"""Persistent ClusterAdapter lifecycle acceptance tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from clawrl.artifacts import ArtifactStore
from clawrl.training.cluster_lifecycle import (
    ClusterLifecycleError,
    ClusterLifecycleWorkflow,
    FixtureClusterLifecycleConfig,
    InjectedClusterControllerCrash,
)


def _spec(root: Path, spec_id: str = "ticket21-spec") -> str:
    return (
        ArtifactStore(root).put("ExperimentSpec", "1.0.0", {"experiment_id": spec_id, "status": "frozen"}).content_hash
    )


def _fresh_resume(root: Path, run_id: str) -> dict[str, object]:
    code = """
import json,sys
from clawrl.training.cluster_lifecycle import ClusterLifecycleWorkflow
s=ClusterLifecycleWorkflow.resume(sys.argv[1],sys.argv[2])
print(json.dumps({
    'outcome_count': len(s.outcomes),
    'run_record': s.run_record.content_hash,
    'status': s.run_record.payload['status'],
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root), run_id],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    decoded = json.loads(completed.stdout)
    assert isinstance(decoded, dict)
    return decoded


def test_graceful_close_recovers_by_provider_query_after_checkpoint_crash(tmp_path: Path) -> None:
    spec_hash = _spec(tmp_path)
    config = FixtureClusterLifecycleConfig("cluster-success-21", "TRAIN_35B")
    with pytest.raises(InjectedClusterControllerCrash, match="checkpoint"):
        ClusterLifecycleWorkflow.run(
            tmp_path,
            config=config,
            spec_hash=spec_hash,
            crash_after_operation="checkpoint",
        )
    fresh = _fresh_resume(tmp_path, config.run_id)
    snapshot = ClusterLifecycleWorkflow.resume(tmp_path, config.run_id)
    assert fresh["run_record"] == snapshot.run_record.content_hash
    assert fresh["status"] == "succeeded"
    assert fresh["outcome_count"] == 6
    record = snapshot.run_record.payload
    assert record["production_smoke"] is False
    assert record["fixture_evidence"] is True
    assert record["closure_operation"] == "graceful_stop"
    assert record["spec_hash"] == spec_hash
    for field, schema in (
        ("log_hash", "ClusterLogBundle"),
        ("checkpoint_hash", "ClusterCheckpointManifest"),
        ("artifact_hash", "ClusterArtifactManifest"),
    ):
        value = record[field]
        assert isinstance(value, str)
        ArtifactStore(tmp_path).read(value, expected_schema_name=schema)
    assert snapshot.recovery_evidence
    assert snapshot.recovery_evidence[-1].payload["operation"] == "checkpoint"
    assert snapshot.recovery_evidence[-1].payload["provider_query_found"] is True
    checkpoint_outcomes = [item for item in snapshot.outcomes if item.payload["operation"] == "checkpoint"]
    assert len(checkpoint_outcomes) == 1


def test_force_cancel_is_a_distinct_terminal_operation(tmp_path: Path) -> None:
    spec_hash = _spec(tmp_path)
    config = FixtureClusterLifecycleConfig("cluster-cancel-21", "TRAIN_122B", close_mode="force_cancel")
    snapshot = ClusterLifecycleWorkflow.run(tmp_path, config=config, spec_hash=spec_hash)
    assert snapshot.run_record.payload["status"] == "canceled"
    assert snapshot.run_record.payload["closure_operation"] == "force_cancel"
    assert snapshot.outcomes[-1].payload["operation"] == "force_cancel"
    assert snapshot.outcomes[-1].payload["status"] == "succeeded"


@pytest.mark.parametrize(
    "operation",
    ["submit", "status", "logs", "artifact", "checkpoint", "graceful_stop", "force_cancel"],
)
def test_every_provider_error_closes_with_durable_failed_run_record(tmp_path: Path, operation: str) -> None:
    spec_hash = _spec(tmp_path)
    close_mode = "force_cancel" if operation == "force_cancel" else "graceful_stop"
    config = FixtureClusterLifecycleConfig(
        f"cluster-error-{operation}",
        "TRAIN_35B",
        close_mode=close_mode,
        fault_operation=operation,
        fault_directive="provider_error",
    )
    snapshot = ClusterLifecycleWorkflow.run(tmp_path, config=config, spec_hash=spec_hash)
    record = snapshot.run_record
    assert record.payload["status"] == "failed"
    assert record.payload["failed_operation"] == operation
    assert record.payload["production_smoke"] is False
    evidence_hash = record.payload["failure_evidence_hash"]
    assert isinstance(evidence_hash, str)
    evidence = ArtifactStore(tmp_path).read(evidence_hash, expected_schema_name="ClusterProviderFailureEvidence")
    assert evidence.payload["operation"] == operation
    assert evidence.payload["outcome_hash"] == snapshot.outcomes[-1].content_hash
    assert _fresh_resume(tmp_path, config.run_id)["run_record"] == record.content_hash


def test_same_run_changed_spec_is_integrity_conflict(tmp_path: Path) -> None:
    first = _spec(tmp_path, "ticket21-spec-a")
    config = FixtureClusterLifecycleConfig("cluster-conflict-21", "TRAIN_35B")
    ClusterLifecycleWorkflow.run(tmp_path, config=config, spec_hash=first)
    second = _spec(tmp_path, "ticket21-spec-b")
    with pytest.raises(RuntimeError, match="CLUSTER_RUN_INPUT_CONFLICT"):
        ClusterLifecycleWorkflow.run(tmp_path, config=config, spec_hash=second)


def test_forged_failure_evidence_fails_closed_on_fresh_resume(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    spec_hash = _spec(tmp_path)
    config = FixtureClusterLifecycleConfig(
        "cluster-forged-evidence-21",
        "TRAIN_35B",
        fault_operation="logs",
        fault_directive="provider_error",
    )
    snapshot = ClusterLifecycleWorkflow.run(tmp_path, config=config, spec_hash=spec_hash)
    evidence_hash = snapshot.run_record.payload["failure_evidence_hash"]
    assert isinstance(evidence_hash, str)
    evidence = store.read(evidence_hash, expected_schema_name="ClusterProviderFailureEvidence")
    forged_evidence = store.put(
        "ClusterProviderFailureEvidence",
        "1.0.0",
        {**evidence.payload, "error_code": "FORGED_PROVIDER_ERROR"},
    )
    forged_record = store.put(
        "RunRecord",
        "1.0.0",
        {**snapshot.run_record.payload, "failure_evidence_hash": forged_evidence.content_hash},
    )
    state_ref = tmp_path / "cluster-lifecycle-runs" / config.run_id / "state.ref"
    state = store.read(state_ref.read_text(encoding="ascii").strip(), expected_schema_name="ClusterWorkflowState")
    forged_state = store.put(
        "ClusterWorkflowState",
        "1.0.0",
        {**state.payload, "run_record_hash": forged_record.content_hash},
    )
    state_ref.write_text(f"{forged_state.content_hash}\n", encoding="ascii")

    with pytest.raises(ClusterLifecycleError, match="failure evidence lineage"):
        ClusterLifecycleWorkflow.resume(tmp_path, config.run_id)
