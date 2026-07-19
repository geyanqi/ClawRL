"""Ticket 18 durable 16 UID x 128 step-ready contract."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import Artifact, ArtifactStore
from clawrl.training.classic_identity import ClassicSourceRow
from clawrl.training.expected_trajectory_set import ExpectedTrajectorySetConfig, ExpectedTrajectorySetWorkflow
from clawrl.training.step_ready_recovery import (
    StepReadyRecoveryConfig,
    StepReadyRecoveryError,
    StepReadyRecoveryWorkflow,
    StepReadyTerminalStop,
)


def _setup(root: Path, run_id: str) -> tuple[StepReadyRecoveryConfig, Artifact, dict[str, Artifact], Artifact]:
    store = ArtifactStore(root)
    sources = tuple(ClassicSourceRow(f"trace-18-{i}", f"uid-18-{i}", f"pack-18-{i}", "prompt") for i in range(16))
    expected_config = ExpectedTrajectorySetConfig(run_id, 18, 128)
    expected = ExpectedTrajectorySetWorkflow.freeze(root, config=expected_config, sources=sources)
    ExpectedTrajectorySetWorkflow.publish_batch(
        root, expected_set=expected, rows=ExpectedTrajectorySetWorkflow.generated_rows(expected_config, sources)
    )
    ExpectedTrajectorySetWorkflow.authorize_scoring(root, expected_set=expected, transport="classic")
    packs = {
        source.uid: store.put(
            "JudgePack",
            "1.0.0",
            {
                "aggregation": "calibrated_scalar",
                "algorithm_contract": "grpo-calibrated-scalar-v1",
                "items_per_turn": 8,
                "judge_pack_id": source.judge_pack_id,
                "judge_pack_version": "1.0.0",
                "prompt_hash": "a" * 64,
                "scalarizer_hash": "b" * 64,
                "scalarizer_version": "1.0.0",
                "scorer_tier": "luna",
                "status": "certified",
                "trace_id": source.trace_id,
                "uid": source.uid,
            },
        )
        for source in sources
    }
    spec = store.put(
        "ExperimentSpec",
        "1.0.0",
        {
            "aggregation": "calibrated_scalar",
            "dataset_version_hash": "c" * 64,
            "dataset_version_id": "dataset-18",
            "experiment_id": "experiment-18",
            "judge_bundle_hash": "d" * 64,
            "trace_set_hash": "e" * 64,
        },
    )
    return StepReadyRecoveryConfig(run_id, 18, spec.content_hash, chunk_size=13), expected, packs, spec


def test_exact_2048_step_ready_and_fresh_resume(tmp_path: Path) -> None:
    config, expected, packs, spec = _setup(tmp_path, "step-ready-18-contract")
    snapshot = StepReadyRecoveryWorkflow.run(
        tmp_path, config=config, expected_set=expected, judge_packs=packs, experiment_spec=spec
    )
    assert snapshot.step_ready.payload["expected_slot_count"] == 2048
    assert len(snapshot.rewards) == 2048
    assert len(snapshot.uid_states) == 16
    assert snapshot.report.payload["queue_only"] is True
    assert snapshot.report.payload["expansion_requested"] is False
    assert snapshot.report.payload["account_creation_requested"] is False
    assert StepReadyRecoveryWorkflow.resume(tmp_path, config).report.content_hash == snapshot.report.content_hash


def test_conflict_and_unknown_spec_fail_closed_without_step_ready(tmp_path: Path) -> None:
    config, expected, packs, spec = _setup(tmp_path, "step-ready-18-stop")
    conflict = StepReadyRecoveryConfig(config.run_id, 18, spec.content_hash, fault_kind="conflicting_result")
    with pytest.raises(StepReadyTerminalStop, match="CONFLICTING_RESULT"):
        StepReadyRecoveryWorkflow.run(
            tmp_path, config=conflict, expected_set=expected, judge_packs=packs, experiment_spec=spec
        )
    assert not (tmp_path / "step-ready-recovery" / config.run_id / "18" / "classic" / "report.ref").exists()
    unknown = StepReadyRecoveryConfig("step-ready-18-unknown", 18, "f" * 64)
    with pytest.raises(StepReadyRecoveryError, match="ExperimentSpec identity"):
        StepReadyRecoveryWorkflow.run(
            tmp_path, config=unknown, expected_set=expected, judge_packs=packs, experiment_spec=spec
        )


def test_production_readiness_remains_blocked(tmp_path: Path) -> None:
    readiness = StepReadyRecoveryWorkflow.production_readiness(tmp_path)
    assert readiness.payload["status"] == "blocked"
    assert readiness.payload["side_effects_permitted"] is False
    assert {item["status"] for item in cast(list[dict[str, str]], readiness.payload["checks"])} == {"blocked"}
