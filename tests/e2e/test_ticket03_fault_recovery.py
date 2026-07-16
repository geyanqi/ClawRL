"""Fault, retry, crash, and fail-closed contracts for Ticket 03."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from clawrl.artifacts import ArtifactStore
from clawrl.data.models import FixtureDataIngestConfig, QueryWindow
from clawrl.data.workflow import GovernedDataIngestWorkflow, InjectedDataIngestCrash
from tests.contracts.test_ticket03_sql_and_dataset_count import APPROVED_SQL
from tests.fixtures.ticket03_data import (
    PRIVATE_SENTINEL,
    stage_data_provider,
)


def fault_config(
    run_id: str,
    *,
    schedule: tuple[str, ...] = ("success",),
    source_fault: str | None = None,
) -> FixtureDataIngestConfig:
    return FixtureDataIngestConfig(
        run_id=run_id,
        query_sql=APPROVED_SQL,
        window=QueryWindow("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
        fault_schedule=schedule,
        retry_limit=4,
        source_fault=source_fault,
    )


def start(root: Path, item: FixtureDataIngestConfig) -> None:
    GovernedDataIngestWorkflow.bootstrap(
        root,
        item,
        role_ingress=stage_data_provider(root),
        epoch=3001,
    )


def resume_process(root: Path, run_id: str) -> dict[str, Any]:
    code = """
import json,sys
from clawrl.data.workflow import GovernedDataIngestWorkflow
s=GovernedDataIngestWorkflow.resume(sys.argv[1],sys.argv[2],epoch=3001)
print(json.dumps({
 'events':[e.payload['event_type'] for e in s.events],
 'terminal':s.terminal,
 'dataset':None if s.dataset_version is None else s.dataset_version.content_hash,
 'decision':None if s.decision_record is None else s.decision_record.content_hash,
},sort_keys=True))
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
    return json.loads(completed.stdout)


def to_query_requested(root: Path, item: FixtureDataIngestConfig) -> None:
    start(root, item)
    assert resume_process(root, item.run_id)["events"][-1] == "QUERY_PLANNED"
    assert resume_process(root, item.run_id)["events"][-1] == "QUERY_REQUESTED"


def finish(root: Path, item: FixtureDataIngestConfig) -> dict[str, Any]:
    if not (root / "runs" / item.run_id / "identity.ref").exists():
        start(root, item)
    state: dict[str, Any] = {"terminal": False}
    for _ in range(32):
        if state["terminal"]:
            return state
        state = resume_process(root, item.run_id)
    raise AssertionError("workflow did not reach a bounded terminal state")


def public_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and "private-boundary" not in path.parts]


def artifact_schemas(root: Path) -> list[str]:
    schemas: list[str] = []
    for path in (root / "artifacts").glob("*.json"):
        value = json.loads(path.read_bytes())
        schemas.append(str(value["schema_name"]))
    return schemas


@pytest.mark.parametrize(
    ("directive", "event_type", "failure_code"),
    [
        ("timeout", "QUERY_RETRY_SCHEDULED", "QUERY_TIMEOUT"),
        ("error", "QUERY_RETRY_SCHEDULED", "QUERY_PROVIDER_ERROR"),
        ("delayed", "QUERY_RETRY_SCHEDULED", "QUERY_DELAYED"),
        ("late", "QUERY_RETRY_SCHEDULED", "QUERY_DELAYED"),
    ],
)
def test_retryable_fault_has_durable_observation_then_recovers_once(
    tmp_path: Path,
    directive: str,
    event_type: str,
    failure_code: str,
) -> None:
    item = fault_config(f"retry-{directive}", schedule=(directive, "success"))
    to_query_requested(tmp_path, item)
    failed_attempt = resume_process(tmp_path, item.run_id)
    assert failed_attempt["events"][-1] == event_type
    store = ArtifactStore(tmp_path)
    event = store.read(
        (tmp_path / "runs" / item.run_id / "events" / f"{len(failed_attempt['events']):020d}.ref").read_text().strip(),
        expected_schema_name="RunEvent",
    )
    details = event.payload["details"]
    assert isinstance(details, dict)
    observation = store.read(
        str(details["observation_hash"]),
        expected_schema_name="DataSourceQueryObservation",
    )
    assert observation.payload["failure_code"] == failure_code
    recovered = resume_process(tmp_path, item.run_id)
    assert recovered["events"][-1] == "CANDIDATES_SANITIZED"
    terminal = finish(tmp_path, item)
    assert terminal["dataset"] is not None
    invocations = list((tmp_path / "private-boundary" / "data-source").rglob("invocations/*.json"))
    assert len(invocations) == 2
    if directive == "late":
        assert list((tmp_path / "private-boundary" / "data-source").rglob("deliveries/quarantine/*.json"))
    assert all(PRIVATE_SENTINEL.encode() not in path.read_bytes() for path in public_files(tmp_path))


def test_permanent_provider_failure_closes_with_sanitized_decision_and_no_dataset(tmp_path: Path) -> None:
    item = fault_config("permanent-provider", schedule=("error", "permanent_failure"))
    terminal = finish(tmp_path, item)
    assert terminal["terminal"] is True
    assert terminal["dataset"] is None
    assert terminal["decision"] is not None
    store = ArtifactStore(tmp_path)
    decision = store.read(str(terminal["decision"]), expected_schema_name="DecisionRecord")
    assert decision.payload["outcome"] == "failed"
    assert decision.payload["reason_code"] == "QUERY_PROVIDER_PERMANENT_FAILURE"
    assert "DatasetVersion" not in artifact_schemas(tmp_path)
    assert all(PRIVATE_SENTINEL.encode() not in path.read_bytes() for path in public_files(tmp_path))


def test_crash_after_private_success_recovers_query_first_without_second_invocation(tmp_path: Path) -> None:
    item = fault_config("crash-private-result")
    to_query_requested(tmp_path, item)
    with pytest.raises(InjectedDataIngestCrash, match="PRIVATE_QUERY_RESULT_COMMITTED"):
        GovernedDataIngestWorkflow.resume(
            tmp_path,
            item.run_id,
            epoch=3001,
            crash_after="PRIVATE_QUERY_RESULT_COMMITTED",
        )
    store = ArtifactStore(tmp_path)
    events = store.read(
        (tmp_path / "runs" / item.run_id / "events" / "00000000000000000003.ref").read_text().strip(),
        expected_schema_name="RunEvent",
    )
    assert events.payload["event_type"] == "QUERY_REQUESTED"
    invocations = list((tmp_path / "private-boundary" / "data-source").rglob("invocations/*.json"))
    assert len(invocations) == 1
    assert resume_process(tmp_path, item.run_id)["events"][-1] == "CANDIDATES_SANITIZED"
    assert len(list((tmp_path / "private-boundary" / "data-source").rglob("invocations/*.json"))) == 1
    assert finish(tmp_path, item)["dataset"] is not None


@pytest.mark.parametrize("source_fault", ["missing_sentinel", "wrong_sentinel", "leak_sentinel"])
def test_sentinel_fault_aborts_before_any_public_candidate_or_dataset(
    tmp_path: Path,
    source_fault: str,
) -> None:
    item = fault_config(f"sentinel-{source_fault}", source_fault=source_fault)
    terminal = finish(tmp_path, item)
    assert terminal["terminal"] is True
    assert terminal["dataset"] is None
    schemas = artifact_schemas(tmp_path)
    assert not {"SanitizedCandidate", "TrainingTrace", "DatasetVersion"}.intersection(schemas)
    assert all(PRIVATE_SENTINEL.encode() not in path.read_bytes() for path in public_files(tmp_path))
    assert any(
        PRIVATE_SENTINEL.encode() in path.read_bytes()
        for path in (tmp_path / "private-boundary").rglob("*")
        if path.is_file()
    )


@pytest.mark.parametrize(
    "source_fault",
    [
        "row_limit",
        "byte_limit",
        "time_limit",
        "unique_99",
        "unique_101",
        "duplicate_semantic_mismatch",
        "missing_field",
        "wrong_purpose",
        "out_of_window",
    ],
)
def test_source_and_post_query_invariant_faults_close_without_partial_dataset(
    tmp_path: Path,
    source_fault: str,
) -> None:
    item = fault_config(f"invariant-{source_fault}", source_fault=source_fault)
    terminal = finish(tmp_path, item)
    assert terminal["terminal"] is True
    assert terminal["dataset"] is None
    assert terminal["decision"] is not None
    assert "DatasetVersion" not in artifact_schemas(tmp_path)
    assert all(PRIVATE_SENTINEL.encode() not in path.read_bytes() for path in public_files(tmp_path))


def test_faulted_run_once_is_idempotent_and_never_creates_scheduler_artifact(tmp_path: Path) -> None:
    item = fault_config("run-once-retry", schedule=("timeout", "success"))
    first = GovernedDataIngestWorkflow.run_once(
        tmp_path,
        item,
        role_ingress=stage_data_provider(tmp_path),
        epoch=3001,
    )
    assert first.terminal and first.dataset_version is not None
    second = GovernedDataIngestWorkflow.run_once(
        tmp_path,
        item,
        role_ingress=stage_data_provider(tmp_path),
        epoch=9001,
    )
    assert second.dataset_version is not None
    assert second.dataset_version.content_hash == first.dataset_version.content_hash
    assert not any(
        token in path.name.lower()
        for path in tmp_path.rglob("*")
        for token in ("scheduler", "scheduled-task", "cron", "recursion")
    )
