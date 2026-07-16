"""Successful, restartable Ticket 03 ingestion workflow contracts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from clawrl.artifacts import ArtifactStore
from clawrl.data.models import DatasetValidationError, FixtureDataIngestConfig, QueryWindow
from clawrl.data.validation import load_training_dataset, validate_dataset_for_experiment
from clawrl.data.workflow import DataIngestSnapshot, GovernedDataIngestWorkflow
from tests.contracts.test_ticket03_sql_and_dataset_count import APPROVED_SQL
from tests.fixtures.ticket03_data import (
    PRIVATE_SENTINEL,
    stage_data_provider,
)

EXPECTED_EVENTS = [
    "RUN_STARTED",
    "QUERY_PLANNED",
    "QUERY_REQUESTED",
    "CANDIDATES_SANITIZED",
    "ROWS_VALIDATED",
    "ROWS_DEDUPED",
    "BADCASES_SELECTED",
    "DATASET_PUBLISHED",
    "DECISION_RECORDED",
    "RUN_CLOSED",
]


def config(
    run_id: str,
    *,
    query_sql: str = APPROVED_SQL,
    sanitizer: str = "sentinel-drop-v1",
    selection: str = "badcase-sha256-v1",
) -> FixtureDataIngestConfig:
    return FixtureDataIngestConfig(
        run_id=run_id,
        query_sql=query_sql,
        window=QueryWindow("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
        sanitizer_policy_version=sanitizer,
        selection_policy_version=selection,
    )


def start(root: Path, item: FixtureDataIngestConfig) -> DataIngestSnapshot:
    return GovernedDataIngestWorkflow.bootstrap(
        root,
        item,
        role_ingress=stage_data_provider(root),
        epoch=3001,
    )


def resume_in_subprocess(root: Path, run_id: str, epoch: int = 3001) -> dict[str, Any]:
    code = """
import json,sys
from clawrl.data.workflow import GovernedDataIngestWorkflow
s=GovernedDataIngestWorkflow.resume(sys.argv[1],sys.argv[2],epoch=int(sys.argv[3]))
print(json.dumps({
  'event_types':[e.payload['event_type'] for e in s.events],
  'dataset_version_hash':None if s.dataset_version is None else s.dataset_version.content_hash,
  'terminal':s.terminal,
},sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root), run_id, str(epoch)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def complete_with_fresh_process_per_stage(root: Path, item: FixtureDataIngestConfig) -> str:
    initial = start(root, item)
    assert [event.payload["event_type"] for event in initial.events] == ["RUN_STARTED"]
    latest: dict[str, Any] = {"terminal": False}
    while not latest["terminal"]:
        latest = resume_in_subprocess(root, item.run_id)
    assert latest["event_types"] == EXPECTED_EVENTS
    dataset_hash = latest["dataset_version_hash"]
    assert isinstance(dataset_hash, str)
    return dataset_hash


def public_file_snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    result: dict[str, tuple[int, int, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "private-boundary" not in path.parts:
            raw = path.read_bytes()
            result[str(path.relative_to(root))] = (path.stat().st_mtime_ns, len(raw), hashlib.sha256(raw).hexdigest())
    return result


def test_fresh_process_every_stage_publishes_verified_data_only_exact_100(tmp_path: Path) -> None:
    dataset_hash = complete_with_fresh_process_per_stage(tmp_path, config("ticket03-success"))
    store = ArtifactStore(tmp_path)
    loaded = load_training_dataset(store, dataset_hash)
    assert loaded.dataset_version.content_hash == dataset_hash
    assert len(loaded.training_traces) == 100
    assert len({trace.payload["trace_id"] for trace in loaded.training_traces}) == 100
    assert all(trace.payload["purpose"] == "training_allowed" for trace in loaded.training_traces)
    assert validate_dataset_for_experiment(store, dataset_hash).dataset_version.content_hash == dataset_hash

    prohibited_keys = {"judge", "judge_bundle", "judge_pack", "label", "teacher_label", "holdout"}
    for artifact in [loaded.dataset_version, *loaded.training_traces]:
        pending: list[object] = [artifact.payload]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                assert prohibited_keys.isdisjoint(value)
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
    sentinel = PRIVATE_SENTINEL.encode()
    public_paths = [path for path in tmp_path.rglob("*") if path.is_file() and "private-boundary" not in path.parts]
    assert public_paths
    assert all(sentinel not in path.read_bytes() for path in public_paths)
    assert any(sentinel in path.read_bytes() for path in (tmp_path / "private-boundary").rglob("*") if path.is_file())


def test_dataset_hash_and_bytes_are_root_and_process_independent(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    first = complete_with_fresh_process_per_stage(left, config("root-left"))
    second = complete_with_fresh_process_per_stage(right, config("root-right"))
    assert first == second
    assert (left / "artifacts" / f"{first}.json").read_bytes() == (right / "artifacts" / f"{second}.json").read_bytes()


def test_query_sanitizer_and_selection_versions_each_change_root_lineage(tmp_path: Path) -> None:
    baseline = complete_with_fresh_process_per_stage(tmp_path / "base", config("base"))
    query_changed = complete_with_fresh_process_per_stage(
        tmp_path / "query",
        config("query", query_sql="  " + APPROVED_SQL.replace(" FROM ", "\nFROM ")),
    )
    sanitizer_changed = complete_with_fresh_process_per_stage(
        tmp_path / "sanitizer",
        config("sanitizer", sanitizer="sentinel-drop-v1.1"),
    )
    selection_changed = complete_with_fresh_process_per_stage(
        tmp_path / "selection",
        config("selection", selection="badcase-sha256-v2"),
    )
    assert len({baseline, query_changed, sanitizer_changed, selection_changed}) == 4
    for root, dataset_hash, expected in (
        (tmp_path / "query", query_changed, ("query_source_hash",)),
        (tmp_path / "sanitizer", sanitizer_changed, ("sanitizer_policy_version",)),
        (tmp_path / "selection", selection_changed, ("selection_policy_version",)),
    ):
        dataset = ArtifactStore(root).read(dataset_hash, expected_schema_name="DatasetVersion")
        lineage = dataset.payload["lineage"]
        assert isinstance(lineage, dict)
        assert all(key in lineage for key in expected)


def test_terminal_run_once_retry_is_read_only_and_returns_same_dataset(tmp_path: Path) -> None:
    item = config("terminal-retry")
    dataset_hash = complete_with_fresh_process_per_stage(tmp_path, item)
    before = public_file_snapshot(tmp_path)
    replay = GovernedDataIngestWorkflow.run_once(
        tmp_path,
        item,
        role_ingress=stage_data_provider(tmp_path),
        epoch=9001,
    )
    after = public_file_snapshot(tmp_path)
    assert replay.terminal
    assert replay.dataset_version is not None
    assert replay.dataset_version.content_hash == dataset_hash
    assert before == after


@pytest.mark.parametrize("purpose", ["judge_only", "eval_only"])
def test_loader_and_experiment_validator_both_reject_nontraining_purpose(tmp_path: Path, purpose: str) -> None:
    dataset_hash = complete_with_fresh_process_per_stage(tmp_path, config(f"purpose-{purpose}"))
    store = ArtifactStore(tmp_path)
    original = store.read(dataset_hash, expected_schema_name="DatasetVersion")
    payload = original.payload
    payload["purpose"] = purpose
    unsafe = store.put("DatasetVersion", "1.0.0", payload)
    with pytest.raises(DatasetValidationError):
        load_training_dataset(store, unsafe.content_hash)
    with pytest.raises(DatasetValidationError):
        validate_dataset_for_experiment(store, unsafe.content_hash)


def test_selection_manifest_is_content_derived_and_contains_complete_mix(tmp_path: Path) -> None:
    dataset_hash = complete_with_fresh_process_per_stage(tmp_path, config("selection-evidence"))
    store = ArtifactStore(tmp_path)
    dataset = store.read(dataset_hash, expected_schema_name="DatasetVersion")
    lineage = dataset.payload["lineage"]
    assert isinstance(lineage, dict)
    selection = store.read(str(lineage["selection_manifest_hash"]), expected_schema_name="BadcaseSelectionManifest")
    rankings = selection.payload["rankings"]
    assert isinstance(rankings, list) and len(rankings) == 100
    assert len({str(item["selection_score"]) for item in rankings if isinstance(item, dict)}) == 100
    assert selection.payload["mix"] == {
        "model": {"fixture-model-a": 50, "fixture-model-b": 50},
        "purpose": {"training_allowed": 100},
        "tool": {"javascript": 25, "none": 25, "python": 25, "regex": 25},
    }
    assert selection.payload["selected_count"] == 100
