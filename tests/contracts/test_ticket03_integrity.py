"""Closed-world disk/ref/boundary integrity contracts for Ticket 03."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clawrl.adapters.sources.fixture import FixtureDataSource
from clawrl.artifacts import ArtifactCorruption, ArtifactStore, canonical_json_bytes, sha256_hex
from clawrl.data.models import DatasetValidationError
from clawrl.data.validation import load_training_dataset
from clawrl.data.workflow import DataIngestWorkflowError, GovernedDataIngestWorkflow
from clawrl.training.run_journal import RunJournalCorruption
from tests.e2e.test_ticket03_fault_recovery import resume_process, to_query_requested
from tests.e2e.test_ticket03_successful_dataset_workflow import (
    complete_with_fresh_process_per_stage,
    config,
)
from tests.fixtures.ticket03_data import (
    PRIVATE_SENTINEL,
    stage_data_provider,
)


def bootstrap(root: Path, run_id: str) -> object:
    return GovernedDataIngestWorkflow.bootstrap(
        root,
        config(run_id),
        role_ingress=stage_data_provider(root),
        epoch=3001,
    )


def replace_last_event_detail(root: Path, run_id: str, key: str, value: str) -> None:
    store = ArtifactStore(root)
    refs = sorted((root / "runs" / run_id / "events").glob("*.ref"))
    ref = refs[-1]
    event = store.read(ref.read_text().strip(), expected_schema_name="RunEvent")
    payload = event.payload
    details = payload["details"]
    assert isinstance(details, dict)
    details[key] = value
    replacement = store.put("RunEvent", "1.0.0", payload)
    ref.write_bytes(f"{replacement.content_hash}\n".encode())


@pytest.mark.parametrize(
    "ref_bytes",
    [
        b"short\n",
        b"g" * 64 + b"\n",
        "测".encode() + b"\n",
        b"f" * 64 + b"\n",
    ],
)
def test_malformed_or_missing_event_artifact_ref_fails_before_progress(tmp_path: Path, ref_bytes: bytes) -> None:
    item = config("bad-event-ref")
    bootstrap(tmp_path, item.run_id)
    ref = tmp_path / "runs" / item.run_id / "events" / "00000000000000000001.ref"
    ref.write_bytes(ref_bytes)
    with pytest.raises((RunJournalCorruption, DataIngestWorkflowError)):
        GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3001)
    assert not list((tmp_path / "private-boundary" / "data-source").rglob("queries/*"))


def test_content_valid_query_plan_substitution_is_rejected_before_query(tmp_path: Path) -> None:
    item = config("query-plan-substitution")
    bootstrap(tmp_path, item.run_id)
    assert resume_process(tmp_path, item.run_id)["events"][-1] == "QUERY_PLANNED"
    store = ArtifactStore(tmp_path)
    event_ref = tmp_path / "runs" / item.run_id / "events" / "00000000000000000002.ref"
    event = store.read(event_ref.read_text().strip(), expected_schema_name="RunEvent")
    details = event.payload["details"]
    assert isinstance(details, dict)
    plan = store.read(str(details["query_plan_hash"]), expected_schema_name="QueryPlan")
    payload = plan.payload
    payload["query_source_hash"] = "f" * 64
    substitute = store.put("QueryPlan", "1.0.0", payload)
    replace_last_event_detail(tmp_path, item.run_id, "query_plan_hash", substitute.content_hash)
    with pytest.raises(DataIngestWorkflowError, match="QueryPlan"):
        GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3001)
    assert not list((tmp_path / "private-boundary" / "data-source").rglob("invocations/*.json"))


@pytest.mark.parametrize("mutation", ["missing", "nonascii"])
def test_committed_private_query_result_is_reread_before_candidates_are_trusted(
    tmp_path: Path,
    mutation: str,
) -> None:
    item = config(f"query-result-{mutation}")
    to_query_requested(tmp_path, item)
    assert resume_process(tmp_path, item.run_id)["events"][-1] == "CANDIDATES_SANITIZED"
    result_path = next((tmp_path / "private-boundary").rglob("success.canonical.json"))
    if mutation == "missing":
        result_path.unlink()
    else:
        result_path.write_bytes("测".encode())
    state = resume_process(tmp_path, item.run_id)
    assert state["events"][-1] == "INGEST_FAILURE_RECORDED"
    assert state["dataset"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_field",
        "bool_purpose",
        "bool_event_time",
        "int_prompt",
        "changed_source_id",
        "bool_boundary_attempt",
        "changed_boundary_result_hash",
        "unknown_major",
        "duplicate_ref",
    ],
)
def test_candidate_set_and_candidate_substitutions_fail_closed(tmp_path: Path, mutation: str) -> None:
    item = config(f"candidate-{mutation}")
    to_query_requested(tmp_path, item)
    assert resume_process(tmp_path, item.run_id)["events"][-1] == "CANDIDATES_SANITIZED"
    store = ArtifactStore(tmp_path)
    event_ref = tmp_path / "runs" / item.run_id / "events" / "00000000000000000004.ref"
    event = store.read(event_ref.read_text().strip(), expected_schema_name="RunEvent")
    details = event.payload["details"]
    assert isinstance(details, dict)
    candidate_set = store.read(str(details["candidate_set_hash"]), expected_schema_name="SanitizedCandidateSet")
    payload = candidate_set.payload
    refs = payload["candidate_refs"]
    assert isinstance(refs, list)
    if mutation == "duplicate_ref":
        refs[1] = refs[0]
        payload["candidate_ref_set_hash"] = sha256_hex(canonical_json_bytes(refs))
    elif mutation == "bool_boundary_attempt":
        payload["boundary_attempt_ordinal"] = True
    elif mutation == "changed_boundary_result_hash":
        payload["boundary_result_hash"] = "f" * 64
    else:
        first = refs[0]
        assert isinstance(first, dict)
        candidate = store.read(str(first["artifact_hash"]), expected_schema_name="SanitizedCandidate")
        candidate_payload = candidate.payload
        schema_version = "2.0.0" if mutation == "unknown_major" else "1.0.0"
        if mutation == "extra_field":
            candidate_payload["unexpected"] = "not-approved"
        elif mutation == "bool_purpose":
            candidate_payload["purpose"] = True
        elif mutation == "bool_event_time":
            candidate_payload["event_time_utc"] = False
        elif mutation == "int_prompt":
            candidate_payload["prompt"] = 7
        elif mutation == "changed_source_id":
            candidate_payload["source_id"] = "fixture-online-traces-v2"
        substitute = store.put("SanitizedCandidate", schema_version, candidate_payload)
        first["artifact_hash"] = substitute.content_hash
        payload["candidate_ref_set_hash"] = sha256_hex(canonical_json_bytes(refs))
    substitute_set = store.put("SanitizedCandidateSet", "1.0.0", payload)
    replace_last_event_detail(tmp_path, item.run_id, "candidate_set_hash", substitute_set.content_hash)
    with pytest.raises((DataIngestWorkflowError, DatasetValidationError, ArtifactCorruption)):
        GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3001)
    assert "DatasetVersion" not in [
        json.loads(path.read_bytes())["schema_name"] for path in (tmp_path / "artifacts").glob("*.json")
    ]


@pytest.mark.parametrize("mutation", ["duplicate_trace", "forbidden_judge", "bool_count", "reordered"])
def test_loader_rejects_content_valid_dataset_substitution_matrix(tmp_path: Path, mutation: str) -> None:
    dataset_hash = complete_with_fresh_process_per_stage(tmp_path, config(f"dataset-{mutation}"))
    store = ArtifactStore(tmp_path)
    dataset = store.read(dataset_hash, expected_schema_name="DatasetVersion")
    payload = dataset.payload
    refs = payload["trace_refs"]
    assert isinstance(refs, list)
    if mutation == "duplicate_trace":
        refs[1] = refs[0]
    elif mutation == "forbidden_judge":
        payload["judge_pack"] = "f" * 64
    elif mutation == "bool_count":
        payload["trace_count"] = True
    elif mutation == "reordered":
        refs[0], refs[1] = refs[1], refs[0]
    substitute = store.put("DatasetVersion", "1.0.0", payload)
    with pytest.raises(DatasetValidationError):
        load_training_dataset(store, substitute.content_hash)


@pytest.mark.parametrize(
    "target",
    ["DatasetVersion", "TrainingTrace", "DatasetPublicationManifest", "DecisionRecord"],
)
def test_terminal_replay_detects_corrupted_terminal_dag(tmp_path: Path, target: str) -> None:
    item = config(f"terminal-corrupt-{target.lower()}")
    complete_with_fresh_process_per_stage(tmp_path, item)
    schema_path = next(
        path
        for path in (tmp_path / "artifacts").glob("*.json")
        if json.loads(path.read_bytes())["schema_name"] == target
    )
    value = json.loads(schema_path.read_bytes())
    value["payload"]["tampered"] = True
    schema_path.write_bytes(canonical_json_bytes(value))
    with pytest.raises((DataIngestWorkflowError, DatasetValidationError, ArtifactCorruption, RunJournalCorruption)):
        GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=9999)


@pytest.mark.parametrize("boundary", ["factory_artifact", "factory_runtime", "query_artifact", "query_runtime"])
def test_boundary_errors_close_sanitized_before_dataset(
    tmp_path: Path,
    boundary: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = config(f"boundary-{boundary}")
    if boundary.startswith("query"):
        to_query_requested(tmp_path, item)
        method = "query"
    else:
        bootstrap(tmp_path, item.run_id)
        method = "open"
    error: Exception = ArtifactCorruption("secret=T03_PRIVATE_SENTINEL_NEVER_EXPORT")
    if boundary.endswith("runtime"):
        error = RuntimeError("secret=T03_PRIVATE_SENTINEL_NEVER_EXPORT")

    def fail(*args: object, **kwargs: object) -> object:
        raise error

    monkeypatch.setattr(FixtureDataSource, method, classmethod(fail) if method == "open" else fail)
    snapshot = None
    for _ in range(16):
        snapshot = GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3001)
        if snapshot.terminal:
            break
    assert snapshot is not None and snapshot.terminal and snapshot.dataset_version is None
    assert snapshot.decision_record is not None
    public = [path for path in tmp_path.rglob("*") if path.is_file() and "private-boundary" not in path.parts]
    assert all(PRIVATE_SENTINEL.encode() not in path.read_bytes() for path in public)


@pytest.mark.parametrize("mutation", ["duplicate_key", "formula_change", "output_change"])
def test_private_role_raw_mutation_fails_closed_on_fresh_resume(tmp_path: Path, mutation: str) -> None:
    item = config(f"private-role-{mutation}")
    bootstrap(tmp_path, item.run_id)
    raw_path = next((tmp_path / "private-boundary").rglob("role-output.raw.json"))
    raw = raw_path.read_bytes()
    if mutation == "duplicate_key":
        raw = b'{"packet_id":"a","packet_id":"b"}'
    elif mutation == "formula_change":
        raw = raw.replace(b"javascript", b"javascripT", 1)
    else:
        raw = raw.replace(b"fixture-online-traces-v1", b"fixture-online-traces-v2", 1)
    raw_path.write_bytes(raw)
    first = resume_process(tmp_path, item.run_id)
    assert first["events"][-1] == "INGEST_FAILURE_RECORDED"
    terminal = resume_process(tmp_path, item.run_id)
    assert terminal["terminal"] is True and terminal["dataset"] is None
    assert not list((tmp_path / "private-boundary").rglob("invocations/*.json"))
