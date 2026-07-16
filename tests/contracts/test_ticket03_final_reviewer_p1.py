"""Regressions for the final independent Ticket 03 P1 review."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from clawrl.adapters.sources.fixture import DataSourceBoundaryError, FixtureDataSource
from clawrl.artifacts import Artifact, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.data.domain import dataset_version_id
from clawrl.data.models import DatasetValidationError
from clawrl.data.validation import expected_validation_manifest, load_training_dataset
from clawrl.data.workflow import DataIngestWorkflowError, GovernedDataIngestWorkflow
from tests.e2e.test_ticket03_fault_recovery import fault_config, finish, resume_process, to_query_requested
from tests.e2e.test_ticket03_successful_dataset_workflow import (
    complete_with_fresh_process_per_stage,
    config,
)
from tests.fixtures.ticket03_data import DATA_PROVIDER_RAW, PRIVATE_SENTINEL, stage_data_provider


def _replace_head_event(root: Path, run_id: str, mutate: Callable[[dict[str, object]], None]) -> None:
    store = ArtifactStore(root)
    ref = sorted((root / "runs" / run_id / "events").glob("*.ref"))[-1]
    event = store.read(ref.read_text().strip(), expected_schema_name="RunEvent")
    payload = event.payload
    details = cast(dict[str, object], payload["details"])
    mutate(details)
    replacement = store.put("RunEvent", "1.0.0", payload)
    ref.write_text(replacement.content_hash + "\n")


def _dataset_with_lineage(store: ArtifactStore, dataset: Artifact, **updates: str) -> Artifact:
    payload = dataset.payload
    lineage = cast(dict[str, JsonValue], payload["lineage"])
    lineage.update(updates)
    without_id = dict(payload)
    without_id.pop("dataset_version_id")
    without_id["lineage"] = lineage
    return store.put(
        "DatasetVersion",
        "1.0.0",
        {"dataset_version_id": dataset_version_id(without_id), **without_id},
    )


def _substitute_run_closed(
    root: Path,
    run_id: str,
    *,
    status: str | None = None,
    reason_code: str | None = None,
) -> None:
    store = ArtifactStore(root)
    event_ref = sorted((root / "runs" / run_id / "events").glob("*.ref"))[-1]
    terminal = store.read(event_ref.read_text().strip(), expected_schema_name="RunEvent")
    details = cast(dict[str, object], terminal.payload["details"])
    closed = store.read(str(details["run_closed_hash"]), expected_schema_name="RunClosed")
    closed_payload = closed.payload
    if status is not None:
        closed_payload["status"] = status
    if reason_code is not None:
        closed_payload["reason_code"] = reason_code
    replacement = store.put("RunClosed", "1.0.0", closed_payload)
    terminal_payload = terminal.payload
    terminal_details = cast(dict[str, object], terminal_payload["details"])
    terminal_details["run_closed_hash"] = replacement.content_hash
    replacement_event = store.put("RunEvent", "1.0.0", terminal_payload)
    event_ref.write_text(replacement_event.content_hash + "\n")
    (root / "runs" / run_id / "closed.ref").write_text(replacement.content_hash + "\n")


def test_base_exchange_preserves_and_fresh_recertifies_private_normalized_output(tmp_path: Path) -> None:
    ingress = stage_data_provider(tmp_path)
    private = tmp_path / "private-boundary" / "data-provider-ingress" / ingress.ingress_hash
    raw = (private / "role-output.raw.json").read_bytes()
    normalized_path = private / "role-output.normalized.canonical.json"
    normalized = normalized_path.read_bytes()
    assert raw == DATA_PROVIDER_RAW
    assert normalized == canonical_json_bytes(json.loads(raw))
    assert PRIVATE_SENTINEL.encode() in normalized
    assert sha256_hex(normalized) == ingress.normalized_output_hash
    audit = ArtifactStore(tmp_path).read(
        ingress.role_invocation_audit_hash,
        expected_schema_name="RoleInvocationAudit",
    )
    assert audit.payload["normalized_output_hash"] == ingress.normalized_output_hash
    assert audit.payload["normalized_output_size"] == len(normalized)

    item = config("base-normalized-recert")
    GovernedDataIngestWorkflow.bootstrap(tmp_path, item, role_ingress=ingress, epoch=3001)
    copied = next((tmp_path / "private-boundary" / "data-source").rglob("role-output.normalized.canonical.json"))
    copied.write_bytes(copied.read_bytes() + b" ")
    failed = GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3001)
    assert failed.events[-1].payload["event_type"] == "INGEST_FAILURE_RECORDED"
    assert failed.dataset_version is None


def test_loader_rejects_bool_rank_in_resigned_selection(tmp_path: Path) -> None:
    dataset_hash = complete_with_fresh_process_per_stage(tmp_path, config("loader-bool-rank"))
    store = ArtifactStore(tmp_path)
    dataset = store.read(dataset_hash, expected_schema_name="DatasetVersion")
    lineage = cast(dict[str, object], dataset.payload["lineage"])
    selection = store.read(str(lineage["selection_manifest_hash"]), expected_schema_name="BadcaseSelectionManifest")
    selection_payload = selection.payload
    rankings = cast(list[dict[str, object]], selection_payload["rankings"])
    rankings[0]["rank"] = True
    replacement = store.put("BadcaseSelectionManifest", "1.0.0", selection_payload)
    substitute = _dataset_with_lineage(store, dataset, selection_manifest_hash=replacement.content_hash)
    with pytest.raises(DatasetValidationError):
        load_training_dataset(store, substitute.content_hash)


def test_loader_rejects_resigned_duplicate_report_id_across_distinct_traces(tmp_path: Path) -> None:
    dataset_hash = complete_with_fresh_process_per_stage(tmp_path, config("report-id-validation"))
    store = ArtifactStore(tmp_path)
    dataset = store.read(dataset_hash, expected_schema_name="DatasetVersion")
    lineage = cast(dict[str, object], dataset.payload["lineage"])
    candidate_set = store.read(str(lineage["candidate_set_hash"]), expected_schema_name="SanitizedCandidateSet")
    refs = cast(list[dict[str, object]], candidate_set.payload["candidate_refs"])
    candidates = [store.read(str(ref["artifact_hash"]), expected_schema_name="SanitizedCandidate") for ref in refs]
    second = candidates[1]
    second_payload = second.payload
    second_payload["report_id"] = candidates[0].payload["report_id"]
    candidates[1] = store.put("SanitizedCandidate", "1.0.0", second_payload)
    with pytest.raises(DatasetValidationError, match="report"):
        expected_validation_manifest(
            candidate_set,
            candidates,
            window_start_utc="2026-01-01T00:00:00Z",
            window_end_utc="2026-01-02T00:00:00Z",
        )

    candidate_set_payload = candidate_set.payload
    rewritten_refs = cast(list[dict[str, object]], candidate_set_payload["candidate_refs"])
    rewritten_refs[1]["artifact_hash"] = candidates[1].content_hash
    rewritten_refs[1]["report_id"] = candidates[1].payload["report_id"]
    candidate_set_payload["candidate_ref_set_hash"] = sha256_hex(canonical_json_bytes(rewritten_refs))
    rewritten_rows = [
        {
            "event_time_utc": candidate.payload["event_time_utc"],
            "ingestion_time_utc": candidate.payload["ingestion_time_utc"],
            "model_id": candidate.payload["model_id"],
            "prompt": candidate.payload["prompt"],
            "purpose": candidate.payload["purpose"],
            "report_id": candidate.payload["report_id"],
            "response": candidate.payload["response"],
            "tool_name": candidate.payload["tool_name"],
            "trace_pk": candidate.payload["source_trace_key"],
        }
        for candidate in candidates
    ]
    rewritten_result = canonical_json_bytes(rewritten_rows)
    candidate_set_payload["boundary_result_hash"] = sha256_hex(rewritten_result)
    candidate_set_payload["boundary_result_size"] = len(rewritten_result)
    rewritten_set = store.put("SanitizedCandidateSet", "1.0.0", candidate_set_payload)

    validation = store.read(str(lineage["validation_manifest_hash"]), expected_schema_name="ValidationManifest")
    validation_payload = validation.payload
    validation_payload["candidate_set_hash"] = rewritten_set.content_hash
    rewritten_validation = store.put("ValidationManifest", "1.0.0", validation_payload)
    substitute = _dataset_with_lineage(
        store,
        dataset,
        candidate_set_hash=rewritten_set.content_hash,
        validation_manifest_hash=rewritten_validation.content_hash,
    )
    with pytest.raises(DatasetValidationError, match="report"):
        load_training_dataset(store, substitute.content_hash)


def test_workflow_rejects_duplicate_primary_report_identity(tmp_path: Path) -> None:
    item = fault_config("duplicate-report-id", source_fault="duplicate_report_id")
    terminal = finish(tmp_path, item)
    assert terminal["terminal"] is True
    assert terminal["dataset"] is None


@pytest.mark.parametrize(
    ("stage", "mutation"),
    [
        ("started", "extra"),
        ("started", "bool-purpose"),
        ("planned", "extra"),
        ("closed", "extra"),
    ],
)
def test_application_event_details_are_closed_world_on_every_resume(
    tmp_path: Path,
    stage: str,
    mutation: str,
) -> None:
    item = config(f"event-details-{stage}-{mutation}")
    GovernedDataIngestWorkflow.bootstrap(
        tmp_path,
        item,
        role_ingress=stage_data_provider(tmp_path),
        epoch=3001,
    )
    if stage == "planned":
        assert resume_process(tmp_path, item.run_id)["events"][-1] == "QUERY_PLANNED"
    elif stage == "closed":
        complete_with_fresh_process_per_stage(tmp_path, item)

    def mutate(details: dict[str, object]) -> None:
        if mutation == "bool-purpose":
            details["purpose"] = True
        else:
            details["unexpected"] = "resigned"

    _replace_head_event(tmp_path, item.run_id, mutate)
    with pytest.raises(DataIngestWorkflowError):
        GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=9001)


@pytest.mark.parametrize("directive", ["timeout", "error", "delayed", "permanent_failure"])
def test_attempt_status_matrix_rejects_resigned_result_fields(tmp_path: Path, directive: str) -> None:
    schedule = ("permanent_failure",) if directive == "permanent_failure" else (directive, "success")
    item = fault_config(f"attempt-matrix-{directive}", schedule=schedule)
    to_query_requested(tmp_path, item)
    resume_process(tmp_path, item.run_id)
    attempt_path = next((tmp_path / "private-boundary").rglob("invocations/*.json"))
    value = json.loads(attempt_path.read_bytes())
    value["result_hash"] = "f" * 64
    value["result_size"] = 1
    attempt_path.write_bytes(canonical_json_bytes(value))
    if directive == "permanent_failure":
        with pytest.raises((DataIngestWorkflowError, DataSourceBoundaryError)):
            GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3001)
    else:
        failed = GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3001)
        assert failed.events[-1].payload["event_type"] == "INGEST_FAILURE_RECORDED"
        assert len(list((tmp_path / "private-boundary").rglob("invocations/*.json"))) == 1


def test_late_delivery_arrives_after_success_and_is_idempotently_quarantined(tmp_path: Path) -> None:
    item = fault_config("out-of-order-late", schedule=("late", "success"))
    to_query_requested(tmp_path, item)
    first = resume_process(tmp_path, item.run_id)
    assert first["events"][-1] == "QUERY_RETRY_SCHEDULED"
    assert not list((tmp_path / "private-boundary").rglob("deliveries/quarantine/*.json"))
    second = resume_process(tmp_path, item.run_id)
    assert second["events"][-1] == "CANDIDATES_SANITIZED"
    quarantine = next((tmp_path / "private-boundary").rglob("deliveries/quarantine/*.json"))
    success = next((tmp_path / "private-boundary").rglob("success.canonical.json"))
    before = (quarantine.read_bytes(), success.read_bytes())
    terminal = finish(tmp_path, item)
    assert terminal["dataset"] is not None
    GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=9001)
    assert (quarantine.read_bytes(), success.read_bytes()) == before
    quarantine.unlink()
    with pytest.raises((DataIngestWorkflowError, DataSourceBoundaryError)):
        GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=9002)


@pytest.mark.parametrize("target", ["blueprint-extra", "query-extra", "query-blueprint-mismatch"])
def test_loader_rejects_resigned_source_contract_substitution(tmp_path: Path, target: str) -> None:
    dataset_hash = complete_with_fresh_process_per_stage(tmp_path, config(f"loader-contract-{target}"))
    store = ArtifactStore(tmp_path)
    dataset = store.read(dataset_hash, expected_schema_name="DatasetVersion")
    lineage = cast(dict[str, object], dataset.payload["lineage"])
    if target == "blueprint-extra":
        blueprint = store.read(str(lineage["blueprint_hash"]), expected_schema_name="DataProviderBlueprint")
        blueprint_payload = blueprint.payload
        blueprint_payload["unexpected"] = "resigned"
        replacement = store.put("DataProviderBlueprint", "1.0.0", blueprint_payload)
        substitute = _dataset_with_lineage(store, dataset, blueprint_hash=replacement.content_hash)
    else:
        plan = store.read(str(lineage["query_plan_hash"]), expected_schema_name="QueryPlan")
        plan_payload = plan.payload
        if target == "query-extra":
            plan_payload["unexpected"] = "resigned"
        else:
            plan_payload["blueprint_hash"] = "f" * 64
        replacement = store.put("QueryPlan", "1.0.0", plan_payload)
        substitute = _dataset_with_lineage(store, dataset, query_plan_hash=replacement.content_hash)
    with pytest.raises(DatasetValidationError):
        load_training_dataset(store, substitute.content_hash)


def test_loader_rejects_sentinel_in_any_resigned_public_dag_artifact(tmp_path: Path) -> None:
    dataset_hash = complete_with_fresh_process_per_stage(tmp_path, config("loader-sentinel-closure"))
    store = ArtifactStore(tmp_path)
    dataset = store.read(dataset_hash, expected_schema_name="DatasetVersion")
    lineage = cast(dict[str, object], dataset.payload["lineage"])
    blueprint = store.read(str(lineage["blueprint_hash"]), expected_schema_name="DataProviderBlueprint")
    blueprint_payload = blueprint.payload
    blueprint_payload["unexpected"] = PRIVATE_SENTINEL.lower()
    replacement = store.put("DataProviderBlueprint", "1.0.0", blueprint_payload)
    substitute = _dataset_with_lineage(store, dataset, blueprint_hash=replacement.content_hash)
    with pytest.raises(DatasetValidationError, match="sentinel"):
        load_training_dataset(store, substitute.content_hash)


@pytest.mark.parametrize(
    ("source_fault", "expected_code"),
    [
        ("time_limit", "QUERY_TIME_LIMIT_EXCEEDED"),
        ("row_limit", "QUERY_ROW_LIMIT_EXCEEDED"),
        ("byte_limit", "QUERY_BYTE_LIMIT_EXCEEDED"),
        ("missing_sentinel", "RAW_FIELD_COMPLETENESS_FAILED"),
        ("wrong_sentinel", "SENTINEL_VALIDATION_FAILED"),
        ("leak_sentinel", "SENTINEL_LEAK_DETECTED"),
        ("missing_field", "RAW_FIELD_COMPLETENESS_FAILED"),
    ],
)
def test_boundary_failure_code_is_derived_from_frozen_source_fault(
    tmp_path: Path,
    source_fault: str,
    expected_code: str,
) -> None:
    item = fault_config(f"boundary-code-{source_fault}", source_fault=source_fault)
    to_query_requested(tmp_path, item)
    failed = resume_process(tmp_path, item.run_id)
    assert failed["events"][-1] == "INGEST_FAILURE_RECORDED"
    attempt_path = next((tmp_path / "private-boundary").rglob("invocations/*.json"))
    attempt = json.loads(attempt_path.read_bytes())
    assert attempt["failure_code"] == expected_code
    attempt["failure_code"] = "QUERY_CARDINALITY_INVALID"
    attempt_path.write_bytes(canonical_json_bytes(attempt))
    with pytest.raises(DataIngestWorkflowError):
        GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3001)


@pytest.mark.parametrize(("source_fault", "raw_should_exist"), [("row_limit", False), ("missing_sentinel", True)])
def test_boundary_failure_raw_presence_is_source_fault_specific(
    tmp_path: Path,
    source_fault: str,
    raw_should_exist: bool,
) -> None:
    item = fault_config(f"boundary-raw-{source_fault}", source_fault=source_fault)
    to_query_requested(tmp_path, item)
    failed = resume_process(tmp_path, item.run_id)
    assert failed["events"][-1] == "INGEST_FAILURE_RECORDED"
    attempt_path = next((tmp_path / "private-boundary").rglob("invocations/*.json"))
    attempt = json.loads(attempt_path.read_bytes())
    raw_path = attempt_path.parents[1] / "online-traces.private.json"
    assert raw_path.exists() is raw_should_exist
    if raw_should_exist:
        raw_path.unlink()
        attempt["raw_result_hash"] = None
        attempt["raw_result_size"] = 0
        attempt["raw_row_count"] = 0
    else:
        store = ArtifactStore(tmp_path)
        journal = GovernedDataIngestWorkflow._journal(tmp_path, store, item.run_id)
        workflow_input, persisted_config = GovernedDataIngestWorkflow._input(store, journal, item.run_id)
        adapter = FixtureDataSource.restore_read_only(
            tmp_path,
            expected_blueprint_hash=str(workflow_input.payload["blueprint_hash"]),
            expected_role_audit_hash=str(workflow_input.payload["role_invocation_audit_hash"]),
            expected_correction_role_audit_hash=str(workflow_input.payload["correction_role_invocation_audit_hash"]),
            expected_skill_hash=str(workflow_input.payload["data_source_skill_hash"]),
            expected_config=persisted_config,
        )
        raw = canonical_json_bytes(adapter._generate_raw_rows())
        raw_path.write_bytes(raw)
        attempt["raw_result_hash"] = sha256_hex(raw)
        attempt["raw_result_size"] = len(raw)
        attempt["raw_row_count"] = len(adapter._generate_raw_rows())
    attempt_path.write_bytes(canonical_json_bytes(attempt))
    with pytest.raises(DataIngestWorkflowError):
        GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3001)


def test_boundary_failure_decision_must_match_private_attempt_code_and_stage(tmp_path: Path) -> None:
    item = fault_config("boundary-decision-crossref", source_fault="row_limit")
    to_query_requested(tmp_path, item)
    failed = resume_process(tmp_path, item.run_id)
    assert failed["events"][-1] == "INGEST_FAILURE_RECORDED"
    store = ArtifactStore(tmp_path)
    event_ref = sorted((tmp_path / "runs" / item.run_id / "events").glob("*.ref"))[-1]
    event = store.read(event_ref.read_text().strip(), expected_schema_name="RunEvent")
    details = cast(dict[str, object], event.payload["details"])
    observation = store.read(str(details["observation_hash"]), expected_schema_name="DataIngestFailureObservation")
    observation_payload = observation.payload
    observation_payload["failure_code"] = "QUERY_CARDINALITY_INVALID"
    replacement_observation = store.put("DataIngestFailureObservation", "1.0.0", observation_payload)
    decision = store.read(str(details["decision_record_hash"]), expected_schema_name="DecisionRecord")
    decision_payload = decision.payload
    decision_payload["reason_code"] = "QUERY_CARDINALITY_INVALID"
    decision_payload["evidence_hashes"] = [replacement_observation.content_hash]
    replacement_decision = store.put("DecisionRecord", "1.0.0", decision_payload)
    event_payload = event.payload
    event_details = cast(dict[str, object], event_payload["details"])
    event_details["decision_record_hash"] = replacement_decision.content_hash
    event_details["observation_hash"] = replacement_observation.content_hash
    replacement_event = store.put("RunEvent", "1.0.0", event_payload)
    event_ref.write_text(replacement_event.content_hash + "\n")
    with pytest.raises(DataIngestWorkflowError):
        GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3001)


def test_terminal_recertification_binds_run_closed_to_decision_path(tmp_path: Path) -> None:
    success_root = tmp_path / "success"
    success_item = config("closed-success-substitution")
    complete_with_fresh_process_per_stage(success_root, success_item)
    _substitute_run_closed(success_root, success_item.run_id, status="failed", reason_code="BENIGN_FAILURE")
    with pytest.raises(DataIngestWorkflowError):
        GovernedDataIngestWorkflow.resume(success_root, success_item.run_id, epoch=9001)

    failure_root = tmp_path / "failure"
    failure_item = fault_config("closed-failure-substitution", schedule=("permanent_failure",))
    terminal = finish(failure_root, failure_item)
    assert terminal["terminal"] is True
    _substitute_run_closed(failure_root, failure_item.run_id, status="succeeded", reason_code="BENIGN_SUCCESS")
    with pytest.raises(DataIngestWorkflowError):
        GovernedDataIngestWorkflow.resume(failure_root, failure_item.run_id, epoch=9001)


@pytest.mark.parametrize("target", ["skill", "candidate-set", "query-request"])
def test_loader_rejects_closed_world_source_dag_substitution(tmp_path: Path, target: str) -> None:
    dataset_hash = complete_with_fresh_process_per_stage(tmp_path, config(f"loader-source-dag-{target}"))
    store = ArtifactStore(tmp_path)
    dataset = store.read(dataset_hash, expected_schema_name="DatasetVersion")
    lineage = cast(dict[str, object], dataset.payload["lineage"])
    if target == "skill":
        skill = store.read(str(lineage["data_source_skill_hash"]), expected_schema_name="DataSourceSkill")
        skill_payload = skill.payload
        limits = cast(dict[str, object], skill_payload["limits"])
        limits["max_rows"] = True
        replacement = store.put("DataSourceSkill", "1.0.0", skill_payload)
        substitute = _dataset_with_lineage(store, dataset, data_source_skill_hash=replacement.content_hash)
        message = "DataSourceSkill"
    else:
        candidate_set = store.read(
            str(lineage["candidate_set_hash"]),
            expected_schema_name="SanitizedCandidateSet",
        )
        candidate_payload = candidate_set.payload
        if target == "candidate-set":
            candidate_payload["boundary_attempt_ordinal"] = True
            message = "SanitizedCandidateSet boundary contract"
        else:
            request = store.read(str(candidate_payload["query_request_hash"]), expected_schema_name="QueryRequest")
            request_payload = request.payload
            request_payload["unexpected"] = "resigned"
            replacement_request = store.put("QueryRequest", "1.0.0", request_payload)
            candidate_payload["query_request_hash"] = replacement_request.content_hash
            message = "QueryRequest"
        replacement_set = store.put("SanitizedCandidateSet", "1.0.0", candidate_payload)
        validation = store.read(
            str(lineage["validation_manifest_hash"]),
            expected_schema_name="ValidationManifest",
        )
        validation_payload = validation.payload
        validation_payload["candidate_set_hash"] = replacement_set.content_hash
        replacement_validation = store.put("ValidationManifest", "1.0.0", validation_payload)
        substitute = _dataset_with_lineage(
            store,
            dataset,
            candidate_set_hash=replacement_set.content_hash,
            validation_manifest_hash=replacement_validation.content_hash,
        )
    with pytest.raises(DatasetValidationError, match=message):
        load_training_dataset(store, substitute.content_hash)
