"""Exact regressions for independent Ticket 03 P1 review findings."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from clawrl.adapters.sources.fixture import DataSourceBoundaryError, FixtureDataSource
from clawrl.artifacts import ArtifactCorruption, ArtifactStore, canonical_json_bytes
from clawrl.data.models import DataContractError, FixtureDataIngestConfig, QueryWindow
from clawrl.data.workflow import GovernedDataIngestWorkflow
from clawrl.training.run_journal import RunIdentityConflict, RunJournal, RunJournalCorruption
from tests.contracts.test_ticket03_sql_and_dataset_count import APPROVED_SQL
from tests.e2e.test_ticket03_fault_recovery import fault_config, finish, resume_process, to_query_requested
from tests.e2e.test_ticket03_successful_dataset_workflow import complete_with_fresh_process_per_stage, config
from tests.fixtures.ticket03_data import (
    DATA_PROVIDER_CORRECTION_RAW,
    PRIVATE_SENTINEL,
    stage_data_provider,
)


def _all_files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _artifact_for_schema(root: Path, schema_name: str) -> Path:
    return next(
        path
        for path in (root / "artifacts").glob("*.json")
        if json.loads(path.read_bytes())["schema_name"] == schema_name
    )


@pytest.mark.parametrize("target", ["failure_observation", "private_role_raw", "private_config"])
def test_failed_terminal_resume_fresh_reads_every_failure_and_source_evidence(tmp_path: Path, target: str) -> None:
    item = fault_config(f"failed-recert-{target}", schedule=("permanent_failure",))
    terminal = finish(tmp_path, item)
    assert terminal["terminal"] and terminal["dataset"] is None
    if target == "failure_observation":
        decision_path = _artifact_for_schema(tmp_path, "DecisionRecord")
        decision = json.loads(decision_path.read_bytes())
        observation_hash = decision["payload"]["evidence_hashes"][0]
        observation_path = tmp_path / "artifacts" / f"{observation_hash}.json"
        value = json.loads(observation_path.read_bytes())
        value["payload"]["unexpected"] = "substitution"
        observation_path.write_bytes(canonical_json_bytes(value))
    elif target == "private_role_raw":
        role_path = next((tmp_path / "private-boundary").rglob("role-output.raw.json"))
        role_path.write_bytes(
            role_path.read_bytes().replace(b"fixture-online-traces-v1", b"fixture-online-traces-v2", 1)
        )
    else:
        config_path = next((tmp_path / "private-boundary").rglob("fixture-config.canonical.json"))
        value = json.loads(config_path.read_bytes())
        value["retry_limit"] = int(value["retry_limit"]) + 1
        config_path.write_bytes(canonical_json_bytes(value))
    with pytest.raises((ArtifactCorruption, DataContractError, DataSourceBoundaryError)):
        GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=9001)


@pytest.mark.parametrize("mismatch", ["query", "window", "packet", "raw", "session"])
def test_terminal_run_once_rejects_different_supplied_identity_without_any_write(
    tmp_path: Path,
    mismatch: str,
) -> None:
    item = config("terminal-identity")
    complete_with_fresh_process_per_stage(tmp_path, item)
    ingress = stage_data_provider(tmp_path)
    supplied = item
    if mismatch == "query":
        supplied = config(item.run_id, query_sql="  " + APPROVED_SQL)
    elif mismatch == "window":
        supplied = FixtureDataIngestConfig(
            run_id=item.run_id,
            query_sql=APPROVED_SQL,
            window=QueryWindow("2025-12-31T23:59:59Z", "2026-01-02T00:00:00Z"),
        )
    elif mismatch == "packet":
        ingress = replace(ingress, input_packet_hash="e" * 64)
    elif mismatch == "raw":
        ingress = replace(ingress, raw_output_hash="d" * 64)
    else:
        ingress = replace(ingress, role_invocation_audit_hash="c" * 64)
    before = _all_files(tmp_path)
    with pytest.raises(RunIdentityConflict):
        GovernedDataIngestWorkflow.run_once(
            tmp_path,
            supplied,
            role_ingress=ingress,
            epoch=9999,
        )
    assert _all_files(tmp_path) == before


@pytest.mark.parametrize(
    "run_id",
    [
        PRIVATE_SENTINEL,
        PRIVATE_SENTINEL.lower(),
        "t03-private-sentinel-never-export",
    ],
)
def test_public_sentinel_variant_is_rejected_before_first_file(run_id: str, tmp_path: Path) -> None:
    item = config(run_id)
    ingress = stage_data_provider(tmp_path)
    before = _all_files(tmp_path)
    with pytest.raises(DataContractError, match="sentinel closure"):
        GovernedDataIngestWorkflow.bootstrap(
            tmp_path,
            item,
            role_ingress=ingress,
            epoch=3001,
        )
    assert _all_files(tmp_path) == before
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        (PRIVATE_SENTINEL, "DATA_SOURCE_BOUNDARY_FAILED"),
        ("QUERY_ROW_LIMIT_EXCEEDED", "QUERY_ROW_LIMIT_EXCEEDED"),
    ],
)
def test_boundary_exception_reason_uses_frozen_code_enum_without_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
    expected: str,
) -> None:
    item = fault_config(f"reason-{expected.lower()}")
    to_query_requested(tmp_path, item)

    def fail(*args: object, **kwargs: object) -> object:
        raise DataSourceBoundaryError(error_code)

    monkeypatch.setattr(FixtureDataSource, "query", fail)
    snapshot = GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3001)
    assert snapshot.decision_record is not None
    assert snapshot.decision_record.payload["reason_code"] == expected
    public_files = {name: raw for name, raw in _all_files(tmp_path).items() if not name.startswith("private-boundary/")}
    assert all(PRIVATE_SENTINEL.encode() not in raw for raw in public_files.values())


def test_config_window_must_equal_provider_window_before_first_write(tmp_path: Path) -> None:
    item = FixtureDataIngestConfig(
        run_id="window-preflight",
        query_sql=APPROVED_SQL,
        window=QueryWindow("2025-12-31T23:59:59Z", "2026-01-02T00:00:00Z"),
    )
    ingress = stage_data_provider(tmp_path)
    before = _all_files(tmp_path)
    with pytest.raises(DataContractError, match="window"):
        GovernedDataIngestWorkflow.bootstrap(
            tmp_path,
            item,
            role_ingress=ingress,
            epoch=3001,
        )
    assert _all_files(tmp_path) == before
    assert not (tmp_path / "runs").exists()


def test_query_plan_request_and_skill_repeat_one_frozen_window_hash(tmp_path: Path) -> None:
    item = config("window-lineage")
    to_query_requested(tmp_path, item)
    store = ArtifactStore(tmp_path)
    plan = store.read(
        next(
            str(cast(dict[str, object], event.payload["details"])["query_plan_hash"])
            for event in RunJournal(
                tmp_path,
                store,
                item.run_id,
                input_schema_name="DataIngestWorkflowInput",
                input_schema_version="1.0.0",
            ).events()
            if event.payload["event_type"] == "QUERY_PLANNED"
        ),
        expected_schema_name="QueryPlan",
    )
    request = store.read(
        next(
            str(cast(dict[str, object], event.payload["details"])["query_request_hash"])
            for event in RunJournal(
                tmp_path,
                store,
                item.run_id,
                input_schema_name="DataIngestWorkflowInput",
                input_schema_version="1.0.0",
            ).events()
            if event.payload["event_type"] == "QUERY_REQUESTED"
        ),
        expected_schema_name="QueryRequest",
    )
    skill = store.read(str(plan.payload["data_source_skill_hash"]), expected_schema_name="DataSourceSkill")
    assert plan.payload["window_hash"] == request.payload["window_hash"] == skill.payload["window_hash"]


def test_attempt_evidence_is_versioned_chained_and_missing_history_aborts_without_new_attempt(tmp_path: Path) -> None:
    item = fault_config("attempt-chain", schedule=("timeout", "success"))
    to_query_requested(tmp_path, item)
    retry = resume_process(tmp_path, item.run_id)
    assert retry["events"][-1] == "QUERY_RETRY_SCHEDULED"
    attempt = next((tmp_path / "private-boundary").rglob("invocations/*.json"))
    payload = json.loads(attempt.read_bytes())
    assert set(payload) >= {
        "attempt_ordinal",
        "directive",
        "input_hash",
        "previous_attempt_hash",
        "query_hash",
        "request_hash",
        "schedule_hash",
        "schema_version",
    }
    attempt.unlink()
    state = resume_process(tmp_path, item.run_id)
    assert state["events"][-1] == "INGEST_FAILURE_RECORDED"
    assert state["dataset"] is None
    assert not list((tmp_path / "private-boundary").rglob("invocations/*.json"))


def test_success_receipt_is_bound_to_attempt_schedule_query_and_input(tmp_path: Path) -> None:
    item = config("receipt-binding")
    complete_with_fresh_process_per_stage(tmp_path, item)
    receipt_path = next((tmp_path / "private-boundary").rglob("success.canonical.json"))
    receipt = json.loads(receipt_path.read_bytes())
    assert set(receipt) >= {"attempt_hash", "input_hash", "query_hash", "schedule_hash"}
    receipt["attempt_hash"] = "f" * 64
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    with pytest.raises(DataSourceBoundaryError):
        GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=9001)


def test_auxiliary_observation_is_verified_before_lifecycle_progress(tmp_path: Path) -> None:
    item = config("aux-integrity")
    GovernedDataIngestWorkflow.bootstrap(
        tmp_path,
        item,
        role_ingress=stage_data_provider(tmp_path),
        epoch=3001,
    )
    store = ArtifactStore(tmp_path)
    journal = RunJournal(
        tmp_path,
        store,
        item.run_id,
        input_schema_name="DataIngestWorkflowInput",
        input_schema_version="1.0.0",
    )
    receipt = journal.record_observation(3001, "HEARTBEAT", {"sequence": 1})
    value = json.loads(receipt.observation.path.read_bytes())
    value["payload"]["run_id"] = "substituted"
    receipt.observation.path.write_bytes(canonical_json_bytes(value))
    before = len(list((tmp_path / "runs" / item.run_id / "events").glob("*.ref")))
    with pytest.raises(RunJournalCorruption):
        GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3001)
    assert len(list((tmp_path / "runs" / item.run_id / "events").glob("*.ref"))) == before


def test_skill_and_selection_publish_the_frozen_observed_data_mix(tmp_path: Path) -> None:
    dataset_hash = complete_with_fresh_process_per_stage(tmp_path, config("approved-mix"))
    store = ArtifactStore(tmp_path)
    dataset = store.read(dataset_hash, expected_schema_name="DatasetVersion")
    lineage = dataset.payload["lineage"]
    assert isinstance(lineage, dict)
    skill = store.read(str(lineage["data_source_skill_hash"]), expected_schema_name="DataSourceSkill")
    selection = store.read(str(lineage["selection_manifest_hash"]), expected_schema_name="BadcaseSelectionManifest")
    approved = skill.payload["approved_data_mix"]
    assert approved == {
        "model": {"fixture-model-a": 50, "fixture-model-b": 50},
        "policy_version": "fixture-online-trace-mix-v1",
        "purpose": {"training_allowed": 100},
        "schema_version": "approved-data-mix/1.0.0",
        "selected_count": 100,
        "tool": {"javascript": 25, "none": 25, "python": 25, "regex": 25},
    }
    assert selection.payload["approved_data_mix_hash"] == lineage["approved_data_mix_hash"]
    assert selection.payload["mix"] == {
        "model": approved["model"],
        "purpose": approved["purpose"],
        "tool": approved["tool"],
    }


def test_correction_exchange_preserves_exact_raw_and_separate_normalization(tmp_path: Path) -> None:
    ingress = stage_data_provider(tmp_path)
    private_root = tmp_path / "private-boundary" / "data-provider-ingress" / ingress.ingress_hash
    raw = (private_root / "correction-role-output.raw.json").read_bytes()
    normalized = (private_root / "correction-role-output.normalized.canonical.json").read_bytes()
    assert raw == DATA_PROVIDER_CORRECTION_RAW
    assert normalized == canonical_json_bytes(json.loads(raw))
    assert normalized != raw
    store = ArtifactStore(tmp_path)
    audit = store.read(
        ingress.correction_role_invocation_audit_hash,
        expected_schema_name="RoleInvocationAudit",
    )
    assert audit.payload["raw_output_hash"] == ingress.correction_raw_output_hash
    assert audit.payload["previous_output_hash"] == ingress.raw_output_hash
    assert audit.payload["normalized_output_hash"] != audit.payload["raw_output_hash"]
    skill = FixtureDataSource.load_public_contract(tmp_path, ingress).base_skill
    assert skill.primary_key == "report_id"
    assert skill.dedupe_key == "trace_pk"
    assert dict(skill.field_mapping)["primary_key"] == "report_id"


def test_dataset_identity_closes_both_data_provider_exchanges(tmp_path: Path) -> None:
    dataset_hash = complete_with_fresh_process_per_stage(tmp_path, config("dual-provider-lineage"))
    store = ArtifactStore(tmp_path)
    dataset = store.read(dataset_hash, expected_schema_name="DatasetVersion")
    lineage = cast(dict[str, object], dataset.payload["lineage"])
    base = store.read(str(lineage["role_invocation_audit_hash"]), expected_schema_name="RoleInvocationAudit")
    correction = store.read(
        str(lineage["correction_role_invocation_audit_hash"]),
        expected_schema_name="RoleInvocationAudit",
    )
    plan = store.read(str(lineage["query_plan_hash"]), expected_schema_name="QueryPlan")
    assert correction.payload["previous_output_hash"] == base.payload["raw_output_hash"]
    assert plan.payload["role_invocation_audit_hash"] == base.content_hash
    assert plan.payload["correction_role_invocation_audit_hash"] == correction.content_hash
