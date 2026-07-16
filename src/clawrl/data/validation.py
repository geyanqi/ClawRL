"""Fail-closed consumers for immutable DatasetVersion DAGs."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.data.domain import (
    CANDIDATE_FIELDS,
    TRAINING_TRACE_FIELDS,
    assert_data_only,
    dataset_version_id,
    selection_score,
    trace_set_hash,
    training_trace_payload,
)
from clawrl.data.models import ApprovedDataMix, DataContractError, DatasetValidationError, DataSourceSkill, QueryWindow
from clawrl.data.safety import assert_public_sentinel_free

_HASH = re.compile(r"^[0-9a-f]{64}$")
_TRACE_ID = re.compile(r"^tt-[0-9a-f]{40}$")
_BASE_NORMALIZED_OUTPUT_HASH = "5f09019f6c81569bcad5079ac3134c0f6a741f3a886f347bf7c2d2bccd70eac5"
_CORRECTION_NORMALIZED_OUTPUT_HASH = "30dd338b9d3024186b4cf063452f04bda7fc44db1d0d274413ebd194c21bc363"
_DATASET_FIELDS = {
    "dataset_version_id",
    "lineage",
    "purpose",
    "source_id",
    "trace_count",
    "trace_refs",
    "trace_set_hash",
}
_LINEAGE_FIELDS = {
    "approved_data_mix_hash",
    "approved_data_mix_policy_version",
    "blueprint_hash",
    "candidate_set_hash",
    "correction_role_invocation_audit_hash",
    "data_source_skill_hash",
    "dedupe_manifest_hash",
    "query_plan_hash",
    "query_source_hash",
    "role_invocation_audit_hash",
    "sanitizer_policy_version",
    "selection_manifest_hash",
    "selection_policy_version",
    "validation_manifest_hash",
}


@dataclass(frozen=True, slots=True)
class LoadedTrainingDataset:
    dataset_version: Artifact
    training_traces: tuple[Artifact, ...]


def expected_validation_manifest(
    candidate_set: Artifact,
    candidates: list[Artifact],
    *,
    window_start_utc: str,
    window_end_utc: str,
) -> dict[str, JsonValue]:
    """Recompute the complete validation evidence from persisted candidates."""

    if len(candidates) != 103:
        raise DatasetValidationError("candidate count is not 103")
    report_ids: set[str] = set()
    logical_trace_keys: set[str] = set()
    for candidate in candidates:
        values = {
            key: candidate.payload.get(key)
            for key in (
                "event_time_utc",
                "ingestion_time_utc",
                "model_id",
                "prompt",
                "purpose",
                "report_id",
                "response",
                "source_trace_key",
                "tool_name",
            )
        }
        if any(type(value) is not str or not value for value in values.values()):
            raise DatasetValidationError("candidate is incomplete")
        event = cast(str, values["event_time_utc"])
        ingestion = cast(str, values["ingestion_time_utc"])
        try:
            ingestion_time = datetime.strptime(ingestion, "%Y-%m-%dT%H:%M:%SZ")
            event_time = datetime.strptime(event, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as error:
            raise DatasetValidationError("candidate time is malformed") from error
        if (
            not window_start_utc < event < window_end_utc
            or ingestion_time <= event_time
            or values["purpose"] != "training_allowed"
        ):
            raise DatasetValidationError("candidate post-query invariant failed")
        report_id = cast(str, values["report_id"])
        if report_id in report_ids:
            raise DatasetValidationError("primary report_id is duplicated")
        report_ids.add(report_id)
        logical_trace_keys.add(cast(str, values["source_trace_key"]))
    if len(report_ids) != 103:
        raise DatasetValidationError("primary report_id cardinality is not 103")
    if len(logical_trace_keys) != 100:
        raise DatasetValidationError("logical trace_pk cardinality is not 100")
    return {
        "candidate_set_hash": candidate_set.content_hash,
        "checks": [
            {"check": "row_count", "observed": 103, "status": "passed"},
            {"check": "field_completeness", "observed": 103, "status": "passed"},
            {"check": "strict_utc_window", "observed": 103, "status": "passed"},
            {"check": "purpose", "observed": 103, "status": "passed"},
            {"check": "primary_report_uniqueness", "observed": 103, "status": "passed"},
            {"check": "logical_trace_cardinality", "observed": 100, "status": "passed"},
        ],
        "sanitized_artifact_bytes": sum(len(candidate.raw_bytes) for candidate in candidates),
        "status": "passed",
        "validated_count": 103,
    }


def expected_dedupe_manifest(
    candidate_set: Artifact,
    validation: Artifact,
    candidates: list[Artifact],
) -> dict[str, JsonValue]:
    """Recompute ordered duplicate groups and highest-ingestion winners."""

    grouped: dict[str, list[Artifact]] = {}
    for candidate in candidates:
        key = candidate.payload.get("source_trace_key")
        if type(key) is not str:
            raise DatasetValidationError("dedupe key is invalid")
        grouped.setdefault(cast(str, key), []).append(candidate)
    winners: list[dict[str, str]] = []
    duplicate_groups: list[dict[str, JsonValue]] = []
    semantics = ("event_time_utc", "model_id", "prompt", "purpose", "response", "tool_name")
    for key in sorted(grouped):
        group = grouped[key]
        baseline = tuple(group[0].payload.get(field) for field in semantics)
        if any(tuple(candidate.payload.get(field) for field in semantics) != baseline for candidate in group):
            raise DatasetValidationError("duplicate semantic conflict")
        ordered = sorted(
            group,
            key=lambda candidate: (
                cast(str, candidate.payload.get("ingestion_time_utc")),
                candidate.content_hash,
            ),
        )
        ingestion_values = [candidate.payload.get("ingestion_time_utc") for candidate in ordered]
        if any(type(value) is not str for value in ingestion_values) or len(set(ingestion_values)) != len(ordered):
            raise DatasetValidationError("duplicate ingestion ordering is ambiguous")
        winner = ordered[-1]
        winners.append({"artifact_hash": winner.content_hash, "source_trace_key": key})
        if len(ordered) > 1:
            duplicate_groups.append(
                {
                    "discarded_hashes": [candidate.content_hash for candidate in ordered[:-1]],
                    "source_trace_key": key,
                    "winner_hash": winner.content_hash,
                }
            )
    if len(candidates) != 103 or len(winners) != 100 or len(duplicate_groups) != 3:
        raise DatasetValidationError("dedupe cardinality failed")
    return {
        "candidate_set_hash": candidate_set.content_hash,
        "duplicate_count": 3,
        "duplicate_groups": cast(JsonValue, duplicate_groups),
        "input_count": 103,
        "semantics": "highest_ingestion_with_identical_sanitized_content",
        "unique_count": 100,
        "validation_manifest_hash": validation.content_hash,
        "winner_refs": cast(JsonValue, winners),
    }


def expected_selection_manifest(
    store: ArtifactStore,
    dedupe: Artifact,
    *,
    policy: str,
    approved_mix: ApprovedDataMix,
) -> dict[str, JsonValue]:
    """Recompute ranking, score contract, and the complete observed mix."""

    winners = dedupe.payload.get("winner_refs")
    if not isinstance(winners, list):
        raise DatasetValidationError("dedupe winners are invalid")
    scored: list[tuple[str, str, str, Artifact]] = []
    for winner in winners:
        if not isinstance(winner, dict) or set(winner) != {"artifact_hash", "source_trace_key"}:
            raise DatasetValidationError("dedupe winner is invalid")
        artifact_hash = winner.get("artifact_hash")
        key = winner.get("source_trace_key")
        if type(artifact_hash) is not str or type(key) is not str:
            raise DatasetValidationError("dedupe winner identity is invalid")
        candidate = store.read(cast(str, artifact_hash), expected_schema_name="SanitizedCandidate")
        scored.append((selection_score(candidate.payload, policy), cast(str, key), cast(str, artifact_hash), candidate))
    scored.sort(key=lambda value: (value[0], value[1]), reverse=True)
    tools: Counter[str] = Counter()
    models: Counter[str] = Counter()
    purposes: Counter[str] = Counter()
    rankings: list[dict[str, JsonValue]] = []
    for rank, (score, key, artifact_hash, candidate) in enumerate(scored, 1):
        rankings.append(
            {
                "artifact_hash": artifact_hash,
                "rank": rank,
                "selected": True,
                "selection_score": score,
                "source_trace_key": key,
            }
        )
        for field, counter in (
            ("tool_name", tools),
            ("model_id", models),
            ("purpose", purposes),
        ):
            value = candidate.payload.get(field)
            if type(value) is not str:
                raise DatasetValidationError("selection mix field is invalid")
            counter[cast(str, value)] += 1
    observed_mix = {
        "model": dict(sorted(models.items())),
        "purpose": dict(sorted(purposes.items())),
        "tool": dict(sorted(tools.items())),
    }
    approved = approved_mix.artifact_payload()
    if (
        observed_mix
        != {
            "model": approved["model"],
            "purpose": approved["purpose"],
            "tool": approved["tool"],
        }
        or len(rankings) != approved_mix.selected_count
    ):
        raise DatasetValidationError("selected content violates the approved data mix")
    return {
        "approved_data_mix_hash": approved_mix.content_hash,
        "approved_data_mix_policy_version": approved_mix.policy_version,
        "dedupe_manifest_hash": dedupe.content_hash,
        "eligible_count": 100,
        "mix": cast(JsonValue, observed_mix),
        "rankings": cast(JsonValue, rankings),
        "score_contract": "sha256-over-sanitized-semantic-content",
        "selected_count": 100,
        "selection_policy_version": policy,
    }


def load_training_dataset(store: ArtifactStore, dataset_hash: str) -> LoadedTrainingDataset:
    """Resolve and fully verify the DatasetVersion DAG for an RL dataloader."""

    if (
        not isinstance(store, ArtifactStore)
        or not isinstance(dataset_hash, str)
        or _HASH.fullmatch(dataset_hash) is None
    ):
        raise DatasetValidationError("DatasetVersion reference is invalid")
    try:
        dataset = store.read(dataset_hash, expected_schema_name="DatasetVersion")
    except Exception as error:
        raise DatasetValidationError("DatasetVersion cannot be resolved") from error
    if dataset.schema_version != "1.0.0" or set(dataset.payload) != _DATASET_FIELDS:
        raise DatasetValidationError("DatasetVersion schema fields are invalid")
    payload = dataset.payload
    _assert_public_data(payload)
    assert_data_only(payload)
    if payload.get("purpose") != "training_allowed":
        raise DatasetValidationError("only training_allowed DatasetVersion may enter the RL loader")
    if payload.get("source_id") != "fixture-online-traces-v1":
        raise DatasetValidationError("DatasetVersion source identity is invalid")
    trace_count = payload.get("trace_count")
    trace_refs = payload.get("trace_refs")
    lineage = payload.get("lineage")
    if (
        not isinstance(trace_count, int)
        or isinstance(trace_count, bool)
        or trace_count != 100
        or not isinstance(trace_refs, list)
        or len(trace_refs) != 100
        or not isinstance(lineage, dict)
        or set(lineage) != _LINEAGE_FIELDS
    ):
        raise DatasetValidationError("DatasetVersion cardinality or lineage is invalid")
    normalized_refs: list[dict[str, str]] = []
    traces: list[Artifact] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for ref in trace_refs:
        if not isinstance(ref, dict) or set(ref) != {"artifact_hash", "trace_id"}:
            raise DatasetValidationError("TrainingTrace ref fields are invalid")
        artifact_hash = ref.get("artifact_hash")
        trace_id = ref.get("trace_id")
        if (
            not isinstance(artifact_hash, str)
            or _HASH.fullmatch(artifact_hash) is None
            or not isinstance(trace_id, str)
            or _TRACE_ID.fullmatch(trace_id) is None
            or artifact_hash in seen_hashes
            or trace_id in seen_ids
        ):
            raise DatasetValidationError("TrainingTrace refs are malformed or duplicated")
        try:
            trace = store.read(artifact_hash, expected_schema_name="TrainingTrace")
        except Exception as error:
            raise DatasetValidationError("TrainingTrace cannot be resolved") from error
        if (
            trace.schema_version != "1.0.0"
            or set(trace.payload) != TRAINING_TRACE_FIELDS
            or trace.payload.get("trace_id") != trace_id
            or trace.payload.get("purpose") != "training_allowed"
        ):
            raise DatasetValidationError("TrainingTrace schema or purpose is invalid")
        assert_data_only(trace.payload)
        _assert_public_data(trace.payload)
        normalized_refs.append({"artifact_hash": artifact_hash, "trace_id": trace_id})
        seen_hashes.add(artifact_hash)
        seen_ids.add(trace_id)
        traces.append(trace)
    if normalized_refs != trace_refs or payload.get("trace_set_hash") != trace_set_hash(normalized_refs):
        raise DatasetValidationError("DatasetVersion trace-set identity is invalid")
    identity_payload = dict(payload)
    stored_id = identity_payload.pop("dataset_version_id", None)
    if stored_id != dataset_version_id(identity_payload):
        raise DatasetValidationError("DatasetVersion domain identity is invalid")
    _verify_lineage(store, cast(dict[str, object], lineage), traces)
    return LoadedTrainingDataset(dataset, tuple(traces))


def validate_dataset_for_experiment(store: ArtifactStore, dataset_hash: str) -> LoadedTrainingDataset:
    """ExperimentSpec's independent purpose/DAG validation seam."""

    loaded = load_training_dataset(store, dataset_hash)
    if loaded.dataset_version.payload.get("purpose") != "training_allowed":
        raise DatasetValidationError("ExperimentSpec cannot bind a non-training DatasetVersion")
    return loaded


def _read_lineage(store: ArtifactStore, lineage: dict[str, object], key: str, schema: str) -> Artifact:
    value = lineage.get(key)
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise DatasetValidationError(f"DatasetVersion {key} is invalid")
    try:
        artifact = store.read(value, expected_schema_name=schema)
    except (ArtifactCorruption, RuntimeError) as error:
        raise DatasetValidationError(f"DatasetVersion {key} cannot be resolved") from error
    if artifact.schema_version != "1.0.0":
        raise DatasetValidationError(f"DatasetVersion {key} schema version is invalid")
    assert_data_only(artifact.payload)
    _assert_public_data(artifact.payload)
    return artifact


def _assert_public_data(value: object) -> None:
    try:
        assert_public_sentinel_free(value)
    except DataContractError as error:
        raise DatasetValidationError("public DatasetVersion DAG violates sentinel closure") from error


def _verify_data_provider_audit_lineage(
    blueprint: Artifact,
    role_audit: Artifact,
    correction_audit: Artifact,
) -> None:
    """Verify both isolated role exchanges and their correction chain."""

    base_fields = {
        "input_hash",
        "input_packet_hash",
        "normalized_output_hash",
        "normalized_output_size",
        "output_hash",
        "packet_id",
        "raw_output_hash",
        "raw_output_size",
        "role_session_lineage",
        "role_type",
        "seed",
    }
    correction_fields = base_fields | {"previous_output_hash", "prior_packet_id"}
    base = role_audit.payload
    correction = correction_audit.payload
    base_hashes = (
        base.get("input_hash"),
        base.get("input_packet_hash"),
        base.get("normalized_output_hash"),
        base.get("output_hash"),
        base.get("raw_output_hash"),
    )
    correction_hashes = (
        correction.get("input_hash"),
        correction.get("input_packet_hash"),
        correction.get("normalized_output_hash"),
        correction.get("output_hash"),
        correction.get("previous_output_hash"),
        correction.get("raw_output_hash"),
    )
    if (
        set(base) != base_fields
        or set(correction) != correction_fields
        or any(type(value) is not str or _HASH.fullmatch(cast(str, value)) is None for value in base_hashes)
        or any(type(value) is not str or _HASH.fullmatch(cast(str, value)) is None for value in correction_hashes)
        or type(base.get("raw_output_size")) is not int
        or cast(int, base.get("raw_output_size")) <= 0
        or type(base.get("normalized_output_size")) is not int
        or cast(int, base.get("normalized_output_size")) <= 0
        or type(correction.get("raw_output_size")) is not int
        or cast(int, correction.get("raw_output_size")) <= 0
        or type(correction.get("normalized_output_size")) is not int
        or cast(int, correction.get("normalized_output_size")) <= 0
        or base.get("normalized_output_hash") != _BASE_NORMALIZED_OUTPUT_HASH
        or base.get("normalized_output_size") != 3410
        or correction.get("normalized_output_hash") != _CORRECTION_NORMALIZED_OUTPUT_HASH
        or correction.get("normalized_output_size") != 861
        or type(base.get("seed")) is not int
        or type(correction.get("seed")) is not int
        or base.get("seed") != 3001
        or correction.get("seed") != 3001
        or base.get("input_hash") != base.get("input_packet_hash")
        or base.get("output_hash") != base.get("raw_output_hash")
        or correction.get("input_hash") != correction.get("input_packet_hash")
        or correction.get("output_hash") != correction.get("raw_output_hash")
        or correction.get("previous_output_hash") != base.get("raw_output_hash")
        or base.get("packet_id") != "dp-ticket03-0001"
        or correction.get("packet_id") != "dp-ticket03-correction-0001"
        or correction.get("prior_packet_id") != base.get("packet_id")
        or base.get("role_type") != "DataProvider"
        or correction.get("role_type") != "DataProviderCorrection"
        or base.get("role_session_lineage") != "/root/ticket03_data_provider"
        or correction.get("role_session_lineage") != "/root/ticket03_data_provider_correction"
    ):
        raise DatasetValidationError("Data Provider role audit lineage is invalid")
    expected_blueprint = {
        "columns": [
            "trace_pk",
            "report_id",
            "event_time_utc",
            "ingestion_time_utc",
            "purpose",
            "prompt",
            "response",
            "tool_name",
            "model_id",
            "private_sentinel",
        ],
        "correction": {
            "corrected_output_hash": correction["raw_output_hash"],
            "dedupe_key": "trace_pk",
            "logical_trace_key": "trace_pk",
            "normalized_output_hash": correction["normalized_output_hash"],
            "packet_id": "dp-ticket03-correction-0001",
            "primary_key": "report_id",
            "prior_packet_id": "dp-ticket03-0001",
            "reason_code": "SOURCE_KEY_CONTRACT_CORRECTED",
            "report_identity_rule_hash": "0d4b96658a8c9ec2ad941ab88efa13fcd4b696d75aa894367599091510755296",
        },
        "dedupe_semantics_hash": "2967e86148f6e331ffd952e73d1ca3aa953c205a7a71bb45dbf00ed0d293881b",
        "duplicate_count": 3,
        "duplicate_overrides": [
            {
                "ingestion_lag_seconds": 3600,
                "logical_index": 7,
                "report_id": "report-100",
                "trace_pk": "trace-007",
            },
            {
                "ingestion_lag_seconds": 3601,
                "logical_index": 42,
                "report_id": "report-101",
                "trace_pk": "trace-042",
            },
            {
                "ingestion_lag_seconds": 3602,
                "logical_index": 99,
                "report_id": "report-102",
                "trace_pk": "trace-099",
            },
        ],
        "field_mapping": {
            "dedupe_key": "trace_pk",
            "event_time": "event_time_utc",
            "ingestion_time": "ingestion_time_utc",
            "logical_trace_key": "trace_pk",
            "model": "model_id",
            "primary_key": "report_id",
            "prompt": "prompt",
            "response": "response",
            "tool": "tool_name",
        },
        "formula": {
            "execution_permitted": False,
            "language": "javascript",
            "source_hash": "4076bd42ff71b18239a4e74e9488668131bb6de8b301b1bdb8ef0ea061f64662",
        },
        "invariant_hashes": [
            "b05cfcdcc354734579914128646c01ee8c969a4925653f15b20d8e07303d0da8",
            "cff9608570d913467f2fb726daacfd2bef335d35ba1f656c17bc1727c42d8757",
            "e454a0017bcf14704718b2b6aca1000c5c318fdc3ff2a0bb093a6bb2b86e0e14",
            "692ce2b017885472af219e4c7cf56a7af54d1bdbcada043c327f7267c2708e8e",
            "5a0073048e87b54acff18857ac3df3bd9f15a18741424a058e5636064b8aa359",
            "193134f540652864c885e753d0ab3295ee8fde2beca6ef9eac3ce22b5a27b8f0",
            "b878edcb7859278226e32a1d888fab38cb1d1bf1c3b88a7f0688f31511582132",
            "065c35a6e5b533916b7c4b7d49524310c57b95c052a5db7b0e412c4ddc124242",
        ],
        "packet_id": "dp-ticket03-0001",
        "purpose": "training_allowed",
        "row_count": 103,
        "sanitizer": {
            "expected_value_hash": "ea55483587214b66e63d5172b8df9f5dd2563af21fe3873561c414e96f6673dc",
            "sensitive_columns": ["private_sentinel"],
            "version": "sentinel-drop-v1",
        },
        "schema_version": "data-provider-blueprint/1.0.0",
        "seed": 3001,
        "source_id": "fixture-online-traces-v1",
        "table": "fixture.online_trace",
        "unique_key_count": 100,
        "window": {"end_utc": "2026-01-02T00:00:00Z", "start_utc": "2026-01-01T00:00:00Z"},
    }
    if blueprint.schema_version != "1.0.0" or blueprint.payload != expected_blueprint:
        raise DatasetValidationError("Data Provider corrected source-key contract is invalid")


def _verify_lineage(store: ArtifactStore, lineage: dict[str, object], traces: list[Artifact]) -> None:
    blueprint = _read_lineage(store, lineage, "blueprint_hash", "DataProviderBlueprint")
    skill = _read_lineage(store, lineage, "data_source_skill_hash", "DataSourceSkill")
    role_audit = _read_lineage(store, lineage, "role_invocation_audit_hash", "RoleInvocationAudit")
    correction_audit = _read_lineage(
        store,
        lineage,
        "correction_role_invocation_audit_hash",
        "RoleInvocationAudit",
    )
    query_plan = _read_lineage(store, lineage, "query_plan_hash", "QueryPlan")
    candidate_set = _read_lineage(store, lineage, "candidate_set_hash", "SanitizedCandidateSet")
    validation = _read_lineage(store, lineage, "validation_manifest_hash", "ValidationManifest")
    dedupe = _read_lineage(store, lineage, "dedupe_manifest_hash", "DedupeManifest")
    selection = _read_lineage(store, lineage, "selection_manifest_hash", "BadcaseSelectionManifest")
    try:
        decoded_skill = DataSourceSkill.from_mapping(cast(dict[str, object], skill.payload))
    except (DataContractError, TypeError, ValueError) as error:
        raise DatasetValidationError("DataSourceSkill closed-world contract is invalid") from error
    if skill.payload != decoded_skill.artifact_payload():
        raise DatasetValidationError("DataSourceSkill artifact is not canonical")
    if blueprint.payload.get("source_id") != "fixture-online-traces-v1":
        raise DatasetValidationError("Data Provider blueprint source is invalid")
    sanitizer = lineage.get("sanitizer_policy_version")
    selection_policy = lineage.get("selection_policy_version")
    query_source_hash = lineage.get("query_source_hash")
    if (
        sanitizer not in {"sentinel-drop-v1", "sentinel-drop-v1.1"}
        or selection_policy not in {"badcase-sha256-v1", "badcase-sha256-v2"}
        or not isinstance(query_source_hash, str)
        or _HASH.fullmatch(query_source_hash) is None
        or query_plan.payload.get("query_source_hash") != query_source_hash
        or query_plan.payload.get("data_source_skill_hash") != skill.content_hash
        or query_plan.payload.get("role_invocation_audit_hash") != role_audit.content_hash
        or query_plan.payload.get("correction_role_invocation_audit_hash") != correction_audit.content_hash
        or lineage.get("approved_data_mix_hash") != skill.payload.get("approved_data_mix_hash")
        or lineage.get("approved_data_mix_policy_version")
        != cast(dict[str, object], skill.payload.get("approved_data_mix", {})).get("policy_version")
    ):
        raise DatasetValidationError("DatasetVersion frozen policy lineage is inconsistent")
    _verify_data_provider_audit_lineage(blueprint, role_audit, correction_audit)
    _verify_query_plan(
        query_plan,
        blueprint=blueprint,
        skill=skill,
        role_audit=role_audit,
        correction_audit=correction_audit,
        lineage=lineage,
    )
    candidate_refs = candidate_set.payload.get("candidate_refs")
    if (
        candidate_set.payload.get("query_plan_hash") != query_plan.content_hash
        or candidate_set.payload.get("candidate_count") != 103
        or not isinstance(candidate_refs, list)
        or len(candidate_refs) != 103
    ):
        raise DatasetValidationError("SanitizedCandidateSet is invalid")
    candidates: dict[str, tuple[str, Artifact]] = {}
    candidate_list: list[Artifact] = []
    for ref in candidate_refs:
        if not isinstance(ref, dict) or set(ref) != {"artifact_hash", "report_id", "source_trace_key"}:
            raise DatasetValidationError("SanitizedCandidate ref fields are invalid")
        artifact_hash = ref.get("artifact_hash")
        source_key = ref.get("source_trace_key")
        report_id = ref.get("report_id")
        if not all(isinstance(value, str) for value in (artifact_hash, source_key, report_id)):
            raise DatasetValidationError("SanitizedCandidate ref values are invalid")
        try:
            candidate = store.read(cast(str, artifact_hash), expected_schema_name="SanitizedCandidate")
        except Exception as error:
            raise DatasetValidationError("SanitizedCandidate cannot be resolved") from error
        if (
            candidate.schema_version != "1.0.0"
            or set(candidate.payload) != CANDIDATE_FIELDS
            or candidate.payload.get("source_trace_key") != source_key
            or candidate.payload.get("report_id") != report_id
            or candidate.payload.get("query_plan_hash") != query_plan.content_hash
            or candidate.payload.get("sanitizer_version") != sanitizer
        ):
            raise DatasetValidationError("SanitizedCandidate is inconsistent")
        assert_data_only(candidate.payload)
        _assert_public_data(candidate.payload)
        key = f"{source_key}\0{report_id}"
        if key in candidates:
            raise DatasetValidationError("SanitizedCandidate refs are duplicated")
        candidates[key] = (cast(str, artifact_hash), candidate)
        candidate_list.append(candidate)
    candidate_set_hash = sha256_hex(canonical_json_bytes(candidate_refs))
    if candidate_set.payload.get("candidate_ref_set_hash") != candidate_set_hash:
        raise DatasetValidationError("SanitizedCandidate set hash is invalid")
    _verify_candidate_set_boundary_contract(
        store,
        candidate_set=candidate_set,
        query_plan=query_plan,
        sanitizer=cast(str, sanitizer),
        candidate_refs=cast(list[dict[str, object]], candidate_refs),
        candidates=candidate_list,
    )
    if (
        validation.payload.get("candidate_set_hash") != candidate_set.content_hash
        or validation.payload.get("validated_count") != 103
        or validation.payload.get("status") != "passed"
    ):
        raise DatasetValidationError("ValidationManifest is invalid")
    parameters = query_plan.payload.get("parameters")
    if not isinstance(parameters, dict):
        raise DatasetValidationError("QueryPlan parameters are invalid")
    start_utc = parameters.get("start_utc")
    end_utc = parameters.get("end_utc")
    if type(start_utc) is not str or type(end_utc) is not str:
        raise DatasetValidationError("QueryPlan window parameters are invalid")
    if validation.payload != expected_validation_manifest(
        candidate_set,
        candidate_list,
        window_start_utc=cast(str, start_utc),
        window_end_utc=cast(str, end_utc),
    ):
        raise DatasetValidationError("ValidationManifest is not content-derived")
    winners = dedupe.payload.get("winner_refs")
    if (
        dedupe.payload.get("candidate_set_hash") != candidate_set.content_hash
        or dedupe.payload.get("validation_manifest_hash") != validation.content_hash
        or dedupe.payload.get("input_count") != 103
        or dedupe.payload.get("unique_count") != 100
        or dedupe.payload.get("duplicate_count") != 3
        or not isinstance(winners, list)
        or len(winners) != 100
    ):
        raise DatasetValidationError("DedupeManifest is invalid")
    if dedupe.payload != expected_dedupe_manifest(candidate_set, validation, candidate_list):
        raise DatasetValidationError("DedupeManifest is not content-derived")
    winner_by_key: dict[str, str] = {}
    for winner in winners:
        if not isinstance(winner, dict) or set(winner) != {"artifact_hash", "source_trace_key"}:
            raise DatasetValidationError("Dedupe winner ref is invalid")
        winner_key = winner.get("source_trace_key")
        artifact_hash = winner.get("artifact_hash")
        if not isinstance(winner_key, str) or not isinstance(artifact_hash, str) or winner_key in winner_by_key:
            raise DatasetValidationError("Dedupe winners are malformed or duplicated")
        if artifact_hash not in {item[0] for item in candidates.values()}:
            raise DatasetValidationError("Dedupe winner is not a source candidate")
        winner_by_key[winner_key] = artifact_hash
    rankings = selection.payload.get("rankings")
    if (
        selection.schema_version != "1.0.0"
        or set(selection.payload)
        != {
            "approved_data_mix_hash",
            "approved_data_mix_policy_version",
            "dedupe_manifest_hash",
            "eligible_count",
            "mix",
            "rankings",
            "score_contract",
            "selected_count",
            "selection_policy_version",
        }
        or selection.payload.get("dedupe_manifest_hash") != dedupe.content_hash
        or selection.payload.get("selection_policy_version") != selection_policy
        or selection.payload.get("approved_data_mix_hash") != lineage.get("approved_data_mix_hash")
        or selection.payload.get("approved_data_mix_policy_version") != lineage.get("approved_data_mix_policy_version")
        or type(selection.payload.get("eligible_count")) is not int
        or selection.payload.get("eligible_count") != 100
        or type(selection.payload.get("selected_count")) is not int
        or selection.payload.get("selected_count") != 100
        or selection.payload.get("score_contract") != "sha256-over-sanitized-semantic-content"
        or not isinstance(rankings, list)
        or len(rankings) != 100
    ):
        raise DatasetValidationError("BadcaseSelectionManifest is invalid")
    selected_hashes: set[str] = set()
    observed_models: dict[str, int] = {}
    observed_purposes: dict[str, int] = {}
    observed_tools: dict[str, int] = {}
    for rank, item in enumerate(rankings, start=1):
        if not isinstance(item, dict) or set(item) != {
            "artifact_hash",
            "rank",
            "selected",
            "selection_score",
            "source_trace_key",
        }:
            raise DatasetValidationError("badcase ranking fields are invalid")
        source_key = item.get("source_trace_key")
        artifact_hash = item.get("artifact_hash")
        score = item.get("selection_score")
        if (
            type(item.get("rank")) is not int
            or item.get("rank") != rank
            or type(item.get("selected")) is not bool
            or item.get("selected") is not True
            or type(source_key) is not str
            or not cast(str, source_key)
            or type(artifact_hash) is not str
            or _HASH.fullmatch(cast(str, artifact_hash)) is None
            or type(score) is not str
            or _HASH.fullmatch(cast(str, score)) is None
            or winner_by_key.get(source_key) != artifact_hash
        ):
            raise DatasetValidationError("badcase ranking order or winner binding is invalid")
        candidate = store.read(cast(str, artifact_hash), expected_schema_name="SanitizedCandidate")
        if item.get("selection_score") != selection_score(candidate.payload, cast(str, selection_policy)):
            raise DatasetValidationError("badcase selection score is not content-derived")
        selected_hashes.add(cast(str, artifact_hash))
        for field, output in (
            ("model_id", observed_models),
            ("purpose", observed_purposes),
            ("tool_name", observed_tools),
        ):
            value = candidate.payload.get(field)
            if not isinstance(value, str):
                raise DatasetValidationError("badcase mix field is invalid")
            output[value] = output.get(value, 0) + 1
    if len(selected_hashes) != 100:
        raise DatasetValidationError("badcase selection is not unique")
    approved_mix = skill.payload.get("approved_data_mix")
    observed_mix = {
        "model": dict(sorted(observed_models.items())),
        "purpose": dict(sorted(observed_purposes.items())),
        "tool": dict(sorted(observed_tools.items())),
    }
    if (
        not isinstance(approved_mix, dict)
        or selection.payload.get("mix") != observed_mix
        or observed_mix
        != {
            "model": approved_mix.get("model"),
            "purpose": approved_mix.get("purpose"),
            "tool": approved_mix.get("tool"),
        }
    ):
        raise DatasetValidationError("badcase selection mix violates the approved policy")
    approved_mix_mapping = skill.payload.get("approved_data_mix")
    if not isinstance(approved_mix_mapping, dict):
        raise DatasetValidationError("DataSourceSkill approved mix is invalid")
    try:
        approved_mix_contract = ApprovedDataMix.from_mapping(cast(dict[str, object], approved_mix_mapping))
    except Exception as error:
        raise DatasetValidationError("DataSourceSkill approved mix cannot be decoded") from error
    if selection.payload != expected_selection_manifest(
        store,
        dedupe,
        policy=cast(str, selection_policy),
        approved_mix=approved_mix_contract,
    ):
        raise DatasetValidationError("BadcaseSelectionManifest is not content-derived")
    trace_candidate_hashes: set[str] = set()
    for trace in traces:
        candidate_hash = trace.payload.get("source_candidate_hash")
        if not isinstance(candidate_hash, str) or candidate_hash not in selected_hashes:
            raise DatasetValidationError("TrainingTrace is not bound to a selected candidate")
        candidate = store.read(candidate_hash, expected_schema_name="SanitizedCandidate")
        if trace.payload != training_trace_payload(candidate_hash, candidate.payload):
            raise DatasetValidationError("TrainingTrace content does not match its selected candidate")
        trace_candidate_hashes.add(candidate_hash)
    if trace_candidate_hashes != selected_hashes:
        raise DatasetValidationError("DatasetVersion does not cover every selected badcase")


def _verify_query_plan(
    query_plan: Artifact,
    *,
    blueprint: Artifact,
    skill: Artifact,
    role_audit: Artifact,
    correction_audit: Artifact,
    lineage: dict[str, object],
) -> None:
    fields = {
        "approved_data_mix_hash",
        "ast",
        "blueprint_hash",
        "correction_role_invocation_audit_hash",
        "data_source_skill_hash",
        "fault_schedule_hash",
        "input_hash",
        "parameters",
        "purpose",
        "query_hash",
        "query_source_hash",
        "role_invocation_audit_hash",
        "sanitizer_policy_version",
        "selection_policy_version",
        "source_id",
        "window_hash",
    }
    ast = {
        "columns": [
            "trace_pk",
            "report_id",
            "event_time_utc",
            "ingestion_time_utc",
            "purpose",
            "prompt",
            "response",
            "tool_name",
            "model_id",
            "private_sentinel",
        ],
        "filters": [
            {"column": "event_time_utc", "operator": ">=", "parameter": "start_utc"},
            {"column": "event_time_utc", "operator": "<", "parameter": "end_utc"},
            {"column": "purpose", "operator": "=", "parameter": "purpose"},
        ],
        "order_by": [
            {"column": "trace_pk", "direction": "ASC"},
            {"column": "ingestion_time_utc", "direction": "ASC"},
        ],
        "statement_type": "SELECT",
        "table": "fixture.online_trace",
    }
    parameters = {
        "end_utc": "2026-01-02T00:00:00Z",
        "purpose": "training_allowed",
        "start_utc": "2026-01-01T00:00:00Z",
    }
    payload = query_plan.payload
    hashes = (
        payload.get("approved_data_mix_hash"),
        payload.get("blueprint_hash"),
        payload.get("correction_role_invocation_audit_hash"),
        payload.get("data_source_skill_hash"),
        payload.get("fault_schedule_hash"),
        payload.get("input_hash"),
        payload.get("query_hash"),
        payload.get("query_source_hash"),
        payload.get("role_invocation_audit_hash"),
        payload.get("window_hash"),
    )
    expected_query_hash = sha256_hex(
        canonical_json_bytes(
            {
                "ast": ast,
                "domain": "governed-query-contract/1.0.0",
                "parameters": parameters,
            }
        )
    )
    approved_mix = skill.payload.get("approved_data_mix_hash")
    if (
        query_plan.schema_version != "1.0.0"
        or set(payload) != fields
        or any(type(value) is not str or _HASH.fullmatch(cast(str, value)) is None for value in hashes)
        or payload.get("ast") != ast
        or payload.get("parameters") != parameters
        or payload.get("purpose") != "training_allowed"
        or payload.get("source_id") != "fixture-online-traces-v1"
        or payload.get("blueprint_hash") != blueprint.content_hash
        or lineage.get("blueprint_hash") != blueprint.content_hash
        or payload.get("data_source_skill_hash") != skill.content_hash
        or payload.get("approved_data_mix_hash") != approved_mix
        or payload.get("role_invocation_audit_hash") != role_audit.content_hash
        or payload.get("correction_role_invocation_audit_hash") != correction_audit.content_hash
        or payload.get("query_hash") != expected_query_hash
        or payload.get("query_source_hash") != lineage.get("query_source_hash")
        or payload.get("sanitizer_policy_version") != lineage.get("sanitizer_policy_version")
        or payload.get("selection_policy_version") != lineage.get("selection_policy_version")
        or payload.get("window_hash") != QueryWindow("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z").content_hash
    ):
        raise DatasetValidationError("QueryPlan source contract is invalid")


def _verify_candidate_set_boundary_contract(
    store: ArtifactStore,
    *,
    candidate_set: Artifact,
    query_plan: Artifact,
    sanitizer: str,
    candidate_refs: list[dict[str, object]],
    candidates: list[Artifact],
) -> None:
    """Recompute the public query/candidate boundary contract without private state."""

    fields = {
        "boundary_attempt_hash",
        "boundary_attempt_ordinal",
        "boundary_input_hash",
        "boundary_private_result_content_hash",
        "boundary_query_hash",
        "boundary_raw_result_hash",
        "boundary_request_hash",
        "boundary_result_hash",
        "boundary_result_size",
        "boundary_schedule_hash",
        "candidate_count",
        "candidate_ref_set_hash",
        "candidate_refs",
        "query_plan_hash",
        "query_request_hash",
        "sanitizer_policy_version",
    }
    payload = candidate_set.payload
    hashes = (
        payload.get("boundary_attempt_hash"),
        payload.get("boundary_input_hash"),
        payload.get("boundary_private_result_content_hash"),
        payload.get("boundary_query_hash"),
        payload.get("boundary_raw_result_hash"),
        payload.get("boundary_request_hash"),
        payload.get("boundary_result_hash"),
        payload.get("boundary_schedule_hash"),
        payload.get("candidate_ref_set_hash"),
        payload.get("query_plan_hash"),
        payload.get("query_request_hash"),
    )
    if (
        candidate_set.schema_version != "1.0.0"
        or set(payload) != fields
        or any(type(value) is not str or _HASH.fullmatch(cast(str, value)) is None for value in hashes)
        or type(payload.get("boundary_attempt_ordinal")) is not int
        or cast(int, payload.get("boundary_attempt_ordinal")) <= 0
        or type(payload.get("boundary_result_size")) is not int
        or cast(int, payload.get("boundary_result_size")) <= 0
        or type(payload.get("candidate_count")) is not int
        or payload.get("candidate_count") != 103
        or payload.get("candidate_refs") != candidate_refs
        or payload.get("candidate_ref_set_hash") != sha256_hex(canonical_json_bytes(candidate_refs))
        or payload.get("query_plan_hash") != query_plan.content_hash
        or payload.get("sanitizer_policy_version") != sanitizer
    ):
        raise DatasetValidationError("SanitizedCandidateSet boundary contract is invalid")

    request_hash = cast(str, payload["query_request_hash"])
    try:
        request = store.read(request_hash, expected_schema_name="QueryRequest")
    except (ArtifactCorruption, RuntimeError) as error:
        raise DatasetValidationError("QueryRequest cannot be resolved") from error
    expected_request = {
        "fault_schedule_hash": query_plan.payload.get("fault_schedule_hash"),
        "idempotency_key": "query-" + query_plan.content_hash[:40],
        "input_hash": query_plan.payload.get("input_hash"),
        "parameters": query_plan.payload.get("parameters"),
        "query_hash": query_plan.payload.get("query_hash"),
        "query_plan_hash": query_plan.content_hash,
        "source_id": query_plan.payload.get("source_id"),
        "window_hash": query_plan.payload.get("window_hash"),
    }
    if request.schema_version != "1.0.0" or request.payload != expected_request:
        raise DatasetValidationError("QueryRequest differs from its verified QueryPlan")
    boundary_request = {
        "ast": query_plan.payload.get("ast"),
        "input_hash": request.payload.get("input_hash"),
        "parameters": request.payload.get("parameters"),
        "query_hash": request.payload.get("query_hash"),
        "query_key": request.payload.get("idempotency_key"),
        "schedule_hash": request.payload.get("fault_schedule_hash"),
        "schema_version": "fixture-query-request/1.0.0",
        "source_id": request.payload.get("source_id"),
        "window_hash": request.payload.get("window_hash"),
    }
    if (
        payload.get("boundary_request_hash") != sha256_hex(canonical_json_bytes(boundary_request))
        or payload.get("boundary_input_hash") != request.payload.get("input_hash")
        or payload.get("boundary_query_hash") != request.payload.get("query_hash")
        or payload.get("boundary_schedule_hash") != request.payload.get("fault_schedule_hash")
    ):
        raise DatasetValidationError("SanitizedCandidateSet boundary contract is invalid")

    result_rows = [
        {
            "event_time_utc": candidate.payload.get("event_time_utc"),
            "ingestion_time_utc": candidate.payload.get("ingestion_time_utc"),
            "model_id": candidate.payload.get("model_id"),
            "prompt": candidate.payload.get("prompt"),
            "purpose": candidate.payload.get("purpose"),
            "report_id": candidate.payload.get("report_id"),
            "response": candidate.payload.get("response"),
            "tool_name": candidate.payload.get("tool_name"),
            "trace_pk": candidate.payload.get("source_trace_key"),
        }
        for candidate in candidates
    ]
    result_bytes = canonical_json_bytes(result_rows)
    if payload.get("boundary_result_hash") != sha256_hex(result_bytes) or payload.get("boundary_result_size") != len(
        result_bytes
    ):
        raise DatasetValidationError("SanitizedCandidateSet boundary result is not content-derived")
