"""Durable deterministic fixture DataSource boundary for Ticket 03.

The Data Provider's JavaScript formula is *never* evaluated.  Its exact frozen
identity is validated, then an independently typed Python generator implements
the declared contract.  Raw role bytes and OnlineTrace rows are confined to
``private-boundary/data-source``; only sanitized rows and hash-only audit
metadata cross the adapter boundary.
"""

from __future__ import annotations

import fcntl
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from clawrl.artifacts import (
    Artifact,
    ArtifactStore,
    JsonValue,
    canonical_json_bytes,
    sha256_hex,
)
from clawrl.data.models import ApprovedDataMix, DataContractError, DataSourceSkill, FixtureDataIngestConfig, QueryWindow
from clawrl.data.safety import assert_public_sentinel_free
from clawrl.data.sql import SelectAst, parse_and_validate_select

_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EXPECTED_RAW_HASH = "e38025307baa56c095df829a7a501cbf8ecf844435e7f3dab873a4cae55add74"
_EXPECTED_FORMULA_HASH = "4076bd42ff71b18239a4e74e9488668131bb6de8b301b1bdb8ef0ea061f64662"
_EXPECTED_SENTINEL_HASH = "ea55483587214b66e63d5172b8df9f5dd2563af21fe3873561c414e96f6673dc"
_EXPECTED_CORRECTION_RAW_HASH = "90167c1bddbac43f407d505a4162bf074658ec23815ed6c06eb0a3b34e9d9644"
_BOUNDARY_FAILURE_BY_SOURCE_FAULT = {
    "byte_limit": ("QUERY_BYTE_LIMIT_EXCEEDED", False),
    "leak_sentinel": ("SENTINEL_LEAK_DETECTED", True),
    "missing_field": ("RAW_FIELD_COMPLETENESS_FAILED", True),
    "missing_sentinel": ("RAW_FIELD_COMPLETENESS_FAILED", True),
    "row_limit": ("QUERY_ROW_LIMIT_EXCEEDED", False),
    "time_limit": ("QUERY_TIME_LIMIT_EXCEEDED", False),
    "wrong_sentinel": ("SENTINEL_VALIDATION_FAILED", True),
}
_EXPECTED_COLUMNS = (
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
)
_EXPECTED_MAPPING = {
    "dedupe_key": "trace_pk",
    "event_time": "event_time_utc",
    "ingestion_time": "ingestion_time_utc",
    "model": "model_id",
    "primary_key": "trace_pk",
    "prompt": "prompt",
    "response": "response",
    "tool": "tool_name",
}
_CORRECTED_SKILL_MAPPING = {**_EXPECTED_MAPPING, "primary_key": "report_id"}
_CORRECTED_PUBLIC_MAPPING = {
    **_CORRECTED_SKILL_MAPPING,
    "logical_trace_key": "trace_pk",
}
_EXPECTED_OVERRIDES = (
    (7, "trace-007", "report-100", 3600),
    (42, "trace-042", "report-101", 3601),
    (99, "trace-099", "report-102", 3602),
)
_PUBLIC_BLUEPRINT_SCHEMA = "DataProviderBlueprint"


class DataSourceBoundaryError(RuntimeError):
    """The fixture source failed closed without exporting raw source values."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON object key")
        output[key] = value
    return output


def _bounded_json(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > 64_000:
        raise DataContractError("Data Provider output byte size is invalid")
    try:
        decoded = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise DataContractError("Data Provider output is not strict UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise DataContractError("Data Provider output must be an object")
    nodes = 0
    pending: list[tuple[object, int]] = [(decoded, 0)]
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > 4_096 or depth > 32:
            raise DataContractError("Data Provider output exceeds structural limits")
        if isinstance(value, dict):
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
        elif value is not None and not isinstance(value, (bool, int, str)):
            raise DataContractError("Data Provider output contains an unsupported value")
    return cast(dict[str, object], decoded)


@dataclass(frozen=True, slots=True, repr=False)
class _DataProviderExchange:
    """Validated private exchange plus a sentinel-free public normalization."""

    skill: DataSourceSkill
    raw_output_hash: str
    raw_output_size: int
    normalized_output_hash: str
    input_packet_hash: str
    packet_id: str
    seed: int
    role_session_lineage: str
    public_blueprint_bytes: bytes
    _raw_output: bytes
    _normalized_output_bytes: bytes
    _input_packet_bytes: bytes
    _sentinel: str
    _overrides: tuple[tuple[int, str, str, int], ...]

    def public_blueprint(self) -> dict[str, JsonValue]:
        decoded = json.loads(self.public_blueprint_bytes)
        if not isinstance(decoded, dict):
            raise DataSourceBoundaryError("PUBLIC_BLUEPRINT_CORRUPTION")
        return cast(dict[str, JsonValue], decoded)


@dataclass(frozen=True, slots=True, repr=False)
class _DataProviderCorrection:
    input_packet_hash: str
    raw_output_hash: str
    raw_output_size: int
    packet_id: str
    prior_packet_id: str
    seed: int
    role_session_lineage: str
    dedupe_semantics: str
    report_identity_rule: str
    _raw_output: bytes
    _input_packet_bytes: bytes
    _normalized_output_bytes: bytes

    @property
    def normalized_output_hash(self) -> str:
        return sha256_hex(self._normalized_output_bytes)


def _parse_data_provider_correction(
    input_packet: dict[str, object],
    raw_output: bytes,
    *,
    role_session_lineage: str,
    previous_output_hash: str,
) -> _DataProviderCorrection:
    expected_input = {
        "packet_id": "dp-ticket03-correction-0001",
        "prior_output_hash": _EXPECTED_RAW_HASH,
        "request": {
            "dedupe_key": "trace_pk",
            "logical_trace_key": "trace_pk",
            "primary_key": "report_id",
            "reason_code": "SOURCE_KEY_CONTRACT_CORRECTION_REQUIRED",
        },
        "schema_version": "data-provider-correction-input/1.0.0",
        "seed": 3001,
    }
    if (
        type(input_packet) is not dict
        or input_packet != expected_input
        or type(input_packet.get("seed")) is not int
        or previous_output_hash != _EXPECTED_RAW_HASH
        or role_session_lineage != "/root/ticket03_data_provider_correction"
    ):
        raise DataContractError("Data Provider correction input is not the frozen allowlisted request")
    decoded = _bounded_json(raw_output)
    # The isolated role's wire bytes are authoritative and intentionally are
    # not required to use our canonical JSON key ordering.  Preserve those
    # exact bytes for provenance while deriving a separate canonical
    # normalization below.
    if sha256_hex(raw_output) != _EXPECTED_CORRECTION_RAW_HASH:
        raise DataContractError("Data Provider correction bytes are not the frozen isolated response")
    expected_fields = {
        "corrected_field_mapping",
        "dedupe_semantics",
        "invariants",
        "packet_id",
        "prior_packet_id",
        "reason_code",
        "report_identity_rule",
        "schema_version",
        "seed",
        "source_id",
    }
    invariants = decoded.get("invariants")
    expected_invariants = {
        "duplicate_logical_keys": ["trace-007", "trace-042", "trace-099"],
        "formula_unchanged": True,
        "raw_row_content_unchanged": True,
        "sanitizer_unchanged": True,
        "unique_report_id_count": 103,
        "unique_trace_pk_count": 100,
        "window_unchanged": True,
    }
    expected_mapping = {
        "dedupe_key": "trace_pk",
        "logical_trace_key": "trace_pk",
        "primary_key": "report_id",
    }
    if (
        set(decoded) != expected_fields
        or decoded.get("schema_version") != "data-provider-correction-output/1.0.0"
        or decoded.get("packet_id") != "dp-ticket03-correction-0001"
        or decoded.get("prior_packet_id") != "dp-ticket03-0001"
        or type(decoded.get("seed")) is not int
        or decoded.get("seed") != 3001
        or decoded.get("source_id") != "fixture-online-traces-v1"
        or decoded.get("corrected_field_mapping") != expected_mapping
        or decoded.get("reason_code") != "SOURCE_KEY_CONTRACT_CORRECTED"
        or decoded.get("report_identity_rule")
        != (
            "report_id is unique for every raw report; trace_pk may repeat and is the logical TrainingTrace "
            "identity after highest-ingestion dedupe"
        )
        or decoded.get("dedupe_semantics")
        != "highest ingestion_time_utc wins for the same trace_pk; sanitized semantic content must match"
        or type(invariants) is not dict
        or invariants != expected_invariants
        or any(type(invariants.get(key)) is not int for key in ("unique_report_id_count", "unique_trace_pk_count"))
        or any(
            type(invariants.get(key)) is not bool or invariants.get(key) is not True
            for key in (
                "formula_unchanged",
                "raw_row_content_unchanged",
                "sanitizer_unchanged",
                "window_unchanged",
            )
        )
    ):
        raise DataContractError("Data Provider correction output contract is invalid")
    normalized = canonical_json_bytes(decoded)
    return _DataProviderCorrection(
        input_packet_hash=sha256_hex(canonical_json_bytes(input_packet)),
        raw_output_hash=sha256_hex(raw_output),
        raw_output_size=len(raw_output),
        packet_id="dp-ticket03-correction-0001",
        prior_packet_id="dp-ticket03-0001",
        seed=3001,
        role_session_lineage=role_session_lineage,
        dedupe_semantics=cast(str, decoded["dedupe_semantics"]),
        report_identity_rule=cast(str, decoded["report_identity_rule"]),
        _raw_output=raw_output,
        _input_packet_bytes=canonical_json_bytes(input_packet),
        _normalized_output_bytes=normalized,
    )


def _apply_data_provider_correction(
    exchange: _DataProviderExchange,
    correction: _DataProviderCorrection,
) -> _DataProviderExchange:
    skill = replace(
        exchange.skill,
        field_mapping=tuple(sorted(_CORRECTED_SKILL_MAPPING.items())),
        primary_key="report_id",
        dedupe_key="trace_pk",
        duplicate_semantics=correction.dedupe_semantics,
        post_query_invariants=(
            "field-completeness",
            "strict-window-and-utc",
            "purpose-match",
            "sentinel-validated-and-dropped",
            "unique-report-identity",
            "duplicate-semantic-equivalence",
            "unique-logical-trace-identity",
        ),
    )
    blueprint = exchange.public_blueprint()
    blueprint["field_mapping"] = cast(JsonValue, _CORRECTED_PUBLIC_MAPPING)
    blueprint["dedupe_semantics_hash"] = sha256_hex(correction.dedupe_semantics.encode("utf-8"))
    blueprint["correction"] = {
        "corrected_output_hash": correction.raw_output_hash,
        "dedupe_key": "trace_pk",
        "logical_trace_key": "trace_pk",
        "packet_id": correction.packet_id,
        "primary_key": "report_id",
        "prior_packet_id": correction.prior_packet_id,
        "normalized_output_hash": correction.normalized_output_hash,
        "reason_code": "SOURCE_KEY_CONTRACT_CORRECTED",
        "report_identity_rule_hash": sha256_hex(correction.report_identity_rule.encode("utf-8")),
    }
    public_bytes = canonical_json_bytes(blueprint)
    return replace(exchange, skill=skill, public_blueprint_bytes=public_bytes)


def _parse_data_provider_exchange(
    input_packet: dict[str, object],
    raw_output: bytes,
    *,
    role_session_lineage: str,
) -> _DataProviderExchange:
    """Validate the one allowlisted role exchange without executing formula text."""

    if not isinstance(input_packet, dict) or set(input_packet) != {"packet_id", "request", "schema_version", "seed"}:
        raise DataContractError("Data Provider input packet fields are invalid")
    request = input_packet.get("request")
    if (
        input_packet.get("schema_version") != "data-provider-input/1.0.0"
        or input_packet.get("packet_id") != "dp-ticket03-0001"
        or input_packet.get("seed") != 3001
        or isinstance(input_packet.get("seed"), bool)
        or not isinstance(request, dict)
        or request
        != {
            "required_purpose": "training_allowed",
            "required_raw_rows": 103,
            "required_unique_keys": 100,
            "window_end_utc": "2026-01-02T00:00:00Z",
            "window_start_utc": "2026-01-01T00:00:00Z",
        }
    ):
        raise DataContractError("Data Provider input packet is not the frozen allowlisted request")
    if not isinstance(role_session_lineage, str) or role_session_lineage != "/root/ticket03_data_provider":
        raise DataContractError("Data Provider role lineage is not the isolated approved task")
    input_packet_bytes = canonical_json_bytes(input_packet)
    decoded = _bounded_json(raw_output)
    if sha256_hex(raw_output) != _EXPECTED_RAW_HASH:
        raise DataContractError("Data Provider output bytes are not the frozen isolated response")
    expected_fields = {
        "columns",
        "dedupe_semantics",
        "duplicate_count",
        "duplicate_overrides",
        "field_mapping",
        "invariants",
        "packet_id",
        "purpose",
        "row_count",
        "row_formula",
        "sanitizer_rule",
        "schema_version",
        "seed",
        "sentinel",
        "source_id",
        "table",
        "unique_key_count",
        "window",
    }
    if set(decoded) != expected_fields:
        raise DataContractError("Data Provider output schema fields are invalid")
    formula = decoded.get("row_formula")
    if not isinstance(formula, dict) or set(formula) != {"language", "source"}:
        raise DataContractError("Data Provider row formula declaration is invalid")
    formula_source = formula.get("source")
    if (
        formula.get("language") != "javascript"
        or not isinstance(formula_source, str)
        or sha256_hex(formula_source.encode("utf-8")) != _EXPECTED_FORMULA_HASH
    ):
        raise DataContractError("Data Provider formula identity is not approved")
    sentinel = decoded.get("sentinel")
    if not isinstance(sentinel, str) or sha256_hex(sentinel.encode("utf-8")) != _EXPECTED_SENTINEL_HASH:
        raise DataContractError("Data Provider sentinel identity is not approved")
    overrides = _validate_overrides(decoded.get("duplicate_overrides"))
    invariants = decoded.get("invariants")
    if (
        not isinstance(invariants, list)
        or len(invariants) != 8
        or not all(isinstance(item, str) for item in invariants)
    ):
        raise DataContractError("Data Provider invariants are invalid")
    mapping = decoded.get("field_mapping")
    window = decoded.get("window")
    columns = decoded.get("columns")
    seed = decoded.get("seed")
    if (
        decoded.get("schema_version") != "data-provider-output/1.0.0"
        or decoded.get("packet_id") != input_packet["packet_id"]
        or seed != input_packet["seed"]
        or isinstance(seed, bool)
        or decoded.get("source_id") != "fixture-online-traces-v1"
        or decoded.get("table") != "fixture.online_trace"
        or columns != list(_EXPECTED_COLUMNS)
        or mapping != _EXPECTED_MAPPING
        or window != {"start_utc": "2026-01-01T00:00:00Z", "end_utc": "2026-01-02T00:00:00Z"}
        or decoded.get("purpose") != "training_allowed"
        or decoded.get("row_count") != 103
        or isinstance(decoded.get("row_count"), bool)
        or decoded.get("unique_key_count") != 100
        or isinstance(decoded.get("unique_key_count"), bool)
        or decoded.get("duplicate_count") != 3
        or isinstance(decoded.get("duplicate_count"), bool)
        or not isinstance(decoded.get("dedupe_semantics"), str)
        or not isinstance(decoded.get("sanitizer_rule"), str)
    ):
        raise DataContractError("Data Provider output contract values are invalid")
    skill = DataSourceSkill(
        source_id="fixture-online-traces-v1",
        table="fixture.online_trace",
        approved_window=QueryWindow("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
        approved_data_mix=ApprovedDataMix(),
        approved_columns=_EXPECTED_COLUMNS,
        purpose="training_allowed",
        field_mapping=tuple(sorted(_EXPECTED_MAPPING.items())),
        event_time_column="event_time_utc",
        ingestion_time_column="ingestion_time_utc",
        primary_key="trace_pk",
        dedupe_key="trace_pk",
        duplicate_semantics=cast(str, decoded["dedupe_semantics"]),
        allowed_filters=(
            "event_time_utc >= :start_utc",
            "event_time_utc < :end_utc",
            "purpose = :purpose",
        ),
        allowed_joins=(),
        sanitizer_version="sentinel-drop-v1",
        sensitive_columns=("private_sentinel",),
        max_rows=103,
        max_bytes=131_072,
        max_elapsed_ticks=8,
        expected_raw_rows=103,
        expected_unique_rows=100,
        selection_count=100,
        selection_policy_version="badcase-sha256-v1",
        post_query_invariants=(
            "field-completeness",
            "strict-window-and-utc",
            "purpose-match",
            "sentinel-validated-and-dropped",
            "duplicate-semantic-equivalence",
            "unique-trace-identity",
        ),
    )
    public_blueprint: dict[str, object] = {
        "columns": list(_EXPECTED_COLUMNS),
        "dedupe_semantics_hash": sha256_hex(cast(str, decoded["dedupe_semantics"]).encode("utf-8")),
        "duplicate_count": 3,
        "duplicate_overrides": [
            {
                "ingestion_lag_seconds": lag,
                "logical_index": index,
                "report_id": report_id,
                "trace_pk": trace_pk,
            }
            for index, trace_pk, report_id, lag in overrides
        ],
        "field_mapping": _EXPECTED_MAPPING,
        "formula": {
            "language": "javascript",
            "source_hash": _EXPECTED_FORMULA_HASH,
            "execution_permitted": False,
        },
        "invariant_hashes": [sha256_hex(cast(str, item).encode("utf-8")) for item in invariants],
        "packet_id": "dp-ticket03-0001",
        "purpose": "training_allowed",
        "row_count": 103,
        "sanitizer": {
            "expected_value_hash": _EXPECTED_SENTINEL_HASH,
            "sensitive_columns": ["private_sentinel"],
            "version": "sentinel-drop-v1",
        },
        "schema_version": "data-provider-blueprint/1.0.0",
        "seed": 3001,
        "source_id": "fixture-online-traces-v1",
        "table": "fixture.online_trace",
        "unique_key_count": 100,
        "window": cast(dict[str, object], window),
    }
    blueprint_bytes = canonical_json_bytes(public_blueprint)
    normalized_output_bytes = canonical_json_bytes(decoded)
    return _DataProviderExchange(
        skill=skill,
        raw_output_hash=sha256_hex(raw_output),
        raw_output_size=len(raw_output),
        normalized_output_hash=sha256_hex(normalized_output_bytes),
        input_packet_hash=sha256_hex(input_packet_bytes),
        packet_id="dp-ticket03-0001",
        seed=3001,
        role_session_lineage=role_session_lineage,
        public_blueprint_bytes=blueprint_bytes,
        _raw_output=raw_output,
        _normalized_output_bytes=normalized_output_bytes,
        _input_packet_bytes=input_packet_bytes,
        _sentinel=sentinel,
        _overrides=overrides,
    )


def _validate_overrides(value: object) -> tuple[tuple[int, str, str, int], ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise DataContractError("Data Provider duplicate overrides are invalid")
    normalized: list[tuple[int, str, str, int]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "ingestion_lag_seconds",
            "logical_index",
            "report_id",
            "trace_pk",
        }:
            raise DataContractError("Data Provider duplicate override fields are invalid")
        index = item.get("logical_index")
        lag = item.get("ingestion_lag_seconds")
        trace_pk = item.get("trace_pk")
        report_id = item.get("report_id")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not isinstance(lag, int)
            or isinstance(lag, bool)
            or not isinstance(trace_pk, str)
            or not isinstance(report_id, str)
        ):
            raise DataContractError("Data Provider duplicate override types are invalid")
        normalized.append((index, trace_pk, report_id, lag))
    result = tuple(normalized)
    if result != _EXPECTED_OVERRIDES:
        raise DataContractError("Data Provider duplicate overrides are not the frozen set")
    return result


@dataclass(frozen=True, slots=True)
class QueryResult:
    status: str
    failure_code: str | None
    raw_row_count: int
    raw_byte_count: int
    elapsed_ticks: int
    request_hash: str
    query_hash: str
    input_hash: str
    schedule_hash: str
    attempt_hash: str
    result_hash: str | None
    result_size: int
    private_result_content_hash: str | None
    late_result_hash: str | None
    raw_result_hash: str | None
    attempt_ordinal: int
    _sanitized_bytes: bytes

    @property
    def sanitized_rows(self) -> tuple[dict[str, JsonValue], ...]:
        decoded = json.loads(self._sanitized_bytes)
        if not isinstance(decoded, list) or not all(isinstance(item, dict) for item in decoded):
            raise DataSourceBoundaryError("SANITIZED_RESULT_CORRUPTION")
        return tuple(cast(dict[str, JsonValue], item) for item in decoded)

    def deduplicate(self) -> tuple[dict[str, JsonValue], ...]:
        if self.status != "success":
            raise DataSourceBoundaryError("QUERY_NOT_SUCCESSFUL")
        grouped: dict[str, list[dict[str, JsonValue]]] = {}
        for row in self.sanitized_rows:
            trace_pk = row.get("trace_pk")
            if not isinstance(trace_pk, str):
                raise DataSourceBoundaryError("DEDUPE_KEY_INVALID")
            grouped.setdefault(trace_pk, []).append(row)
        output: list[dict[str, JsonValue]] = []
        semantic_fields = ("event_time_utc", "model_id", "prompt", "purpose", "response", "tool_name")
        for trace_pk in sorted(grouped):
            candidates = grouped[trace_pk]
            first_semantics = tuple(candidates[0].get(field) for field in semantic_fields)
            if any(tuple(item.get(field) for field in semantic_fields) != first_semantics for item in candidates[1:]):
                raise DataSourceBoundaryError("DUPLICATE_SEMANTIC_CONFLICT")
            ordered = sorted(candidates, key=lambda item: cast(str, item["ingestion_time_utc"]))
            if len({cast(str, item["ingestion_time_utc"]) for item in ordered}) != len(ordered):
                raise DataSourceBoundaryError("DUPLICATE_INGESTION_CONFLICT")
            output.append(dict(ordered[-1]))
        return tuple(output)


@dataclass(frozen=True, slots=True)
class QueryAttemptEvidence:
    attempt_hash: str
    attempt_ordinal: int
    directive: str
    status: str
    failure_code: str | None
    request_hash: str
    query_hash: str
    input_hash: str
    schedule_hash: str
    previous_attempt_hash: str | None
    result_hash: str | None
    result_size: int
    late_result_hash: str | None
    raw_result_hash: str | None
    raw_result_size: int
    raw_row_count: int
    elapsed_ticks: int
    clock_start_tick: int
    clock_end_tick: int
    clock_policy_hash: str


@dataclass(frozen=True, slots=True)
class DataProviderIngressReceipt:
    """Opaque hash-only handle to an adapter-private staged role exchange."""

    ingress_hash: str
    blueprint_hash: str
    data_source_skill_hash: str
    input_packet_hash: str
    raw_output_hash: str
    normalized_output_hash: str
    role_invocation_audit_hash: str
    correction_input_packet_hash: str
    correction_raw_output_hash: str
    correction_normalized_output_hash: str
    correction_role_invocation_audit_hash: str

    def __post_init__(self) -> None:
        for value in (
            self.ingress_hash,
            self.blueprint_hash,
            self.data_source_skill_hash,
            self.input_packet_hash,
            self.raw_output_hash,
            self.normalized_output_hash,
            self.role_invocation_audit_hash,
            self.correction_input_packet_hash,
            self.correction_raw_output_hash,
            self.correction_normalized_output_hash,
            self.correction_role_invocation_audit_hash,
        ):
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise DataContractError("Data Provider ingress receipt contains an invalid hash")

    def artifact_payload(self) -> dict[str, str]:
        return {
            "blueprint_hash": self.blueprint_hash,
            "data_source_skill_hash": self.data_source_skill_hash,
            "ingress_hash": self.ingress_hash,
            "input_packet_hash": self.input_packet_hash,
            "raw_output_hash": self.raw_output_hash,
            "normalized_output_hash": self.normalized_output_hash,
            "role_invocation_audit_hash": self.role_invocation_audit_hash,
            "correction_input_packet_hash": self.correction_input_packet_hash,
            "correction_raw_output_hash": self.correction_raw_output_hash,
            "correction_normalized_output_hash": self.correction_normalized_output_hash,
            "correction_role_invocation_audit_hash": self.correction_role_invocation_audit_hash,
        }


@dataclass(frozen=True, slots=True)
class DataProviderPublicContract:
    """Sentinel-free public artifacts resolved from an opaque ingress receipt."""

    blueprint_artifact: Artifact
    role_audit_artifact: Artifact
    correction_role_audit_artifact: Artifact
    base_skill_artifact: Artifact
    base_skill: DataSourceSkill


class FixtureDataSource:
    """Restartable adapter whose private ledger determines fault/retry state."""

    def __init__(
        self,
        root: Path,
        exchange: _DataProviderExchange,
        config: FixtureDataIngestConfig,
        *,
        blueprint_artifact: Artifact,
        role_audit_artifact: Artifact,
        correction_role_audit_artifact: Artifact,
        skill_artifact: Artifact,
        correction: _DataProviderCorrection,
    ) -> None:
        self.root = root
        self.exchange = exchange
        self.config = config
        self.private_root = root / "private-boundary" / "data-source" / exchange.skill.source_id
        self.blueprint_artifact = blueprint_artifact
        self.role_audit_artifact = role_audit_artifact
        self.correction_role_audit_artifact = correction_role_audit_artifact
        self.skill_artifact = skill_artifact
        self.correction = correction

    @classmethod
    def stage_private_role_exchange(
        cls,
        root: str | Path,
        *,
        input_packet: dict[str, object],
        raw_role_output: bytes,
        role_session_lineage: str,
        correction_input_packet: dict[str, object],
        correction_raw_role_output: bytes,
        correction_role_session_lineage: str,
    ) -> DataProviderIngressReceipt:
        """Validate/stage exact role bytes behind the adapter-private boundary."""

        exchange = _parse_data_provider_exchange(
            input_packet,
            raw_role_output,
            role_session_lineage=role_session_lineage,
        )
        correction = _parse_data_provider_correction(
            correction_input_packet,
            correction_raw_role_output,
            role_session_lineage=correction_role_session_lineage,
            previous_output_hash=exchange.raw_output_hash,
        )
        corrected_exchange = _apply_data_provider_correction(exchange, correction)
        store = ArtifactStore(root)
        blueprint = store.put(_PUBLIC_BLUEPRINT_SCHEMA, "1.0.0", corrected_exchange.public_blueprint())
        base_skill = store.put("DataSourceSkill", "1.0.0", corrected_exchange.skill.artifact_payload())
        role_audit_payload = cls._role_audit_payload(exchange)
        role_audit = store.put("RoleInvocationAudit", "1.0.0", role_audit_payload)
        correction_audit = store.put(
            "RoleInvocationAudit",
            "1.0.0",
            cls._correction_role_audit_payload(exchange, correction, corrected_exchange),
        )
        identity = {
            "blueprint_hash": blueprint.content_hash,
            "data_source_skill_hash": base_skill.content_hash,
            "input_packet_hash": exchange.input_packet_hash,
            "raw_output_hash": exchange.raw_output_hash,
            "normalized_output_hash": exchange.normalized_output_hash,
            "role_invocation_audit_hash": role_audit.content_hash,
            "correction_input_packet_hash": correction.input_packet_hash,
            "correction_raw_output_hash": correction.raw_output_hash,
            "correction_normalized_output_hash": correction.normalized_output_hash,
            "correction_role_invocation_audit_hash": correction_audit.content_hash,
        }
        ingress_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "domain": "data-provider-private-ingress/1.0.0",
                    "identity": identity,
                }
            )
        )
        receipt = DataProviderIngressReceipt(
            ingress_hash=ingress_hash,
            blueprint_hash=blueprint.content_hash,
            data_source_skill_hash=base_skill.content_hash,
            input_packet_hash=exchange.input_packet_hash,
            raw_output_hash=exchange.raw_output_hash,
            normalized_output_hash=exchange.normalized_output_hash,
            role_invocation_audit_hash=role_audit.content_hash,
            correction_input_packet_hash=correction.input_packet_hash,
            correction_raw_output_hash=correction.raw_output_hash,
            correction_normalized_output_hash=correction.normalized_output_hash,
            correction_role_invocation_audit_hash=correction_audit.content_hash,
        )
        private_root = Path(root) / "private-boundary" / "data-provider-ingress" / ingress_hash
        ArtifactStore.durable_mkdir(private_root)
        ArtifactStore._publish(private_root / "role-output.raw.json", exchange._raw_output)
        ArtifactStore._publish(
            private_root / "role-output.normalized.canonical.json",
            exchange._normalized_output_bytes,
        )
        ArtifactStore._publish(private_root / "role-input.canonical.json", exchange._input_packet_bytes)
        ArtifactStore._publish(private_root / "correction-role-output.raw.json", correction._raw_output)
        ArtifactStore._publish(
            private_root / "correction-role-output.normalized.canonical.json",
            correction._normalized_output_bytes,
        )
        ArtifactStore._publish(
            private_root / "correction-role-input.canonical.json",
            correction._input_packet_bytes,
        )
        ArtifactStore._publish(
            private_root / "ingress-receipt.canonical.json",
            canonical_json_bytes(
                {
                    **receipt.artifact_payload(),
                    "raw_output_size": exchange.raw_output_size,
                    "normalized_output_size": len(exchange._normalized_output_bytes),
                    "correction_raw_output_size": correction.raw_output_size,
                    "correction_normalized_output_hash": correction.normalized_output_hash,
                    "correction_normalized_output_size": len(correction._normalized_output_bytes),
                    "schema_version": "data-provider-ingress/1.0.0",
                }
            ),
        )
        return receipt

    @staticmethod
    def _role_audit_payload(exchange: _DataProviderExchange) -> dict[str, object]:
        return {
            "input_hash": exchange.input_packet_hash,
            "input_packet_hash": exchange.input_packet_hash,
            "normalized_output_hash": exchange.normalized_output_hash,
            "normalized_output_size": len(exchange._normalized_output_bytes),
            "output_hash": exchange.raw_output_hash,
            "packet_id": exchange.packet_id,
            "raw_output_hash": exchange.raw_output_hash,
            "raw_output_size": exchange.raw_output_size,
            "role_session_lineage": exchange.role_session_lineage,
            "role_type": "DataProvider",
            "seed": exchange.seed,
        }

    @staticmethod
    def _correction_role_audit_payload(
        exchange: _DataProviderExchange,
        correction: _DataProviderCorrection,
        corrected_exchange: _DataProviderExchange,
    ) -> dict[str, object]:
        return {
            "input_hash": correction.input_packet_hash,
            "input_packet_hash": correction.input_packet_hash,
            "normalized_output_hash": correction.normalized_output_hash,
            "normalized_output_size": len(correction._normalized_output_bytes),
            "output_hash": correction.raw_output_hash,
            "packet_id": correction.packet_id,
            "previous_output_hash": exchange.raw_output_hash,
            "prior_packet_id": correction.prior_packet_id,
            "raw_output_hash": correction.raw_output_hash,
            "raw_output_size": correction.raw_output_size,
            "role_session_lineage": correction.role_session_lineage,
            "role_type": "DataProviderCorrection",
            "seed": correction.seed,
        }

    @classmethod
    def load_public_contract(
        cls,
        root: str | Path,
        receipt: DataProviderIngressReceipt,
    ) -> DataProviderPublicContract:
        """Resolve only sentinel-free public artifacts; never reads role bytes."""

        if type(receipt) is not DataProviderIngressReceipt:
            raise DataContractError("opaque Data Provider ingress receipt is required")
        store = ArtifactStore(root)
        blueprint = store.read(receipt.blueprint_hash, expected_schema_name=_PUBLIC_BLUEPRINT_SCHEMA)
        role_audit = store.read(receipt.role_invocation_audit_hash, expected_schema_name="RoleInvocationAudit")
        correction_role_audit = store.read(
            receipt.correction_role_invocation_audit_hash,
            expected_schema_name="RoleInvocationAudit",
        )
        base_skill_artifact = store.read(receipt.data_source_skill_hash, expected_schema_name="DataSourceSkill")
        base_skill = DataSourceSkill.from_mapping(cast(dict[str, object], base_skill_artifact.payload))
        if (
            role_audit.schema_version != "1.0.0"
            or role_audit.payload.get("input_packet_hash") != receipt.input_packet_hash
            or role_audit.payload.get("raw_output_hash") != receipt.raw_output_hash
            or role_audit.payload.get("normalized_output_hash") != receipt.normalized_output_hash
            or correction_role_audit.schema_version != "1.0.0"
            or correction_role_audit.payload.get("input_packet_hash") != receipt.correction_input_packet_hash
            or correction_role_audit.payload.get("raw_output_hash") != receipt.correction_raw_output_hash
            or correction_role_audit.payload.get("normalized_output_hash") != receipt.correction_normalized_output_hash
            or correction_role_audit.payload.get("previous_output_hash") != receipt.raw_output_hash
            or base_skill.artifact_payload() != base_skill_artifact.payload
        ):
            raise DataSourceBoundaryError("PUBLIC_SOURCE_ARTIFACT_SUBSTITUTION")
        return DataProviderPublicContract(
            blueprint,
            role_audit,
            correction_role_audit,
            base_skill_artifact,
            base_skill,
        )

    @staticmethod
    def effective_skill(base_skill: DataSourceSkill, config: FixtureDataIngestConfig) -> DataSourceSkill:
        if config.window != base_skill.approved_window:
            raise DataContractError("config window does not match the approved Data Provider window")
        return replace(
            base_skill,
            approved_data_mix=replace(
                base_skill.approved_data_mix,
                policy_version=config.data_mix_policy_version,
            ),
            sanitizer_version=config.sanitizer_policy_version,
            selection_policy_version=config.selection_policy_version,
        )

    @classmethod
    def _load_private_exchange(
        cls,
        root: str | Path,
        receipt: DataProviderIngressReceipt,
    ) -> tuple[_DataProviderExchange, _DataProviderCorrection]:
        private_root = Path(root) / "private-boundary" / "data-provider-ingress" / receipt.ingress_hash
        try:
            raw_output = (private_root / "role-output.raw.json").read_bytes()
            normalized_output = (private_root / "role-output.normalized.canonical.json").read_bytes()
            input_bytes = (private_root / "role-input.canonical.json").read_bytes()
            correction_raw_output = (private_root / "correction-role-output.raw.json").read_bytes()
            correction_normalized_output = (
                private_root / "correction-role-output.normalized.canonical.json"
            ).read_bytes()
            correction_input_bytes = (private_root / "correction-role-input.canonical.json").read_bytes()
            manifest_bytes = (private_root / "ingress-receipt.canonical.json").read_bytes()
            input_packet = json.loads(input_bytes, object_pairs_hook=_strict_object)
            correction_input_packet = json.loads(correction_input_bytes, object_pairs_hook=_strict_object)
            manifest = json.loads(manifest_bytes, object_pairs_hook=_strict_object)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise DataSourceBoundaryError("PRIVATE_BOUNDARY_STATE_CORRUPTION") from error
        if (
            not isinstance(input_packet, dict)
            or not isinstance(correction_input_packet, dict)
            or not isinstance(manifest, dict)
            or canonical_json_bytes(input_packet) != input_bytes
            or canonical_json_bytes(correction_input_packet) != correction_input_bytes
            or canonical_json_bytes(manifest) != manifest_bytes
            or manifest
            != {
                **receipt.artifact_payload(),
                "raw_output_size": len(raw_output),
                "normalized_output_size": len(normalized_output),
                "correction_raw_output_size": len(correction_raw_output),
                "correction_normalized_output_hash": sha256_hex(correction_normalized_output),
                "correction_normalized_output_size": len(correction_normalized_output),
                "schema_version": "data-provider-ingress/1.0.0",
            }
        ):
            raise DataSourceBoundaryError("PRIVATE_BOUNDARY_STATE_CORRUPTION")
        exchange = _parse_data_provider_exchange(
            cast(dict[str, object], input_packet),
            raw_output,
            role_session_lineage="/root/ticket03_data_provider",
        )
        correction = _parse_data_provider_correction(
            cast(dict[str, object], correction_input_packet),
            correction_raw_output,
            role_session_lineage="/root/ticket03_data_provider_correction",
            previous_output_hash=exchange.raw_output_hash,
        )
        if (
            exchange._normalized_output_bytes != normalized_output
            or exchange.normalized_output_hash != receipt.normalized_output_hash
            or correction._normalized_output_bytes != correction_normalized_output
            or correction.normalized_output_hash != receipt.correction_normalized_output_hash
        ):
            raise DataSourceBoundaryError("PRIVATE_BOUNDARY_STATE_CORRUPTION")
        corrected_exchange = _apply_data_provider_correction(exchange, correction)
        public = cls.load_public_contract(root, receipt)
        if (
            exchange.input_packet_hash != receipt.input_packet_hash
            or exchange.raw_output_hash != receipt.raw_output_hash
            or correction.input_packet_hash != receipt.correction_input_packet_hash
            or correction.raw_output_hash != receipt.correction_raw_output_hash
            or corrected_exchange.public_blueprint() != public.blueprint_artifact.payload
            or corrected_exchange.skill.artifact_payload() != public.base_skill_artifact.payload
            or cls._role_audit_payload(exchange) != public.role_audit_artifact.payload
            or cls._correction_role_audit_payload(exchange, correction, corrected_exchange)
            != public.correction_role_audit_artifact.payload
        ):
            raise DataSourceBoundaryError("PRIVATE_BOUNDARY_STATE_CORRUPTION")
        return corrected_exchange, correction

    @classmethod
    def bootstrap(
        cls,
        root: str | Path,
        receipt: DataProviderIngressReceipt,
        config: FixtureDataIngestConfig,
    ) -> FixtureDataSource:
        if type(receipt) is not DataProviderIngressReceipt or not isinstance(config, FixtureDataIngestConfig):
            raise DataContractError("validated fixture ingress receipt and config are required")
        exchange, correction = cls._load_private_exchange(root, receipt)
        effective_skill = cls.effective_skill(exchange.skill, config)
        exchange = replace(exchange, skill=effective_skill)
        # Revalidate SQL before creating any boundary directory.
        parse_and_validate_select(config.query_sql, effective_skill)
        root_path = Path(root)
        private_root = root_path / "private-boundary" / "data-source" / exchange.skill.source_id
        ArtifactStore.durable_mkdir(private_root / "queries")
        ArtifactStore._publish(private_root / "role-output.raw.json", exchange._raw_output)
        ArtifactStore._publish(
            private_root / "role-output.normalized.canonical.json",
            exchange._normalized_output_bytes,
        )
        ArtifactStore._publish(private_root / "role-input.canonical.json", exchange._input_packet_bytes)
        ArtifactStore._publish(private_root / "correction-role-output.raw.json", correction._raw_output)
        ArtifactStore._publish(
            private_root / "correction-role-output.normalized.canonical.json",
            correction._normalized_output_bytes,
        )
        ArtifactStore._publish(
            private_root / "correction-role-input.canonical.json",
            correction._input_packet_bytes,
        )
        ArtifactStore._publish(
            private_root / "fixture-config.canonical.json",
            canonical_json_bytes(config.artifact_payload()),
        )
        store = ArtifactStore(root_path)
        blueprint = store.read(receipt.blueprint_hash, expected_schema_name=_PUBLIC_BLUEPRINT_SCHEMA)
        skill_artifact = store.put("DataSourceSkill", "1.0.0", exchange.skill.artifact_payload())
        role_audit = store.read(receipt.role_invocation_audit_hash, expected_schema_name="RoleInvocationAudit")
        correction_role_audit = store.read(
            receipt.correction_role_invocation_audit_hash,
            expected_schema_name="RoleInvocationAudit",
        )
        return cls(
            root_path,
            exchange,
            config,
            blueprint_artifact=blueprint,
            role_audit_artifact=role_audit,
            correction_role_audit_artifact=correction_role_audit,
            skill_artifact=skill_artifact,
            correction=correction,
        )

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        expected_blueprint_hash: str,
        expected_role_audit_hash: str,
        expected_correction_role_audit_hash: str,
        expected_skill_hash: str,
        expected_config: FixtureDataIngestConfig,
    ) -> FixtureDataSource:
        """Reconstruct the query-capable adapter from verified persisted state."""

        return cls._restore_verified(
            root,
            expected_blueprint_hash=expected_blueprint_hash,
            expected_role_audit_hash=expected_role_audit_hash,
            expected_correction_role_audit_hash=expected_correction_role_audit_hash,
            expected_skill_hash=expected_skill_hash,
            expected_config=expected_config,
        )

    @classmethod
    def restore_read_only(
        cls,
        root: str | Path,
        *,
        expected_blueprint_hash: str,
        expected_role_audit_hash: str,
        expected_correction_role_audit_hash: str,
        expected_skill_hash: str,
        expected_config: FixtureDataIngestConfig,
    ) -> FixtureDataSource:
        """Verify and reconstruct persisted state without publishing or querying.

        Terminal replay uses this path so a closed run can revalidate the complete
        public/private lineage while remaining structurally unable to enter the
        query-capable boundary factory hook.
        """

        return cls._restore_verified(
            root,
            expected_blueprint_hash=expected_blueprint_hash,
            expected_role_audit_hash=expected_role_audit_hash,
            expected_correction_role_audit_hash=expected_correction_role_audit_hash,
            expected_skill_hash=expected_skill_hash,
            expected_config=expected_config,
        )

    @classmethod
    def _restore_verified(
        cls,
        root: str | Path,
        *,
        expected_blueprint_hash: str,
        expected_role_audit_hash: str,
        expected_correction_role_audit_hash: str,
        expected_skill_hash: str,
        expected_config: FixtureDataIngestConfig,
    ) -> FixtureDataSource:
        """Shared read-only reconstruction used by open and terminal verify."""

        root_path = Path(root)
        private_root = root_path / "private-boundary" / "data-source" / "fixture-online-traces-v1"
        try:
            raw_output = (private_root / "role-output.raw.json").read_bytes()
            normalized_output = (private_root / "role-output.normalized.canonical.json").read_bytes()
            input_bytes = (private_root / "role-input.canonical.json").read_bytes()
            correction_raw_output = (private_root / "correction-role-output.raw.json").read_bytes()
            correction_normalized_output = (
                private_root / "correction-role-output.normalized.canonical.json"
            ).read_bytes()
            correction_input_bytes = (private_root / "correction-role-input.canonical.json").read_bytes()
            config_bytes = (private_root / "fixture-config.canonical.json").read_bytes()
            input_packet = json.loads(input_bytes, object_pairs_hook=_strict_object)
            correction_input_packet = json.loads(correction_input_bytes, object_pairs_hook=_strict_object)
            config_value = json.loads(config_bytes, object_pairs_hook=_strict_object)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise DataSourceBoundaryError("PRIVATE_BOUNDARY_STATE_CORRUPTION") from error
        if (
            not isinstance(input_packet, dict)
            or not isinstance(correction_input_packet, dict)
            or not isinstance(config_value, dict)
            or canonical_json_bytes(input_packet) != input_bytes
            or canonical_json_bytes(correction_input_packet) != correction_input_bytes
            or canonical_json_bytes(config_value) != config_bytes
        ):
            raise DataSourceBoundaryError("PRIVATE_BOUNDARY_STATE_CORRUPTION")
        exchange = _parse_data_provider_exchange(
            cast(dict[str, object], input_packet),
            raw_output,
            role_session_lineage="/root/ticket03_data_provider",
        )
        base_exchange = exchange
        correction = _parse_data_provider_correction(
            cast(dict[str, object], correction_input_packet),
            correction_raw_output,
            role_session_lineage="/root/ticket03_data_provider_correction",
            previous_output_hash=exchange.raw_output_hash,
        )
        if (
            exchange._normalized_output_bytes != normalized_output
            or correction._normalized_output_bytes != correction_normalized_output
        ):
            raise DataSourceBoundaryError("PRIVATE_BOUNDARY_STATE_CORRUPTION")
        exchange = _apply_data_provider_correction(exchange, correction)
        config = FixtureDataIngestConfig.from_mapping(cast(dict[str, object], config_value))
        if config.artifact_payload() != expected_config.artifact_payload():
            raise DataSourceBoundaryError("PRIVATE_BOUNDARY_CONFIG_SUBSTITUTION")
        effective_skill = cls.effective_skill(exchange.skill, config)
        exchange = replace(exchange, skill=effective_skill)
        store = ArtifactStore(root_path)
        try:
            blueprint = store.read(expected_blueprint_hash, expected_schema_name=_PUBLIC_BLUEPRINT_SCHEMA)
            role_audit = store.read(expected_role_audit_hash, expected_schema_name="RoleInvocationAudit")
            correction_role_audit = store.read(
                expected_correction_role_audit_hash,
                expected_schema_name="RoleInvocationAudit",
            )
            skill = store.read(expected_skill_hash, expected_schema_name="DataSourceSkill")
        except Exception as error:
            raise DataSourceBoundaryError("PUBLIC_SOURCE_ARTIFACT_CORRUPTION") from error
        if (
            blueprint.payload != exchange.public_blueprint()
            or skill.payload != effective_skill.artifact_payload()
            or role_audit.payload != cls._role_audit_payload(base_exchange)
            or correction_role_audit.payload != cls._correction_role_audit_payload(base_exchange, correction, exchange)
        ):
            raise DataSourceBoundaryError("PUBLIC_SOURCE_ARTIFACT_SUBSTITUTION")
        return cls(
            root_path,
            exchange,
            config,
            blueprint_artifact=blueprint,
            role_audit_artifact=role_audit,
            correction_role_audit_artifact=correction_role_audit,
            skill_artifact=skill,
            correction=correction,
        )

    @staticmethod
    def fault_schedule_hash(config: FixtureDataIngestConfig) -> str:
        return sha256_hex(
            canonical_json_bytes(
                {
                    "directives": list(config.fault_schedule),
                    "domain": "fixture-fault-schedule/1.0.0",
                    "source_fault": config.source_fault,
                }
            )
        )

    @staticmethod
    def query_contract_hash(ast: SelectAst, parameters: Mapping[str, object]) -> str:
        return sha256_hex(
            canonical_json_bytes(
                {
                    "ast": ast.artifact_payload(),
                    "domain": "governed-query-contract/1.0.0",
                    "parameters": dict(parameters),
                }
            )
        )

    @staticmethod
    def clock_policy_hash() -> str:
        return sha256_hex(
            canonical_json_bytes(
                {
                    "domain": "fixture-query-clock/1.0.0",
                    "elapsed_rule": "end_tick_minus_start_tick",
                    "start_rule": "(attempt_ordinal_minus_one)_times_16",
                }
            )
        )

    @staticmethod
    def boundary_request_payload(
        *,
        query_key: str,
        ast: SelectAst,
        parameters: Mapping[str, object],
        input_hash: str,
        query_hash: str,
        schedule_hash: str,
        source_id: str,
        window_hash: str,
    ) -> dict[str, object]:
        return {
            "ast": ast.artifact_payload(),
            "input_hash": input_hash,
            "parameters": dict(parameters),
            "query_hash": query_hash,
            "query_key": query_key,
            "schedule_hash": schedule_hash,
            "schema_version": "fixture-query-request/1.0.0",
            "source_id": source_id,
            "window_hash": window_hash,
        }

    @classmethod
    def boundary_request_hash(
        cls,
        *,
        query_key: str,
        ast: SelectAst,
        parameters: Mapping[str, object],
        input_hash: str,
        query_hash: str,
        schedule_hash: str,
        source_id: str,
        window_hash: str,
    ) -> str:
        return sha256_hex(
            canonical_json_bytes(
                cls.boundary_request_payload(
                    query_key=query_key,
                    ast=ast,
                    parameters=parameters,
                    input_hash=input_hash,
                    query_hash=query_hash,
                    schedule_hash=schedule_hash,
                    source_id=source_id,
                    window_hash=window_hash,
                )
            )
        )

    def query(
        self,
        *,
        query_key: str,
        ast: SelectAst,
        parameters: Mapping[str, object],
        input_hash: str,
        query_hash: str,
    ) -> QueryResult:
        """Serialize one idempotency key across processes until its receipt commits."""

        if not isinstance(query_key, str) or _SAFE_KEY.fullmatch(query_key) is None:
            raise DataSourceBoundaryError("INVALID_QUERY_KEY")
        if re.fullmatch(r"[0-9a-f]{64}", input_hash) is None or re.fullmatch(r"[0-9a-f]{64}", query_hash) is None:
            raise DataSourceBoundaryError("INVALID_QUERY_REQUEST_HASH")
        query_dir = self.private_root / "queries" / query_key
        ArtifactStore.durable_mkdir(query_dir)
        lock_path = query_dir / "query.lock"
        ArtifactStore.durable_touch(lock_path)
        with lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                return self._query_unlocked(
                    query_key=query_key,
                    ast=ast,
                    parameters=parameters,
                    input_hash=input_hash,
                    query_hash=query_hash,
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _query_unlocked(
        self,
        *,
        query_key: str,
        ast: SelectAst,
        parameters: Mapping[str, object],
        input_hash: str,
        query_hash: str,
    ) -> QueryResult:
        if not isinstance(query_key, str) or _SAFE_KEY.fullmatch(query_key) is None:
            raise DataSourceBoundaryError("INVALID_QUERY_KEY")
        approved_ast = parse_and_validate_select(self.config.query_sql, self.exchange.skill)
        if ast != approved_ast:
            raise DataSourceBoundaryError("QUERY_AST_SUBSTITUTION")
        expected_parameters = {
            "end_utc": self.config.window.end_utc,
            "purpose": self.exchange.skill.purpose,
            "start_utc": self.config.window.start_utc,
        }
        if parameters != expected_parameters:
            raise DataSourceBoundaryError("QUERY_PARAMETERS_OUTSIDE_POLICY")
        if query_hash != self.query_contract_hash(ast, parameters):
            raise DataSourceBoundaryError("QUERY_AST_SUBSTITUTION")
        schedule_hash = self.fault_schedule_hash(self.config)
        request_bytes = canonical_json_bytes(
            self.boundary_request_payload(
                query_key=query_key,
                ast=ast,
                parameters=parameters,
                input_hash=input_hash,
                query_hash=query_hash,
                schedule_hash=schedule_hash,
                source_id=self.exchange.skill.source_id,
                window_hash=self.exchange.skill.approved_window.content_hash,
            )
        )
        request_hash = sha256_hex(request_bytes)
        query_dir = self.private_root / "queries" / query_key
        ArtifactStore.durable_mkdir(query_dir / "invocations")
        ArtifactStore._publish(query_dir / "request.canonical.json", request_bytes)
        attempts = self._verify_attempt_chain_unlocked(
            query_key=query_key,
            expected_request_hash=request_hash,
            expected_query_hash=query_hash,
            expected_input_hash=input_hash,
            expected_schedule_hash=schedule_hash,
        )
        success_path = query_dir / "success.canonical.json"
        if attempts and attempts[-1].status == "success":
            return self._read_success(
                success_path,
                request_hash,
                query_key,
                expected_attempt=attempts[-1],
            )
        attempt_ordinal = len(attempts) + 1
        if attempt_ordinal > len(self.config.fault_schedule):
            raise DataSourceBoundaryError("RETRY_POLICY_EXHAUSTED")
        directive = self.config.fault_schedule[attempt_ordinal - 1]
        elapsed_ticks = (
            self.exchange.skill.max_elapsed_ticks + 1
            if self.config.source_fault == "time_limit" and directive == "success"
            else attempt_ordinal
        )
        clock_start_tick = (attempt_ordinal - 1) * 16
        clock_end_tick = clock_start_tick + elapsed_ticks
        previous_attempt_hash = attempts[-1].attempt_hash if attempts else None

        def commit_attempt(
            *,
            status: str,
            failure_code: str | None,
            result_hash: str | None = None,
            result_size: int = 0,
            late_result_hash: str | None = None,
            raw_result_hash: str | None = None,
            raw_result_size: int = 0,
            raw_row_count: int = 0,
        ) -> QueryAttemptEvidence:
            payload = {
                "attempt_ordinal": attempt_ordinal,
                "clock_end_tick": clock_end_tick,
                "clock_policy_hash": self.clock_policy_hash(),
                "clock_start_tick": clock_start_tick,
                "directive": directive,
                "elapsed_ticks": elapsed_ticks,
                "failure_code": failure_code,
                "input_hash": input_hash,
                "late_result_hash": late_result_hash,
                "previous_attempt_hash": previous_attempt_hash,
                "query_hash": query_hash,
                "query_key": query_key,
                "raw_result_hash": raw_result_hash,
                "raw_result_size": raw_result_size,
                "raw_row_count": raw_row_count,
                "request_hash": request_hash,
                "result_hash": result_hash,
                "result_size": result_size,
                "schedule_hash": schedule_hash,
                "schema_version": "fixture-query-attempt/1.0.0",
                "status": status,
            }
            raw = canonical_json_bytes(payload)
            ArtifactStore._publish(
                query_dir / "invocations" / f"{attempt_ordinal:020d}.json",
                raw,
            )
            return self._decode_attempt(raw, expected_ordinal=attempt_ordinal, expected_previous=previous_attempt_hash)

        if elapsed_ticks > self.exchange.skill.max_elapsed_ticks:
            commit_attempt(status="boundary_failure", failure_code="QUERY_TIME_LIMIT_EXCEEDED")
            raise DataSourceBoundaryError("QUERY_TIME_LIMIT_EXCEEDED")
        if directive != "success":
            if directive == "late":
                raw_rows = self._generate_raw_rows()
                late_bytes = canonical_json_bytes(raw_rows)
                self._schedule_late_delivery(
                    query_dir=query_dir,
                    attempt_ordinal=attempt_ordinal,
                    due_tick=attempt_ordinal * 16 + 1,
                    request_hash=request_hash,
                    query_hash=query_hash,
                    input_hash=input_hash,
                    schedule_hash=schedule_hash,
                    payload=late_bytes,
                    row_count=len(raw_rows),
                )
            failure_code = {
                "delayed": "QUERY_DELAYED",
                "error": "QUERY_PROVIDER_ERROR",
                "late": "QUERY_DELAYED",
                "permanent_failure": "QUERY_PROVIDER_PERMANENT_FAILURE",
                "timeout": "QUERY_TIMEOUT",
            }[directive]
            status = "delayed" if directive == "late" else directive
            attempt = commit_attempt(status=status, failure_code=failure_code)
            return QueryResult(
                status=status,
                failure_code=failure_code,
                raw_row_count=0,
                raw_byte_count=0,
                elapsed_ticks=elapsed_ticks,
                request_hash=request_hash,
                query_hash=query_hash,
                input_hash=input_hash,
                schedule_hash=schedule_hash,
                attempt_hash=attempt.attempt_hash,
                result_hash=None,
                result_size=0,
                private_result_content_hash=None,
                late_result_hash=None,
                raw_result_hash=None,
                attempt_ordinal=attempt_ordinal,
                _sanitized_bytes=b"[]",
            )
        raw_rows = self._generate_raw_rows()
        raw_bytes = canonical_json_bytes(raw_rows)
        if len(raw_rows) > self.exchange.skill.max_rows:
            commit_attempt(status="boundary_failure", failure_code="QUERY_ROW_LIMIT_EXCEEDED")
            raise DataSourceBoundaryError("QUERY_ROW_LIMIT_EXCEEDED")
        if len(raw_bytes) > self.exchange.skill.max_bytes:
            commit_attempt(status="boundary_failure", failure_code="QUERY_BYTE_LIMIT_EXCEEDED")
            raise DataSourceBoundaryError("QUERY_BYTE_LIMIT_EXCEEDED")
        if len(raw_rows) != self.exchange.skill.expected_raw_rows:
            commit_attempt(status="boundary_failure", failure_code="QUERY_CARDINALITY_INVALID")
            raise DataSourceBoundaryError("QUERY_CARDINALITY_INVALID")
        raw_result_hash = sha256_hex(raw_bytes)
        # This is the only persisted OnlineTrace batch and it remains private.
        ArtifactStore._publish(query_dir / "online-traces.private.json", raw_bytes)
        try:
            sanitized = self._validate_and_sanitize(raw_rows)
        except DataSourceBoundaryError as error:
            code = (
                error.args[0] if len(error.args) == 1 and type(error.args[0]) is str else "DATA_SOURCE_BOUNDARY_FAILED"
            )
            commit_attempt(
                status="boundary_failure",
                failure_code=cast(str, code),
                raw_result_hash=raw_result_hash,
                raw_result_size=len(raw_bytes),
                raw_row_count=len(raw_rows),
            )
            raise
        sanitized_bytes = canonical_json_bytes(sanitized)
        result_hash = sha256_hex(sanitized_bytes)
        result_path = query_dir / "results" / f"{result_hash}.sanitized.json"
        ArtifactStore._publish(result_path, sanitized_bytes)
        attempt = commit_attempt(
            status="success",
            failure_code=None,
            result_hash=result_hash,
            result_size=len(sanitized_bytes),
            raw_result_hash=raw_result_hash,
            raw_result_size=len(raw_bytes),
            raw_row_count=len(raw_rows),
        )
        success = canonical_json_bytes(
            {
                "attempt_hash": attempt.attempt_hash,
                "attempt_ordinal": attempt_ordinal,
                "clock_end_tick": clock_end_tick,
                "clock_policy_hash": self.clock_policy_hash(),
                "clock_start_tick": clock_start_tick,
                "elapsed_ticks": elapsed_ticks,
                "input_hash": input_hash,
                "query_hash": query_hash,
                "query_key": query_key,
                "raw_byte_count": len(raw_bytes),
                "raw_result_hash": raw_result_hash,
                "raw_row_count": len(raw_rows),
                "request_hash": request_hash,
                "result_hash": result_hash,
                "result_size": len(sanitized_bytes),
                "schedule_hash": schedule_hash,
                "schema_version": "fixture-query-success/1.0.0",
                "status": "success",
            }
        )
        ArtifactStore._publish(success_path, success)
        self._deliver_due_deliveries(
            query_dir=query_dir,
            attempts=(*attempts, attempt),
            current_tick=clock_end_tick,
            covering_attempt=attempt,
        )
        return self._read_success(success_path, request_hash, query_key, expected_attempt=attempt)

    @staticmethod
    def _schedule_late_delivery(
        *,
        query_dir: Path,
        attempt_ordinal: int,
        due_tick: int,
        request_hash: str,
        query_hash: str,
        input_hash: str,
        schedule_hash: str,
        payload: bytes,
        row_count: int,
    ) -> None:
        payload_hash = sha256_hex(payload)
        schedule = canonical_json_bytes(
            {
                "attempt_ordinal": attempt_ordinal,
                "due_tick": due_tick,
                "input_hash": input_hash,
                "payload_hash": payload_hash,
                "payload_size": len(payload),
                "query_hash": query_hash,
                "raw_row_count": row_count,
                "request_hash": request_hash,
                "schedule_hash": schedule_hash,
                "schema_version": "fixture-late-delivery-schedule/1.0.0",
            }
        )
        ArtifactStore._publish(query_dir / "deliveries" / "sealed" / f"{payload_hash}.private.json", payload)
        ArtifactStore._publish(
            query_dir / "deliveries" / "scheduled" / f"{attempt_ordinal:020d}.json",
            schedule,
        )

    def _deliver_due_deliveries(
        self,
        *,
        query_dir: Path,
        attempts: tuple[QueryAttemptEvidence, ...],
        current_tick: int,
        covering_attempt: QueryAttemptEvidence,
    ) -> None:
        if covering_attempt.status != "success":
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
        scheduled_dir = query_dir / "deliveries" / "scheduled"
        if not scheduled_dir.exists():
            return
        for path in sorted(scheduled_dir.glob("*.json")):
            try:
                schedule_bytes = path.read_bytes()
                schedule = json.loads(schedule_bytes, object_pairs_hook=_strict_object)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error
            if not isinstance(schedule, dict) or canonical_json_bytes(schedule) != schedule_bytes:
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            late_ordinal = schedule.get("attempt_ordinal")
            due_tick = schedule.get("due_tick")
            if (
                type(late_ordinal) is not int
                or type(due_tick) is not int
                or cast(int, late_ordinal) >= covering_attempt.attempt_ordinal
                or cast(int, due_tick) > current_tick
                or cast(int, late_ordinal) > len(attempts)
                or attempts[cast(int, late_ordinal) - 1].directive != "late"
            ):
                continue
            payload_hash = cast(str, schedule.get("payload_hash"))
            sealed_path = query_dir / "deliveries" / "sealed" / f"{payload_hash}.private.json"
            try:
                payload = sealed_path.read_bytes()
            except OSError as error:
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error
            arrived_path = (
                query_dir / "deliveries" / "arrived" / f"{cast(int, late_ordinal):020d}-{payload_hash}.private.json"
            )
            receipt_path = query_dir / "deliveries" / "quarantine" / f"{cast(int, late_ordinal):020d}.json"
            receipt = canonical_json_bytes(
                {
                    "arrival_tick": current_tick,
                    "covering_attempt_hash": covering_attempt.attempt_hash,
                    "covering_attempt_ordinal": covering_attempt.attempt_ordinal,
                    "delivery_schedule_hash": sha256_hex(schedule_bytes),
                    "late_attempt_ordinal": late_ordinal,
                    "payload_hash": payload_hash,
                    "payload_size": len(payload),
                    "raw_row_count": schedule.get("raw_row_count"),
                    "reason_code": "OUT_OF_ORDER_RESULT_QUARANTINED",
                    "schema_version": "fixture-late-delivery-quarantine/1.0.0",
                    "status": "quarantined",
                }
            )
            ArtifactStore._publish(arrived_path, payload)
            ArtifactStore._publish(receipt_path, receipt)

    @classmethod
    def _decode_attempt(
        cls,
        raw: bytes,
        *,
        expected_ordinal: int,
        expected_previous: str | None,
    ) -> QueryAttemptEvidence:
        try:
            value = json.loads(raw, object_pairs_hook=_strict_object)
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error
        fields = {
            "attempt_ordinal",
            "clock_end_tick",
            "clock_policy_hash",
            "clock_start_tick",
            "directive",
            "elapsed_ticks",
            "failure_code",
            "input_hash",
            "late_result_hash",
            "previous_attempt_hash",
            "query_hash",
            "query_key",
            "raw_result_hash",
            "raw_result_size",
            "raw_row_count",
            "request_hash",
            "result_hash",
            "result_size",
            "schedule_hash",
            "schema_version",
            "status",
        }
        if not isinstance(value, dict) or set(value) != fields or canonical_json_bytes(value) != raw:
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
        integers = (
            value.get("attempt_ordinal"),
            value.get("clock_end_tick"),
            value.get("clock_start_tick"),
            value.get("elapsed_ticks"),
            value.get("raw_result_size"),
            value.get("raw_row_count"),
            value.get("result_size"),
        )
        hash_fields = (
            value.get("clock_policy_hash"),
            value.get("input_hash"),
            value.get("query_hash"),
            value.get("request_hash"),
            value.get("schedule_hash"),
        )
        optional_hashes = (
            value.get("late_result_hash"),
            value.get("previous_attempt_hash"),
            value.get("raw_result_hash"),
            value.get("result_hash"),
        )
        if (
            value.get("schema_version") != "fixture-query-attempt/1.0.0"
            or value.get("attempt_ordinal") != expected_ordinal
            or value.get("previous_attempt_hash") != expected_previous
            or any(type(item) is not int or cast(int, item) < 0 for item in integers)
            or cast(int, value.get("attempt_ordinal")) <= 0
            or any(
                type(item) is not str or re.fullmatch(r"[0-9a-f]{64}", cast(str, item)) is None for item in hash_fields
            )
            or any(
                item is not None and (type(item) is not str or re.fullmatch(r"[0-9a-f]{64}", cast(str, item)) is None)
                for item in optional_hashes
            )
            or type(value.get("directive")) is not str
            or type(value.get("query_key")) is not str
            or type(value.get("status")) is not str
            or (value.get("failure_code") is not None and type(value.get("failure_code")) is not str)
            or value.get("clock_policy_hash") != cls.clock_policy_hash()
            or cast(int, value.get("clock_start_tick")) != (expected_ordinal - 1) * 16
            or cast(int, value.get("clock_end_tick")) - cast(int, value.get("clock_start_tick"))
            != value.get("elapsed_ticks")
        ):
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
        return QueryAttemptEvidence(
            attempt_hash=sha256_hex(raw),
            attempt_ordinal=expected_ordinal,
            directive=cast(str, value["directive"]),
            status=cast(str, value["status"]),
            failure_code=cast(str | None, value["failure_code"]),
            request_hash=cast(str, value["request_hash"]),
            query_hash=cast(str, value["query_hash"]),
            input_hash=cast(str, value["input_hash"]),
            schedule_hash=cast(str, value["schedule_hash"]),
            previous_attempt_hash=cast(str | None, value["previous_attempt_hash"]),
            result_hash=cast(str | None, value["result_hash"]),
            result_size=cast(int, value["result_size"]),
            late_result_hash=cast(str | None, value["late_result_hash"]),
            raw_result_hash=cast(str | None, value["raw_result_hash"]),
            raw_result_size=cast(int, value["raw_result_size"]),
            raw_row_count=cast(int, value["raw_row_count"]),
            elapsed_ticks=cast(int, value["elapsed_ticks"]),
            clock_start_tick=cast(int, value["clock_start_tick"]),
            clock_end_tick=cast(int, value["clock_end_tick"]),
            clock_policy_hash=cast(str, value["clock_policy_hash"]),
        )

    def verify_attempt_chain(
        self,
        *,
        query_key: str,
        expected_request_hash: str,
        expected_query_hash: str,
        expected_input_hash: str,
        expected_schedule_hash: str,
    ) -> tuple[QueryAttemptEvidence, ...]:
        """Read a stable ledger snapshot while excluding an in-flight commit."""

        query_dir = self.private_root / "queries" / query_key
        lock_path = query_dir / "query.lock"
        try:
            with lock_path.open("rb") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
                try:
                    return self._verify_attempt_chain_unlocked(
                        query_key=query_key,
                        expected_request_hash=expected_request_hash,
                        expected_query_hash=expected_query_hash,
                        expected_input_hash=expected_input_hash,
                        expected_schedule_hash=expected_schedule_hash,
                    )
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except OSError as error:
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error

    def _verify_attempt_chain_unlocked(
        self,
        *,
        query_key: str,
        expected_request_hash: str,
        expected_query_hash: str,
        expected_input_hash: str,
        expected_schedule_hash: str,
    ) -> tuple[QueryAttemptEvidence, ...]:
        query_dir = self.private_root / "queries" / query_key
        try:
            request_bytes = (query_dir / "request.canonical.json").read_bytes()
            request = json.loads(request_bytes, object_pairs_hook=_strict_object)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error
        if (
            not isinstance(request, dict)
            or canonical_json_bytes(request) != request_bytes
            or request
            != self.boundary_request_payload(
                query_key=query_key,
                ast=parse_and_validate_select(self.config.query_sql, self.exchange.skill),
                parameters={
                    "end_utc": self.config.window.end_utc,
                    "purpose": self.exchange.skill.purpose,
                    "start_utc": self.config.window.start_utc,
                },
                input_hash=expected_input_hash,
                query_hash=expected_query_hash,
                schedule_hash=expected_schedule_hash,
                source_id=self.exchange.skill.source_id,
                window_hash=self.exchange.skill.approved_window.content_hash,
            )
            or sha256_hex(request_bytes) != expected_request_hash
            or request.get("schema_version") != "fixture-query-request/1.0.0"
            or request.get("query_key") != query_key
            or request.get("query_hash") != expected_query_hash
            or request.get("input_hash") != expected_input_hash
            or request.get("schedule_hash") != expected_schedule_hash
            or request.get("window_hash") != self.exchange.skill.approved_window.content_hash
        ):
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
        attempts_dir = query_dir / "invocations"
        if not attempts_dir.exists():
            return ()
        try:
            paths = sorted(attempts_dir.iterdir())
        except OSError as error:
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error
        if any(not path.is_file() for path in paths) or [path.name for path in paths] != [
            f"{ordinal:020d}.json" for ordinal in range(1, len(paths) + 1)
        ]:
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
        result: list[QueryAttemptEvidence] = []
        previous: str | None = None
        failure_codes = {
            "delayed": "QUERY_DELAYED",
            "error": "QUERY_PROVIDER_ERROR",
            "permanent_failure": "QUERY_PROVIDER_PERMANENT_FAILURE",
            "timeout": "QUERY_TIMEOUT",
        }
        for ordinal, path in enumerate(paths, 1):
            try:
                raw = path.read_bytes()
            except OSError as error:
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error
            attempt = self._decode_attempt(raw, expected_ordinal=ordinal, expected_previous=previous)
            expected_directive = (
                self.config.fault_schedule[ordinal - 1] if ordinal <= len(self.config.fault_schedule) else None
            )
            expected_elapsed = (
                self.exchange.skill.max_elapsed_ticks + 1
                if expected_directive == "success" and self.config.source_fault == "time_limit"
                else ordinal
            )
            if (
                attempt.directive != expected_directive
                or attempt.request_hash != expected_request_hash
                or attempt.query_hash != expected_query_hash
                or attempt.input_hash != expected_input_hash
                or attempt.schedule_hash != expected_schedule_hash
                or attempt.elapsed_ticks != expected_elapsed
            ):
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            expected_boundary_raw: bool | None = None
            if attempt.directive == "success":
                boundary_failure = _BOUNDARY_FAILURE_BY_SOURCE_FAULT.get(self.config.source_fault or "")
                expected_status = "boundary_failure" if boundary_failure is not None else "success"
                expected_failure = boundary_failure[0] if boundary_failure is not None else None
                expected_boundary_raw = boundary_failure[1] if boundary_failure is not None else None
                if attempt.status != expected_status or attempt.failure_code != expected_failure:
                    raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            elif attempt.directive == "late":
                if attempt.status != "delayed" or attempt.failure_code != "QUERY_DELAYED":
                    raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            elif attempt.status != attempt.directive or attempt.failure_code != failure_codes.get(attempt.directive):
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            self._verify_attempt_outcome_fields(attempt)
            if expected_boundary_raw is not None:
                raw_fields_present = (
                    attempt.raw_result_hash is not None and attempt.raw_result_size > 0 and attempt.raw_row_count > 0
                )
                raw_file_present = (query_dir / "online-traces.private.json").is_file()
                if raw_fields_present is not expected_boundary_raw or raw_file_present is not expected_boundary_raw:
                    raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            if attempt.late_result_hash is not None:
                late_path = query_dir / f"late-{ordinal:020d}.private.json"
                try:
                    late_bytes = late_path.read_bytes()
                except OSError as error:
                    raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error
                if sha256_hex(late_bytes) != attempt.late_result_hash:
                    raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            elif (query_dir / f"late-{ordinal:020d}.private.json").exists():
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            if attempt.raw_result_hash is not None:
                try:
                    raw_result = (query_dir / "online-traces.private.json").read_bytes()
                    raw_rows = json.loads(raw_result, object_pairs_hook=_strict_object)
                except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                    raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error
                if (
                    not isinstance(raw_rows, list)
                    or canonical_json_bytes(raw_rows) != raw_result
                    or sha256_hex(raw_result) != attempt.raw_result_hash
                    or len(raw_result) != attempt.raw_result_size
                    or len(raw_rows) != attempt.raw_row_count
                ):
                    raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
                expected_raw = canonical_json_bytes(self._generate_raw_rows())
                if raw_result != expected_raw:
                    raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            elif attempt.status == "boundary_failure" and (query_dir / "online-traces.private.json").exists():
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            if attempt.result_hash is not None:
                try:
                    result_bytes = (query_dir / "results" / f"{attempt.result_hash}.sanitized.json").read_bytes()
                    result_rows = json.loads(result_bytes, object_pairs_hook=_strict_object)
                except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                    raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error
                if (
                    not isinstance(result_rows, list)
                    or any(not isinstance(row, dict) for row in result_rows)
                    or canonical_json_bytes(result_rows) != result_bytes
                    or sha256_hex(result_bytes) != attempt.result_hash
                    or len(result_bytes) != attempt.result_size
                    or len(result_rows) != attempt.raw_row_count
                ):
                    raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            previous = attempt.attempt_hash
            result.append(attempt)
        if any(attempt.status in {"success", "permanent_failure", "boundary_failure"} for attempt in result[:-1]):
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
        self._verify_delivery_queue(query_dir, tuple(result))
        return tuple(result)

    def _verify_delivery_queue(
        self,
        query_dir: Path,
        attempts: tuple[QueryAttemptEvidence, ...],
    ) -> None:
        delivery_root = query_dir / "deliveries"
        late_attempts = [attempt for attempt in attempts if attempt.directive == "late"]
        if not late_attempts:
            if delivery_root.exists() and any(delivery_root.rglob("*")):
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            return
        scheduled_dir = delivery_root / "scheduled"
        sealed_dir = delivery_root / "sealed"
        arrived_dir = delivery_root / "arrived"
        quarantine_dir = delivery_root / "quarantine"
        try:
            scheduled_paths = sorted(scheduled_dir.glob("*.json"))
            quarantine_paths = sorted(quarantine_dir.glob("*.json")) if quarantine_dir.exists() else []
            arrived_paths = sorted(arrived_dir.glob("*.private.json")) if arrived_dir.exists() else []
            sealed_paths = sorted(sealed_dir.glob("*.private.json"))
        except OSError as error:
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error
        if [path.name for path in scheduled_paths] != [
            f"{attempt.attempt_ordinal:020d}.json" for attempt in late_attempts
        ]:
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
        expected_quarantine_names: list[str] = []
        expected_arrived_names: list[str] = []
        expected_sealed_names: list[str] = []
        for attempt, path in zip(late_attempts, scheduled_paths, strict=True):
            try:
                schedule_bytes = path.read_bytes()
                schedule = json.loads(schedule_bytes, object_pairs_hook=_strict_object)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error
            if not isinstance(schedule, dict) or canonical_json_bytes(schedule) != schedule_bytes:
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            payload_hash = schedule.get("payload_hash")
            if type(payload_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", cast(str, payload_hash)) is None:
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            sealed_path = sealed_dir / f"{payload_hash}.private.json"
            try:
                payload = sealed_path.read_bytes()
                rows = json.loads(payload, object_pairs_hook=_strict_object)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error
            expected_schedule = {
                "attempt_ordinal": attempt.attempt_ordinal,
                "due_tick": attempt.attempt_ordinal * 16 + 1,
                "input_hash": attempt.input_hash,
                "payload_hash": sha256_hex(payload),
                "payload_size": len(payload),
                "query_hash": attempt.query_hash,
                "raw_row_count": len(rows) if isinstance(rows, list) else -1,
                "request_hash": attempt.request_hash,
                "schedule_hash": attempt.schedule_hash,
                "schema_version": "fixture-late-delivery-schedule/1.0.0",
            }
            if (
                not isinstance(rows, list)
                or any(not isinstance(row, dict) for row in rows)
                or canonical_json_bytes(rows) != payload
                or schedule != expected_schedule
            ):
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
            expected_sealed_names.append(f"{payload_hash}.private.json")
            covering = next(
                (
                    candidate
                    for candidate in attempts
                    if candidate.attempt_ordinal > attempt.attempt_ordinal and candidate.status == "success"
                ),
                None,
            )
            receipt_path = quarantine_dir / f"{attempt.attempt_ordinal:020d}.json"
            arrived_name = f"{attempt.attempt_ordinal:020d}-{payload_hash}.private.json"
            arrived_path = arrived_dir / arrived_name
            if covering is None:
                if receipt_path.exists() or arrived_path.exists():
                    raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
                continue
            expected_quarantine_names.append(receipt_path.name)
            expected_arrived_names.append(arrived_name)
            try:
                receipt_bytes = receipt_path.read_bytes()
                receipt = json.loads(receipt_bytes, object_pairs_hook=_strict_object)
                arrived = arrived_path.read_bytes()
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error
            expected_receipt = {
                "arrival_tick": covering.clock_end_tick,
                "covering_attempt_hash": covering.attempt_hash,
                "covering_attempt_ordinal": covering.attempt_ordinal,
                "delivery_schedule_hash": sha256_hex(schedule_bytes),
                "late_attempt_ordinal": attempt.attempt_ordinal,
                "payload_hash": payload_hash,
                "payload_size": len(payload),
                "raw_row_count": len(rows),
                "reason_code": "OUT_OF_ORDER_RESULT_QUARANTINED",
                "schema_version": "fixture-late-delivery-quarantine/1.0.0",
                "status": "quarantined",
            }
            if canonical_json_bytes(receipt) != receipt_bytes or receipt != expected_receipt or arrived != payload:
                raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
        if (
            [path.name for path in quarantine_paths] != expected_quarantine_names
            or [path.name for path in arrived_paths] != expected_arrived_names
            or [path.name for path in sealed_paths] != sorted(set(expected_sealed_names))
        ):
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")

    @staticmethod
    def _verify_attempt_outcome_fields(attempt: QueryAttemptEvidence) -> None:
        """Enforce the status-specific closed-world attempt field matrix."""

        if attempt.status == "success":
            valid = (
                attempt.failure_code is None
                and attempt.late_result_hash is None
                and attempt.result_hash is not None
                and attempt.result_size > 0
                and attempt.raw_result_hash is not None
                and attempt.raw_result_size > 0
                and attempt.raw_row_count > 0
            )
        elif attempt.status in {"timeout", "error", "delayed", "permanent_failure"}:
            valid = (
                attempt.failure_code is not None
                and attempt.late_result_hash is None
                and attempt.result_hash is None
                and attempt.result_size == 0
                and attempt.raw_result_hash is None
                and attempt.raw_result_size == 0
                and attempt.raw_row_count == 0
            )
        elif attempt.status == "late":
            valid = (
                attempt.failure_code == "QUERY_LATE_RESULT_QUARANTINED"
                and attempt.late_result_hash is not None
                and attempt.result_hash is None
                and attempt.result_size == 0
                and attempt.raw_result_hash is None
                and attempt.raw_result_size == 0
                and attempt.raw_row_count == 0
            )
        elif attempt.status == "boundary_failure":
            raw_absent = attempt.raw_result_hash is None and attempt.raw_result_size == 0 and attempt.raw_row_count == 0
            raw_present = (
                attempt.raw_result_hash is not None and attempt.raw_result_size > 0 and attempt.raw_row_count > 0
            )
            valid = (
                attempt.failure_code is not None
                and attempt.late_result_hash is None
                and attempt.result_hash is None
                and attempt.result_size == 0
                and (raw_absent or raw_present)
            )
        else:
            valid = False
        if not valid:
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")

    def read_committed_result(
        self,
        *,
        query_key: str,
        expected_request_hash: str,
        expected_query_hash: str,
        expected_input_hash: str,
        expected_schedule_hash: str,
    ) -> QueryResult:
        """Read and verify a committed sanitized receipt without querying or rebuilding raw rows."""

        if not isinstance(query_key, str) or _SAFE_KEY.fullmatch(query_key) is None:
            raise DataSourceBoundaryError("INVALID_QUERY_KEY")
        if not isinstance(expected_request_hash, str) or re.fullmatch(r"[0-9a-f]{64}", expected_request_hash) is None:
            raise DataSourceBoundaryError("INVALID_QUERY_REQUEST_HASH")
        path = self.private_root / "queries" / query_key / "success.canonical.json"
        if not path.is_file():
            raise DataSourceBoundaryError("QUERY_RESULT_RECEIPT_MISSING")
        attempts = self.verify_attempt_chain(
            query_key=query_key,
            expected_request_hash=expected_request_hash,
            expected_query_hash=expected_query_hash,
            expected_input_hash=expected_input_hash,
            expected_schedule_hash=expected_schedule_hash,
        )
        if not attempts or attempts[-1].status != "success":
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
        return self._read_success(path, expected_request_hash, query_key, expected_attempt=attempts[-1])

    def _read_success(
        self,
        path: Path,
        request_hash: str,
        query_key: str,
        *,
        expected_attempt: QueryAttemptEvidence,
    ) -> QueryResult:
        try:
            receipt_bytes = path.read_bytes()
            value = json.loads(receipt_bytes, object_pairs_hook=_strict_object)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION") from error
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "attempt_hash",
                "attempt_ordinal",
                "clock_end_tick",
                "clock_policy_hash",
                "clock_start_tick",
                "elapsed_ticks",
                "input_hash",
                "query_hash",
                "query_key",
                "raw_byte_count",
                "raw_result_hash",
                "raw_row_count",
                "request_hash",
                "result_hash",
                "result_size",
                "schedule_hash",
                "schema_version",
                "status",
            }
            or canonical_json_bytes(value) != receipt_bytes
            or value.get("request_hash") != request_hash
            or value.get("query_key") != query_key
            or value.get("status") != "success"
            or value.get("schema_version") != "fixture-query-success/1.0.0"
            or expected_attempt.status != "success"
            or expected_attempt.failure_code is not None
            or expected_attempt.late_result_hash is not None
        ):
            raise DataSourceBoundaryError("QUERY_KEY_CONFLICT")
        result_hash = value.get("result_hash")
        result_size = value.get("result_size")
        attempt_ordinal = value.get("attempt_ordinal")
        numeric_fields = (
            attempt_ordinal,
            value.get("elapsed_ticks"),
            value.get("raw_byte_count"),
            value.get("raw_row_count"),
            result_size,
            value.get("clock_start_tick"),
            value.get("clock_end_tick"),
        )
        if (
            not isinstance(result_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", result_hash) is None
            or any(type(item) is not int or cast(int, item) < 0 for item in numeric_fields)
            or value.get("attempt_ordinal") != expected_attempt.attempt_ordinal
            or value.get("attempt_hash") != expected_attempt.attempt_hash
            or value.get("input_hash") != expected_attempt.input_hash
            or value.get("query_hash") != expected_attempt.query_hash
            or value.get("schedule_hash") != expected_attempt.schedule_hash
            or value.get("clock_policy_hash") != expected_attempt.clock_policy_hash
            or value.get("clock_start_tick") != expected_attempt.clock_start_tick
            or value.get("clock_end_tick") != expected_attempt.clock_end_tick
            or value.get("elapsed_ticks") != expected_attempt.elapsed_ticks
            or value.get("result_hash") != expected_attempt.result_hash
            or value.get("result_size") != expected_attempt.result_size
            or value.get("raw_result_hash") != expected_attempt.raw_result_hash
            or value.get("raw_byte_count") != expected_attempt.raw_result_size
            or value.get("raw_row_count") != expected_attempt.raw_row_count
        ):
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
        result_path = path.parent / "results" / f"{result_hash}.sanitized.json"
        try:
            sanitized_bytes = result_path.read_bytes()
            rows = json.loads(sanitized_bytes, object_pairs_hook=_strict_object)
            raw_bytes = (path.parent / "online-traces.private.json").read_bytes()
            raw_rows = json.loads(raw_bytes, object_pairs_hook=_strict_object)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise DataSourceBoundaryError("QUERY_RESULT_PAYLOAD_UNAVAILABLE") from error
        if (
            not isinstance(rows, list)
            or any(not isinstance(row, dict) for row in rows)
            or canonical_json_bytes(rows) != sanitized_bytes
            or len(sanitized_bytes) != result_size
            or sha256_hex(sanitized_bytes) != result_hash
            or len(rows) != value.get("raw_row_count")
            or not isinstance(raw_rows, list)
            or canonical_json_bytes(raw_rows) != raw_bytes
            or sha256_hex(raw_bytes) != value.get("raw_result_hash")
            or len(raw_bytes) != value.get("raw_byte_count")
            or len(raw_rows) != value.get("raw_row_count")
            or len(raw_rows) > self.exchange.skill.max_rows
            or len(raw_bytes) > self.exchange.skill.max_bytes
            or cast(int, value.get("clock_end_tick")) - cast(int, value.get("clock_start_tick"))
            != value.get("elapsed_ticks")
            or cast(int, value.get("elapsed_ticks")) > self.exchange.skill.max_elapsed_ticks
        ):
            raise DataSourceBoundaryError("QUERY_LEDGER_CORRUPTION")
        return QueryResult(
            status="success",
            failure_code=None,
            raw_row_count=cast(int, value["raw_row_count"]),
            raw_byte_count=cast(int, value["raw_byte_count"]),
            elapsed_ticks=cast(int, value["elapsed_ticks"]),
            request_hash=request_hash,
            query_hash=expected_attempt.query_hash,
            input_hash=expected_attempt.input_hash,
            schedule_hash=expected_attempt.schedule_hash,
            attempt_hash=expected_attempt.attempt_hash,
            result_hash=result_hash,
            result_size=cast(int, result_size),
            private_result_content_hash=sha256_hex(receipt_bytes),
            late_result_hash=None,
            raw_result_hash=expected_attempt.raw_result_hash,
            attempt_ordinal=cast(int, attempt_ordinal),
            _sanitized_bytes=sanitized_bytes,
        )

    def _generate_raw_rows(self) -> list[dict[str, object]]:
        """Typed equivalent of the approved formula; source text is never executed."""

        rows = [self._make_raw_row(index, f"report-{index:03d}", 30 + (index % 7)) for index in range(100)]
        rows.extend(
            self._make_raw_row(index, report_id, lag) for index, _trace_pk, report_id, lag in self.exchange._overrides
        )
        fault = self.config.source_fault
        if fault == "missing_sentinel":
            rows[0].pop("private_sentinel")
        elif fault == "wrong_sentinel":
            rows[0]["private_sentinel"] = "invalid-private-marker"
        elif fault == "leak_sentinel":
            rows[0]["prompt"] = cast(str, rows[0]["prompt"]) + self.exchange._sentinel
        elif fault == "row_limit":
            rows.append(self._make_raw_row(100, "report-103", 30))
        elif fault == "byte_limit":
            rows[0]["prompt"] = cast(str, rows[0]["prompt"]) + ("x" * 140_000)
        elif fault == "unique_99":
            reference = rows[98]
            for row in rows:
                if row["trace_pk"] == "trace-099":
                    row["trace_pk"] = "trace-098"
                    for field in ("event_time_utc", "model_id", "prompt", "purpose", "response", "tool_name"):
                        row[field] = reference[field]
        elif fault == "unique_101":
            for row in rows:
                if row["report_id"] == "report-100":
                    row["trace_pk"] = "trace-100"
                    break
        elif fault == "duplicate_semantic_mismatch":
            for row in rows:
                if row["report_id"] == "report-100":
                    row["prompt"] = cast(str, row["prompt"]) + "-conflict"
                    break
        elif fault == "duplicate_report_id":
            rows[1]["report_id"] = rows[0]["report_id"]
        elif fault == "missing_field":
            rows[0].pop("response")
        elif fault == "wrong_purpose":
            rows[0]["purpose"] = "judge_only"
        elif fault == "out_of_window":
            rows[0]["event_time_utc"] = "2025-12-31T23:59:59Z"
        return rows

    def _make_raw_row(self, index: int, report_id: str, lag_seconds: int) -> dict[str, object]:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        event = base + timedelta(seconds=(index + 1) * 600)
        ingestion = event + timedelta(seconds=lag_seconds)
        return {
            "event_time_utc": event.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ingestion_time_utc": ingestion.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_id": ("fixture-model-a", "fixture-model-b")[index % 2],
            "private_sentinel": self.exchange._sentinel,
            "prompt": f"seed-{self.exchange.seed}-prompt-{index:03d}-code-{self.exchange.seed + index * 17}",
            "purpose": "training_allowed",
            "report_id": report_id,
            "response": (f"seed-{self.exchange.seed}-response-{index:03d}-value-{self.exchange.seed * 3 + index * 29}"),
            "tool_name": ("python", "regex", "none", "javascript")[index % 4],
            "trace_pk": f"trace-{index:03d}",
        }

    def _validate_and_sanitize(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        expected_fields = set(_EXPECTED_COLUMNS)
        sanitized: list[dict[str, object]] = []
        for row in rows:
            if set(row) != expected_fields:
                raise DataSourceBoundaryError("RAW_FIELD_COMPLETENESS_FAILED")
            if row.get("private_sentinel") != self.exchange._sentinel:
                raise DataSourceBoundaryError("SENTINEL_VALIDATION_FAILED")
            clean = {key: value for key, value in row.items() if key != "private_sentinel"}
            try:
                assert_public_sentinel_free(clean)
            except DataContractError as error:
                raise DataSourceBoundaryError("SENTINEL_LEAK_DETECTED") from error
            if self.exchange._sentinel in canonical_json_bytes(clean).decode("utf-8"):
                raise DataSourceBoundaryError("SENTINEL_LEAK_DETECTED")
            sanitized.append(clean)
        return sanitized
