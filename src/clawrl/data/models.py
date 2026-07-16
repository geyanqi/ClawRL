"""Typed contracts for governed data ingestion.

The types in this module deliberately contain no database client.  They are the
shared domain/application inputs used by both fixture and production profiles;
only the boundary factory differs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast

from clawrl.artifacts import MAX_SAFE_INTEGER, JsonValue, canonical_json_bytes, sha256_hex

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_PURPOSES = {"training_allowed", "judge_only", "eval_only"}
_FAULTS = {"timeout", "error", "delayed", "late", "success", "permanent_failure"}
_SOURCE_FAULTS = {
    "byte_limit",
    "duplicate_semantic_mismatch",
    "duplicate_report_id",
    "leak_sentinel",
    "missing_field",
    "missing_sentinel",
    "out_of_window",
    "row_limit",
    "time_limit",
    "unique_101",
    "unique_99",
    "wrong_purpose",
    "wrong_sentinel",
}


class DataContractError(ValueError):
    """A data contract is malformed or violates a fail-closed invariant."""


class DatasetValidationError(RuntimeError):
    """A persisted DatasetVersion DAG is malformed or unsafe to consume."""


@dataclass(frozen=True, slots=True)
class QueryWindow:
    start_utc: str
    end_utc: str

    def __post_init__(self) -> None:
        if not isinstance(self.start_utc, str) or not isinstance(self.end_utc, str):
            raise DataContractError("query window bounds must be UTC strings")
        if _UTC.fullmatch(self.start_utc) is None or _UTC.fullmatch(self.end_utc) is None:
            raise DataContractError("query window bounds must use second-precision UTC Z format")
        if self.start_utc >= self.end_utc:
            raise DataContractError("query window must be non-empty and ordered")

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {"end_utc": self.end_utc, "start_utc": self.start_utc}

    @property
    def content_hash(self) -> str:
        return sha256_hex(
            canonical_json_bytes(
                {
                    "domain": "approved-query-window/1.0.0",
                    "window": self.artifact_payload(),
                }
            )
        )


@dataclass(frozen=True, slots=True)
class ApprovedDataMix:
    """Versioned closed-world quotas for selected training content."""

    policy_version: str = "fixture-online-trace-mix-v1"
    selected_count: int = 100
    purpose: tuple[tuple[str, int], ...] = (("training_allowed", 100),)
    tool: tuple[tuple[str, int], ...] = (
        ("javascript", 25),
        ("none", 25),
        ("python", 25),
        ("regex", 25),
    )
    model: tuple[tuple[str, int], ...] = (
        ("fixture-model-a", 50),
        ("fixture-model-b", 50),
    )
    schema_version: Literal["approved-data-mix/1.0.0"] = "approved-data-mix/1.0.0"

    def __post_init__(self) -> None:
        if self.schema_version != "approved-data-mix/1.0.0":
            raise DataContractError("approved data mix schema is unsupported")
        if self.policy_version not in {"fixture-online-trace-mix-v1", "fixture-online-trace-mix-v2"}:
            raise DataContractError("approved data mix policy is unsupported")
        if type(self.selected_count) is not int or self.selected_count != 100:
            raise DataContractError("approved data mix selected_count must be exactly 100")
        expected = {
            "purpose": (("training_allowed", 100),),
            "tool": (("javascript", 25), ("none", 25), ("python", 25), ("regex", 25)),
            "model": (("fixture-model-a", 50), ("fixture-model-b", 50)),
        }
        for name, quotas in (("purpose", self.purpose), ("tool", self.tool), ("model", self.model)):
            if type(quotas) is not tuple or quotas != expected[name]:
                raise DataContractError(f"approved data mix {name} quotas are invalid")
            if any(type(key) is not str or type(count) is not int or count <= 0 for key, count in quotas):
                raise DataContractError(f"approved data mix {name} quota types are invalid")
            if sum(count for _key, count in quotas) != self.selected_count:
                raise DataContractError(f"approved data mix {name} quotas do not sum to selected_count")

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> ApprovedDataMix:
        if set(value) != {"model", "policy_version", "purpose", "schema_version", "selected_count", "tool"}:
            raise DataContractError("approved data mix fields are invalid")

        def quotas(name: str) -> tuple[tuple[str, int], ...]:
            raw = value.get(name)
            if not isinstance(raw, dict) or any(
                type(key) is not str or type(count) is not int for key, count in raw.items()
            ):
                raise DataContractError(f"approved data mix {name} quota mapping is invalid")
            return tuple(sorted(cast(dict[str, int], raw).items()))

        return cls(
            policy_version=cast(str, value.get("policy_version")),
            selected_count=cast(int, value.get("selected_count")),
            purpose=quotas("purpose"),
            tool=quotas("tool"),
            model=quotas("model"),
            schema_version=cast(Literal["approved-data-mix/1.0.0"], value.get("schema_version")),
        )

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "model": dict(self.model),
            "policy_version": self.policy_version,
            "purpose": dict(self.purpose),
            "schema_version": self.schema_version,
            "selected_count": self.selected_count,
            "tool": dict(self.tool),
        }

    @property
    def content_hash(self) -> str:
        return sha256_hex(canonical_json_bytes(self.artifact_payload()))


@dataclass(frozen=True, slots=True)
class DataSourceSkill:
    """Versioned, closed-world source governance declaration."""

    source_id: str
    table: str
    approved_window: QueryWindow
    approved_data_mix: ApprovedDataMix
    approved_columns: tuple[str, ...]
    purpose: Literal["training_allowed", "judge_only", "eval_only"]
    field_mapping: tuple[tuple[str, str], ...]
    event_time_column: str
    ingestion_time_column: str
    primary_key: str
    dedupe_key: str
    duplicate_semantics: str
    allowed_filters: tuple[str, ...]
    allowed_joins: tuple[str, ...]
    sanitizer_version: str
    sensitive_columns: tuple[str, ...]
    max_rows: int
    max_bytes: int
    max_elapsed_ticks: int
    expected_raw_rows: int
    expected_unique_rows: int
    selection_count: int
    selection_policy_version: str
    post_query_invariants: tuple[str, ...]
    schema_version: Literal["data-source-skill/1.0.0"] = "data-source-skill/1.0.0"

    def __post_init__(self) -> None:
        if self.schema_version != "data-source-skill/1.0.0":
            raise DataContractError("unsupported DataSourceSkill schema")
        if not isinstance(self.source_id, str) or _SAFE_ID.fullmatch(self.source_id) is None:
            raise DataContractError("source_id is invalid")
        if self.table != "fixture.online_trace":
            raise DataContractError("DataSourceSkill table is not approved")
        if not isinstance(self.approved_window, QueryWindow):
            raise DataContractError("DataSourceSkill approved_window is invalid")
        if not isinstance(self.approved_data_mix, ApprovedDataMix):
            raise DataContractError("DataSourceSkill approved_data_mix is invalid")
        if self.purpose not in _PURPOSES:
            raise DataContractError("DataSourceSkill purpose is invalid")
        for name, value in (
            ("approved_columns", self.approved_columns),
            ("allowed_filters", self.allowed_filters),
            ("sensitive_columns", self.sensitive_columns),
            ("post_query_invariants", self.post_query_invariants),
        ):
            if not isinstance(value, tuple) or not value or not all(isinstance(item, str) and item for item in value):
                raise DataContractError(f"{name} must be a non-empty tuple of strings")
        if not isinstance(self.allowed_joins, tuple) or not all(
            isinstance(item, str) and item for item in self.allowed_joins
        ):
            raise DataContractError("allowed_joins must be a tuple of strings")
        if len(set(self.approved_columns)) != len(self.approved_columns):
            raise DataContractError("approved columns must be unique")
        if not isinstance(self.field_mapping, tuple) or not all(
            isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], str) and isinstance(pair[1], str)
            for pair in self.field_mapping
        ):
            raise DataContractError("field_mapping must be ordered string pairs")
        mappings = dict(self.field_mapping)
        if set(mappings) != {
            "dedupe_key",
            "event_time",
            "ingestion_time",
            "model",
            "primary_key",
            "prompt",
            "response",
            "tool",
        }:
            raise DataContractError("field_mapping must declare every governed semantic field")
        referenced = {
            *mappings.values(),
            self.event_time_column,
            self.ingestion_time_column,
            self.primary_key,
            self.dedupe_key,
            *self.sensitive_columns,
        }
        if not referenced <= set(self.approved_columns):
            raise DataContractError("DataSourceSkill references an unapproved column")
        if mappings["event_time"] != self.event_time_column or mappings["ingestion_time"] != self.ingestion_time_column:
            raise DataContractError("time field mappings conflict with declared time columns")
        if mappings["primary_key"] != self.primary_key or mappings["dedupe_key"] != self.dedupe_key:
            raise DataContractError("identity field mappings conflict with declared keys")
        if not isinstance(self.duplicate_semantics, str) or not self.duplicate_semantics:
            raise DataContractError("duplicate semantics are required")
        if not isinstance(self.sanitizer_version, str) or _SAFE_ID.fullmatch(self.sanitizer_version) is None:
            raise DataContractError("sanitizer_version is invalid")
        if (
            not isinstance(self.selection_policy_version, str)
            or _SAFE_ID.fullmatch(self.selection_policy_version) is None
        ):
            raise DataContractError("selection_policy_version is invalid")
        for name, numeric_value in (
            ("max_rows", self.max_rows),
            ("max_bytes", self.max_bytes),
            ("max_elapsed_ticks", self.max_elapsed_ticks),
            ("expected_raw_rows", self.expected_raw_rows),
            ("expected_unique_rows", self.expected_unique_rows),
            ("selection_count", self.selection_count),
        ):
            if (
                not isinstance(numeric_value, int)
                or isinstance(numeric_value, bool)
                or not 0 < numeric_value <= MAX_SAFE_INTEGER
            ):
                raise DataContractError(f"{name} must be a positive safe integer")
        if self.expected_raw_rows > self.max_rows:
            raise DataContractError("expected rows exceed the approved row limit")
        if self.selection_count != 100 or self.expected_unique_rows != 100:
            raise DataContractError("Ticket 03 requires exactly 100 selected unique rows")
        if self.approved_data_mix.selected_count != self.selection_count:
            raise DataContractError("approved data mix conflicts with selection_count")

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> DataSourceSkill:
        expected = {
            "allowed_filters",
            "allowed_joins",
            "approved_columns",
            "approved_data_mix",
            "approved_data_mix_hash",
            "approved_window",
            "dedupe_key",
            "duplicate_semantics",
            "event_time_column",
            "expected_raw_rows",
            "expected_unique_rows",
            "field_mapping",
            "ingestion_time_column",
            "limits",
            "post_query_invariants",
            "primary_key",
            "purpose",
            "sanitizer_version",
            "schema_version",
            "selection_count",
            "selection_policy_version",
            "sensitive_columns",
            "source_id",
            "table",
            "window_hash",
        }
        if set(value) != expected:
            raise DataContractError("DataSourceSkill artifact fields are invalid")

        def strings(name: str) -> tuple[str, ...]:
            raw = value.get(name)
            if not isinstance(raw, list) or any(type(item) is not str for item in raw):
                raise DataContractError(f"DataSourceSkill {name} is invalid")
            return tuple(cast(list[str], raw))

        raw_window = value.get("approved_window")
        raw_mix = value.get("approved_data_mix")
        raw_mapping = value.get("field_mapping")
        raw_limits = value.get("limits")
        if (
            not isinstance(raw_window, dict)
            or set(raw_window) != {"end_utc", "start_utc"}
            or not isinstance(raw_mix, dict)
            or not isinstance(raw_mapping, dict)
            or any(type(key) is not str or type(item) is not str for key, item in raw_mapping.items())
            or not isinstance(raw_limits, dict)
            or set(raw_limits) != {"max_bytes", "max_elapsed_ticks", "max_rows"}
        ):
            raise DataContractError("DataSourceSkill nested fields are invalid")
        mix = ApprovedDataMix.from_mapping(cast(dict[str, object], raw_mix))
        window = QueryWindow(
            cast(str, raw_window.get("start_utc")),
            cast(str, raw_window.get("end_utc")),
        )
        if value.get("approved_data_mix_hash") != mix.content_hash or value.get("window_hash") != window.content_hash:
            raise DataContractError("DataSourceSkill policy hashes are invalid")
        return cls(
            source_id=cast(str, value.get("source_id")),
            table=cast(str, value.get("table")),
            approved_window=window,
            approved_data_mix=mix,
            approved_columns=strings("approved_columns"),
            purpose=cast(Literal["training_allowed", "judge_only", "eval_only"], value.get("purpose")),
            field_mapping=tuple(sorted(cast(dict[str, str], raw_mapping).items())),
            event_time_column=cast(str, value.get("event_time_column")),
            ingestion_time_column=cast(str, value.get("ingestion_time_column")),
            primary_key=cast(str, value.get("primary_key")),
            dedupe_key=cast(str, value.get("dedupe_key")),
            duplicate_semantics=cast(str, value.get("duplicate_semantics")),
            allowed_filters=strings("allowed_filters"),
            allowed_joins=strings("allowed_joins"),
            sanitizer_version=cast(str, value.get("sanitizer_version")),
            sensitive_columns=strings("sensitive_columns"),
            max_rows=cast(int, raw_limits.get("max_rows")),
            max_bytes=cast(int, raw_limits.get("max_bytes")),
            max_elapsed_ticks=cast(int, raw_limits.get("max_elapsed_ticks")),
            expected_raw_rows=cast(int, value.get("expected_raw_rows")),
            expected_unique_rows=cast(int, value.get("expected_unique_rows")),
            selection_count=cast(int, value.get("selection_count")),
            selection_policy_version=cast(str, value.get("selection_policy_version")),
            post_query_invariants=strings("post_query_invariants"),
            schema_version=cast(Literal["data-source-skill/1.0.0"], value.get("schema_version")),
        )

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "approved_data_mix": self.approved_data_mix.artifact_payload(),
            "approved_data_mix_hash": self.approved_data_mix.content_hash,
            "approved_window": self.approved_window.artifact_payload(),
            "allowed_filters": list(self.allowed_filters),
            "allowed_joins": list(self.allowed_joins),
            "approved_columns": list(self.approved_columns),
            "dedupe_key": self.dedupe_key,
            "duplicate_semantics": self.duplicate_semantics,
            "event_time_column": self.event_time_column,
            "expected_raw_rows": self.expected_raw_rows,
            "expected_unique_rows": self.expected_unique_rows,
            "field_mapping": {key: value for key, value in self.field_mapping},
            "ingestion_time_column": self.ingestion_time_column,
            "limits": {
                "max_bytes": self.max_bytes,
                "max_elapsed_ticks": self.max_elapsed_ticks,
                "max_rows": self.max_rows,
            },
            "post_query_invariants": list(self.post_query_invariants),
            "primary_key": self.primary_key,
            "purpose": self.purpose,
            "sanitizer_version": self.sanitizer_version,
            "schema_version": self.schema_version,
            "selection_count": self.selection_count,
            "selection_policy_version": self.selection_policy_version,
            "sensitive_columns": list(self.sensitive_columns),
            "source_id": self.source_id,
            "table": self.table,
            "window_hash": self.approved_window.content_hash,
        }


@dataclass(frozen=True, slots=True)
class FixtureDataIngestConfig:
    """Fixture boundary controls; structurally incapable of carrying credentials."""

    run_id: str
    query_sql: str
    window: QueryWindow
    fault_schedule: tuple[str, ...] = ("success",)
    retry_limit: int = 4
    sanitizer_policy_version: str = "sentinel-drop-v1"
    selection_policy_version: str = "badcase-sha256-v1"
    data_mix_policy_version: str = "fixture-online-trace-mix-v1"
    source_fault: str | None = None
    execution_profile: Literal["fixture"] = "fixture"

    def __post_init__(self) -> None:
        if self.execution_profile != "fixture":
            raise DataContractError("fixture data ingest requires execution_profile='fixture'")
        if not isinstance(self.run_id, str) or _SAFE_ID.fullmatch(self.run_id) is None:
            raise DataContractError("run_id is invalid")
        if not isinstance(self.query_sql, str) or not self.query_sql:
            raise DataContractError("query_sql is required")
        if not isinstance(self.window, QueryWindow):
            raise DataContractError("window must be a QueryWindow")
        if not isinstance(self.fault_schedule, tuple) or not self.fault_schedule:
            raise DataContractError("fault_schedule must be non-empty")
        if len(self.fault_schedule) > 16 or any(item not in _FAULTS for item in self.fault_schedule):
            raise DataContractError("fault_schedule contains an unsupported directive")
        if self.fault_schedule[-1] not in {"success", "permanent_failure"}:
            raise DataContractError("fixture fault schedule must reach success or permanent failure")
        if any(item in {"success", "permanent_failure"} for item in self.fault_schedule[:-1]):
            raise DataContractError("fixture fault schedule cannot continue after a terminal directive")
        if (
            not isinstance(self.retry_limit, int)
            or isinstance(self.retry_limit, bool)
            or not 0 <= self.retry_limit <= 15
        ):
            raise DataContractError("retry_limit must be a safe integer from 0 through 15")
        if len(self.fault_schedule) - 1 > self.retry_limit:
            raise DataContractError("fault schedule requires more retries than the frozen policy permits")
        if self.sanitizer_policy_version not in {"sentinel-drop-v1", "sentinel-drop-v1.1"}:
            raise DataContractError("sanitizer policy version is not in the fixture registry")
        if self.selection_policy_version not in {"badcase-sha256-v1", "badcase-sha256-v2"}:
            raise DataContractError("selection policy version is not in the fixture registry")
        if self.data_mix_policy_version not in {"fixture-online-trace-mix-v1", "fixture-online-trace-mix-v2"}:
            raise DataContractError("data mix policy version is not in the fixture registry")
        if self.source_fault is not None and self.source_fault not in _SOURCE_FAULTS:
            raise DataContractError("source_fault is not in the deterministic fixture fault registry")

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> FixtureDataIngestConfig:
        allowed = {
            "data_mix_policy_version",
            "execution_profile",
            "fault_schedule",
            "query_sql",
            "retry_limit",
            "run_id",
            "sanitizer_policy_version",
            "selection_policy_version",
            "source_fault",
            "window",
        }
        extra = sorted(set(value) - allowed)
        if extra:
            raise DataContractError(f"fixture data config cannot carry production field {extra[0]!r}")
        raw_window = value.get("window")
        if not isinstance(raw_window, dict) or set(raw_window) != {"end_utc", "start_utc"}:
            raise DataContractError("window mapping is invalid")
        schedule = value.get("fault_schedule", ["success"])
        if not isinstance(schedule, list) or not all(isinstance(item, str) for item in schedule):
            raise DataContractError("fault_schedule must be a string list")
        return cls(
            run_id=cast(str, value.get("run_id")),
            query_sql=cast(str, value.get("query_sql")),
            window=QueryWindow(
                start_utc=cast(str, raw_window.get("start_utc")),
                end_utc=cast(str, raw_window.get("end_utc")),
            ),
            fault_schedule=tuple(schedule),
            retry_limit=cast(int, value.get("retry_limit", 4)),
            sanitizer_policy_version=cast(str, value.get("sanitizer_policy_version", "sentinel-drop-v1")),
            selection_policy_version=cast(str, value.get("selection_policy_version", "badcase-sha256-v1")),
            data_mix_policy_version=cast(
                str,
                value.get("data_mix_policy_version", "fixture-online-trace-mix-v1"),
            ),
            source_fault=cast(str | None, value.get("source_fault")),
            execution_profile=cast(Literal["fixture"], value.get("execution_profile", "fixture")),
        )

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "data_mix_policy_version": self.data_mix_policy_version,
            "execution_profile": self.execution_profile,
            "fault_schedule": list(self.fault_schedule),
            "query_sql": self.query_sql,
            "retry_limit": self.retry_limit,
            "run_id": self.run_id,
            "sanitizer_policy_version": self.sanitizer_policy_version,
            "selection_policy_version": self.selection_policy_version,
            "source_fault": self.source_fault,
            "window": self.window.artifact_payload(),
        }


@dataclass(frozen=True, slots=True)
class ProductionDataIngestConfig:
    """References required to consider constructing a real data adapter."""

    run_id: str
    schema_artifact_hash: str | None = None
    credential_ref: str | None = None
    governance_approval_hash: str | None = None
    approved_table: str | None = None
    approved_columns_hash: str | None = None
    sanitizer_approval_hash: str | None = None
    execution_profile: Literal["production"] = "production"

    def readiness_payload(self) -> dict[str, JsonValue]:
        checks: list[JsonValue] = []

        def check(field: str, value: object, *, content_hash: bool = False) -> None:
            if value is None:
                checks.append({"code": f"MISSING_{field}", "status": "blocked"})
            elif not isinstance(value, str) or not value or (content_hash and _HASH.fullmatch(value) is None):
                checks.append({"code": f"INVALID_{field}", "status": "blocked"})

        if not isinstance(self.execution_profile, str) or self.execution_profile != "production":
            checks.append({"code": "INVALID_EXECUTION_PROFILE", "status": "blocked"})
        if not isinstance(self.run_id, str) or _SAFE_ID.fullmatch(self.run_id) is None:
            checks.append({"code": "INVALID_RUN_ID", "status": "blocked"})
        check("DATA_SCHEMA", self.schema_artifact_hash, content_hash=True)
        credential = self.credential_ref
        if credential is None:
            checks.append({"code": "MISSING_DATA_CREDENTIAL_REF", "status": "blocked"})
        elif (
            not isinstance(credential, str)
            or not credential
            or len(credential) > 256
            or not credential.isascii()
            or any(character.isspace() for character in credential)
        ):
            checks.append({"code": "INVALID_DATA_CREDENTIAL_REF", "status": "blocked"})
        check("DATA_GOVERNANCE_APPROVAL", self.governance_approval_hash, content_hash=True)
        table = self.approved_table
        if table is None:
            checks.append({"code": "MISSING_APPROVED_TABLE", "status": "blocked"})
        elif (
            not isinstance(table, str)
            or not table.isascii()
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", table) is None
        ):
            checks.append({"code": "INVALID_APPROVED_TABLE", "status": "blocked"})
        check("APPROVED_COLUMNS", self.approved_columns_hash, content_hash=True)
        check("SANITIZER_APPROVAL", self.sanitizer_approval_hash, content_hash=True)
        checks.append({"code": "PRODUCTION_DATA_SOURCE_CONTRACT_UNAVAILABLE", "status": "blocked"})
        return {
            "checks": checks,
            "execution_profile": "production",
            "phase": "DATA_INGEST",
            "side_effects_permitted": False,
            "status": "blocked",
        }


DataIngestConfig = FixtureDataIngestConfig | ProductionDataIngestConfig
