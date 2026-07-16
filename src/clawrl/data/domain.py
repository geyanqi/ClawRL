"""Pure deterministic identity functions shared by ingestion and consumers."""

from __future__ import annotations

from collections.abc import Mapping

from clawrl.artifacts import JsonValue, canonical_json_bytes, sha256_hex
from clawrl.data.models import DatasetValidationError

TRAINING_TRACE_FIELDS = {
    "event_time_utc",
    "ingestion_time_utc",
    "model_id",
    "prompt",
    "purpose",
    "response",
    "sanitizer_version",
    "source_candidate_hash",
    "source_id",
    "source_trace_key",
    "tool_name",
    "trace_id",
}
CANDIDATE_FIELDS = {
    "event_time_utc",
    "ingestion_time_utc",
    "model_id",
    "prompt",
    "purpose",
    "query_plan_hash",
    "report_id",
    "response",
    "sanitizer_version",
    "source_id",
    "source_trace_key",
    "tool_name",
}
FORBIDDEN_DATA_KEYS = {
    "eval_dataset",
    "holdout",
    "holdout_ref",
    "judge",
    "judge_bundle",
    "judge_pack",
    "label",
    "labels",
    "teacher_label",
}


def selection_score(candidate: Mapping[str, object], policy_version: str) -> str:
    if policy_version not in {"badcase-sha256-v1", "badcase-sha256-v2"}:
        raise DatasetValidationError("unsupported badcase selection policy")
    required = (
        "event_time_utc",
        "ingestion_time_utc",
        "model_id",
        "prompt",
        "response",
        "source_trace_key",
        "tool_name",
    )
    values: dict[str, object] = {}
    for key in required:
        value = candidate.get(key)
        if not isinstance(value, str) or not value:
            raise DatasetValidationError("candidate cannot be scored without complete semantic content")
        values[key] = value
    return sha256_hex(
        canonical_json_bytes(
            {
                "candidate": values,
                "domain": f"governed-badcase-selection/{policy_version}",
            }
        )
    )


def training_trace_payload(candidate_hash: str, candidate: Mapping[str, object]) -> dict[str, JsonValue]:
    if set(candidate) != CANDIDATE_FIELDS:
        raise DatasetValidationError("SanitizedCandidate fields are invalid")
    semantic: dict[str, JsonValue] = {}
    for key in (
        "event_time_utc",
        "ingestion_time_utc",
        "model_id",
        "prompt",
        "purpose",
        "response",
        "sanitizer_version",
        "source_id",
        "source_trace_key",
        "tool_name",
    ):
        value = candidate.get(key)
        if not isinstance(value, str) or not value:
            raise DatasetValidationError("SanitizedCandidate semantic fields are invalid")
        semantic[key] = value
    identity = sha256_hex(
        canonical_json_bytes(
            {
                "domain": "training-trace-identity/1.0.0",
                "semantic": semantic,
            }
        )
    )
    return {
        **semantic,
        "source_candidate_hash": candidate_hash,
        "trace_id": f"tt-{identity[:40]}",
    }


def dataset_version_id(payload_without_id: Mapping[str, object]) -> str:
    return (
        "dv-"
        + sha256_hex(
            canonical_json_bytes(
                {
                    "dataset": dict(payload_without_id),
                    "domain": "dataset-version-identity/1.0.0",
                }
            )
        )[:40]
    )


def trace_set_hash(trace_refs: list[dict[str, str]]) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "domain": "training-trace-set/1.0.0",
                "trace_refs": trace_refs,
            }
        )
    )


def assert_data_only(value: object) -> None:
    pending = [value]
    nodes = 0
    while pending:
        item = pending.pop()
        nodes += 1
        if nodes > 100_000:
            raise DatasetValidationError("data artifact graph exceeds validation limits")
        if isinstance(item, dict):
            if FORBIDDEN_DATA_KEYS.intersection(item):
                raise DatasetValidationError("data-only artifact contains a prohibited Judge/eval field")
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
