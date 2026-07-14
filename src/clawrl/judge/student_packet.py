"""Versioned, leak-resistant Student Judge input packet boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping

from clawrl.artifacts import (
    MAX_SAFE_INTEGER,
    CanonicalizationError,
    JsonValue,
    canonical_json_bytes,
    sha256_hex,
)

STUDENT_PACKET_FIELDS = (
    "schema_version",
    "role",
    "seed",
    "packet_id",
    "run_id",
    "global_step",
    "trace_id",
    "trajectory",
    "judge_pack",
    "rubric",
    "output_schema",
    "allowlisted_fields",
)
TRAJECTORY_FIELDS = {"trajectory_id", "prompt", "response"}
JUDGE_PACK_FIELDS = {
    "judge_pack_id",
    "items_per_turn",
    "reward_schema_version",
    "scalarizer_version",
    "dimensions",
    "scalarizer",
}
OUTPUT_SCHEMA_FIELDS = {"schema_version", "required"}
OUTPUT_FIELDS = (
    "packet_id",
    "seed",
    "dimensions",
    "overall_scalar",
    "confidence",
    "failure_tags",
    "evidence",
    "turn_local_tie_groups",
)
_PROHIBITED_KEY_TERMS = ("teacher", "holdout", "generator", "credential", "production")
_PROHIBITED_ROLE_VALUE = re.compile(
    r"(?:^|[^a-z0-9])(?:teacher[\s._-]*(?:only|label|secret)|holdout[\s._-]*only)(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


class StudentPacketValidationError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.code = "ROLE_PACKET_NOT_ALLOWLISTED"
        self.detail = detail


def validate_student_packet_structure(packet: Mapping[str, object]) -> None:
    """Reject any field not in the frozen Student-visible v1 schema."""

    value = dict(packet)
    allowlisted = value.get("allowlisted_fields")
    if allowlisted != list(STUDENT_PACKET_FIELDS) or set(value) != set(STUDENT_PACKET_FIELDS):
        raise StudentPacketValidationError("student packet does not match the versioned adapter allowlist")
    if _contains_prohibited_content(value):
        raise StudentPacketValidationError("student packet exposes a prohibited role field")
    trajectory = value.get("trajectory")
    judge_pack = value.get("judge_pack")
    output_schema = value.get("output_schema")
    rubric = value.get("rubric")
    if not isinstance(trajectory, dict) or not isinstance(judge_pack, dict):
        raise StudentPacketValidationError("student packet trajectory or JudgePack is invalid")
    validate_judge_pack_structure(judge_pack)
    dimensions = judge_pack.get("dimensions")
    seed = value.get("seed")
    global_step = value.get("global_step")
    if (
        value.get("schema_version") != "student-judge-input/1.0.0"
        or value.get("role") != "student_judge"
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= MAX_SAFE_INTEGER
        or not isinstance(global_step, int)
        or isinstance(global_step, bool)
        or not 0 <= global_step <= MAX_SAFE_INTEGER
        or not _is_safe_id(value.get("packet_id"))
        or not _is_safe_id(value.get("run_id"))
        or not _is_safe_id(value.get("trace_id"))
        or set(trajectory) != TRAJECTORY_FIELDS
        or not _is_safe_id(trajectory.get("trajectory_id"))
        or not isinstance(trajectory.get("prompt"), str)
        or not trajectory.get("prompt")
        or not isinstance(trajectory.get("response"), str)
        or not trajectory.get("response")
        or set(judge_pack) != JUDGE_PACK_FIELDS
        or not isinstance(output_schema, dict)
        or set(output_schema) != OUTPUT_SCHEMA_FIELDS
        or output_schema.get("schema_version") != "student-judge-output/1.0.0"
        or output_schema.get("required") != list(OUTPUT_FIELDS)
        or not isinstance(dimensions, dict)
        or not isinstance(rubric, dict)
        or set(rubric) != set(dimensions)
        or not all(isinstance(item, str) and item for item in rubric.values())
    ):
        raise StudentPacketValidationError("student packet nested fields do not match the versioned schema")


def validate_judge_pack_structure(judge_pack: Mapping[str, object]) -> None:
    """Validate the closed-world Student-visible JudgePack v1 contract."""

    value = dict(judge_pack)
    if set(value) != JUDGE_PACK_FIELDS or _contains_prohibited_content(value):
        raise StudentPacketValidationError("student packet exposes a prohibited role field")
    items_per_turn = value.get("items_per_turn")
    dimensions = value.get("dimensions")
    scalarizer = value.get("scalarizer")
    if (
        not _is_safe_id(value.get("judge_pack_id"))
        or not isinstance(items_per_turn, int)
        or isinstance(items_per_turn, bool)
        or not 0 < items_per_turn <= MAX_SAFE_INTEGER
        or value.get("reward_schema_version") != "reward-schema/1.0.0"
        or value.get("scalarizer_version") != "scalarizer/1.0.0"
        or not isinstance(dimensions, dict)
        or not dimensions
        or not isinstance(scalarizer, dict)
        or set(scalarizer) != {"formula", "min", "max"}
    ):
        raise StudentPacketValidationError("student packet trajectory or JudgePack is invalid")
    for name, contract in dimensions.items():
        if not _is_safe_id(name) or not isinstance(contract, dict) or set(contract) != {"min", "max"}:
            raise StudentPacketValidationError("student packet nested fields do not match the versioned schema")
        minimum = contract.get("min")
        maximum = contract.get("max")
        if (
            not isinstance(minimum, int)
            or isinstance(minimum, bool)
            or not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or not -MAX_SAFE_INTEGER <= minimum <= maximum <= MAX_SAFE_INTEGER
        ):
            raise StudentPacketValidationError("student packet trajectory or JudgePack is invalid")
    formula = scalarizer.get("formula")
    scalar_min = scalarizer.get("min")
    scalar_max = scalarizer.get("max")
    formula_names = [name.strip() for name in formula.split("+")] if isinstance(formula, str) else []
    if (
        not formula_names
        or len(formula_names) != len(set(formula_names))
        or set(formula_names) != set(dimensions)
        or not isinstance(scalar_min, int)
        or isinstance(scalar_min, bool)
        or not isinstance(scalar_max, int)
        or isinstance(scalar_max, bool)
        or scalar_max <= scalar_min
        or not -MAX_SAFE_INTEGER <= scalar_min <= scalar_max <= MAX_SAFE_INTEGER
    ):
        raise StudentPacketValidationError("student packet trajectory or JudgePack is invalid")


def sanitized_rejection_payload(
    packet: Mapping[str, object], error: StudentPacketValidationError
) -> dict[str, JsonValue]:
    """Describe rejection without persisting any caller-controlled key or value."""

    try:
        input_bytes = canonical_json_bytes(dict(packet))
    except (CanonicalizationError, MemoryError, RecursionError):
        input_hash: str | None = None
        input_byte_size: int | None = None
    else:
        input_hash = sha256_hex(input_bytes)
        input_byte_size = len(input_bytes)
    return {
        "failure_code": error.code,
        "failure_detail": error.detail,
        "input_byte_size": input_byte_size,
        "input_hash": input_hash,
        "packet_schema": "student-judge-input/1.x-rejected",
        "status": "rejected",
    }


def _contains_prohibited_content(value: object) -> bool:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, nested in current.items():
                if isinstance(key, str) and any(term in key.lower() for term in _PROHIBITED_KEY_TERMS):
                    return True
                pending.append(nested)
        elif isinstance(current, list):
            pending.extend(current)
        elif isinstance(current, str) and _PROHIBITED_ROLE_VALUE.search(current) is not None:
            return True
    return False


def _is_safe_id(value: object) -> bool:
    return isinstance(value, str) and _SAFE_ID.fullmatch(value) is not None
