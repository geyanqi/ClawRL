"""Closed-world inputs for Ticket 06 TRY_4 and Sol fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast

from clawrl.artifacts import MAX_SAFE_INTEGER, JsonValue

_HASH = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_DIRECTIVES = {"timeout", "delayed", "success", "permanent_failure"}
SolEvidenceMode = Literal["calibrated_scalar", "group_relative_only", "invalid_calibrated_variance"]


class Luna4ContractError(ValueError):
    """A TRY_4 input is malformed or attempts to widen the contract."""


def _safe_id(value: object, name: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(cast(str, value)) is None:
        raise Luna4ContractError(f"{name} is invalid")
    return cast(str, value)


def _hash(value: object, name: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise Luna4ContractError(f"{name} must be a SHA-256 digest")
    return cast(str, value)


@dataclass(frozen=True, slots=True)
class FixtureLuna4Config:
    run_id: str
    try8_run_id: str
    try8_exhaustion_hash: str
    holdout_seed: int
    role_seed: int
    sol_evidence_mode: SolEvidenceMode = "calibrated_scalar"
    fault_schedule: tuple[str, ...] = ("success",)
    output_fault: str | None = None
    holdout_generator_profile_id: str = "fixture-holdout-generator-v1"
    holdout_generator_model_id: str = "fixture-semantic-scenarios-v2"
    schema_version: Literal["fixture-luna4-config/1.0.0"] = "fixture-luna4-config/1.0.0"

    def __post_init__(self) -> None:
        _safe_id(self.run_id, "run_id")
        _safe_id(self.try8_run_id, "try8_run_id")
        _hash(self.try8_exhaustion_hash, "TRY_8 exhaustion hash")
        if (
            self.schema_version != "fixture-luna4-config/1.0.0"
            or type(self.holdout_seed) is not int
            or not 0 <= self.holdout_seed <= MAX_SAFE_INTEGER
            or type(self.role_seed) is not int
            or not 0 <= self.role_seed <= MAX_SAFE_INTEGER
            or self.sol_evidence_mode not in {"calibrated_scalar", "group_relative_only", "invalid_calibrated_variance"}
            or type(self.fault_schedule) is not tuple
            or not 1 <= len(self.fault_schedule) <= 8
            or any(type(item) is not str or item not in _DIRECTIVES for item in self.fault_schedule)
            or self.fault_schedule[-1] not in {"success", "permanent_failure"}
            or any(item in {"success", "permanent_failure"} for item in self.fault_schedule[:-1])
            or self.output_fault
            not in {None, "short_response", "duplicate_response", "fit_reuse", "historical_reuse", "wrong_count"}
            or (self.output_fault is not None and self.fault_schedule[-1] != "success")
        ):
            raise Luna4ContractError("fixture Luna@4 config is invalid")
        _safe_id(self.holdout_generator_profile_id, "holdout generator profile")
        _safe_id(self.holdout_generator_model_id, "holdout generator model")

    @property
    def holdout_inference_payload(self) -> dict[str, JsonValue]:
        return {
            "max_output_characters": 4096,
            "min_output_characters": 32,
            "schema_version": "holdout-generator-inference-config/1.0.0",
            "temperature_micros": 800_000,
            "top_p_micros": 950_000,
        }

    @property
    def immutable_input_payload(self) -> dict[str, JsonValue]:
        return {
            "fault_schedule": list(self.fault_schedule),
            "holdout_generator_inference": self.holdout_inference_payload,
            "holdout_generator_model_id": self.holdout_generator_model_id,
            "holdout_generator_profile_id": self.holdout_generator_profile_id,
            "holdout_seed": self.holdout_seed,
            "output_fault": self.output_fault,
            "role_seed": self.role_seed,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "sol_evidence_mode": self.sol_evidence_mode,
            "try8_exhaustion_hash": self.try8_exhaustion_hash,
            "try8_run_id": self.try8_run_id,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> FixtureLuna4Config:
        try:
            item = cls(
                run_id=cast(str, value["run_id"]),
                try8_run_id=cast(str, value["try8_run_id"]),
                try8_exhaustion_hash=cast(str, value["try8_exhaustion_hash"]),
                holdout_seed=cast(int, value["holdout_seed"]),
                role_seed=cast(int, value["role_seed"]),
                sol_evidence_mode=cast(SolEvidenceMode, value["sol_evidence_mode"]),
                fault_schedule=tuple(cast(list[str], value["fault_schedule"])),
                output_fault=cast(str | None, value["output_fault"]),
                holdout_generator_profile_id=cast(str, value["holdout_generator_profile_id"]),
                holdout_generator_model_id=cast(str, value["holdout_generator_model_id"]),
                schema_version=cast(Literal["fixture-luna4-config/1.0.0"], value["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise Luna4ContractError("fixture Luna@4 config mapping is invalid") from error
        if value != item.immutable_input_payload:
            raise Luna4ContractError("fixture Luna@4 config has unknown or altered fields")
        return item


@dataclass(frozen=True, slots=True)
class ProductionLuna4Config:
    alignment_policy_approval_hash: str | None = None
    sol_model_approval_hash: str | None = None
    luna_model_approval_hash: str | None = None
    holdout_generator_approval_hash: str | None = None
    reward_schema_approval_hash: str | None = None
    scalarizer_approval_hash: str | None = None
    algorithm_contract_approval_hash: str | None = None
    fallback_policy_approval_hash: str | None = None
    human_escalation_sink_approval_hash: str | None = None
    base_prompt_approval_hash: str | None = None
    trace_prompt_approval_hash: str | None = None
    permanent_trace_governance_approval_hash: str | None = None


def classify_sol_evidence(scalars: list[int], mode: SolEvidenceMode) -> dict[str, JsonValue]:
    """Classify calibrated variance separately from relative-order-only evidence."""

    exact = (
        len(scalars) == 32
        and all(type(item) is int and 0 <= item <= 100_000_000 for item in scalars)
        and mode in {"calibrated_scalar", "group_relative_only", "invalid_calibrated_variance"}
    )
    mean = sum(scalars) // 32 if exact else 0
    variance = sum(abs(item - mean) for item in scalars) // 32 if exact else 0
    unique = len(set(scalars)) if exact else 0
    valid = mode == "calibrated_scalar" and variance > 0 and unique > 1
    relative = exact and unique > 1
    reason = (
        "SOL_CALIBRATED_SCALAR_VALID"
        if valid
        else "SOL_GROUP_RELATIVE_ONLY_UNSUPPORTED_V1"
        if mode == "group_relative_only"
        else "SOL_CALIBRATED_VARIANCE_INVALID"
    )
    return {
        "calibrated_label_count": 32 if exact and mode != "group_relative_only" else 0,
        "calibrated_scalar_valid": valid,
        "mean_scalar_micros": mean if exact else None,
        "reason_code": reason,
        "relative_order_available": relative,
        "reward_strategy": "calibrated_scalar",
        "scalar_variance_micros": variance if exact and mode != "invalid_calibrated_variance" else None,
        "source_mode": mode,
        "training_authorized": valid,
        "unique_scalar_count": unique,
    }
