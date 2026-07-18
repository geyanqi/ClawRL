"""Closed-world inputs for Ticket 07 Luna@16/@32 expansion."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from clawrl.artifacts import MAX_SAFE_INTEGER, JsonValue

_HASH = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_DIRECTIVES = {"timeout", "delayed", "success", "permanent_failure"}


class LunaExpansionContractError(ValueError):
    """An expansion input is malformed or widens a frozen boundary."""


def _safe_id(value: object, name: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(cast(str, value)) is None:
        raise LunaExpansionContractError(f"{name} is invalid")
    return cast(str, value)


def _hash(value: object, name: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise LunaExpansionContractError(f"{name} must be a SHA-256 digest")
    return cast(str, value)


@dataclass(frozen=True, slots=True)
class FixtureLunaExpansionConfig:
    run_id: str
    source_run_id: str
    source_certification_report_hash: str
    alignment_policy_hash: str
    holdout_seed: int
    role_seed: int
    fault_schedule: tuple[str, ...] = ("success",)
    output_fault: str | None = None
    holdout_generator_profile_id: str = "fixture-holdout-generator-v1"
    holdout_generator_model_id: str = "fixture-semantic-scenarios-v2"
    schema_version: Literal["fixture-luna-expansion-config/1.0.0"] = "fixture-luna-expansion-config/1.0.0"

    def __post_init__(self) -> None:
        _safe_id(self.run_id, "run_id")
        _safe_id(self.source_run_id, "source_run_id")
        _hash(self.source_certification_report_hash, "source certification report hash")
        _hash(self.alignment_policy_hash, "alignment policy hash")
        if (
            self.schema_version != "fixture-luna-expansion-config/1.0.0"
            or type(self.holdout_seed) is not int
            or not 0 <= self.holdout_seed <= MAX_SAFE_INTEGER - 20_006
            or type(self.role_seed) is not int
            or not 0 <= self.role_seed <= MAX_SAFE_INTEGER - 100
            or type(self.fault_schedule) is not tuple
            or not 1 <= len(self.fault_schedule) <= 8
            or any(type(item) is not str or item not in _DIRECTIVES for item in self.fault_schedule)
            or self.fault_schedule[-1] not in {"success", "permanent_failure"}
            or any(item in {"success", "permanent_failure"} for item in self.fault_schedule[:-1])
            or self.output_fault
            not in {None, "short_response", "duplicate_response", "fit_reuse", "historical_reuse", "wrong_count"}
            or (self.output_fault is not None and self.fault_schedule[-1] != "success")
        ):
            raise LunaExpansionContractError("fixture Luna expansion config is invalid")
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
            "alignment_policy_hash": self.alignment_policy_hash,
            "fault_schedule": list(self.fault_schedule),
            "holdout_generator_inference": self.holdout_inference_payload,
            "holdout_generator_model_id": self.holdout_generator_model_id,
            "holdout_generator_profile_id": self.holdout_generator_profile_id,
            "holdout_seed": self.holdout_seed,
            "output_fault": self.output_fault,
            "role_seed": self.role_seed,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "source_certification_report_hash": self.source_certification_report_hash,
            "source_run_id": self.source_run_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FixtureLunaExpansionConfig:
        try:
            item = cls(
                run_id=cast(str, value["run_id"]),
                source_run_id=cast(str, value["source_run_id"]),
                source_certification_report_hash=cast(str, value["source_certification_report_hash"]),
                alignment_policy_hash=cast(str, value["alignment_policy_hash"]),
                holdout_seed=cast(int, value["holdout_seed"]),
                role_seed=cast(int, value["role_seed"]),
                fault_schedule=tuple(cast(list[str], value["fault_schedule"])),
                output_fault=cast(str | None, value["output_fault"]),
                holdout_generator_profile_id=cast(str, value["holdout_generator_profile_id"]),
                holdout_generator_model_id=cast(str, value["holdout_generator_model_id"]),
                schema_version=cast(Literal["fixture-luna-expansion-config/1.0.0"], value["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LunaExpansionContractError("fixture Luna expansion config mapping is invalid") from error
        if dict(value) != item.immutable_input_payload:
            raise LunaExpansionContractError("fixture Luna expansion config has unknown or altered fields")
        return item


@dataclass(frozen=True, slots=True)
class ProductionLunaExpansionConfig:
    source_luna8_pack_approval_hash: str | None = None
    alignment_policy_approval_hash: str | None = None
    sol_model_approval_hash: str | None = None
    luna_model_approval_hash: str | None = None
    holdout_generator_approval_hash: str | None = None
    reward_schema_approval_hash: str | None = None
    scalarizer_approval_hash: str | None = None
    algorithm_contract_approval_hash: str | None = None
    base_prompt_approval_hash: str | None = None
    trace_prompt_approval_hash: str | None = None
    permanent_trace_governance_approval_hash: str | None = None
