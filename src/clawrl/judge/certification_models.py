"""Immutable Ticket 05 contracts for Luna@8 certification."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from clawrl.artifacts import MAX_SAFE_INTEGER, JsonValue, canonical_json_bytes, sha256_hex

_HASH = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_DIRECTIVES = {"timeout", "delayed", "success", "permanent_failure"}
_OUTPUT_FAULTS = {None, "short_response", "duplicate_response", "fit_reuse", "historical_reuse", "wrong_count"}


class CertificationContractError(ValueError):
    """A closed-world certification contract is malformed."""


def _safe_id(value: object, name: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(cast(str, value)) is None:
        raise CertificationContractError(f"{name} is invalid")
    return cast(str, value)


def _hash(value: object, name: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise CertificationContractError(f"{name} must be a SHA-256 hex digest")
    return cast(str, value)


def _integer(value: object, name: str, *, minimum: int = 0, maximum: int = MAX_SAFE_INTEGER) -> int:
    if type(value) is not int or not minimum <= cast(int, value) <= maximum:
        raise CertificationContractError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return cast(int, value)


@dataclass(frozen=True, slots=True)
class BaseJudgePrompt:
    prompt_id: str
    text: str
    schema_version: Literal["base-judge-prompt/1.0.0"] = "base-judge-prompt/1.0.0"

    def __post_init__(self) -> None:
        _safe_id(self.prompt_id, "BaseJudgePrompt prompt_id")
        if (
            self.schema_version != "base-judge-prompt/1.0.0"
            or type(self.text) is not str
            or not 32 <= len(self.text) <= 16_384
        ):
            raise CertificationContractError("BaseJudgePrompt contract is invalid")

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {"prompt_id": self.prompt_id, "schema_version": self.schema_version, "text": self.text}


@dataclass(frozen=True, slots=True)
class TraceJudgePrompt:
    prompt_id: str
    trace_id: str
    candidate_index: int
    parent_prompt_hash: str
    text: str
    schema_version: Literal["trace-judge-prompt/1.0.0"] = "trace-judge-prompt/1.0.0"

    def __post_init__(self) -> None:
        _safe_id(self.prompt_id, "TraceJudgePrompt prompt_id")
        _safe_id(self.trace_id, "TraceJudgePrompt trace_id")
        _integer(self.candidate_index, "candidate_index", minimum=1, maximum=3)
        _hash(self.parent_prompt_hash, "parent prompt hash")
        if (
            self.schema_version != "trace-judge-prompt/1.0.0"
            or type(self.text) is not str
            or not 32 <= len(self.text) <= 16_384
        ):
            raise CertificationContractError("TraceJudgePrompt contract is invalid")

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "candidate_index": self.candidate_index,
            "parent_prompt_hash": self.parent_prompt_hash,
            "prompt_id": self.prompt_id,
            "schema_version": self.schema_version,
            "text": self.text,
            "trace_id": self.trace_id,
        }


@dataclass(frozen=True, slots=True)
class AlignmentPolicy:
    """All numeric promotion thresholds, frozen before holdout generation."""

    policy_id: str
    max_mean_absolute_error_micros: int
    max_p95_absolute_error_micros: int
    max_absolute_bias_micros: int
    min_pairwise_agreement_micros: int
    min_student_variance_micros: int
    max_failure_rate_micros: int
    required_item_count: int = 32
    max_prompt_attempts: int = 3
    schema_version: Literal["alignment-policy/1.0.0"] = "alignment-policy/1.0.0"

    def __post_init__(self) -> None:
        _safe_id(self.policy_id, "AlignmentPolicy policy_id")
        if self.schema_version != "alignment-policy/1.0.0":
            raise CertificationContractError("AlignmentPolicy schema is unsupported")
        for name in (
            "max_mean_absolute_error_micros",
            "max_p95_absolute_error_micros",
            "max_absolute_bias_micros",
            "min_pairwise_agreement_micros",
            "min_student_variance_micros",
            "max_failure_rate_micros",
        ):
            _integer(getattr(self, name), name, maximum=100_000_000)
        if self.required_item_count != 32 or self.max_prompt_attempts != 3:
            raise CertificationContractError("Luna@8 policy cardinality and attempt limits are frozen at 32 and 3")

    @classmethod
    def fixture_default(cls) -> AlignmentPolicy:
        return cls(
            policy_id="fixture-luna8-policy-v1",
            max_mean_absolute_error_micros=12_000_000,
            max_p95_absolute_error_micros=24_000_000,
            max_absolute_bias_micros=8_000_000,
            min_pairwise_agreement_micros=780_000,
            min_student_variance_micros=2_000_000,
            max_failure_rate_micros=125_000,
        )

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "max_absolute_bias_micros": self.max_absolute_bias_micros,
            "max_failure_rate_micros": self.max_failure_rate_micros,
            "max_mean_absolute_error_micros": self.max_mean_absolute_error_micros,
            "max_p95_absolute_error_micros": self.max_p95_absolute_error_micros,
            "max_prompt_attempts": self.max_prompt_attempts,
            "min_pairwise_agreement_micros": self.min_pairwise_agreement_micros,
            "min_student_variance_micros": self.min_student_variance_micros,
            "policy_id": self.policy_id,
            "required_item_count": self.required_item_count,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class RewardSchema:
    schema_id: str = "fixture-calibrated-scalar-reward-v1"
    minimum_micros: int = 0
    maximum_micros: int = 100_000_000
    aggregation: Literal["calibrated_scalar"] = "calibrated_scalar"
    schema_version: Literal["reward-schema/1.0.0"] = "reward-schema/1.0.0"

    def __post_init__(self) -> None:
        _safe_id(self.schema_id, "RewardSchema schema_id")
        if (
            self.schema_version != "reward-schema/1.0.0"
            or self.aggregation != "calibrated_scalar"
            or self.minimum_micros != 0
            or self.maximum_micros != 100_000_000
        ):
            raise CertificationContractError("RewardSchema must use the frozen calibrated_scalar range")

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "aggregation": self.aggregation,
            "maximum_micros": self.maximum_micros,
            "minimum_micros": self.minimum_micros,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class Scalarizer:
    scalarizer_id: str = "fixture-dimension-mean-v1"
    dimension_weights_micros: tuple[tuple[str, int], ...] = (
        ("correctness", 250_000),
        ("reasoning_quality", 250_000),
        ("task_completion", 250_000),
        ("tool_discipline", 250_000),
    )
    schema_version: Literal["scalarizer/1.0.0"] = "scalarizer/1.0.0"

    def __post_init__(self) -> None:
        _safe_id(self.scalarizer_id, "Scalarizer scalarizer_id")
        expected = ("correctness", "reasoning_quality", "task_completion", "tool_discipline")
        if (
            self.schema_version != "scalarizer/1.0.0"
            or tuple(name for name, _ in self.dimension_weights_micros) != expected
            or sum(weight for _, weight in self.dimension_weights_micros) != 1_000_000
            or any(type(weight) is not int or weight <= 0 for _, weight in self.dimension_weights_micros)
        ):
            raise CertificationContractError("Scalarizer weights are invalid")

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "dimension_weights_micros": {name: weight for name, weight in self.dimension_weights_micros},
            "scalarizer_id": self.scalarizer_id,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class LunaInferenceConfig:
    model_id: str = "fixture-luna-student-v1"
    max_output_tokens: int = 4096
    temperature_micros: int = 0
    top_p_micros: int = 1_000_000
    schema_version: Literal["luna-inference-config/1.0.0"] = "luna-inference-config/1.0.0"

    def __post_init__(self) -> None:
        _safe_id(self.model_id, "Luna model_id")
        if (
            self.schema_version != "luna-inference-config/1.0.0"
            or self.max_output_tokens != 4096
            or self.temperature_micros != 0
            or self.top_p_micros != 1_000_000
        ):
            raise CertificationContractError("Luna inference config is invalid")

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "max_output_tokens": self.max_output_tokens,
            "model_id": self.model_id,
            "schema_version": self.schema_version,
            "temperature_micros": self.temperature_micros,
            "top_p_micros": self.top_p_micros,
        }


@dataclass(frozen=True, slots=True)
class FixtureLuna8Config:
    run_id: str
    ticket04_run_id: str
    teacher_label_set_hash: str
    base_prompt: BaseJudgePrompt
    trace_prompt: TraceJudgePrompt
    policy: AlignmentPolicy
    luna_inference: LunaInferenceConfig
    reward_schema: RewardSchema
    scalarizer: Scalarizer
    algorithm_contract_hash: str
    trace_prompt_derivation_hash: str | None
    holdout_seed: int
    role_seed: int
    fault_schedule: tuple[str, ...] = ("success",)
    output_fault: str | None = None
    holdout_generator_profile_id: str = "fixture-holdout-generator-v1"
    holdout_generator_model_id: str = "fixture-semantic-scenarios-v2"
    schema_version: Literal["fixture-luna8-config/1.0.0", "fixture-luna8-config/1.1.0"] = "fixture-luna8-config/1.1.0"

    def __post_init__(self) -> None:
        _safe_id(self.run_id, "run_id")
        _safe_id(self.ticket04_run_id, "ticket04_run_id")
        _hash(self.teacher_label_set_hash, "TeacherLabelSet hash")
        _hash(self.algorithm_contract_hash, "algorithm contract hash")
        if self.schema_version == "fixture-luna8-config/1.1.0":
            _hash(self.trace_prompt_derivation_hash, "trace prompt derivation hash")
        elif self.schema_version != "fixture-luna8-config/1.0.0" or self.trace_prompt_derivation_hash is not None:
            raise CertificationContractError("fixture Luna@8 config schema evolution is invalid")
        _integer(self.holdout_seed, "holdout_seed")
        _integer(self.role_seed, "role_seed")
        if (
            type(self.base_prompt) is not BaseJudgePrompt
            or type(self.trace_prompt) is not TraceJudgePrompt
            or type(self.policy) is not AlignmentPolicy
            or type(self.luna_inference) is not LunaInferenceConfig
            or type(self.reward_schema) is not RewardSchema
            or type(self.scalarizer) is not Scalarizer
        ):
            raise CertificationContractError("fixture Luna@8 config dependencies are invalid")
        if (
            type(self.fault_schedule) is not tuple
            or not 1 <= len(self.fault_schedule) <= 8
            or any(type(value) is not str or value not in _DIRECTIVES for value in self.fault_schedule)
            or self.fault_schedule[-1] not in {"success", "permanent_failure"}
            or any(value in {"success", "permanent_failure"} for value in self.fault_schedule[:-1])
        ):
            raise CertificationContractError("holdout fault schedule is invalid")
        if self.output_fault not in _OUTPUT_FAULTS or (
            self.output_fault is not None and self.fault_schedule[-1] != "success"
        ):
            raise CertificationContractError("holdout output fault is invalid")
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
        payload: dict[str, JsonValue] = {
            "algorithm_contract_hash": self.algorithm_contract_hash,
            "base_prompt": self.base_prompt.artifact_payload(),
            "fault_schedule": list(self.fault_schedule),
            "holdout_generator_inference": self.holdout_inference_payload,
            "holdout_generator_model_id": self.holdout_generator_model_id,
            "holdout_generator_profile_id": self.holdout_generator_profile_id,
            "holdout_seed": self.holdout_seed,
            "luna_inference": self.luna_inference.artifact_payload(),
            "output_fault": self.output_fault,
            "policy": self.policy.artifact_payload(),
            "reward_schema": self.reward_schema.artifact_payload(),
            "role_seed": self.role_seed,
            "run_id": self.run_id,
            "scalarizer": self.scalarizer.artifact_payload(),
            "schema_version": self.schema_version,
            "teacher_label_set_hash": self.teacher_label_set_hash,
            "ticket04_run_id": self.ticket04_run_id,
            "trace_prompt": self.trace_prompt.artifact_payload(),
        }
        if self.schema_version == "fixture-luna8-config/1.1.0":
            payload["trace_prompt_derivation_hash"] = cast(str, self.trace_prompt_derivation_hash)
        return payload

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> FixtureLuna8Config:
        try:
            base = cast(dict[str, object], value["base_prompt"])
            trace = cast(dict[str, object], value["trace_prompt"])
            policy = cast(dict[str, object], value["policy"])
            luna = cast(dict[str, object], value["luna_inference"])
            reward = cast(dict[str, object], value["reward_schema"])
            scalar = cast(dict[str, object], value["scalarizer"])
            weights = cast(dict[str, object], scalar["dimension_weights_micros"])
            item = cls(
                run_id=cast(str, value["run_id"]),
                ticket04_run_id=cast(str, value["ticket04_run_id"]),
                teacher_label_set_hash=cast(str, value["teacher_label_set_hash"]),
                base_prompt=BaseJudgePrompt(
                    prompt_id=cast(str, base["prompt_id"]),
                    text=cast(str, base["text"]),
                    schema_version=cast(Literal["base-judge-prompt/1.0.0"], base["schema_version"]),
                ),
                trace_prompt=TraceJudgePrompt(
                    prompt_id=cast(str, trace["prompt_id"]),
                    trace_id=cast(str, trace["trace_id"]),
                    candidate_index=cast(int, trace["candidate_index"]),
                    parent_prompt_hash=cast(str, trace["parent_prompt_hash"]),
                    text=cast(str, trace["text"]),
                    schema_version=cast(Literal["trace-judge-prompt/1.0.0"], trace["schema_version"]),
                ),
                policy=AlignmentPolicy(
                    policy_id=cast(str, policy["policy_id"]),
                    max_mean_absolute_error_micros=cast(int, policy["max_mean_absolute_error_micros"]),
                    max_p95_absolute_error_micros=cast(int, policy["max_p95_absolute_error_micros"]),
                    max_absolute_bias_micros=cast(int, policy["max_absolute_bias_micros"]),
                    min_pairwise_agreement_micros=cast(int, policy["min_pairwise_agreement_micros"]),
                    min_student_variance_micros=cast(int, policy["min_student_variance_micros"]),
                    max_failure_rate_micros=cast(int, policy["max_failure_rate_micros"]),
                    required_item_count=cast(int, policy["required_item_count"]),
                    max_prompt_attempts=cast(int, policy["max_prompt_attempts"]),
                    schema_version=cast(Literal["alignment-policy/1.0.0"], policy["schema_version"]),
                ),
                luna_inference=LunaInferenceConfig(
                    model_id=cast(str, luna["model_id"]),
                    max_output_tokens=cast(int, luna["max_output_tokens"]),
                    temperature_micros=cast(int, luna["temperature_micros"]),
                    top_p_micros=cast(int, luna["top_p_micros"]),
                    schema_version=cast(Literal["luna-inference-config/1.0.0"], luna["schema_version"]),
                ),
                reward_schema=RewardSchema(
                    schema_id=cast(str, reward["schema_id"]),
                    minimum_micros=cast(int, reward["minimum_micros"]),
                    maximum_micros=cast(int, reward["maximum_micros"]),
                    aggregation=cast(Literal["calibrated_scalar"], reward["aggregation"]),
                    schema_version=cast(Literal["reward-schema/1.0.0"], reward["schema_version"]),
                ),
                scalarizer=Scalarizer(
                    scalarizer_id=cast(str, scalar["scalarizer_id"]),
                    dimension_weights_micros=tuple(
                        (cast(str, key), cast(int, item_value)) for key, item_value in weights.items()
                    ),
                    schema_version=cast(Literal["scalarizer/1.0.0"], scalar["schema_version"]),
                ),
                algorithm_contract_hash=cast(str, value["algorithm_contract_hash"]),
                trace_prompt_derivation_hash=cast(str | None, value.get("trace_prompt_derivation_hash")),
                holdout_seed=cast(int, value["holdout_seed"]),
                role_seed=cast(int, value["role_seed"]),
                fault_schedule=tuple(cast(list[str], value["fault_schedule"])),
                output_fault=cast(str | None, value["output_fault"]),
                holdout_generator_profile_id=cast(str, value["holdout_generator_profile_id"]),
                holdout_generator_model_id=cast(str, value["holdout_generator_model_id"]),
                schema_version=cast(
                    Literal["fixture-luna8-config/1.0.0", "fixture-luna8-config/1.1.0"],
                    value["schema_version"],
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CertificationContractError("fixture Luna@8 config mapping is invalid") from error
        if value != item.immutable_input_payload:
            raise CertificationContractError("fixture Luna@8 config contains unknown or altered fields")
        return item


@dataclass(frozen=True, slots=True)
class ProductionLuna8Config:
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


def derive_fit_calibrated_trace_prompt(candidate_prompt: str, aggregate_diagnostics: Mapping[str, object]) -> str:
    """Deterministically distill fit-only optimizer diagnostics into a candidate prompt."""

    if type(candidate_prompt) is not str or not 32 <= len(candidate_prompt) <= 12_000:
        raise CertificationContractError("fit optimizer candidate prompt is invalid")
    means = aggregate_diagnostics.get("mean_dimension_scores")
    scalar = aggregate_diagnostics.get("scalar_micros")
    if (
        aggregate_diagnostics.get("fit_item_count") != 32
        or not isinstance(means, dict)
        or set(means) != {"correctness", "reasoning_quality", "task_completion", "tool_discipline"}
        or not isinstance(scalar, dict)
        or type(scalar.get("mean")) is not str
    ):
        raise CertificationContractError("fit optimizer aggregate diagnostics are invalid")
    values: dict[str, str] = {}
    for name in ("correctness", "reasoning_quality", "task_completion", "tool_discipline"):
        value = means.get(name)
        if type(value) is not str or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", cast(str, value)) is None:
            raise CertificationContractError("fit optimizer dimension mean is invalid")
        values[name] = cast(str, value)
    scalar_mean = cast(str, scalar["mean"])
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", scalar_mean) is None:
        raise CertificationContractError("fit optimizer scalar mean is invalid")
    result = (
        f"{candidate_prompt} Fit-derived numeric calibration, frozen from 32 fit labels before any holdout: "
        f"correctness mean {values['correctness']} points, reasoning_quality mean "
        f"{values['reasoning_quality']} points, task_completion mean {values['task_completion']} points, "
        f"tool_discipline mean {values['tool_discipline']} points, and calibrated scalar mean {scalar_mean} micros. "
        "Use these only as anchors for semantically similar visible responses; adjust independently for concrete "
        "observable differences, preserve item-specific evidence, and never infer teacher labels or holdout history."
    )
    if len(result) > 16_384:
        raise CertificationContractError("fit-derived TraceJudgePrompt exceeds its contract")
    return result


def payload_hash(payload: dict[str, JsonValue]) -> str:
    return sha256_hex(canonical_json_bytes(payload))
