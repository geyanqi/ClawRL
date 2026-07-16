"""Closed-world Ticket 04 contracts shared by fixture and production workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast

from clawrl.artifacts import MAX_SAFE_INTEGER, JsonValue, canonical_json_bytes, sha256_hex

_HASH = re.compile(r"^[0-9a-f]{64}$")
_TRACE_ID = re.compile(r"^tt-[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_FAULT_DIRECTIVES = {"timeout", "delayed", "success", "permanent_failure"}
_OUTPUT_FAULTS = {None, "short_response", "duplicate_response", "copied_online_response", "wrong_count", "out_of_order"}


class FitContractError(ValueError):
    """A Ticket 04 input or immutable domain object is malformed."""


class FitValidationError(RuntimeError):
    """A persisted Ticket 04 artifact graph cannot be recertified."""


def _hash(value: object, name: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise FitContractError(f"{name} must be a SHA-256 hex digest")
    return cast(str, value)


def _safe_id(value: object, name: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(cast(str, value)) is None:
        raise FitContractError(f"{name} is invalid")
    return cast(str, value)


def _trace_id(value: object) -> str:
    if type(value) is not str or _TRACE_ID.fullmatch(cast(str, value)) is None:
        raise FitContractError("trace_id is invalid")
    return cast(str, value)


def _safe_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= cast(int, value) <= MAX_SAFE_INTEGER:
        raise FitContractError(f"{name} must be a safe integer")
    return cast(int, value)


@dataclass(frozen=True, slots=True)
class InitialEvalRubric:
    """Frozen fixture rubric supplied to the isolated Sol TeacherScorer."""

    rubric_id: str
    dimensions: tuple[tuple[str, str], ...]
    scalar_contract: str
    schema_version: Literal["initial-eval-rubric/1.0.0"] = "initial-eval-rubric/1.0.0"

    def __post_init__(self) -> None:
        if self.schema_version != "initial-eval-rubric/1.0.0":
            raise FitContractError("Initial Eval Rubric schema is unsupported")
        _safe_id(self.rubric_id, "rubric_id")
        expected_names = ("correctness", "reasoning_quality", "task_completion", "tool_discipline")
        if (
            type(self.dimensions) is not tuple
            or tuple(name for name, _instruction in self.dimensions) != expected_names
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not str
                or type(pair[1]) is not str
                or not pair[1]
                for pair in self.dimensions
            )
        ):
            raise FitContractError("Initial Eval Rubric dimensions are invalid")
        if self.scalar_contract != "integer-dimensions-0-to-100-with-calibrated-mean-micros":
            raise FitContractError("Initial Eval Rubric scalar contract is unsupported")

    @classmethod
    def fixture_default(cls) -> InitialEvalRubric:
        return cls(
            rubric_id="fixture-initial-eval-rubric-v1",
            dimensions=(
                ("correctness", "Score whether the response reaches a correct result supported by its reasoning."),
                ("reasoning_quality", "Score whether the reasoning is coherent, specific, and internally consistent."),
                ("task_completion", "Score whether every explicit part of the prompt is completed."),
                ("tool_discipline", "Score whether any tool use is necessary, bounded, and accurately represented."),
            ),
            scalar_contract="integer-dimensions-0-to-100-with-calibrated-mean-micros",
        )

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> InitialEvalRubric:
        if set(value) != {"dimensions", "rubric_id", "scalar_contract", "schema_version"}:
            raise FitContractError("Initial Eval Rubric fields are invalid")
        dimensions = value.get("dimensions")
        if not isinstance(dimensions, dict):
            raise FitContractError("Initial Eval Rubric dimensions are invalid")
        return cls(
            rubric_id=cast(str, value.get("rubric_id")),
            dimensions=tuple((cast(str, key), cast(str, item)) for key, item in dimensions.items()),
            scalar_contract=cast(str, value.get("scalar_contract")),
            schema_version=cast(Literal["initial-eval-rubric/1.0.0"], value.get("schema_version")),
        )

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "dimensions": {name: instruction for name, instruction in self.dimensions},
            "rubric_id": self.rubric_id,
            "scalar_contract": self.scalar_contract,
            "schema_version": self.schema_version,
        }

    @property
    def content_hash(self) -> str:
        return sha256_hex(canonical_json_bytes(self.artifact_payload()))


@dataclass(frozen=True, slots=True)
class SolInferenceConfig:
    """Frozen Sol identity and inference behavior; never shared with other roles."""

    model_id: str
    decoding: tuple[tuple[str, int | str], ...]
    schema_version: Literal["sol-inference-config/1.0.0"] = "sol-inference-config/1.0.0"

    def __post_init__(self) -> None:
        if self.schema_version != "sol-inference-config/1.0.0":
            raise FitContractError("Sol inference schema is unsupported")
        _safe_id(self.model_id, "Sol model_id")
        expected = (
            ("max_output_tokens", 4096),
            ("temperature_micros", 0),
            ("top_p_micros", 1000000),
            ("version", "fixture-sol-decoding-v1"),
        )
        if self.decoding != expected:
            raise FitContractError("Sol inference decoding contract is invalid")

    @classmethod
    def fixture_default(cls) -> SolInferenceConfig:
        return cls(
            model_id="fixture-sol-teacher-v1",
            decoding=(
                ("max_output_tokens", 4096),
                ("temperature_micros", 0),
                ("top_p_micros", 1000000),
                ("version", "fixture-sol-decoding-v1"),
            ),
        )

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> SolInferenceConfig:
        if set(value) != {"decoding", "model_id", "schema_version"}:
            raise FitContractError("Sol inference fields are invalid")
        decoding = value.get("decoding")
        if not isinstance(decoding, dict):
            raise FitContractError("Sol inference decoding is invalid")
        pairs: list[tuple[str, int | str]] = []
        for key, item in decoding.items():
            if type(key) is not str or type(item) not in {int, str}:
                raise FitContractError("Sol inference decoding is invalid")
            pairs.append((key, cast(int | str, item)))
        return cls(
            model_id=cast(str, value.get("model_id")),
            decoding=tuple(pairs),
            schema_version=cast(Literal["sol-inference-config/1.0.0"], value.get("schema_version")),
        )

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "decoding": cast(dict[str, JsonValue], dict(self.decoding)),
            "model_id": self.model_id,
            "schema_version": self.schema_version,
        }

    @property
    def content_hash(self) -> str:
        return sha256_hex(canonical_json_bytes(self.artifact_payload()))


@dataclass(frozen=True, slots=True)
class GeneratorSlot:
    slot_id: str
    slot_index: int
    slot_seed: int
    origin: Literal["fresh_rollout", "online_response"]

    def __post_init__(self) -> None:
        _safe_id(self.slot_id, "generator slot_id")
        _safe_integer(self.slot_index, "generator slot_index")
        _safe_integer(self.slot_seed, "generator slot seed")
        if self.origin not in {"fresh_rollout", "online_response"}:
            raise FitContractError("generator slot origin is invalid")

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "origin": self.origin,
            "slot_id": self.slot_id,
            "slot_index": self.slot_index,
            "slot_seed": self.slot_seed,
        }


@dataclass(frozen=True, slots=True)
class GeneratorPlan:
    """Frozen, exact-32 fit rollout declaration."""

    dataset_version_hash: str
    training_trace_hash: str
    trace_id: str
    seed: int
    include_online_response: bool
    generator_profile_id: str
    generator_model_id: str
    inference_config_hash: str
    target_count: int
    slots: tuple[GeneratorSlot, ...]
    schema_version: Literal["generator-plan/1.0.0"] = "generator-plan/1.0.0"

    def __post_init__(self) -> None:
        if self.schema_version != "generator-plan/1.0.0":
            raise FitContractError("GeneratorPlan schema is unsupported")
        _hash(self.dataset_version_hash, "DatasetVersion hash")
        _hash(self.training_trace_hash, "TrainingTrace hash")
        _trace_id(self.trace_id)
        _safe_integer(self.seed, "GeneratorPlan seed")
        if type(self.include_online_response) is not bool:
            raise FitContractError("include_online_response must be a boolean")
        _safe_id(self.generator_profile_id, "generator_profile_id")
        _safe_id(self.generator_model_id, "generator_model_id")
        _hash(self.inference_config_hash, "generator inference config hash")
        if type(self.target_count) is not int or self.target_count != 32:
            raise FitContractError("GeneratorPlan target_count must be exactly 32")
        if type(self.slots) is not tuple or len(self.slots) != 32:
            raise FitContractError("GeneratorPlan must contain exactly 32 slots")
        if any(type(slot) is not GeneratorSlot for slot in self.slots):
            raise FitContractError("GeneratorPlan slot type is invalid")
        if [slot.slot_index for slot in self.slots] != list(range(32)):
            raise FitContractError("GeneratorPlan slots are out of order")
        if len({slot.slot_id for slot in self.slots}) != 32 or len({slot.slot_seed for slot in self.slots}) != 32:
            raise FitContractError("GeneratorPlan slots are not unique")
        online_positions = [slot.slot_index for slot in self.slots if slot.origin == "online_response"]
        expected_positions = [0] if self.include_online_response else []
        if online_positions != expected_positions:
            raise FitContractError("online response slot does not match the explicit plan declaration")

    @classmethod
    def create(
        cls,
        *,
        dataset_version_hash: str,
        training_trace_hash: str,
        trace_id: str,
        seed: int,
        include_online_response: bool,
        generator_profile_id: str,
        generator_model_id: str,
        inference_config_hash: str,
        target_count: int = 32,
    ) -> GeneratorPlan:
        _safe_integer(seed, "GeneratorPlan seed")
        if type(target_count) is not int or target_count != 32:
            raise FitContractError("GeneratorPlan target_count must be exactly 32")
        if type(include_online_response) is not bool:
            raise FitContractError("include_online_response must be a boolean")
        slots: list[GeneratorSlot] = []
        for index in range(32):
            material = sha256_hex(
                canonical_json_bytes(
                    {
                        "domain": "fit-generator-slot/1.0.0",
                        "index": index,
                        "seed": seed,
                        "trace_id": trace_id,
                    }
                )
            )
            slots.append(
                GeneratorSlot(
                    slot_id=f"gs-{material[:40]}",
                    slot_index=index,
                    slot_seed=int(material[:13], 16),
                    origin=("online_response" if include_online_response and index == 0 else "fresh_rollout"),
                )
            )
        return cls(
            dataset_version_hash=dataset_version_hash,
            training_trace_hash=training_trace_hash,
            trace_id=trace_id,
            seed=seed,
            include_online_response=include_online_response,
            generator_profile_id=generator_profile_id,
            generator_model_id=generator_model_id,
            inference_config_hash=inference_config_hash,
            target_count=target_count,
            slots=tuple(slots),
        )

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> GeneratorPlan:
        if (
            set(value)
            != {
                "dataset_version_hash",
                "generator_lineage",
                "include_online_response",
                "schema_version",
                "seed",
                "slots",
                "split",
                "target_count",
                "trace_id",
                "training_trace_hash",
            }
            or value.get("split") != "fit"
        ):
            raise FitContractError("GeneratorPlan fields are invalid")
        lineage = value.get("generator_lineage")
        raw_slots = value.get("slots")
        if (
            not isinstance(lineage, dict)
            or set(lineage)
            != {
                "generator_model_id",
                "generator_profile_id",
                "inference_config_hash",
            }
            or not isinstance(raw_slots, list)
        ):
            raise FitContractError("GeneratorPlan nested fields are invalid")
        slots: list[GeneratorSlot] = []
        for item in raw_slots:
            if not isinstance(item, dict) or set(item) != {"origin", "slot_id", "slot_index", "slot_seed"}:
                raise FitContractError("GeneratorPlan slot fields are invalid")
            slots.append(
                GeneratorSlot(
                    slot_id=cast(str, item.get("slot_id")),
                    slot_index=cast(int, item.get("slot_index")),
                    slot_seed=cast(int, item.get("slot_seed")),
                    origin=cast(Literal["fresh_rollout", "online_response"], item.get("origin")),
                )
            )
        return cls(
            dataset_version_hash=cast(str, value.get("dataset_version_hash")),
            training_trace_hash=cast(str, value.get("training_trace_hash")),
            trace_id=cast(str, value.get("trace_id")),
            seed=cast(int, value.get("seed")),
            include_online_response=cast(bool, value.get("include_online_response")),
            generator_profile_id=cast(str, lineage.get("generator_profile_id")),
            generator_model_id=cast(str, lineage.get("generator_model_id")),
            inference_config_hash=cast(str, lineage.get("inference_config_hash")),
            target_count=cast(int, value.get("target_count")),
            slots=tuple(slots),
            schema_version=cast(Literal["generator-plan/1.0.0"], value.get("schema_version")),
        )

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "dataset_version_hash": self.dataset_version_hash,
            "generator_lineage": {
                "generator_model_id": self.generator_model_id,
                "generator_profile_id": self.generator_profile_id,
                "inference_config_hash": self.inference_config_hash,
            },
            "include_online_response": self.include_online_response,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "slots": [slot.artifact_payload() for slot in self.slots],
            "split": "fit",
            "target_count": self.target_count,
            "trace_id": self.trace_id,
            "training_trace_hash": self.training_trace_hash,
        }

    @property
    def content_hash(self) -> str:
        return sha256_hex(canonical_json_bytes(self.artifact_payload()))


@dataclass(frozen=True, slots=True)
class FixtureFitConfig:
    run_id: str
    dataset_version_hash: str
    trace_id: str
    generator_seed: int
    role_seed: int
    rubric: InitialEvalRubric
    sol_inference: SolInferenceConfig
    include_online_response: bool = False
    fault_schedule: tuple[str, ...] = ("success",)
    output_fault: str | None = None
    generator_profile_id: str = "fixture-fit-generator-v1"
    generator_model_id: str = "fixture-content-derived-v1"
    schema_version: Literal["fixture-fit-config/1.0.0"] = "fixture-fit-config/1.0.0"

    def __post_init__(self) -> None:
        if self.schema_version != "fixture-fit-config/1.0.0":
            raise FitContractError("fixture fit config schema is unsupported")
        _safe_id(self.run_id, "run_id")
        _hash(self.dataset_version_hash, "DatasetVersion hash")
        _trace_id(self.trace_id)
        _safe_integer(self.generator_seed, "generator_seed")
        _safe_integer(self.role_seed, "role_seed")
        if type(self.rubric) is not InitialEvalRubric or type(self.sol_inference) is not SolInferenceConfig:
            raise FitContractError("frozen rubric and Sol inference config are required")
        if type(self.include_online_response) is not bool:
            raise FitContractError("include_online_response must be a boolean")
        if (
            type(self.fault_schedule) is not tuple
            or not 1 <= len(self.fault_schedule) <= 8
            or any(type(item) is not str or item not in _FAULT_DIRECTIVES for item in self.fault_schedule)
            or self.fault_schedule[-1] not in {"success", "permanent_failure"}
            or any(item in {"success", "permanent_failure"} for item in self.fault_schedule[:-1])
        ):
            raise FitContractError("generator fault schedule is invalid")
        if self.output_fault not in _OUTPUT_FAULTS:
            raise FitContractError("generator output fault is invalid")
        if self.output_fault is not None and self.fault_schedule[-1] != "success":
            raise FitContractError("output faults require a successful boundary attempt")
        _safe_id(self.generator_profile_id, "generator_profile_id")
        _safe_id(self.generator_model_id, "generator_model_id")

    @property
    def generator_inference_payload(self) -> dict[str, JsonValue]:
        return {
            "max_output_characters": 4096,
            "min_output_characters": 24,
            "schema_version": "generator-inference-config/1.0.0",
            "temperature_micros": 700000,
            "top_p_micros": 950000,
        }

    @property
    def generator_inference_hash(self) -> str:
        return sha256_hex(canonical_json_bytes(self.generator_inference_payload))

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "dataset_version_hash": self.dataset_version_hash,
            "fault_schedule": list(self.fault_schedule),
            "generator_inference": self.generator_inference_payload,
            "generator_model_id": self.generator_model_id,
            "generator_profile_id": self.generator_profile_id,
            "generator_seed": self.generator_seed,
            "include_online_response": self.include_online_response,
            "output_fault": self.output_fault,
            "role_seed": self.role_seed,
            "rubric": self.rubric.artifact_payload(),
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "sol_inference": self.sol_inference.artifact_payload(),
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> FixtureFitConfig:
        if set(value) != {
            "dataset_version_hash",
            "fault_schedule",
            "generator_inference",
            "generator_model_id",
            "generator_profile_id",
            "generator_seed",
            "include_online_response",
            "output_fault",
            "role_seed",
            "rubric",
            "run_id",
            "schema_version",
            "sol_inference",
            "trace_id",
        }:
            raise FitContractError("fixture fit config fields are invalid")
        rubric = value.get("rubric")
        sol = value.get("sol_inference")
        schedule = value.get("fault_schedule")
        if not isinstance(rubric, dict) or not isinstance(sol, dict) or not isinstance(schedule, list):
            raise FitContractError("fixture fit config nested fields are invalid")
        item = cls(
            run_id=cast(str, value.get("run_id")),
            dataset_version_hash=cast(str, value.get("dataset_version_hash")),
            trace_id=cast(str, value.get("trace_id")),
            generator_seed=cast(int, value.get("generator_seed")),
            role_seed=cast(int, value.get("role_seed")),
            rubric=InitialEvalRubric.from_mapping(cast(dict[str, object], rubric)),
            sol_inference=SolInferenceConfig.from_mapping(cast(dict[str, object], sol)),
            include_online_response=cast(bool, value.get("include_online_response")),
            fault_schedule=tuple(cast(list[str], schedule)),
            output_fault=cast(str | None, value.get("output_fault")),
            generator_profile_id=cast(str, value.get("generator_profile_id")),
            generator_model_id=cast(str, value.get("generator_model_id")),
            schema_version=cast(Literal["fixture-fit-config/1.0.0"], value.get("schema_version")),
        )
        if value.get("generator_inference") != item.generator_inference_payload:
            raise FitContractError("fixture generator inference contract is invalid")
        return item


@dataclass(frozen=True, slots=True)
class ProductionJudgeCertifyConfig:
    sol_model_approval_hash: str | None = None
    initial_eval_rubric_approval_hash: str | None = None
    generator_approval_hash: str | None = None
    permanent_trace_governance_approval_hash: str | None = None
    schema_version: Literal["production-judge-certify-config/1.0.0"] = "production-judge-certify-config/1.0.0"
