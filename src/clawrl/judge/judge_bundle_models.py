"""Closed-world inputs for total calibrated JudgeBundle publication."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from clawrl.artifacts import JsonValue

_HASH = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_STRATEGIES = {"calibrated_scalar", "hierarchical_rank", "local_microgroup_rank"}


class JudgeBundleContractError(ValueError):
    """A bundle/compiler input is malformed or widens v1 semantics."""


def _safe_id(value: object, name: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(cast(str, value)) is None:
        raise JudgeBundleContractError(f"{name} is invalid")
    return cast(str, value)


def _hash(value: object, name: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise JudgeBundleContractError(f"{name} must be a SHA-256 digest")
    return cast(str, value)


@dataclass(frozen=True, slots=True)
class FixtureJudgeBundleConfig:
    run_id: str
    dataset_version_hash: str
    reward_schema_hash: str
    scalarizer_hash: str
    algorithm_contract_hash: str
    aggregation: str = "calibrated_scalar"
    schema_version: str = "fixture-judge-bundle-config/1.0.0"

    def __post_init__(self) -> None:
        _safe_id(self.run_id, "run_id")
        _hash(self.dataset_version_hash, "dataset version hash")
        _hash(self.reward_schema_hash, "reward schema hash")
        _hash(self.scalarizer_hash, "scalarizer hash")
        _hash(self.algorithm_contract_hash, "algorithm contract hash")
        if self.aggregation not in _STRATEGIES or self.schema_version != "fixture-judge-bundle-config/1.0.0":
            raise JudgeBundleContractError("fixture JudgeBundle config is invalid")

    @property
    def immutable_input_payload(self) -> dict[str, JsonValue]:
        return {
            "aggregation": self.aggregation,
            "algorithm_contract_hash": self.algorithm_contract_hash,
            "dataset_version_hash": self.dataset_version_hash,
            "reward_schema_hash": self.reward_schema_hash,
            "run_id": self.run_id,
            "scalarizer_hash": self.scalarizer_hash,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FixtureJudgeBundleConfig:
        try:
            item = cls(
                run_id=cast(str, value["run_id"]),
                dataset_version_hash=cast(str, value["dataset_version_hash"]),
                reward_schema_hash=cast(str, value["reward_schema_hash"]),
                scalarizer_hash=cast(str, value["scalarizer_hash"]),
                algorithm_contract_hash=cast(str, value["algorithm_contract_hash"]),
                aggregation=cast(str, value["aggregation"]),
                schema_version=cast(str, value["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise JudgeBundleContractError("fixture JudgeBundle config mapping is invalid") from error
        if dict(value) != item.immutable_input_payload:
            raise JudgeBundleContractError("fixture JudgeBundle config has unknown or altered fields")
        return item


@dataclass(frozen=True, slots=True)
class FixtureExperimentSpecConfig:
    experiment_id: str
    dataset_version_id: str
    dataset_version_hash: str
    judge_bundle_hash: str
    trace_set_hash: str
    aggregation: str = "calibrated_scalar"
    schema_version: str = "fixture-experiment-spec/1.0.0"

    def __post_init__(self) -> None:
        _safe_id(self.experiment_id, "experiment_id")
        _safe_id(self.dataset_version_id, "dataset_version_id")
        _hash(self.dataset_version_hash, "dataset version hash")
        _hash(self.judge_bundle_hash, "JudgeBundle hash")
        _hash(self.trace_set_hash, "trace set hash")
        if self.aggregation not in _STRATEGIES or self.schema_version != "fixture-experiment-spec/1.0.0":
            raise JudgeBundleContractError("fixture ExperimentSpec is invalid")


@dataclass(frozen=True, slots=True)
class ProductionJudgeBundleConfig:
    dataset_approval_hash: str | None = None
    certification_controller_approval_hash: str | None = None
    reward_schema_approval_hash: str | None = None
    scalarizer_approval_hash: str | None = None
    algorithm_contract_approval_hash: str | None = None
    permanent_trace_governance_approval_hash: str | None = None
