"""Closed-world contracts for preregistering the Future-100 evaluation protocol."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast

from clawrl.artifacts import MAX_SAFE_INTEGER, JsonValue
from clawrl.judge.fit_models import InitialEvalRubric

_HASH = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


class FinalEvaluationContractError(ValueError):
    """A protocol input widens or weakens the frozen v1 semantics."""


def _safe_id(value: object, name: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(cast(str, value)) is None:
        raise FinalEvaluationContractError(f"{name} is invalid")
    return cast(str, value)


def _hash(value: object, name: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise FinalEvaluationContractError(f"{name} must be a SHA-256 digest")
    return cast(str, value)


@dataclass(frozen=True, slots=True)
class EvaluationEnvironment:
    environment_id: str = "fixture-final-eval-environment-v1"
    schema_version: Literal["evaluation-environment/1.0.0"] = "evaluation-environment/1.0.0"

    def __post_init__(self) -> None:
        _safe_id(self.environment_id, "environment_id")
        if self.schema_version != "evaluation-environment/1.0.0":
            raise FinalEvaluationContractError("EvaluationEnvironment schema is unsupported")

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "backend_differences": {"request_transport": "non-semantic"},
            "environment_id": self.environment_id,
            "evaluator_visible_trajectory_schema": {
                "fields": ["prompt", "response", "tool_transcript"],
                "schema_version": "evaluator-visible-trajectory/1.0.0",
            },
            "prompt_wrapper": {"template": "{{prompt}}", "version": "identity-wrapper-v1"},
            "schema_version": self.schema_version,
            "semantic_decoding": {
                "max_output_tokens": 4096,
                "temperature_micros": 0,
                "top_p_micros": 1_000_000,
            },
            "tool_harness_policy": {
                "network": "denied",
                "policy_version": "fixture-final-eval-tools-v1",
                "tool_allowlist": ["javascript", "python", "regex"],
            },
        }


@dataclass(frozen=True, slots=True)
class PromptIdentityNormalizer:
    algorithm_version: str = "prompt-identity-nfkc-lf-trim-v1"
    schema_version: Literal["prompt-identity-normalizer/1.0.0"] = "prompt-identity-normalizer/1.0.0"

    def __post_init__(self) -> None:
        _safe_id(self.algorithm_version, "identity normalizer version")
        if self.schema_version != "prompt-identity-normalizer/1.0.0":
            raise FinalEvaluationContractError("PromptIdentityNormalizer schema is unsupported")

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "algorithm_version": self.algorithm_version,
            "hash_algorithm": "sha256",
            "schema_version": self.schema_version,
            "steps": ["unicode_nfkc", "crlf_to_lf", "trim_outer_whitespace"],
        }

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> PromptIdentityNormalizer:
        item = cls(algorithm_version=cast(str, value.get("algorithm_version")))
        if value != item.artifact_payload():
            raise FinalEvaluationContractError("PromptIdentityNormalizer fields are invalid")
        return item


@dataclass(frozen=True, slots=True)
class FinalEvaluationProtocolConfig:
    protocol_id: str
    rubric: InitialEvalRubric
    environment: EvaluationEnvironment
    sample_seed: int
    balanced_ab_seed: int
    source_contract_id: str = "fixture-future-hard-prompts-v1"
    identity_normalizer_version: str = "prompt-identity-nfkc-lf-trim-v1"
    window_duration_seconds: int = 86_400
    schema_version: Literal["final-evaluation-protocol-config/1.0.0"] = "final-evaluation-protocol-config/1.0.0"

    def __post_init__(self) -> None:
        _safe_id(self.protocol_id, "protocol_id")
        _safe_id(self.source_contract_id, "source_contract_id")
        _safe_id(self.identity_normalizer_version, "identity_normalizer_version")
        if type(self.rubric) is not InitialEvalRubric or type(self.environment) is not EvaluationEnvironment:
            raise FinalEvaluationContractError("raw rubric and semantic environment are required")
        if (
            type(self.sample_seed) is not int
            or type(self.balanced_ab_seed) is not int
            or not 0 <= self.sample_seed <= MAX_SAFE_INTEGER
            or not 0 <= self.balanced_ab_seed <= MAX_SAFE_INTEGER
            or self.sample_seed == self.balanced_ab_seed
            or self.window_duration_seconds != 86_400
            or self.schema_version != "final-evaluation-protocol-config/1.0.0"
        ):
            raise FinalEvaluationContractError("protocol seeds, window, or schema are invalid")

    def canonical_payload(
        self,
        *,
        environment_hash: str,
        decision_record_hash: str,
        data_source_skill_hash: str,
        identity_normalizer_hash: str,
        idempotency_key_schema_hash: str,
        prompt_wrapper_hash: str,
        base_generation_config_hash: str,
        trained_generation_config_hash: str,
    ) -> dict[str, JsonValue]:
        _hash(environment_hash, "environment_hash")
        _hash(decision_record_hash, "decision_record_hash")
        _hash(data_source_skill_hash, "data_source_skill_hash")
        _hash(identity_normalizer_hash, "identity_normalizer_hash")
        _hash(idempotency_key_schema_hash, "idempotency_key_schema_hash")
        _hash(prompt_wrapper_hash, "prompt_wrapper_hash")
        _hash(base_generation_config_hash, "base_generation_config_hash")
        _hash(trained_generation_config_hash, "trained_generation_config_hash")
        return {
            "balanced_ab_assignment": {
                "algorithm": "sha256-seeded-permutation-v1",
                "seed": self.balanced_ab_seed,
                "trained_as_a_count": 50,
                "trained_as_b_count": 50,
            },
            "decision_record_hash": decision_record_hash,
            "data_source_skill_hash": data_source_skill_hash,
            "evaluation_environment_hash": environment_hash,
            "generation_configs": {
                "base_generation_config_hash": base_generation_config_hash,
                "prompt_wrapper_hash": prompt_wrapper_hash,
                "trained_generation_config_hash": trained_generation_config_hash,
            },
            "identity_exclusion": {
                "algorithm_version": self.identity_normalizer_version,
                "normalizer_hash": identity_normalizer_hash,
                "comparison": "normalized_identity_hash_not_in_all_candidate_development_datasets",
                "semantic_near_duplicate_detection": "out_of_scope_v1",
            },
            "initial_eval_rubric_raw": self.rubric.artifact_payload(),
            "invalid_rules": {
                "campaign_replacement": "forbidden_v1",
                "malformed_committed_result": "invalid",
                "missing_item": "invalid",
                "non_idempotent_unknown_outcome": "invalid",
                "protocol_schema_drift": "invalid",
            },
            "predicate_contract": {
                "clauses": [
                    {"field": "difficulty", "operator": "eq", "value": "hard"},
                    {"field": "purpose", "operator": "eq", "value": "eval_only"},
                ],
                "version": "future-hard-predicate/1.0.0",
            },
            "protocol_id": self.protocol_id,
            "query_schema": {
                "event_time_field": "event_time_utc",
                "fields": [
                    "provider_row_id",
                    "prompt",
                    "difficulty",
                    "purpose",
                    "event_time_utc",
                    "ingestion_time_utc",
                ],
                "ingestion_time_field": "ingestion_time_utc",
                "source_contract_id": self.source_contract_id,
                "data_source_skill_hash": data_source_skill_hash,
                "version": "future-hard-query/1.0.0",
            },
            "retry_idempotency": {
                "committed_results_per_item": 1,
                "idempotency_key_schema_hash": idempotency_key_schema_hash,
                "provider_retry": "only_confirmed_no_result_or_same_idempotency_key",
                "retry_key_behavior": "reuse_exact_same_key",
                "unknown_outcome": "invalid",
            },
            "sample": {"algorithm": "sha256-seeded-without-replacement-v1", "seed": self.sample_seed, "size": 100},
            "schema_version": "final-evaluation-protocol/1.0.0",
            "time_eligibility": {
                "event_time": "strictly_greater_than_candidate_freeze_t0",
                "ingestion_time": "strictly_greater_than_candidate_freeze_t0",
            },
            "verdict_rules": {
                "allowed": ["A", "B", "tie"],
                "base_or_tie_win_credit": 0,
                "failed_max_trained_wins": 59,
                "passed_min_trained_wins": 60,
                "unseal_after_committed_verdicts": 100,
            },
            "window": {
                "extension_count": 1,
                "extension_duration_seconds": self.window_duration_seconds,
                "initial_duration_seconds": self.window_duration_seconds,
                "on_under_100": "extend_once_then_invalid",
                "reapply_same_predicate_to_union": True,
            },
        }


@dataclass(frozen=True, slots=True)
class ProductionFinalEvaluationConfig:
    trusted_clock_approval_hash: str | None = None
    data_source_approval_hash: str | None = None
    sealed_storage_approval_hash: str | None = None
    provider_contract_approval_hash: str | None = None
