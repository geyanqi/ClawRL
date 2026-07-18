"""Ticket 09 canonical preregistration contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from clawrl.artifacts import ArtifactStore
from clawrl.data.models import DataSourceSkill
from clawrl.evaluation.protocol_models import (
    EvaluationEnvironment,
    FinalEvaluationProtocolConfig,
    ProductionFinalEvaluationConfig,
)
from clawrl.evaluation.protocol_workflow import FinalEvaluationProtocolRegistry
from clawrl.judge.fit_models import InitialEvalRubric


def _config() -> FinalEvaluationProtocolConfig:
    return FinalEvaluationProtocolConfig(
        protocol_id="fixture-future-100-v1",
        rubric=InitialEvalRubric.fixture_default(),
        environment=EvaluationEnvironment(),
        sample_seed=90_001,
        balanced_ab_seed=90_002,
    )


def test_canonical_protocol_freezes_every_semantic_rule_in_hash(tmp_path: Path) -> None:
    snapshot = FinalEvaluationProtocolRegistry.preregister(tmp_path, campaign_id="campaign-09", config=_config())
    payload = snapshot.protocol.payload
    assert payload["initial_eval_rubric_raw"] == InitialEvalRubric.fixture_default().artifact_payload()
    assert payload["time_eligibility"] == {
        "event_time": "strictly_greater_than_candidate_freeze_t0",
        "ingestion_time": "strictly_greater_than_candidate_freeze_t0",
    }
    assert payload["window"] == {
        "extension_count": 1,
        "extension_duration_seconds": 86_400,
        "initial_duration_seconds": 86_400,
        "on_under_100": "extend_once_then_invalid",
        "reapply_same_predicate_to_union": True,
    }
    assert cast(dict[str, object], payload["sample"])["size"] == 100
    assert cast(dict[str, object], payload["verdict_rules"])["passed_min_trained_wins"] == 60
    assert cast(dict[str, object], payload["verdict_rules"])["failed_max_trained_wins"] == 59
    assert cast(dict[str, object], payload["balanced_ab_assignment"])["trained_as_a_count"] == 50
    assert cast(dict[str, object], payload["balanced_ab_assignment"])["trained_as_b_count"] == 50
    assert cast(dict[str, object], payload["retry_idempotency"])["unknown_outcome"] == "invalid"
    assert cast(dict[str, object], payload["retry_idempotency"])["retry_key_behavior"] == "reuse_exact_same_key"
    assert cast(dict[str, object], payload["invalid_rules"])["protocol_schema_drift"] == "invalid"
    query = cast(dict[str, object], payload["query_schema"])
    assert {"difficulty", "purpose"}.issubset(cast(list[str], query["fields"]))
    store = ArtifactStore(tmp_path)
    skill = store.read(cast(str, payload["data_source_skill_hash"]), expected_schema_name="DataSourceSkill")
    assert DataSourceSkill.from_mapping(cast(dict[str, object], skill.payload)).artifact_payload() == skill.payload
    identity = cast(dict[str, object], payload["identity_exclusion"])
    store.read(cast(str, identity["normalizer_hash"]), expected_schema_name="PromptIdentityNormalizer")
    retry = cast(dict[str, object], payload["retry_idempotency"])
    key_schema = store.read(
        cast(str, retry["idempotency_key_schema_hash"]), expected_schema_name="ProviderIdempotencyKeySchema"
    )
    assert "attempt" not in cast(list[str], key_schema.payload["key_fields"])
    generation = cast(dict[str, object], payload["generation_configs"])
    wrapper = store.read(
        cast(str, generation["prompt_wrapper_hash"]), expected_schema_name="FinalEvaluationPromptWrapper"
    )
    base = store.read(
        cast(str, generation["base_generation_config_hash"]), expected_schema_name="SemanticGenerationConfig"
    )
    trained = store.read(
        cast(str, generation["trained_generation_config_hash"]), expected_schema_name="SemanticGenerationConfig"
    )
    assert wrapper.payload == {"template": "{{prompt}}", "version": "identity-wrapper-v1"}
    assert base.payload["model_role"] == "base" and trained.payload["model_role"] == "trained"
    for key in (
        "checkpoint_binding",
        "evaluation_environment_hash",
        "prompt_wrapper_hash",
        "semantic_decoding",
        "semantic_equivalence_required",
    ):
        assert base.payload[key] == trained.payload[key]
    assert snapshot.receipt.payload["status"] == "preregistered_before_candidate_results"


def test_production_final_eval_is_explicitly_blocked(tmp_path: Path) -> None:
    report = FinalEvaluationProtocolRegistry.production_readiness(tmp_path, ProductionFinalEvaluationConfig())
    persisted = ArtifactStore(tmp_path).read(report.content_hash, expected_schema_name="ReadinessReport")
    assert persisted.payload["phase"] == "FINAL_EVAL"
    assert persisted.payload["status"] == "blocked"
    assert persisted.payload["side_effects_permitted"] is False
    assert len(cast(list[object], persisted.payload["checks"])) == 4
