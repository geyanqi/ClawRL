"""Ticket 04 frozen fit-generation and role-isolation contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from clawrl.adapters.scorers.role_isolation import RoleIsolationIngress, RoleIsolationOutputError
from clawrl.artifacts import ArtifactStore, canonical_json_bytes, sha256_hex
from clawrl.judge.fit_models import (
    FitContractError,
    FixtureFitConfig,
    GeneratorPlan,
    InitialEvalRubric,
    SolInferenceConfig,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def test_generator_plan_is_closed_world_and_exactly_32_unique_slots() -> None:
    plan = GeneratorPlan.create(
        dataset_version_hash=HASH_A,
        training_trace_hash=HASH_B,
        trace_id="tt-" + "1" * 40,
        seed=4004,
        include_online_response=False,
        generator_profile_id="fixture-fit-generator-v1",
        generator_model_id="fixture-content-derived-v1",
        inference_config_hash=HASH_C,
    )
    payload = plan.artifact_payload()
    slots = cast(list[dict[str, Any]], payload["slots"])
    assert isinstance(slots, list) and len(slots) == 32
    assert [slot["slot_index"] for slot in slots] == list(range(32))
    assert len({slot["slot_id"] for slot in slots}) == 32
    assert {slot["origin"] for slot in slots} == {"fresh_rollout"}
    assert plan.content_hash == sha256_hex(canonical_json_bytes(payload))

    online = replace(
        plan,
        include_online_response=True,
        slots=GeneratorPlan.create(
            dataset_version_hash=HASH_A,
            training_trace_hash=HASH_B,
            trace_id="tt-" + "1" * 40,
            seed=4004,
            include_online_response=True,
            generator_profile_id="fixture-fit-generator-v1",
            generator_model_id="fixture-content-derived-v1",
            inference_config_hash=HASH_C,
        ).slots,
    )
    online_slots = cast(list[dict[str, Any]], online.artifact_payload()["slots"])
    assert sum(slot["origin"] == "online_response" for slot in online_slots) == 1
    assert online_slots[0]["origin"] == "online_response"


@pytest.mark.parametrize("field,value", [("target_count", 31), ("target_count", True), ("seed", True)])
def test_generator_plan_rejects_wrong_cardinality_and_bool_integers(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "dataset_version_hash": HASH_A,
        "training_trace_hash": HASH_B,
        "trace_id": "tt-" + "1" * 40,
        "seed": 4004,
        "include_online_response": False,
        "generator_profile_id": "fixture-fit-generator-v1",
        "generator_model_id": "fixture-content-derived-v1",
        "inference_config_hash": HASH_C,
        "target_count": 32,
    }
    kwargs[field] = value
    with pytest.raises(FitContractError):
        GeneratorPlan.create(**kwargs)  # type: ignore[arg-type]


def test_role_input_schema_versions_have_exactly_one_artifact_publisher() -> None:
    source_root = Path(__file__).parents[2] / "src" / "clawrl"
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(source_root.rglob("*.py")))
    for schema_name in ("PromptOptimizerInputPacket", "AlignmentAuditorInputPacket"):
        assert source.count(f'.put("{schema_name}", "1.0.0"') == 1
    fit_models_source = (source_root / "judge" / "fit_models.py").read_text(encoding="utf-8")
    assert "build_prompt_optimizer_packet" not in fit_models_source
    assert "build_alignment_auditor_packet" not in fit_models_source


@pytest.mark.parametrize(
    ("role_type", "schema_name", "legacy_payload"),
    [
        (
            "PromptOptimizer",
            "PromptOptimizerInputPacket",
            {
                "aggregate_teacher_diagnostics": {"label_count": 32, "variance_micros": 1},
                "base_prompt_hash": HASH_A,
                "fit_trajectory_content_hashes": [f"{index:064x}" for index in range(1, 33)],
                "output_schema": {"required": ["candidate_prompt"]},
                "packet_id": "legacy-optimizer",
                "role": "prompt_optimizer",
                "schema_version": "prompt-optimizer-input/1.0.0",
                "seed": 1,
                "teacher_label_set_hash": HASH_B,
                "trace_id": "tt-" + "1" * 40,
            },
        ),
        (
            "AlignmentAuditor",
            "AlignmentAuditorInputPacket",
            {
                "alignment_policy_hash": HASH_A,
                "candidate_prompt_hash": HASH_B,
                "holdout_content_hashes": [f"{index:064x}" for index in range(1, 33)],
                "holdout_set_hash": HASH_C,
                "output_schema": {"required": ["verdict"]},
                "packet_id": "legacy-auditor",
                "role": "alignment_auditor",
                "schema_version": "alignment-auditor-input/1.0.0",
                "seed": 1,
                "trace_id": "tt-" + "1" * 40,
            },
        ),
    ],
)
def test_legacy_multimeaning_role_packets_fail_before_role_call(
    tmp_path: Path,
    role_type: str,
    schema_name: str,
    legacy_payload: dict[str, Any],
) -> None:
    payload = dict(legacy_payload)
    payload["allowlisted_fields"] = sorted([*payload, "allowlisted_fields"])
    packet = ArtifactStore(tmp_path).put(schema_name, "1.0.0", payload)
    with pytest.raises(RoleIsolationOutputError):
        RoleIsolationIngress.validate_input_packet(
            tmp_path,
            role_type=cast(Any, role_type),
            input_packet_hash=packet.content_hash,
        )


def test_fixture_config_requires_real_frozen_rubric_and_sol_contract() -> None:
    rubric = InitialEvalRubric.fixture_default()
    sol = SolInferenceConfig.fixture_default()
    config = FixtureFitConfig(
        run_id="ticket04-contract",
        dataset_version_hash=HASH_A,
        trace_id="tt-" + "1" * 40,
        generator_seed=4004,
        role_seed=4304,
        rubric=rubric,
        sol_inference=sol,
    )
    assert config.rubric.content_hash != config.sol_inference.content_hash
    with pytest.raises(FitContractError):
        replace(config, generator_seed=True)  # type: ignore[arg-type]
