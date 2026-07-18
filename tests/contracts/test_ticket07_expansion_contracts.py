"""Ticket 07 closed-world config and scorer packet contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from clawrl.adapters.scorers.certification_roles import (
    CertificationRoleOutputError,
    output_schema_contract,
    ticket07_fixture_role_lineage,
)
from clawrl.artifacts import sha256_hex
from clawrl.judge.luna_expansion_models import (
    FixtureLunaExpansionConfig,
    LunaExpansionContractError,
    ProductionLunaExpansionConfig,
)
from clawrl.judge.luna_expansion_workflow import LunaExpansionWorkflow


def _config() -> FixtureLunaExpansionConfig:
    return FixtureLunaExpansionConfig(
        run_id="ticket07-contract",
        source_run_id="ticket05-source",
        source_certification_report_hash=sha256_hex(b"source-report"),
        alignment_policy_hash=sha256_hex(b"policy"),
        holdout_seed=7007,
        role_seed=7707,
        fault_schedule=("timeout", "delayed", "success"),
    )


def test_expansion_config_is_closed_world_and_policy_is_run_identity() -> None:
    config = _config()
    assert FixtureLunaExpansionConfig.from_mapping(config.immutable_input_payload) == config
    widened = dict(config.immutable_input_payload)
    widened["training_step_holdout"] = True
    with pytest.raises(LunaExpansionContractError):
        FixtureLunaExpansionConfig.from_mapping(widened)
    with pytest.raises(LunaExpansionContractError):
        replace(config, fault_schedule=("success", "timeout"))
    assert replace(config, alignment_policy_hash=sha256_hex(b"new-policy")) != config


def test_student_supports_16_32_while_teacher_is_fixed_to_four() -> None:
    assert output_schema_contract("StudentJudge", items_per_turn=16)["turns_semantics"] == {
        "exact_item_count": 32,
        "turn_count": "derived_from_packet_scoring_session",
        "wire_type": "integer",
    }
    assert output_schema_contract("StudentJudge", items_per_turn=32)["turns_semantics"] == {
        "exact_item_count": 32,
        "turn_count": "derived_from_packet_scoring_session",
        "wire_type": "integer",
    }
    with pytest.raises(CertificationRoleOutputError):
        output_schema_contract("TeacherScorer", items_per_turn=16)
    assert ticket07_fixture_role_lineage("TeacherScorer").startswith("/fixture/ticket07_")


def test_production_expansion_is_fail_closed_without_real_boundaries(tmp_path: Path) -> None:
    report = LunaExpansionWorkflow.production_readiness(tmp_path, ProductionLunaExpansionConfig())
    assert report.payload["status"] == "blocked"
    assert report.payload["side_effects_permitted"] is False
    checks = cast(list[dict[str, object]], report.payload["checks"])
    assert {item["code"] for item in checks} >= {
        "SOURCE_LUNA8_PACK_APPROVAL_UNAVAILABLE",
        "SOL_MODEL_CONFIGURATION_UNAVAILABLE",
        "LUNA_MODEL_CONFIGURATION_UNAVAILABLE",
    }
