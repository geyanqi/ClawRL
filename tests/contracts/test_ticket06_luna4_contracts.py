"""Closed-world and fail-closed contracts for Ticket 06."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from clawrl.adapters.scorers.certification_roles import output_schema_contract, ticket06_fixture_role_lineage
from clawrl.adapters.scorers.sol_fallback import FixtureSolFallbackEvidenceAdapter, SolFallbackBoundaryError
from clawrl.artifacts import ArtifactStore
from clawrl.judge.luna4_models import FixtureLuna4Config, Luna4ContractError, classify_sol_evidence


def _config() -> FixtureLuna4Config:
    return FixtureLuna4Config(
        run_id="ticket06-contract",
        try8_run_id="ticket05-exhausted",
        try8_exhaustion_hash="a" * 64,
        holdout_seed=6006,
        role_seed=6606,
    )


def test_ticket06_config_is_closed_world_and_role_turns_are_packet_derived() -> None:
    config = _config()
    assert FixtureLuna4Config.from_mapping(cast(dict[str, object], config.immutable_input_payload)) == config
    assert output_schema_contract("TeacherScorer", items_per_turn=4)["turns_semantics"] == {
        "exact_item_count": 32,
        "turn_count": "derived_from_packet_scoring_session",
        "wire_type": "integer",
    }
    assert output_schema_contract("TeacherScorer")["turns_semantics"] == {
        "exact_turn_count": 4,
        "wire_type": "integer",
    }
    assert ticket06_fixture_role_lineage("TeacherScorer").startswith("/fixture/ticket06_")

    mutated = dict(config.immutable_input_payload)
    mutated["hidden_search"] = True
    with pytest.raises(Luna4ContractError):
        FixtureLuna4Config.from_mapping(cast(dict[str, object], mutated))
    with pytest.raises(Luna4ContractError):
        replace(config, fault_schedule=("success", "success"))
    with pytest.raises(Luna4ContractError):
        replace(config, sol_evidence_mode="rubric_loosened")  # type: ignore[arg-type]


def test_sol_evidence_modes_are_not_conflated_or_used_to_loosen_v1() -> None:
    ordered = [index * 1_000_000 for index in range(32)]
    valid = classify_sol_evidence(ordered, "calibrated_scalar")
    relative = classify_sol_evidence(ordered, "group_relative_only")
    invalid = classify_sol_evidence(ordered, "invalid_calibrated_variance")
    assert valid["calibrated_scalar_valid"] is True and valid["training_authorized"] is True
    assert relative["reason_code"] == "SOL_GROUP_RELATIVE_ONLY_UNSUPPORTED_V1"
    assert relative["relative_order_available"] is True and relative["training_authorized"] is False
    assert invalid["reason_code"] == "SOL_CALIBRATED_VARIANCE_INVALID"
    assert invalid["scalar_variance_micros"] is None and invalid["training_authorized"] is False


def test_sol_fallback_boundary_is_stateful_idempotent_conflict_detecting_and_fail_closed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    label_set = store.put(
        "TeacherLabelSet",
        "1.0.0",
        {"labels": [{"scalar_micros": index * 1_000_000} for index in range(32)]},
    )
    adapter = FixtureSolFallbackEvidenceAdapter(tmp_path, store)
    request, observation = adapter.execute(
        run_id="ticket06-sol-boundary-contract", teacher_label_set=label_set, mode="calibrated_scalar"
    )
    replay_request, replay_observation = adapter.execute(
        run_id="ticket06-sol-boundary-contract", teacher_label_set=label_set, mode="calibrated_scalar"
    )
    assert replay_request.content_hash == request.content_hash
    assert replay_observation.content_hash == observation.content_hash
    with pytest.raises(SolFallbackBoundaryError):
        adapter.execute(
            run_id="ticket06-sol-boundary-contract", teacher_label_set=label_set, mode="group_relative_only"
        )
    ref = tmp_path / "boundaries" / "sol-fallback-evidence" / "ticket06-sol-boundary-contract" / "observation.ref"
    ref.unlink()
    with pytest.raises(SolFallbackBoundaryError):
        adapter.verify(
            request=request,
            observation=observation,
            teacher_label_set=label_set,
            mode="calibrated_scalar",
        )
