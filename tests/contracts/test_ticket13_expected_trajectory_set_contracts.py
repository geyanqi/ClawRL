"""Ticket 13 ExpectedTrajectorySet contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import JsonValue
from clawrl.training.classic_identity import ClassicSourceRow, ClassicTrajectoryRow
from clawrl.training.expected_trajectory_set import (
    ExpectedTrajectorySetConfig,
    ExpectedTrajectorySetError,
    ExpectedTrajectorySetWorkflow,
    ProductionExpectedTrajectorySetConfig,
)
from clawrl.training.v1_transfer_identity import V1RewardIdentityError, V1RewardManager


def _sources() -> tuple[ClassicSourceRow, ...]:
    return (
        ClassicSourceRow("trace-001", "uid-001", "judge-pack-001", "prompt one"),
        ClassicSourceRow("trace-002", "uid-002", "judge-pack-002", "prompt two"),
    )


def test_expected_set_freezes_configurable_uid_cartesian_product_and_mapping(tmp_path: Path) -> None:
    config = ExpectedTrajectorySetConfig("expected-contract-13", 21, rollout_count=3)
    frozen = ExpectedTrajectorySetWorkflow.freeze(tmp_path, config=config, sources=_sources())
    slots = cast(list[dict[str, JsonValue]], frozen.payload["slots"])
    assert len(slots) == 6
    assert [(item["uid"], item["rollout_index"]) for item in slots] == [
        ("uid-001", 0),
        ("uid-001", 1),
        ("uid-001", 2),
        ("uid-002", 0),
        ("uid-002", 1),
        ("uid-002", 2),
    ]
    assert all(item["global_step"] == 21 and item["expected_rollout_count"] == 3 for item in slots)
    assert {(item["trace_id"], item["judge_pack_id"]) for item in slots} == {
        ("trace-001", "judge-pack-001"),
        ("trace-002", "judge-pack-002"),
    }


def test_slot_publish_is_idempotent_but_mapping_step_and_content_conflicts_fail_closed(tmp_path: Path) -> None:
    config = ExpectedTrajectorySetConfig("expected-conflict-13", 9, rollout_count=2)
    frozen = ExpectedTrajectorySetWorkflow.freeze(tmp_path, config=config, sources=_sources())
    row = ExpectedTrajectorySetWorkflow.generated_rows(config, _sources())[0]
    first = ExpectedTrajectorySetWorkflow.publish_slot(tmp_path, expected_set=frozen, row=row)
    assert (
        ExpectedTrajectorySetWorkflow.publish_slot(tmp_path, expected_set=frozen, row=row).content_hash
        == first.content_hash
    )
    changed = ClassicTrajectoryRow(row.identity, row.prompt, "changed response")
    with pytest.raises(ExpectedTrajectorySetError, match="TRAJECTORY_STABLE_KEY_CONFLICT"):
        ExpectedTrajectorySetWorkflow.publish_slot(tmp_path, expected_set=frozen, row=changed)
    cross_step = ClassicTrajectoryRow(
        type(row.identity)(
            row.identity.run_id,
            row.identity.global_step + 1,
            row.identity.trace_id,
            row.identity.uid,
            row.identity.rollout_index,
            row.identity.expected_rollout_count,
            row.identity.judge_pack_id,
        ),
        row.prompt,
        row.response,
    )
    with pytest.raises(ExpectedTrajectorySetError, match="not in frozen ExpectedTrajectorySet"):
        ExpectedTrajectorySetWorkflow.publish_slot(tmp_path, expected_set=frozen, row=cross_step)


def test_v1_reward_manager_invokes_the_shared_expected_set_validator(tmp_path: Path) -> None:
    config = ExpectedTrajectorySetConfig("expected-v1-manager-13", 11, rollout_count=2)
    frozen = ExpectedTrajectorySetWorkflow.freeze(tmp_path, config=config, sources=_sources())
    ExpectedTrajectorySetWorkflow.publish_batch(
        tmp_path,
        expected_set=frozen,
        rows=ExpectedTrajectorySetWorkflow.generated_rows(config, _sources()),
    )
    authorization = V1RewardManager.authorize_expected_trajectory_set(tmp_path, frozen)
    assert authorization.payload["transport"] == "v1"
    assert authorization.payload["validator_id"] == "shared-expected-trajectory-set/1.0.0"

    incomplete_root = tmp_path / "incomplete"
    incomplete = ExpectedTrajectorySetWorkflow.freeze(incomplete_root, config=config, sources=_sources())
    with pytest.raises(V1RewardIdentityError, match="failed closed"):
        V1RewardManager.authorize_expected_trajectory_set(incomplete_root, incomplete)


@pytest.mark.parametrize("phase", ["TRAIN_35B", "TRAIN_122B"])
def test_production_train_readiness_is_blocked_without_verified_runtime(tmp_path: Path, phase: str) -> None:
    report = ExpectedTrajectorySetWorkflow.production_readiness(
        tmp_path / phase,
        ProductionExpectedTrajectorySetConfig(cast(str, phase)),
    )
    checks = cast(list[dict[str, JsonValue]], report.payload["checks"])
    assert report.payload["status"] == "blocked"
    assert report.payload["side_effects_permitted"] is False
    assert "REAL_VERL_EXPECTED_SET_ADAPTER_UNVERIFIED" in {item["code"] for item in checks}
