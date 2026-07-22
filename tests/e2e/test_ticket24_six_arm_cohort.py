from pathlib import Path
from typing import cast

import pytest

from clawrl.governor.six_arm_cohort import (
    InjectedSixArmCohortCrash,
    SixArmCohortConfig,
    SixArmCohortError,
    SixArmCohortWorkflow,
)


def test_six_logical_roles_freeze_protocol_and_promote_once(tmp_path: Path) -> None:
    snapshot = SixArmCohortWorkflow.run(tmp_path, config=SixArmCohortConfig("ticket24-success"))
    assert snapshot.terminal
    assert [item.payload["logical_role"] for item in snapshot.arms] == [
        "control",
        "exploration-1",
        "exploration-2",
        "exploration-3",
        "exploration-4",
        "control-replication",
    ]
    assert all(
        item.payload["harness_steps"] == ["submit", "status", "logs", "artifact", "checkpoint"]
        for item in snapshot.arms
    )
    assert snapshot.promotion is not None
    assert snapshot.promotion.payload["winner_logical_role"] == "exploration-4"
    assert cast(str, snapshot.promotion.payload["causal_boundary"]).endswith("black_box_descriptive_only")
    assert snapshot.state.payload["next_index"] == 6


def test_failures_and_policy_stops_are_terminal_before_unique_promotion(tmp_path: Path) -> None:
    config = SixArmCohortConfig(
        "ticket24-mixed",
        arm_outcomes=("success", "deterministic_failure", "policy_stop", "success", "success", "success"),
    )
    snapshot = SixArmCohortWorkflow.run(tmp_path, config=config)
    assert [item.payload["status"] for item in snapshot.arms] == [
        "success",
        "deterministic_failure",
        "policy_stop",
        "success",
        "success",
        "success",
    ]
    assert snapshot.promotion is not None
    assert snapshot.promotion.payload["winner_logical_role"] == "exploration-4"


def test_child_crash_resumes_idempotently_and_fences_stale_controller(tmp_path: Path) -> None:
    config = SixArmCohortConfig("ticket24-crash")
    with pytest.raises(InjectedSixArmCohortCrash):
        SixArmCohortWorkflow.run(tmp_path, config=config, crash_after_role="exploration-1")
    resumed = SixArmCohortWorkflow.resume(tmp_path, config.cohort_id)
    again = SixArmCohortWorkflow.resume(tmp_path, config.cohort_id)
    assert resumed.state.content_hash == again.state.content_hash
    assert [item.payload["logical_role"] for item in resumed.arms].count("exploration-1") == 1
    with pytest.raises(SixArmCohortError, match="STALE_COHORT_CONTROLLER_EPOCH"):
        SixArmCohortWorkflow.resume(tmp_path, config.cohort_id, controller_epoch=0)


def test_new_epoch_takeover_and_tie_fail_closed(tmp_path: Path) -> None:
    config = SixArmCohortConfig("ticket24-takeover", scores=(80, 70, 60, 50, 40, 30))
    with pytest.raises(InjectedSixArmCohortCrash):
        SixArmCohortWorkflow.run(tmp_path, config=config, crash_after_role="exploration-1")
    resumed = SixArmCohortWorkflow.resume(tmp_path, config.cohort_id, controller_epoch=2)
    assert resumed.terminal
    with pytest.raises(SixArmCohortError, match="STALE_COHORT_CONTROLLER_EPOCH"):
        SixArmCohortWorkflow.resume(tmp_path, config.cohort_id, controller_epoch=1)

    with pytest.raises(SixArmCohortError, match="unique winner"):
        SixArmCohortWorkflow.run(
            tmp_path,
            config=SixArmCohortConfig("ticket24-tie", scores=(80, 80, 60, 50, 40, 30)),
        )


def test_future_evaluation_dataset_is_not_visible_and_production_is_blocked(tmp_path: Path) -> None:
    with pytest.raises(SixArmCohortError, match="Future EvaluationDataset"):
        SixArmCohortConfig("ticket24-future", future_evaluation_dataset_hash="a" * 64)
    readiness = SixArmCohortWorkflow.production_readiness(tmp_path)
    assert readiness.payload["status"] == "blocked"
    assert readiness.payload["side_effects_permitted"] is False
    assert readiness.payload["submit_attempted"] is False
