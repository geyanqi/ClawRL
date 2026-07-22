from __future__ import annotations

from pathlib import Path

import pytest

from clawrl.artifacts import ArtifactStore
from clawrl.harness.journal import HarnessJournal
from clawrl.training.hard_failure_stop import (
    HardFailureKind,
    HardFailureObservation,
    HardFailureStopConfig,
    HardFailureStopError,
    HardFailureStopWorkflow,
    StaleHardFailureController,
)


@pytest.mark.parametrize(
    ("payload", "kind"),
    [
        ({"metric": float("nan")}, HardFailureKind.NAN_INF),
        ({"failure_code": "CUDA_OOM"}, HardFailureKind.OOM),
        ({"crash_count": 3}, HardFailureKind.REPEATED_CRASH),
        ({"heartbeat_stalled": True}, HardFailureKind.HEARTBEAT_STALLED),
        ({"scheduler_stalled": True}, HardFailureKind.SCHEDULER_STALLED),
        ({"failure_code": "CFS_CHECKSUM_CORRUPTION"}, HardFailureKind.CFS_CORRUPTION),
        ({"failure_code": "REWARD_ATTEMPTS_EXHAUSTED"}, HardFailureKind.PERMANENT_REWARD_FAILURE),
    ],
)
def test_hard_failure_classification(payload: dict[str, object], kind: HardFailureKind) -> None:
    assert HardFailureStopWorkflow.classify(payload) == kind


def test_cancel_and_reward_terminal_stop_are_exactly_once(tmp_path: Path) -> None:
    config = HardFailureStopConfig("ticket22-run")
    first = HardFailureStopWorkflow.run(tmp_path, config=config, observation={"failure_code": "OOM"})
    second = HardFailureStopWorkflow.run(tmp_path, config=config, observation={"failure_code": "late-CFS-corruption"})
    assert second.terminal.content_hash == first.terminal.content_hash
    assert second.reward_terminal_stop.content_hash == first.reward_terminal_stop.content_hash
    assert [o.payload["operation"] for o in second.cluster.outcomes] == ["force_cancel"]
    assert first.terminal.payload["optimizer_update_permitted"] is False
    assert first.reward_terminal_stop.payload["reward_publish_permitted"] is False
    journal = HarnessJournal(tmp_path, ArtifactStore(tmp_path), "hard-failure-ticket22-run")
    assert [item.schema_name for item in journal.strict_transition_snapshot()] == [
        "DecisionProposal",
        "ActionPlan",
        "HarnessDecision",
        "ActionObservation",
        "DecisionOutcome",
    ]
    assert first.terminal.payload["audit_event_hash"]


def test_stale_controller_cannot_close_or_emergency_train(tmp_path: Path) -> None:
    config = HardFailureStopConfig("ticket22-stale", controller_epoch=3)
    with pytest.raises(StaleHardFailureController):
        HardFailureStopWorkflow.run(tmp_path, config=config, observation="OOM", controller_epoch=2)
    assert HardFailureStopWorkflow.emergency_action_allowed("force_cancel")
    assert not HardFailureStopWorkflow.emergency_action_allowed("submit")
    assert not HardFailureStopWorkflow.emergency_action_allowed("query")
    assert not HardFailureStopWorkflow.emergency_action_allowed("scorer")


def test_production_readiness_is_blocked(tmp_path: Path) -> None:
    report = HardFailureStopWorkflow.production_readiness(tmp_path)
    assert report.payload["status"] == "blocked"
    assert report.payload["side_effects_permitted"] is False


def test_production_execution_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(HardFailureStopError, match="production hard-failure stop is blocked"):
        HardFailureStopWorkflow.run(
            tmp_path,
            config=HardFailureStopConfig("ticket22-production"),
            observation="OOM",
            execution_profile="production",
        )

    with pytest.raises(HardFailureStopError, match="execution_profile must be fixture or production"):
        HardFailureStopWorkflow.run(
            tmp_path,
            config=HardFailureStopConfig("ticket22-unknown-profile"),
            observation="OOM",
            execution_profile="PRODUCTION",
        )


def test_typed_observation_run_identity_is_fenced(tmp_path: Path) -> None:
    with pytest.raises(HardFailureStopError, match="observation run identity changed"):
        HardFailureStopWorkflow.run(
            tmp_path,
            config=HardFailureStopConfig("ticket22-run"),
            observation=HardFailureObservation("other-run", HardFailureKind.OOM),
        )


def test_run_id_with_cluster_safe_colon_keeps_harness_durable(tmp_path: Path) -> None:
    snapshot = HardFailureStopWorkflow.run(
        tmp_path,
        config=HardFailureStopConfig("ticket22:colon"),
        observation="OOM",
    )
    assert snapshot.terminal.payload["run_id"] == "ticket22:colon"
