from pathlib import Path

import pytest

from clawrl.artifacts import ArtifactStore
from clawrl.harness.journal import HarnessJournal
from clawrl.training.soft_two_phase_stop import (
    InjectedSoftStopControllerCrash,
    SoftTwoPhaseStopConfig,
    SoftTwoPhaseStopWorkflow,
)


def test_frozen_policy_evidence_and_transition_lineage(tmp_path: Path) -> None:
    config = SoftTwoPhaseStopConfig("ticket23-contract", grace_period=3, budget_exhausted=True)
    snapshot = SoftTwoPhaseStopWorkflow.run(tmp_path, config=config, observation={"kl": 0.8, "evidence": "obs-1"})
    assert snapshot.terminal is not None
    assert snapshot.grace is not None and snapshot.grace.payload["configured_period"] == 3
    assert snapshot.cluster is not None
    assert [
        item.schema_name
        for item in HarnessJournal(
            tmp_path, ArtifactStore(tmp_path), SoftTwoPhaseStopWorkflow._proposal_id(config.run_id)
        ).strict_transition_snapshot()
    ] == ["DecisionProposal", "ActionPlan", "HarnessDecision", "ActionObservation", "DecisionOutcome"]
    assert [item.payload["operation"] for item in snapshot.cluster.outcomes] == [
        "submit",
        "status",
        "logs",
        "artifact",
        "checkpoint",
        "force_cancel",
    ]
    assert snapshot.action_plan is not None
    assert snapshot.action_plan.payload["actions"] == ["checkpoint", "grace", "force_cancel"]
    assert snapshot.action_plan.payload["budget_cost"] == 0


@pytest.mark.parametrize("fault", ["timeout", "failure"])
def test_checkpoint_fault_recovers_to_cancel_with_terminal_evidence(tmp_path: Path, fault: str) -> None:
    snapshot = SoftTwoPhaseStopWorkflow.run(
        tmp_path,
        config=SoftTwoPhaseStopConfig("ticket23-checkpoint-" + fault, checkpoint_fault=fault),
        observation={"judge_failure_rate": 1},
    )
    assert snapshot.cluster is not None
    assert snapshot.cluster.run_record.payload["failed_operation"] == "checkpoint"
    assert snapshot.cluster.run_record.payload["closure_operation"] == "force_cancel"
    assert snapshot.terminal is not None and snapshot.terminal.payload["status"] == "canceled"


def test_reward_only_increase_continues_without_stop(tmp_path: Path) -> None:
    snapshot = SoftTwoPhaseStopWorkflow.run(
        tmp_path,
        config=SoftTwoPhaseStopConfig("ticket23-reward"),
        observation={"reward_delta": 2.0},
    )
    assert snapshot.continued is True
    assert snapshot.terminal is None
    assert snapshot.stop_request is not None and snapshot.stop_request.payload["stop"] is False


def test_crash_after_checkpoint_is_provider_query_recovered(tmp_path: Path) -> None:
    config = SoftTwoPhaseStopConfig("ticket23-recovery")
    with pytest.raises(InjectedSoftStopControllerCrash):
        SoftTwoPhaseStopWorkflow.run(tmp_path, config=config, observation={"entropy": 0.0}, crash_after="checkpoint")
    recovered = SoftTwoPhaseStopWorkflow.resume(tmp_path, config.run_id)
    assert recovered.terminal is not None and recovered.terminal.payload["status"] == "canceled"
    assert recovered.cluster is not None
    assert recovered.cluster.recovery_evidence


def test_production_soft_stop_is_fail_closed(tmp_path: Path) -> None:
    report = SoftTwoPhaseStopWorkflow.production_readiness(tmp_path)
    assert report.payload["status"] == "blocked"
    assert report.payload["side_effects_permitted"] is False
    with pytest.raises(RuntimeError, match="production soft two-phase stop is blocked"):
        SoftTwoPhaseStopWorkflow.run(
            tmp_path,
            config=SoftTwoPhaseStopConfig("ticket23-production"),
            observation={"kl": 1},
            execution_profile="production",
        )
