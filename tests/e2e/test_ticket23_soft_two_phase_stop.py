from pathlib import Path

import pytest

from clawrl.artifacts import ArtifactStore
from clawrl.harness.journal import HarnessJournal
from clawrl.training.soft_two_phase_stop import (
    InjectedSoftStopControllerCrash,
    SoftStopError,
    SoftTwoPhaseStopConfig,
    SoftTwoPhaseStopWorkflow,
)


def test_soft_stop_is_idempotent_across_late_observation_and_grace_crash(tmp_path: Path) -> None:
    config = SoftTwoPhaseStopConfig("ticket23-e2e", grace_period=2)
    try:
        SoftTwoPhaseStopWorkflow.run(
            tmp_path,
            config=config,
            observation={"length_ratio": 3},
            crash_after="grace",
        )
    except InjectedSoftStopControllerCrash:
        pass
    first = SoftTwoPhaseStopWorkflow.resume(tmp_path, config.run_id)
    second = SoftTwoPhaseStopWorkflow.run(tmp_path, config=config, observation={"kl": 99})
    assert first.terminal is not None and second.terminal is not None
    assert first.terminal.content_hash == second.terminal.content_hash
    assert second.cluster is not None
    assert [item.payload["operation"] for item in second.cluster.outcomes].count("force_cancel") == 1
    journal = HarnessJournal(tmp_path, ArtifactStore(tmp_path), SoftTwoPhaseStopWorkflow._proposal_id(config.run_id))
    assert len(journal.strict_transition_snapshot()) == 5


def test_original_config_and_new_epoch_take_over_after_grace_crash(tmp_path: Path) -> None:
    config = SoftTwoPhaseStopConfig("ticket23-takeover", grace_period=2)
    with pytest.raises(InjectedSoftStopControllerCrash):
        SoftTwoPhaseStopWorkflow.run(
            tmp_path,
            config=config,
            observation={"kl": 1.0},
            crash_after="grace",
        )
    recovered = SoftTwoPhaseStopWorkflow.run(
        tmp_path,
        config=config,
        observation={"kl": 99.0},
        controller_epoch=2,
    )
    assert recovered.terminal is not None
    assert recovered.terminal.payload["status"] == "canceled"


def test_terminal_policy_ref_substitution_fails_closed(tmp_path: Path) -> None:
    config = SoftTwoPhaseStopConfig("ticket23-policy-tamper")
    SoftTwoPhaseStopWorkflow.run(tmp_path, config=config, observation={"kl": 1.0})
    store = ArtifactStore(tmp_path)
    alternate = store.put(
        "MonitoringPolicy",
        "1.0.0",
        {
            "entropy_floor": "0.20",
            "grace_period": 1,
            "judge_failure_rate": "0.50",
            "kl_threshold": "0.20",
            "length_ratio": "2.00",
            "policy_id": "alternate",
            "policy_version": "fixture-monitoring-policy/1.0.0",
        },
    )
    ref = tmp_path / "soft-two-phase-stops" / config.run_id / "policy.ref"
    ref.write_text(alternate.content_hash + "\n", encoding="ascii")
    with pytest.raises(SoftStopError, match="lineage"):
        SoftTwoPhaseStopWorkflow.resume(tmp_path, config.run_id)
