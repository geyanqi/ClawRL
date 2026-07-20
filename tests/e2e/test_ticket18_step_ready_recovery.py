"""Ticket 18 crash/restart and recoverable-fault E2E checks."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from clawrl.training.step_ready_recovery import InjectedStepReadyRecoveryCrash, StepReadyRecoveryWorkflow
from tests.contracts.test_ticket18_step_ready_recovery import _setup


def test_controller_restart_reuses_committed_uid_states(tmp_path: Path) -> None:
    config, expected, packs, spec = _setup(tmp_path, "step-ready-18-crash")
    crashing = config.__class__(
        config.run_id,
        config.global_step,
        config.experiment_spec_hash,
        trainer_path=config.trainer_path,
        chunk_size=config.chunk_size,
        crash_after_uid=4,
    )
    with pytest.raises(InjectedStepReadyRecoveryCrash):
        StepReadyRecoveryWorkflow.run(
            tmp_path, config=crashing, expected_set=expected, judge_packs=packs, experiment_spec=spec
        )
    resumed = StepReadyRecoveryWorkflow.run(
        tmp_path, config=crashing, expected_set=expected, judge_packs=packs, experiment_spec=spec
    )
    assert len(resumed.rewards) == 2048
    assert resumed.report.payload["status"] == "step_ready"


def test_late_result_is_quarantined_but_step_ready_is_unique(tmp_path: Path) -> None:
    config, expected, packs, spec = _setup(tmp_path, "step-ready-18-late")
    late = config.__class__(config.run_id, config.global_step, config.experiment_spec_hash, fault_kind="late_result")
    snapshot = StepReadyRecoveryWorkflow.run(
        tmp_path, config=late, expected_set=expected, judge_packs=packs, experiment_spec=spec
    )
    assert snapshot.step_ready.payload["status"] == "step_ready"
    assert len(cast(list[str], snapshot.report.payload["fault_event_hashes"])) == 17
