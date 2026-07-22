# ruff: noqa: E501

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from clawrl.training.hard_failure_stop import (
    HardFailureStopConfig,
    HardFailureStopWorkflow,
    InjectedHardFailureControllerCrash,
    StaleHardFailureController,
)


def test_fresh_process_recovers_after_cancel_provider_commit(tmp_path: Path) -> None:
    config = HardFailureStopConfig("ticket22-fresh")
    with pytest.raises(InjectedHardFailureControllerCrash):
        HardFailureStopWorkflow.run(
            tmp_path,
            config=config,
            observation={"failure_code": "REWARD_ATTEMPTS_EXHAUSTED"},
            crash_after="cluster_cancel",
        )
    code = """
import json, sys
from clawrl.training.hard_failure_stop import HardFailureStopWorkflow
s = HardFailureStopWorkflow.resume(sys.argv[1], sys.argv[2])
print(json.dumps({'terminal': s.terminal.content_hash,
                  'cluster': s.cluster.run_record.content_hash,
                  'cancel_count': sum(1 for x in s.cluster.outcomes if x.payload['operation'] == 'force_cancel')}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    first = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), config.run_id],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    recovered = json.loads(first.stdout)
    resumed = HardFailureStopWorkflow.resume(tmp_path, config.run_id)
    assert recovered["terminal"] == resumed.terminal.content_hash
    assert recovered["cluster"] == resumed.cluster.run_record.content_hash
    assert recovered["cancel_count"] == 1

    with pytest.raises(StaleHardFailureController):
        HardFailureStopWorkflow.resume(tmp_path, config.run_id, controller_epoch=0)
