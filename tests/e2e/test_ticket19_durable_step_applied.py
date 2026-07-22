"""Ticket 19 fresh-process recovery seam."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from clawrl.training.durable_step_applied import DurableStepAppliedConfig, DurableStepAppliedWorkflow
from tests.contracts.test_ticket19_durable_step_applied import _setup


def test_resume_loads_marker_checkpoint_model_and_optimizer_in_fresh_process(tmp_path: Path) -> None:
    spec, ready = _setup(tmp_path, "durable-19-fresh")
    config = DurableStepAppliedConfig("durable-19-fresh", 0, spec.content_hash)
    snapshot = DurableStepAppliedWorkflow.run(tmp_path, config=config, step_ready=ready, experiment_spec=spec)
    code = """
import json, sys
from clawrl.training.durable_step_applied import DurableStepAppliedWorkflow, DurableStepAppliedConfig
s = DurableStepAppliedWorkflow.resume(sys.argv[1], DurableStepAppliedConfig(sys.argv[2], 0, sys.argv[3]))
print(json.dumps({'report': s.report.content_hash, 'checkpoint': s.checkpoint.content_hash,
                  'marker': s.step_applied.content_hash, 'model': s.model_state.content_hash,
                  'optimizer': s.optimizer_state.content_hash}, sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), config.run_id, spec.content_hash],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    resumed = json.loads(completed.stdout)
    assert resumed == {
        "checkpoint": snapshot.checkpoint.content_hash,
        "marker": snapshot.step_applied.content_hash,
        "model": snapshot.model_state.content_hash,
        "optimizer": snapshot.optimizer_state.content_hash,
        "report": snapshot.report.content_hash,
    }
