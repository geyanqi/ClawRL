"""Ticket 19 checkpoint-first and StepApplied contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import ArtifactStore, canonical_json_bytes, sha256_hex
from clawrl.training.durable_step_applied import (
    DurableStepAppliedConfig,
    DurableStepAppliedTerminalStop,
    DurableStepAppliedWorkflow,
    InjectedDurableStepAppliedCrash,
)


def _setup(root: Path, run_id: str = "durable-19"):
    store = ArtifactStore(root)
    spec = store.put("ExperimentSpec", "1.0.0", {"experiment_id": run_id, "status": "frozen"})
    hashes = ["a" * 64, "b" * 64]
    ready = store.put(
        "StepReady",
        "1.0.0",
        {
            "expected_set_hash": "c" * 64,
            "expected_slot_count": len(hashes),
            "experiment_spec_hash": spec.content_hash,
            "global_step": 0,
            "optimizer_update": False,
            "reward_hashes": hashes,
            "reward_root_hash": sha256_hex(canonical_json_bytes(hashes)),
            "run_id": run_id,
            "status": "step_ready",
            "trainer_path": "classic",
            "uid_count": 1,
            "uid_state_hashes": [],
        },
    )
    return spec, ready


def test_update_checkpoint_and_marker_are_durable_and_not_replayed(tmp_path: Path) -> None:
    spec, ready = _setup(tmp_path)
    config = DurableStepAppliedConfig("durable-19", 0, spec.content_hash)
    first = DurableStepAppliedWorkflow.run(tmp_path, config=config, step_ready=ready, experiment_spec=spec)
    second = DurableStepAppliedWorkflow.run(tmp_path, config=config, step_ready=ready, experiment_spec=spec)
    assert first.checkpoint.content_hash == second.checkpoint.content_hash
    assert first.step_applied.content_hash == second.step_applied.content_hash
    assert first.report.content_hash == second.report.content_hash
    assert first.step_applied.payload["checkpoint_hash"] == first.checkpoint.content_hash
    assert first.checkpoint.payload["reward_set_hash"] == ready.payload["reward_root_hash"]
    assert first.model_state.payload["global_step"] == first.optimizer_state.payload["global_step"] == 0


@pytest.mark.parametrize("fault", ["crash_after_checkpoint", "crash_after_update"])
def test_crash_windows_recover_without_a_second_applied_marker(tmp_path: Path, fault: str) -> None:
    spec, ready = _setup(tmp_path, f"durable-19-{fault}")
    config = DurableStepAppliedConfig(f"durable-19-{fault}", 0, spec.content_hash, fault_kind=fault)
    with pytest.raises(InjectedDurableStepAppliedCrash):
        DurableStepAppliedWorkflow.run(tmp_path, config=config, step_ready=ready, experiment_spec=spec)
    recovered = DurableStepAppliedWorkflow.run(tmp_path, config=config, step_ready=ready, experiment_spec=spec)
    assert recovered.report.payload["status"] == "step_applied"
    assert recovered.step_applied.payload["checkpoint_hash"] == recovered.checkpoint.content_hash
    assert (
        len(list((tmp_path / "durable-step-applied" / config.run_id / "0" / "classic").glob("step-applied.ref"))) == 1
    )


def test_checkpoint_only_backfills_marker_and_marker_without_checkpoint_stops(tmp_path: Path) -> None:
    spec, ready = _setup(tmp_path)
    config = DurableStepAppliedConfig("durable-19", 0, spec.content_hash, fault_kind="crash_after_checkpoint")
    with pytest.raises(InjectedDurableStepAppliedCrash):
        DurableStepAppliedWorkflow.run(tmp_path, config=config, step_ready=ready, experiment_spec=spec)
    namespace = tmp_path / "durable-step-applied" / config.run_id / "0" / "classic"
    assert (namespace / "checkpoint.ref").exists()
    assert not (namespace / "step-applied.ref").exists()
    recovered = DurableStepAppliedWorkflow.run(tmp_path, config=config, step_ready=ready, experiment_spec=spec)
    assert recovered.step_applied.payload["checkpoint_hash"] == recovered.checkpoint.content_hash

    marker_only_root = tmp_path / "marker-only"
    spec2, ready2 = _setup(marker_only_root, "marker-only")
    config2 = DurableStepAppliedConfig("marker-only", 0, spec2.content_hash)
    completed = DurableStepAppliedWorkflow.run(
        marker_only_root, config=config2, step_ready=ready2, experiment_spec=spec2
    )
    (marker_only_root / "durable-step-applied" / "marker-only" / "0" / "classic" / "checkpoint.ref").unlink()
    with pytest.raises(DurableStepAppliedTerminalStop, match="STEP_APPLIED_CHECKPOINT_MISSING"):
        DurableStepAppliedWorkflow.run(marker_only_root, config=config2, step_ready=ready2, experiment_spec=spec2)
    assert completed.step_applied.payload["status"] == "applied"


def test_production_backend_readiness_is_blocked(tmp_path: Path) -> None:
    readiness = DurableStepAppliedWorkflow.production_readiness(tmp_path)
    assert readiness.payload["status"] == "blocked"
    assert readiness.payload["side_effects_permitted"] is False
    checks = cast(list[dict[str, str]], readiness.payload["checks"])
    assert {item["status"] for item in checks} == {"blocked"}
