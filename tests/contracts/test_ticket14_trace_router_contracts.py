"""Ticket 14 single-trace Router contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import Artifact, ArtifactStore
from clawrl.router.trace_router import FixtureTraceRouter, RouterCapacityConfig, TraceRouterConfig
from clawrl.training.classic_identity import ClassicSourceRow
from clawrl.training.expected_trajectory_set import ExpectedTrajectorySetConfig, ExpectedTrajectorySetWorkflow


def _expected(root: Path, run_id: str) -> Artifact:
    config = ExpectedTrajectorySetConfig(run_id, 7, 128)
    return ExpectedTrajectorySetWorkflow.run(
        root,
        config=config,
        sources=(ClassicSourceRow("trace-001", "uid-001", "judge-pack-001", "prompt"),),
        arrival_ordinals=tuple(reversed(range(128))),
        transport="classic",
    ).expected_set


def _pack(root: Path, items: int, *, tier: str = "luna") -> Artifact:
    return ArtifactStore(root).put(
        "JudgePack",
        "1.0.0",
        {
            "aggregation": "calibrated_scalar",
            "algorithm_contract": "grpo-calibrated-scalar-v1",
            "items_per_turn": items,
            "judge_pack_id": "judge-pack-001",
            "judge_pack_version": "1.0.0",
            "prompt_hash": "a" * 64,
            "scalarizer_hash": "b" * 64,
            "scalarizer_version": "1.0.0",
            "scorer_tier": tier,
            "status": "certified",
            "trace_id": "trace-001",
            "uid": "uid-001",
        },
    )


@pytest.mark.parametrize(
    ("items", "tier", "wave_count"),
    [(4, "luna", 32), (8, "luna", 16), (16, "luna", 8), (32, "luna", 4), (4, "sol", 32)],
)
def test_pack_selected_wave_size_and_terminal_variants(tmp_path: Path, items: int, tier: str, wave_count: int) -> None:
    run_id = f"router-{tier}-{items}"
    snapshot = FixtureTraceRouter.run(
        tmp_path,
        config=TraceRouterConfig(run_id, 7, "uid-001", 1, RouterCapacityConfig(5)),
        expected_set=_expected(tmp_path, run_id),
        judge_pack=_pack(tmp_path, items, tier=tier),
    )
    assert len(snapshot.results) == 128
    assert len(snapshot.waves) == wave_count
    assert all(cast(int, wave.payload["active_subthreads"]) <= 5 for wave in snapshot.waves)
    assert snapshot.report.payload["sol_shadow_called"] is False
    assert snapshot.report.payload["prompt_optimized_during_training"] is False


def test_global_capacity_queues_excess_without_admission_rejection(tmp_path: Path) -> None:
    expected = _expected(tmp_path, "router-capacity")
    snapshot = FixtureTraceRouter.run(
        tmp_path,
        config=TraceRouterConfig("router-capacity", 7, "uid-001", 1, RouterCapacityConfig(2)),
        expected_set=expected,
        judge_pack=_pack(tmp_path, 8),
    )
    assert all(wave.payload["active_subthreads"] == 2 for wave in snapshot.waves)
    assert all(wave.payload["queued_item_count"] == 6 for wave in snapshot.waves)
    assert all(wave.payload["admission_policy"] == "queue_only_no_predicted_delay_rejection" for wave in snapshot.waves)


def test_production_router_readiness_is_blocked(tmp_path: Path) -> None:
    for phase in ("TRAIN_35B", "TRAIN_122B"):
        report = FixtureTraceRouter.production_readiness(tmp_path / phase, phase)
        assert report.payload["status"] == "blocked"
        assert report.payload["side_effects_permitted"] is False
