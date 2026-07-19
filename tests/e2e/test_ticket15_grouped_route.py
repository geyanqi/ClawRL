"""Ticket 15 grouped route durable acceptance tests."""

import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import ArtifactStore
from clawrl.router.grouped_route import (
    FixtureGroupedToolRouter,
    GroupedRouteConfig,
    InjectedGroupedRouteCrash,
)
from clawrl.training.classic_identity import ClassicSourceRow
from clawrl.training.expected_trajectory_set import ExpectedTrajectorySetConfig, ExpectedTrajectorySetWorkflow


def _inputs(root: Path, run_id: str):
    cfg = ExpectedTrajectorySetConfig(run_id, 7, 128)
    source = (ClassicSourceRow("trace-001", "uid-001", "judge-pack-001", "prompt"),)
    expected = ExpectedTrajectorySetWorkflow.run(
        root, config=cfg, sources=source, arrival_ordinals=tuple(reversed(range(128))), transport="classic"
    ).expected_set
    pack = ArtifactStore(root).put(
        "JudgePack",
        "1.0.0",
        {
            "aggregation": "calibrated_scalar",
            "algorithm_contract": "grpo-calibrated-scalar-v1",
            "items_per_turn": 8,
            "judge_pack_id": "judge-pack-001",
            "judge_pack_version": "1.0.0",
            "prompt_hash": "a" * 64,
            "scalarizer_hash": "b" * 64,
            "scalarizer_version": "1.0.0",
            "scorer_tier": "luna",
            "status": "certified",
            "trace_id": "trace-001",
            "uid": "uid-001",
        },
    )
    return expected, pack


def test_grouped_route_persists_rewards_denials_and_fresh_resume(tmp_path: Path) -> None:
    expected, pack = _inputs(tmp_path, "grouped-route")
    report = FixtureGroupedToolRouter.run(
        tmp_path, config=GroupedRouteConfig("grouped-route", 7, "uid-001"), expected_set=expected, judge_pack=pack
    )
    assert report.payload["status"] == "resolved"
    assert len(cast(list[str], report.payload["reward_hashes"])) == 128
    assert len(cast(list[str], report.payload["decision_hashes"])) == 4
    assert FixtureGroupedToolRouter.resume(tmp_path, "grouped-route", 7, "uid-001").content_hash == report.content_hash
    code = (
        "from clawrl.router.grouped_route import FixtureGroupedToolRouter; "
        "print(FixtureGroupedToolRouter.resume(r'" + str(tmp_path) + "', 'grouped-route', 7, 'uid-001').content_hash)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    fresh = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert fresh.stdout.strip() == report.content_hash


def test_crash_before_report_is_restartable_and_production_blocked(tmp_path: Path) -> None:
    expected, pack = _inputs(tmp_path, "grouped-crash")
    with pytest.raises(InjectedGroupedRouteCrash):
        FixtureGroupedToolRouter.run(
            tmp_path,
            config=GroupedRouteConfig("grouped-crash", 7, "uid-001"),
            expected_set=expected,
            judge_pack=pack,
            crash_after_group=1,
        )
    report = FixtureGroupedToolRouter.run(
        tmp_path, config=GroupedRouteConfig("grouped-crash", 7, "uid-001"), expected_set=expected, judge_pack=pack
    )
    assert report.payload["production_readiness"] == "blocked"
    readiness = FixtureGroupedToolRouter.production_readiness(tmp_path / "production")
    assert readiness.payload["status"] == "blocked"
