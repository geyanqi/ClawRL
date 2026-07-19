"""Persistent Ticket 14 Router acceptance tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import ArtifactStore, JsonValue
from clawrl.router.trace_router import (
    FixtureTraceRouter,
    InjectedRouterCrash,
    RouterCapacityConfig,
    RouterFenceAuthority,
    TraceRouterConfig,
    TraceRouterError,
)
from clawrl.training.classic_identity import ClassicSourceRow
from clawrl.training.expected_trajectory_set import (
    ExpectedTrajectorySetConfig,
    ExpectedTrajectorySetWorkflow,
)


def _setup(root: Path, run_id: str, *, complete: bool = True, rollout_count: int = 128):
    config = ExpectedTrajectorySetConfig(run_id, 9, rollout_count)
    source = (ClassicSourceRow("trace-001", "uid-001", "judge-pack-001", "prompt"),)
    expected = ExpectedTrajectorySetWorkflow.freeze(root, config=config, sources=source)
    rows = ExpectedTrajectorySetWorkflow.generated_rows(config, source)
    for row in rows if complete else rows[:-1]:
        ExpectedTrajectorySetWorkflow.publish_slot(root, expected_set=expected, row=row)
    if complete:
        ExpectedTrajectorySetWorkflow.authorize_scoring(root, expected_set=expected, transport="classic")
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


def _fresh(root: Path, run_id: str) -> dict[str, object]:
    code = """
import json,sys
from clawrl.router.trace_router import FixtureTraceRouter
s=FixtureTraceRouter.resume(sys.argv[1],sys.argv[2],9,'uid-001')
print(json.dumps({'report':s.report.content_hash,'result_count':len(s.results),'session':s.session.payload['session_id'],'attempt':s.session.payload['attempt']},sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root), run_id], check=True, capture_output=True, text=True, env=env
    )
    return json.loads(completed.stdout)


def test_crash_restart_uses_new_session_and_committed_events_only(tmp_path: Path) -> None:
    expected, pack = _setup(tmp_path, "router-restart")
    config = TraceRouterConfig("router-restart", 9, "uid-001", 3, RouterCapacityConfig(3))
    with pytest.raises(InjectedRouterCrash):
        FixtureTraceRouter.run(tmp_path, config=config, expected_set=expected, judge_pack=pack, crash_after_wave=0)
    snapshot = FixtureTraceRouter.run(tmp_path, config=config, expected_set=expected, judge_pack=pack)
    assert snapshot.session.payload["attempt"] == 2
    assert snapshot.session.payload["hidden_memory_restored"] is False
    assert snapshot.waves[0].payload["session_hash"] != snapshot.session.content_hash
    assert len({item.payload["trajectory_manifest_hash"] for item in snapshot.results}) == 128
    fresh = _fresh(tmp_path, "router-restart")
    assert fresh["report"] == snapshot.report.content_hash
    assert fresh["result_count"] == 128
    assert fresh["attempt"] == 2


def test_crash_epoch_handover_uses_isolated_input_session_and_results(tmp_path: Path) -> None:
    expected, pack = _setup(tmp_path, "router-handover")
    epoch_one = TraceRouterConfig("router-handover", 9, "uid-001", 1, RouterCapacityConfig(3))
    with pytest.raises(InjectedRouterCrash):
        FixtureTraceRouter.run(
            tmp_path,
            config=epoch_one,
            expected_set=expected,
            judge_pack=pack,
            crash_after_wave=0,
        )
    authority = RouterFenceAuthority(tmp_path, "router-handover")
    authority.advance(2)
    snapshot = FixtureTraceRouter.run(
        tmp_path,
        config=TraceRouterConfig("router-handover", 9, "uid-001", 2, RouterCapacityConfig(3)),
        expected_set=expected,
        judge_pack=pack,
    )
    base = tmp_path / "trace-router-runs" / "router-handover" / "9" / "uid-001"
    assert (base / "epochs" / "1" / "input.ref").exists()
    assert (base / "epochs" / "2" / "input.ref").exists()
    assert all(result.payload["resolver_epoch"] == 2 for result in snapshot.results)
    assert snapshot.session.payload["attempt"] == 1
    assert _fresh(tmp_path, "router-handover")["report"] == snapshot.report.content_hash


def test_fence_concurrent_advance_is_monotonic_and_durable(tmp_path: Path) -> None:
    authority = RouterFenceAuthority(tmp_path, "router-fence-race")
    authority.advance(1)

    def advance(epoch: int) -> str:
        try:
            return authority.advance(epoch).content_hash
        except TraceRouterError:
            return "stale"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(advance, (2, 3)))
    current = authority.current()
    assert current.payload["epoch"] == 3
    assert outcomes.count("stale") <= 1
    assert (tmp_path / "router-resolver-epochs" / "router-fence-race" / "current.ref").read_bytes() == (
        f"{current.content_hash}\n".encode("ascii")
    )


def test_invalid_result_fails_closed_without_report(tmp_path: Path) -> None:
    expected, pack = _setup(tmp_path, "router-invalid")
    with pytest.raises(TraceRouterError, match="INVALID_JUDGE_RESULT"):
        FixtureTraceRouter.run(
            tmp_path,
            config=TraceRouterConfig("router-invalid", 9, "uid-001", 1, RouterCapacityConfig(5), fault_wave=0),
            expected_set=expected,
            judge_pack=pack,
        )
    assert not (tmp_path / "trace-router-runs" / "router-invalid" / "9" / "uid-001" / "report.ref").exists()


def test_incomplete_expected_set_blocks_before_session_or_result(tmp_path: Path) -> None:
    expected, pack = _setup(tmp_path, "router-incomplete", complete=False)
    with pytest.raises(TraceRouterError, match="EXPECTED_TRAJECTORY_SET_INCOMPLETE"):
        FixtureTraceRouter.run(
            tmp_path,
            config=TraceRouterConfig("router-incomplete", 9, "uid-001", 1, RouterCapacityConfig(5)),
            expected_set=expected,
            judge_pack=pack,
        )
    namespace = tmp_path / "trace-router-runs" / "router-incomplete" / "9" / "uid-001"
    assert not namespace.exists()


def test_complete_but_non_128_set_and_stale_fence_fail_before_router_side_effects(tmp_path: Path) -> None:
    expected, pack = _setup(tmp_path / "count", "router-count-64", rollout_count=64)
    with pytest.raises(TraceRouterError, match="exactly 128"):
        FixtureTraceRouter.run(
            tmp_path / "count",
            config=TraceRouterConfig("router-count-64", 9, "uid-001", 1, RouterCapacityConfig(5)),
            expected_set=expected,
            judge_pack=pack,
        )
    assert not (tmp_path / "count" / "trace-router-runs").exists()

    expected, pack = _setup(tmp_path / "fence", "router-stale-fence")
    RouterFenceAuthority(tmp_path / "fence", "router-stale-fence").advance(2)
    with pytest.raises(TraceRouterError, match="STALE_ROUTER_RESOLVER_EPOCH"):
        FixtureTraceRouter.run(
            tmp_path / "fence",
            config=TraceRouterConfig("router-stale-fence", 9, "uid-001", 1, RouterCapacityConfig(5)),
            expected_set=expected,
            judge_pack=pack,
        )
    assert not (tmp_path / "fence" / "trace-router-runs").exists()


def test_pack_mapping_and_crafted_result_refs_fail_closed(tmp_path: Path) -> None:
    expected, pack = _setup(tmp_path, "router-crafted")
    wrong_payload = dict(pack.payload)
    wrong_payload["trace_id"] = "trace-other"
    wrong_pack = ArtifactStore(tmp_path).put("JudgePack", "1.0.0", wrong_payload)
    config = TraceRouterConfig("router-crafted", 9, "uid-001", 1, RouterCapacityConfig(5))
    with pytest.raises(TraceRouterError, match="does not match"):
        FixtureTraceRouter.run(tmp_path, config=config, expected_set=expected, judge_pack=wrong_pack)
    assert not (tmp_path / "trace-router-runs").exists()

    snapshot = FixtureTraceRouter.run(tmp_path, config=config, expected_set=expected, judge_pack=pack)
    allowed = {
        item.content_hash for item in ExpectedTrajectorySetWorkflow.resume(tmp_path, "router-crafted", 9).manifests
    }
    for field, value in (
        ("evidence", []),
        ("failure_tags", [1]),
        ("turn_local_tie_groups", [["not-a-hash"]]),
        ("session_hash", "f" * 64),
        ("judge_pack_hash", "e" * 64),
    ):
        payload = dict(snapshot.results[0].payload)
        payload[field] = cast(JsonValue, value)
        forged = ArtifactStore(tmp_path).put("FencedJudgeResult", "1.0.0", payload)
        with pytest.raises(TraceRouterError, match="INVALID_JUDGE_RESULT"):
            FixtureTraceRouter._validate_result(
                forged,
                config,
                pack,
                snapshot.session,
                allowed,
                {cast(str, snapshot.results[0].payload["trajectory_manifest_hash"])},
            )

    payload = dict(snapshot.results[0].payload)
    payload["turn_local_tie_groups"] = [[cast(str, snapshot.results[1].payload["trajectory_manifest_hash"])]]
    forged_tie = ArtifactStore(tmp_path).put("FencedJudgeResult", "1.0.0", payload)
    with pytest.raises(TraceRouterError, match="INVALID_JUDGE_RESULT"):
        FixtureTraceRouter._validate_result(
            forged_tie,
            config,
            pack,
            snapshot.session,
            allowed,
            {cast(str, snapshot.results[0].payload["trajectory_manifest_hash"])},
        )


def test_forged_single_wave_report_fails_closed(tmp_path: Path) -> None:
    expected, pack = _setup(tmp_path, "router-forged-wave")
    snapshot = FixtureTraceRouter.run(
        tmp_path,
        config=TraceRouterConfig("router-forged-wave", 9, "uid-001", 1, RouterCapacityConfig(5)),
        expected_set=expected,
        judge_pack=pack,
    )
    forged_wave = ArtifactStore(tmp_path).put(
        "RouterWave",
        "1.0.0",
        {
            **snapshot.waves[0].payload,
            "active_subthreads": 5,
            "queued_item_count": 123,
            "result_hashes": [item.content_hash for item in snapshot.results],
        },
    )
    report_payload = dict(snapshot.report.payload)
    report_payload["wave_hashes"] = [forged_wave.content_hash]
    forged_report = ArtifactStore(tmp_path).put("TraceRouterReport", "1.0.0", report_payload)
    report_ref = tmp_path / "trace-router-runs" / "router-forged-wave" / "9" / "uid-001" / "epochs" / "1" / "report.ref"
    report_ref.write_text(f"{forged_report.content_hash}\n", encoding="ascii")
    with pytest.raises(TraceRouterError, match="wave cardinality"):
        FixtureTraceRouter.resume(tmp_path, "router-forged-wave", 9, "uid-001")
