"""Ticket 16 classic RewardLoop grouped calibrated-scalar acceptance tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import Artifact, ArtifactStore
from clawrl.router.trace_router import RouterCapacityConfig
from clawrl.training.classic_grouped_reward import (
    ClassicGroupedRewardConfig,
    ClassicGroupedRewardError,
    ClassicGroupedRewardWorkflow,
    InjectedClassicGroupedRewardCrash,
)
from clawrl.training.classic_identity import ClassicSourceRow
from clawrl.training.expected_trajectory_set import ExpectedTrajectorySetConfig, ExpectedTrajectorySetWorkflow


def _setup(root: Path, run_id: str) -> tuple[ClassicGroupedRewardConfig, Artifact, Artifact, Artifact]:
    store = ArtifactStore(root)
    source = (ClassicSourceRow("trace-16", "uid-16", "pack-16", "prompt"),)
    expected_config = ExpectedTrajectorySetConfig(run_id, 16, 128)
    expected = ExpectedTrajectorySetWorkflow.freeze(root, config=expected_config, sources=source)
    for row in ExpectedTrajectorySetWorkflow.generated_rows(expected_config, source):
        ExpectedTrajectorySetWorkflow.publish_slot(root, expected_set=expected, row=row)
    ExpectedTrajectorySetWorkflow.authorize_scoring(root, expected_set=expected, transport="classic")
    pack = store.put(
        "JudgePack",
        "1.0.0",
        {
            "aggregation": "calibrated_scalar",
            "algorithm_contract": "grpo-calibrated-scalar-v1",
            "items_per_turn": 8,
            "judge_pack_id": "pack-16",
            "judge_pack_version": "1.0.0",
            "prompt_hash": "a" * 64,
            "scalarizer_hash": "b" * 64,
            "scalarizer_version": "1.0.0",
            "scorer_tier": "luna",
            "status": "certified",
            "trace_id": "trace-16",
            "uid": "uid-16",
        },
    )
    spec = store.put(
        "ExperimentSpec",
        "1.0.0",
        {
            "aggregation": "calibrated_scalar",
            "dataset_version_hash": "c" * 64,
            "dataset_version_id": "dataset-16",
            "experiment_id": "experiment-16",
            "judge_bundle_hash": "d" * 64,
            "trace_set_hash": "e" * 64,
        },
    )
    config = ClassicGroupedRewardConfig(
        run_id,
        16,
        "uid-16",
        spec.content_hash,
        chunk_size=13,
        capacity=RouterCapacityConfig(5),
        prompt_hash="a" * 64,
        scalarizer_hash="b" * 64,
        scalarizer_version="1.0.0",
        algorithm_contract="grpo-calibrated-scalar-v1",
    )
    return config, expected, pack, spec


def _fresh(root: Path, config: ClassicGroupedRewardConfig) -> dict[str, object]:
    code = """
import json,sys
from clawrl.training.classic_grouped_reward import ClassicGroupedRewardConfig,ClassicGroupedRewardWorkflow
c=ClassicGroupedRewardConfig(sys.argv[2],16,"uid-16",sys.argv[3],chunk_size=13)
s=ClassicGroupedRewardWorkflow.resume(sys.argv[1],c)
print(json.dumps({'report':s.report.content_hash,'count':len(s.rewards),'tensor':len(s.trainer_output.payload['reward_micros'])},sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-c", code, str(root), config.run_id, config.experiment_spec_hash],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return cast(dict[str, object], json.loads(result.stdout))


def test_async_group_barrier_worker_reorder_and_fresh_resume(tmp_path: Path) -> None:
    config, expected, pack, spec = _setup(tmp_path, "classic-16")
    snapshot = ClassicGroupedRewardWorkflow.run(
        tmp_path, config=config, expected_set=expected, judge_pack=pack, experiment_spec=spec
    )
    assert len(snapshot.rewards) == 128
    assert snapshot.trainer_output.payload["reward_count"] == 128
    extra = cast(list[dict[str, object]], snapshot.trainer_output.payload["extra_info"])
    assert len(extra) == 128
    assert {item["run_id"] for item in extra} == {config.run_id}
    assert {item["global_step"] for item in extra} == {config.global_step}
    assert {item["uid"] for item in extra} == {config.uid}
    assert [item["rollout_index"] for item in extra] == list(range(128))
    assert all(
        {"dimensions", "confidence_basis_points", "failure_tags", "evidence", "turn_local_tie_groups",
         "judge_pack_hash", "judge_pack_version", "scalarizer_hash", "scalarizer_version", "session_hash",
         "thread_ref", "turn_ref", "trajectory_manifest_hash", "judge_result_hash", "trace_id"}
        <= set(item)
        for item in extra
    )
    assert [item.payload["rollout_index"] for item in snapshot.rewards] == list(range(128))
    assert _fresh(tmp_path, config)["report"] == snapshot.report.content_hash


def test_missing_nan_and_version_fail_closed_without_trainer_tensor(tmp_path: Path) -> None:
    for kind in ("missing", "nan", "version"):
        root = tmp_path / kind
        config, expected, pack, spec = _setup(root, f"classic-16-{kind}")
        bad = ClassicGroupedRewardConfig(
            config.run_id,
            config.global_step,
            config.uid,
            config.experiment_spec_hash,
            resolver_epoch=config.resolver_epoch,
            chunk_size=config.chunk_size,
            capacity=config.capacity,
            prompt_hash=config.prompt_hash,
            scalarizer_hash=config.scalarizer_hash,
            scalarizer_version=config.scalarizer_version,
            algorithm_contract=config.algorithm_contract,
            fault_kind=kind,
            fault_index=3,
        )
        with pytest.raises(ClassicGroupedRewardError):
            ClassicGroupedRewardWorkflow.run(
                root, config=bad, expected_set=expected, judge_pack=pack, experiment_spec=spec
            )
        assert not (root / "classic-grouped-reward" / bad.run_id / "16" / "uid-16" / "report.ref").exists()


def test_reserved_strategy_and_frozen_mutation_rejected_before_router(tmp_path: Path) -> None:
    config, expected, pack, spec = _setup(tmp_path, "classic-16-mutation")
    with pytest.raises(ClassicGroupedRewardError, match="UNSUPPORTED_REWARD_STRATEGY"):
        ClassicGroupedRewardWorkflow.run(
            tmp_path,
            config=ClassicGroupedRewardConfig(
                config.run_id,
                config.global_step,
                config.uid,
                config.experiment_spec_hash,
                aggregation="hierarchical_rank",
            ),
            expected_set=expected,
            judge_pack=pack,
            experiment_spec=spec,
        )
    with pytest.raises(ClassicGroupedRewardError, match="mutation"):
        ClassicGroupedRewardWorkflow.run(
            tmp_path,
            config=ClassicGroupedRewardConfig(
                config.run_id, config.global_step, config.uid, config.experiment_spec_hash, prompt_hash="f" * 64
            ),
            expected_set=expected,
            judge_pack=pack,
            experiment_spec=spec,
        )
    assert not (tmp_path / "classic-grouped-reward").exists()


def test_crash_after_request_publication_resumes_idempotently(tmp_path: Path) -> None:
    config, expected, pack, spec = _setup(tmp_path, "classic-16-crash")
    with pytest.raises(InjectedClassicGroupedRewardCrash):
        ClassicGroupedRewardWorkflow.run(
            tmp_path,
            config=config,
            expected_set=expected,
            judge_pack=pack,
            experiment_spec=spec,
            crash_after_requests=1,
        )
    snapshot = ClassicGroupedRewardWorkflow.run(
        tmp_path, config=config, expected_set=expected, judge_pack=pack, experiment_spec=spec
    )
    assert len(snapshot.rewards) == 128


def test_production_readiness_is_blocked_without_real_verl_or_cfs(tmp_path: Path) -> None:
    readiness = ClassicGroupedRewardWorkflow.production_readiness(tmp_path)
    assert readiness.payload["status"] == "blocked"
    assert readiness.payload["side_effects_permitted"] is False
    assert {item["status"] for item in cast(list[dict[str, str]], readiness.payload["checks"])} == {"blocked"}
