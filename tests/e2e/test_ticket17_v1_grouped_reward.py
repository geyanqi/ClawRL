"""Ticket 17 v1/TransferQueue grouped reward acceptance tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import Artifact, ArtifactStore
from clawrl.router.trace_router import RouterCapacityConfig
from clawrl.training.classic_identity import ClassicSourceRow
from clawrl.training.expected_trajectory_set import ExpectedTrajectorySetConfig, ExpectedTrajectorySetWorkflow
from clawrl.training.v1_grouped_reward import (
    InjectedV1GroupedRewardCrash,
    V1GroupedRewardConfig,
    V1GroupedRewardError,
    V1GroupedRewardWorkflow,
)


def _setup(root: Path, run_id: str) -> tuple[V1GroupedRewardConfig, Artifact, Artifact, Artifact]:
    store = ArtifactStore(root)
    source = (ClassicSourceRow("trace-17", "uid-17", "pack-17", "prompt"),)
    expected_config = ExpectedTrajectorySetConfig(run_id, 17, 128)
    expected = ExpectedTrajectorySetWorkflow.freeze(root, config=expected_config, sources=source)
    for row in ExpectedTrajectorySetWorkflow.generated_rows(expected_config, source):
        ExpectedTrajectorySetWorkflow.publish_slot(root, expected_set=expected, row=row)
    ExpectedTrajectorySetWorkflow.authorize_scoring(root, expected_set=expected, transport="v1")
    pack = store.put(
        "JudgePack",
        "1.0.0",
        {
            "aggregation": "calibrated_scalar",
            "algorithm_contract": "grpo-calibrated-scalar-v1",
            "items_per_turn": 8,
            "judge_pack_id": "pack-17",
            "judge_pack_version": "1.0.0",
            "prompt_hash": "a" * 64,
            "scalarizer_hash": "b" * 64,
            "scalarizer_version": "1.0.0",
            "scorer_tier": "luna",
            "status": "certified",
            "trace_id": "trace-17",
            "uid": "uid-17",
        },
    )
    spec = store.put(
        "ExperimentSpec",
        "1.0.0",
        {
            "aggregation": "calibrated_scalar",
            "dataset_version_hash": "c" * 64,
            "dataset_version_id": "dataset-17",
            "experiment_id": "experiment-17",
            "judge_bundle_hash": "d" * 64,
            "trace_set_hash": "e" * 64,
        },
    )
    config = V1GroupedRewardConfig(
        run_id,
        17,
        "uid-17",
        spec.content_hash,
        chunk_size=13,
        capacity=RouterCapacityConfig(5),
        prompt_hash="a" * 64,
        scalarizer_hash="b" * 64,
        scalarizer_version="1.0.0",
        algorithm_contract="grpo-calibrated-scalar-v1",
    )
    return config, expected, pack, spec


def test_v1_dispatches_full_group_and_preserves_trainer_identity(tmp_path: Path) -> None:
    config, expected, pack, spec = _setup(tmp_path, "v1-17")
    snapshot = V1GroupedRewardWorkflow.run(
        tmp_path, config=config, expected_set=expected, judge_pack=pack, experiment_spec=spec
    )
    assert len(snapshot.rewards) == 128
    assert snapshot.trainer_output.payload["reward_count"] == 128
    extra = cast(list[dict[str, object]], snapshot.trainer_output.payload["extra_info"])
    assert [item["rollout_index"] for item in extra] == list(range(128))
    assert all(
        {
            "dimensions",
            "confidence_basis_points",
            "failure_tags",
            "evidence",
            "turn_local_tie_groups",
            "judge_pack_hash",
            "judge_pack_version",
            "scalarizer_hash",
            "scalarizer_version",
            "session_hash",
            "thread_ref",
            "turn_ref",
            "trajectory_manifest_hash",
            "trace_id",
        }
        <= set(item)
        for item in extra
    )
    assert len(cast(list[object], snapshot.dump.payload["dispatch_hashes"])) == 128
    assert len(cast(list[object], snapshot.dump.payload["return_hashes"])) == 128


def test_v1_checkpoint_crash_reissues_and_resumes(tmp_path: Path) -> None:
    config, expected, pack, spec = _setup(tmp_path, "v1-17-crash")
    with pytest.raises(InjectedV1GroupedRewardCrash):
        V1GroupedRewardWorkflow.run(
            tmp_path,
            config=config,
            expected_set=expected,
            judge_pack=pack,
            experiment_spec=spec,
            crash_after_checkpoint=True,
        )
    snapshot = V1GroupedRewardWorkflow.run(
        tmp_path, config=config, expected_set=expected, judge_pack=pack, experiment_spec=spec
    )
    assert len(snapshot.rewards) == 128
    assert len(cast(list[object], snapshot.dump.payload["reissue_hashes"])) == 127


def test_v1_reserved_strategy_rejected_before_router(tmp_path: Path) -> None:
    config, expected, pack, spec = _setup(tmp_path, "v1-17-reserved")
    bad = V1GroupedRewardConfig(
        config.run_id,
        config.global_step,
        config.uid,
        config.experiment_spec_hash,
        aggregation="hierarchical_rank",
    )
    with pytest.raises(V1GroupedRewardError, match="UNSUPPORTED_REWARD_STRATEGY"):
        V1GroupedRewardWorkflow.run(tmp_path, config=bad, expected_set=expected, judge_pack=pack, experiment_spec=spec)
    assert not (tmp_path / "v1-grouped-reward").exists()


def test_v1_missing_slot_never_emits_tensor(tmp_path: Path) -> None:
    config, expected, pack, spec = _setup(tmp_path, "v1-17-missing")
    bad = V1GroupedRewardConfig(
        config.run_id,
        config.global_step,
        config.uid,
        config.experiment_spec_hash,
        prompt_hash=config.prompt_hash,
        scalarizer_hash=config.scalarizer_hash,
        scalarizer_version=config.scalarizer_version,
        algorithm_contract=config.algorithm_contract,
        fault_kind="missing",
        fault_index=3,
    )
    with pytest.raises(V1GroupedRewardError):
        V1GroupedRewardWorkflow.run(tmp_path, config=bad, expected_set=expected, judge_pack=pack, experiment_spec=spec)
    assert not (tmp_path / "v1-grouped-reward" / bad.run_id / "17" / "uid-17" / "report.ref").exists()
