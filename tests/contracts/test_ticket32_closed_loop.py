"""Focused fixture closed-loop composition contracts."""

from pathlib import Path
from typing import cast

from clawrl.artifacts import ArtifactStore
from clawrl.fixtures.closed_loop import FixtureClosedLoopConfig, FixtureClosedLoopWorkflow


def test_fixture_closed_loop_has_complete_lineage_and_reward_cardinality(tmp_path: Path) -> None:
    snapshot = FixtureClosedLoopWorkflow.run_fixture(tmp_path)
    assert snapshot.status == "succeeded"
    assert snapshot.phase_matrix == {
        "DATA_INGEST": "green",
        "JUDGE_CERTIFY": "green",
        "TRAIN_35B": "green",
        "TRAIN_122B": "green",
        "FINAL_EVAL": "green",
    }
    assert len(snapshot.rewards) == 16 * 128
    assert snapshot.run_record is not None
    assert snapshot.decision_record is not None
    assert snapshot.transfer_candidate is not None
    assert snapshot.final_eval is not None
    run_record = snapshot.run_record
    transfer_candidate = snapshot.transfer_candidate
    final_eval = snapshot.final_eval
    reward_manifest = snapshot.reward_manifest
    assert reward_manifest is not None
    assert final_eval.payload["trained_wins"] == 60
    assert transfer_candidate.payload["run_record_hash"] == run_record.content_hash
    assert final_eval.payload["run_record_hash"] == run_record.content_hash
    assert run_record.payload["reward_hash"] == reward_manifest.content_hash


def test_first_missing_or_conflicting_artifact_stops_before_downstream_side_effects(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    dataset = store.put("DatasetVersion", "1.0.0", {"purpose": "training_allowed", "trace_count": 100})
    missing_bundle = "0" * 64
    config = FixtureClosedLoopConfig("ticket32-fail", dataset.content_hash, missing_bundle, "1" * 64)
    snapshot = FixtureClosedLoopWorkflow.run(tmp_path, config=config)
    assert snapshot.status == "failed"
    assert snapshot.phase == "DATA_INGEST"
    assert snapshot.rewards == ()
    assert snapshot.run_record is None
    assert snapshot.transfer_candidate is None
    assert snapshot.final_eval is None
    assert "scorer" not in snapshot.side_effects
    assert "optimizer" not in snapshot.side_effects
    assert "cluster" not in snapshot.side_effects


def test_phase_matrix_does_not_inherit_later_authority_from_earlier_green_gates(tmp_path: Path) -> None:
    snapshot = FixtureClosedLoopWorkflow.run_fixture(tmp_path)
    assert snapshot.run_record is not None
    run_record = snapshot.run_record
    config = FixtureClosedLoopConfig(
        "fixture-closed-loop",
        cast(str, run_record.payload["dataset_version_hash"]),
        cast(str, run_record.payload["judge_bundle_hash"]),
        cast(str, run_record.payload["experiment_spec_hash"]),
    )
    matrix = FixtureClosedLoopWorkflow.phase_matrix(tmp_path, config=config)
    assert matrix["DATA_INGEST"] == "green"
    assert matrix["JUDGE_CERTIFY"] == "green"
    assert matrix["TRAIN_35B"] == "green"
    assert matrix["TRAIN_122B"] == "green"
    assert matrix["FINAL_EVAL"] == "green"
