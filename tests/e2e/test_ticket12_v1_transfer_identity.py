"""Persistent Ticket 12 v1 asynchronous TransferQueue acceptance tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import ArtifactStore, JsonValue
from clawrl.training.classic_identity import ClassicSourceRow
from clawrl.training.v1_transfer_identity import (
    InjectedV1ControllerCrash,
    V1IdentityConfig,
    V1RewardIdentityError,
    V1RewardIdentityWorkflow,
)


def _sources(suffix: str = "") -> tuple[ClassicSourceRow, ...]:
    return (
        ClassicSourceRow("trace-001", "uid-001", "judge-pack-001", f"raw prompt one{suffix}"),
        ClassicSourceRow("trace-002", "uid-002", "judge-pack-002", f"raw prompt two{suffix}"),
    )


def _fresh_resume(root: Path, run_id: str) -> dict[str, object]:
    code = """
import json,sys
from clawrl.training.v1_transfer_identity import V1RewardIdentityWorkflow
s=V1RewardIdentityWorkflow.resume(sys.argv[1],sys.argv[2])
print(json.dumps({
    'checkpoint': s.checkpoint.content_hash,
    'dump': s.dump.content_hash,
    'report': s.report.content_hash,
    'return_count': len(s.returns),
    'slot_ordinals': [row['slot_ordinal'] for row in s.dump.payload['rows']],
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root), run_id],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    decoded = json.loads(completed.stdout)
    assert isinstance(decoded, dict)
    return decoded


def test_out_of_order_return_checkpoint_reissue_and_fresh_resume_keep_original_ordinals(tmp_path: Path) -> None:
    config = V1IdentityConfig("v1-reissue-12", 19, rollout_count=3, initial_return_ordinals=(4, 0))
    with pytest.raises(InjectedV1ControllerCrash, match="checkpoint"):
        V1RewardIdentityWorkflow.run(
            tmp_path,
            config=config,
            sources=_sources(),
            crash_after_checkpoint=True,
        )
    snapshot = V1RewardIdentityWorkflow.resume(tmp_path, config.run_id)
    fresh = _fresh_resume(tmp_path, config.run_id)
    assert fresh["report"] == snapshot.report.content_hash
    assert fresh["dump"] == snapshot.dump.content_hash
    assert fresh["checkpoint"] == snapshot.checkpoint.content_hash
    assert fresh["slot_ordinals"] == list(range(6))
    assert fresh["return_count"] == 6

    arrivals = [item.payload["slot_ordinal"] for item in snapshot.returns]
    assert arrivals[:2] == [4, 0]
    assert arrivals != sorted(cast(list[int], arrivals))
    checkpoint = snapshot.checkpoint.payload
    assert checkpoint["returned_slot_ordinals"] == [0, 4]
    assert checkpoint["outstanding_slot_ordinals"] == [1, 2, 3, 5]
    assert [item.payload["slot_ordinal"] for item in snapshot.reissues] == [5, 3, 2, 1]

    rows = cast(list[dict[str, JsonValue]], snapshot.dump.payload["rows"])
    assert [row["slot_ordinal"] for row in rows] == list(range(6))
    for row in rows:
        identity = cast(dict[str, JsonValue], row["identity"])
        assert row["global_step"] == identity["global_step"] == 19
        assert identity["run_id"] == config.run_id
        assert identity["expected_rollout_count"] == 3
        assert isinstance(row["trajectory_hash"], str)
    assert snapshot.report.payload["classic_compatibility_preserved"] is True
    assert snapshot.report.payload["cfs_integration_claimed"] is False
    assert snapshot.report.payload["production_smoke"] is False


def test_same_run_replay_is_idempotent_and_changed_input_fails_closed(tmp_path: Path) -> None:
    config = V1IdentityConfig("v1-conflict-12", 5, rollout_count=2, initial_return_ordinals=(3, 0))
    first = V1RewardIdentityWorkflow.run(tmp_path, config=config, sources=_sources())
    replay = V1RewardIdentityWorkflow.run(tmp_path, config=config, sources=_sources())
    assert replay.report.content_hash == first.report.content_hash
    with pytest.raises(V1RewardIdentityError, match="V1_RUN_INPUT_CONFLICT"):
        V1RewardIdentityWorkflow.run(tmp_path, config=config, sources=_sources(" changed"))
    assert _fresh_resume(tmp_path, config.run_id)["report"] == first.report.content_hash


@pytest.mark.parametrize("fault_kind", ["missing_identity", "ordinal_mismatch", "global_step_mismatch"])
def test_corrupt_return_fails_closed_without_dump(tmp_path: Path, fault_kind: str) -> None:
    config = V1IdentityConfig(
        f"v1-fault-{fault_kind}",
        8,
        rollout_count=2,
        initial_return_ordinals=(3, 0),
        fault_slot_ordinal=2,
        fault_kind=fault_kind,
    )
    with pytest.raises(V1RewardIdentityError, match="V1_TRANSFER_IDENTITY_CORRUPTION"):
        V1RewardIdentityWorkflow.run(tmp_path, config=config, sources=_sources())
    schemas = [json.loads(path.read_text())["schema_name"] for path in (tmp_path / "artifacts").glob("*.json")]
    assert "V1TransferIdentityFailure" in schemas
    assert "V1RewardTrajectoryDump" not in schemas
    assert "V1RewardIdentityReport" not in schemas


def test_resume_rejects_forged_dump_ordinal_even_with_coherent_report_reference(tmp_path: Path) -> None:
    config = V1IdentityConfig("v1-forged-12", 6, rollout_count=2, initial_return_ordinals=(3, 0))
    snapshot = V1RewardIdentityWorkflow.run(tmp_path, config=config, sources=_sources())
    store = ArtifactStore(tmp_path)
    dump_payload = cast(dict[str, JsonValue], json.loads(json.dumps(snapshot.dump.payload)))
    rows = cast(list[dict[str, JsonValue]], dump_payload["rows"])
    rows[0]["slot_ordinal"] = 1
    forged_dump = store.put("V1RewardTrajectoryDump", "1.0.0", dump_payload)
    report_payload = cast(dict[str, JsonValue], json.loads(json.dumps(snapshot.report.payload)))
    report_payload["dump_hash"] = forged_dump.content_hash
    forged_report = store.put("V1RewardIdentityReport", "1.0.0", report_payload)
    report_ref = tmp_path / "v1-transfer-runs" / config.run_id / "report.ref"
    report_ref.write_text(f"{forged_report.content_hash}\n", encoding="ascii")
    with pytest.raises(V1RewardIdentityError, match="dump|ordinal"):
        V1RewardIdentityWorkflow.resume(tmp_path, config.run_id)


def test_resume_rejects_forged_checkpoint_slot_hash_mapping(tmp_path: Path) -> None:
    config = V1IdentityConfig("v1-forged-checkpoint-12", 7, rollout_count=2, initial_return_ordinals=(3, 0))
    snapshot = V1RewardIdentityWorkflow.run(tmp_path, config=config, sources=_sources())
    store = ArtifactStore(tmp_path)
    checkpoint_payload = cast(dict[str, JsonValue], json.loads(json.dumps(snapshot.checkpoint.payload)))
    dispatches = cast(list[dict[str, JsonValue]], checkpoint_payload["dispatch_hashes_by_slot"])
    dispatches[0]["dispatch_hash"] = cast(str, dispatches[1]["dispatch_hash"])
    forged_checkpoint = store.put("V1TransferQueueCheckpoint", "1.0.0", checkpoint_payload)
    dump_payload = cast(dict[str, JsonValue], json.loads(json.dumps(snapshot.dump.payload)))
    dump_payload["checkpoint_hash"] = forged_checkpoint.content_hash
    forged_dump = store.put("V1RewardTrajectoryDump", "1.0.0", dump_payload)
    report_payload = cast(dict[str, JsonValue], json.loads(json.dumps(snapshot.report.payload)))
    report_payload["checkpoint_hash"] = forged_checkpoint.content_hash
    report_payload["dump_hash"] = forged_dump.content_hash
    forged_report = store.put("V1RewardIdentityReport", "1.0.0", report_payload)
    queue_checkpoint_ref = tmp_path / "fixture-transfer-queue" / config.run_id / "checkpoint.ref"
    workflow_checkpoint_ref = tmp_path / "v1-transfer-runs" / config.run_id / "checkpoint.ref"
    report_ref = tmp_path / "v1-transfer-runs" / config.run_id / "report.ref"
    for ref, content_hash in (
        (queue_checkpoint_ref, forged_checkpoint.content_hash),
        (workflow_checkpoint_ref, forged_checkpoint.content_hash),
        (report_ref, forged_report.content_hash),
    ):
        ref.write_text(f"{content_hash}\n", encoding="ascii")

    with pytest.raises(V1RewardIdentityError, match="checkpoint slot/hash mapping"):
        V1RewardIdentityWorkflow.resume(tmp_path, config.run_id)


def test_fresh_resume_rejects_slot_dispatch_ref_rebound_to_another_valid_slot(tmp_path: Path) -> None:
    config = V1IdentityConfig("v1-slot-ref-tamper-12", 10, rollout_count=2, initial_return_ordinals=(3, 0))
    snapshot = V1RewardIdentityWorkflow.run(tmp_path, config=config, sources=_sources())
    slot_zero_ref = tmp_path / "fixture-transfer-queue" / config.run_id / "slots" / "0" / "dispatch.ref"
    slot_zero_ref.write_text(f"{snapshot.dispatches[1].content_hash}\n", encoding="ascii")

    with pytest.raises(V1RewardIdentityError, match="dispatch ref|slot ref"):
        V1RewardIdentityWorkflow.resume(tmp_path, config.run_id)
