"""Ticket 12 verl v1 reward identity and TransferQueue contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import ArtifactStore, JsonValue
from clawrl.training.classic_identity import ClassicSourceRow
from clawrl.training.v1_transfer_identity import (
    PersistentFixtureTransferQueue,
    ProductionV1IdentityConfig,
    TransferQueueSlot,
    V1IdentityConfig,
    V1RewardEnvelope,
    V1RewardIdentityError,
    V1RewardIdentityWorkflow,
    V1RewardManager,
)


def _sources() -> tuple[ClassicSourceRow, ...]:
    return (
        ClassicSourceRow("trace-001", "uid-001", "judge-pack-001", "raw prompt one"),
        ClassicSourceRow("trace-002", "uid-002", "judge-pack-002", "raw prompt two"),
    )


def test_v1_reward_envelope_preserves_wide_identity_beyond_raw_prompt() -> None:
    config = V1IdentityConfig("v1-contract-12", 17, rollout_count=3, initial_return_ordinals=(4, 0))
    envelopes = V1RewardManager.prefactor(config, _sources())
    assert [item.slot_ordinal for item in envelopes] == list(range(6))
    assert [item.global_step for item in envelopes] == [17] * 6
    for envelope in envelopes:
        payload = envelope.artifact_payload()
        assert set(payload) == {"global_step", "identity", "raw_prompt", "slot_ordinal"}
        identity = cast(dict[str, JsonValue], payload["identity"])
        assert set(identity) == {
            "expected_rollout_count",
            "global_step",
            "judge_pack_id",
            "rollout_index",
            "run_id",
            "trace_id",
            "uid",
        }
        assert identity["global_step"] == envelope.global_step == 17
        assert identity["expected_rollout_count"] == 3


def test_transfer_queue_replays_same_slot_and_rejects_duplicate_or_conflicting_content(tmp_path: Path) -> None:
    config = V1IdentityConfig("v1-queue-contract-12", 3, rollout_count=2, initial_return_ordinals=(3, 0))
    envelope = V1RewardManager.prefactor(config, _sources())[0]
    slot = TransferQueueSlot.from_envelope(envelope)
    queue = PersistentFixtureTransferQueue(tmp_path, config.run_id)
    dispatch = queue.dispatch(slot)
    assert queue.dispatch(slot).content_hash == dispatch.content_hash
    returned = queue.commit_return(dispatch, "fixture response")
    assert queue.commit_return(dispatch, "fixture response").content_hash == returned.content_hash

    other = V1RewardManager.prefactor(config, _sources())[1]
    conflicting_slot = TransferQueueSlot(
        0,
        V1RewardEnvelope(0, other.identity, other.raw_prompt, other.global_step),
    )
    with pytest.raises(V1RewardIdentityError, match="TRANSFER_SLOT_CONFLICT"):
        queue.dispatch(conflicting_slot)
    with pytest.raises(V1RewardIdentityError, match="TRANSFER_RETURN_CONFLICT"):
        queue.commit_return(dispatch, "different response")


def test_public_reissue_rejects_forged_outstanding_checkpoint(tmp_path: Path) -> None:
    config = V1IdentityConfig("v1-forged-reissue-12", 4, rollout_count=2, initial_return_ordinals=(3, 0))
    envelope = V1RewardManager.prefactor(config, _sources())[0]
    queue = PersistentFixtureTransferQueue(tmp_path, config.run_id)
    dispatch = queue.dispatch(TransferQueueSlot.from_envelope(envelope))
    forged = ArtifactStore(tmp_path).put(
        "V1TransferQueueCheckpoint",
        "1.0.0",
        {
            "dispatch_hashes_by_slot": [],
            "outstanding_slot_ordinals": [0],
            "return_hashes_by_slot": [],
            "returned_slot_ordinals": [],
            "run_id": config.run_id,
            "status": "durable",
        },
    )
    with pytest.raises(V1RewardIdentityError, match="checkpoint"):
        queue.reissue(dispatch, forged)


@pytest.mark.parametrize("phase", ["TRAIN_35B", "TRAIN_122B"])
def test_real_v1_modes_remain_train_blocked_without_verifiable_runtime(tmp_path: Path, phase: str) -> None:
    report = V1RewardIdentityWorkflow.production_readiness(
        tmp_path / phase,
        ProductionV1IdentityConfig(
            phase=cast(str, phase),
            variant="v1/colocated-async-transfer-queue",
        ),
    )
    checks = cast(list[dict[str, JsonValue]], report.payload["checks"])
    assert report.payload["status"] == "blocked"
    assert report.payload["phase"] == phase
    assert report.payload["side_effects_permitted"] is False
    assert report.payload["cfs_integration_claimed"] is False
    assert report.payload["production_smoke"] is False
    assert "REAL_VERL_V1_RUNTIME_UNVERIFIED" in {item["code"] for item in checks}
    ArtifactStore(tmp_path / phase).read(report.content_hash, expected_schema_name="ReadinessReport")
