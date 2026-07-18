"""Ticket 10 CFS capability and reward identity contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pytest

from clawrl.artifacts import ArtifactStore, JsonValue
from clawrl.training.reward_roundtrip import (
    FixtureCfsBackend,
    FixtureCfsConfig,
    ProductionCfsRewardConfig,
    RewardRoundtripWorkflow,
    RewardSlotKey,
)


def test_stable_reward_key_is_content_derived_and_closed() -> None:
    left = RewardSlotKey("run-10", 7, "uid-1", 3, "judge-pack-1")
    right = RewardSlotKey("run-10", 7, "uid-1", 3, "judge-pack-1")
    assert left.content_hash == right.content_hash
    assert left.payload() == {
        "global_step": 7,
        "judge_pack_id": "judge-pack-1",
        "rollout_index": 3,
        "run_id": "run-10",
        "uid": "uid-1",
    }
    with pytest.raises(ValueError):
        RewardSlotKey("../escape", 7, "uid-1", 3, "judge-pack-1")


@pytest.mark.parametrize(
    ("fault", "status", "winner_count", "partial"),
    [
        ("valid", "qualified", 1, False),
        ("overwrite", "blocked", 2, False),
        ("partial_visibility", "blocked", 1, True),
        ("cleanup_failure", "qualified", 1, False),
    ],
)
def test_probe_is_dedicated_no_listing_and_records_faults(
    tmp_path: Path, fault: str, status: str, winner_count: int, partial: bool
) -> None:
    config = FixtureCfsConfig(
        "backend-10",
        fault=cast(Literal["valid", "overwrite", "partial_visibility", "cleanup_failure"], fault),
    )
    evidence = FixtureCfsBackend(tmp_path, config).probe()
    assert evidence.payload["status"] == status
    assert evidence.payload["winner_count"] == winner_count
    assert evidence.payload["partial_payload_visible"] is partial
    assert evidence.payload["directory_listing_used"] is False
    client_results = evidence.payload["client_results"]
    candidate_payloads = evidence.payload["candidate_payloads"]
    assert isinstance(client_results, list) and len(client_results) == 2
    assert isinstance(candidate_payloads, list) and len(candidate_payloads) == 2
    assert all(isinstance(item, dict) for item in client_results)
    assert all(isinstance(item, dict) for item in candidate_payloads)
    client_rows = cast(list[dict[str, JsonValue]], client_results)
    candidate_rows = cast(list[dict[str, JsonValue]], candidate_payloads)
    assert len({str(item["mount_view_id"]) for item in client_rows}) == 2
    assert all(type(item["payload_size"]) is int and cast(int, item["payload_size"]) > 0 for item in candidate_rows)
    if partial:
        assert evidence.payload["consumer_status"] == "partial_or_invalid_payload_visible"
        assert evidence.payload["checksum_valid"] is False
        assert evidence.payload["payload_size"] == 0
    else:
        assert evidence.payload["consumer_status"] == "verified"
        assert evidence.payload["checksum_valid"] is True
        assert any(
            item["payload_hash"] == evidence.payload["payload_hash"]
            and item["payload_size"] == evidence.payload["payload_size"]
            for item in candidate_rows
        )
    if fault == "cleanup_failure":
        assert evidence.payload["cleanup_status"] == "failed"
        assert any(item["retained_after_cleanup"] is True for item in client_rows)
    assert FixtureCfsBackend(tmp_path, config).probe().content_hash == evidence.content_hash


def test_production_reward_roundtrip_is_blocked(tmp_path: Path) -> None:
    report = RewardRoundtripWorkflow.production_readiness(tmp_path, ProductionCfsRewardConfig())
    persisted = ArtifactStore(tmp_path).read(report.content_hash, expected_schema_name="ReadinessReport")
    assert persisted.payload["status"] == "blocked"
    assert persisted.payload["side_effects_permitted"] is False
    assert persisted.payload["phase"] == "TRAIN_35B"
    checks = persisted.payload["checks"]
    assert isinstance(checks, list)
    assert any(isinstance(item, dict) and item.get("code") == "REWARD_ATTEMPT_POLICY_UNAVAILABLE" for item in checks)
