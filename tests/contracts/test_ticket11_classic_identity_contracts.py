"""Ticket 11 classic verl identity contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import ArtifactStore, JsonValue
from clawrl.training.classic_identity import (
    ClassicBatch,
    ClassicIdentityConfig,
    ClassicIdentityError,
    ClassicIdentityWorkflow,
    ClassicRewardManager,
    ClassicSourceRow,
    ClassicTrajectoryIdentity,
    ProductionClassicIdentityConfig,
)


def _sources() -> tuple[ClassicSourceRow, ...]:
    return (
        ClassicSourceRow("trace-001", "uid-001", "judge-pack-001", "prompt one"),
        ClassicSourceRow("trace-002", "uid-002", "judge-pack-002", "prompt two"),
    )


def test_wide_prefactor_assigns_closed_stable_identity_before_generation() -> None:
    config = ClassicIdentityConfig("classic-run-11", 7, rollout_count=3, chunk_size=4)
    batch = ClassicRewardManager().prefactor(config, _sources())
    assert len(batch.rows) == 6
    assert batch.global_steps == (7, 7, 7, 7, 7, 7)
    for uid in ("uid-001", "uid-002"):
        identities = [row.identity for row in batch.rows if row.identity.uid == uid]
        assert [identity.rollout_index for identity in identities] == [0, 1, 2]
        assert all(identity.expected_rollout_count == 3 for identity in identities)
        assert all(identity.run_id == "classic-run-11" and identity.global_step == 7 for identity in identities)
    payload = batch.rows[0].identity.artifact_payload()
    assert set(payload) == {
        "expected_rollout_count",
        "global_step",
        "judge_pack_id",
        "rollout_index",
        "run_id",
        "trace_id",
        "uid",
    }
    assert ClassicTrajectoryIdentity.from_mapping(payload).content_hash == batch.rows[0].identity.content_hash


def test_batch_rejects_missing_duplicate_and_batch_step_mismatch() -> None:
    config = ClassicIdentityConfig("classic-run-11", 7, rollout_count=2, chunk_size=3)
    valid = ClassicRewardManager().prefactor(config, _sources())
    payload = valid.artifact_payload()
    rows_payload = cast(list[JsonValue], payload["rows"])

    missing = dict(payload)
    missing["rows"] = rows_payload[:-1]
    missing["global_steps"] = cast(list[JsonValue], list(valid.global_steps)[:-1])
    with pytest.raises(ClassicIdentityError, match="rollout index set"):
        ClassicBatch.from_mapping(missing)

    duplicate = dict(payload)
    duplicate_rows = list(rows_payload)
    duplicate_rows[-1] = duplicate_rows[-2]
    duplicate["rows"] = duplicate_rows
    with pytest.raises(ClassicIdentityError, match="duplicate|rollout index set"):
        ClassicBatch.from_mapping(duplicate)

    mismatch = dict(payload)
    mismatch["global_steps"] = cast(list[JsonValue], [7, 7, 7, 8])
    with pytest.raises(ClassicIdentityError, match="batch global_steps"):
        ClassicBatch.from_mapping(mismatch)


def test_unverified_classic_variant_and_real_verl_remain_train_blocked(tmp_path: Path) -> None:
    unverified = ClassicIdentityWorkflow.production_readiness(
        tmp_path / "unverified", ProductionClassicIdentityConfig("classic/v2-unknown")
    )
    verified = ClassicIdentityWorkflow.production_readiness(
        tmp_path / "verified", ProductionClassicIdentityConfig("classic/v1-wide-prefactor")
    )
    assert unverified.payload["status"] == verified.payload["status"] == "blocked"
    assert unverified.payload["side_effects_permitted"] is False
    unverified_checks = cast(list[dict[str, JsonValue]], unverified.payload["checks"])
    verified_checks = cast(list[dict[str, JsonValue]], verified.payload["checks"])
    assert "UNVERIFIED_CLASSIC_VARIANT" in {item["code"] for item in unverified_checks}
    assert "REAL_VERL_RUNTIME_UNAVAILABLE" in {item["code"] for item in verified_checks}
    assert verified.payload["cfs_integration_claimed"] is False
    ArtifactStore(tmp_path / "verified").read(verified.content_hash, expected_schema_name="ReadinessReport")
