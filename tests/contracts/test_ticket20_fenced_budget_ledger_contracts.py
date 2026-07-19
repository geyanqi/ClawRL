"""Ticket 20 fenced BudgetLedger contracts."""

from pathlib import Path

import pytest

from clawrl.governor.budget_ledger import (
    BudgetLedgerError,
    BudgetLimits,
    BudgetUsage,
    FixtureBudgetLedger,
    ProductionBudgetLedgerConfig,
)


def _limits(**overrides: int) -> BudgetLimits:
    values = {name: 100 for name in BudgetUsage.dimension_names()}
    values["gpu_count"] = 96
    values.update(overrides)
    return BudgetLimits.from_mapping(values)


def test_missing_or_negative_budget_dimension_means_zero_permission() -> None:
    limits = BudgetLimits.from_mapping({"gpu_count": 8, "gpu_hours_millis": -1})
    assert limits.values["gpu_count"] == 8
    assert limits.values["gpu_hours_millis"] == 0
    assert all(
        limits.values[name] == 0
        for name in BudgetUsage.dimension_names()
        if name not in {"gpu_count", "gpu_hours_millis"}
    )


def test_reserve_precedes_harness_allow_and_split_children_cannot_bypass_total(tmp_path: Path) -> None:
    ledger = FixtureBudgetLedger.create(tmp_path, "campaign-20", _limits(jobs=2, gpu_hours_millis=10), epoch=1)
    usage = BudgetUsage(jobs=1, gpu_hours_millis=5)
    first = ledger.reserve("child-1", usage, epoch=1, expected_revision=0)
    second = ledger.reserve("child-2", usage, epoch=1, expected_revision=1)
    assert first.reservation is not None and second.reservation is not None
    assert first.decision.payload["decision"] == second.decision.payload["decision"] == "allow"
    assert first.decision.payload["reservation_hash"] == first.reservation.content_hash
    third = ledger.reserve("child-3", BudgetUsage(jobs=1), epoch=1, expected_revision=2)
    assert third.decision.payload["decision"] == "deny"
    assert third.reservation is None
    assert third.state.payload["active_reservation_hashes"] == [
        first.reservation.content_hash,
        second.reservation.content_hash,
    ]


def test_gpu_hard_cap_and_production_readiness(tmp_path: Path) -> None:
    ledger = FixtureBudgetLedger.create(tmp_path / "fixture", "gpu-cap-20", _limits(gpu_count=120), epoch=1)
    assert (
        ledger.reserve("gpu-96", BudgetUsage(gpu_count=96), epoch=1, expected_revision=0).decision.payload["decision"]
        == "allow"
    )
    assert (
        ledger.reserve("gpu-over", BudgetUsage(gpu_count=1), epoch=1, expected_revision=1).decision.payload["decision"]
        == "deny"
    )
    readiness = FixtureBudgetLedger.production_readiness(
        tmp_path / "production", ProductionBudgetLedgerConfig("TRAIN_35B")
    )
    assert readiness.payload["status"] == "blocked"
    assert readiness.payload["side_effects_permitted"] is False


def test_stale_revision_and_epoch_fail_closed(tmp_path: Path) -> None:
    ledger = FixtureBudgetLedger.create(tmp_path, "fenced-20", _limits(), epoch=2)
    ledger.reserve("child", BudgetUsage(jobs=1), epoch=2, expected_revision=0)
    with pytest.raises(BudgetLedgerError, match="STALE_LEDGER_REVISION"):
        ledger.reserve("stale-revision", BudgetUsage(jobs=1), epoch=2, expected_revision=0)
    ledger.advance_epoch(3, expected_revision=1)
    with pytest.raises(BudgetLedgerError, match="STALE_LEDGER_EPOCH"):
        ledger.reserve("stale-epoch", BudgetUsage(jobs=1), epoch=2, expected_revision=2)


def test_duplicate_action_is_idempotent_and_control_bools_fail_closed(tmp_path: Path) -> None:
    ledger = FixtureBudgetLedger.create(tmp_path, "idempotent-20", _limits(), epoch=1)
    first = ledger.reserve("same-child", BudgetUsage(jobs=1), epoch=1, expected_revision=0)
    replay = ledger.reserve("same-child", BudgetUsage(jobs=1), epoch=1, expected_revision=0)
    assert first.reservation is not None and replay.reservation is not None
    assert replay.reservation.content_hash == first.reservation.content_hash
    assert replay.decision.content_hash == first.decision.content_hash
    assert replay.state.payload["revision"] == 1
    with pytest.raises(BudgetLedgerError, match="ACTION_IDEMPOTENCY_CONFLICT"):
        ledger.reserve("same-child", BudgetUsage(jobs=2), epoch=1, expected_revision=1)
    with pytest.raises(BudgetLedgerError, match="control fields"):
        ledger.reserve("bool-epoch", BudgetUsage(), epoch=True, expected_revision=1)
    with pytest.raises(BudgetLedgerError, match="control fields"):
        ledger.reserve("bool-emergency", BudgetUsage(), epoch=1, expected_revision=1, emergency=1)  # type: ignore[arg-type]
