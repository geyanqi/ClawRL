"""Persistent Ticket 20 BudgetLedger acceptance tests."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from clawrl.governor.budget_ledger import BudgetLedgerError, BudgetLimits, BudgetUsage, FixtureBudgetLedger


def _limits() -> BudgetLimits:
    return BudgetLimits.from_mapping(
        {
            "gpu_count": 8,
            "gpu_hours_millis": 100,
            "jobs": 2,
            "cohorts": 1,
            "queries": 4,
            "rows": 1000,
            "scorer_calls": 10,
            "scorer_spend_micros": 5000,
            "wall_clock_ticks": 100,
        }
    )


def _fresh(root: Path, campaign_id: str) -> dict[str, object]:
    code = """
import json,sys
from clawrl.governor.budget_ledger import FixtureBudgetLedger
s=FixtureBudgetLedger.resume(sys.argv[1],sys.argv[2])
print(json.dumps({'revision':s.state.payload['revision'],'epoch':s.state.payload['epoch'],'state':s.state.content_hash,'active':s.state.payload['active_reservation_hashes'],'actual':s.state.payload['actual_usage']},sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root), campaign_id], check=True, capture_output=True, text=True, env=env
    )
    return json.loads(completed.stdout)


def test_reserve_reconcile_restart_and_exhaustion_preserve_inflight(tmp_path: Path) -> None:
    ledger = FixtureBudgetLedger.create(tmp_path, "durable-20", _limits(), epoch=1)
    reservation = ledger.reserve(
        "child-a",
        BudgetUsage(
            gpu_count=8,
            gpu_hours_millis=60,
            jobs=1,
            queries=2,
            rows=400,
            scorer_calls=4,
            scorer_spend_micros=2000,
            wall_clock_ticks=50,
        ),
        epoch=1,
        expected_revision=0,
    )
    assert reservation.reservation is not None
    denied = ledger.reserve("child-b", BudgetUsage(gpu_count=1), epoch=1, expected_revision=1)
    assert denied.decision.payload["decision"] == "deny"
    assert reservation.reservation.content_hash in cast(list[str], denied.state.payload["active_reservation_hashes"])
    fresh = _fresh(tmp_path, "durable-20")
    assert fresh["revision"] == 1 and fresh["active"] == [reservation.reservation.content_hash]
    reconciled = FixtureBudgetLedger.resume(tmp_path, "durable-20").ledger.reconcile(
        reservation.reservation.content_hash,
        BudgetUsage(
            gpu_count=8,
            gpu_hours_millis=55,
            jobs=1,
            queries=2,
            rows=390,
            scorer_calls=4,
            scorer_spend_micros=1900,
            wall_clock_ticks=48,
        ),
        epoch=1,
        expected_revision=1,
    )
    assert reconciled.state.payload["active_reservation_hashes"] == []
    assert _fresh(tmp_path, "durable-20")["state"] == reconciled.state.content_hash


def test_budget_replacement_conflict_and_emergency_risk_reduction_bypass_exhaustion(tmp_path: Path) -> None:
    ledger = FixtureBudgetLedger.create(tmp_path, "emergency-20", _limits(), epoch=4)
    work = ledger.reserve("all-jobs", BudgetUsage(jobs=2), epoch=4, expected_revision=0)
    assert work.reservation is not None
    assert work.decision.payload["decision"] == "allow"
    emergency = ledger.reserve(
        "cancel-risk",
        BudgetUsage(),
        epoch=4,
        expected_revision=1,
        emergency=True,
        risk_reducing=True,
    )
    assert emergency.decision.payload["decision"] == "allow"
    assert emergency.decision.payload["reason_code"] == "EMERGENCY_RISK_REDUCTION"
    assert emergency.reservation is None
    assert emergency.state.payload["active_reservation_hashes"] == [work.reservation.content_hash]
    with pytest.raises(BudgetLedgerError, match="BUDGET_REPLACEMENT_CONFLICT"):
        FixtureBudgetLedger.create(
            tmp_path,
            "emergency-20",
            BudgetLimits.from_mapping({**_limits().values, "jobs": 3}),
            epoch=4,
        )


def test_reconcile_rejects_actual_above_reserved_and_stale_writer(tmp_path: Path) -> None:
    ledger = FixtureBudgetLedger.create(tmp_path, "reconcile-20", _limits(), epoch=1)
    result = ledger.reserve("child", BudgetUsage(jobs=1, rows=10), epoch=1, expected_revision=0)
    assert result.reservation is not None
    with pytest.raises(BudgetLedgerError, match="ACTUAL_EXCEEDS_RESERVATION"):
        ledger.reconcile(
            result.reservation.content_hash,
            BudgetUsage(jobs=1, rows=11),
            epoch=1,
            expected_revision=1,
        )
    ledger.advance_epoch(2, expected_revision=1)
    with pytest.raises(BudgetLedgerError, match="STALE_LEDGER_EPOCH"):
        ledger.reconcile(
            result.reservation.content_hash,
            BudgetUsage(jobs=1, rows=10),
            epoch=1,
            expected_revision=2,
        )
    with pytest.raises(BudgetLedgerError, match="STALE_LEDGER_EPOCH"):
        ledger.reconcile(
            result.reservation.content_hash,
            BudgetUsage(jobs=1, rows=10),
            epoch=2,
            expected_revision=2,
        )
