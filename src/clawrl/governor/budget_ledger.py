"""Fenced, durable, multi-dimensional fixture BudgetLedger."""

from __future__ import annotations

import fcntl
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from clawrl.artifacts import (
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    ImmutableArtifactConflict,
    JsonValue,
    canonical_json_bytes,
    sha256_hex,
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_DIMS = (
    "gpu_count",
    "gpu_hours_millis",
    "jobs",
    "cohorts",
    "queries",
    "rows",
    "scorer_calls",
    "scorer_spend_micros",
    "wall_clock_ticks",
)
_PHASES = {"TRAIN_35B", "TRAIN_122B"}


class BudgetLedgerError(RuntimeError):
    """Budget mutation was stale, conflicting, or unauthorized."""


def _safe_id(value: object, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(cast(str, value)) is None:
        raise BudgetLedgerError(f"{field} is invalid")
    return cast(str, value)


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    gpu_count: int = 0
    gpu_hours_millis: int = 0
    jobs: int = 0
    cohorts: int = 0
    queries: int = 0
    rows: int = 0
    scorer_calls: int = 0
    scorer_spend_micros: int = 0
    wall_clock_ticks: int = 0

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in self.values.values()):
            raise BudgetLedgerError("budget usage values must be non-negative integers")

    @classmethod
    def dimension_names(cls) -> tuple[str, ...]:
        return _DIMS

    @classmethod
    def from_mapping(cls, value: object) -> BudgetUsage:
        if not isinstance(value, dict) or set(value) != set(_DIMS):
            raise BudgetLedgerError("budget usage fields are invalid")
        return cls(**{name: value[name] for name in _DIMS})

    @property
    def values(self) -> dict[str, int]:
        return {name: cast(int, getattr(self, name)) for name in _DIMS}

    def payload(self) -> dict[str, JsonValue]:
        return dict(self.values)


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    values: dict[str, int]

    @classmethod
    def from_mapping(cls, value: object) -> BudgetLimits:
        if not isinstance(value, dict):
            raise BudgetLedgerError("budget limits must be a mapping")
        normalized: dict[str, int] = {}
        for name in _DIMS:
            candidate = value.get(name, 0)
            normalized[name] = candidate if type(candidate) is int and candidate >= 0 else 0
        return cls(normalized)

    def __post_init__(self) -> None:
        if set(self.values) != set(_DIMS) or any(type(value) is not int or value < 0 for value in self.values.values()):
            raise BudgetLedgerError("normalized budget limits are invalid")

    def payload(self) -> dict[str, JsonValue]:
        return dict(self.values)


@dataclass(frozen=True, slots=True)
class ProductionBudgetLedgerConfig:
    phase: str

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise BudgetLedgerError("production budget phase is invalid")


@dataclass(frozen=True, slots=True)
class BudgetActionResult:
    state: Artifact
    decision: Artifact
    reservation: Artifact | None


@dataclass(frozen=True, slots=True)
class BudgetReconcileResult:
    state: Artifact
    reconciliation: Artifact


@dataclass(frozen=True, slots=True)
class BudgetLedgerSnapshot:
    ledger: FixtureBudgetLedger
    budget: Artifact
    state: Artifact


class FixtureBudgetLedger:
    def __init__(self, root: str | Path, campaign_id: str) -> None:
        self.root = Path(root)
        self.campaign_id = _safe_id(campaign_id, "campaign_id")
        self.store = ArtifactStore(root)
        self.namespace = self.root / "budget-ledgers" / self.campaign_id

    @classmethod
    def create(
        cls,
        root: str | Path,
        campaign_id: str,
        limits: BudgetLimits,
        *,
        epoch: int,
    ) -> FixtureBudgetLedger:
        if type(epoch) is not int or epoch <= 0:
            raise BudgetLedgerError("budget ledger epoch is invalid")
        ledger = cls(root, campaign_id)
        ArtifactStore.durable_mkdir(ledger.namespace)
        ArtifactStore.durable_touch(ledger.namespace / "writer.lock")
        budget = ledger.store.put(
            "AutonomyBudget",
            "1.0.0",
            {
                "campaign_id": campaign_id,
                "gpu_hard_cap": 96,
                "limits": limits.payload(),
                "status": "frozen",
            },
        )
        ledger._publish_once(ledger.namespace / "budget.ref", budget, "BUDGET_REPLACEMENT_CONFLICT")
        state_ref = ledger.namespace / "state.ref"
        if not state_ref.exists():
            zero = BudgetUsage().payload()
            state = ledger.store.put(
                "BudgetLedgerState",
                "1.0.0",
                {
                    "active_reservation_hashes": [],
                    "actual_usage": zero,
                    "budget_hash": budget.content_hash,
                    "campaign_id": campaign_id,
                    "epoch": epoch,
                    "previous_state_hash": None,
                    "reserved_usage": zero,
                    "revision": 0,
                },
            )
            ledger._replace_state(state)
        snapshot = cls.resume(root, campaign_id)
        if snapshot.state.payload["epoch"] != epoch:
            raise BudgetLedgerError("BUDGET_REPLACEMENT_CONFLICT:epoch changed")
        return snapshot.ledger

    @classmethod
    def resume(cls, root: str | Path, campaign_id: str) -> BudgetLedgerSnapshot:
        ledger = cls(root, campaign_id)
        try:
            budget = ledger.store.read(
                (ledger.namespace / "budget.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="AutonomyBudget",
            )
            state = ledger.store.read(
                (ledger.namespace / "state.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="BudgetLedgerState",
            )
        except (ArtifactCorruption, OSError) as error:
            raise BudgetLedgerError("budget ledger cannot be recovered") from error
        ledger._validate_budget(budget)
        ledger._validate_state(state, budget)
        return BudgetLedgerSnapshot(ledger, budget, state)

    @staticmethod
    def production_readiness(root: str | Path, config: ProductionBudgetLedgerConfig) -> Artifact:
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": "AUTONOMY_BUDGET_APPROVAL_MISSING", "status": "blocked"},
                    {"code": "EXTERNAL_FENCE_AUTHORITY_UNVERIFIED", "status": "blocked"},
                ],
                "execution_profile": "production",
                "phase": config.phase,
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )

    def reserve(
        self,
        action_id: str,
        worst_case: BudgetUsage,
        *,
        epoch: int,
        expected_revision: int,
        emergency: bool = False,
        risk_reducing: bool = False,
    ) -> BudgetActionResult:
        _safe_id(action_id, "action_id")
        if (
            type(epoch) is not int
            or epoch <= 0
            or type(expected_revision) is not int
            or expected_revision < 0
            or type(emergency) is not bool
            or type(risk_reducing) is not bool
        ):
            raise BudgetLedgerError("budget reserve control fields are invalid")
        with self._locked():
            snapshot = self.resume(self.root, self.campaign_id)
            budget = BudgetLimits.from_mapping(snapshot.budget.payload["limits"])
            plan = self.store.put(
                "ActionPlan",
                "1.0.0",
                {
                    "action_id": action_id,
                    "budget_hash": snapshot.budget.content_hash,
                    "campaign_id": self.campaign_id,
                    "emergency": emergency,
                    "idempotency_key": sha256_hex(
                        canonical_json_bytes(
                            {
                                "action_id": action_id,
                                "budget_hash": snapshot.budget.content_hash,
                                "campaign_id": self.campaign_id,
                                "worst_case": worst_case.payload(),
                            }
                        )
                    ),
                    "risk_reducing": risk_reducing,
                    "worst_case": worst_case.payload(),
                },
            )
            existing = self._existing_action(action_id, plan, snapshot.state)
            if existing is not None:
                return existing
            self._fence(snapshot.state, epoch, expected_revision)
            if emergency:
                if not risk_reducing or any(worst_case.values.values()):
                    raise BudgetLedgerError("emergency exemption requires a zero-cost risk-reducing action")
                exemption = self.store.put(
                    "EmergencyBudgetExemption",
                    "1.0.0",
                    {
                        "action_plan_hash": plan.content_hash,
                        "campaign_id": self.campaign_id,
                        "reason_code": "EMERGENCY_RISK_REDUCTION",
                    },
                )
                decision = self._decision(plan, snapshot.state, "allow", "EMERGENCY_RISK_REDUCTION", None, exemption)
                result = BudgetActionResult(snapshot.state, decision, None)
                self._record_action(action_id, plan, result)
                return result

            reserved = BudgetUsage.from_mapping(snapshot.state.payload["reserved_usage"])
            actual = BudgetUsage.from_mapping(snapshot.state.payload["actual_usage"])
            allowed = self._fits(budget, reserved, actual, worst_case)
            if not allowed:
                decision = self._decision(plan, snapshot.state, "deny", "BUDGET_EXHAUSTED", None, None)
                result = BudgetActionResult(snapshot.state, decision, None)
                self._record_action(action_id, plan, result)
                return result
            reservation = self.store.put(
                "BudgetReservation",
                "1.0.0",
                {
                    "action_plan_hash": plan.content_hash,
                    "budget_hash": snapshot.budget.content_hash,
                    "campaign_id": self.campaign_id,
                    "epoch": epoch,
                    "reserved_at_revision": expected_revision,
                    "status": "active",
                    "worst_case": worst_case.payload(),
                },
            )
            new_reserved = {name: reserved.values[name] + worst_case.values[name] for name in _DIMS}
            state = self._new_state(
                snapshot.state,
                epoch=epoch,
                reserved=BudgetUsage.from_mapping(new_reserved),
                actual=actual,
                active=[
                    *cast(list[str], snapshot.state.payload["active_reservation_hashes"]),
                    reservation.content_hash,
                ],
            )
            self._replace_state(state)
            decision = self._decision(plan, state, "allow", "BUDGET_RESERVED", reservation, None)
            result = BudgetActionResult(state, decision, reservation)
            self._record_action(action_id, plan, result)
            return result

    def reconcile(
        self,
        reservation_hash: str,
        actual_usage: BudgetUsage,
        *,
        epoch: int,
        expected_revision: int,
    ) -> BudgetReconcileResult:
        if type(epoch) is not int or epoch <= 0 or type(expected_revision) is not int or expected_revision < 0:
            raise BudgetLedgerError("budget reconcile control fields are invalid")
        with self._locked():
            snapshot = self.resume(self.root, self.campaign_id)
            self._fence(snapshot.state, epoch, expected_revision)
            try:
                reservation = self.store.read(reservation_hash, expected_schema_name="BudgetReservation")
            except ArtifactCorruption as error:
                raise BudgetLedgerError("reservation cannot be recovered") from error
            active = cast(list[str], snapshot.state.payload["active_reservation_hashes"])
            if reservation_hash not in active or reservation.payload.get("campaign_id") != self.campaign_id:
                raise BudgetLedgerError("reservation is not active in this campaign")
            if reservation.payload.get("epoch") != epoch:
                raise BudgetLedgerError("STALE_LEDGER_EPOCH")
            worst = BudgetUsage.from_mapping(reservation.payload["worst_case"])
            if any(actual_usage.values[name] > worst.values[name] for name in _DIMS):
                raise BudgetLedgerError("ACTUAL_EXCEEDS_RESERVATION")
            reserved = BudgetUsage.from_mapping(snapshot.state.payload["reserved_usage"])
            cumulative = BudgetUsage.from_mapping(snapshot.state.payload["actual_usage"])
            new_reserved = {name: reserved.values[name] - worst.values[name] for name in _DIMS}
            new_actual = {
                name: (
                    max(cumulative.values[name], actual_usage.values[name])
                    if name == "gpu_count"
                    else cumulative.values[name] + actual_usage.values[name]
                )
                for name in _DIMS
            }
            reconciliation = self.store.put(
                "BudgetReconciliation",
                "1.0.0",
                {
                    "actual_usage": actual_usage.payload(),
                    "campaign_id": self.campaign_id,
                    "epoch": epoch,
                    "reservation_hash": reservation.content_hash,
                    "status": "completed",
                },
            )
            state = self._new_state(
                snapshot.state,
                epoch=epoch,
                reserved=BudgetUsage.from_mapping(new_reserved),
                actual=BudgetUsage.from_mapping(new_actual),
                active=[item for item in active if item != reservation_hash],
            )
            self._replace_state(state)
            return BudgetReconcileResult(state, reconciliation)

    def advance_epoch(self, epoch: int, *, expected_revision: int) -> Artifact:
        if type(epoch) is not int or type(expected_revision) is not int or expected_revision < 0:
            raise BudgetLedgerError("budget epoch advance control fields are invalid")
        with self._locked():
            snapshot = self.resume(self.root, self.campaign_id)
            current = cast(int, snapshot.state.payload["epoch"])
            if snapshot.state.payload["revision"] != expected_revision:
                raise BudgetLedgerError("STALE_LEDGER_REVISION")
            if type(epoch) is not int or epoch <= current:
                raise BudgetLedgerError("STALE_LEDGER_EPOCH")
            state = self._new_state(
                snapshot.state,
                epoch=epoch,
                reserved=BudgetUsage.from_mapping(snapshot.state.payload["reserved_usage"]),
                actual=BudgetUsage.from_mapping(snapshot.state.payload["actual_usage"]),
                active=cast(list[str], snapshot.state.payload["active_reservation_hashes"]),
            )
            self._replace_state(state)
            return state

    @staticmethod
    def _fits(limits: BudgetLimits, reserved: BudgetUsage, actual: BudgetUsage, request: BudgetUsage) -> bool:
        for name in _DIMS:
            if name == "gpu_count":
                if reserved.values[name] + request.values[name] > min(96, limits.values[name]):
                    return False
            elif actual.values[name] + reserved.values[name] + request.values[name] > limits.values[name]:
                return False
        return True

    def _new_state(
        self,
        previous: Artifact,
        *,
        epoch: int,
        reserved: BudgetUsage,
        actual: BudgetUsage,
        active: list[str],
    ) -> Artifact:
        return self.store.put(
            "BudgetLedgerState",
            "1.0.0",
            {
                "active_reservation_hashes": active,
                "actual_usage": actual.payload(),
                "budget_hash": previous.payload["budget_hash"],
                "campaign_id": self.campaign_id,
                "epoch": epoch,
                "previous_state_hash": previous.content_hash,
                "reserved_usage": reserved.payload(),
                "revision": cast(int, previous.payload["revision"]) + 1,
            },
        )

    def _decision(
        self,
        plan: Artifact,
        state: Artifact,
        decision_value: str,
        reason: str,
        reservation: Artifact | None,
        exemption: Artifact | None,
    ) -> Artifact:
        return self.store.put(
            "HarnessDecision",
            "1.0.0",
            {
                "action_plan_hash": plan.content_hash,
                "budget_state_hash": state.content_hash,
                "decision": decision_value,
                "emergency_exemption_hash": None if exemption is None else exemption.content_hash,
                "reason_code": reason,
                "reservation_hash": None if reservation is None else reservation.content_hash,
            },
        )

    def _existing_action(
        self,
        action_id: str,
        plan: Artifact,
        current_state: Artifact,
    ) -> BudgetActionResult | None:
        ref = self.namespace / "actions" / action_id / "record.ref"
        if not ref.exists():
            return None
        try:
            record = self.store.read(
                ref.read_text(encoding="ascii").strip(),
                expected_schema_name="BudgetActionRecord",
            )
        except (ArtifactCorruption, OSError) as error:
            raise BudgetLedgerError("budget action record cannot be recovered") from error
        if (
            set(record.payload) != {"action_plan_hash", "decision_hash", "reservation_hash", "state_hash"}
            or record.payload.get("action_plan_hash") != plan.content_hash
        ):
            raise BudgetLedgerError("ACTION_IDEMPOTENCY_CONFLICT")
        decision_hash = record.payload.get("decision_hash")
        reservation_hash = record.payload.get("reservation_hash")
        if type(decision_hash) is not str or reservation_hash is not None and type(reservation_hash) is not str:
            raise BudgetLedgerError("budget action record hashes are invalid")
        decision = self.store.read(decision_hash, expected_schema_name="HarnessDecision")
        reservation = (
            None
            if reservation_hash is None
            else self.store.read(reservation_hash, expected_schema_name="BudgetReservation")
        )
        return BudgetActionResult(current_state, decision, reservation)

    def _record_action(self, action_id: str, plan: Artifact, result: BudgetActionResult) -> None:
        record = self.store.put(
            "BudgetActionRecord",
            "1.0.0",
            {
                "action_plan_hash": plan.content_hash,
                "decision_hash": result.decision.content_hash,
                "reservation_hash": None if result.reservation is None else result.reservation.content_hash,
                "state_hash": result.state.content_hash,
            },
        )
        namespace = self.namespace / "actions" / action_id
        ArtifactStore.durable_mkdir(namespace)
        self._publish_once(namespace / "record.ref", record, "ACTION_IDEMPOTENCY_CONFLICT")

    @staticmethod
    def _fence(state: Artifact, epoch: int, revision: int) -> None:
        if state.payload.get("epoch") != epoch:
            raise BudgetLedgerError("STALE_LEDGER_EPOCH")
        if state.payload.get("revision") != revision:
            raise BudgetLedgerError("STALE_LEDGER_REVISION")

    def _validate_budget(self, budget: Artifact) -> None:
        if (
            set(budget.payload) != {"campaign_id", "gpu_hard_cap", "limits", "status"}
            or budget.payload.get("campaign_id") != self.campaign_id
            or budget.payload.get("gpu_hard_cap") != 96
            or budget.payload.get("status") != "frozen"
        ):
            raise BudgetLedgerError("AutonomyBudget is invalid")
        BudgetLimits.from_mapping(budget.payload.get("limits"))

    def _validate_state(self, state: Artifact, budget: Artifact) -> None:
        active = state.payload.get("active_reservation_hashes")
        if (
            set(state.payload)
            != {
                "active_reservation_hashes",
                "actual_usage",
                "budget_hash",
                "campaign_id",
                "epoch",
                "previous_state_hash",
                "reserved_usage",
                "revision",
            }
            or state.payload.get("budget_hash") != budget.content_hash
            or state.payload.get("campaign_id") != self.campaign_id
            or type(state.payload.get("epoch")) is not int
            or cast(int, state.payload["epoch"]) <= 0
            or type(state.payload.get("revision")) is not int
            or not isinstance(active, list)
            or len(active) != len(set(cast(list[str], active)))
        ):
            raise BudgetLedgerError("BudgetLedger state is invalid")
        BudgetUsage.from_mapping(state.payload.get("reserved_usage"))
        BudgetUsage.from_mapping(state.payload.get("actual_usage"))
        for item in active:
            if type(item) is not str:
                raise BudgetLedgerError("active reservation hash is invalid")
            reservation = self.store.read(item, expected_schema_name="BudgetReservation")
            if reservation.payload.get("campaign_id") != self.campaign_id:
                raise BudgetLedgerError("active reservation campaign changed")

    def _publish_once(self, ref: Path, artifact: Artifact, code: str) -> None:
        try:
            ArtifactStore._publish(ref, f"{artifact.content_hash}\n".encode("ascii"))
            return
        except ImmutableArtifactConflict:
            pass
        existing = ref.read_text(encoding="ascii").strip()
        if existing == artifact.content_hash:
            return
        raise BudgetLedgerError(code)

    def _replace_state(self, state: Artifact) -> None:
        ref = self.namespace / "state.ref"
        temporary = ref.with_name(f".{ref.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(f"{state.content_hash}\n".encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, ref)
        directory = os.open(self.namespace, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _locked(self):
        class _Lock:
            def __init__(inner_self, path: Path) -> None:
                inner_self.stream = path.open("rb")

            def __enter__(inner_self):
                fcntl.flock(inner_self.stream.fileno(), fcntl.LOCK_EX)

            def __exit__(inner_self, *_: object) -> None:
                fcntl.flock(inner_self.stream.fileno(), fcntl.LOCK_UN)
                inner_self.stream.close()

        return _Lock(self.namespace / "writer.lock")
