"""Durable, bounded Governor iteration fixture (Ticket 25).

This module is intentionally small but production-shaped: a tick commits one
immutable proposal and one finite child DAG.  Every child has an idempotency
key, a worst-case ledger reservation and a five-step Harness audit.  The
fixture profile is executable; production readiness is fail-closed.
"""

from __future__ import annotations

import fcntl
import importlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from clawrl.adapters.sources.fixture import DataProviderIngressReceipt
from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, canonical_json_bytes, sha256_hex
from clawrl.data.models import (
    FixtureDataIngestConfig,
    QueryWindow,
)
from clawrl.data.validation import load_training_dataset, validate_dataset_for_experiment
from clawrl.data.workflow import GovernedDataIngestWorkflow
from clawrl.governor.budget_ledger import BudgetLedgerError, BudgetLimits, BudgetUsage, FixtureBudgetLedger
from clawrl.governor.six_arm_cohort import SixArmCohortWorkflow
from clawrl.harness.journal import HarnessJournal
from clawrl.router.trace_router import RouterCapacityConfig

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_BRANCHES = {"new_dataset", "new_experiment", "stop_and_transfer"}
_HARN = ["submit", "status", "logs", "artifact", "checkpoint"]


class BoundedGovernorError(RuntimeError):
    """Governor input, policy, capacity, or durable state failed closed."""


class InjectedBoundedGovernorCrash(BoundedGovernorError):
    """Test-only host crash after a child outcome has been committed."""


@dataclass(frozen=True, slots=True)
class BoundedGovernorConfig:
    governor_id: str
    cohort_id: str
    branch: Literal["new_dataset", "new_experiment", "stop_and_transfer"] = "stop_and_transfer"
    candidate_id: str | None = None
    controller_epoch: int = 1
    max_children: int = 1
    router_capacity: RouterCapacityConfig = RouterCapacityConfig(1)
    changed_dimensions: tuple[str, ...] = ()
    future_evaluation_dataset_hash: str | None = None
    reserved_aggregation: bool = False
    execution_profile: Literal["fixture", "production"] = "fixture"

    def __post_init__(self) -> None:
        for name, value in (("governor_id", self.governor_id), ("cohort_id", self.cohort_id)):
            if type(value) is not str or _ID.fullmatch(value) is None:
                raise BoundedGovernorError(f"{name} is invalid")
        if self.branch not in _BRANCHES:
            raise BoundedGovernorError("branch is invalid")
        if self.candidate_id is not None and (
            type(self.candidate_id) is not str or _ID.fullmatch(self.candidate_id) is None
        ):
            raise BoundedGovernorError("candidate_id is invalid")
        if type(self.controller_epoch) is not int or self.controller_epoch <= 0:
            raise BoundedGovernorError("controller_epoch is invalid")
        if type(self.max_children) is not int or not 1 <= self.max_children <= 8:
            raise BoundedGovernorError("max_children is invalid")
        if not isinstance(self.changed_dimensions, tuple) or any(
            type(item) is not str for item in self.changed_dimensions
        ):
            raise BoundedGovernorError("changed_dimensions is invalid")
        if self.reserved_aggregation or "aggregation" in self.changed_dimensions:
            raise BoundedGovernorError("RESERVED_AGGREGATION_REJECTED")
        if self.future_evaluation_dataset_hash is not None:
            raise BoundedGovernorError("Future EvaluationDataset is outside Governor visibility boundary")
        if self.execution_profile not in {"fixture", "production"}:
            raise BoundedGovernorError("execution_profile must be fixture or production")


@dataclass(frozen=True, slots=True)
class BoundedGovernorSnapshot:
    state: Artifact
    proposal: Artifact
    plan: Artifact
    children: tuple[Artifact, ...]
    transfer_candidate: Artifact | None
    summary: Artifact | None

    @property
    def terminal(self) -> bool:
        return self.state.payload.get("status") == "terminal"


def _replace_ref(ref: Path, digest: str) -> None:
    ArtifactStore.durable_mkdir(ref.parent)
    temp = ref.with_name(f".{ref.name}.tmp")
    temp.write_text(digest + "\n", encoding="ascii")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, ref)


def _publish_once(ref: Path, digest: str) -> None:
    try:
        ArtifactStore._publish(ref, f"{digest}\n".encode("ascii"))
    except Exception as error:
        try:
            if ref.read_text(encoding="ascii").strip() == digest:
                return
        except OSError:
            pass
        raise BoundedGovernorError("immutable Governor input conflict") from error


class BoundedGovernorWorkflow:
    """Run one finite Governor tick and recover it without repeating children."""

    @staticmethod
    def production_readiness(root: str | Path, *, phase: str = "TRAIN_35B") -> Artifact:
        if phase not in {"TRAIN_35B", "TRAIN_122B"}:
            raise BoundedGovernorError("phase is invalid")
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": "GOVERNOR_EXTERNAL_AUTHORITY_UNVERIFIED", "status": "blocked"},
                    {"code": "ROUTER_CAPACITY_AUTHORITY_UNVERIFIED", "status": "blocked"},
                ],
                "execution_profile": "production",
                "phase": phase,
                "side_effects_permitted": False,
                "status": "blocked",
                "submit_attempted": False,
            },
        )

    @classmethod
    def run(
        cls, root: str | Path, *, config: BoundedGovernorConfig, crash_after_child: int | None = None
    ) -> BoundedGovernorSnapshot:
        if config.execution_profile == "production":
            raise BoundedGovernorError("production Governor is blocked until external authority is verified")
        store = ArtifactStore(root)
        path = Path(root) / "governor-iterations" / config.governor_id
        ArtifactStore.durable_mkdir(path)
        cohort = cls._cohort(root, config.cohort_id)
        input_body = {
            "cohort_id": config.cohort_id,
            "cohort_state_hash": cohort.content_hash,
            "branch": config.branch,
            "controller_epoch": config.controller_epoch,
            "governor_id": config.governor_id,
            "max_children": config.max_children,
            "router_capacity": config.router_capacity.max_global_subthreads,
            "changed_dimensions": list(config.changed_dimensions),
            "candidate_id": config.candidate_id,
        }
        input_art = store.put("GovernorInput", "1.0.0", input_body)
        _publish_once(path / "input.ref", input_art.content_hash)
        if not (path / "state.ref").exists():
            state = store.put(
                "GovernorState",
                "1.0.0",
                {
                    "child_hashes": [],
                    "controller_epoch": config.controller_epoch,
                    "input_hash": input_art.content_hash,
                    "next_index": 0,
                    "plan_hash": None,
                    "proposal_hash": None,
                    "summary_hash": None,
                    "transfer_candidate_hash": None,
                    "status": "active",
                },
            )
            _replace_ref(path / "state.ref", state.content_hash)
        return cls._drive(root, config, crash_after_child)

    @classmethod
    def resume(
        cls, root: str | Path, governor_id: str, *, controller_epoch: int | None = None
    ) -> BoundedGovernorSnapshot:
        if type(governor_id) is not str or _ID.fullmatch(governor_id) is None:
            raise BoundedGovernorError("governor_id is invalid")
        path = Path(root) / "governor-iterations" / governor_id
        store = ArtifactStore(root)
        try:
            inp = store.read(
                (path / "input.ref").read_text(encoding="ascii").strip(), expected_schema_name="GovernorInput"
            )
            epoch = cast(int, inp.payload["controller_epoch"])
            if controller_epoch is not None and controller_epoch < epoch:
                raise BoundedGovernorError("STALE_GOVERNOR_CONTROLLER_EPOCH")
            branch = cast(Literal["new_dataset", "new_experiment", "stop_and_transfer"], inp.payload["branch"])
            config = BoundedGovernorConfig(
                governor_id,
                cast(str, inp.payload["cohort_id"]),
                branch=branch,
                controller_epoch=max(epoch, controller_epoch or epoch),
                max_children=cast(int, inp.payload["max_children"]),
                router_capacity=RouterCapacityConfig(cast(int, inp.payload.get("router_capacity", 1))),
                changed_dimensions=tuple(cast(list[str], inp.payload.get("changed_dimensions", []))),
                candidate_id=cast(str | None, inp.payload.get("candidate_id")),
            )
        except (OSError, ArtifactCorruption, KeyError, TypeError) as error:
            raise BoundedGovernorError("Governor input cannot be recovered") from error
        return cls._drive(root, config, None)

    @classmethod
    def bind_run_record(cls, root: str | Path, governor_id: str, *, run_record_hash: str) -> Artifact:
        """Publish a terminal candidate carrying the durable 122B run lineage.

        The original Governor candidate remains the immutable recertification
        input.  This derived candidate is a workflow-owned handoff record, so
        callers never manufacture a TransferCandidate after the gated run.
        """
        if _HASH.fullmatch(run_record_hash) is None:
            raise BoundedGovernorError("run_record_hash is invalid")
        snapshot = cls.resume(root, governor_id)
        candidate = snapshot.transfer_candidate
        if candidate is None or not snapshot.terminal:
            raise BoundedGovernorError("Governor candidate is not terminal")
        store = ArtifactStore(root)
        run = store.read(run_record_hash, expected_schema_name="RunRecord")
        if run.payload.get("status") != "succeeded" or run.payload.get("phase") != "TRAIN_122B":
            raise BoundedGovernorError("122B run is not a terminal transfer input")
        existing = candidate.payload.get("run_record_hash")
        if existing is not None and existing != run_record_hash:
            raise BoundedGovernorError("TransferCandidate run lineage conflict")
        if existing == run_record_hash:
            return candidate
        return store.put(
            "TransferCandidate",
            "1.0.0",
            {
                **candidate.payload,
                "run_record_hash": run_record_hash,
                "source_transfer_candidate_hash": candidate.content_hash,
            },
        )

    @classmethod
    def _cohort(cls, root: str | Path, cohort_id: str) -> Artifact:
        store = ArtifactStore(root)
        try:
            # Re-run the cohort's own immutable recovery/terminal validator before
            # consuming its evidence; a forged state.ref with only status=terminal
            # must never authorize a Governor tick.
            cohort_snapshot = SixArmCohortWorkflow.resume(root, cohort_id)
            if not cohort_snapshot.terminal or cohort_snapshot.promotion is None:
                raise BoundedGovernorError("approved cohort terminal barrier is incomplete")
            state = store.read(
                (Path(root) / "six-arm-cohorts" / cohort_id / "state.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="SixArmCohortState",
            )
        except (OSError, ArtifactCorruption) as error:
            raise BoundedGovernorError("approved cohort is unavailable") from error
        if state.payload.get("status") != "terminal":
            raise BoundedGovernorError("Governor requires a terminal approved cohort")
        return state

    @classmethod
    def _drive(
        cls, root: str | Path, config: BoundedGovernorConfig, crash_after_child: int | None
    ) -> BoundedGovernorSnapshot:
        path = Path(root) / "governor-iterations" / config.governor_id
        ArtifactStore.durable_touch(path / "controller.lock")
        with (path / "controller.lock").open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                return cls._advance(root, config, crash_after_child)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @classmethod
    def _advance(
        cls, root: str | Path, config: BoundedGovernorConfig, crash_after_child: int | None
    ) -> BoundedGovernorSnapshot:
        store = ArtifactStore(root)
        path = Path(root) / "governor-iterations" / config.governor_id
        state = store.read(
            (path / "state.ref").read_text(encoding="ascii").strip(), expected_schema_name="GovernorState"
        )
        inp = store.read((path / "input.ref").read_text(encoding="ascii").strip(), expected_schema_name="GovernorInput")
        persisted_epoch = state.payload.get("controller_epoch")
        if (
            state.payload.get("input_hash") != inp.content_hash
            or type(persisted_epoch) is not int
            or cast(int, persisted_epoch) > config.controller_epoch
        ):
            raise BoundedGovernorError("Governor state tamper or stale fencing detected")
        if config.controller_epoch > cast(int, persisted_epoch) + 1:
            raise BoundedGovernorError("UNAUTHORIZED_GOVERNOR_EPOCH_JUMP")
        if config.controller_epoch > cast(int, persisted_epoch):
            state = store.put("GovernorState", "1.0.0", {**state.payload, "controller_epoch": config.controller_epoch})
            _replace_ref(path / "state.ref", state.content_hash)
        proposal_hash = state.payload.get("proposal_hash")
        proposal = (
            store.read(cast(str, proposal_hash), expected_schema_name="DecisionProposal")
            if isinstance(proposal_hash, str)
            else cls._proposal(store, config, inp)
        )
        if not isinstance(proposal_hash, str):
            _replace_ref(path / "proposal.ref", proposal.content_hash)
        plan_hash = state.payload.get("plan_hash")
        if isinstance(plan_hash, str):
            plan = store.read(plan_hash, expected_schema_name="ActionPlan")
        else:
            plan = cls._plan(store, config, proposal)
            _replace_ref(path / "plan.ref", plan.content_hash)
            state = store.put(
                "GovernorState",
                "1.0.0",
                {**state.payload, "proposal_hash": proposal.content_hash, "plan_hash": plan.content_hash},
            )
            _replace_ref(path / "state.ref", state.content_hash)
        if plan.payload.get("proposal_hash") != proposal.content_hash:
            raise BoundedGovernorError("Governor ActionPlan proposal binding is invalid")
        children = [
            store.read(item, expected_schema_name="GovernorChild")
            for item in cast(list[str], state.payload.get("child_hashes", []))
        ]
        for child in children:
            if child.payload.get("plan_hash") != plan.content_hash or child.payload.get("branch") != config.branch:
                raise BoundedGovernorError("Governor child DAG binding is invalid")
            expected_key = sha256_hex(
                canonical_json_bytes(
                    {
                        "action_id": child.payload.get("action_id"),
                        "plan_hash": plan.content_hash,
                        "index": child.payload.get("child_index"),
                    }
                )
            )
            if child.payload.get("idempotency_key") != expected_key:
                raise BoundedGovernorError("Governor child idempotency binding is invalid")
        status = state.payload.get("status")
        if status not in {"active", "deferred", "terminal"}:
            raise BoundedGovernorError("Governor state status is invalid")
        if status == "terminal" and (
            not isinstance(state.payload.get("summary_hash"), str) or len(children) != config.max_children
        ):
            raise BoundedGovernorError("terminal Governor state is incomplete")
        if state.payload.get("status") == "deferred":
            deferred_summary = store.read(
                cast(str, state.payload["summary_hash"]), expected_schema_name="ExperimentSummary"
            )
            return BoundedGovernorSnapshot(state, proposal, plan, tuple(children), None, deferred_summary)
        ledger = cls._ledger(root, config)
        ledger_snapshot = FixtureBudgetLedger.resume(root, config.governor_id)
        ledger_epoch = cast(int, ledger_snapshot.state.payload["epoch"])
        if config.controller_epoch < ledger_epoch:
            raise BoundedGovernorError("STALE_GOVERNOR_CONTROLLER_EPOCH")
        if config.controller_epoch > ledger_epoch:
            ledger.advance_epoch(
                config.controller_epoch,
                expected_revision=cast(int, ledger_snapshot.state.payload["revision"]),
            )
            ledger = FixtureBudgetLedger.resume(root, config.governor_id).ledger
        # Capacity is a hard deny/defer boundary.  It never causes an implicit
        # Router resize or a recursive spawn.
        if not children and config.max_children > config.router_capacity.max_global_subthreads:
            child = cls._child(store, config, plan, 0)
            deferred = store.put(
                "GovernorChild",
                "1.0.0",
                {
                    **child.payload,
                    "status": "deferred",
                    "completion_condition": "router_capacity_available",
                    "reservation_hash": None,
                    "router_decision": "defer",
                    "harness_steps": _HARN,
                    "phase": "TRAIN_35B",
                    "readiness": "ready",
                },
            )
            children = [deferred]
            deferred_summary = store.put(
                "ExperimentSummary",
                "1.0.0",
                {
                    "branch": config.branch,
                    "evidence_hashes": [x.content_hash for x in children],
                    "causal_boundary": "controlled_comparison_only; black_box_descriptive_only",
                    "status": "deferred",
                    "reason_code": "ROUTER_CAPACITY_DENY",
                },
            )
            state = store.put(
                "GovernorState",
                "1.0.0",
                {
                    **state.payload,
                    "child_hashes": [deferred.content_hash],
                    "next_index": 1,
                    "summary_hash": deferred_summary.content_hash,
                    "status": "deferred",
                },
            )
            _replace_ref(path / "state.ref", state.content_hash)
            return BoundedGovernorSnapshot(state, proposal, plan, tuple(children), None, deferred_summary)
        # Recovery fence: a host crash after publishing a terminal child but
        # before ledger reconcile is repaired exactly once from the committed
        # child record.
        ledger_state = FixtureBudgetLedger.resume(root, config.governor_id).state
        for existing in children:
            reservation_hash = existing.payload.get("reservation_hash")
            if isinstance(reservation_hash, str) and reservation_hash in cast(
                list[str], ledger_state.payload["active_reservation_hashes"]
            ):
                actual = BudgetUsage(
                    jobs=1,
                    queries=1 if config.branch == "new_dataset" else 0,
                    rows=100 if config.branch == "new_dataset" else 0,
                )
                ledger.reconcile(
                    reservation_hash,
                    actual,
                    epoch=config.controller_epoch,
                    expected_revision=cast(int, ledger_state.payload["revision"]),
                )
                ledger_state = FixtureBudgetLedger.resume(root, config.governor_id).state
        while len(children) < config.max_children:
            index = len(children)
            child = cls._child(store, config, plan, index)
            usage = BudgetUsage(
                jobs=1,
                queries=1 if config.branch == "new_dataset" else 0,
                rows=100 if config.branch == "new_dataset" else 0,
            )
            revision = cast(int, FixtureBudgetLedger.resume(root, config.governor_id).state.payload["revision"])
            reservation_result = ledger.reserve(
                cast(str, child.payload["action_id"]), usage, epoch=config.controller_epoch, expected_revision=revision
            )
            if reservation_result.reservation is None:
                denied = store.put(
                    "GovernorChild",
                    "1.0.0",
                    {
                        **child.payload,
                        "status": "deferred",
                        "completion_condition": "budget_or_router_available",
                        "reservation_hash": None,
                    },
                )
                children.append(denied)
                break
            reservation_hash = reservation_result.reservation.content_hash
            completed = store.put(
                "GovernorChild",
                "1.0.0",
                {
                    **child.payload,
                    "status": "completed",
                    "reservation_hash": reservation_hash,
                    "budget_reconciled": True,
                    "harness_steps": _HARN,
                    "harness_purpose": "governor_child_execution",
                    "phase": "TRAIN_35B",
                    "readiness": "ready",
                    "harness_transition_hashes": cls._harness_transitions(store, child, config.controller_epoch),
                },
            )
            children.append(completed)
            _replace_ref(
                path / "state.ref",
                store.put(
                    "GovernorState",
                    "1.0.0",
                    {**state.payload, "child_hashes": [x.content_hash for x in children], "next_index": len(children)},
                ).content_hash,
            )
            state = store.read(
                (path / "state.ref").read_text(encoding="ascii").strip(), expected_schema_name="GovernorState"
            )
            ledger_state = FixtureBudgetLedger.resume(root, config.governor_id).state
            if reservation_hash in cast(list[str], ledger_state.payload["active_reservation_hashes"]):
                ledger.reconcile(
                    reservation_hash,
                    usage,
                    epoch=config.controller_epoch,
                    expected_revision=cast(int, ledger_state.payload["revision"]),
                )
            if crash_after_child is not None and index == crash_after_child:
                raise InjectedBoundedGovernorCrash(f"injected crash after child {index}")
        if len(children) >= config.max_children:
            result = cls._materialize_result(store, config, children, proposal)
            state = store.put(
                "GovernorState",
                "1.0.0",
                {
                    **state.payload,
                    "status": "terminal",
                    "summary_hash": result[0].content_hash,
                    "transfer_candidate_hash": result[1].content_hash if result[1] else None,
                    "child_hashes": [x.content_hash for x in children],
                    "next_index": len(children),
                },
            )
            _replace_ref(path / "state.ref", state.content_hash)
        summary: Artifact | None = (
            store.read(cast(str, state.payload["summary_hash"]), expected_schema_name="ExperimentSummary")
            if state.payload.get("summary_hash")
            else None
        )
        candidate = (
            store.read(cast(str, state.payload["transfer_candidate_hash"]), expected_schema_name="TransferCandidate")
            if state.payload.get("transfer_candidate_hash")
            else None
        )
        return BoundedGovernorSnapshot(state, proposal, plan, tuple(children), candidate, summary)

    @staticmethod
    def _proposal(store: ArtifactStore, config: BoundedGovernorConfig, inp: Artifact) -> Artifact:
        return store.put(
            "DecisionProposal",
            "1.0.0",
            {
                "governor_id": config.governor_id,
                "cohort_evidence_hash": inp.payload["cohort_state_hash"],
                "hypothesis": f"bounded-{config.branch}-improves-validated-result",
                "evidence": [inp.payload["cohort_state_hash"]],
                "config_diff": {"changed_dimensions": list(config.changed_dimensions)},
                "worst_case_budget": BudgetUsage(
                    jobs=config.max_children,
                    queries=1 if config.branch == "new_dataset" else 0,
                    rows=100 if config.branch == "new_dataset" else 0,
                ).payload(),
                "expected_result": "one immutable transfer candidate"
                if config.branch == "stop_and_transfer"
                else "one certified finite iteration",
                "phase": "TRAIN_35B",
                "purpose": "governor_decision",
            },
        )

    @staticmethod
    def _plan(store: ArtifactStore, config: BoundedGovernorConfig, proposal: Artifact) -> Artifact:
        return store.put(
            "ActionPlan",
            "1.0.0",
            {
                "proposal_hash": proposal.content_hash,
                "governor_id": config.governor_id,
                "bounded": True,
                "max_children": config.max_children,
                "child_ids": [f"{config.governor_id}:child:{i}" for i in range(config.max_children)],
                "dag_depth": 1,
                "recursive_spawn": False,
                "phase": "TRAIN_35B",
                "purpose": "governor_action_plan",
            },
        )

    @staticmethod
    def _child(store: ArtifactStore, config: BoundedGovernorConfig, plan: Artifact, index: int) -> Artifact:
        action_id = f"{config.governor_id}:child:{index}"
        key = sha256_hex(canonical_json_bytes({"action_id": action_id, "plan_hash": plan.content_hash, "index": index}))
        return store.put(
            "GovernorChild",
            "1.0.0",
            {
                "action_id": action_id,
                "child_index": index,
                "plan_hash": plan.content_hash,
                "idempotency_key": key,
                "completion_condition": "terminal_artifact_and_harness_outcome",
                "status": "planned",
                "branch": config.branch,
            },
        )

    @staticmethod
    def _harness_transitions(store: ArtifactStore, child: Artifact, epoch: int) -> list[str]:
        """Commit the exact five HarnessJournal transitions for a child."""
        proposal_id = "governor-harness-" + child.content_hash[:24]
        input_art = store.put("GovernorHarnessInput", "1.0.0", {"child_hash": child.content_hash})
        config_art = store.put("GovernorHarnessConfig", "1.0.0", {"phase": "TRAIN_35B"})
        policy_art = store.put("GovernorHarnessPolicy", "1.0.0", {"purpose": "governor_child_execution"})
        capability_art = store.put("GovernorHarnessCapability", "1.0.0", {"policy_hash": policy_art.content_hash})
        journal = HarnessJournal(store.root, store, proposal_id)
        journal.reserve_identity(
            input_art.content_hash, config_art.content_hash, policy_art.content_hash, capability_art.content_hash
        )
        existing = journal.transitions()
        if len(existing) == 5:
            journal.verify()
            return [item.content_hash for item in existing]
        journal.claim_epoch(epoch)
        previous: str | None = None
        transitions: list[str] = []
        for sequence, schema in enumerate(
            ("DecisionProposal", "ActionPlan", "HarnessDecision", "ActionObservation", "DecisionOutcome"), start=1
        ):
            item = journal.append(
                epoch,
                schema,
                {
                    "child_hash": child.content_hash,
                    "idempotency_key": child.payload["idempotency_key"],
                    "phase": _HARN[sequence - 1],
                    "purpose": "governor_child_execution",
                    "terminal": schema == "DecisionOutcome",
                },
                expected_sequence=sequence,
                expected_previous_hash=previous,
            )
            transitions.append(item.content_hash)
            previous = item.content_hash
        journal.verify()
        return transitions

    @staticmethod
    def _ledger(root: str | Path, config: BoundedGovernorConfig) -> FixtureBudgetLedger:
        try:
            return FixtureBudgetLedger.resume(root, config.governor_id).ledger
        except BudgetLedgerError as error:
            namespace = Path(root) / "budget-ledgers" / config.governor_id
            if (namespace / "budget.ref").exists() or (namespace / "state.ref").exists():
                raise BoundedGovernorError("budget ledger is corrupt; refusing to rebuild") from error
            return FixtureBudgetLedger.create(
                root,
                config.governor_id,
                BudgetLimits.from_mapping({"jobs": config.max_children, "queries": 1, "rows": 100}),
                epoch=config.controller_epoch,
            )

    @classmethod
    def _materialize_result(
        cls, store: ArtifactStore, config: BoundedGovernorConfig, children: list[Artifact], proposal: Artifact
    ) -> tuple[Artifact, Artifact | None]:
        if config.branch == "new_dataset":
            result = cls._run_canonical_dataset_workflow(store, config)
            summary = store.put(
                "ExperimentSummary",
                "1.0.0",
                {
                    "branch": config.branch,
                    "dataset_version_hash": result.content_hash,
                    "evidence_hashes": [x.content_hash for x in children],
                    "causal_boundary": "controlled_comparison_only; black_box_descriptive_only",
                    "status": "iteration_proposed",
                },
            )
            return summary, None

        if config.branch == "new_experiment":
            return cls._materialize_experiment_result(store, config, children, proposal)

        candidate_id = config.candidate_id or f"{config.governor_id}-candidate"
        summary = store.put(
            "ExperimentSummary",
            "1.0.0",
            {
                "branch": "stop_and_transfer",
                "evidence_hashes": [x.content_hash for x in children],
                "causal_boundary": "controlled_comparison_only; black_box_descriptive_only",
                "status": "terminal",
            },
        )
        candidate = store.put(
            "TransferCandidate",
            "1.0.0",
            {
                "candidate_id": candidate_id,
                "cohort_id": config.cohort_id,
                "summary_hash": summary.content_hash,
                "evidence_hashes": [x.content_hash for x in children],
                "status": "immutable",
                "terminal_action": "stop_and_transfer",
            },
        )
        return summary, candidate

    @staticmethod
    def _run_canonical_dataset_workflow(store: ArtifactStore, config: BoundedGovernorConfig) -> Artifact:
        """Run the existing governed fixture DataSource/SQL/lineage workflow.

        The role packet is loaded through an allowlisted fixture provider module; no
        final DatasetVersion is constructed here.  Production execution is rejected
        before this helper can be reached.
        """
        try:
            packet_module = importlib.import_module("tests.fixtures.ticket03_data")
            stage = cast(Callable[[Path], object], packet_module.stage_data_provider)
            receipt = stage(store.root)
        except (ImportError, AttributeError, TypeError, ValueError) as error:
            raise BoundedGovernorError("fixture Data Provider packet is unavailable") from error
        run_id = re.sub(r"[^A-Za-z0-9._-]", "-", f"ticket25-{config.governor_id}-dataset")
        ingest = FixtureDataIngestConfig(
            run_id=run_id,
            query_sql=(
                "SELECT trace_pk, report_id, event_time_utc, ingestion_time_utc, purpose, prompt, response, "
                "tool_name, model_id, private_sentinel FROM fixture.online_trace "
                "WHERE event_time_utc >= :start_utc AND event_time_utc < :end_utc AND purpose = :purpose "
                "ORDER BY trace_pk ASC, ingestion_time_utc ASC"
            ),
            window=QueryWindow("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
        )
        snapshot = GovernedDataIngestWorkflow.bootstrap(
            store.root, ingest, role_ingress=cast(DataProviderIngressReceipt, receipt), epoch=config.controller_epoch
        )
        while not snapshot.terminal:
            snapshot = GovernedDataIngestWorkflow.resume(store.root, run_id, epoch=config.controller_epoch)
        if snapshot.dataset_version is None:
            raise BoundedGovernorError("fixture DatasetVersion workflow did not reach a terminal artifact")
        load_training_dataset(store, snapshot.dataset_version.content_hash)
        validate_dataset_for_experiment(store, snapshot.dataset_version.content_hash)
        return snapshot.dataset_version

    @classmethod
    def _materialize_experiment_result(
        cls, store: ArtifactStore, config: BoundedGovernorConfig, children: list[Artifact], proposal: Artifact
    ) -> tuple[Artifact, Artifact | None]:
        """Create a new recertified ExperimentSpec without mutating frozen specs."""
        recert = store.put(
            "ExperimentRecertification",
            "1.0.0",
            {
                "status": "certified",
                "changed_dimensions": list(config.changed_dimensions),
                "evidence_hashes": [x.content_hash for x in children],
                "purpose": "training_allowed",
                "phase": "TRAIN_35B",
            },
        )
        prompt_art = store.put(
            "PromptArtifact",
            "1.0.0",
            {
                "governor_id": config.governor_id,
                "version": "prompt/v2",
                "changed": "prompt" in config.changed_dimensions,
            },
        )
        scalarizer_art = store.put(
            "ScalarizerArtifact",
            "1.0.0",
            {
                "governor_id": config.governor_id,
                "version": "scalarizer/v2",
                "changed": "scalarizer" in config.changed_dimensions,
            },
        )
        algorithm_art = store.put(
            "AlgorithmArtifact",
            "1.0.0",
            {
                "governor_id": config.governor_id,
                "version": "algorithm/v2",
                "changed": "algorithm" in config.changed_dimensions,
            },
        )
        cohort_state = store.read(
            cast(str, proposal.payload["cohort_evidence_hash"]), expected_schema_name="SixArmCohortState"
        )
        protocol = store.read(cast(str, cohort_state.payload["protocol_hash"]), expected_schema_name="ProtocolManifest")
        frozen = cast(dict[str, object], protocol.payload.get("frozen", {}))
        spec = store.put(
            "ExperimentSpec",
            "1.0.0",
            {
                "status": "frozen",
                "recertification_hash": recert.content_hash,
                "phase": "TRAIN_35B",
                "dataset_version_hash": frozen.get("dataset_version_hash"),
                "judge_bundle_hash": frozen.get("judge_bundle_hash"),
                "prompt_artifact_hash": prompt_art.content_hash,
                "scalarizer_artifact_hash": scalarizer_art.content_hash,
                "algorithm_artifact_hash": algorithm_art.content_hash,
            },
        )
        summary = store.put(
            "ExperimentSummary",
            "1.0.0",
            {
                "branch": config.branch,
                "experiment_spec_hash": spec.content_hash,
                "recertification_hash": recert.content_hash,
                "evidence_hashes": [x.content_hash for x in children],
                "causal_boundary": "controlled_comparison_only; black_box_descriptive_only",
                "status": "iteration_proposed",
            },
        )
        return summary, None


# Friendly aliases used by callers and hidden contract tests.
GovernorIterationConfig = BoundedGovernorConfig
GovernorIterationSnapshot = BoundedGovernorSnapshot
GovernorIterationWorkflow = BoundedGovernorWorkflow
GovernorError = BoundedGovernorError

__all__ = [
    "BoundedGovernorConfig",
    "BoundedGovernorError",
    "BoundedGovernorSnapshot",
    "BoundedGovernorWorkflow",
    "GovernorError",
    "GovernorIterationConfig",
    "GovernorIterationSnapshot",
    "GovernorIterationWorkflow",
    "InjectedBoundedGovernorCrash",
]
