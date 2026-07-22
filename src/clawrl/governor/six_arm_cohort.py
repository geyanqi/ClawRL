"""Durable fixture implementation of the Ticket 24 six-arm 35B cohort.

The implementation deliberately uses the existing content addressed artifact store,
fenced budget ledger, and fixture ClusterAdapter.  It is not an adapter for a real
cluster: ``production_readiness`` always fails closed.  This makes the fixture useful
for contract and restart tests without accidentally granting production authority.
"""

from __future__ import annotations

import fcntl
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, canonical_json_bytes, sha256_hex
from clawrl.governor.budget_ledger import BudgetLimits, BudgetUsage, FixtureBudgetLedger
from clawrl.training.cluster_lifecycle import (
    ClusterLifecycleWorkflow,
    FixtureClusterLifecycleConfig,
    InjectedClusterControllerCrash,
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_ROLES = ("control", "exploration-1", "exploration-2", "exploration-3", "exploration-4", "control-replication")
_TERMINAL = {"success", "deterministic_failure", "policy_stop"}


class SixArmCohortError(RuntimeError):
    """A six-arm cohort violated its frozen protocol or durable state."""


class InjectedSixArmCohortCrash(RuntimeError):
    """Test-only crash after a child has reached a durable cluster outcome."""


@dataclass(frozen=True, slots=True)
class SixArmCohortConfig:
    cohort_id: str
    gpu_per_arm: int = 16
    gpu_hours_millis_per_arm: int = 100
    max_active_gpu: int = 32
    controller_epoch: int = 1
    arm_outcomes: tuple[str, ...] = ("success",) * 6
    scores: tuple[int, ...] = (60, 70, 80, 90, 100, 60)
    dataset_version_hash: str | None = None
    judge_bundle_hash: str | None = None
    protocol_hash: str | None = None
    future_evaluation_dataset_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cohort_id, str) or _ID.fullmatch(self.cohort_id) is None:
            raise SixArmCohortError("cohort_id is invalid")
        if type(self.gpu_per_arm) is not int or not 1 <= self.gpu_per_arm <= 96:
            raise SixArmCohortError("gpu_per_arm is invalid")
        if type(self.max_active_gpu) is not int or not 1 <= self.max_active_gpu <= 96:
            raise SixArmCohortError("max_active_gpu is invalid")
        if self.gpu_per_arm > self.max_active_gpu:
            raise SixArmCohortError("gpu_per_arm exceeds frozen max_active_gpu")
        if type(self.controller_epoch) is not int or self.controller_epoch <= 0:
            raise SixArmCohortError("controller_epoch is invalid")
        if len(self.arm_outcomes) != 6 or any(item not in _TERMINAL for item in self.arm_outcomes):
            raise SixArmCohortError("arm_outcomes must contain six terminal fixture outcomes")
        if len(self.scores) != 6 or any(type(item) is not int for item in self.scores):
            raise SixArmCohortError("scores must contain six integers")
        if self.future_evaluation_dataset_hash is not None:
            raise SixArmCohortError("Future EvaluationDataset is outside the cohort visibility boundary")
        for field in ("dataset_version_hash", "judge_bundle_hash", "protocol_hash"):
            value = getattr(self, field)
            if value is not None and (_HASH.fullmatch(value) is None):
                raise SixArmCohortError(f"{field} is invalid")


# Friendly name used by callers that use the fixture terminology.
FixtureSixArmCohortConfig = SixArmCohortConfig


@dataclass(frozen=True, slots=True)
class SixArmCohortSnapshot:
    state: Artifact
    arms: tuple[Artifact, ...]
    promotion: Artifact | None
    protocol: Artifact

    @property
    def terminal(self) -> bool:
        return self.state.payload.get("status") == "terminal"


def _replace_ref(ref: Path, content_hash: str) -> None:
    ArtifactStore.durable_mkdir(ref.parent)
    temporary = ref.parent / f".{ref.name}.tmp"
    with temporary.open("wb") as stream:
        stream.write(f"{content_hash}\n".encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, ref)


class SixArmCohortWorkflow:
    """Run and recover one immutable, six-logical-role cohort."""

    @staticmethod
    def production_readiness(root: str | Path, *, phase: str = "TRAIN_35B") -> Artifact:
        if phase not in {"TRAIN_35B", "TRAIN_122B"}:
            raise SixArmCohortError("phase is invalid")
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": "PRODUCTION_CLUSTER_ADAPTER_UNVERIFIED", "status": "blocked"},
                    {"code": "AUTONOMY_BUDGET_APPROVAL_MISSING", "status": "blocked"},
                    {"code": "COHORT_CONTROLLER_EXTERNAL_FENCE_UNVERIFIED", "status": "blocked"},
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
        cls,
        root: str | Path,
        *,
        config: SixArmCohortConfig,
        crash_after_role: str | None = None,
        execution_profile: str = "fixture",
    ) -> SixArmCohortSnapshot:
        if execution_profile not in {"fixture", "production"}:
            raise SixArmCohortError("execution_profile must be fixture or production")
        if execution_profile == "production":
            raise SixArmCohortError(
                "production six-arm cohort is blocked until cluster and budget adapters are verified"
            )
        store = ArtifactStore(root)
        protocol, specs = cls._materialize_protocol(store, config)
        cohort_root = Path(root) / "six-arm-cohorts" / config.cohort_id
        ArtifactStore.durable_mkdir(cohort_root)
        input_artifact = store.put(
            "SixArmCohortInput",
            "1.0.0",
            {
                "cohort_id": config.cohort_id,
                "controller_epoch": config.controller_epoch,
                "protocol_hash": protocol.content_hash,
                "spec_hashes": [item.content_hash for item in specs],
            },
        )
        cls._publish_once(cohort_root / "input.ref", input_artifact.content_hash, "COHORT_INPUT_CONFLICT")
        state_ref = cohort_root / "state.ref"
        if not state_ref.exists():
            state = store.put(
                "SixArmCohortState",
                "1.0.0",
                {
                    "arm_hashes": [],
                    "cohort_id": config.cohort_id,
                    "controller_epoch": config.controller_epoch,
                    "input_hash": input_artifact.content_hash,
                    "next_index": 0,
                    "promotion_hash": None,
                    "protocol_hash": protocol.content_hash,
                    "status": "active",
                },
            )
            _replace_ref(state_ref, state.content_hash)
        else:
            durable_state = store.read(
                state_ref.read_text(encoding="ascii").strip(), expected_schema_name="SixArmCohortState"
            )
            durable_epoch = durable_state.payload.get("controller_epoch")
            if type(durable_epoch) is not int:
                raise SixArmCohortError("cohort state controller fence is invalid")
            if durable_epoch > config.controller_epoch:
                raise SixArmCohortError("STALE_COHORT_CONTROLLER_EPOCH")
        return cls._drive(root, config, protocol, specs, crash_after_role)

    @classmethod
    def resume(cls, root: str | Path, cohort_id: str, *, controller_epoch: int | None = None) -> SixArmCohortSnapshot:
        if _ID.fullmatch(cohort_id) is None:
            raise SixArmCohortError("cohort_id is invalid")
        store = ArtifactStore(root)
        cohort_root = Path(root) / "six-arm-cohorts" / cohort_id
        try:
            input_artifact = store.read(
                (cohort_root / "input.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="SixArmCohortInput",
            )
            protocol_hash = cast(str, input_artifact.payload["protocol_hash"])
            protocol = store.read(protocol_hash, expected_schema_name="ProtocolManifest")
            specs = tuple(
                store.read(item, expected_schema_name="ExperimentSpec")
                for item in cast(list[str], input_artifact.payload["spec_hashes"])
            )
        except (OSError, ArtifactCorruption, KeyError, TypeError) as error:
            raise SixArmCohortError("cohort input cannot be recovered") from error
        epoch = cast(int, input_artifact.payload["controller_epoch"])
        if controller_epoch is not None and controller_epoch < epoch:
            raise SixArmCohortError("STALE_COHORT_CONTROLLER_EPOCH")
        config = SixArmCohortConfig(cohort_id=cohort_id, controller_epoch=max(epoch, controller_epoch or epoch))
        state = store.read(
            (cohort_root / "state.ref").read_text(encoding="ascii").strip(), expected_schema_name="SixArmCohortState"
        )
        state_epoch = state.payload.get("controller_epoch")
        if type(state_epoch) is not int:
            raise SixArmCohortError("cohort state controller fence is invalid")
        if controller_epoch is not None and controller_epoch < state_epoch:
            raise SixArmCohortError("STALE_COHORT_CONTROLLER_EPOCH")
        if state_epoch > config.controller_epoch:
            config = replace(config, controller_epoch=state_epoch)
        # Outcome choices and scores are deliberately not mutable on resume; the
        # persisted ExperimentSpec is the source of truth for all child identity.
        return cls._drive(root, config, protocol, specs, None)

    @classmethod
    def _drive(
        cls,
        root: str | Path,
        config: SixArmCohortConfig,
        protocol: Artifact,
        specs: tuple[Artifact, ...],
        crash_after_role: str | None,
    ) -> SixArmCohortSnapshot:
        cohort_root = Path(root) / "six-arm-cohorts" / config.cohort_id
        lock_path = cohort_root / "controller.lock"
        ArtifactStore.durable_touch(lock_path)
        with lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                while True:
                    snapshot = cls._advance(root, config, protocol, specs, crash_after_role)
                    if snapshot is not None:
                        return snapshot
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @classmethod
    def _advance(
        cls,
        root: str | Path,
        config: SixArmCohortConfig,
        protocol: Artifact,
        specs: tuple[Artifact, ...],
        crash_after_role: str | None,
    ) -> SixArmCohortSnapshot | None:
        store = ArtifactStore(root)
        cohort_root = Path(root) / "six-arm-cohorts" / config.cohort_id
        state = store.read(
            (cohort_root / "state.ref").read_text(encoding="ascii").strip(), expected_schema_name="SixArmCohortState"
        )
        persisted_epoch = state.payload.get("controller_epoch")
        if (
            state.payload.get("protocol_hash") != protocol.content_hash
            or type(persisted_epoch) is not int
            or config.controller_epoch < persisted_epoch
        ):
            raise SixArmCohortError("cohort state protocol or controller fence changed")
        if config.controller_epoch > persisted_epoch:
            state = store.put(
                "SixArmCohortState",
                "1.0.0",
                {**state.payload, "controller_epoch": config.controller_epoch},
            )
            _replace_ref(cohort_root / "state.ref", state.content_hash)
        arm_hashes = cast(list[str], state.payload["arm_hashes"])
        index = cast(int, state.payload["next_index"])
        if state.payload.get("status") == "terminal":
            promotion_hash = state.payload.get("promotion_hash")
            promotion = (
                store.read(cast(str, promotion_hash), expected_schema_name="PromotionDecision")
                if isinstance(promotion_hash, str)
                else None
            )
            return SixArmCohortSnapshot(
                state,
                tuple(store.read(item, expected_schema_name="CohortArm") for item in arm_hashes),
                promotion,
                protocol,
            )
        if len(specs) != 6 or not 0 <= index <= 6:
            raise SixArmCohortError("six logical roles are not present")
        if index == 6:
            promotion = cls._promote(store, config, arm_hashes, protocol)
            terminal = store.put(
                "SixArmCohortState",
                "1.0.0",
                {**state.payload, "promotion_hash": promotion.content_hash, "status": "terminal"},
            )
            _replace_ref(cohort_root / "state.ref", terminal.content_hash)
            return SixArmCohortSnapshot(
                terminal,
                tuple(store.read(item, expected_schema_name="CohortArm") for item in arm_hashes),
                promotion,
                protocol,
            )

        role = _ROLES[index]
        spec = specs[index]
        schedule = cast(
            dict[str, dict[str, object]], cast(dict[str, object], protocol.payload["frozen"])["fixture_arm_schedule"]
        )
        expected = cast(str, schedule[role]["expected_outcome"])
        arm = cls._run_arm(root, config, protocol, spec, role, expected, index, crash_after_role)
        arm_hashes = [*arm_hashes, arm.content_hash]
        next_state = store.put(
            "SixArmCohortState", "1.0.0", {**state.payload, "arm_hashes": arm_hashes, "next_index": index + 1}
        )
        _replace_ref(cohort_root / "state.ref", next_state.content_hash)
        # A child crash intentionally leaves state at the previous durable index;
        # retrying resumes through ClusterAdapter idempotency and then advances.
        return None

    @classmethod
    def _run_arm(
        cls,
        root: str | Path,
        config: SixArmCohortConfig,
        protocol: Artifact,
        spec: Artifact,
        role: str,
        expected: str,
        index: int,
        crash_after_role: str | None,
    ) -> Artifact:
        store = ArtifactStore(root)
        budget_root = Path(root) / "budget-ledgers" / config.cohort_id
        action_id = f"{config.cohort_id}:{role}"
        operation_epoch = config.controller_epoch
        if (budget_root / "state.ref").exists():
            budget_snapshot = FixtureBudgetLedger.resume(root, config.cohort_id)
            campaign = budget_snapshot.ledger
            current_epoch = cast(int, budget_snapshot.state.payload["epoch"])
            if config.controller_epoch < current_epoch:
                raise SixArmCohortError("STALE_COHORT_CONTROLLER_EPOCH")
            if config.controller_epoch > current_epoch:
                active = cast(list[str], budget_snapshot.state.payload["active_reservation_hashes"])
                existing_epoch: int | None = None
                for reservation_hash in active:
                    reservation = store.read(reservation_hash, expected_schema_name="BudgetReservation")
                    plan = store.read(
                        cast(str, reservation.payload["action_plan_hash"]), expected_schema_name="ActionPlan"
                    )
                    if plan.payload.get("action_id") == action_id:
                        existing_epoch = cast(int, reservation.payload["epoch"])
                        break
                if existing_epoch is None:
                    campaign.advance_epoch(
                        config.controller_epoch,
                        expected_revision=cast(int, budget_snapshot.state.payload["revision"]),
                    )
                    campaign = FixtureBudgetLedger.resume(root, config.cohort_id).ledger
                else:
                    operation_epoch = existing_epoch
        else:
            campaign = FixtureBudgetLedger.create(
                root,
                config.cohort_id,
                BudgetLimits.from_mapping(
                    {"gpu_count": 96, "gpu_hours_millis": config.gpu_hours_millis_per_arm * 6, "jobs": 6, "cohorts": 1}
                ),
                epoch=config.controller_epoch,
            )
        revision = cast(int, FixtureBudgetLedger.resume(root, config.cohort_id).state.payload["revision"])
        usage = BudgetUsage(gpu_count=config.gpu_per_arm, gpu_hours_millis=config.gpu_hours_millis_per_arm, jobs=1)
        reservation_result = campaign.reserve(action_id, usage, epoch=operation_epoch, expected_revision=revision)
        if reservation_result.reservation is None:
            raise SixArmCohortError("BUDGET_EXHAUSTED before child submit")
        reservation = reservation_result.reservation
        if expected == "policy_stop":
            cluster_config = FixtureClusterLifecycleConfig(
                f"{config.cohort_id}:{role}", "TRAIN_35B", close_mode="force_cancel"
            )
        elif expected == "deterministic_failure":
            cluster_config = FixtureClusterLifecycleConfig(
                f"{config.cohort_id}:{role}", "TRAIN_35B", fault_operation="status", fault_directive="provider_error"
            )
        else:
            cluster_config = FixtureClusterLifecycleConfig(f"{config.cohort_id}:{role}", "TRAIN_35B")
        try:
            cluster = ClusterLifecycleWorkflow.run(root, config=cluster_config, spec_hash=spec.content_hash)
        except InjectedClusterControllerCrash as error:
            raise InjectedSixArmCohortCrash(f"child {role} crashed") from error
        status = (
            "success"
            if cluster.run_record.payload["status"] == "succeeded"
            else "policy_stop"
            if cluster.run_record.payload["status"] == "canceled"
            else "deterministic_failure"
        )
        if status != expected:
            raise SixArmCohortError("fixture child terminal outcome does not match frozen policy")
        if crash_after_role == role:
            raise InjectedSixArmCohortCrash(f"injected crash after child {role}")
        current_revision = cast(int, FixtureBudgetLedger.resume(root, config.cohort_id).state.payload["revision"])
        campaign.reconcile(reservation.content_hash, usage, epoch=operation_epoch, expected_revision=current_revision)
        if operation_epoch < config.controller_epoch:
            after_reconcile = FixtureBudgetLedger.resume(root, config.cohort_id)
            after_reconcile.ledger.advance_epoch(
                config.controller_epoch, expected_revision=cast(int, after_reconcile.state.payload["revision"])
            )
        schedule = cast(
            dict[str, dict[str, object]], cast(dict[str, object], protocol.payload["frozen"])["fixture_arm_schedule"]
        )
        score_value = schedule[role].get("score", config.scores[index])
        if type(score_value) is not int:
            raise SixArmCohortError("frozen fixture score is invalid")
        score = score_value
        return store.put(
            "CohortArm",
            "1.0.0",
            {
                "causal_boundary": "controlled" if role in {"control", "control-replication"} else "black_box",
                "cluster_run_id": cluster.run_record.payload["run_id"],
                "cluster_run_record_hash": cluster.run_record.content_hash,
                "cohort_id": config.cohort_id,
                "experiment_spec_hash": spec.content_hash,
                "harness_steps": ["submit", "status", "logs", "artifact", "checkpoint"],
                "independent_seed": role == "control-replication",
                "logical_role": role,
                "promotion_score": score,
                "protocol_hash": protocol.content_hash,
                "queue_position": index,
                "queued_before_submit": index > 0,
                "resource_reused": index > 0,
                "reservation_hash": reservation.content_hash,
                "status": status,
            },
        )

    @staticmethod
    def _promote(
        store: ArtifactStore, config: SixArmCohortConfig, arm_hashes: list[str], protocol: Artifact
    ) -> Artifact:
        arms = [store.read(item, expected_schema_name="CohortArm") for item in arm_hashes]
        if (
            len(arms) != 6
            or {item.payload.get("logical_role") for item in arms} != set(_ROLES)
            or any(item.payload.get("status") not in _TERMINAL for item in arms)
        ):
            raise SixArmCohortError("promotion requires all six terminal logical roles")
        eligible = [item for item in arms if item.payload.get("status") == "success"]
        if not eligible:
            raise SixArmCohortError("promotion requires one successful candidate")
        top_score = max(cast(int, item.payload["promotion_score"]) for item in eligible)
        top = [item for item in eligible if item.payload.get("promotion_score") == top_score]
        if len(top) != 1:
            raise SixArmCohortError("promotion requires a unique winner")
        winner = top[0]
        return store.put(
            "PromotionDecision",
            "1.0.0",
            {
                "causal_boundary": "controlled_comparison_only; black_box_descriptive_only",
                "cohort_id": config.cohort_id,
                "eligible_arm_hashes": [item.content_hash for item in eligible],
                "protocol_hash": protocol.content_hash,
                "status": "promoted",
                "winner_arm_hash": winner.content_hash,
                "winner_logical_role": winner.payload["logical_role"],
            },
        )

    @classmethod
    def _materialize_protocol(
        cls, store: ArtifactStore, config: SixArmCohortConfig
    ) -> tuple[Artifact, tuple[Artifact, ...]]:
        dataset = (
            store.read(config.dataset_version_hash, expected_schema_name="DatasetVersion")
            if config.dataset_version_hash
            else store.put(
                "DatasetVersion",
                "1.0.0",
                {
                    "dataset_id": "ticket24-train-35b",
                    "future_evaluation": False,
                    "purpose": "training_allowed",
                    "version": "v1",
                },
            )
        )
        judge = (
            store.read(config.judge_bundle_hash, expected_schema_name="JudgeBundle")
            if config.judge_bundle_hash
            else store.put(
                "JudgeBundle",
                "1.0.0",
                {"bundle_id": "ticket24-judge", "dataset_hash": dataset.content_hash, "purpose": "training-evaluation"},
            )
        )
        if dataset.payload.get("purpose") != "training_allowed" or dataset.payload.get("future_evaluation") is True:
            raise SixArmCohortError("cohort DatasetVersion is not training-only")
        if judge.payload.get("dataset_hash") != dataset.content_hash:
            raise SixArmCohortError("JudgeBundle does not bind the frozen DatasetVersion")
        frozen = {
            "aggregation": "calibrated-mean/v1",
            "algorithm": "fixture-ppo/v1",
            "dataset_version_hash": dataset.content_hash,
            "generation": {"max_tokens": 256, "temperature_millis": 700},
            "judge_bundle_hash": judge.content_hash,
            "monitoring_policy": {"max_kl_millis": 250, "max_length_ratio_millis": 2000},
            "prompt": "ticket24-fixed-prompt/v1",
            "promotion_policy": {"requires_all_terminal": True, "unique_winner": True},
            "resources": {"gpu_count": config.gpu_per_arm, "phase": "TRAIN_35B"},
            "reward_schema": "ticket24-reward/v1",
            "scalarizer": "ticket24-scalarizer/v1",
            "fixture_arm_schedule": {
                role: {"expected_outcome": config.arm_outcomes[index], "score": config.scores[index]}
                for index, role in enumerate(_ROLES)
            },
        }
        cohort_spec = store.put(
            "CohortSpec",
            "1.0.0",
            {
                "cohort_id": config.cohort_id,
                "control_arm_id": "control",
                "exploration_arm_ids": list(_ROLES[1:5]),
                "logical_roles": list(_ROLES),
                "replication_of": "control",
                "status": "frozen",
            },
        )
        policy = store.put(
            "CohortPolicy",
            "1.0.0",
            {
                "cohort_id": config.cohort_id,
                "logical_roles": list(_ROLES),
                "status": "frozen",
                "resources": {"gpu_hard_cap": 96, "max_active_gpu": config.max_active_gpu},
            },
        )
        promotion = store.put("PromotionPolicy", "1.0.0", cast(dict[str, object], frozen["promotion_policy"]))
        monitoring = store.put("MonitoringPolicy", "1.0.0", cast(dict[str, object], frozen["monitoring_policy"]))
        protocol_body = {
            "cohort_policy_hash": policy.content_hash,
            "cohort_spec_hash": cohort_spec.content_hash,
            "frozen": frozen,
            "monitoring_policy_hash": monitoring.content_hash,
            "promotion_policy_hash": promotion.content_hash,
            "roles": list(_ROLES),
            "version": "ticket24-protocol/1.0.0",
        }
        protocol_digest = sha256_hex(canonical_json_bytes(protocol_body))
        protocol = store.put("ProtocolManifest", "1.0.0", {**protocol_body, "protocol_hash": protocol_digest})
        if config.protocol_hash is not None and config.protocol_hash != protocol.content_hash:
            raise SixArmCohortError("protocol hash does not match frozen inputs")
        specs: list[Artifact] = []
        for index, role in enumerate(_ROLES):
            role_frozen = {
                **frozen,
                "arm_role": role,
                "seed": 1000 + index,
                "protocol_hash": protocol.content_hash,
                "run_id": f"{config.cohort_id}:{role}",
            }
            if role.startswith("exploration-"):
                role_frozen["algorithm"] = f"fixture-ppo/exploration-{index}"
            specs.append(store.put("ExperimentSpec", "1.0.0", {**role_frozen, "status": "frozen"}))
        return protocol, tuple(specs)

    @staticmethod
    def _publish_once(ref: Path, content_hash: str, code: str) -> None:
        try:
            ArtifactStore._publish(ref, f"{content_hash}\n".encode("ascii"))
        except Exception as error:
            try:
                if ref.read_text(encoding="ascii").strip() == content_hash:
                    return
            except OSError:
                pass
            raise SixArmCohortError(f"{code}: immutable input changed") from error


__all__ = [
    "FixtureSixArmCohortConfig",
    "InjectedSixArmCohortCrash",
    "SixArmCohortConfig",
    "SixArmCohortError",
    "SixArmCohortSnapshot",
    "SixArmCohortWorkflow",
]
