"""Durable, production-shaped recovery of a complete training step.

Ticket 18 deliberately sits above the classic and v1 grouped-reward seams.  A
step is not ready when one UID has produced a tensor: all sixteen UID groups
must have their frozen 128-slot barrier.  This module provides the small
controller state machine used by the fixture tests.  Every controller
decision is represented by an immutable content-addressed artifact; the
``report.ref`` file is only a pointer to the final, validated report.

The fixture accepts fault names so tests can exercise recovery without a real
cluster or CFS.  Faults which can be recovered are recorded as durable fault
events.  A conflicting result or exhausted retry policy produces a typed stop
request and never publishes a step-ready artifact (nor a partial/neutral
reward tensor).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from clawrl.artifacts import (
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    UnknownSchemaMajor,
    canonical_json_bytes,
    sha256_hex,
)
from clawrl.router.trace_router import (
    FixtureTraceRouter,
    RouterCapacityConfig,
    RouterFenceAuthority,
    TraceRouterError,
)
from clawrl.training.classic_grouped_reward import (
    ClassicGroupedRewardConfig,
    ClassicGroupedRewardError,
    ClassicGroupedRewardWorkflow,
)
from clawrl.training.expected_trajectory_set import ExpectedTrajectorySetError, ExpectedTrajectorySetWorkflow
from clawrl.training.v1_grouped_reward import V1GroupedRewardConfig, V1GroupedRewardError, V1GroupedRewardWorkflow

_HASH = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_UID_COUNT = 16
_ROLLOUT_COUNT = 128
_SLOT_COUNT = _UID_COUNT * _ROLLOUT_COUNT
_FAULT_ALIASES = {
    "duplicate_result": "duplicate",
    "conflicting_result": "conflict",
    "out_of_order_attempt": "out_of_order",
    "resolver_restart": "resolver_restart",
    "router_restart": "router_restart",
    "controller_restart": "controller_restart",
    "late_result_quarantine": "late_result",
    "attempt_exhaustion": "exhaustion",
    "attempts_exhausted": "exhaustion",
    "retry_exhaustion": "exhaustion",
}
_VALID_FAULTS = {
    None,
    "duplicate",
    "conflict",
    "out_of_order",
    "lease_expiry",
    "resolver_restart",
    "router_restart",
    "controller_restart",
    "late_result",
    "exhaustion",
}
_EXPERIMENT_FIELDS = {
    "aggregation",
    "dataset_version_hash",
    "dataset_version_id",
    "experiment_id",
    "judge_bundle_hash",
    "trace_set_hash",
}
_STATE_FIELDS = {
    "attempt_count",
    "attempt_hashes",
    "expected_count",
    "global_step",
    "grouped_report_hash",
    "late_quarantine_hashes",
    "resolver_epoch",
    "reward_hashes",
    "router_restart_count",
    "run_id",
    "status",
    "uid",
    "uid_ordinal",
}


class StepReadyRecoveryError(RuntimeError):
    """The step-ready controller failed closed before an optimizer update."""


class StepReadyTerminalStop(StepReadyRecoveryError):
    """A typed terminal stop request was durably written."""


class InjectedStepReadyRecoveryCrash(StepReadyRecoveryError):
    """Synthetic process/controller crash after a UID group commits."""


def _hash(value: object, field: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise StepReadyRecoveryError(f"{field} is not a content hash")
    return cast(str, value)


def _id(value: object, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(cast(str, value)) is None:
        raise StepReadyRecoveryError(f"{field} is invalid")
    return cast(str, value)


def _fault(value: str | None) -> str | None:
    return None if value is None else _FAULT_ALIASES.get(value, value)


@dataclass(frozen=True, slots=True)
class StepReadyRecoveryConfig:
    run_id: str
    global_step: int
    experiment_spec_hash: str
    trainer_path: str = "classic"
    resolver_epoch: int = 1
    chunk_size: int = 16
    capacity: RouterCapacityConfig = RouterCapacityConfig(5)
    fault_kind: str | None = None
    fault_index: int | None = None
    crash_after_uid: int | None = None
    max_attempts: int = 3
    lease_ticks: int = 4

    def __post_init__(self) -> None:
        _id(self.run_id, "run_id")
        if type(self.global_step) is not int or self.global_step < 0:
            raise StepReadyRecoveryError("global_step is invalid")
        _hash(self.experiment_spec_hash, "experiment_spec_hash")
        if self.trainer_path not in {"classic", "v1"}:
            raise StepReadyRecoveryError("trainer_path is invalid")
        if type(self.resolver_epoch) is not int or self.resolver_epoch < 1:
            raise StepReadyRecoveryError("resolver_epoch is invalid")
        if type(self.chunk_size) is not int or not 1 <= self.chunk_size <= _ROLLOUT_COUNT:
            raise StepReadyRecoveryError("chunk_size is invalid")
        normalized = _fault(self.fault_kind)
        if normalized not in _VALID_FAULTS:
            raise StepReadyRecoveryError("unknown recovery fault")
        if normalized is None and self.fault_index is not None:
            raise StepReadyRecoveryError("fault_index requires a fault_kind")
        if normalized in {"duplicate", "out_of_order", "lease_expiry"} and self.fault_index is None:
            object.__setattr__(self, "fault_index", 0)
        if self.fault_index is not None and (
            type(self.fault_index) is not int or not 0 <= self.fault_index < _SLOT_COUNT
        ):
            raise StepReadyRecoveryError("fault_index is invalid")
        if self.crash_after_uid is not None and (
            type(self.crash_after_uid) is not int or not 0 <= self.crash_after_uid < _UID_COUNT
        ):
            raise StepReadyRecoveryError("crash_after_uid is invalid")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise StepReadyRecoveryError("max_attempts is invalid")
        if type(self.lease_ticks) is not int or self.lease_ticks < 1:
            raise StepReadyRecoveryError("lease_ticks is invalid")
        object.__setattr__(self, "fault_kind", normalized)


# Short aliases used in a few trainer integrations.
StepRecoveryConfig = StepReadyRecoveryConfig


@dataclass(frozen=True, slots=True)
class StepReadyRecoverySnapshot:
    report: Artifact
    step_ready: Artifact
    uid_states: tuple[Artifact, ...]
    rewards: tuple[Artifact, ...]


class StepReadyRecoveryWorkflow:
    """Recover all 2,048 frozen slots into one durable ``StepReady``."""

    @staticmethod
    def _namespace(root: str | Path, config: StepReadyRecoveryConfig) -> Path:
        return Path(root) / "step-ready-recovery" / config.run_id / str(config.global_step) / config.trainer_path

    @staticmethod
    def _effective_epoch(config: StepReadyRecoveryConfig) -> int:
        return (
            config.resolver_epoch + 1
            if config.fault_kind
            in {
                "resolver_restart",
                "router_restart",
                "controller_restart",
            }
            else config.resolver_epoch
        )

    @classmethod
    def production_readiness(cls, root: str | Path, *, phase: str = "TRAIN_35B") -> Artifact:
        if phase not in {"TRAIN_35B", "TRAIN_122B"}:
            raise StepReadyRecoveryError("production phase is invalid")
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": "PRODUCTION_PERMANENT_TRACE_GOVERNANCE_UNGREEN", "status": "blocked"},
                    {"code": "PRODUCTION_CFS_ROUTER_RESOLVER_UNVERIFIED", "status": "blocked"},
                    {"code": "PRODUCTION_TRAINER_STEP_READY_UNVERIFIED", "status": "blocked"},
                ],
                "execution_profile": "production",
                "phase": phase,
                "scorer_request_attempted": False,
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: StepReadyRecoveryConfig,
        expected_set: Artifact,
        judge_packs: Mapping[str, Artifact] | Sequence[Artifact],
        experiment_spec: Artifact,
        crash_after_uid: int | None = None,
    ) -> StepReadyRecoverySnapshot:
        store = ArtifactStore(root)
        packs = cls._pack_map(judge_packs)
        cls._validate_inputs(root, config, expected_set, packs, experiment_spec)
        effective_epoch = cls._effective_epoch(config)
        try:
            authority = RouterFenceAuthority(root, config.run_id)
            if effective_epoch != config.resolver_epoch:
                try:
                    current_epoch = cast(int, authority.current().payload["epoch"])
                except TraceRouterError:
                    current_epoch = None
                if current_epoch is None:
                    authority.ensure(config.resolver_epoch)
                    authority.advance(effective_epoch)
                elif current_epoch == config.resolver_epoch:
                    authority.advance(effective_epoch)
                elif current_epoch != effective_epoch:
                    raise TraceRouterError("STALE_ROUTER_RESOLVER_EPOCH")
                else:
                    authority.ensure(effective_epoch)
            else:
                authority.ensure(effective_epoch)
        except TraceRouterError as error:
            raise StepReadyRecoveryError("router fencing epoch is stale") from error
        namespace = cls._namespace(root, config)
        run_root = Path(root) / "step-ready-recovery" / config.run_id / str(config.global_step)
        if run_root.exists():
            for sibling in run_root.iterdir():
                if sibling.name != config.trainer_path and (sibling / "report.ref").exists():
                    raise StepReadyRecoveryError("trainer path mutation changed durable run")
        report_ref = namespace / "report.ref"
        if report_ref.exists():
            return cls._resume_checked(root, config, expected_set, packs, experiment_spec)

        # Fatal faults are rejected before any grouped reward call.  This is
        # important: an incomplete step never exposes a partial tensor.
        if config.fault_kind in {"conflict", "exhaustion"}:
            reason = "CONFLICTING_RESULT" if config.fault_kind == "conflict" else "REWARD_ATTEMPTS_EXHAUSTED"
            raise cls._terminal_stop(root, config, expected_set, experiment_spec, reason)
        if (
            config.fault_kind in {"duplicate", "out_of_order", "lease_expiry", "late_result"}
            and config.max_attempts < 2
        ):
            raise cls._terminal_stop(root, config, expected_set, experiment_spec, "REWARD_ATTEMPTS_EXHAUSTED")

        ArtifactStore.durable_mkdir(namespace)
        frozen = store.put(
            "StepReadyFrozenSet",
            "1.0.0",
            {
                "expected_set_hash": expected_set.content_hash,
                "expected_slot_count": _SLOT_COUNT,
                "experiment_spec_hash": experiment_spec.content_hash,
                "global_step": config.global_step,
                "resolver_epoch": effective_epoch,
                "run_id": config.run_id,
                "trainer_path": config.trainer_path,
                "uid_count": _UID_COUNT,
            },
        )
        cls._publish_ref(namespace / "frozen.ref", frozen)

        uid_values = cls._uids(expected_set)
        if len(uid_values) != _UID_COUNT:
            raise StepReadyRecoveryError("step-ready requires exactly 16 UIDs")
        states: list[Artifact] = []
        rewards: list[Artifact] = []
        stop_after = crash_after_uid if crash_after_uid is not None else config.crash_after_uid
        crash_marker = namespace / "crash-injected.ref"
        for ordinal, uid in enumerate(uid_values):
            state_ref = namespace / "uids" / uid / "state.ref"
            try:
                if state_ref.exists():
                    state = store.read(
                        state_ref.read_text(encoding="ascii").strip(), expected_schema_name="RecoveryUidState"
                    )
                    state_payload = state.payload
                    state_reward_values = state_payload.get("reward_hashes")
                    state_attempt_values = state_payload.get("attempt_hashes")
                    if (
                        state.schema_version != "1.0.0"
                        or set(state_payload) != _STATE_FIELDS
                        or state_payload.get("uid") != uid
                        or state_payload.get("uid_ordinal") != ordinal
                        or state_payload.get("status") != "resolved"
                        or state_payload.get("expected_count") != _ROLLOUT_COUNT
                        or state_payload.get("resolver_epoch") != effective_epoch
                        or type(state_payload.get("attempt_count")) is not int
                        or not isinstance(state_reward_values, list)
                        or len(state_reward_values) != _ROLLOUT_COUNT
                        or any(type(value) is not str for value in state_reward_values)
                        or not isinstance(state_attempt_values, list)
                        or any(type(value) is not str for value in state_attempt_values)
                        or len(state_attempt_values) != cast(int, state_payload.get("attempt_count", 0)) * 2
                        or cast(int, state_payload.get("attempt_count", 0)) > config.max_attempts
                    ):
                        raise StepReadyRecoveryError("durable UID state is incomplete")
                    uid_rewards = tuple(
                        store.read(
                            value,
                            expected_schema_name=(
                                "ClassicGroupedReward" if config.trainer_path == "classic" else "V1TransferQueueReward"
                            ),
                        )
                        for value in cast(list[str], state_reward_values)
                    )
                    cls._validate_uid_commitment(
                        store, state, uid_rewards, config, expected_set, packs[uid], experiment_spec
                    )
                else:
                    pack = packs[uid]
                    uid_config = cls._group_config(config, uid)
                    if config.trainer_path == "classic":
                        grouped_classic = ClassicGroupedRewardWorkflow.run(
                            root,
                            config=cast(ClassicGroupedRewardConfig, uid_config),
                            expected_set=expected_set,
                            judge_pack=pack,
                            experiment_spec=experiment_spec,
                        )
                        uid_rewards = grouped_classic.rewards
                        grouped_report_hash = grouped_classic.report.content_hash
                    else:
                        grouped_v1 = V1GroupedRewardWorkflow.run(
                            root,
                            config=cast(V1GroupedRewardConfig, uid_config),
                            expected_set=expected_set,
                            judge_pack=pack,
                            experiment_spec=experiment_spec,
                        )
                        uid_rewards = grouped_v1.rewards
                        grouped_report_hash = grouped_v1.report.content_hash
                    attempt_hashes = cls._attempt_artifacts(store, config, uid, ordinal, uid_rewards)
                    quarantine_hashes: list[str] = []
                    if config.fault_kind in {"out_of_order", "lease_expiry", "late_result"} and (
                        config.fault_index is None or config.fault_index // _ROLLOUT_COUNT == ordinal
                    ):
                        quarantine = store.put(
                            "LateResultQuarantine",
                            "1.0.0",
                            {
                                "late_result_hash": uid_rewards[-1].content_hash,
                                "reason": (
                                    "lease_expired"
                                    if config.fault_kind == "lease_expiry"
                                    else "out_of_order"
                                    if config.fault_kind == "out_of_order"
                                    else "resolved_reward_already_committed"
                                ),
                                "status": "late_quarantined",
                                "run_id": config.run_id,
                                "global_step": config.global_step,
                                "uid": uid,
                            },
                        )
                        quarantine_hashes.append(quarantine.content_hash)
                    state = store.put(
                        "RecoveryUidState",
                        "1.0.0",
                        {
                            "attempt_count": len(attempt_hashes) // 2,
                            "attempt_hashes": attempt_hashes,
                            "expected_count": _ROLLOUT_COUNT,
                            "global_step": config.global_step,
                            "grouped_report_hash": grouped_report_hash,
                            "late_quarantine_hashes": quarantine_hashes,
                            "resolver_epoch": effective_epoch,
                            "reward_hashes": [item.content_hash for item in uid_rewards],
                            "router_restart_count": 1 if config.fault_kind == "router_restart" else 0,
                            "run_id": config.run_id,
                            "status": "resolved",
                            "uid": uid,
                            "uid_ordinal": ordinal,
                        },
                    )
                    cls._publish_ref(state_ref, state)
                    cls._validate_uid_commitment(store, state, uid_rewards, config, expected_set, pack, experiment_spec)
                states.append(state)
                rewards.extend(uid_rewards)
            except (ClassicGroupedRewardError, V1GroupedRewardError, StepReadyRecoveryError) as error:
                raise StepReadyRecoveryError(f"UID {uid} did not reach trace-ready") from error
            if stop_after is not None and ordinal == stop_after and not crash_marker.exists():
                crash = store.put(
                    "RecoveryCrashEvent",
                    "1.0.0",
                    {"uid": uid, "uid_ordinal": ordinal, "status": "injected", "run_id": config.run_id},
                )
                cls._publish_ref(crash_marker, crash)
                raise InjectedStepReadyRecoveryCrash("controller crashed after durable UID state")

        fault_hashes = cls._fault_events(store, namespace, config, states, rewards)
        if len(states) != _UID_COUNT or len(rewards) != _SLOT_COUNT:
            raise StepReadyRecoveryError("step-ready barrier is incomplete")
        reward_hashes = [item.content_hash for item in rewards]
        step_ready = store.put(
            "StepReady",
            "1.0.0",
            {
                "expected_slot_count": _SLOT_COUNT,
                "expected_set_hash": expected_set.content_hash,
                "experiment_spec_hash": experiment_spec.content_hash,
                "global_step": config.global_step,
                "reward_hashes": reward_hashes,
                "reward_root_hash": sha256_hex(canonical_json_bytes(reward_hashes)),
                "run_id": config.run_id,
                "status": "step_ready",
                "trainer_path": config.trainer_path,
                "uid_state_hashes": [item.content_hash for item in states],
                "uid_count": _UID_COUNT,
                "optimizer_update": False,
            },
        )
        cls._publish_ref(namespace / "step-ready.ref", step_ready)
        report = store.put(
            "StepReadyRecoveryReport",
            "1.0.0",
            {
                "capacity": config.capacity.max_global_subthreads,
                "chunk_size": config.chunk_size,
                "crash_after_uid": stop_after,
                "controller_restart_count": 1 if config.fault_kind == "controller_restart" else 0,
                "expected_set_hash": expected_set.content_hash,
                "expected_slot_count": _SLOT_COUNT,
                "experiment_spec_hash": experiment_spec.content_hash,
                "fault_index": config.fault_index,
                "fault_kind": config.fault_kind,
                "fault_event_hashes": fault_hashes,
                "global_step": config.global_step,
                "max_attempts": config.max_attempts,
                "lease_ticks": config.lease_ticks,
                "judge_pack_hashes": {uid: packs[uid].content_hash for uid in uid_values},
                "production_readiness": "blocked",
                "queue_only": True,
                "account_creation_requested": False,
                "expansion_requested": False,
                "reward_root_hash": step_ready.payload["reward_root_hash"],
                "run_id": config.run_id,
                "status": "step_ready",
                "step_ready_hash": step_ready.content_hash,
                "trainer_path": config.trainer_path,
                "uid_count": _UID_COUNT,
                "uid_state_hashes": [item.content_hash for item in states],
                "resolver_epoch": effective_epoch,
            },
        )
        cls._publish_ref(report_ref, report)
        return cls.resume(root, config)

    @classmethod
    def resume(cls, root: str | Path, config: StepReadyRecoveryConfig) -> StepReadyRecoverySnapshot:
        run_root = Path(root) / "step-ready-recovery" / config.run_id / str(config.global_step)
        if run_root.exists():
            for sibling in run_root.iterdir():
                if sibling.name != config.trainer_path and (sibling / "report.ref").exists():
                    raise StepReadyRecoveryError("trainer path mutation changed durable run")
        return cls._resume_checked(root, config, None, None, None)

    @classmethod
    def _resume_checked(
        cls,
        root: str | Path,
        config: StepReadyRecoveryConfig,
        expected_set: Artifact | None,
        packs: Mapping[str, Artifact] | None,
        experiment_spec: Artifact | None,
    ) -> StepReadyRecoverySnapshot:
        store = ArtifactStore(root)
        namespace = cls._namespace(root, config)
        effective_epoch = cls._effective_epoch(config)
        try:
            try:
                RouterFenceAuthority(root, config.run_id).ensure(effective_epoch)
            except TraceRouterError as error:
                raise StepReadyRecoveryError("router fencing epoch is stale") from error
            report = store.read(
                (namespace / "report.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="StepReadyRecoveryReport",
            )
            step_ready = store.read(cast(str, report.payload["step_ready_hash"]), expected_schema_name="StepReady")
            expected_report_fields = {
                "capacity",
                "chunk_size",
                "crash_after_uid",
                "controller_restart_count",
                "expected_set_hash",
                "expected_slot_count",
                "experiment_spec_hash",
                "fault_event_hashes",
                "fault_kind",
                "global_step",
                "max_attempts",
                "lease_ticks",
                "fault_index",
                "judge_pack_hashes",
                "production_readiness",
                "queue_only",
                "account_creation_requested",
                "expansion_requested",
                "reward_root_hash",
                "run_id",
                "status",
                "step_ready_hash",
                "trainer_path",
                "uid_count",
                "uid_state_hashes",
                "resolver_epoch",
            }
            expected_step_fields = {
                "expected_slot_count",
                "expected_set_hash",
                "experiment_spec_hash",
                "global_step",
                "reward_hashes",
                "reward_root_hash",
                "run_id",
                "status",
                "trainer_path",
                "uid_state_hashes",
                "uid_count",
                "optimizer_update",
            }
            if report.schema_version != "1.0.0" or set(report.payload) != expected_report_fields:
                raise StepReadyRecoveryError("step-ready report schema is invalid")
            if step_ready.schema_version != "1.0.0" or set(step_ready.payload) != expected_step_fields:
                raise StepReadyRecoveryError("step-ready schema is invalid")
            if expected_set is not None and report.payload.get("expected_set_hash") != expected_set.content_hash:
                raise StepReadyRecoveryError("expected set identity changed on resume")
            expected_snapshot = ExpectedTrajectorySetWorkflow.resume(root, config.run_id, config.global_step)
            if expected_snapshot.expected_set.content_hash != report.payload.get("expected_set_hash"):
                raise StepReadyRecoveryError("expected set identity changed on resume")
            expected_manifests: dict[tuple[str, int], str] = {}
            for manifest in expected_snapshot.manifests:
                slot = manifest.payload.get("slot_key")
                if isinstance(slot, dict) and type(slot.get("uid")) is str and type(slot.get("rollout_index")) is int:
                    expected_manifests[(cast(str, slot["uid"]), cast(int, slot["rollout_index"]))] = (
                        manifest.content_hash
                    )
            expected_uid_set = {uid for uid, _index in expected_manifests}
            stored_pack_map = report.payload.get("judge_pack_hashes")
            if not isinstance(stored_pack_map, dict) or set(stored_pack_map) != expected_uid_set:
                raise StepReadyRecoveryError("step-ready UID set does not match ExpectedTrajectorySet")
            if (
                experiment_spec is not None
                and report.payload.get("experiment_spec_hash") != experiment_spec.content_hash
            ):
                raise StepReadyRecoveryError("experiment identity changed on resume")
            if report.payload.get("run_id") != config.run_id or report.payload.get("global_step") != config.global_step:
                raise StepReadyRecoveryError("step-ready run identity changed")
            if report.payload.get("experiment_spec_hash") != config.experiment_spec_hash:
                raise StepReadyRecoveryError("experiment identity changed on resume")
            if (
                report.payload.get("trainer_path") != config.trainer_path
                or report.payload.get("fault_kind") != config.fault_kind
                or report.payload.get("fault_index") != config.fault_index
                or report.payload.get("capacity") != config.capacity.max_global_subthreads
                or report.payload.get("chunk_size") != config.chunk_size
                or report.payload.get("crash_after_uid") != config.crash_after_uid
                or report.payload.get("resolver_epoch") != effective_epoch
                or report.payload.get("max_attempts") != config.max_attempts
                or report.payload.get("lease_ticks") != config.lease_ticks
            ):
                raise StepReadyRecoveryError("step-ready config mutation changed durable run")
            if packs is not None:
                stored_packs = report.payload.get("judge_pack_hashes")
                if (
                    not isinstance(stored_packs, dict)
                    or {uid: packs[uid].content_hash for uid in packs} != stored_packs
                ):
                    raise StepReadyRecoveryError("JudgePack identity changed on resume")
            if report.payload.get("status") != "step_ready" or step_ready.payload.get("status") != "step_ready":
                raise StepReadyRecoveryError("step-ready status is invalid")
            if (
                step_ready.payload.get("optimizer_update") is not False
                or report.payload.get("queue_only") is not True
                or report.payload.get("account_creation_requested") is not False
                or report.payload.get("expansion_requested") is not False
            ):
                raise StepReadyRecoveryError("step-ready side-effect policy is invalid")
            fault_values = report.payload.get("fault_event_hashes")
            if not isinstance(fault_values, list) or any(type(value) is not str for value in fault_values):
                raise StepReadyRecoveryError("fault event lineage is invalid")
            if config.fault_kind is None and fault_values:
                raise StepReadyRecoveryError("unexpected fault events on clean step")
            if config.fault_kind is not None:
                if not fault_values:
                    raise StepReadyRecoveryError("recovery fault event is missing")
                event = store.read(cast(str, fault_values[0]), expected_schema_name="RecoveryFaultEvent")
                if event.payload.get("fault_kind") != config.fault_kind:
                    raise StepReadyRecoveryError("recovery fault identity changed")
            hashes = report.payload.get("uid_state_hashes")
            if not isinstance(hashes, list) or len(hashes) != _UID_COUNT:
                raise StepReadyRecoveryError("step-ready UID barrier is incomplete")
            states = tuple(store.read(cast(str, value), expected_schema_name="RecoveryUidState") for value in hashes)
            if len({cast(str, state.payload.get("uid")) for state in states}) != _UID_COUNT:
                raise StepReadyRecoveryError("step-ready UID identity is duplicated")
            reward_hashes = step_ready.payload.get("reward_hashes")
            if (
                not isinstance(reward_hashes, list)
                or len(reward_hashes) != _SLOT_COUNT
                or any(type(value) is not str for value in reward_hashes)
            ):
                raise StepReadyRecoveryError("step-ready reward barrier is incomplete")
            typed_reward_hashes = cast(list[str], reward_hashes)
            reward_schema = "ClassicGroupedReward" if config.trainer_path == "classic" else "V1TransferQueueReward"
            rewards = tuple(store.read(value, expected_schema_name=reward_schema) for value in typed_reward_hashes)
            if [item.content_hash for item in rewards] != typed_reward_hashes:
                raise StepReadyRecoveryError("step-ready reward identity changed")
            if step_ready.payload.get("reward_root_hash") != sha256_hex(canonical_json_bytes(reward_hashes)):
                raise StepReadyRecoveryError("step-ready reward root changed")
            state_reward_hashes: list[str] = []
            state_reward_offset = 0
            for state in states:
                payload = state.payload
                state_fields = {
                    "attempt_count",
                    "attempt_hashes",
                    "expected_count",
                    "global_step",
                    "grouped_report_hash",
                    "late_quarantine_hashes",
                    "resolver_epoch",
                    "reward_hashes",
                    "router_restart_count",
                    "run_id",
                    "status",
                    "uid",
                    "uid_ordinal",
                }
                if (
                    state.schema_version != "1.0.0"
                    or set(payload) != state_fields
                    or payload.get("uid_ordinal") != len(state_reward_hashes) // _ROLLOUT_COUNT
                ):
                    raise StepReadyRecoveryError("UID durable state schema is invalid")
                grouped_schema = (
                    "ClassicGroupedRewardReport" if config.trainer_path == "classic" else "V1GroupedRewardReport"
                )
                grouped_report = store.read(
                    cast(str, payload["grouped_report_hash"]), expected_schema_name=grouped_schema
                )
                if (
                    grouped_report.payload.get("run_id") != config.run_id
                    or grouped_report.payload.get("global_step") != config.global_step
                    or grouped_report.payload.get("uid") != payload.get("uid")
                    or grouped_report.payload.get("expected_set_hash") != report.payload.get("expected_set_hash")
                    or grouped_report.payload.get("experiment_spec_hash") != config.experiment_spec_hash
                ):
                    raise StepReadyRecoveryError("grouped report lineage changed")
                state_hashes = payload.get("reward_hashes")
                if (
                    payload.get("status") != "resolved"
                    or payload.get("expected_count") != _ROLLOUT_COUNT
                    or payload.get("run_id") != config.run_id
                    or payload.get("global_step") != config.global_step
                    or type(payload.get("uid_ordinal")) is not int
                    or not 0 <= cast(int, payload.get("uid_ordinal")) < _UID_COUNT
                    or not isinstance(state_hashes, list)
                    or len(state_hashes) != _ROLLOUT_COUNT
                    or any(type(value) is not str for value in state_hashes)
                ):
                    raise StepReadyRecoveryError("partial UID reward state cannot be recovered")
                attempt_hashes = payload.get("attempt_hashes")
                if (
                    not isinstance(attempt_hashes, list)
                    or any(type(value) is not str for value in attempt_hashes)
                    or len(attempt_hashes) != cast(int, payload.get("attempt_count", 0)) * 2
                ):
                    raise StepReadyRecoveryError("attempt transition lineage is incomplete")
                typed_attempt_hashes = cast(list[str], attempt_hashes)
                targeted_uid = config.fault_index is not None and config.fault_index // _ROLLOUT_COUNT == payload.get(
                    "uid_ordinal"
                )
                for index in range(0, len(typed_attempt_hashes), 2):
                    lease = store.read(typed_attempt_hashes[index], expected_schema_name="RecoveryAttemptLease")
                    result = store.read(typed_attempt_hashes[index + 1], expected_schema_name="RecoveryAttemptResult")
                    lease_payload = lease.payload
                    result_payload = result.payload
                    ordinal = index // 2 + 1
                    if (
                        set(lease_payload)
                        != {
                            "attempt_ordinal",
                            "fault_index",
                            "fault_kind",
                            "injected_slot_index",
                            "lease_expires_tick",
                            "lease_ticks",
                            "resolver_epoch",
                            "run_id",
                            "status",
                            "uid",
                        }
                        or set(result_payload)
                        != {
                            "attempt_hash",
                            "attempt_ordinal",
                            "fault_index",
                            "fault_kind",
                            "injected_slot_index",
                            "result_hash",
                            "run_id",
                            "status",
                            "uid",
                        }
                        or lease_payload.get("attempt_ordinal") != ordinal
                        or result_payload.get("attempt_ordinal") != ordinal
                        or lease_payload.get("lease_ticks") != config.lease_ticks
                        or lease_payload.get("lease_expires_tick") != config.lease_ticks * ordinal
                        or lease_payload.get("resolver_epoch") != effective_epoch
                        or lease_payload.get("status")
                        != (
                            "expired"
                            if targeted_uid and config.fault_kind == "lease_expiry" and ordinal == 1
                            else "committed"
                        )
                        or lease_payload.get("fault_kind") != config.fault_kind
                        or lease_payload.get("fault_index") != config.fault_index
                        or result_payload.get("fault_kind") != config.fault_kind
                        or result_payload.get("fault_index") != config.fault_index
                        or type(result_payload.get("injected_slot_index")) is not int
                        or not 0 <= cast(int, result_payload.get("injected_slot_index")) < _ROLLOUT_COUNT
                        or result_payload.get("status")
                        != (
                            "duplicate_deduplicated"
                            if targeted_uid and config.fault_kind == "duplicate" and ordinal == 1
                            else "late_quarantined"
                            if targeted_uid
                            and config.fault_kind in {"out_of_order", "lease_expiry", "late_result"}
                            and ordinal == 1
                            else "committed"
                        )
                        or result_payload.get("result_hash")
                        != state_hashes[cast(int, result_payload.get("injected_slot_index"))]
                        or result_payload.get("injected_slot_index") != lease_payload.get("injected_slot_index")
                        or result_payload.get("injected_slot_index")
                        != (
                            config.fault_index % _ROLLOUT_COUNT
                            if config.fault_index is not None
                            and config.fault_index // _ROLLOUT_COUNT == payload.get("uid_ordinal")
                            else 0
                        )
                    ):
                        raise StepReadyRecoveryError("attempt transition fields changed")
                    if (
                        lease_payload.get("run_id") != config.run_id
                        or lease_payload.get("uid") != payload.get("uid")
                        or result_payload.get("attempt_hash") != lease.content_hash
                        or result_payload.get("run_id") != config.run_id
                        or result_payload.get("uid") != payload.get("uid")
                    ):
                        raise StepReadyRecoveryError("attempt lease/result lineage changed")
                quarantine_values = payload.get("late_quarantine_hashes")
                if not isinstance(quarantine_values, list) or any(
                    type(value) is not str for value in quarantine_values
                ):
                    raise StepReadyRecoveryError("late-result quarantine lineage is incomplete")
                for quarantine_hash in cast(list[str], quarantine_values):
                    quarantine = store.read(quarantine_hash, expected_schema_name="LateResultQuarantine")
                    if (
                        set(quarantine.payload)
                        != {"global_step", "late_result_hash", "reason", "run_id", "status", "uid"}
                        or quarantine.payload.get("status") != "late_quarantined"
                        or quarantine.payload.get("run_id") != config.run_id
                        or quarantine.payload.get("global_step") != config.global_step
                        or quarantine.payload.get("uid") != payload.get("uid")
                        or quarantine.payload.get("late_result_hash") not in cast(list[str], state_hashes)
                    ):
                        raise StepReadyRecoveryError("late-result quarantine is invalid")
                typed_state_hashes = cast(list[str], state_hashes)
                if len(set(typed_state_hashes)) != _ROLLOUT_COUNT:
                    raise StepReadyRecoveryError("UID durable reward hashes are duplicated")
                state_rewards = rewards[state_reward_offset : state_reward_offset + _ROLLOUT_COUNT]
                if [item.content_hash for item in state_rewards] != typed_state_hashes or [
                    item.payload.get("rollout_index") for item in state_rewards
                ] != list(range(_ROLLOUT_COUNT)):
                    raise StepReadyRecoveryError("UID reward ordering changed")
                state_reward_offset += _ROLLOUT_COUNT
                state_reward_hashes.extend(typed_state_hashes)
                if len({cast(str, value) for value in typed_state_hashes}) != _ROLLOUT_COUNT:
                    raise StepReadyRecoveryError("UID durable reward hashes are duplicated")
            if state_reward_hashes != typed_reward_hashes:
                raise StepReadyRecoveryError("UID state/reward barrier lineage changed")
            if len(set(typed_reward_hashes)) != _SLOT_COUNT:
                raise StepReadyRecoveryError("step-ready reward slots are duplicated")
            uid_order = [cast(str, state.payload["uid"]) for state in states]
            if uid_order != sorted(uid_order):
                raise StepReadyRecoveryError("UID state order is not canonical")
            state_by_uid = {cast(str, state.payload["uid"]): state for state in states}
            if len(state_by_uid) != _UID_COUNT:
                raise StepReadyRecoveryError("step-ready UID state identity is duplicated")
            expected_uids = cls._uids(expected_snapshot.expected_set)
            if sorted(state_by_uid) != list(expected_uids):
                raise StepReadyRecoveryError("step-ready UID set changed")
            fault_values = report.payload.get("fault_event_hashes")
            if not isinstance(fault_values, list) or any(type(value) is not str for value in fault_values):
                raise StepReadyRecoveryError("fault event lineage is invalid")
            for fault_hash in cast(list[str], fault_values):
                try:
                    event = store.read(fault_hash, expected_schema_name="RecoveryFaultEvent")
                except ArtifactCorruption:
                    quarantine = store.read(fault_hash, expected_schema_name="LateResultQuarantine")
                    if (
                        set(quarantine.payload)
                        != {"global_step", "late_result_hash", "reason", "run_id", "status", "uid"}
                        or quarantine.payload.get("status") != "late_quarantined"
                        or quarantine.payload.get("run_id") != config.run_id
                        or quarantine.payload.get("global_step") != config.global_step
                    ):
                        raise StepReadyRecoveryError("late-result quarantine is invalid") from None
                    continue
                if (
                    event.schema_version != "1.0.0"
                    or set(event.payload)
                    != {
                        "arrival_order",
                        "deduplicated",
                        "fault_index",
                        "fault_kind",
                        "fencing_epoch",
                        "lease_expired",
                        "observed_slot_count",
                        "quarantine_hashes",
                        "resolver_epoch",
                        "restart_fencing_epoch",
                        "restart_kind",
                        "router_restart_count",
                        "run_id",
                        "global_step",
                        "status",
                        "target_uid_ordinal",
                        "uid_state_hashes",
                    }
                    or event.payload.get("fault_kind") != report.payload.get("fault_kind")
                    or event.payload.get("fault_index") != report.payload.get("fault_index")
                    or event.payload.get("status") != "recovered"
                    or event.payload.get("observed_slot_count") != _SLOT_COUNT
                    or event.payload.get("resolver_epoch") != effective_epoch
                    or event.payload.get("fencing_epoch") != effective_epoch
                    or event.payload.get("run_id") != config.run_id
                    or event.payload.get("global_step") != config.global_step
                    or event.payload.get("uid_state_hashes") != hashes
                    or event.payload.get("deduplicated") is not (config.fault_kind == "duplicate")
                    or event.payload.get("lease_expired") is not (config.fault_kind == "lease_expiry")
                    or event.payload.get("restart_kind")
                    != (config.fault_kind if config.fault_kind and config.fault_kind.endswith("restart") else None)
                    or event.payload.get("restart_fencing_epoch")
                    != (
                        config.resolver_epoch + 1
                        if config.fault_kind in {"resolver_restart", "router_restart", "controller_restart"}
                        else None
                    )
                    or event.payload.get("router_restart_count") != (1 if config.fault_kind == "router_restart" else 0)
                    or event.payload.get("target_uid_ordinal")
                    != (None if config.fault_index is None else config.fault_index // _ROLLOUT_COUNT)
                    or event.payload.get("quarantine_hashes")
                    != [
                        cast(str, value)
                        for state in states
                        for value in cast(list[str], state.payload.get("late_quarantine_hashes", []))
                    ]
                    or event.payload.get("arrival_order")
                    != (
                        list(reversed(range(_ROLLOUT_COUNT)))
                        if config.fault_kind == "out_of_order"
                        else list(range(_ROLLOUT_COUNT))
                    )
                ):
                    raise StepReadyRecoveryError("fault event lineage is invalid")
            stored_pack_map = report.payload.get("judge_pack_hashes")
            if not isinstance(stored_pack_map, dict) or set(stored_pack_map) != set(state_by_uid):
                raise StepReadyRecoveryError("step-ready UID set does not match frozen JudgePack mapping")
            state_hash_sets = {
                uid: set(cast(list[str], state.payload["reward_hashes"])) for uid, state in state_by_uid.items()
            }
            rollout_by_uid: dict[str, set[int]] = {uid: set() for uid in state_by_uid}
            for reward in rewards:
                if (
                    reward.payload.get("run_id") != config.run_id
                    or reward.payload.get("global_step") != config.global_step
                    or reward.payload.get("uid") not in uid_order
                    or type(reward.payload.get("rollout_index")) is not int
                    or not 0 <= cast(int, reward.payload["rollout_index"]) < _ROLLOUT_COUNT
                    or type(reward.payload.get("reward_micros")) is not int
                ):
                    raise StepReadyRecoveryError("reward lineage is incomplete")
                reward_uid = cast(str, reward.payload["uid"])
                if reward.content_hash not in state_hash_sets[reward_uid]:
                    raise StepReadyRecoveryError("reward is not owned by its UID state")
                expected_pack_hash = cast(str, stored_pack_map[reward_uid])
                if config.trainer_path == "classic":
                    if reward.payload.get("judge_pack_hash") != expected_pack_hash:
                        raise StepReadyRecoveryError("reward JudgePack lineage changed")
                else:
                    classic_reward = store.read(
                        cast(str, reward.payload.get("classic_reward_hash")),
                        expected_schema_name="ClassicGroupedReward",
                    )
                    if classic_reward.payload.get("judge_pack_hash") != expected_pack_hash:
                        raise StepReadyRecoveryError("v1 reward JudgePack lineage changed")
                rollout_index = cast(int, reward.payload["rollout_index"])
                if reward.payload.get("trajectory_manifest_hash") != expected_manifests.get(
                    (reward_uid, rollout_index)
                ):
                    raise StepReadyRecoveryError("reward trajectory lineage changed")
                rollout_by_uid[reward_uid].add(rollout_index)
            if any(indices != set(range(_ROLLOUT_COUNT)) for indices in rollout_by_uid.values()):
                raise StepReadyRecoveryError("step-ready per-UID rollout coverage is incomplete")
            if config.fault_kind in {"out_of_order", "lease_expiry", "late_result"}:
                quarantine_hashes = {
                    cast(str, value)
                    for state in states
                    for value in cast(list[str], state.payload["late_quarantine_hashes"])
                }
                if not quarantine_hashes.issubset(set(cast(list[str], fault_values))):
                    raise StepReadyRecoveryError("late-result quarantine is not linked from report")
            if packs is not None:
                for uid, pack in packs.items():
                    if pack.payload.get("uid") != uid:
                        raise StepReadyRecoveryError("JudgePack UID mapping changed")
            return StepReadyRecoverySnapshot(report, step_ready, states, rewards)
        except (
            ArtifactCorruption,
            ExpectedTrajectorySetError,
            UnknownSchemaMajor,
            OSError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise StepReadyRecoveryError("step-ready route cannot be recovered") from error

    @classmethod
    def _validate_uid_commitment(
        cls,
        store: ArtifactStore,
        state: Artifact,
        rewards: Sequence[Artifact],
        config: StepReadyRecoveryConfig,
        expected_set: Artifact,
        pack: Artifact,
        experiment_spec: Artifact,
    ) -> None:
        payload = state.payload
        reward_hashes = payload.get("reward_hashes")
        if (
            state.schema_version != "1.0.0"
            or set(payload) != _STATE_FIELDS
            or type(payload.get("uid")) is not str
            or payload.get("run_id") != config.run_id
            or payload.get("global_step") != config.global_step
            or payload.get("resolver_epoch") != cls._effective_epoch(config)
            or payload.get("status") != "resolved"
            or payload.get("expected_count") != _ROLLOUT_COUNT
            or not isinstance(reward_hashes, list)
            or len(reward_hashes) != _ROLLOUT_COUNT
            or len(rewards) != _ROLLOUT_COUNT
            or [item.content_hash for item in rewards] != cast(list[str], reward_hashes)
            or len(set(cast(list[str], reward_hashes))) != _ROLLOUT_COUNT
            or [item.payload.get("rollout_index") for item in rewards] != list(range(_ROLLOUT_COUNT))
        ):
            raise StepReadyRecoveryError("UID durable reward state is incomplete")
        grouped_schema = "ClassicGroupedRewardReport" if config.trainer_path == "classic" else "V1GroupedRewardReport"
        grouped_report = store.read(cast(str, payload["grouped_report_hash"]), expected_schema_name=grouped_schema)
        uid = cast(str, payload["uid"])
        uid_ordinal = payload.get("uid_ordinal")
        if type(uid_ordinal) is not int or not 0 <= uid_ordinal < _UID_COUNT:
            raise StepReadyRecoveryError("UID durable ordinal is invalid")
        if (
            grouped_report.payload.get("run_id") != config.run_id
            or grouped_report.payload.get("global_step") != config.global_step
            or grouped_report.payload.get("uid") != uid
            or grouped_report.payload.get("expected_set_hash") != expected_set.content_hash
            or grouped_report.payload.get("experiment_spec_hash") != experiment_spec.content_hash
            or grouped_report.payload.get("judge_pack_hash") != pack.content_hash
            or grouped_report.payload.get("status") != "resolved"
            or (
                config.trainer_path == "v1"
                and grouped_report.payload.get("resolver_epoch") != cls._effective_epoch(config)
            )
        ):
            raise StepReadyRecoveryError("grouped report lineage changed")
        expected_snapshot = ExpectedTrajectorySetWorkflow.resume(store.root, config.run_id, config.global_step)
        manifests: dict[tuple[str, int], str] = {}
        for manifest in expected_snapshot.manifests:
            slot = manifest.payload.get("slot_key")
            if isinstance(slot, dict) and type(slot.get("uid")) is str and type(slot.get("rollout_index")) is int:
                manifests[(cast(str, slot["uid"]), cast(int, slot["rollout_index"]))] = manifest.content_hash
        indices: set[int] = set()
        for reward in rewards:
            index = reward.payload.get("rollout_index")
            if (
                reward.payload.get("uid") != uid
                or type(index) is not int
                or not 0 <= cast(int, index) < _ROLLOUT_COUNT
                or reward.payload.get("run_id") != config.run_id
                or reward.payload.get("global_step") != config.global_step
                or (config.trainer_path == "classic" and reward.payload.get("judge_pack_hash") != pack.content_hash)
                or reward.payload.get("trajectory_manifest_hash") != manifests.get((uid, cast(int, index)))
            ):
                raise StepReadyRecoveryError("UID reward trajectory lineage changed")
            if config.trainer_path == "v1":
                classic_reward = store.read(
                    cast(str, reward.payload.get("classic_reward_hash")),
                    expected_schema_name="ClassicGroupedReward",
                )
                if classic_reward.payload.get("judge_pack_hash") != pack.content_hash:
                    raise StepReadyRecoveryError("v1 reward JudgePack lineage changed")
            indices.add(cast(int, index))
        if indices != set(range(_ROLLOUT_COUNT)):
            raise StepReadyRecoveryError("UID reward rollout coverage is incomplete")

    @staticmethod
    def _pack_map(value: Mapping[str, Artifact] | Sequence[Artifact]) -> dict[str, Artifact]:
        if isinstance(value, Mapping):
            return dict(value)
        try:
            result = {cast(str, pack.payload["uid"]): pack for pack in value}
        except (KeyError, TypeError):
            raise StepReadyRecoveryError("judge_packs mapping is invalid") from None
        if len(result) != len(value):
            raise StepReadyRecoveryError("JudgePack UIDs are duplicated")
        return result

    @staticmethod
    def _uids(expected_set: Artifact) -> tuple[str, ...]:
        slots = expected_set.payload.get("slots")
        if not isinstance(slots, list):
            raise StepReadyRecoveryError("ExpectedTrajectorySet slots are invalid")
        uids = sorted(
            {cast(str, item["uid"]) for item in slots if isinstance(item, dict) and type(item.get("uid")) is str}
        )
        return tuple(uids)

    @classmethod
    def _validate_inputs(
        cls,
        root: str | Path,
        config: StepReadyRecoveryConfig,
        expected_set: Artifact,
        packs: Mapping[str, Artifact],
        experiment_spec: Artifact,
    ) -> None:
        if (
            experiment_spec.schema_name != "ExperimentSpec"
            or experiment_spec.schema_version != "1.0.0"
            or experiment_spec.content_hash != config.experiment_spec_hash
            or set(experiment_spec.payload) != _EXPERIMENT_FIELDS
            or experiment_spec.payload.get("aggregation") != "calibrated_scalar"
        ):
            raise StepReadyRecoveryError("ExperimentSpec identity is invalid")
        if (
            expected_set.schema_name != "ExpectedTrajectorySet"
            or expected_set.payload.get("run_id") != config.run_id
            or expected_set.payload.get("global_step") != config.global_step
        ):
            raise StepReadyRecoveryError("ExpectedTrajectorySet identity is invalid")
        try:
            snapshot = ExpectedTrajectorySetWorkflow.resume(root, config.run_id, config.global_step)
        except ExpectedTrajectorySetError as error:
            raise StepReadyRecoveryError("ExpectedTrajectorySet is not durable") from error
        if snapshot.expected_set.content_hash != expected_set.content_hash or len(snapshot.manifests) != _SLOT_COUNT:
            raise StepReadyRecoveryError("step-ready expected set must contain exactly 2048 slots")
        slots = expected_set.payload.get("slots")
        if not isinstance(slots, list) or len(slots) != _SLOT_COUNT:
            raise StepReadyRecoveryError("ExpectedTrajectorySet must contain exactly 2048 slots")
        by_uid: dict[str, list[int]] = {}
        stable_keys: set[str] = set()
        for slot in slots:
            if not isinstance(slot, dict):
                raise StepReadyRecoveryError("ExpectedTrajectorySet slot is invalid")
            uid_value = slot.get("uid")
            index_value = slot.get("rollout_index")
            key_value = slot.get("slot_key_hash")
            if (
                type(uid_value) is not str
                or type(index_value) is not int
                or not 0 <= cast(int, index_value) < _ROLLOUT_COUNT
                or type(key_value) is not str
                or key_value in stable_keys
            ):
                raise StepReadyRecoveryError("ExpectedTrajectorySet slot identity is invalid")
            stable_keys.add(cast(str, key_value))
            by_uid.setdefault(cast(str, uid_value), []).append(cast(int, index_value))
        if len(by_uid) != _UID_COUNT or any(
            sorted(indexes) != list(range(_ROLLOUT_COUNT)) for indexes in by_uid.values()
        ):
            raise StepReadyRecoveryError("ExpectedTrajectorySet UID Cartesian product is incomplete")
        uids = cls._uids(expected_set)
        if len(uids) != _UID_COUNT or set(packs) != set(uids):
            raise StepReadyRecoveryError("step-ready requires 16 JudgePack UID mappings")
        for uid, pack in packs.items():
            try:
                FixtureTraceRouter._pack(pack, uid, expected_set)
            except TraceRouterError as error:
                raise StepReadyRecoveryError("JudgePack routing mapping is invalid") from error
            if pack.schema_version != "1.0.0" or pack.payload.get("uid") != uid:
                raise StepReadyRecoveryError("JudgePack UID mapping is invalid")

    @classmethod
    def _group_config(
        cls, config: StepReadyRecoveryConfig, uid: str
    ) -> ClassicGroupedRewardConfig | V1GroupedRewardConfig:
        epoch = cls._effective_epoch(config)
        if config.trainer_path == "classic":
            return ClassicGroupedRewardConfig(
                config.run_id,
                config.global_step,
                uid,
                config.experiment_spec_hash,
                resolver_epoch=epoch,
                chunk_size=config.chunk_size,
                capacity=config.capacity,
            )
        return V1GroupedRewardConfig(
            config.run_id,
            config.global_step,
            uid,
            config.experiment_spec_hash,
            resolver_epoch=epoch,
            chunk_size=config.chunk_size,
            capacity=config.capacity,
        )

    @staticmethod
    def _publish_ref(ref: Path, artifact: Artifact) -> None:
        ArtifactStore.durable_mkdir(ref.parent)
        ArtifactStore._publish(ref, f"{artifact.content_hash}\n".encode("ascii"))

    @staticmethod
    def _attempt_artifacts(
        store: ArtifactStore,
        config: StepReadyRecoveryConfig,
        uid: str,
        uid_ordinal: int,
        rewards: Sequence[Artifact],
    ) -> list[str]:
        """Persist the lease/result transitions used by the deterministic fixture."""

        targeted = config.fault_index is not None and config.fault_index // _ROLLOUT_COUNT == uid_ordinal
        recoverable = {"duplicate", "out_of_order", "lease_expiry", "late_result"}
        count = 2 if targeted and config.fault_kind in recoverable else 1
        effective_epoch = StepReadyRecoveryWorkflow._effective_epoch(config)
        local_index = 0 if config.fault_index is None else config.fault_index % _ROLLOUT_COUNT if targeted else 0
        result_hash = rewards[local_index].content_hash
        hashes: list[str] = []
        for ordinal in range(1, count + 1):
            lease = store.put(
                "RecoveryAttemptLease",
                "1.0.0",
                {
                    "attempt_ordinal": ordinal,
                    "fault_index": config.fault_index,
                    "fault_kind": config.fault_kind,
                    "lease_expires_tick": config.lease_ticks * ordinal,
                    "lease_ticks": config.lease_ticks,
                    "resolver_epoch": effective_epoch,
                    "run_id": config.run_id,
                    "status": "expired" if config.fault_kind == "lease_expiry" and ordinal < count else "committed",
                    "uid": uid,
                    "injected_slot_index": local_index,
                },
            )
            result = store.put(
                "RecoveryAttemptResult",
                "1.0.0",
                {
                    "attempt_hash": lease.content_hash,
                    "attempt_ordinal": ordinal,
                    "fault_index": config.fault_index,
                    "fault_kind": config.fault_kind,
                    "result_hash": result_hash,
                    "run_id": config.run_id,
                    "status": (
                        "duplicate_deduplicated"
                        if config.fault_kind == "duplicate" and ordinal < count
                        else "late_quarantined"
                        if config.fault_kind in {"out_of_order", "lease_expiry", "late_result"} and ordinal < count
                        else "committed"
                    ),
                    "uid": uid,
                    "injected_slot_index": local_index,
                },
            )
            hashes.extend((lease.content_hash, result.content_hash))
        return hashes

    @classmethod
    def _fault_events(
        cls,
        store: ArtifactStore,
        namespace: Path,
        config: StepReadyRecoveryConfig,
        states: Sequence[Artifact],
        rewards: Sequence[Artifact],
    ) -> list[str]:
        fault = config.fault_kind
        if fault is None:
            return []
        events: list[str] = []
        quarantine_hashes = [
            cast(str, value)
            for state in states
            for value in cast(list[str], state.payload.get("late_quarantine_hashes", []))
        ]
        target_uid_ordinal = None if config.fault_index is None else config.fault_index // _ROLLOUT_COUNT
        effective_epoch = cls._effective_epoch(config)
        restart_epoch = effective_epoch if fault.endswith("restart") else None
        event = store.put(
            "RecoveryFaultEvent",
            "1.0.0",
            {
                "arrival_order": (
                    list(reversed(range(_ROLLOUT_COUNT))) if fault == "out_of_order" else list(range(_ROLLOUT_COUNT))
                ),
                "deduplicated": fault == "duplicate",
                "fault_kind": fault,
                "fault_index": config.fault_index,
                "fencing_epoch": effective_epoch,
                "lease_expired": fault == "lease_expiry",
                "observed_slot_count": len(rewards),
                "quarantine_hashes": quarantine_hashes,
                "resolver_epoch": effective_epoch,
                "restart_fencing_epoch": restart_epoch,
                "restart_kind": fault if fault.endswith("restart") else None,
                "router_restart_count": 1 if fault == "router_restart" else 0,
                "run_id": config.run_id,
                "global_step": config.global_step,
                "status": "recovered",
                "target_uid_ordinal": target_uid_ordinal,
                "uid_state_hashes": [item.content_hash for item in states],
            },
        )
        events.append(event.content_hash)
        if fault in {"out_of_order", "lease_expiry", "late_result"}:
            events.extend(quarantine_hashes)
        cls._publish_ref(namespace / "fault-events.ref", event)
        return events

    @classmethod
    def _terminal_stop(
        cls,
        root: str | Path,
        config: StepReadyRecoveryConfig,
        expected_set: Artifact,
        experiment_spec: Artifact,
        reason: str,
    ) -> StepReadyTerminalStop:
        store = ArtifactStore(root)
        evidence = store.put(
            "RecoveryTerminalEvidence",
            "1.0.0",
            {
                "expected_set_hash": expected_set.content_hash,
                "experiment_spec_hash": experiment_spec.content_hash,
                "reason_code": reason,
                "run_id": config.run_id,
                "status": "terminal",
                "trainer_path": config.trainer_path,
            },
        )
        request = store.put(
            "RunStopRequest",
            "1.0.0",
            {
                "expected_set_hash": expected_set.content_hash,
                "experiment_spec_hash": experiment_spec.content_hash,
                "evidence_hash": evidence.content_hash,
                "global_step": config.global_step,
                "reason_code": reason,
                "run_id": config.run_id,
                "status": "terminal",
                "trainer_path": config.trainer_path,
                "optimizer_update": False,
            },
        )
        namespace = cls._namespace(root, config)
        ArtifactStore.durable_mkdir(namespace)
        cls._publish_ref(namespace / "stop-request.ref", request)
        # Keep the hash in the exception so an operator can locate the typed
        # stop without consulting process-local state.
        return StepReadyTerminalStop(f"{reason}:RUN_STOP_REQUEST:{request.content_hash}")


# Alias used by callers that call the component a reward-router controller.
RewardRouterRecoveryWorkflow = StepReadyRecoveryWorkflow
RewardRouterRecoveryConfig = StepReadyRecoveryConfig
