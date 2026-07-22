# ruff: noqa: E501

"""Deterministic, fail-closed handling of hard training failures (Ticket 22).

This module deliberately keeps the monitor/controller boundary small.  A monitor
only writes immutable observations; the fenced controller is the sole writer of
the terminal manifest and asks the fixture ClusterAdapter to perform one
idempotent ``force_cancel``.  No reward tensor or optimizer update is produced
on this path.
"""

from __future__ import annotations

import fcntl
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.harness.journal import HarnessJournal, HarnessJournalError
from clawrl.training.cluster_lifecycle import (
    ClusterLifecycleError,
    ClusterLifecycleSnapshot,
    ClusterLifecycleWorkflow,
    FixtureClusterLifecycleConfig,
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


class HardFailureStopError(RuntimeError):
    """The hard-failure controller failed closed."""


class StaleHardFailureController(HardFailureStopError):
    """A controller epoch no longer owns the run."""


class InjectedHardFailureControllerCrash(HardFailureStopError):
    """Fixture crash after the provider cancel has committed."""


HardFailureTerminalStop = HardFailureStopError


class HardFailureKind(StrEnum):
    NAN_INF = "NAN_INF"
    OOM = "OOM"
    REPEATED_CRASH = "REPEATED_CRASH"
    HEARTBEAT_STALLED = "HEARTBEAT_STALLED"
    SCHEDULER_STALLED = "SCHEDULER_STALLED"
    CFS_CORRUPTION = "CFS_CORRUPTION"
    PERMANENT_REWARD_FAILURE = "PERMANENT_REWARD_FAILURE"


# Names used by integrations which predate the enum.
HardFailureClass = HardFailureKind
FailureKind = HardFailureKind
HardFailureCrash = InjectedHardFailureControllerCrash


_ALIASES: dict[str, HardFailureKind] = {
    "NAN": HardFailureKind.NAN_INF,
    "INF": HardFailureKind.NAN_INF,
    "NAN_INF": HardFailureKind.NAN_INF,
    "NON_FINITE": HardFailureKind.NAN_INF,
    "OUT_OF_MEMORY": HardFailureKind.OOM,
    "OOM": HardFailureKind.OOM,
    "MEMORY": HardFailureKind.OOM,
    "REPEATED_CRASH": HardFailureKind.REPEATED_CRASH,
    "DUPLICATE_CRASH": HardFailureKind.REPEATED_CRASH,
    "HEARTBEAT_STALLED": HardFailureKind.HEARTBEAT_STALLED,
    "HEARTBEAT_TIMEOUT": HardFailureKind.HEARTBEAT_STALLED,
    "SCHEDULER_STALLED": HardFailureKind.SCHEDULER_STALLED,
    "SCHEDULER_TIMEOUT": HardFailureKind.SCHEDULER_STALLED,
    "CFS_CORRUPTION": HardFailureKind.CFS_CORRUPTION,
    "CFS_CORRUPT": HardFailureKind.CFS_CORRUPTION,
    "PERMANENT_REWARD_FAILURE": HardFailureKind.PERMANENT_REWARD_FAILURE,
    "REWARD_ATTEMPTS_EXHAUSTED": HardFailureKind.PERMANENT_REWARD_FAILURE,
    "REWARD_FAILURE_PERMANENT": HardFailureKind.PERMANENT_REWARD_FAILURE,
    "CRASH": HardFailureKind.REPEATED_CRASH,
}


def _safe_id(value: object, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(cast(str, value)) is None:
        raise HardFailureStopError(f"{field} is invalid")
    return cast(str, value)


def _hash(value: object, field: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise HardFailureStopError(f"{field} is invalid")
    return cast(str, value)


def _json_safe(value: object) -> JsonValue:
    """Convert monitor values to the artifact domain without hiding non-finite data."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "-Inf" if value < 0 else "Inf"
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return cast(JsonValue, value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(v) for v in value]
    return str(value)


def classify_hard_failure(
    observation: Artifact | Mapping[str, object] | str | HardFailureObservation,
) -> HardFailureKind:
    """Classify a monitor observation using a deterministic closed-world table."""
    payload: Mapping[str, object]
    if isinstance(observation, HardFailureObservation):
        payload = {"failure_kind": str(observation.failure_kind), **(observation.details or {})}
    elif isinstance(observation, Artifact):
        payload = observation.payload
        details = payload.get("details")
        if isinstance(details, dict):
            payload = {**payload, **details}
    elif isinstance(observation, Mapping):
        payload = observation
    else:
        payload = {"failure_code": observation}
    explicit = payload.get("failure_kind") or payload.get("failure_class") or payload.get("reason_code")
    if isinstance(explicit, str):
        normalized = explicit.strip().upper().replace("-", "_").replace(" ", "_")
        if normalized in _ALIASES:
            return _ALIASES[normalized]

    def has_nonfinite(value: object) -> bool:
        if isinstance(value, float):
            return not math.isfinite(value)
        if isinstance(value, Mapping):
            return any(has_nonfinite(item) for item in value.values())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return any(has_nonfinite(item) for item in value)
        return False

    if has_nonfinite(payload):
        return HardFailureKind.NAN_INF
    values = " ".join(str(payload.get(k, "")) for k in ("failure_code", "code", "error", "message", "status"))
    upper = values.upper().replace("-", "_")
    if any(x in upper for x in ("NAN", "INF", "NONFINITE", "NON_FINITE")):
        return HardFailureKind.NAN_INF
    if any(x in upper for x in ("OUT_OF_MEMORY", "OOM", "CUDA_ERROR_OUT_OF_MEMORY")):
        return HardFailureKind.OOM
    crash_count = payload.get("crash_count", 0)
    if (
        bool(payload.get("repeated_crash"))
        or (type(crash_count) is int and crash_count >= 2)
        or "REPEATED_CRASH" in upper
        or "CRASH_LOOP" in upper
    ):
        return HardFailureKind.REPEATED_CRASH
    if bool(payload.get("heartbeat_stalled")) or (
        "HEARTBEAT" in upper and any(x in upper for x in ("STALL", "STALE", "TIMEOUT", "MISSING", "DEAD"))
    ):
        return HardFailureKind.HEARTBEAT_STALLED
    if bool(payload.get("scheduler_stalled")) or (
        "SCHEDULER" in upper and any(x in upper for x in ("STALL", "STALE", "TIMEOUT", "MISSING", "DEAD"))
    ):
        return HardFailureKind.SCHEDULER_STALLED
    if "CFS" in upper and any(x in upper for x in ("CORRUPT", "CHECKSUM", "MISSING", "INVALID")):
        return HardFailureKind.CFS_CORRUPTION
    if (
        "REWARD" in upper and any(x in upper for x in ("PERMANENT", "EXHAUST", "INVALID", "FAIL"))
    ) or "REWARD_ATTEMPTS_EXHAUSTED" in upper:
        return HardFailureKind.PERMANENT_REWARD_FAILURE
    raise HardFailureStopError("observation is not a recognized hard failure")


@dataclass(frozen=True, slots=True)
class HardFailureStopConfig:
    run_id: str
    # Kept as the second positional field to match the other training
    # controller configs (run_id, experiment_spec_hash).
    experiment_spec_hash: str = ""
    phase: str = "TRAIN_35B"
    controller_epoch: int = 1
    close_mode: str = "force_cancel"
    failure_kind: str | None = None
    fault_kind: str | None = None
    # Keyword spelling accepted by early fixture callers.
    spec_hash: str = ""

    def __post_init__(self) -> None:
        _safe_id(self.run_id, "run_id")
        if self.phase not in {"TRAIN_35B", "TRAIN_122B"}:
            raise HardFailureStopError("phase is invalid")
        if self.experiment_spec_hash and self.spec_hash and self.experiment_spec_hash != self.spec_hash:
            raise HardFailureStopError("experiment_spec_hash and spec_hash conflict")
        resolved = self.spec_hash or self.experiment_spec_hash
        if resolved:
            _hash(resolved, "spec_hash")
            object.__setattr__(self, "spec_hash", resolved)
        if type(self.controller_epoch) is not int or self.controller_epoch <= 0:
            raise HardFailureStopError("controller_epoch is invalid")
        if self.close_mode != "force_cancel":
            raise HardFailureStopError("hard failure requires force_cancel")


# Friendly aliases used by fixture callers.
HardFailureConfig = HardFailureStopConfig
FixtureHardFailureConfig = HardFailureStopConfig


@dataclass(frozen=True, slots=True)
class HardFailureObservation:
    """Typed monitor value before it is persisted as an ``Observation``."""

    run_id: str
    failure_kind: HardFailureKind | str
    sequence: int = 1
    controller_epoch: int = 1
    details: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _safe_id(self.run_id, "run_id")
        if type(self.sequence) is not int or self.sequence < 1:
            raise HardFailureStopError("observation sequence is invalid")
        if type(self.controller_epoch) is not int or self.controller_epoch < 1:
            raise HardFailureStopError("observation controller epoch is invalid")
        classify_hard_failure({"failure_kind": str(self.failure_kind)})

    def payload(self) -> dict[str, JsonValue]:
        details = _json_safe(self.details or {})
        return {
            "controller_epoch": self.controller_epoch,
            "details": cast(dict[str, JsonValue], details),
            "failure_kind": str(self.failure_kind),
            "observation_sequence": self.sequence,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class HardFailureStopSnapshot:
    terminal: Artifact
    observation: Artifact
    stop_request: Artifact
    harness_decision: Artifact
    action_plan: Artifact
    action_observation: Artifact
    reward_terminal_stop: Artifact
    cluster: ClusterLifecycleSnapshot

    @property
    def run_closed(self) -> Artifact:
        return self.terminal

    @property
    def cancel(self) -> Artifact:
        return self.action_observation

    @property
    def trainer_reward(self) -> None:
        return None

    @property
    def optimizer_update(self) -> None:
        return None


class HardFailureStopWorkflow:
    """Durable fixture monitor/controller with process-safe exactly-once closure."""

    @staticmethod
    def classify(observation: Artifact | Mapping[str, object] | str) -> HardFailureKind:
        return classify_hard_failure(observation)

    @staticmethod
    def production_readiness(root: str | Path, *, phase: str = "TRAIN_35B") -> Artifact:
        if phase not in {"TRAIN_35B", "TRAIN_122B"}:
            raise HardFailureStopError("phase is invalid")
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": "PRODUCTION_MONITOR_FENCE_UNVERIFIED", "status": "blocked"},
                    {"code": "PRODUCTION_CLUSTER_CANCEL_UNVERIFIED", "status": "blocked"},
                    {"code": "PRODUCTION_HARNESS_STOP_UNVERIFIED", "status": "blocked"},
                ],
                "execution_profile": "production",
                "phase": phase,
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )

    @staticmethod
    def emergency_action_allowed(action: str) -> bool:
        return action == "force_cancel"

    @staticmethod
    def _harness_proposal_id(run_id: str) -> str:
        # HarnessJournal's filesystem identifier is narrower than the training
        # run identifier (which also permits ':'); hash only the incompatible
        # form so ordinary fixture paths remain inspectable.
        return (
            f"hard-failure-{run_id}"
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id)
            else f"hard-failure-{sha256_hex(run_id.encode('utf-8', 'surrogatepass'))}"
        )

    @classmethod
    def authorize_emergency(cls, action: str) -> None:
        """Validate the zero-cost emergency exemption's narrow capability."""
        if not cls.emergency_action_allowed(action):
            raise HardFailureStopError("emergency exemption cannot authorize training/query/scorer work")

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: HardFailureStopConfig,
        observation: Artifact | Mapping[str, object] | str | HardFailureObservation | None = None,
        controller_epoch: int | None = None,
        observations: Sequence[Artifact | Mapping[str, object] | str | HardFailureObservation] | None = None,
        experiment_spec: Artifact | None = None,
        crash_after: str | None = None,
        **_ignored: object,
    ) -> HardFailureStopSnapshot:
        profile = _ignored.get("execution_profile")
        if profile is not None and profile not in {"fixture", "production"}:
            raise HardFailureStopError("execution_profile must be fixture or production")
        if profile == "production":
            raise HardFailureStopError(
                "production hard-failure stop is blocked until monitor, Harness, and Cluster adapters are verified"
            )
        store = ArtifactStore(root)
        if observation is None and observations:
            observation = observations[0]
        if experiment_spec is not None:
            if experiment_spec.schema_name != "ExperimentSpec":
                raise HardFailureStopError("experiment_spec must be ExperimentSpec")
            if config.spec_hash and config.spec_hash != experiment_spec.content_hash:
                raise HardFailureStopError("ExperimentSpec identity conflict")
            if not config.spec_hash:
                config = replace(config, experiment_spec_hash=experiment_spec.content_hash)
        epoch = config.controller_epoch if controller_epoch is None else controller_epoch
        if type(epoch) is not int or epoch <= 0:
            raise StaleHardFailureController("controller epoch is invalid")
        if observation is None:
            requested = config.failure_kind or config.fault_kind
            if requested is None:
                raise HardFailureStopError("hard-failure observation is required")
            observation = {"failure_kind": requested}
        kind = classify_hard_failure(observation)
        if isinstance(observation, HardFailureObservation):
            if observation.run_id != config.run_id:
                raise HardFailureStopError("observation run identity changed")
            obs = store.put("Observation", "1.0.0", observation.payload())
        elif isinstance(observation, Artifact):
            if observation.schema_name != "Observation":
                raise HardFailureStopError("hard-failure monitor input must be Observation")
            if observation.payload.get("run_id") not in {None, config.run_id}:
                raise HardFailureStopError("observation run identity changed")
            obs = observation
        else:
            safe = _json_safe(observation if isinstance(observation, Mapping) else {"failure_code": observation})
            details = cast(dict[str, JsonValue], safe)
            obs = store.put(
                "Observation", "1.0.0", {"details": details, "observation_type": kind.value, "run_id": config.run_id}
            )
        root_path = Path(root) / "hard-failure-stops" / config.run_id
        ArtifactStore.durable_mkdir(root_path)
        ArtifactStore.durable_touch(root_path / "controller.lock")
        with (root_path / "controller.lock").open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                return cls._run_locked(root, root_path, config, epoch, kind, obs, crash_after=crash_after)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @classmethod
    def resume(cls, root: str | Path, run_id: str, *, controller_epoch: int | None = None) -> HardFailureStopSnapshot:
        _safe_id(run_id, "run_id")
        path = Path(root) / "hard-failure-stops" / run_id
        try:
            config_artifact = ArtifactStore(root).read(
                (path / "input.ref").read_text().strip(), expected_schema_name="HardFailureStopInput"
            )
            payload = config_artifact.payload
            config = HardFailureStopConfig(
                run_id,
                experiment_spec_hash=cast(str, payload["spec_hash"]),
                phase=cast(str, payload["phase"]),
                controller_epoch=cast(int, payload["controller_epoch"]),
            )
            obs = ArtifactStore(root).read(
                (path / "observation.ref").read_text().strip(), expected_schema_name="Observation"
            )
        except (OSError, ArtifactCorruption, KeyError, TypeError) as error:
            raise HardFailureStopError("hard-failure stop cannot be recovered") from error
        return cls.run(
            root,
            config=config,
            observation=obs,
            controller_epoch=config.controller_epoch if controller_epoch is None else controller_epoch,
        )

    @classmethod
    def _run_locked(
        cls,
        root: str | Path,
        path: Path,
        config: HardFailureStopConfig,
        epoch: int,
        kind: HardFailureKind,
        obs: Artifact,
        *,
        crash_after: str | None = None,
    ) -> HardFailureStopSnapshot:
        store = ArtifactStore(root)
        state_ref = path / "state.ref"
        if state_ref.exists():
            if not (path / "input.ref").exists():
                raise HardFailureStopError("hard-failure input ref is missing")
            committed_input = store.read(
                (path / "input.ref").read_text().strip(), expected_schema_name="HardFailureStopInput"
            )
            if (
                committed_input.payload.get("run_id") != config.run_id
                or committed_input.payload.get("phase") != config.phase
            ):
                raise HardFailureStopError("hard-failure run identity changed")
            committed_spec = committed_input.payload.get("spec_hash")
            if config.spec_hash and committed_spec and config.spec_hash != committed_spec:
                raise HardFailureStopError("ExperimentSpec identity conflict")
            state = store.read(state_ref.read_text().strip(), expected_schema_name="HardFailureStopState")
            current_epoch = state.payload.get("controller_epoch")
            if type(current_epoch) is not int or epoch < current_epoch:
                raise StaleHardFailureController("controller fencing epoch is stale")
            terminal_hash = state.payload.get("terminal_hash")
            if isinstance(terminal_hash, str):
                return cls._snapshot_from_hashes(store, terminal_hash, path)
        elif epoch != config.controller_epoch:
            if not (path / "input.ref").exists():
                raise StaleHardFailureController("controller fencing epoch is stale")
            try:
                persisted = store.read(
                    (path / "input.ref").read_text().strip(), expected_schema_name="HardFailureStopInput"
                )
                persisted_epoch = persisted.payload.get("controller_epoch")
            except (OSError, ArtifactCorruption) as error:
                raise HardFailureStopError("hard-failure input cannot be recovered") from error
            if type(persisted_epoch) is not int or epoch < persisted_epoch:
                raise StaleHardFailureController("controller fencing epoch is stale")
        if not config.spec_hash:
            spec = store.put("ExperimentSpec", "1.0.0", {"experiment_id": config.run_id, "status": "frozen"})
            spec_hash = spec.content_hash
            config = replace(config, spec_hash=spec_hash)
        else:
            spec_hash = config.spec_hash
            store.read(spec_hash, expected_schema_name="ExperimentSpec")
        input_artifact = store.put(
            "HardFailureStopInput",
            "1.0.0",
            {
                "controller_epoch": config.controller_epoch,
                "phase": config.phase,
                "run_id": config.run_id,
                "spec_hash": config.spec_hash,
            },
        )
        cls._publish_once(path / "input.ref", input_artifact.content_hash)
        cls._publish_once(path / "observation.ref", obs.content_hash)
        stop_request = store.put(
            "HardFailureStopRequest",
            "1.0.0",
            {
                "failure_kind": kind.value,
                "observation_hash": obs.content_hash,
                "run_id": config.run_id,
                "spec_hash": spec_hash,
                "status": "terminal",
            },
        )
        proposal_id = cls._harness_proposal_id(config.run_id)
        journal = HarnessJournal(root, store, proposal_id)
        harness_config = store.put(
            "FixtureHarnessConfig",
            "1.0.0",
            {
                "execution_profile": "fixture",
                "policy_id": "hard-failure-emergency",
                "allowed_action_types": ["cluster.force_cancel"],
            },
        )
        harness_policy = store.put(
            "HarnessPolicy",
            "1.0.0",
            {"policy_id": "hard-failure-emergency", "emergency_only": True, "spec_hash": spec_hash},
        )
        policy_capability = store.put(
            "HarnessPolicyCapability",
            "1.0.0",
            {"action_type": "cluster.force_cancel", "policy_hash": harness_policy.content_hash, "spec_hash": spec_hash},
        )
        workflow_input = store.put(
            "HarnessWorkflowInput",
            "1.0.0",
            {
                "action": {"action_type": "cluster.force_cancel", "input_hash": stop_request.content_hash},
                "caller": "hard-failure-controller",
                "phase": config.phase,
                "spec_hash": spec_hash,
            },
        )
        try:
            journal.reserve_identity(
                workflow_input.content_hash,
                harness_config.content_hash,
                harness_policy.content_hash,
                policy_capability.content_hash,
            )
            journal.claim_epoch(epoch)
            transitions = journal.strict_transition_snapshot()
            if not transitions:
                decision_proposal = journal.append(
                    epoch,
                    "DecisionProposal",
                    {
                        "action_type": "cluster.force_cancel",
                        "caller": "hard-failure-controller",
                        "input_hash": stop_request.content_hash,
                        "phase": config.phase,
                        "policy_capability_hash": policy_capability.content_hash,
                        "policy_hash": harness_policy.content_hash,
                        "spec_hash": spec_hash,
                    },
                    expected_sequence=1,
                    expected_previous_hash=None,
                )
                proposal = journal.append(
                    epoch,
                    "ActionPlan",
                    {
                        "action_type": "cluster.force_cancel",
                        "budget_cost": 0,
                        "emergency": True,
                        "input_hash": stop_request.content_hash,
                        "run_id": config.run_id,
                        "spec_hash": spec_hash,
                    },
                    expected_sequence=2,
                    expected_previous_hash=decision_proposal.content_hash,
                )
                decision = journal.append(
                    epoch,
                    "HarnessDecision",
                    {
                        "action_type": "cluster.force_cancel",
                        "decision": "allow",
                        "emergency": True,
                        "reason_code": "HARD_FAILURE_SAFETY_STOP",
                        "input_hash": stop_request.content_hash,
                        "plan_hash": proposal.content_hash,
                        "run_id": config.run_id,
                        "spec_hash": spec_hash,
                    },
                    expected_sequence=3,
                    expected_previous_hash=proposal.content_hash,
                )
            else:
                if len(transitions) < 3 or transitions[2].schema_name != "HarnessDecision":
                    raise HardFailureStopError("hard-failure Harness journal is incomplete")
                proposal = transitions[1]
                decision = transitions[2]
                journal.verify()
        except HarnessJournalError as error:
            raise HardFailureStopError("hard-failure Harness authorization failed closed") from error
        cluster_config = FixtureClusterLifecycleConfig(
            config.run_id,
            cast(Literal["TRAIN_35B", "TRAIN_122B"], config.phase),
            close_mode="force_cancel",
            fault_operation="force_cancel",
            fault_directive="emergency_stop",
        )
        try:
            cluster = ClusterLifecycleWorkflow.run(root, config=cluster_config, spec_hash=spec_hash)
        except ClusterLifecycleError as error:
            raise HardFailureStopError("cluster cancel did not produce durable outcome") from error
        if crash_after in {"cancel", "cluster_cancel", "provider_cancel"}:
            raise InjectedHardFailureControllerCrash("injected crash after cluster cancel")
        record = cluster.run_record
        transitions = journal.strict_transition_snapshot()
        if len(transitions) >= 5 and transitions[3].schema_name == "ActionObservation":
            action_observation = transitions[3]
            harness_outcome = transitions[4] if len(transitions) >= 5 else None
        else:
            action_observation = journal.append(
                epoch,
                "ActionObservation",
                {
                    "action_type": "cluster.force_cancel",
                    "cluster_run_record_hash": record.content_hash,
                    "idempotency_key": sha256_hex(
                        canonical_json_bytes({"run_id": config.run_id, "action": "force_cancel", "epoch": epoch})
                    ),
                    "plan_hash": proposal.content_hash,
                    "run_id": config.run_id,
                    "spec_hash": spec_hash,
                    "status": "succeeded" if record.payload.get("closure_operation") == "force_cancel" else "failed",
                },
                expected_sequence=4,
                expected_previous_hash=decision.content_hash,
            )
            harness_outcome = journal.append(
                epoch,
                "DecisionOutcome",
                {
                    "action_observation_hash": action_observation.content_hash,
                    "decision_hash": decision.content_hash,
                    "plan_hash": proposal.content_hash,
                    "run_id": config.run_id,
                    "spec_hash": spec_hash,
                    "status": "failed",
                    "terminal": True,
                },
                expected_sequence=5,
                expected_previous_hash=action_observation.content_hash,
            )
        if harness_outcome is None or harness_outcome.schema_name != "DecisionOutcome":
            raise HardFailureStopError("hard-failure Harness outcome is missing")
        audit_event = store.put(
            "AuditEvent",
            "1.0.0",
            {
                "action_plan_hash": proposal.content_hash,
                "decision_hash": decision.content_hash,
                "outcome_hash": harness_outcome.content_hash,
                "run_id": config.run_id,
                "spec_hash": spec_hash,
                "terminal": True,
            },
        )
        reward_stop = store.put(
            "RewardTerminalStop",
            "1.0.0",
            {
                "failure_kind": kind.value,
                "observation_hash": obs.content_hash,
                "optimizer_update_hash": None,
                "optimizer_update_permitted": False,
                "reward_publish_permitted": False,
                "run_id": config.run_id,
                "spec_hash": spec_hash,
                "status": "terminal",
                "trainer_reward_hash": None,
            },
        )
        terminal = store.put(
            "RunClosed",
            "1.0.0",
            {
                "action_observation_hash": action_observation.content_hash,
                "controller_epoch": epoch,
                "failure_kind": kind.value,
                "harness_decision_hash": decision.content_hash,
                "harness_outcome_hash": harness_outcome.content_hash,
                "audit_event_hash": audit_event.content_hash,
                "observation_hash": obs.content_hash,
                "action_plan_hash": proposal.content_hash,
                "optimizer_update_hash": None,
                "optimizer_update_permitted": False,
                "reason_code": kind.value,
                "reward_terminal_stop_hash": reward_stop.content_hash,
                "run_id": config.run_id,
                "spec_hash": spec_hash,
                "stop_request_hash": stop_request.content_hash,
                "status": "failed",
                "trainer_reward_hash": None,
            },
        )
        state = store.put(
            "HardFailureStopState",
            "1.0.0",
            {
                "controller_epoch": epoch,
                "failure_kind": kind.value,
                "harness_decision_hash": decision.content_hash,
                "harness_outcome_hash": harness_outcome.content_hash,
                "audit_event_hash": audit_event.content_hash,
                "observation_hash": obs.content_hash,
                "run_id": config.run_id,
                "spec_hash": spec_hash,
                "status": "terminal",
                "terminal_hash": terminal.content_hash,
            },
        )
        cls._publish_once(path / "state.ref", state.content_hash)
        for name, artifact in (
            ("request", stop_request),
            ("proposal", proposal),
            ("decision", decision),
            ("action-observation", action_observation),
            ("reward-stop", reward_stop),
            ("audit-event", audit_event),
        ):
            cls._publish_once(path / f"{name}.ref", artifact.content_hash)
        return HardFailureStopSnapshot(
            terminal, obs, stop_request, decision, proposal, action_observation, reward_stop, cluster
        )

    @staticmethod
    def _publish_once(path: Path, value: str) -> None:
        if path.exists():
            if path.read_text().strip() != value:
                raise HardFailureStopError("durable hard-failure ref conflict")
            return
        ArtifactStore._publish(path, f"{value}\n".encode("ascii"))

    @classmethod
    def _snapshot_from_hashes(cls, store: ArtifactStore, terminal_hash: str, path: Path) -> HardFailureStopSnapshot:
        terminal = store.read(terminal_hash, expected_schema_name="RunClosed")
        input_artifact = store.read(
            (path / "input.ref").read_text().strip(), expected_schema_name="HardFailureStopInput"
        )
        if set(input_artifact.payload) != {"controller_epoch", "phase", "run_id", "spec_hash"}:
            raise HardFailureStopError("hard-failure input lineage is invalid")
        run_id = input_artifact.payload.get("run_id")
        phase = input_artifact.payload.get("phase")
        spec_hash = input_artifact.payload.get("spec_hash")
        if (
            type(run_id) is not str
            or type(phase) is not str
            or type(spec_hash) is not str
            or terminal.payload.get("run_id") != run_id
            or terminal.payload.get("spec_hash") != spec_hash
        ):
            raise HardFailureStopError("hard-failure terminal identity changed")
        _hash(spec_hash, "spec_hash")
        store.read(spec_hash, expected_schema_name="ExperimentSpec")
        state = store.read((path / "state.ref").read_text().strip(), expected_schema_name="HardFailureStopState")
        if (
            set(state.payload)
            != {
                "audit_event_hash",
                "controller_epoch",
                "failure_kind",
                "harness_decision_hash",
                "harness_outcome_hash",
                "observation_hash",
                "run_id",
                "spec_hash",
                "status",
                "terminal_hash",
            }
            or state.payload.get("terminal_hash") != terminal_hash
            or state.payload.get("run_id") != run_id
            or state.payload.get("spec_hash") != spec_hash
            or state.payload.get("status") != "terminal"
        ):
            raise HardFailureStopError("hard-failure state lineage is invalid")
        obs = store.read((path / "observation.ref").read_text().strip(), expected_schema_name="Observation")
        if obs.payload.get("run_id") not in {None, run_id}:
            raise HardFailureStopError("hard-failure observation lineage is invalid")
        action_observation = store.read(
            cast(str, terminal.payload["action_observation_hash"]), expected_schema_name="ActionObservation"
        )
        reward_stop = store.read(
            cast(str, terminal.payload["reward_terminal_stop_hash"]), expected_schema_name="RewardTerminalStop"
        )
        proposal = store.read((path / "proposal.ref").read_text().strip(), expected_schema_name="ActionPlan")
        decision = store.read((path / "decision.ref").read_text().strip(), expected_schema_name="HarnessDecision")
        request = store.read((path / "request.ref").read_text().strip(), expected_schema_name="HardFailureStopRequest")
        expected_terminal = {
            "action_observation_hash",
            "action_plan_hash",
            "audit_event_hash",
            "controller_epoch",
            "failure_kind",
            "harness_decision_hash",
            "harness_outcome_hash",
            "observation_hash",
            "optimizer_update_hash",
            "optimizer_update_permitted",
            "reason_code",
            "reward_terminal_stop_hash",
            "run_id",
            "spec_hash",
            "status",
            "stop_request_hash",
            "trainer_reward_hash",
        }
        if (
            set(terminal.payload) != expected_terminal
            or terminal.payload.get("observation_hash") != obs.content_hash
            or terminal.payload.get("stop_request_hash") != request.content_hash
            or terminal.payload.get("action_plan_hash") != proposal.content_hash
            or terminal.payload.get("harness_decision_hash") != decision.content_hash
            or type(terminal.payload.get("harness_outcome_hash")) is not str
            or type(terminal.payload.get("audit_event_hash")) is not str
            or terminal.payload.get("status") != "failed"
        ):
            raise HardFailureStopError("hard-failure terminal lineage is invalid")
        if (
            set(request.payload) != {"failure_kind", "observation_hash", "run_id", "spec_hash", "status"}
            or request.payload.get("observation_hash") != obs.content_hash
            or request.payload.get("run_id") != run_id
            or request.payload.get("spec_hash") != spec_hash
            or request.payload.get("status") != "terminal"
        ):
            raise HardFailureStopError("hard-failure request lineage is invalid")
        if (
            not {
                "action_type",
                "budget_cost",
                "controller_epoch",
                "emergency",
                "input_hash",
                "run_id",
                "spec_hash",
                "previous_transition_hash",
                "proposal_id",
                "workflow_sequence",
            }.issubset(proposal.payload)
            or proposal.payload.get("input_hash") != request.content_hash
            or proposal.payload.get("run_id") != run_id
            or proposal.payload.get("spec_hash") != spec_hash
            or proposal.payload.get("emergency") is not True
        ):
            raise HardFailureStopError("hard-failure action plan lineage is invalid")
        if (
            not {
                "action_type",
                "controller_epoch",
                "decision",
                "emergency",
                "input_hash",
                "plan_hash",
                "reason_code",
                "run_id",
                "spec_hash",
                "previous_transition_hash",
                "proposal_id",
                "workflow_sequence",
            }.issubset(decision.payload)
            or decision.payload.get("plan_hash") != proposal.content_hash
            or decision.payload.get("input_hash") != request.content_hash
            or decision.payload.get("run_id") != run_id
            or decision.payload.get("spec_hash") != spec_hash
            or decision.payload.get("decision") != "allow"
        ):
            raise HardFailureStopError("hard-failure harness decision lineage is invalid")
        if (
            action_observation.payload.get("cluster_run_record_hash") is None
            or action_observation.payload.get("plan_hash") != proposal.content_hash
            or action_observation.payload.get("run_id") != run_id
            or action_observation.payload.get("spec_hash") != spec_hash
        ):
            raise HardFailureStopError("hard-failure action observation lineage is invalid")
        outcome = store.read(
            cast(str, terminal.payload["harness_outcome_hash"]), expected_schema_name="DecisionOutcome"
        )
        if (
            outcome.payload.get("terminal") is not True
            or outcome.payload.get("action_observation_hash") != action_observation.content_hash
            or outcome.payload.get("decision_hash") != decision.content_hash
            or outcome.payload.get("plan_hash") != proposal.content_hash
            or outcome.payload.get("spec_hash") != spec_hash
        ):
            raise HardFailureStopError("hard-failure Harness outcome lineage is invalid")
        audit_event = store.read(cast(str, terminal.payload["audit_event_hash"]), expected_schema_name="AuditEvent")
        if (
            audit_event.payload.get("action_plan_hash") != proposal.content_hash
            or audit_event.payload.get("decision_hash") != decision.content_hash
            or audit_event.payload.get("outcome_hash") != outcome.content_hash
            or audit_event.payload.get("run_id") != run_id
            or audit_event.payload.get("spec_hash") != spec_hash
            or audit_event.payload.get("terminal") is not True
        ):
            raise HardFailureStopError("hard-failure audit lineage is invalid")
        journal = HarnessJournal(path.parents[1], store, cls._harness_proposal_id(run_id))
        journal.verify()
        transitions = journal.strict_transition_snapshot()
        if (
            len(transitions) != 5
            or transitions[1].content_hash != proposal.content_hash
            or transitions[2].content_hash != decision.content_hash
            or transitions[3].content_hash != action_observation.content_hash
            or transitions[4].content_hash != outcome.content_hash
        ):
            raise HardFailureStopError("hard-failure Harness transition lineage is invalid")
        if (
            reward_stop.payload.get("observation_hash") != obs.content_hash
            or reward_stop.payload.get("run_id") != run_id
            or reward_stop.payload.get("spec_hash") != spec_hash
            or reward_stop.payload.get("optimizer_update_permitted") is not False
        ):
            raise HardFailureStopError("hard-failure reward stop lineage is invalid")
        config = HardFailureStopConfig(run_id, experiment_spec_hash=spec_hash, phase=cast(str, phase))
        cluster = ClusterLifecycleWorkflow.resume(path.parents[1], config.run_id)
        return HardFailureStopSnapshot(
            terminal, obs, request, decision, proposal, action_observation, reward_stop, cluster
        )


# Short names make the fixture pleasant to use in integration tests.
HardFailureWorkflow = HardFailureStopWorkflow
HardFailureController = HardFailureStopWorkflow
HardFailureMonitor = HardFailureStopWorkflow
FencedRunController = HardFailureStopWorkflow
FixtureHardFailureSimulator = HardFailureStopWorkflow
