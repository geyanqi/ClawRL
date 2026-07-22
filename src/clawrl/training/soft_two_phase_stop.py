# ruff: noqa: E501

"""Production-shaped fixture for Governor soft two-phase stopping.

The monitor is intentionally evidence-only.  A frozen :class:`MonitoringPolicy`
and immutable Observation produce one bounded Harness plan; the plan drives the
existing durable ClusterLifecycle (checkpoint, configured grace, force_cancel).
No trainer, scalarizer, prompt, or generation setting is changed on this path.
"""

from __future__ import annotations

import fcntl
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.harness.journal import HarnessJournal, HarnessJournalError
from clawrl.training.cluster_lifecycle import (
    ClusterLifecycleSnapshot,
    ClusterLifecycleWorkflow,
    FixtureClusterLifecycleConfig,
    InjectedClusterControllerCrash,
    _replace_ref,
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PHASES = {"TRAIN_35B", "TRAIN_122B"}


class SoftStopError(RuntimeError):
    """Soft-stop controller failed closed."""


class StaleSoftStopController(SoftStopError):
    """A fenced controller epoch no longer owns the run."""


class InjectedSoftStopControllerCrash(SoftStopError):
    """Fixture crash after a durable phase boundary."""


class SoftStopKind(StrEnum):
    KL_DIVERGENCE = "KL_DIVERGENCE"
    ENTROPY_COLLAPSE = "ENTROPY_COLLAPSE"
    LENGTH_ANOMALY = "LENGTH_ANOMALY"
    JUDGE_FAILURE = "JUDGE_FAILURE"


def _id(value: object, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(cast(str, value)) is None:
        raise SoftStopError(f"{field} is invalid")
    return cast(str, value)


def _hash(value: object, field: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise SoftStopError(f"{field} is invalid")
    return cast(str, value)


def _safe(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return cast(JsonValue, value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe(v) for v in value]
    return str(value)


def _numeric(value: object) -> float:
    if not isinstance(value, (str, int, float)):
        raise TypeError("value is not numeric")
    return float(value)


@dataclass(frozen=True, slots=True)
class MonitoringPolicy:
    """Frozen thresholds and grace; values are strings to remain canonical JSON."""

    policy_id: str = "fixture-monitoring-policy"
    policy_version: str = "fixture-monitoring-policy/1.0.0"
    grace_period: int = 1
    kl_threshold: str = "0.20"
    entropy_floor: str = "0.10"
    length_ratio: str = "2.00"
    judge_failure_rate: str = "0.50"

    def __post_init__(self) -> None:
        _id(self.policy_id, "policy_id")
        if type(self.policy_version) is not str or not self.policy_version:
            raise SoftStopError("policy_version is invalid")
        if type(self.grace_period) is not int or self.grace_period < 0:
            raise SoftStopError("grace_period is invalid")
        for name in ("kl_threshold", "entropy_floor", "length_ratio", "judge_failure_rate"):
            if type(getattr(self, name)) is not str:
                raise SoftStopError(f"{name} is invalid")

    def payload(self) -> dict[str, JsonValue]:
        return {
            "entropy_floor": self.entropy_floor,
            "grace_period": self.grace_period,
            "judge_failure_rate": self.judge_failure_rate,
            "kl_threshold": self.kl_threshold,
            "length_ratio": self.length_ratio,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True, slots=True)
class SoftTwoPhaseStopConfig:
    run_id: str
    experiment_spec_hash: str = ""
    phase: str = "TRAIN_35B"
    controller_epoch: int = 1
    grace_period: int | None = None
    grace_ticks: int | None = None
    configured_grace: int | None = None
    grace: int | None = None
    monitoring_policy: MonitoringPolicy | Artifact | Mapping[str, object] | None = None
    checkpoint_fault: str | None = None  # timeout or failure
    spec_hash: str = ""
    budget_exhausted: bool = False

    def __post_init__(self) -> None:
        _id(self.run_id, "run_id")
        if self.phase not in _PHASES:
            raise SoftStopError("phase is invalid")
        if type(self.controller_epoch) is not int or self.controller_epoch < 1:
            raise SoftStopError("controller_epoch is invalid")
        if self.experiment_spec_hash and self.spec_hash and self.experiment_spec_hash != self.spec_hash:
            raise SoftStopError("experiment_spec_hash and spec_hash conflict")
        resolved = self.spec_hash or self.experiment_spec_hash
        if resolved:
            _hash(resolved, "spec_hash")
            object.__setattr__(self, "spec_hash", resolved)
        aliases = [
            value
            for value in (self.grace_period, self.grace_ticks, self.configured_grace, self.grace)
            if value is not None
        ]
        if any(type(value) is not int or value < 0 for value in aliases):
            raise SoftStopError("grace_period is invalid")
        if aliases and any(value != aliases[0] for value in aliases):
            raise SoftStopError("grace period aliases conflict")
        if aliases and self.grace_period is None:
            object.__setattr__(self, "grace_period", aliases[0])
        if self.checkpoint_fault not in {None, "timeout", "failure", "provider_error"}:
            raise SoftStopError("checkpoint_fault is invalid")
        if type(self.budget_exhausted) is not bool:
            raise SoftStopError("budget_exhausted is invalid")


@dataclass(frozen=True, slots=True)
class SoftTwoPhaseStopObservation:
    run_id: str
    metrics: Mapping[str, object] | None = None
    evidence: Mapping[str, object] | None = None
    sequence: int = 1
    controller_epoch: int = 1
    reward_delta: object | None = None

    def __post_init__(self) -> None:
        _id(self.run_id, "run_id")
        if type(self.sequence) is not int or self.sequence < 1:
            raise SoftStopError("observation sequence is invalid")
        if type(self.controller_epoch) is not int or self.controller_epoch < 1:
            raise SoftStopError("observation controller epoch is invalid")

    def payload(self) -> dict[str, JsonValue]:
        metrics = dict(self.metrics or {})
        if self.reward_delta is not None:
            metrics.setdefault("reward_delta", _safe(self.reward_delta))
        return {
            "controller_epoch": self.controller_epoch,
            "evidence": cast(dict[str, JsonValue], _safe(self.evidence or {})),
            "metrics": cast(dict[str, JsonValue], _safe(metrics)),
            "observation_sequence": self.sequence,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class SoftTwoPhaseStopSnapshot:
    terminal: Artifact | None
    observation: Artifact
    stop_request: Artifact | None
    harness_decision: Artifact | None
    action_plan: Artifact | None
    action_observation: Artifact | None
    decision_outcome: Artifact | None
    cluster: ClusterLifecycleSnapshot | None
    monitoring_policy: Artifact
    grace: Artifact | None = None
    continued: bool = False

    @property
    def run_closed(self) -> Artifact | None:
        return self.terminal

    @property
    def outcome(self) -> Artifact | None:
        return self.decision_outcome


class SoftTwoPhaseStopWorkflow:
    """Durable, fenced, exactly-once checkpoint→grace→cancel fixture."""

    @staticmethod
    def production_readiness(root: str | Path, *, phase: str = "TRAIN_35B") -> Artifact:
        if phase not in _PHASES:
            raise SoftStopError("phase is invalid")
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": "PRODUCTION_MONITOR_POLICY_UNVERIFIED", "status": "blocked"},
                    {"code": "PRODUCTION_CLUSTER_CHECKPOINT_UNVERIFIED", "status": "blocked"},
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
        return action in {"checkpoint", "force_cancel"}

    @staticmethod
    def classify(
        observation: Artifact | Mapping[str, object] | SoftTwoPhaseStopObservation,
        policy: Artifact | MonitoringPolicy | Mapping[str, object] | None = None,
    ) -> SoftStopKind | None:
        payload: Mapping[str, object]
        if isinstance(observation, SoftTwoPhaseStopObservation):
            payload = observation.metrics or {}
        elif isinstance(observation, Artifact):
            candidate = observation.payload.get("metrics")
            payload = (
                cast(Mapping[str, object], candidate)
                if isinstance(candidate, Mapping)
                else cast(Mapping[str, object], observation.payload)
            )
        else:
            payload = observation
        explicit = payload.get("signal") or payload.get("failure_kind") or payload.get("reason_code")
        if isinstance(explicit, str):
            normalized = explicit.upper().replace("-", "_").replace(" ", "_")
            for item in SoftStopKind:
                if normalized == item.value:
                    return item
        # A reward increase alone is expressly not a stop signal.
        candidates: list[tuple[str, SoftStopKind]] = [
            ("kl", SoftStopKind.KL_DIVERGENCE),
            ("kl_divergence", SoftStopKind.KL_DIVERGENCE),
            ("entropy_collapse", SoftStopKind.ENTROPY_COLLAPSE),
            ("entropy", SoftStopKind.ENTROPY_COLLAPSE),
            ("length_anomaly", SoftStopKind.LENGTH_ANOMALY),
            ("length_ratio", SoftStopKind.LENGTH_ANOMALY),
            ("judge_failure_rate", SoftStopKind.JUDGE_FAILURE),
            ("judge_failures", SoftStopKind.JUDGE_FAILURE),
        ]
        policy_payload: Mapping[str, object] = (
            cast(Mapping[str, object], policy.payload)
            if isinstance(policy, Artifact)
            else cast(Mapping[str, object], policy.payload())
            if isinstance(policy, MonitoringPolicy)
            else policy or cast(Mapping[str, object], MonitoringPolicy().payload())
        )
        try:
            raw_thresholds = [
                policy_payload.get("kl_threshold", "0.20"),
                policy_payload.get("entropy_floor", "0.10"),
                policy_payload.get("length_ratio", "2.00"),
                policy_payload.get("judge_failure_rate", "0.50"),
            ]
            kl_threshold, entropy_floor, length_ratio, judge_failure_rate = (
                _numeric(value) for value in raw_thresholds
            )
        except (TypeError, ValueError) as error:
            raise SoftStopError("monitoring policy thresholds are invalid") from error
        for key, kind in candidates:
            if key in payload:
                value = payload[key]
                if value is True:
                    return kind
                try:
                    numeric = float(value) if isinstance(value, (int, float, str)) else None
                except ValueError:
                    numeric = None
                if numeric is None:
                    continue
                if kind is SoftStopKind.KL_DIVERGENCE and numeric > kl_threshold:
                    return kind
                if kind is SoftStopKind.ENTROPY_COLLAPSE and numeric < entropy_floor:
                    return kind
                if kind is SoftStopKind.LENGTH_ANOMALY and numeric > length_ratio:
                    return kind
                if kind is SoftStopKind.JUDGE_FAILURE and numeric > judge_failure_rate:
                    return kind
        return None

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: SoftTwoPhaseStopConfig,
        observation: Artifact | Mapping[str, object] | SoftTwoPhaseStopObservation | None = None,
        monitoring_policy: MonitoringPolicy | Artifact | Mapping[str, object] | None = None,
        controller_epoch: int | None = None,
        crash_after: str | None = None,
        execution_profile: str = "fixture",
        **_: object,
    ) -> SoftTwoPhaseStopSnapshot:
        if execution_profile not in {"fixture", "production"}:
            raise SoftStopError("execution_profile must be fixture or production")
        if execution_profile == "production":
            raise SoftStopError(
                "production soft two-phase stop is blocked until monitoring, Harness, and Cluster adapters are verified"
            )
        epoch = config.controller_epoch if controller_epoch is None else controller_epoch
        if type(epoch) is not int or epoch < 1 or epoch < config.controller_epoch:
            raise StaleSoftStopController("controller epoch is stale")
        store = ArtifactStore(root)
        policy = monitoring_policy or config.monitoring_policy or MonitoringPolicy()
        policy_artifact = cls._policy_artifact(store, policy)
        if observation is None:
            raise SoftStopError("soft-stop observation is required")
        obs = cls._observation_artifact(store, config, observation)
        kind = cls.classify(obs, policy_artifact)
        # Reward-only progress returns an immutable continue evidence record.
        if kind is None:
            continued = store.put(
                "SoftStopRequest",
                "1.0.0",
                {
                    "observation_hash": obs.content_hash,
                    "policy_hash": policy_artifact.content_hash,
                    "reason_code": "REWARD_ONLY_PROGRESS",
                    "run_id": config.run_id,
                    "stop": False,
                    "status": "continued",
                },
            )
            return SoftTwoPhaseStopSnapshot(
                None, obs, continued, None, None, None, None, None, policy_artifact, None, True
            )
        path = Path(root) / "soft-two-phase-stops" / config.run_id
        ArtifactStore.durable_mkdir(path)
        ArtifactStore.durable_touch(path / "controller.lock")
        with (path / "controller.lock").open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                return cls._run_locked(root, path, config, epoch, kind, obs, policy_artifact, crash_after)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @classmethod
    def resume(cls, root: str | Path, run_id: str, *, controller_epoch: int | None = None) -> SoftTwoPhaseStopSnapshot:
        _id(run_id, "run_id")
        path = Path(root) / "soft-two-phase-stops" / run_id
        store = ArtifactStore(root)
        try:
            input_artifact = store.read((path / "input.ref").read_text().strip(), expected_schema_name="SoftStopInput")
            payload = input_artifact.payload
            config = SoftTwoPhaseStopConfig(
                run_id,
                experiment_spec_hash=cast(str, payload["spec_hash"]),
                phase=cast(str, payload["phase"]),
                controller_epoch=cast(int, payload["controller_epoch"]),
                grace_period=cast(int, payload["grace_period"]),
                checkpoint_fault=cast(str | None, payload.get("checkpoint_fault")),
                budget_exhausted=payload.get("budget_exhausted") is True,
            )
            obs = store.read((path / "observation.ref").read_text().strip(), expected_schema_name="Observation")
            policy = store.read((path / "policy.ref").read_text().strip(), expected_schema_name="MonitoringPolicy")
        except (OSError, ArtifactCorruption, KeyError, TypeError) as error:
            raise SoftStopError("soft stop cannot be recovered") from error
        return cls.run(
            root, config=config, observation=obs, monitoring_policy=policy, controller_epoch=controller_epoch
        )

    @classmethod
    def _run_locked(
        cls,
        root: str | Path,
        path: Path,
        config: SoftTwoPhaseStopConfig,
        epoch: int,
        kind: SoftStopKind,
        obs: Artifact,
        policy: Artifact,
        crash_after: str | None,
    ) -> SoftTwoPhaseStopSnapshot:
        store = ArtifactStore(root)
        if (path / "state.ref").exists():
            input_artifact = store.read((path / "input.ref").read_text().strip(), expected_schema_name="SoftStopInput")
            if set(input_artifact.payload) != {
                "budget_exhausted",
                "checkpoint_fault",
                "controller_epoch",
                "grace_period",
                "phase",
                "run_id",
                "spec_hash",
            }:
                raise SoftStopError("soft-stop input lineage is invalid")
            if (
                input_artifact.payload.get("run_id") != config.run_id
                or input_artifact.payload.get("phase") != config.phase
            ):
                raise SoftStopError("soft-stop run identity changed")
            durable_spec = input_artifact.payload.get("spec_hash")
            if type(durable_spec) is not str:
                raise SoftStopError("soft-stop spec identity is invalid")
            _hash(durable_spec, "spec_hash")
            store.read(durable_spec, expected_schema_name="ExperimentSpec")
            config = replace(
                config,
                spec_hash=durable_spec,
                experiment_spec_hash=durable_spec,
                grace_period=cast(int, input_artifact.payload["grace_period"]),
                checkpoint_fault=cast(str | None, input_artifact.payload.get("checkpoint_fault")),
                budget_exhausted=input_artifact.payload.get("budget_exhausted") is True,
            )
            durable_obs = store.read((path / "observation.ref").read_text().strip(), expected_schema_name="Observation")
            durable_policy = store.read(
                (path / "policy.ref").read_text().strip(), expected_schema_name="MonitoringPolicy"
            )
            obs = durable_obs
            policy = durable_policy
            state = store.read((path / "state.ref").read_text().strip(), expected_schema_name="SoftTwoPhaseStopState")
            current_epoch = state.payload.get("controller_epoch")
            if type(current_epoch) is not int or epoch < current_epoch:
                raise StaleSoftStopController("controller fencing epoch is stale")
            if state.payload.get("run_id") != config.run_id:
                raise SoftStopError("soft-stop run identity changed")
            if state.payload.get("terminal_hash"):
                return cls._snapshot_from_state(root, path, state)
            try:
                committed_input = store.read(
                    (path / "input.ref").read_text().strip(), expected_schema_name="SoftStopInput"
                )
                input_payload = committed_input.payload
                committed_spec = input_payload.get("spec_hash")
                if (
                    input_payload.get("run_id") != config.run_id
                    or input_payload.get("phase") != config.phase
                    or type(committed_spec) is not str
                    or (config.spec_hash and config.spec_hash != committed_spec)
                ):
                    raise SoftStopError("soft-stop frozen input identity changed")
                config = replace(
                    config,
                    spec_hash=cast(str, committed_spec),
                    grace_period=cast(int, input_payload.get("grace_period")),
                )
                committed_obs = store.read(
                    (path / "observation.ref").read_text().strip(), expected_schema_name="Observation"
                )
                committed_policy = store.read(
                    (path / "policy.ref").read_text().strip(), expected_schema_name="MonitoringPolicy"
                )
                if (
                    committed_obs.content_hash != obs.content_hash
                    or committed_policy.content_hash != policy.content_hash
                ):
                    raise SoftStopError("soft-stop frozen observation or policy changed")
            except (OSError, ArtifactCorruption, KeyError, TypeError) as error:
                raise SoftStopError("soft-stop durable input cannot be recovered") from error
            if state.payload.get("grace_hash"):
                grace = store.read(cast(str, state.payload["grace_hash"]), expected_schema_name="SoftStopGrace")
            else:
                grace = None
            journal = HarnessJournal(root, store, cls._proposal_id(config.run_id))
            try:
                journal.claim_epoch(epoch)
            except HarnessJournalError as error:
                raise StaleSoftStopController("soft-stop Harness fencing epoch is stale") from error
            transitions = journal.strict_transition_snapshot()
            request_ref = store.read((path / "request.ref").read_text().strip(), expected_schema_name="SoftStopRequest")
            plan_ref = store.read((path / "plan.ref").read_text().strip(), expected_schema_name="ActionPlan")
            decision_ref = store.read(
                (path / "decision.ref").read_text().strip(), expected_schema_name="HarnessDecision"
            )
            if (
                len(transitions) < 3
                or transitions[1].content_hash != plan_ref.content_hash
                or transitions[2].content_hash != decision_ref.content_hash
                or request_ref.payload.get("evidence_hash") != obs.content_hash
                or request_ref.payload.get("monitoring_policy_hash") != policy.content_hash
                or request_ref.payload.get("spec_hash") != config.spec_hash
            ):
                raise SoftStopError("soft-stop durable action lineage is invalid")
        else:
            grace = None
            spec_hash = config.spec_hash
            if not spec_hash:
                spec = store.put("ExperimentSpec", "1.0.0", {"experiment_id": config.run_id, "status": "frozen"})
                spec_hash = spec.content_hash
                config = replace(config, spec_hash=spec_hash)
            else:
                store.read(spec_hash, expected_schema_name="ExperimentSpec")
            input_artifact = store.put(
                "SoftStopInput",
                "1.0.0",
                {
                    "budget_exhausted": config.budget_exhausted,
                    "checkpoint_fault": config.checkpoint_fault,
                    "controller_epoch": config.controller_epoch,
                    "grace_period": config.grace_period
                    if config.grace_period is not None
                    else cast(int, policy.payload["grace_period"]),
                    "phase": config.phase,
                    "run_id": config.run_id,
                    "spec_hash": spec_hash,
                },
            )
            cls._publish_once(path / "input.ref", input_artifact.content_hash)
            cls._publish_once(path / "observation.ref", obs.content_hash)
            cls._publish_once(path / "policy.ref", policy.content_hash)
            config = replace(
                config, spec_hash=spec_hash, grace_period=cast(int, input_artifact.payload["grace_period"])
            )
            stop_request = store.put(
                "SoftStopRequest",
                "1.0.0",
                {
                    "evidence_hash": obs.content_hash,
                    "monitoring_policy_hash": policy.content_hash,
                    "reason_code": kind.value,
                    "run_id": config.run_id,
                    "spec_hash": spec_hash,
                    "status": "proposed",
                },
            )
            proposal_id = cls._proposal_id(config.run_id)
            journal = HarnessJournal(root, store, proposal_id)
            harness_config = store.put(
                "FixtureHarnessConfig",
                "1.0.0",
                {
                    "execution_profile": "fixture",
                    "policy_id": "soft-stop",
                    "allowed_action_types": ["cluster.soft_stop"],
                },
            )
            harness_policy = store.put(
                "HarnessPolicy",
                "1.0.0",
                {"policy_id": "soft-stop", "monitoring_policy_hash": policy.content_hash, "spec_hash": spec_hash},
            )
            capability = store.put(
                "HarnessPolicyCapability",
                "1.0.0",
                {
                    "action_type": "cluster.soft_stop",
                    "policy_hash": harness_policy.content_hash,
                    "spec_hash": spec_hash,
                },
            )
            workflow_input = store.put(
                "HarnessWorkflowInput",
                "1.0.0",
                {
                    "action": {"action_type": "cluster.soft_stop", "input_hash": stop_request.content_hash},
                    "caller": "governor",
                    "phase": config.phase,
                    "spec_hash": spec_hash,
                },
            )
            try:
                journal.reserve_identity(
                    workflow_input.content_hash,
                    harness_config.content_hash,
                    harness_policy.content_hash,
                    capability.content_hash,
                )
                journal.claim_epoch(epoch)
                proposal = journal.append(
                    epoch,
                    "DecisionProposal",
                    {
                        "action_type": "cluster.soft_stop",
                        "caller": "governor",
                        "evidence_hash": obs.content_hash,
                        "input_hash": stop_request.content_hash,
                        "monitoring_policy_hash": policy.content_hash,
                        "phase": config.phase,
                        "policy_capability_hash": capability.content_hash,
                        "policy_hash": harness_policy.content_hash,
                        "spec_hash": spec_hash,
                    },
                    expected_sequence=1,
                    expected_previous_hash=None,
                )
                action_names = ["checkpoint", "grace", "force_cancel"]
                idempotency_keys = [
                    sha256_hex(
                        canonical_json_bytes(
                            {
                                "action": action,
                                "domain": "soft-stop-action/1.0.0",
                                "run_id": config.run_id,
                                "spec_hash": spec_hash,
                            }
                        )
                    )
                    for action in action_names
                ]
                plan = journal.append(
                    epoch,
                    "ActionPlan",
                    {
                        "action_type": "cluster.soft_stop",
                        "actions": action_names,
                        "action_idempotency_keys": idempotency_keys,
                        "budget_cost": 0,
                        "emergency": config.budget_exhausted,
                        "input_hash": stop_request.content_hash,
                        "run_id": config.run_id,
                        "spec_hash": spec_hash,
                    },
                    expected_sequence=2,
                    expected_previous_hash=proposal.content_hash,
                )
                decision = journal.append(
                    epoch,
                    "HarnessDecision",
                    {
                        "action_type": "cluster.soft_stop",
                        "decision": "allow",
                        "emergency": config.budget_exhausted,
                        "reason_code": "SOFT_MONITORING_THRESHOLD",
                        "input_hash": stop_request.content_hash,
                        "plan_hash": plan.content_hash,
                        "run_id": config.run_id,
                        "spec_hash": spec_hash,
                    },
                    expected_sequence=3,
                    expected_previous_hash=plan.content_hash,
                )
            except HarnessJournalError as error:
                raise SoftStopError("soft-stop Harness authorization failed closed") from error
            grace = store.put(
                "SoftStopGrace",
                "1.0.0",
                {
                    "configured_period": config.grace_period,
                    "elapsed": True,
                    "policy_hash": policy.content_hash,
                    "run_id": config.run_id,
                    "status": "elapsed",
                },
            )
            state = store.put(
                "SoftTwoPhaseStopState",
                "1.0.0",
                {
                    "controller_epoch": epoch,
                    "grace_hash": grace.content_hash,
                    "run_id": config.run_id,
                    "spec_hash": spec_hash,
                    "stage": "grace",
                    "terminal_hash": None,
                },
            )
            cls._publish_once(path / "state.ref", state.content_hash)
            for name, artifact in (
                ("request", stop_request),
                ("proposal", proposal),
                ("plan", plan),
                ("decision", decision),
                ("grace", grace),
            ):
                cls._publish_once(path / f"{name}.ref", artifact.content_hash)
            if crash_after == "grace":
                raise InjectedSoftStopControllerCrash("injected crash after configured grace")
        # ClusterLifecycle's force_cancel plan is exactly checkpoint → cancel;
        # this pre-bound grace artifact is the configured waiting boundary.
        cluster_config = FixtureClusterLifecycleConfig(
            config.run_id,
            cast(Literal["TRAIN_35B", "TRAIN_122B"], config.phase),
            close_mode="force_cancel",
            fault_operation="checkpoint" if config.checkpoint_fault else None,
            fault_directive=(
                "soft_stop_checkpoint_timeout"
                if config.checkpoint_fault == "timeout"
                else "soft_stop_checkpoint_failure"
                if config.checkpoint_fault in {"failure", "provider_error"}
                else None
            ),
        )
        try:
            cluster_crash_operation = (
                "checkpoint"
                if crash_after == "checkpoint"
                else "force_cancel"
                if crash_after in {"force_cancel", "cancel", "cluster_cancel", "provider_cancel"}
                else None
            )
            cluster = (
                ClusterLifecycleWorkflow.resume(root, config.run_id)
                if (Path(root) / "cluster-lifecycle-runs" / config.run_id / "input.ref").exists()
                else ClusterLifecycleWorkflow.run(
                    root,
                    config=cluster_config,
                    spec_hash=config.spec_hash,
                    crash_after_operation=cluster_crash_operation,
                )
            )
        except InjectedClusterControllerCrash as error:
            if crash_after in {"checkpoint", "force_cancel", "cancel", "cluster_cancel", "provider_cancel"}:
                raise InjectedSoftStopControllerCrash(f"injected crash after {crash_after} provider commit") from error
            cluster = ClusterLifecycleWorkflow.resume(root, config.run_id)
        journal = HarnessJournal(root, store, cls._proposal_id(config.run_id))
        try:
            journal.claim_epoch(epoch)
        except HarnessJournalError as error:
            raise StaleSoftStopController("soft-stop Harness epoch is stale") from error
        transitions = journal.strict_transition_snapshot()
        plan = transitions[1]
        decision = transitions[2]
        if len(transitions) == 3:
            action_observation = journal.append(
                epoch,
                "ActionObservation",
                {
                    "action_type": "cluster.soft_stop",
                    "cluster_run_record_hash": cluster.run_record.content_hash,
                    "idempotency_key": sha256_hex(
                        canonical_json_bytes({"run_id": config.run_id, "action": "soft_stop", "epoch": epoch})
                    ),
                    "plan_hash": plan.content_hash,
                    "run_id": config.run_id,
                    "spec_hash": config.spec_hash,
                    "status": "succeeded" if cluster.run_record.payload.get("status") == "canceled" else "failed",
                },
                expected_sequence=4,
                expected_previous_hash=decision.content_hash,
            )
        else:
            action_observation = transitions[3]
        transitions = journal.strict_transition_snapshot()
        if len(transitions) == 4:
            audit = store.put(
                "AuditEvent",
                "1.0.0",
                {
                    "action_observation_hash": action_observation.content_hash,
                    "decision_hash": decision.content_hash,
                    "plan_hash": plan.content_hash,
                    "run_id": config.run_id,
                    "spec_hash": config.spec_hash,
                    "status": "canceled" if cluster.run_record.payload.get("status") == "canceled" else "failed",
                },
            )
            outcome = journal.append(
                epoch,
                "DecisionOutcome",
                {
                    "action_observation_hash": action_observation.content_hash,
                    "audit_event_hash": audit.content_hash,
                    "decision_hash": decision.content_hash,
                    "plan_hash": plan.content_hash,
                    "run_id": config.run_id,
                    "spec_hash": config.spec_hash,
                    "status": "canceled" if cluster.run_record.payload.get("status") == "canceled" else "failed",
                    "terminal": True,
                },
                expected_sequence=5,
                expected_previous_hash=action_observation.content_hash,
            )
        else:
            outcome = transitions[4]
        terminal = store.put(
            "RunClosed",
            "1.0.0",
            {
                "action_observation_hash": action_observation.content_hash,
                "audit_event_hash": outcome.payload.get("audit_event_hash"),
                "controller_epoch": epoch,
                "harness_decision_hash": decision.content_hash,
                "harness_outcome_hash": outcome.content_hash,
                "monitoring_policy_hash": policy.content_hash,
                "observation_hash": obs.content_hash,
                "action_plan_hash": plan.content_hash,
                "run_id": config.run_id,
                "spec_hash": config.spec_hash,
                "status": outcome.payload["status"],
                "stop_request_hash": store.read(
                    (path / "request.ref").read_text().strip(), expected_schema_name="SoftStopRequest"
                ).content_hash,
            },
        )
        state = store.put(
            "SoftTwoPhaseStopState",
            "1.0.0",
            {
                "controller_epoch": epoch,
                "grace_hash": grace.content_hash if grace else None,
                "run_id": config.run_id,
                "spec_hash": config.spec_hash,
                "stage": "terminal",
                "terminal_hash": terminal.content_hash,
            },
        )
        _replace_ref(path / "state.ref", state.content_hash)
        return SoftTwoPhaseStopSnapshot(
            terminal,
            obs,
            store.read((path / "request.ref").read_text().strip(), expected_schema_name="SoftStopRequest"),
            decision,
            plan,
            action_observation,
            outcome,
            cluster,
            policy,
            grace,
            False,
        )

    @staticmethod
    def _policy_artifact(store: ArtifactStore, policy: MonitoringPolicy | Artifact | Mapping[str, object]) -> Artifact:
        if isinstance(policy, Artifact):
            if policy.schema_name != "MonitoringPolicy":
                raise SoftStopError("monitoring_policy must be MonitoringPolicy")
            return policy
        if isinstance(policy, MonitoringPolicy):
            model = policy
        else:
            value = dict(policy)
            model = MonitoringPolicy(
                policy_id=cast(str, value.get("policy_id", "fixture-monitoring-policy")),
                policy_version=cast(str, value.get("policy_version", "fixture-monitoring-policy/1.0.0")),
                grace_period=cast(int, value.get("grace_period", 1)),
                kl_threshold=cast(str, value.get("kl_threshold", "0.20")),
                entropy_floor=cast(str, value.get("entropy_floor", "0.10")),
                length_ratio=cast(str, value.get("length_ratio", "2.00")),
                judge_failure_rate=cast(str, value.get("judge_failure_rate", "0.50")),
            )
        return store.put("MonitoringPolicy", "1.0.0", model.payload())

    @staticmethod
    def _observation_artifact(
        store: ArtifactStore,
        config: SoftTwoPhaseStopConfig,
        observation: Artifact | Mapping[str, object] | SoftTwoPhaseStopObservation,
    ) -> Artifact:
        if isinstance(observation, Artifact):
            if observation.schema_name != "Observation" or observation.payload.get("run_id") not in {
                None,
                config.run_id,
            }:
                raise SoftStopError("soft-stop observation identity changed")
            return observation
        if isinstance(observation, SoftTwoPhaseStopObservation):
            if observation.run_id != config.run_id:
                raise SoftStopError("observation run identity changed")
            if observation.controller_epoch != config.controller_epoch:
                raise SoftStopError("observation controller epoch changed")
            return store.put("Observation", "1.0.0", observation.payload())
        return store.put(
            "Observation",
            "1.0.0",
            {
                "evidence": cast(dict[str, JsonValue], _safe(observation)),
                "metrics": cast(dict[str, JsonValue], _safe(observation)),
                "run_id": config.run_id,
            },
        )

    @staticmethod
    def _proposal_id(run_id: str) -> str:
        return (
            f"soft-stop-{run_id}"
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}", run_id)
            else f"soft-stop-{sha256_hex(run_id.encode())}"
        )

    @staticmethod
    def _publish_once(path: Path, value: str) -> None:
        if path.exists():
            if path.read_text().strip() != value:
                raise SoftStopError("durable soft-stop ref conflict")
            return
        ArtifactStore._publish(path, f"{value}\n".encode("ascii"))

    @classmethod
    def _snapshot_from_state(cls, root: str | Path, path: Path, state: Artifact) -> SoftTwoPhaseStopSnapshot:
        store = ArtifactStore(root)
        if (
            set(state.payload) != {"controller_epoch", "grace_hash", "run_id", "spec_hash", "stage", "terminal_hash"}
            or state.payload.get("stage") != "terminal"
            or type(state.payload.get("run_id")) is not str
            or type(state.payload.get("spec_hash")) is not str
        ):
            raise SoftStopError("soft-stop terminal state lineage is invalid")
        terminal = store.read(cast(str, state.payload["terminal_hash"]), expected_schema_name="RunClosed")
        obs = store.read((path / "observation.ref").read_text().strip(), expected_schema_name="Observation")
        policy = store.read((path / "policy.ref").read_text().strip(), expected_schema_name="MonitoringPolicy")
        request = store.read((path / "request.ref").read_text().strip(), expected_schema_name="SoftStopRequest")
        plan_ref = store.read((path / "plan.ref").read_text().strip(), expected_schema_name="ActionPlan")
        decision_ref = store.read((path / "decision.ref").read_text().strip(), expected_schema_name="HarnessDecision")
        if (
            terminal.payload.get("run_id") != state.payload.get("run_id")
            or terminal.payload.get("spec_hash") != state.payload.get("spec_hash")
            or terminal.payload.get("observation_hash") != obs.content_hash
            or terminal.payload.get("monitoring_policy_hash") != policy.content_hash
            or terminal.payload.get("stop_request_hash") != request.content_hash
        ):
            raise SoftStopError("soft-stop terminal lineage is invalid")
        journal = HarnessJournal(root, store, cls._proposal_id(cast(str, state.payload["run_id"])))
        transitions = journal.strict_transition_snapshot()
        if (
            len(transitions) != 5
            or plan_ref.content_hash != transitions[1].content_hash
            or decision_ref.content_hash != transitions[2].content_hash
            or terminal.payload.get("harness_decision_hash") != transitions[2].content_hash
            or terminal.payload.get("action_plan_hash") != transitions[1].content_hash
            or terminal.payload.get("action_observation_hash") != transitions[3].content_hash
            or terminal.payload.get("harness_outcome_hash") != transitions[4].content_hash
        ):
            raise SoftStopError("soft-stop Harness terminal lineage is invalid")
        outcome = transitions[4]
        audit_hash = outcome.payload.get("audit_event_hash")
        if type(audit_hash) is not str:
            raise SoftStopError("soft-stop AuditEvent lineage is invalid")
        audit = store.read(audit_hash, expected_schema_name="AuditEvent")
        if (
            audit.payload.get("action_observation_hash") != transitions[3].content_hash
            or audit.payload.get("decision_hash") != transitions[2].content_hash
            or audit.payload.get("plan_hash") != transitions[1].content_hash
            or audit.payload.get("run_id") != state.payload.get("run_id")
            or audit.payload.get("spec_hash") != state.payload.get("spec_hash")
            or terminal.payload.get("audit_event_hash") != audit.content_hash
        ):
            raise SoftStopError("soft-stop AuditEvent lineage is invalid")
        cluster = ClusterLifecycleWorkflow.resume(root, cast(str, state.payload["run_id"]))
        grace = (
            store.read(cast(str, state.payload["grace_hash"]), expected_schema_name="SoftStopGrace")
            if state.payload.get("grace_hash")
            else None
        )
        return SoftTwoPhaseStopSnapshot(
            terminal,
            obs,
            request,
            transitions[2],
            transitions[1],
            transitions[3],
            transitions[4],
            cluster,
            policy,
            grace,
            False,
        )


SoftStopWorkflow = SoftTwoPhaseStopWorkflow
SoftStopController = SoftTwoPhaseStopWorkflow
SoftStopMonitor = SoftTwoPhaseStopWorkflow
SoftStopConfig = SoftTwoPhaseStopConfig
SoftStopObservation = SoftTwoPhaseStopObservation
SoftFailureKind = SoftStopKind
SoftFailureStopConfig = SoftTwoPhaseStopConfig
SoftFailureStopWorkflow = SoftTwoPhaseStopWorkflow
SoftFailureStopObservation = SoftTwoPhaseStopObservation
SoftAnomalyStopConfig = SoftTwoPhaseStopConfig
SoftAnomalyStopWorkflow = SoftTwoPhaseStopWorkflow
