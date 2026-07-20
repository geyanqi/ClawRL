"""Checkpoint-first, durable exactly-once optimizer-step fixture.

The real trainer is deliberately not used here.  ``FixtureOptimizerAdapter``
is a small stateful adapter whose state is represented by immutable artifacts;
the controller's only commit point is a synchronous checkpoint followed by a
single ``StepApplied`` marker.  This makes all crash windows observable from a
new process and keeps the fixture useful as a contract seam for the real
trainer integration.
"""

from __future__ import annotations

import re
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

_HASH = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_PHASES = {"TRAIN_35B", "TRAIN_122B"}


class DurableStepAppliedError(RuntimeError):
    """A step could not be recovered without risking a duplicate update."""


class DurableStepAppliedTerminalStop(DurableStepAppliedError):
    """A typed, durable run-stop request was written."""


class InjectedDurableStepAppliedCrash(DurableStepAppliedError):
    """Synthetic controller crash at a checkpoint protocol boundary."""


# Names used by a few integrations and by the ticket wording.
StepAppliedError = DurableStepAppliedError
StepAppliedTerminalStop = DurableStepAppliedTerminalStop
InjectedStepAppliedCrash = InjectedDurableStepAppliedCrash


def _hash(value: object, field: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise DurableStepAppliedError(f"{field} is not a content hash")
    return cast(str, value)


def _id(value: object, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(cast(str, value)) is None:
        raise DurableStepAppliedError(f"{field} is invalid")
    return cast(str, value)


_FAULT_ALIASES = {
    "after_update": "crash_after_update",
    "checkpoint": "crash_after_checkpoint",
    "after_checkpoint": "crash_after_checkpoint",
    "missing_marker": "crash_after_checkpoint",
    "marker_missing": "crash_after_checkpoint",
    "async_checkpoint": "async_save",
    "durability_unconfirmed": "durability_unknown",
    "unstable_state_hash": "unstable_hash",
    "optimizer_state_unrecoverable": "optimizer_unrecoverable",
}
_FAULTS = {
    None,
    "crash_after_update",
    "crash_after_checkpoint",
    "async_save",
    "durability_unknown",
    "unstable_hash",
    "optimizer_unrecoverable",
    "marker_without_checkpoint",
    "corrupt_checkpoint",
}


@dataclass(frozen=True, slots=True)
class DurableStepAppliedConfig:
    run_id: str
    global_step: int
    experiment_spec_hash: str
    trainer_path: str = "classic"
    phase: str = "TRAIN_35B"
    fault_kind: str | None = None
    # ``crash_after`` is accepted for parity with other fixture controllers.
    crash_after: str | None = None
    backend: str = "sync"
    backend_kind: str = "sync"
    checkpoint_backend: str = "sync"
    durability_confirmed: bool = True
    stable_state_hash: bool = True
    optimizer_state_recoverable: bool = True

    def __post_init__(self) -> None:
        _id(self.run_id, "run_id")
        if type(self.global_step) is not int or self.global_step < 0:
            raise DurableStepAppliedError("global_step is invalid")
        _hash(self.experiment_spec_hash, "experiment_spec_hash")
        if self.trainer_path not in {"classic", "v1"}:
            raise DurableStepAppliedError("trainer_path is invalid")
        if self.phase not in _PHASES:
            raise DurableStepAppliedError("training phase is invalid")
        fault = self.fault_kind if self.fault_kind is not None else self.crash_after
        if fault is not None:
            fault = _FAULT_ALIASES.get(fault, fault)
        if fault not in _FAULTS:
            raise DurableStepAppliedError("unknown durable-step fault")
        if self.fault_kind is not None and self.crash_after is not None:
            if _FAULT_ALIASES.get(self.fault_kind, self.fault_kind) != fault:
                raise DurableStepAppliedError("fault_kind and crash_after conflict")
        if type(self.durability_confirmed) is not bool or type(self.stable_state_hash) is not bool:
            raise DurableStepAppliedError("durability controls must be booleans")
        if type(self.optimizer_state_recoverable) is not bool:
            raise DurableStepAppliedError("optimizer recovery control must be boolean")
        object.__setattr__(self, "fault_kind", fault)


DurableStepConfig = DurableStepAppliedConfig
StepAppliedConfig = DurableStepAppliedConfig


@dataclass(frozen=True, slots=True)
class DurableStepAppliedSnapshot:
    report: Artifact
    checkpoint: Artifact
    step_applied: Artifact
    model_state: Artifact
    optimizer_state: Artifact

    @property
    def marker(self) -> Artifact:
        return self.step_applied

    @property
    def checkpoint_manifest(self) -> Artifact:
        return self.checkpoint

    @property
    def step_applied_marker(self) -> Artifact:
        return self.step_applied


class FixtureOptimizerAdapter:
    """Deterministic, durable fake optimizer/model backend.

    ``update`` never mutates a mutable in-memory model.  It creates immutable
    model and optimizer state artifacts whose payload carries parent hashes and
    the exact reward-set identity.  A fresh process therefore sees the same
    state and can verify whether an update was already committed.
    """

    def __init__(self, root: str | Path, run_id: str, trainer_path: str = "classic") -> None:
        self.root = Path(root)
        self.store = ArtifactStore(root)
        self.run_id = _id(run_id, "run_id")
        self.trainer_path = trainer_path
        self.namespace = self.root / "fixture-optimizer" / self.run_id / trainer_path
        ArtifactStore.durable_mkdir(self.namespace)

    def _genesis(self, spec_hash: str) -> tuple[Artifact, Artifact]:
        model_ref = self.namespace / "genesis-model.ref"
        optim_ref = self.namespace / "genesis-optimizer.ref"
        if model_ref.exists() and optim_ref.exists():
            model = self.store.read(model_ref.read_text(encoding="ascii").strip(), expected_schema_name="ModelState")
            optimizer = self.store.read(
                optim_ref.read_text(encoding="ascii").strip(), expected_schema_name="OptimizerState"
            )
            if (
                model.payload.get("run_id") != self.run_id
                or model.payload.get("experiment_spec_hash") != spec_hash
                or model.payload.get("global_step") != -1
                or model.payload.get("status") != "genesis"
                or model.payload.get("parent_model_state_hash") is not None
                or model.payload.get("optimizer_state_hash") is not None
                or optimizer.payload.get("run_id") != self.run_id
                or optimizer.payload.get("experiment_spec_hash") != spec_hash
                or optimizer.payload.get("global_step") != -1
                or optimizer.payload.get("status") != "genesis"
                or optimizer.payload.get("parent_optimizer_state_hash") is not None
                or optimizer.payload.get("model_state_hash") != model.content_hash
                or optimizer.payload.get("recoverable") is not True
            ):
                raise DurableStepAppliedError("genesis state lineage changed")
            return (
                model,
                optimizer,
            )
        model = self.store.put(
            "ModelState",
            "1.0.0",
            {
                "global_step": -1,
                "model_parameters": [0, 0, 0],
                "optimizer_state_hash": None,
                "parent_model_state_hash": None,
                "reward_set_hash": None,
                "run_id": self.run_id,
                "status": "genesis",
                "experiment_spec_hash": spec_hash,
            },
        )
        optimizer = self.store.put(
            "OptimizerState",
            "1.0.0",
            {
                "global_step": -1,
                "optimizer_slots": [0, 0, 0],
                "parent_optimizer_state_hash": None,
                "model_state_hash": model.content_hash,
                "reward_set_hash": None,
                "run_id": self.run_id,
                "status": "genesis",
                "experiment_spec_hash": spec_hash,
                "recoverable": True,
            },
        )
        ArtifactStore._publish(model_ref, f"{model.content_hash}\n".encode("ascii"))
        ArtifactStore._publish(optim_ref, f"{optimizer.content_hash}\n".encode("ascii"))
        return model, optimizer

    @staticmethod
    def _delta(reward_set_hash: str, index: int) -> int:
        return int(reward_set_hash[index * 8 : index * 8 + 8], 16) % 100000

    def update(
        self,
        *,
        config: DurableStepAppliedConfig,
        reward_set_hash: str,
        previous_model: Artifact,
        previous_optimizer: Artifact,
        step_ready: Artifact,
    ) -> tuple[Artifact, Artifact]:
        del step_ready
        model_values = cast(list[int], previous_model.payload["model_parameters"])
        optimizer_values = cast(list[int], previous_optimizer.payload["optimizer_slots"])
        deltas = [self._delta(reward_set_hash, index) for index in range(3)]
        model = self.store.put(
            "ModelState",
            "1.0.0",
            {
                "experiment_spec_hash": config.experiment_spec_hash,
                "global_step": config.global_step,
                "model_parameters": [model_values[i] + deltas[i] for i in range(3)],
                "optimizer_state_hash": None,
                "parent_model_state_hash": previous_model.content_hash,
                "reward_set_hash": reward_set_hash,
                "run_id": config.run_id,
                "status": "updated",
            },
        )
        optimizer = self.store.put(
            "OptimizerState",
            "1.0.0",
            {
                "experiment_spec_hash": config.experiment_spec_hash,
                "global_step": config.global_step,
                "model_state_hash": model.content_hash,
                "optimizer_slots": [optimizer_values[i] + deltas[i] for i in range(3)],
                "parent_optimizer_state_hash": previous_optimizer.content_hash,
                "recoverable": True,
                "reward_set_hash": reward_set_hash,
                "run_id": config.run_id,
                "status": "updated",
            },
        )
        return model, optimizer

    def genesis(self, spec_hash: str) -> tuple[Artifact, Artifact]:
        return self._genesis(spec_hash)


@dataclass(frozen=True, slots=True)
class _Validated:
    checkpoint: Artifact
    model: Artifact
    optimizer: Artifact


class DurableStepAppliedWorkflow:
    """Apply one step-ready reward set with checkpoint-first exactly-once."""

    @staticmethod
    def _namespace(root: str | Path, config: DurableStepAppliedConfig) -> Path:
        return Path(root) / "durable-step-applied" / config.run_id / str(config.global_step) / config.trainer_path

    @staticmethod
    def _publish(ref: Path, artifact: Artifact) -> None:
        ArtifactStore.durable_mkdir(ref.parent)
        ArtifactStore._publish(ref, f"{artifact.content_hash}\n".encode("ascii"))

    @classmethod
    def _stop(cls, root: str | Path, config: DurableStepAppliedConfig, reason: str, evidence: str | None = None):
        store = ArtifactStore(root)
        ev = store.put(
            "DurableStepAppliedEvidence",
            "1.0.0",
            {
                "experiment_spec_hash": config.experiment_spec_hash,
                "global_step": config.global_step,
                "reason_code": reason,
                "run_id": config.run_id,
                "status": "terminal",
                "evidence_hash": evidence,
            },
        )
        req = store.put(
            "RunStopRequest",
            "1.0.0",
            {
                "evidence_hash": ev.content_hash,
                "experiment_spec_hash": config.experiment_spec_hash,
                "global_step": config.global_step,
                "optimizer_update": False,
                "reason_code": reason,
                "run_id": config.run_id,
                "status": "terminal",
            },
        )
        namespace = cls._namespace(root, config)
        cls._publish(namespace / "stop-request.ref", req)
        return DurableStepAppliedTerminalStop(f"{reason}:RUN_STOP_REQUEST:{req.content_hash}")

    @classmethod
    def _validate_step_ready(
        cls, config: DurableStepAppliedConfig, step_ready: Artifact, experiment_spec: Artifact
    ) -> str:
        if step_ready.schema_name != "StepReady" or step_ready.schema_version != "1.0.0":
            raise DurableStepAppliedError("StepReady identity is invalid")
        if (
            experiment_spec.schema_name != "ExperimentSpec"
            or experiment_spec.content_hash != config.experiment_spec_hash
        ):
            raise DurableStepAppliedError("ExperimentSpec identity is invalid")
        payload = step_ready.payload
        if (
            payload.get("run_id") != config.run_id
            or payload.get("global_step") != config.global_step
            or payload.get("experiment_spec_hash") != experiment_spec.content_hash
            or payload.get("trainer_path") != config.trainer_path
            or payload.get("status") != "step_ready"
            or payload.get("optimizer_update") is not False
        ):
            raise DurableStepAppliedError("StepReady lineage is invalid")
        values = payload.get("reward_hashes")
        if (
            not isinstance(values, list)
            or not values
            or any(type(item) is not str or _HASH.fullmatch(cast(str, item)) is None for item in values)
            or payload.get("trainer_path") != config.trainer_path
            or payload.get("expected_slot_count") != len(values)
        ):
            raise DurableStepAppliedError("StepReady reward set is incomplete")
        reward_hashes = cast(list[str], values)
        reward_root = payload.get("reward_root_hash")
        if type(reward_root) is not str or reward_root != sha256_hex(canonical_json_bytes(reward_hashes)):
            raise DurableStepAppliedError("StepReady reward root is invalid")
        return cast(str, reward_root)

    @classmethod
    def _validate_states(
        cls,
        config: DurableStepAppliedConfig,
        checkpoint: Artifact,
        store: ArtifactStore,
        reward_root: str,
    ) -> _Validated:
        if checkpoint.schema_name != "DurableCheckpointManifest" or checkpoint.schema_version != "1.0.0":
            raise DurableStepAppliedError("checkpoint schema is invalid")
        p = checkpoint.payload
        fields = {
            "checkpoint_id",
            "durability",
            "experiment_spec_hash",
            "global_step",
            "model_state_hash",
            "optimizer_state_hash",
            "previous_checkpoint_hash",
            "reward_set_hash",
            "run_id",
            "status",
        }
        if set(p) != fields or p.get("status") != "committed" or p.get("durability") != "synchronous":
            raise DurableStepAppliedError("checkpoint manifest is invalid")
        if (
            p.get("run_id") != config.run_id
            or p.get("global_step") != config.global_step
            or p.get("experiment_spec_hash") != config.experiment_spec_hash
            or p.get("reward_set_hash") != reward_root
            or p.get("checkpoint_id") != f"{config.run_id}:{config.global_step}:{config.trainer_path}"
        ):
            raise DurableStepAppliedError("checkpoint lineage is invalid")
        previous_hash = p.get("previous_checkpoint_hash")
        if config.global_step == 0:
            if previous_hash is not None:
                raise DurableStepAppliedError("genesis checkpoint has a previous checkpoint")
        else:
            _hash(previous_hash, "previous_checkpoint_hash")
            previous_ref = (
                store.root
                / "durable-step-applied"
                / config.run_id
                / str(config.global_step - 1)
                / config.trainer_path
                / "checkpoint.ref"
            )
            if not previous_ref.exists() or previous_ref.read_text(encoding="ascii").strip() != previous_hash:
                raise DurableStepAppliedError("previous checkpoint reference lineage is invalid")
            previous = store.read(cast(str, previous_hash), expected_schema_name="DurableCheckpointManifest")
            if (
                previous.payload.get("global_step") != config.global_step - 1
                or previous.payload.get("run_id") != config.run_id
                or previous.payload.get("experiment_spec_hash") != config.experiment_spec_hash
                or previous.payload.get("status") != "committed"
                or previous.payload.get("durability") != "synchronous"
                or previous.payload.get("checkpoint_id")
                != f"{config.run_id}:{config.global_step - 1}:{config.trainer_path}"
            ):
                raise DurableStepAppliedError("previous checkpoint lineage is invalid")
            previous_config = config.__class__(
                config.run_id,
                config.global_step - 1,
                config.experiment_spec_hash,
                config.trainer_path,
                config.phase,
            )
            cls._validate_states(
                previous_config,
                previous,
                store,
                cast(str, previous.payload.get("reward_set_hash")),
            )
            previous_namespace = cls._namespace(store.root, previous_config)
            previous_marker = cls._read_ref(store, previous_namespace / "step-applied.ref", "StepApplied")
            if previous_marker is None:
                raise DurableStepAppliedError("previous StepApplied marker is missing")
            cls._validate_marker(
                previous_marker,
                previous_config,
                previous,
                cast(str, previous.payload.get("reward_set_hash")),
            )
        model = store.read(cast(str, p["model_state_hash"]), expected_schema_name="ModelState")
        optimizer = store.read(cast(str, p["optimizer_state_hash"]), expected_schema_name="OptimizerState")
        if (
            model.payload.get("run_id") != config.run_id
            or model.payload.get("global_step") != config.global_step
            or model.payload.get("experiment_spec_hash") != config.experiment_spec_hash
            or model.payload.get("reward_set_hash") != reward_root
            or model.payload.get("optimizer_state_hash") is not None
            or optimizer.payload.get("run_id") != config.run_id
            or optimizer.payload.get("global_step") != config.global_step
            or optimizer.payload.get("experiment_spec_hash") != config.experiment_spec_hash
            or optimizer.payload.get("reward_set_hash") != reward_root
            or optimizer.payload.get("model_state_hash") != model.content_hash
            or optimizer.payload.get("recoverable") is not True
            or model.payload.get("status") != "updated"
            or optimizer.payload.get("status") != "updated"
        ):
            raise DurableStepAppliedError("model/optimizer state lineage is invalid")
        if config.global_step == 0:
            genesis_root = store.root / "fixture-optimizer" / config.run_id / config.trainer_path
            try:
                genesis_model = store.read(
                    (genesis_root / "genesis-model.ref").read_text(encoding="ascii").strip(),
                    expected_schema_name="ModelState",
                )
                genesis_optimizer = store.read(
                    (genesis_root / "genesis-optimizer.ref").read_text(encoding="ascii").strip(),
                    expected_schema_name="OptimizerState",
                )
            except (OSError, ArtifactCorruption, UnknownSchemaMajor) as error:
                raise DurableStepAppliedError("genesis state is missing") from error
            if (
                genesis_model.payload.get("run_id") != config.run_id
                or genesis_model.payload.get("experiment_spec_hash") != config.experiment_spec_hash
                or genesis_model.payload.get("global_step") != -1
                or genesis_model.payload.get("status") != "genesis"
                or genesis_model.payload.get("parent_model_state_hash") is not None
                or genesis_model.payload.get("optimizer_state_hash") is not None
                or genesis_optimizer.payload.get("run_id") != config.run_id
                or genesis_optimizer.payload.get("experiment_spec_hash") != config.experiment_spec_hash
                or genesis_optimizer.payload.get("global_step") != -1
                or genesis_optimizer.payload.get("status") != "genesis"
                or genesis_optimizer.payload.get("parent_optimizer_state_hash") is not None
                or genesis_optimizer.payload.get("model_state_hash") != genesis_model.content_hash
                or genesis_optimizer.payload.get("recoverable") is not True
                or model.payload.get("parent_model_state_hash") != genesis_model.content_hash
                or optimizer.payload.get("parent_optimizer_state_hash") != genesis_optimizer.content_hash
            ):
                raise DurableStepAppliedError("initial optimizer parent lineage is invalid")
        if config.global_step > 0:
            previous = store.read(
                cast(str, p["previous_checkpoint_hash"]), expected_schema_name="DurableCheckpointManifest"
            )
            if (
                set(previous.payload) != fields
                or previous.payload.get("status") != "committed"
                or previous.payload.get("durability") != "synchronous"
                or previous.payload.get("checkpoint_id")
                != f"{config.run_id}:{config.global_step - 1}:{config.trainer_path}"
                or previous.payload.get("global_step") != config.global_step - 1
                or previous.payload.get("run_id") != config.run_id
                or previous.payload.get("experiment_spec_hash") != config.experiment_spec_hash
            ):
                raise DurableStepAppliedError("previous checkpoint manifest is invalid")
            previous_namespace = (
                store.root / "durable-step-applied" / config.run_id / str(config.global_step - 1) / config.trainer_path
            )
            previous_marker_ref = previous_namespace / "step-applied.ref"
            if not previous_marker_ref.exists():
                raise DurableStepAppliedError("previous StepApplied marker is missing")
            try:
                previous_marker = store.read(
                    previous_marker_ref.read_text(encoding="ascii").strip(), expected_schema_name="StepApplied"
                )
            except (OSError, ArtifactCorruption, UnknownSchemaMajor) as error:
                raise DurableStepAppliedError("previous StepApplied marker is invalid") from error
            previous_config = DurableStepAppliedConfig(
                config.run_id,
                config.global_step - 1,
                config.experiment_spec_hash,
                trainer_path=config.trainer_path,
                phase=config.phase,
            )
            previous_reward_root = previous_marker.payload.get("reward_set_hash")
            if type(previous_reward_root) is not str:
                raise DurableStepAppliedError("previous StepApplied reward lineage is invalid")
            cls._validate_marker(previous_marker, previous_config, previous, previous_reward_root)
            previous_model_hash = cast(str, previous.payload.get("model_state_hash"))
            previous_optimizer_hash = cast(str, previous.payload.get("optimizer_state_hash"))
            if (
                model.payload.get("parent_model_state_hash") != previous_model_hash
                or optimizer.payload.get("parent_optimizer_state_hash") != previous_optimizer_hash
            ):
                raise DurableStepAppliedError("model/optimizer parent lineage is invalid")
        return _Validated(checkpoint, model, optimizer)

    @classmethod
    def _validate_marker(
        cls, marker: Artifact, config: DurableStepAppliedConfig, checkpoint: Artifact, reward_root: str
    ) -> None:
        if marker.schema_name != "StepApplied" or marker.schema_version != "1.0.0":
            raise DurableStepAppliedError("StepApplied schema is invalid")
        fields = {
            "checkpoint_hash",
            "experiment_spec_hash",
            "global_step",
            "model_state_hash",
            "optimizer_state_hash",
            "reward_set_hash",
            "run_id",
            "status",
        }
        p = marker.payload
        if set(p) != fields or p.get("status") != "applied":
            raise DurableStepAppliedError("StepApplied marker is invalid")
        if (
            p.get("checkpoint_hash") != checkpoint.content_hash
            or p.get("run_id") != config.run_id
            or p.get("global_step") != config.global_step
            or p.get("experiment_spec_hash") != config.experiment_spec_hash
            or p.get("reward_set_hash") != reward_root
            or p.get("model_state_hash") != checkpoint.payload.get("model_state_hash")
            or p.get("optimizer_state_hash") != checkpoint.payload.get("optimizer_state_hash")
        ):
            raise DurableStepAppliedError("StepApplied/checkpoint mismatch")

    @classmethod
    def _read_ref(cls, store: ArtifactStore, ref: Path, schema: str) -> Artifact | None:
        if not ref.exists():
            return None
        try:
            return store.read(ref.read_text(encoding="ascii").strip(), expected_schema_name=schema)
        except (OSError, UnicodeError, ValueError, ArtifactCorruption, UnknownSchemaMajor) as error:
            raise DurableStepAppliedError("durable reference cannot be read") from error

    @classmethod
    def _previous_states(
        cls, root: str | Path, config: DurableStepAppliedConfig, spec_hash: str
    ) -> tuple[Artifact, Artifact, str | None]:
        adapter = FixtureOptimizerAdapter(root, config.run_id, config.trainer_path)
        if config.global_step == 0:
            model, optimizer = adapter.genesis(spec_hash)
            return model, optimizer, None
        previous = config.__class__(config.run_id, config.global_step - 1, spec_hash, config.trainer_path, config.phase)
        namespace = cls._namespace(root, previous)
        store = ArtifactStore(root)
        checkpoint = cls._read_ref(store, namespace / "checkpoint.ref", "DurableCheckpointManifest")
        marker = cls._read_ref(store, namespace / "step-applied.ref", "StepApplied")
        if checkpoint is None or marker is None:
            raise DurableStepAppliedError("previous StepApplied checkpoint is missing")
        reward_root = cast(str, marker.payload.get("reward_set_hash"))
        validated = cls._validate_states(previous, checkpoint, store, reward_root)
        cls._validate_marker(marker, previous, checkpoint, reward_root)
        return validated.model, validated.optimizer, checkpoint.content_hash

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: DurableStepAppliedConfig,
        step_ready: Artifact,
        experiment_spec: Artifact,
        optimizer_adapter: FixtureOptimizerAdapter | None = None,
        adapter: FixtureOptimizerAdapter | None = None,
    ) -> DurableStepAppliedSnapshot:
        store = ArtifactStore(root)
        try:
            reward_root = cls._validate_step_ready(config, step_ready, experiment_spec)
        except (DurableStepAppliedError, ArtifactCorruption) as error:
            raise DurableStepAppliedError("step-ready input cannot be consumed") from error
        namespace = cls._namespace(root, config)
        checkpoint_ref, marker_ref = namespace / "checkpoint.ref", namespace / "step-applied.ref"
        try:
            marker = cls._read_ref(store, marker_ref, "StepApplied")
            checkpoint = cls._read_ref(store, checkpoint_ref, "DurableCheckpointManifest")
            if marker is not None and checkpoint is None:
                raise cls._stop(root, config, "STEP_APPLIED_CHECKPOINT_MISSING")
            if marker is not None and checkpoint is not None:
                validated = cls._validate_states(config, checkpoint, store, reward_root)
                cls._validate_marker(marker, config, checkpoint, reward_root)
                return cls._report(root, config, step_ready, marker, validated)
            if checkpoint is not None:
                validated = cls._validate_states(config, checkpoint, store, reward_root)
                if config.fault_kind == "crash_after_checkpoint" and not (namespace / "checkpoint-crash.ref").exists():
                    crash = store.put(
                        "StepAppliedCrashEvent",
                        "1.0.0",
                        {
                            "checkpoint_hash": checkpoint.content_hash,
                            "run_id": config.run_id,
                            "global_step": config.global_step,
                            "status": "injected",
                        },
                    )
                    cls._publish(namespace / "checkpoint-crash.ref", crash)
                    raise InjectedDurableStepAppliedCrash("crash after durable checkpoint")
                marker = cls._marker(store, config, checkpoint, reward_root)
                cls._publish(marker_ref, marker)
                return cls._report(root, config, step_ready, marker, validated)
            unsupported = (
                config.backend_kind != "sync"
                or config.backend != "sync"
                or config.checkpoint_backend != "sync"
                or not config.durability_confirmed
                or not config.stable_state_hash
                or not config.optimizer_state_recoverable
                or config.fault_kind in {"async_save", "durability_unknown", "unstable_hash", "optimizer_unrecoverable"}
            )
            if unsupported:
                raise cls._stop(root, config, "TRAIN_CHECKPOINT_BACKEND_UNREADY")
            previous_model, previous_optimizer, previous_checkpoint_hash = cls._previous_states(
                root, config, experiment_spec.content_hash
            )
            opt = optimizer_adapter or adapter or FixtureOptimizerAdapter(root, config.run_id, config.trainer_path)
            model, optimizer = opt.update(
                config=config,
                reward_set_hash=reward_root,
                previous_model=previous_model,
                previous_optimizer=previous_optimizer,
                step_ready=step_ready,
            )
            if config.fault_kind == "crash_after_update" and not (namespace / "update-crash.ref").exists():
                # The state is deliberately left unreferenced: without a
                # committed checkpoint the next process must replay from the
                # previous checkpoint.
                event = store.put(
                    "StepAppliedCrashEvent",
                    "1.0.0",
                    {
                        "global_step": config.global_step,
                        "model_state_hash": model.content_hash,
                        "optimizer_state_hash": optimizer.content_hash,
                        "run_id": config.run_id,
                        "status": "updated_not_checkpointed",
                    },
                )
                cls._publish(namespace / "update-crash.ref", event)
                raise InjectedDurableStepAppliedCrash("crash after optimizer update before checkpoint")
            checkpoint = store.put(
                "DurableCheckpointManifest",
                "1.0.0",
                {
                    "checkpoint_id": f"{config.run_id}:{config.global_step}:{config.trainer_path}",
                    "durability": "synchronous",
                    "experiment_spec_hash": config.experiment_spec_hash,
                    "global_step": config.global_step,
                    "model_state_hash": model.content_hash,
                    "optimizer_state_hash": optimizer.content_hash,
                    "previous_checkpoint_hash": previous_checkpoint_hash,
                    "reward_set_hash": reward_root,
                    "run_id": config.run_id,
                    "status": "committed",
                },
            )
            cls._publish(checkpoint_ref, checkpoint)
            if config.fault_kind == "crash_after_checkpoint" and not (namespace / "checkpoint-crash.ref").exists():
                crash = store.put(
                    "StepAppliedCrashEvent",
                    "1.0.0",
                    {
                        "checkpoint_hash": checkpoint.content_hash,
                        "run_id": config.run_id,
                        "global_step": config.global_step,
                        "status": "injected",
                    },
                )
                cls._publish(namespace / "checkpoint-crash.ref", crash)
                raise InjectedDurableStepAppliedCrash("crash after durable checkpoint")
            marker = cls._marker(store, config, checkpoint, reward_root)
            cls._publish(marker_ref, marker)
            validated = cls._validate_states(config, checkpoint, store, reward_root)
            return cls._report(root, config, step_ready, marker, validated)
        except DurableStepAppliedTerminalStop:
            raise
        except InjectedDurableStepAppliedCrash:
            raise
        except (
            DurableStepAppliedError,
            ArtifactCorruption,
            UnknownSchemaMajor,
            OSError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            if isinstance(error, DurableStepAppliedTerminalStop):
                raise
            raise cls._stop(root, config, "DURABLE_STEP_CORRUPTION") from error

    @classmethod
    def _marker(
        cls, store: ArtifactStore, config: DurableStepAppliedConfig, checkpoint: Artifact, reward_root: str
    ) -> Artifact:
        return store.put(
            "StepApplied",
            "1.0.0",
            {
                "checkpoint_hash": checkpoint.content_hash,
                "experiment_spec_hash": config.experiment_spec_hash,
                "global_step": config.global_step,
                "model_state_hash": checkpoint.payload["model_state_hash"],
                "optimizer_state_hash": checkpoint.payload["optimizer_state_hash"],
                "reward_set_hash": reward_root,
                "run_id": config.run_id,
                "status": "applied",
            },
        )

    @classmethod
    def _report(
        cls,
        root: str | Path,
        config: DurableStepAppliedConfig,
        step_ready: Artifact | None,
        marker: Artifact,
        validated: _Validated,
        *,
        reward_root: str | None = None,
    ) -> DurableStepAppliedSnapshot:
        store = ArtifactStore(root)
        effective_reward_root = reward_root
        if effective_reward_root is None and step_ready is not None:
            effective_reward_root = cast(str, step_ready.payload["reward_root_hash"])
        if effective_reward_root is None:
            raise DurableStepAppliedError("reward-set hash is missing from durable report")
        report = store.put(
            "DurableStepAppliedReport",
            "1.0.0",
            {
                "checkpoint_hash": validated.checkpoint.content_hash,
                "experiment_spec_hash": config.experiment_spec_hash,
                "global_step": config.global_step,
                "marker_hash": marker.content_hash,
                "model_state_hash": validated.model.content_hash,
                "optimizer_state_hash": validated.optimizer.content_hash,
                "reward_set_hash": effective_reward_root,
                "run_id": config.run_id,
                "status": "step_applied",
                "update_count": 1,
            },
        )
        namespace = cls._namespace(root, config)
        cls._publish(namespace / "report.ref", report)
        return DurableStepAppliedSnapshot(report, validated.checkpoint, marker, validated.model, validated.optimizer)

    @classmethod
    def resume(cls, root: str | Path, config: DurableStepAppliedConfig) -> DurableStepAppliedSnapshot:
        store = ArtifactStore(root)
        namespace = cls._namespace(root, config)
        try:
            report = cls._read_ref(store, namespace / "report.ref", "DurableStepAppliedReport")
            checkpoint = cls._read_ref(store, namespace / "checkpoint.ref", "DurableCheckpointManifest")
            marker = cls._read_ref(store, namespace / "step-applied.ref", "StepApplied")
        except DurableStepAppliedError as error:
            raise cls._stop(root, config, "DURABLE_STEP_CORRUPTION") from error
        if marker is not None and checkpoint is None:
            raise cls._stop(root, config, "STEP_APPLIED_CHECKPOINT_MISSING")
        if checkpoint is None:
            raise cls._stop(root, config, "DURABLE_STEP_CORRUPTION")
        # A controller may have crashed after the synchronous checkpoint
        # commit and before publishing the marker (or report).  Verify the
        # checkpoint and backfill only the marker; never invoke the optimizer.
        if marker is None:
            reward_root = cast(str, checkpoint.payload.get("reward_set_hash"))
            try:
                validated = cls._validate_states(config, checkpoint, store, reward_root)
                marker = cls._marker(store, config, checkpoint, reward_root)
                cls._publish(namespace / "step-applied.ref", marker)
                if report is None:
                    return cls._report(
                        root,
                        config,
                        None,
                        marker,
                        validated,
                        reward_root=reward_root,
                    )
            except (
                DurableStepAppliedError,
                ArtifactCorruption,
                UnknownSchemaMajor,
                OSError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                raise cls._stop(root, config, "DURABLE_STEP_CORRUPTION") from error
        if report is None or marker is None:
            raise cls._stop(root, config, "DURABLE_STEP_CORRUPTION")
        reward_root = cast(str, marker.payload.get("reward_set_hash"))
        try:
            validated = cls._validate_states(config, checkpoint, store, reward_root)
            cls._validate_marker(marker, config, checkpoint, reward_root)
            expected = {
                "checkpoint_hash",
                "experiment_spec_hash",
                "global_step",
                "marker_hash",
                "model_state_hash",
                "optimizer_state_hash",
                "reward_set_hash",
                "run_id",
                "status",
                "update_count",
            }
            if (
                report.schema_version != "1.0.0"
                or set(report.payload) != expected
                or report.payload.get("status") != "step_applied"
                or report.payload.get("update_count") != 1
                or report.payload.get("checkpoint_hash") != checkpoint.content_hash
                or report.payload.get("marker_hash") != marker.content_hash
                or report.payload.get("run_id") != config.run_id
                or report.payload.get("global_step") != config.global_step
                or report.payload.get("experiment_spec_hash") != config.experiment_spec_hash
                or report.payload.get("reward_set_hash") != reward_root
                or report.payload.get("model_state_hash") != validated.model.content_hash
                or report.payload.get("optimizer_state_hash") != validated.optimizer.content_hash
            ):
                raise DurableStepAppliedError("StepApplied report is invalid")
            return DurableStepAppliedSnapshot(report, checkpoint, marker, validated.model, validated.optimizer)
        except (
            DurableStepAppliedError,
            ArtifactCorruption,
            UnknownSchemaMajor,
            OSError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise cls._stop(root, config, "DURABLE_STEP_CORRUPTION") from error

    @classmethod
    def production_readiness(cls, root: str | Path, *, phase: str = "TRAIN_35B") -> Artifact:
        if phase not in _PHASES:
            raise DurableStepAppliedError("training phase is invalid")
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": "ASYNC_CHECKPOINT_SAVE_UNSUPPORTED", "status": "blocked"},
                    {"code": "DURABILITY_CONFIRMATION_UNAVAILABLE", "status": "blocked"},
                    {"code": "STABLE_STATE_HASH_UNAVAILABLE", "status": "blocked"},
                    {"code": "OPTIMIZER_STATE_RECOVERY_UNSUPPORTED", "status": "blocked"},
                ],
                "execution_profile": "production",
                "phase": phase,
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )


StepAppliedWorkflow = DurableStepAppliedWorkflow
CheckpointFirstWorkflow = DurableStepAppliedWorkflow
