"""Persistent verl v1 reward identity and fixture TransferQueue migration."""

from __future__ import annotations

import fcntl
import re
from collections.abc import Mapping
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
from clawrl.training.classic_identity import (
    ClassicIdentityConfig,
    ClassicIdentityError,
    ClassicRewardManager,
    ClassicSourceRow,
    ClassicTrajectoryIdentity,
    ClassicTrajectoryRow,
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_VARIANT = "v1/colocated-async-transfer-queue"
_PHASES = {"TRAIN_35B", "TRAIN_122B"}
_FAULTS = {"missing_identity", "ordinal_mismatch", "global_step_mismatch"}


class V1RewardIdentityError(RuntimeError):
    """The v1 reward or TransferQueue identity failed closed."""


class InjectedV1ControllerCrash(V1RewardIdentityError):
    """Synthetic controller crash after a durable queue checkpoint."""


def _safe_id(value: object, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(cast(str, value)) is None:
        raise V1RewardIdentityError(f"{field} is invalid")
    return cast(str, value)


def _integer(value: object, field: str, *, minimum: int = 0, maximum: int = 9_007_199_254_740_991) -> int:
    if type(value) is not int or not minimum <= cast(int, value) <= maximum:
        raise V1RewardIdentityError(f"{field} is invalid")
    return cast(int, value)


@dataclass(frozen=True, slots=True)
class V1IdentityConfig:
    run_id: str
    global_step: int
    rollout_count: int = 2
    initial_return_ordinals: tuple[int, ...] = (0,)
    variant: str = _VARIANT
    fault_slot_ordinal: int | None = None
    fault_kind: str | None = None

    def __post_init__(self) -> None:
        _safe_id(self.run_id, "run_id")
        _integer(self.global_step, "global_step")
        _integer(self.rollout_count, "rollout_count", minimum=1, maximum=128)
        if self.variant != _VARIANT:
            raise V1RewardIdentityError("fixture v1 variant is unsupported")
        if not self.initial_return_ordinals or any(
            type(item) is not int or item < 0 for item in self.initial_return_ordinals
        ):
            raise V1RewardIdentityError("initial return ordinals are invalid")
        if len(self.initial_return_ordinals) != len(set(self.initial_return_ordinals)):
            raise V1RewardIdentityError("initial return ordinals are duplicated")
        if (self.fault_slot_ordinal is None) != (self.fault_kind is None):
            raise V1RewardIdentityError("fault slot and kind must be configured together")
        if self.fault_slot_ordinal is not None:
            _integer(self.fault_slot_ordinal, "fault_slot_ordinal")
        if self.fault_kind is not None and self.fault_kind not in _FAULTS:
            raise V1RewardIdentityError("v1 fixture fault kind is invalid")

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "fault_kind": self.fault_kind,
            "fault_slot_ordinal": self.fault_slot_ordinal,
            "global_step": self.global_step,
            "initial_return_ordinals": list(self.initial_return_ordinals),
            "rollout_count": self.rollout_count,
            "run_id": self.run_id,
            "variant": self.variant,
        }


@dataclass(frozen=True, slots=True)
class ProductionV1IdentityConfig:
    phase: str
    variant: str

    def __post_init__(self) -> None:
        if self.phase not in _PHASES or type(self.variant) is not str or not self.variant:
            raise V1RewardIdentityError("production v1 identity config is invalid")


@dataclass(frozen=True, slots=True)
class V1RewardEnvelope:
    slot_ordinal: int
    identity: ClassicTrajectoryIdentity
    raw_prompt: str
    global_step: int

    def __post_init__(self) -> None:
        _integer(self.slot_ordinal, "slot_ordinal")
        if type(self.raw_prompt) is not str or not self.raw_prompt:
            raise V1RewardIdentityError("v1 raw prompt is invalid")
        if self.global_step != self.identity.global_step:
            raise V1RewardIdentityError("v1 batch global_step does not match sample identity")

    @classmethod
    def from_mapping(cls, value: object) -> V1RewardEnvelope:
        if not isinstance(value, Mapping) or set(value) != {"global_step", "identity", "raw_prompt", "slot_ordinal"}:
            raise V1RewardIdentityError("v1 reward envelope fields are invalid")
        raw_prompt = value.get("raw_prompt")
        if type(raw_prompt) is not str:
            raise V1RewardIdentityError("v1 raw prompt is invalid")
        try:
            identity = ClassicTrajectoryIdentity.from_mapping(value.get("identity"))
        except ClassicIdentityError as error:
            raise V1RewardIdentityError("v1 reward envelope identity is invalid") from error
        return cls(
            _integer(value.get("slot_ordinal"), "slot_ordinal"),
            identity,
            raw_prompt,
            _integer(value.get("global_step"), "global_step"),
        )

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "global_step": self.global_step,
            "identity": self.identity.artifact_payload(),
            "raw_prompt": self.raw_prompt,
            "slot_ordinal": self.slot_ordinal,
        }

    @property
    def content_hash(self) -> str:
        return sha256_hex(
            canonical_json_bytes({"domain": "v1-reward-envelope/1.0.0", "value": self.artifact_payload()})
        )


@dataclass(frozen=True, slots=True)
class TransferQueueSlot:
    original_slot_ordinal: int
    envelope: V1RewardEnvelope

    def __post_init__(self) -> None:
        if self.original_slot_ordinal != self.envelope.slot_ordinal:
            raise V1RewardIdentityError("TransferQueue original ordinal changed")

    @classmethod
    def from_envelope(cls, envelope: V1RewardEnvelope) -> TransferQueueSlot:
        return cls(envelope.slot_ordinal, envelope)

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "envelope": self.envelope.artifact_payload(),
            "envelope_hash": self.envelope.content_hash,
            "original_slot_ordinal": self.original_slot_ordinal,
        }


class V1RewardManager:
    """Compatibility seam that reuses the classic wide identity contract."""

    @staticmethod
    def prefactor(config: V1IdentityConfig, sources: tuple[ClassicSourceRow, ...]) -> tuple[V1RewardEnvelope, ...]:
        try:
            classic = ClassicRewardManager.prefactor(
                ClassicIdentityConfig(
                    config.run_id,
                    config.global_step,
                    rollout_count=config.rollout_count,
                    chunk_size=max(1, config.rollout_count),
                ),
                sources,
            )
        except ClassicIdentityError as error:
            raise V1RewardIdentityError("classic identity prefactor rejected v1 input") from error
        return tuple(
            V1RewardEnvelope(ordinal, row.identity, row.prompt, classic.global_steps[ordinal])
            for ordinal, row in enumerate(classic.rows)
        )

    @staticmethod
    def authorize_expected_trajectory_set(root: str | Path, expected_set: Artifact) -> Artifact:
        from clawrl.training.expected_trajectory_set import (
            ExpectedTrajectorySetError,
            ExpectedTrajectorySetWorkflow,
        )

        try:
            return ExpectedTrajectorySetWorkflow.authorize_scoring(
                root,
                expected_set=expected_set,
                transport="v1",
            )
        except ExpectedTrajectorySetError as error:
            raise V1RewardIdentityError("v1 ExpectedTrajectorySet validation failed closed") from error


class PersistentFixtureTransferQueue:
    """Stateful deterministic queue keyed only by persisted original ordinals."""

    def __init__(self, root: str | Path, run_id: str) -> None:
        self.root = Path(root)
        self.run_id = _safe_id(run_id, "run_id")
        self.store = ArtifactStore(root)
        self.namespace = self.root / "fixture-transfer-queue" / self.run_id
        ArtifactStore.durable_mkdir(self.namespace)

    def dispatch(self, slot: TransferQueueSlot) -> Artifact:
        envelope = slot.envelope
        if envelope.identity.run_id != self.run_id:
            raise V1RewardIdentityError("TransferQueue run identity changed")
        artifact = self.store.put(
            "V1TransferDispatch",
            "1.0.0",
            {
                **slot.artifact_payload(),
                "global_step": envelope.global_step,
                "run_id": self.run_id,
                "status": "dispatched",
            },
        )
        return self._publish_slot_ref(slot.original_slot_ordinal, "dispatch.ref", artifact, "TRANSFER_SLOT_CONFLICT")

    def commit_return(self, dispatch: Artifact, response: str, *, fault_kind: str | None = None) -> Artifact:
        envelope = self._validate_dispatch(dispatch)
        persisted_dispatch = self._read_slot_ref(envelope.slot_ordinal, "dispatch.ref", "V1TransferDispatch")
        if persisted_dispatch.content_hash != dispatch.content_hash:
            raise V1RewardIdentityError("TransferQueue dispatch ref changed before return")
        if type(response) is not str or not response:
            raise V1RewardIdentityError("TransferQueue response is invalid")
        row = ClassicTrajectoryRow(envelope.identity, envelope.raw_prompt, response)
        payload: dict[str, JsonValue] = {
            "dispatch_hash": dispatch.content_hash,
            "global_step": envelope.global_step,
            "identity": envelope.identity.artifact_payload(),
            "identity_hash": envelope.identity.content_hash,
            "raw_prompt": envelope.raw_prompt,
            "response": response,
            "run_id": self.run_id,
            "slot_ordinal": envelope.slot_ordinal,
            "status": "returned",
            "trajectory": row.artifact_payload(),
            "trajectory_hash": row.content_hash,
        }
        if fault_kind == "missing_identity":
            del payload["identity"]
        elif fault_kind == "ordinal_mismatch":
            payload["slot_ordinal"] = envelope.slot_ordinal + 1
        elif fault_kind == "global_step_mismatch":
            payload["global_step"] = envelope.global_step + 1
        elif fault_kind is not None:
            raise V1RewardIdentityError("unknown TransferQueue fault")
        artifact = self.store.put("V1TransferReturn", "1.0.0", payload)
        return self._publish_slot_ref(envelope.slot_ordinal, "return.ref", artifact, "TRANSFER_RETURN_CONFLICT")

    def checkpoint(self, dispatches: tuple[Artifact, ...], returns: tuple[Artifact, ...]) -> Artifact:
        by_ordinal = {self._validate_dispatch(item).slot_ordinal: item for item in dispatches}
        if sorted(by_ordinal) != list(range(len(dispatches))) or len(by_ordinal) != len(dispatches):
            raise V1RewardIdentityError("TransferQueue dispatch set is not a total ordinal set")
        returns_by_ordinal: dict[int, Artifact] = {}
        for item in returns:
            envelope, _ = self.validate_return(item, by_ordinal)
            if envelope.slot_ordinal in returns_by_ordinal:
                raise V1RewardIdentityError("TransferQueue checkpoint contains a duplicate return")
            returns_by_ordinal[envelope.slot_ordinal] = item
        returned = sorted(returns_by_ordinal)
        outstanding = sorted(set(by_ordinal) - set(returned))
        checkpoint = self.store.put(
            "V1TransferQueueCheckpoint",
            "1.0.0",
            {
                "dispatch_hashes_by_slot": [
                    {"dispatch_hash": by_ordinal[item].content_hash, "slot_ordinal": item}
                    for item in sorted(by_ordinal)
                ],
                "outstanding_slot_ordinals": outstanding,
                "return_hashes_by_slot": [
                    {"return_hash": returns_by_ordinal[item].content_hash, "slot_ordinal": item} for item in returned
                ],
                "returned_slot_ordinals": returned,
                "run_id": self.run_id,
                "status": "durable",
            },
        )
        return self._publish_ref(self.namespace / "checkpoint.ref", checkpoint, "TRANSFER_CHECKPOINT_CONFLICT")

    def reissue(self, dispatch: Artifact, checkpoint: Artifact) -> Artifact:
        envelope = self._validate_dispatch(dispatch)
        persisted_dispatch = self._read_slot_ref(envelope.slot_ordinal, "dispatch.ref", "V1TransferDispatch")
        if persisted_dispatch.content_hash != dispatch.content_hash:
            raise V1RewardIdentityError("TransferQueue dispatch ref changed before reissue")
        self.validate_checkpoint(checkpoint)
        outstanding = checkpoint.payload.get("outstanding_slot_ordinals")
        dispatch_entries = cast(list[dict[str, JsonValue]], checkpoint.payload["dispatch_hashes_by_slot"])
        checkpoint_dispatch = next(
            (item for item in dispatch_entries if item["slot_ordinal"] == envelope.slot_ordinal),
            None,
        )
        if (
            not isinstance(outstanding, list)
            or envelope.slot_ordinal not in outstanding
            or checkpoint_dispatch is None
            or checkpoint_dispatch["dispatch_hash"] != dispatch.content_hash
        ):
            raise V1RewardIdentityError("TransferQueue attempted to reissue a resolved slot")
        artifact = self.store.put(
            "V1TransferReissue",
            "1.0.0",
            {
                "checkpoint_hash": checkpoint.content_hash,
                "dispatch_hash": dispatch.content_hash,
                "envelope": envelope.artifact_payload(),
                "envelope_hash": envelope.content_hash,
                "original_slot_ordinal": envelope.slot_ordinal,
                "run_id": self.run_id,
                "slot_ordinal": envelope.slot_ordinal,
                "status": "reissued",
            },
        )
        return self._publish_slot_ref(envelope.slot_ordinal, "reissue.ref", artifact, "TRANSFER_REISSUE_CONFLICT")

    def _validate_dispatch(self, artifact: Artifact) -> V1RewardEnvelope:
        expected = {
            "envelope",
            "envelope_hash",
            "global_step",
            "original_slot_ordinal",
            "run_id",
            "status",
        }
        if artifact.schema_name != "V1TransferDispatch" or set(artifact.payload) != expected:
            raise V1RewardIdentityError("TransferQueue dispatch fields are invalid")
        envelope = V1RewardEnvelope.from_mapping(artifact.payload.get("envelope"))
        if (
            artifact.payload.get("envelope_hash") != envelope.content_hash
            or artifact.payload.get("global_step") != envelope.global_step
            or artifact.payload.get("original_slot_ordinal") != envelope.slot_ordinal
            or artifact.payload.get("run_id") != self.run_id
            or artifact.payload.get("status") != "dispatched"
        ):
            raise V1RewardIdentityError("TransferQueue dispatch identity is invalid")
        return envelope

    def validate_return(
        self, artifact: Artifact, dispatches_by_ordinal: Mapping[int, Artifact]
    ) -> tuple[V1RewardEnvelope, ClassicTrajectoryRow]:
        expected = {
            "dispatch_hash",
            "global_step",
            "identity",
            "identity_hash",
            "raw_prompt",
            "response",
            "run_id",
            "slot_ordinal",
            "status",
            "trajectory",
            "trajectory_hash",
        }
        if artifact.schema_name != "V1TransferReturn" or set(artifact.payload) != expected:
            raise V1RewardIdentityError("TransferQueue return fields are invalid")
        ordinal = _integer(artifact.payload.get("slot_ordinal"), "slot_ordinal")
        dispatch = dispatches_by_ordinal.get(ordinal)
        if dispatch is None:
            raise V1RewardIdentityError("TransferQueue return ordinal was never dispatched")
        envelope = self._validate_dispatch(dispatch)
        try:
            identity = ClassicTrajectoryIdentity.from_mapping(artifact.payload.get("identity"))
            trajectory = ClassicTrajectoryRow.from_mapping(artifact.payload.get("trajectory"))
        except ClassicIdentityError as error:
            raise V1RewardIdentityError("TransferQueue returned identity is invalid") from error
        if (
            artifact.payload.get("dispatch_hash") != dispatch.content_hash
            or artifact.payload.get("global_step") != envelope.global_step
            or identity != envelope.identity
            or artifact.payload.get("identity_hash") != identity.content_hash
            or artifact.payload.get("raw_prompt") != envelope.raw_prompt
            or artifact.payload.get("response") != trajectory.response
            or artifact.payload.get("run_id") != self.run_id
            or artifact.payload.get("status") != "returned"
            or trajectory.identity != identity
            or trajectory.prompt != envelope.raw_prompt
            or artifact.payload.get("trajectory_hash") != trajectory.content_hash
        ):
            raise V1RewardIdentityError("TransferQueue return identity/content changed")
        return envelope, trajectory

    def validate_checkpoint(self, checkpoint: Artifact) -> None:
        expected = {
            "dispatch_hashes_by_slot",
            "outstanding_slot_ordinals",
            "return_hashes_by_slot",
            "returned_slot_ordinals",
            "run_id",
            "status",
        }
        if (
            checkpoint.schema_name != "V1TransferQueueCheckpoint"
            or set(checkpoint.payload) != expected
            or checkpoint.payload.get("run_id") != self.run_id
            or checkpoint.payload.get("status") != "durable"
        ):
            raise V1RewardIdentityError("TransferQueue checkpoint is invalid")
        try:
            persisted_checkpoint_hash = (self.namespace / "checkpoint.ref").read_text(encoding="ascii").strip()
        except OSError as error:
            raise V1RewardIdentityError("TransferQueue checkpoint was not durably published") from error
        if persisted_checkpoint_hash != checkpoint.content_hash:
            raise V1RewardIdentityError("TransferQueue checkpoint ref changed")
        dispatch_entries = checkpoint.payload.get("dispatch_hashes_by_slot")
        return_entries = checkpoint.payload.get("return_hashes_by_slot")
        returned = checkpoint.payload.get("returned_slot_ordinals")
        outstanding = checkpoint.payload.get("outstanding_slot_ordinals")
        if (
            not isinstance(dispatch_entries, list)
            or not dispatch_entries
            or not isinstance(return_entries, list)
            or not isinstance(returned, list)
            or not isinstance(outstanding, list)
        ):
            raise V1RewardIdentityError("TransferQueue checkpoint slot maps are invalid")
        dispatches_by_ordinal: dict[int, Artifact] = {}
        try:
            for entry in dispatch_entries:
                if not isinstance(entry, dict) or set(entry) != {"dispatch_hash", "slot_ordinal"}:
                    raise V1RewardIdentityError("TransferQueue checkpoint dispatch entry is invalid")
                ordinal = _integer(entry.get("slot_ordinal"), "checkpoint dispatch ordinal")
                dispatch_hash = entry.get("dispatch_hash")
                if type(dispatch_hash) is not str or _HASH.fullmatch(dispatch_hash) is None:
                    raise V1RewardIdentityError("TransferQueue checkpoint dispatch hash is invalid")
                dispatch = self.store.read(dispatch_hash, expected_schema_name="V1TransferDispatch")
                envelope = self._validate_dispatch(dispatch)
                persisted_dispatch = self._read_slot_ref(ordinal, "dispatch.ref", "V1TransferDispatch")
                if ordinal in dispatches_by_ordinal or envelope.slot_ordinal != ordinal:
                    raise V1RewardIdentityError("TransferQueue checkpoint dispatch ordinal changed")
                if persisted_dispatch.content_hash != dispatch.content_hash:
                    raise V1RewardIdentityError("TransferQueue checkpoint dispatch ref changed")
                dispatches_by_ordinal[ordinal] = dispatch
            if sorted(dispatches_by_ordinal) != list(range(len(dispatches_by_ordinal))):
                raise V1RewardIdentityError("TransferQueue checkpoint dispatch set is not total")

            returns_by_ordinal: dict[int, Artifact] = {}
            for entry in return_entries:
                if not isinstance(entry, dict) or set(entry) != {"return_hash", "slot_ordinal"}:
                    raise V1RewardIdentityError("TransferQueue checkpoint return entry is invalid")
                ordinal = _integer(entry.get("slot_ordinal"), "checkpoint return ordinal")
                return_hash = entry.get("return_hash")
                if type(return_hash) is not str or _HASH.fullmatch(return_hash) is None:
                    raise V1RewardIdentityError("TransferQueue checkpoint return hash is invalid")
                returned_artifact = self.store.read(return_hash, expected_schema_name="V1TransferReturn")
                envelope, _ = self.validate_return(returned_artifact, dispatches_by_ordinal)
                persisted_return = self._read_slot_ref(ordinal, "return.ref", "V1TransferReturn")
                if ordinal in returns_by_ordinal or envelope.slot_ordinal != ordinal:
                    raise V1RewardIdentityError("TransferQueue checkpoint return ordinal changed")
                if persisted_return.content_hash != returned_artifact.content_hash:
                    raise V1RewardIdentityError("TransferQueue checkpoint return ref changed")
                returns_by_ordinal[ordinal] = returned_artifact
        except ArtifactCorruption as error:
            raise V1RewardIdentityError("TransferQueue checkpoint references corrupt artifacts") from error

        expected_returned = sorted(returns_by_ordinal)
        expected_outstanding = sorted(set(dispatches_by_ordinal) - set(returns_by_ordinal))
        if returned != expected_returned or outstanding != expected_outstanding:
            raise V1RewardIdentityError("TransferQueue checkpoint returned/outstanding partition is invalid")

    def _publish_slot_ref(self, ordinal: int, name: str, artifact: Artifact, code: str) -> Artifact:
        namespace = self.namespace / "slots" / str(ordinal)
        ArtifactStore.durable_mkdir(namespace)
        return self._publish_ref(namespace / name, artifact, code)

    def _read_slot_ref(self, ordinal: int, name: str, schema: str) -> Artifact:
        try:
            content_hash = (self.namespace / "slots" / str(ordinal) / name).read_text(encoding="ascii").strip()
            if _HASH.fullmatch(content_hash) is None:
                raise ArtifactCorruption("slot ref hash is invalid")
            return self.store.read(content_hash, expected_schema_name=schema)
        except (ArtifactCorruption, OSError) as error:
            raise V1RewardIdentityError("TransferQueue persisted slot ref is invalid") from error

    def _publish_ref(self, ref: Path, artifact: Artifact, code: str) -> Artifact:
        try:
            ArtifactStore._publish(ref, f"{artifact.content_hash}\n".encode("ascii"))
            return artifact
        except ImmutableArtifactConflict:
            pass
        try:
            existing_hash = ref.read_text(encoding="ascii").strip()
            existing = self.store.read(existing_hash, expected_schema_name=artifact.schema_name)
        except (ArtifactCorruption, OSError) as error:
            raise V1RewardIdentityError(f"{code}:existing ref is invalid") from error
        if existing.content_hash == artifact.content_hash:
            return existing
        conflict = self.store.put(
            "V1TransferQueueConflict",
            "1.0.0",
            {"conflicting_hash": artifact.content_hash, "existing_hash": existing.content_hash, "reason_code": code},
        )
        raise V1RewardIdentityError(f"{code}:{conflict.content_hash}")


@dataclass(frozen=True, slots=True)
class V1IdentitySnapshot:
    report: Artifact
    dump: Artifact
    checkpoint: Artifact
    dispatches: tuple[Artifact, ...]
    returns: tuple[Artifact, ...]
    reissues: tuple[Artifact, ...]


class V1RewardIdentityWorkflow:
    @staticmethod
    def production_readiness(root: str | Path, config: ProductionV1IdentityConfig) -> Artifact:
        checks = []
        if config.variant != _VARIANT:
            checks.append({"code": "UNVERIFIED_VERL_V1_VARIANT", "status": "blocked"})
        else:
            checks.extend(
                [
                    {"code": "REAL_VERL_V1_RUNTIME_UNVERIFIED", "status": "blocked"},
                    {"code": "TRANSFER_QUEUE_PROVIDER_UNVERIFIED", "status": "blocked"},
                    {"code": "COLOCATED_REWARD_ADAPTER_UNVERIFIED", "status": "blocked"},
                ]
            )
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "cfs_integration_claimed": False,
                "checks": checks,
                "execution_profile": "production",
                "phase": config.phase,
                "production_smoke": False,
                "side_effects_permitted": False,
                "status": "blocked",
                "variant": config.variant,
            },
        )

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: V1IdentityConfig,
        sources: tuple[ClassicSourceRow, ...],
        crash_after_checkpoint: bool = False,
    ) -> V1IdentitySnapshot:
        store = ArtifactStore(root)
        run_root = Path(root) / "v1-transfer-runs" / config.run_id
        ArtifactStore.durable_mkdir(run_root)
        input_artifact = store.put(
            "V1RewardIdentityRunInput",
            "1.0.0",
            {"config": config.artifact_payload(), "sources": [item.artifact_payload() for item in sources]},
        )
        cls._publish_ref(store, run_root / "input.ref", input_artifact, "V1_RUN_INPUT_CONFLICT")
        return cls._locked_execute(root, input_artifact, config, sources, crash_after_checkpoint)

    @classmethod
    def resume(cls, root: str | Path, run_id: str) -> V1IdentitySnapshot:
        _safe_id(run_id, "run_id")
        store = ArtifactStore(root)
        run_root = Path(root) / "v1-transfer-runs" / run_id
        try:
            input_artifact = store.read(
                (run_root / "input.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="V1RewardIdentityRunInput",
            )
        except (ArtifactCorruption, OSError) as error:
            raise V1RewardIdentityError("v1 transfer input cannot be recovered") from error
        config, sources = cls._input_contract(input_artifact)
        if config.run_id != run_id:
            raise V1RewardIdentityError("v1 resume run identity changed")
        return cls._locked_execute(root, input_artifact, config, sources, False)

    @classmethod
    def _locked_execute(
        cls,
        root: str | Path,
        input_artifact: Artifact,
        config: V1IdentityConfig,
        sources: tuple[ClassicSourceRow, ...],
        crash_after_checkpoint: bool,
    ) -> V1IdentitySnapshot:
        run_root = Path(root) / "v1-transfer-runs" / config.run_id
        lock_path = run_root / "controller.lock"
        ArtifactStore.durable_touch(lock_path)
        with lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if (run_root / "report.ref").exists():
                    return cls._recover(root, input_artifact, config, sources)
                return cls._execute(root, input_artifact, config, sources, crash_after_checkpoint)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @classmethod
    def _execute(
        cls,
        root: str | Path,
        input_artifact: Artifact,
        config: V1IdentityConfig,
        sources: tuple[ClassicSourceRow, ...],
        crash_after_checkpoint: bool,
    ) -> V1IdentitySnapshot:
        store = ArtifactStore(root)
        run_root = Path(root) / "v1-transfer-runs" / config.run_id
        envelopes = V1RewardManager.prefactor(config, sources)
        envelopes_by_ordinal = {item.slot_ordinal: item for item in envelopes}
        all_ordinals = set(envelopes_by_ordinal)
        initial = list(config.initial_return_ordinals)
        if not set(initial) < all_ordinals:
            raise V1RewardIdentityError("initial return ordinals must be a strict subset of queue slots")
        queue = PersistentFixtureTransferQueue(root, config.run_id)
        dispatches = tuple(queue.dispatch(TransferQueueSlot.from_envelope(item)) for item in envelopes)
        by_ordinal = {cast(int, item.payload["original_slot_ordinal"]): item for item in dispatches}

        returns: list[Artifact] = []
        try:
            for ordinal in initial:
                returns.append(cls._return(queue, by_ordinal[ordinal], envelopes_by_ordinal[ordinal], config))
            checkpoint = queue.checkpoint(dispatches, tuple(returns))
            cls._publish_ref(store, run_root / "checkpoint.ref", checkpoint, "V1_CHECKPOINT_CONFLICT")
            if crash_after_checkpoint:
                raise InjectedV1ControllerCrash("injected v1 controller crash after durable checkpoint")
            outstanding = cast(list[int], checkpoint.payload["outstanding_slot_ordinals"])
            reissues = tuple(queue.reissue(by_ordinal[item], checkpoint) for item in reversed(outstanding))
            for reissue in reissues:
                ordinal = cast(int, reissue.payload["original_slot_ordinal"])
                returns.append(cls._return(queue, by_ordinal[ordinal], envelopes_by_ordinal[ordinal], config))
        except InjectedV1ControllerCrash:
            raise
        except V1RewardIdentityError as error:
            failure = store.put(
                "V1TransferIdentityFailure",
                "1.0.0",
                {
                    "fault_kind": config.fault_kind,
                    "fault_slot_ordinal": config.fault_slot_ordinal,
                    "input_hash": input_artifact.content_hash,
                    "reason_code": "V1_TRANSFER_IDENTITY_CORRUPTION",
                    "status": "failed",
                },
            )
            raise V1RewardIdentityError(f"V1_TRANSFER_IDENTITY_CORRUPTION:{failure.content_hash}") from error

        decision = store.put(
            "DecisionRecord",
            "1.0.0",
            {
                "decision": "ADOPT_VERL_V1_TRANSFER_QUEUE_IDENTITY_FIXTURE",
                "rationale": (
                    "Persist original ordinals and reuse classic wide identity; real verl and CFS remain blocked."
                ),
                "scope": "fixture_only",
            },
        )
        contract = store.put(
            "V1TransferIdentityContract",
            "1.0.0",
            {
                "classic_identity_compatible": True,
                "decision_record_hash": decision.content_hash,
                "identity_fields": list(ClassicTrajectoryIdentity.__dataclass_fields__),
                "ordinal_source": "persisted_prefactor_slot",
                "variant": _VARIANT,
            },
        )
        returns_by_ordinal = {cast(int, item.payload["slot_ordinal"]): item for item in returns}
        rows = [cls._dump_row(queue, returns_by_ordinal[item], by_ordinal) for item in sorted(returns_by_ordinal)]
        identity_set_hash = sha256_hex(
            canonical_json_bytes(sorted(envelope.identity.content_hash for envelope in envelopes))
        )
        dump = store.put(
            "V1RewardTrajectoryDump",
            "1.0.0",
            {
                "cfs_integration_claimed": False,
                "checkpoint_hash": checkpoint.content_hash,
                "classic_compatibility_preserved": True,
                "identity_set_hash": identity_set_hash,
                "input_hash": input_artifact.content_hash,
                "return_arrival_ordinals": [item.payload["slot_ordinal"] for item in returns],
                "reissue_hashes": [item.content_hash for item in reissues],
                "rows": rows,
                "variant": _VARIANT,
            },
        )
        report = store.put(
            "V1RewardIdentityReport",
            "1.0.0",
            {
                "cfs_integration_claimed": False,
                "checkpoint_hash": checkpoint.content_hash,
                "classic_compatibility_preserved": True,
                "contract_hash": contract.content_hash,
                "decision_record_hash": decision.content_hash,
                "dispatch_hashes": [item.content_hash for item in dispatches],
                "dump_hash": dump.content_hash,
                "identity_set_hash": identity_set_hash,
                "input_hash": input_artifact.content_hash,
                "production_smoke": False,
                "reissue_hashes": [item.content_hash for item in reissues],
                "return_hashes": [item.content_hash for item in returns],
                "row_count": len(rows),
                "run_id": config.run_id,
                "status": "passed",
                "variant": _VARIANT,
            },
        )
        cls._publish_ref(store, run_root / "report.ref", report, "V1_REPORT_CONFLICT")
        return cls._recover(root, input_artifact, config, sources)

    @staticmethod
    def _return(
        queue: PersistentFixtureTransferQueue,
        dispatch: Artifact,
        envelope: V1RewardEnvelope,
        config: V1IdentityConfig,
    ) -> Artifact:
        response = "fixture-v1-generation/" + sha256_hex(canonical_json_bytes(envelope.artifact_payload()))[:24]
        fault = config.fault_kind if config.fault_slot_ordinal == envelope.slot_ordinal else None
        artifact = queue.commit_return(dispatch, response, fault_kind=fault)
        queue.validate_return(artifact, {envelope.slot_ordinal: dispatch})
        return artifact

    @staticmethod
    def _dump_row(
        queue: PersistentFixtureTransferQueue, returned: Artifact, dispatches: Mapping[int, Artifact]
    ) -> dict[str, JsonValue]:
        envelope, trajectory = queue.validate_return(returned, dispatches)
        return {
            "global_step": envelope.global_step,
            "identity": envelope.identity.artifact_payload(),
            "raw_prompt": envelope.raw_prompt,
            "response": trajectory.response,
            "return_hash": returned.content_hash,
            "slot_ordinal": envelope.slot_ordinal,
            "trajectory_hash": trajectory.content_hash,
        }

    @classmethod
    def _recover(
        cls,
        root: str | Path,
        input_artifact: Artifact,
        config: V1IdentityConfig,
        sources: tuple[ClassicSourceRow, ...],
    ) -> V1IdentitySnapshot:
        store = ArtifactStore(root)
        run_root = Path(root) / "v1-transfer-runs" / config.run_id
        try:
            report = store.read(
                (run_root / "report.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="V1RewardIdentityReport",
            )
        except (ArtifactCorruption, OSError) as error:
            raise V1RewardIdentityError("v1 report cannot be recovered") from error
        expected_report = {
            "cfs_integration_claimed",
            "checkpoint_hash",
            "classic_compatibility_preserved",
            "contract_hash",
            "decision_record_hash",
            "dispatch_hashes",
            "dump_hash",
            "identity_set_hash",
            "input_hash",
            "production_smoke",
            "reissue_hashes",
            "return_hashes",
            "row_count",
            "run_id",
            "status",
            "variant",
        }
        if (
            set(report.payload) != expected_report
            or report.payload.get("input_hash") != input_artifact.content_hash
            or report.payload.get("run_id") != config.run_id
            or report.payload.get("status") != "passed"
            or report.payload.get("variant") != _VARIANT
            or report.payload.get("classic_compatibility_preserved") is not True
            or report.payload.get("cfs_integration_claimed") is not False
            or report.payload.get("production_smoke") is not False
        ):
            raise V1RewardIdentityError("v1 identity report is invalid")
        envelopes = V1RewardManager.prefactor(config, sources)
        envelopes_by_ordinal = {item.slot_ordinal: item for item in envelopes}
        identity_set_hash = sha256_hex(
            canonical_json_bytes(sorted(envelope.identity.content_hash for envelope in envelopes))
        )
        if report.payload.get("identity_set_hash") != identity_set_hash:
            raise V1RewardIdentityError("v1 report identity root changed")
        contract_hash = report.payload.get("contract_hash")
        decision_hash = report.payload.get("decision_record_hash")
        if type(contract_hash) is not str or type(decision_hash) is not str:
            raise V1RewardIdentityError("v1 contract lineage hashes are invalid")
        contract = store.read(contract_hash, expected_schema_name="V1TransferIdentityContract")
        decision = store.read(decision_hash, expected_schema_name="DecisionRecord")
        if (
            set(contract.payload)
            != {
                "classic_identity_compatible",
                "decision_record_hash",
                "identity_fields",
                "ordinal_source",
                "variant",
            }
            or contract.payload.get("classic_identity_compatible") is not True
            or contract.payload.get("decision_record_hash") != decision.content_hash
            or contract.payload.get("identity_fields") != list(ClassicTrajectoryIdentity.__dataclass_fields__)
            or contract.payload.get("ordinal_source") != "persisted_prefactor_slot"
            or contract.payload.get("variant") != _VARIANT
            or set(decision.payload) != {"decision", "rationale", "scope"}
            or decision.payload.get("decision") != "ADOPT_VERL_V1_TRANSFER_QUEUE_IDENTITY_FIXTURE"
            or decision.payload.get("scope") != "fixture_only"
        ):
            raise V1RewardIdentityError("v1 identity contract lineage is invalid")
        dispatch_hashes = cls._hash_list(report.payload.get("dispatch_hashes"), len(envelopes), "dispatch")
        queue = PersistentFixtureTransferQueue(root, config.run_id)
        dispatches = tuple(store.read(item, expected_schema_name="V1TransferDispatch") for item in dispatch_hashes)
        by_ordinal: dict[int, Artifact] = {}
        for dispatch in dispatches:
            actual = queue._validate_dispatch(dispatch)
            ordinal = actual.slot_ordinal
            if ordinal not in envelopes_by_ordinal or ordinal in by_ordinal:
                raise V1RewardIdentityError("v1 dispatch ordinal set is invalid")
            if actual.artifact_payload() != envelopes_by_ordinal[ordinal].artifact_payload():
                raise V1RewardIdentityError("v1 dispatch changed persisted slot identity")
            persisted_dispatch = queue._read_slot_ref(ordinal, "dispatch.ref", "V1TransferDispatch")
            if persisted_dispatch.content_hash != dispatch.content_hash:
                raise V1RewardIdentityError("v1 persisted dispatch ref/report mapping changed")
            by_ordinal[ordinal] = dispatch
        if sorted(by_ordinal) != list(range(len(envelopes))):
            raise V1RewardIdentityError("v1 dispatch ordinal set is incomplete")
        return_hashes = cls._hash_list(report.payload.get("return_hashes"), len(envelopes), "return")
        returns = tuple(store.read(item, expected_schema_name="V1TransferReturn") for item in return_hashes)
        returned_ordinals: list[int] = []
        for item in returns:
            envelope, trajectory = queue.validate_return(item, by_ordinal)
            expected_response = (
                "fixture-v1-generation/" + sha256_hex(canonical_json_bytes(envelope.artifact_payload()))[:24]
            )
            if trajectory.response != expected_response:
                raise V1RewardIdentityError("v1 returned trajectory content changed")
            persisted_return = queue._read_slot_ref(envelope.slot_ordinal, "return.ref", "V1TransferReturn")
            if persisted_return.content_hash != item.content_hash:
                raise V1RewardIdentityError("v1 persisted return ref/report mapping changed")
            returned_ordinals.append(envelope.slot_ordinal)
        if sorted(returned_ordinals) != list(range(len(envelopes))):
            raise V1RewardIdentityError("v1 return ordinal set is incomplete or duplicated")
        outstanding = sorted(set(range(len(envelopes))) - set(config.initial_return_ordinals))
        expected_arrival = [*config.initial_return_ordinals, *reversed(outstanding)]
        if returned_ordinals != expected_arrival:
            raise V1RewardIdentityError("v1 return arrival lineage changed")
        checkpoint_hash = report.payload.get("checkpoint_hash")
        if type(checkpoint_hash) is not str:
            raise V1RewardIdentityError("v1 checkpoint hash is invalid")
        try:
            checkpoint_ref = (run_root / "checkpoint.ref").read_text(encoding="ascii").strip()
            queue_checkpoint_ref = (queue.namespace / "checkpoint.ref").read_text(encoding="ascii").strip()
        except OSError as error:
            raise V1RewardIdentityError("v1 checkpoint ref is unavailable") from error
        if checkpoint_ref != checkpoint_hash or queue_checkpoint_ref != checkpoint_hash:
            raise V1RewardIdentityError("v1 checkpoint ref/report conflict")
        checkpoint = store.read(checkpoint_hash, expected_schema_name="V1TransferQueueCheckpoint")
        try:
            queue.validate_checkpoint(checkpoint)
        except V1RewardIdentityError as error:
            raise V1RewardIdentityError("v1 checkpoint slot/hash mapping changed") from error
        if checkpoint.payload.get("returned_slot_ordinals") != sorted(config.initial_return_ordinals):
            raise V1RewardIdentityError("v1 checkpoint returned ordinal set changed")
        if checkpoint.payload.get("outstanding_slot_ordinals") != outstanding:
            raise V1RewardIdentityError("v1 checkpoint outstanding ordinal set changed")
        returns_by_ordinal = {cast(int, item.payload["slot_ordinal"]): item for item in returns}
        if checkpoint.payload.get("dispatch_hashes_by_slot") != [
            {"dispatch_hash": by_ordinal[item].content_hash, "slot_ordinal": item} for item in sorted(by_ordinal)
        ] or checkpoint.payload.get("return_hashes_by_slot") != [
            {"return_hash": returns_by_ordinal[item].content_hash, "slot_ordinal": item}
            for item in sorted(config.initial_return_ordinals)
        ]:
            raise V1RewardIdentityError("v1 checkpoint slot/hash mapping changed")
        reissue_hashes = cls._hash_list(report.payload.get("reissue_hashes"), len(outstanding), "reissue")
        reissues = tuple(store.read(item, expected_schema_name="V1TransferReissue") for item in reissue_hashes)
        for expected_ordinal, reissue in zip(reversed(outstanding), reissues, strict=True):
            persisted_reissue = queue._read_slot_ref(expected_ordinal, "reissue.ref", "V1TransferReissue")
            if (
                set(reissue.payload)
                != {
                    "checkpoint_hash",
                    "dispatch_hash",
                    "envelope",
                    "envelope_hash",
                    "original_slot_ordinal",
                    "run_id",
                    "slot_ordinal",
                    "status",
                }
                or reissue.payload.get("checkpoint_hash") != checkpoint.content_hash
                or reissue.payload.get("dispatch_hash") != by_ordinal[expected_ordinal].content_hash
                or reissue.payload.get("original_slot_ordinal") != expected_ordinal
                or reissue.payload.get("slot_ordinal") != expected_ordinal
                or reissue.payload.get("envelope") != envelopes_by_ordinal[expected_ordinal].artifact_payload()
                or reissue.payload.get("envelope_hash") != envelopes_by_ordinal[expected_ordinal].content_hash
                or reissue.payload.get("run_id") != config.run_id
                or reissue.payload.get("status") != "reissued"
                or persisted_reissue.content_hash != reissue.content_hash
            ):
                raise V1RewardIdentityError("v1 reissue changed original slot identity")
        dump_hash = report.payload.get("dump_hash")
        if type(dump_hash) is not str:
            raise V1RewardIdentityError("v1 dump hash is invalid")
        dump = store.read(dump_hash, expected_schema_name="V1RewardTrajectoryDump")
        cls._validate_dump(dump, report, input_artifact, checkpoint, returns, reissues, queue, by_ordinal)
        return V1IdentitySnapshot(report, dump, checkpoint, dispatches, returns, reissues)

    @classmethod
    def _validate_dump(
        cls,
        dump: Artifact,
        report: Artifact,
        input_artifact: Artifact,
        checkpoint: Artifact,
        returns: tuple[Artifact, ...],
        reissues: tuple[Artifact, ...],
        queue: PersistentFixtureTransferQueue,
        dispatches: Mapping[int, Artifact],
    ) -> None:
        expected = {
            "cfs_integration_claimed",
            "checkpoint_hash",
            "classic_compatibility_preserved",
            "identity_set_hash",
            "input_hash",
            "return_arrival_ordinals",
            "reissue_hashes",
            "rows",
            "variant",
        }
        rows = dump.payload.get("rows")
        if (
            set(dump.payload) != expected
            or dump.payload.get("input_hash") != input_artifact.content_hash
            or dump.payload.get("checkpoint_hash") != checkpoint.content_hash
            or dump.payload.get("identity_set_hash") != report.payload.get("identity_set_hash")
            or dump.payload.get("classic_compatibility_preserved") is not True
            or dump.payload.get("cfs_integration_claimed") is not False
            or dump.payload.get("variant") != _VARIANT
            or dump.payload.get("return_arrival_ordinals") != [item.payload["slot_ordinal"] for item in returns]
            or dump.payload.get("reissue_hashes") != [item.content_hash for item in reissues]
            or not isinstance(rows, list)
            or len(rows) != len(returns)
            or report.payload.get("row_count") != len(rows)
        ):
            raise V1RewardIdentityError("v1 dump lineage is invalid")
        returned_by_ordinal = {cast(int, item.payload["slot_ordinal"]): item for item in returns}
        expected_rows = [
            cls._dump_row(queue, returned_by_ordinal[item], dispatches) for item in sorted(returned_by_ordinal)
        ]
        if rows != expected_rows:
            raise V1RewardIdentityError("v1 dump ordinal/content mapping changed")

    @staticmethod
    def _input_contract(input_artifact: Artifact) -> tuple[V1IdentityConfig, tuple[ClassicSourceRow, ...]]:
        if set(input_artifact.payload) != {"config", "sources"}:
            raise V1RewardIdentityError("v1 run input fields are invalid")
        value = input_artifact.payload.get("config")
        sources_value = input_artifact.payload.get("sources")
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "fault_kind",
                "fault_slot_ordinal",
                "global_step",
                "initial_return_ordinals",
                "rollout_count",
                "run_id",
                "variant",
            }
            or not isinstance(sources_value, list)
        ):
            raise V1RewardIdentityError("v1 run input contract is invalid")
        initial = value.get("initial_return_ordinals")
        if not isinstance(initial, list):
            raise V1RewardIdentityError("v1 initial return ordinals are invalid")
        config = V1IdentityConfig(
            run_id=_safe_id(value.get("run_id"), "run_id"),
            global_step=_integer(value.get("global_step"), "global_step"),
            rollout_count=_integer(value.get("rollout_count"), "rollout_count", minimum=1, maximum=128),
            initial_return_ordinals=tuple(_integer(item, "initial return ordinal") for item in initial),
            variant=cast(str, value.get("variant")),
            fault_slot_ordinal=cast(int | None, value.get("fault_slot_ordinal")),
            fault_kind=cast(str | None, value.get("fault_kind")),
        )
        try:
            sources = tuple(ClassicSourceRow.from_mapping(item) for item in sources_value)
        except ClassicIdentityError as error:
            raise V1RewardIdentityError("v1 source input is invalid") from error
        return config, sources

    @staticmethod
    def _hash_list(value: object, count: int, field: str) -> tuple[str, ...]:
        if not isinstance(value, list) or len(value) != count:
            raise V1RewardIdentityError(f"v1 {field} hash cardinality is invalid")
        result: list[str] = []
        for item in value:
            if type(item) is not str or _HASH.fullmatch(item) is None:
                raise V1RewardIdentityError(f"v1 {field} hash is invalid")
            result.append(item)
        return tuple(result)

    @staticmethod
    def _publish_ref(store: ArtifactStore, ref: Path, artifact: Artifact, code: str) -> None:
        try:
            ArtifactStore._publish(ref, f"{artifact.content_hash}\n".encode("ascii"))
            return
        except ImmutableArtifactConflict:
            pass
        try:
            existing = ref.read_text(encoding="ascii").strip()
        except OSError as error:
            raise V1RewardIdentityError(f"{code}:ref unavailable") from error
        if existing == artifact.content_hash:
            return
        conflict = store.put(
            "V1TransferQueueConflict",
            "1.0.0",
            {"conflicting_hash": artifact.content_hash, "existing_hash": existing, "reason_code": code},
        )
        raise V1RewardIdentityError(f"{code}:{conflict.content_hash}")
