"""Frozen ExpectedTrajectorySet and pre-scorer slot validation."""

from __future__ import annotations

import re
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
from clawrl.training.reward_roundtrip import (
    FixtureCfsBackend,
    FixtureCfsConfig,
    RewardRoundtripError,
    RewardRoundtripWorkflow,
    RewardSlotKey,
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PHASES = {"TRAIN_35B", "TRAIN_122B"}
_TRANSPORTS = {"classic", "v1"}
_VALIDATOR_ID = "shared-expected-trajectory-set/1.0.0"


class ExpectedTrajectorySetError(RuntimeError):
    """Expected trajectory cardinality or immutable mapping failed closed."""


def _safe_id(value: object, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(cast(str, value)) is None:
        raise ExpectedTrajectorySetError(f"{field} is invalid")
    return cast(str, value)


def _integer(value: object, field: str, *, minimum: int = 0, maximum: int = 9_007_199_254_740_991) -> int:
    if type(value) is not int or not minimum <= cast(int, value) <= maximum:
        raise ExpectedTrajectorySetError(f"{field} is invalid")
    return cast(int, value)


@dataclass(frozen=True, slots=True)
class ExpectedTrajectorySetConfig:
    run_id: str
    global_step: int
    rollout_count: int

    def __post_init__(self) -> None:
        _safe_id(self.run_id, "run_id")
        _integer(self.global_step, "global_step")
        _integer(self.rollout_count, "rollout_count", minimum=1, maximum=128)


@dataclass(frozen=True, slots=True)
class ProductionExpectedTrajectorySetConfig:
    phase: str

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ExpectedTrajectorySetError("production ExpectedTrajectorySet phase is invalid")


@dataclass(frozen=True, slots=True)
class ExpectedTrajectorySetSnapshot:
    expected_set: Artifact
    manifests: tuple[Artifact, ...]
    authorization: Artifact


class ExpectedTrajectorySetWorkflow:
    @staticmethod
    def production_readiness(root: str | Path, config: ProductionExpectedTrajectorySetConfig) -> Artifact:
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": "REAL_VERL_EXPECTED_SET_ADAPTER_UNVERIFIED", "status": "blocked"},
                    {"code": "PRODUCTION_PUBLISH_IF_ABSENT_UNVERIFIED", "status": "blocked"},
                ],
                "execution_profile": "production",
                "phase": config.phase,
                "scorer_request_attempted": False,
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )

    @classmethod
    def freeze(
        cls,
        root: str | Path,
        *,
        config: ExpectedTrajectorySetConfig,
        sources: tuple[ClassicSourceRow, ...],
    ) -> Artifact:
        if not sources or len({item.uid for item in sources}) != len(sources):
            raise ExpectedTrajectorySetError("ExpectedTrajectorySet source UID set is invalid")
        store = ArtifactStore(root)
        probe = FixtureCfsBackend(
            root,
            FixtureCfsConfig(f"expected-set-{sha256_hex(config.run_id.encode())[:16]}"),
        ).probe()
        try:
            RewardRoundtripWorkflow._validate_probe(probe)
        except RewardRoundtripError as error:
            raise ExpectedTrajectorySetError("publish-if-absent backend is unqualified") from error
        slots: list[dict[str, JsonValue]] = []
        for source in sources:
            for rollout_index in range(config.rollout_count):
                key = RewardSlotKey(
                    config.run_id,
                    config.global_step,
                    source.uid,
                    rollout_index,
                    source.judge_pack_id,
                )
                slots.append(
                    {
                        "expected_rollout_count": config.rollout_count,
                        "global_step": config.global_step,
                        "judge_pack_id": source.judge_pack_id,
                        "rollout_index": rollout_index,
                        "run_id": config.run_id,
                        "slot_key_hash": key.content_hash,
                        "trace_id": source.trace_id,
                        "uid": source.uid,
                    }
                )
        artifact = store.put(
            "ExpectedTrajectorySet",
            "1.0.0",
            {
                "expected_slot_count": len(slots),
                "global_step": config.global_step,
                "probe_evidence_hash": probe.content_hash,
                "rollout_count": config.rollout_count,
                "run_id": config.run_id,
                "slots": slots,
                "validator_id": _VALIDATOR_ID,
            },
        )
        namespace = cls._namespace(root, config.run_id, config.global_step)
        ArtifactStore.durable_mkdir(namespace)
        cls._publish_ref(store, namespace / "expected-set.ref", artifact, "EXPECTED_SET_CONFLICT")
        return cls._read_expected_set(root, config.run_id, config.global_step)

    @staticmethod
    def generated_rows(
        config: ExpectedTrajectorySetConfig, sources: tuple[ClassicSourceRow, ...]
    ) -> tuple[ClassicTrajectoryRow, ...]:
        try:
            batch = ClassicRewardManager.prefactor(
                ClassicIdentityConfig(
                    config.run_id,
                    config.global_step,
                    rollout_count=config.rollout_count,
                    chunk_size=config.rollout_count,
                ),
                sources,
            )
        except ClassicIdentityError as error:
            raise ExpectedTrajectorySetError(
                "classic identity contract rejected ExpectedTrajectorySet input"
            ) from error
        return tuple(
            ClassicTrajectoryRow(
                row.identity,
                row.prompt,
                "fixture-trajectory/" + sha256_hex(canonical_json_bytes(row.artifact_payload()))[:24],
            )
            for row in batch.rows
        )

    @classmethod
    def publish_slot(
        cls,
        root: str | Path,
        *,
        expected_set: Artifact,
        row: ClassicTrajectoryRow,
    ) -> Artifact:
        slots = cls._validate_expected_set(root, expected_set)
        identity = row.identity
        key = RewardSlotKey(
            identity.run_id,
            identity.global_step,
            identity.uid,
            identity.rollout_index,
            identity.judge_pack_id,
        )
        expected = slots.get(key.content_hash)
        if expected is None or not cls._identity_matches_slot(identity, expected):
            raise ExpectedTrajectorySetError("trajectory slot is not in frozen ExpectedTrajectorySet")
        if row.response is None:
            raise ExpectedTrajectorySetError("trajectory slot has no generated response")
        store = ArtifactStore(root)
        manifest = store.put(
            "TrajectoryManifest",
            "1.0.0",
            {
                "slot_key": key.payload(),
                "slot_key_hash": key.content_hash,
                "trajectory": row.artifact_payload(),
            },
        )
        slot_root = Path(root) / "reward-slots" / key.content_hash
        ArtifactStore.durable_mkdir(slot_root)
        cls._publish_ref(store, slot_root / "trajectory.ref", manifest, "TRAJECTORY_STABLE_KEY_CONFLICT")
        return cls._read_manifest(root, expected, key.content_hash)

    @classmethod
    def publish_batch(
        cls,
        root: str | Path,
        *,
        expected_set: Artifact,
        rows: tuple[ClassicTrajectoryRow, ...],
    ) -> tuple[Artifact, ...]:
        hashes = [item.identity.content_hash for item in rows]
        if len(hashes) != len(set(hashes)):
            raise ExpectedTrajectorySetError("trajectory batch contains a duplicate identity")
        return tuple(cls.publish_slot(root, expected_set=expected_set, row=item) for item in rows)

    @classmethod
    def authorize_scoring(
        cls,
        root: str | Path,
        *,
        expected_set: Artifact,
        transport: str,
    ) -> Artifact:
        if transport not in _TRANSPORTS:
            raise ExpectedTrajectorySetError("reward transport is unsupported")
        slots = cls._validate_expected_set(root, expected_set)
        manifests: list[Artifact] = []
        missing: list[str] = []
        for slot_key_hash, slot in slots.items():
            try:
                manifests.append(cls._read_manifest(root, slot, slot_key_hash))
            except ExpectedTrajectorySetError:
                missing.append(slot_key_hash)
        if missing:
            blocked = ArtifactStore(root).put(
                "ScorerRequestBlocked",
                "1.0.0",
                {
                    "expected_set_hash": expected_set.content_hash,
                    "missing_slot_key_hashes": missing,
                    "reason_code": "EXPECTED_TRAJECTORY_SET_INCOMPLETE",
                    "scorer_called": False,
                },
            )
            raise ExpectedTrajectorySetError(f"EXPECTED_TRAJECTORY_SET_INCOMPLETE:{blocked.content_hash}")
        manifest_set_hash = sha256_hex(canonical_json_bytes(sorted(item.content_hash for item in manifests)))
        authorization = ArtifactStore(root).put(
            "ScorerRequestAuthorization",
            "1.0.0",
            {
                "expected_set_hash": expected_set.content_hash,
                "manifest_set_hash": manifest_set_hash,
                "scorer_called": False,
                "slot_count": len(manifests),
                "status": "scoring_authorized",
                "transport": transport,
                "validator_id": _VALIDATOR_ID,
            },
        )
        namespace = cls._namespace(
            root,
            cast(str, expected_set.payload["run_id"]),
            cast(int, expected_set.payload["global_step"]),
        )
        cls._publish_ref(
            ArtifactStore(root),
            namespace / "authorization.ref",
            authorization,
            "SCORER_AUTHORIZATION_CONFLICT",
        )
        return authorization

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: ExpectedTrajectorySetConfig,
        sources: tuple[ClassicSourceRow, ...],
        arrival_ordinals: tuple[int, ...],
        transport: str,
    ) -> ExpectedTrajectorySetSnapshot:
        expected_set = cls.freeze(root, config=config, sources=sources)
        rows = cls.generated_rows(config, sources)
        if sorted(arrival_ordinals) != list(range(len(rows))):
            raise ExpectedTrajectorySetError("arrival ordinals are missing or duplicated")
        rows_by_ordinal = {ordinal: row for ordinal, row in enumerate(rows)}
        cls.publish_batch(
            root,
            expected_set=expected_set,
            rows=tuple(rows_by_ordinal[item] for item in arrival_ordinals),
        )
        if transport == "v1":
            from clawrl.training.v1_transfer_identity import V1RewardManager

            V1RewardManager.authorize_expected_trajectory_set(root, expected_set)
        else:
            cls.authorize_scoring(root, expected_set=expected_set, transport=transport)
        return cls.resume(root, config.run_id, config.global_step)

    @classmethod
    def resume(cls, root: str | Path, run_id: str, global_step: int) -> ExpectedTrajectorySetSnapshot:
        expected_set = cls._read_expected_set(root, run_id, global_step)
        slots = cls._validate_expected_set(root, expected_set)
        namespace = cls._namespace(root, run_id, global_step)
        try:
            authorization = ArtifactStore(root).read(
                (namespace / "authorization.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="ScorerRequestAuthorization",
            )
        except (ArtifactCorruption, OSError) as error:
            raise ExpectedTrajectorySetError("scorer authorization cannot be recovered") from error
        manifests = tuple(cls._read_manifest(root, slot, key) for key, slot in slots.items())
        expected_authorization_fields = {
            "expected_set_hash",
            "manifest_set_hash",
            "scorer_called",
            "slot_count",
            "status",
            "transport",
            "validator_id",
        }
        manifest_set_hash = sha256_hex(canonical_json_bytes(sorted(item.content_hash for item in manifests)))
        if (
            set(authorization.payload) != expected_authorization_fields
            or authorization.payload.get("expected_set_hash") != expected_set.content_hash
            or authorization.payload.get("manifest_set_hash") != manifest_set_hash
            or authorization.payload.get("slot_count") != len(manifests)
            or authorization.payload.get("status") != "scoring_authorized"
            or authorization.payload.get("transport") not in _TRANSPORTS
            or authorization.payload.get("validator_id") != _VALIDATOR_ID
            or authorization.payload.get("scorer_called") is not False
        ):
            raise ExpectedTrajectorySetError("scorer authorization lineage is invalid")
        return ExpectedTrajectorySetSnapshot(expected_set, manifests, authorization)

    @classmethod
    def _validate_expected_set(cls, root: str | Path, artifact: Artifact) -> dict[str, dict[str, JsonValue]]:
        expected_fields = {
            "expected_slot_count",
            "global_step",
            "probe_evidence_hash",
            "rollout_count",
            "run_id",
            "slots",
            "validator_id",
        }
        slots_value = artifact.payload.get("slots")
        if (
            artifact.schema_name != "ExpectedTrajectorySet"
            or set(artifact.payload) != expected_fields
            or artifact.payload.get("validator_id") != _VALIDATOR_ID
            or not isinstance(slots_value, list)
        ):
            raise ExpectedTrajectorySetError("ExpectedTrajectorySet fields are invalid")
        run_id = _safe_id(artifact.payload.get("run_id"), "run_id")
        step = _integer(artifact.payload.get("global_step"), "global_step")
        count = _integer(artifact.payload.get("rollout_count"), "rollout_count", minimum=1, maximum=128)
        probe_hash = artifact.payload.get("probe_evidence_hash")
        if type(probe_hash) is not str or _HASH.fullmatch(probe_hash) is None:
            raise ExpectedTrajectorySetError("ExpectedTrajectorySet probe hash is invalid")
        probe = ArtifactStore(root).read(probe_hash, expected_schema_name="CfsCapabilityProbeEvidence")
        try:
            RewardRoundtripWorkflow._validate_probe(probe)
        except RewardRoundtripError as error:
            raise ExpectedTrajectorySetError("ExpectedTrajectorySet probe is unqualified") from error
        slots: dict[str, dict[str, JsonValue]] = {}
        mapping_by_uid: dict[str, tuple[str, str]] = {}
        indexes_by_uid: dict[str, list[int]] = {}
        for value in slots_value:
            if not isinstance(value, dict) or set(value) != {
                "expected_rollout_count",
                "global_step",
                "judge_pack_id",
                "rollout_index",
                "run_id",
                "slot_key_hash",
                "trace_id",
                "uid",
            }:
                raise ExpectedTrajectorySetError("ExpectedTrajectorySet slot fields are invalid")
            uid = _safe_id(value.get("uid"), "uid")
            trace_id = _safe_id(value.get("trace_id"), "trace_id")
            pack = _safe_id(value.get("judge_pack_id"), "judge_pack_id")
            index = _integer(value.get("rollout_index"), "rollout_index", maximum=count - 1)
            if (
                value.get("run_id") != run_id
                or value.get("global_step") != step
                or value.get("expected_rollout_count") != count
            ):
                raise ExpectedTrajectorySetError("ExpectedTrajectorySet slot identity changed")
            key = RewardSlotKey(run_id, step, uid, index, pack)
            if value.get("slot_key_hash") != key.content_hash or key.content_hash in slots:
                raise ExpectedTrajectorySetError("ExpectedTrajectorySet stable key is invalid or duplicated")
            if uid in mapping_by_uid and mapping_by_uid[uid] != (trace_id, pack):
                raise ExpectedTrajectorySetError("ExpectedTrajectorySet Trace/JudgePack mapping changed")
            mapping_by_uid[uid] = (trace_id, pack)
            indexes_by_uid.setdefault(uid, []).append(index)
            slots[key.content_hash] = value
        if artifact.payload.get("expected_slot_count") != len(slots) or any(
            sorted(indexes) != list(range(count)) for indexes in indexes_by_uid.values()
        ):
            raise ExpectedTrajectorySetError("ExpectedTrajectorySet Cartesian product is incomplete")
        return slots

    @staticmethod
    def _identity_matches_slot(identity: ClassicTrajectoryIdentity, slot: dict[str, JsonValue]) -> bool:
        return identity.artifact_payload() == {
            "expected_rollout_count": slot["expected_rollout_count"],
            "global_step": slot["global_step"],
            "judge_pack_id": slot["judge_pack_id"],
            "rollout_index": slot["rollout_index"],
            "run_id": slot["run_id"],
            "trace_id": slot["trace_id"],
            "uid": slot["uid"],
        }

    @classmethod
    def _read_manifest(cls, root: str | Path, slot: dict[str, JsonValue], slot_key_hash: str) -> Artifact:
        try:
            manifest = ArtifactStore(root).read(
                (Path(root) / "reward-slots" / slot_key_hash / "trajectory.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="TrajectoryManifest",
            )
        except (ArtifactCorruption, OSError) as error:
            raise ExpectedTrajectorySetError("expected trajectory slot is missing or corrupt") from error
        try:
            row = ClassicTrajectoryRow.from_mapping(manifest.payload.get("trajectory"))
        except ClassicIdentityError as error:
            raise ExpectedTrajectorySetError("TrajectoryManifest row is invalid") from error
        if (
            set(manifest.payload) != {"slot_key", "slot_key_hash", "trajectory"}
            or manifest.payload.get("slot_key_hash") != slot_key_hash
            or manifest.payload.get("slot_key")
            != RewardSlotKey(
                row.identity.run_id,
                row.identity.global_step,
                row.identity.uid,
                row.identity.rollout_index,
                row.identity.judge_pack_id,
            ).payload()
            or not cls._identity_matches_slot(row.identity, slot)
            or row.response is None
        ):
            raise ExpectedTrajectorySetError("TrajectoryManifest stable identity/content is invalid")
        return manifest

    @classmethod
    def _read_expected_set(cls, root: str | Path, run_id: str, global_step: int) -> Artifact:
        namespace = cls._namespace(root, _safe_id(run_id, "run_id"), _integer(global_step, "global_step"))
        try:
            artifact = ArtifactStore(root).read(
                (namespace / "expected-set.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="ExpectedTrajectorySet",
            )
        except (ArtifactCorruption, OSError) as error:
            raise ExpectedTrajectorySetError("ExpectedTrajectorySet cannot be recovered") from error
        cls._validate_expected_set(root, artifact)
        return artifact

    @staticmethod
    def _namespace(root: str | Path, run_id: str, global_step: int) -> Path:
        return Path(root) / "expected-trajectory-sets" / run_id / str(global_step)

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
            raise ExpectedTrajectorySetError(f"{code}:ref unavailable") from error
        if existing == artifact.content_hash:
            return
        conflict = store.put(
            "ExpectedTrajectorySetConflict",
            "1.0.0",
            {"conflicting_hash": artifact.content_hash, "existing_hash": existing, "reason_code": code},
        )
        raise ExpectedTrajectorySetError(f"{code}:{conflict.content_hash}")
