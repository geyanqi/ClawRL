"""Persistent single-trace 128-rollout fixture Router."""

from __future__ import annotations

import fcntl
import math
import os
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
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
from clawrl.training.expected_trajectory_set import ExpectedTrajectorySetError, ExpectedTrajectorySetWorkflow

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_DIMS = ("correctness", "reasoning_quality", "task_completion", "tool_discipline")


class TraceRouterError(RuntimeError):
    """Router input, capacity, fencing, or Judge result failed closed."""


class InjectedRouterCrash(TraceRouterError):
    """Synthetic crash after a committed wave."""


@dataclass(frozen=True, slots=True)
class RouterCapacityConfig:
    max_global_subthreads: int

    def __post_init__(self) -> None:
        if type(self.max_global_subthreads) is not int or not 1 <= self.max_global_subthreads <= 5:
            raise TraceRouterError("RouterCapacityConfig is invalid")


@dataclass(frozen=True, slots=True)
class TraceRouterConfig:
    run_id: str
    global_step: int
    uid: str
    resolver_epoch: int
    capacity: RouterCapacityConfig
    fault_wave: int | None = None

    def __post_init__(self) -> None:
        if (
            any(type(item) is not str or _ID.fullmatch(item) is None for item in (self.run_id, self.uid))
            or type(self.global_step) is not int
            or self.global_step < 0
            or type(self.resolver_epoch) is not int
            or self.resolver_epoch <= 0
            or self.fault_wave is not None
            and (type(self.fault_wave) is not int or self.fault_wave < 0)
        ):
            raise TraceRouterError("TraceRouterConfig is invalid")


@dataclass(frozen=True, slots=True)
class TraceRouterSnapshot:
    report: Artifact
    session: Artifact
    waves: tuple[Artifact, ...]
    results: tuple[Artifact, ...]


class RouterFenceAuthority:
    def __init__(self, root: str | Path, run_id: str) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.store = ArtifactStore(root)
        self.namespace = self.root / "router-resolver-epochs" / run_id

    @contextmanager
    def _writer_lock(self) -> Iterator[None]:
        ArtifactStore.durable_mkdir(self.namespace)
        lock_path = self.namespace / "writer.lock"
        ArtifactStore.durable_touch(lock_path)
        descriptor = os.open(lock_path, os.O_RDWR)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _validate_epoch(self, artifact: Artifact) -> int:
        initial_epoch: int | None = None
        newer_epoch: int | None = None
        seen: set[str] = set()
        cursor = artifact
        while True:
            payload = cursor.payload
            previous = payload.get("previous_epoch_hash")
            epoch = payload.get("epoch")
            if (
                cursor.schema_version != "1.0.0"
                or set(payload) != {"epoch", "previous_epoch_hash", "run_id"}
                or payload.get("run_id") != self.run_id
                or type(epoch) is not int
                or cast(int, epoch) <= 0
                or newer_epoch is not None
                and cast(int, epoch) >= newer_epoch
                or previous is not None
                and (type(previous) is not str or _HASH.fullmatch(previous) is None)
                or cursor.content_hash in seen
            ):
                raise TraceRouterError("router resolver epoch chain is invalid")
            if initial_epoch is None:
                initial_epoch = cast(int, epoch)
            seen.add(cursor.content_hash)
            if previous is None:
                return initial_epoch
            newer_epoch = cast(int, epoch)
            cursor = self.store.read(previous, expected_schema_name="RouterResolverEpoch")

    def _read_current_locked(self) -> Artifact | None:
        ref = self.namespace / "current.ref"
        try:
            raw_ref = ref.read_bytes()
        except FileNotFoundError:
            return None
        if len(raw_ref) != 65 or raw_ref[-1:] != b"\n":
            raise TraceRouterError("router resolver epoch ref is invalid")
        try:
            content_hash = raw_ref[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise TraceRouterError("router resolver epoch ref is invalid") from error
        if _HASH.fullmatch(content_hash) is None:
            raise TraceRouterError("router resolver epoch ref is invalid")
        current = self.store.read(content_hash, expected_schema_name="RouterResolverEpoch")
        self._validate_epoch(current)
        return current

    def _replace_current_locked(self, artifact: Artifact) -> None:
        ref = self.namespace / "current.ref"
        temporary = ref.with_name(f".{ref.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(f"{artifact.content_hash}\n".encode("ascii"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, ref)
            directory_descriptor = os.open(self.namespace, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def current(self) -> Artifact:
        with self._writer_lock():
            current = self._read_current_locked()
            if current is None:
                raise TraceRouterError("router resolver epoch is unavailable")
            return current

    def ensure(self, epoch: int) -> Artifact:
        if type(epoch) is not int or epoch <= 0:
            raise TraceRouterError("router resolver epoch is invalid")
        with self._writer_lock():
            current = self._read_current_locked()
            if current is None:
                artifact = self.store.put(
                    "RouterResolverEpoch",
                    "1.0.0",
                    {"epoch": epoch, "previous_epoch_hash": None, "run_id": self.run_id},
                )
                self._replace_current_locked(artifact)
                return artifact
            if self._validate_epoch(current) != epoch:
                raise TraceRouterError("STALE_ROUTER_RESOLVER_EPOCH")
            return current

    def advance(self, epoch: int) -> Artifact:
        if type(epoch) is not int or epoch <= 0:
            raise TraceRouterError("router resolver epoch is invalid")
        with self._writer_lock():
            current = self._read_current_locked()
            if current is not None and epoch <= self._validate_epoch(current):
                raise TraceRouterError("STALE_ROUTER_RESOLVER_EPOCH")
            artifact = self.store.put(
                "RouterResolverEpoch",
                "1.0.0",
                {
                    "epoch": epoch,
                    "previous_epoch_hash": None if current is None else current.content_hash,
                    "run_id": self.run_id,
                },
            )
            self._replace_current_locked(artifact)
            return artifact


class FixtureTraceRouter:
    @staticmethod
    def production_readiness(root: str | Path, phase: str) -> Artifact:
        if phase not in {"TRAIN_35B", "TRAIN_122B"}:
            raise TraceRouterError("router production phase is invalid")
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": "REAL_CODEX_ROUTER_UNAVAILABLE", "status": "blocked"},
                    {"code": "ROUTER_CAPACITY_APPROVAL_MISSING", "status": "blocked"},
                    {"code": "EXTERNAL_ROUTER_FENCE_AUTHORITY_UNVERIFIED", "status": "blocked"},
                ],
                "execution_profile": "production",
                "phase": phase,
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: TraceRouterConfig,
        expected_set: Artifact,
        judge_pack: Artifact,
        crash_after_wave: int | None = None,
    ) -> TraceRouterSnapshot:
        store = ArtifactStore(root)
        base_namespace = Path(root) / "trace-router-runs" / config.run_id / str(config.global_step) / config.uid
        namespace = base_namespace / "epochs" / str(config.resolver_epoch)
        authorization = cls._expected_gate(root, config, expected_set)
        manifests = cls._manifests(root, expected_set, config.uid)
        pack = cls._pack(judge_pack, config.uid, expected_set)
        RouterFenceAuthority(root, config.run_id).ensure(config.resolver_epoch)
        identity = store.put(
            "TraceRouterIdentity",
            "1.0.0",
            {
                "authorization_hash": authorization.content_hash,
                "capacity": config.capacity.max_global_subthreads,
                "expected_set_hash": expected_set.content_hash,
                "global_step": config.global_step,
                "judge_pack_hash": judge_pack.content_hash,
                "run_id": config.run_id,
                "uid": config.uid,
            },
        )
        cls._publish_once(base_namespace / "identity.ref", identity)
        ArtifactStore.durable_mkdir(namespace)
        input_artifact = store.put(
            "TraceRouterInput",
            "1.0.0",
            {
                "authorization_hash": authorization.content_hash,
                "capacity": config.capacity.max_global_subthreads,
                "expected_set_hash": expected_set.content_hash,
                "fault_wave": config.fault_wave,
                "global_step": config.global_step,
                "identity_hash": identity.content_hash,
                "judge_pack_hash": judge_pack.content_hash,
                "resolver_epoch": config.resolver_epoch,
                "run_id": config.run_id,
                "uid": config.uid,
            },
        )
        cls._publish_once(namespace / "input.ref", input_artifact)
        if (namespace / "report.ref").exists():
            return cls.resume(root, config.run_id, config.global_step, config.uid)
        attempt = 1
        while (namespace / f"session-{attempt}.ref").exists():
            attempt += 1
        session = store.put(
            "ScoringSession",
            "1.0.0",
            {
                "attempt": attempt,
                "hidden_memory_restored": False,
                "items_per_turn": pack["items_per_turn"],
                "judge_pack_hash": judge_pack.content_hash,
                "run_id": config.run_id,
                "session_id": f"fixture-router-session-{attempt}-{input_artifact.content_hash[:12]}",
                "sol_shadow_called": False,
                "global_step": config.global_step,
                "uid": config.uid,
            },
        )
        cls._publish_once(namespace / f"session-{attempt}.ref", session)
        items_per_turn = cast(int, pack["items_per_turn"])
        wave_count = math.ceil(len(manifests) / items_per_turn)
        waves: list[Artifact] = []
        results: list[Artifact] = []
        for wave_index in range(wave_count):
            wave_ref = namespace / "waves" / f"{wave_index}.ref"
            if wave_ref.exists():
                wave = store.read(wave_ref.read_text().strip(), expected_schema_name="RouterWave")
                wave_results = [
                    store.read(cast(str, item), expected_schema_name="FencedJudgeResult")
                    for item in cast(list[str], wave.payload["result_hashes"])
                ]
            else:
                batch = manifests[wave_index * items_per_turn : (wave_index + 1) * items_per_turn]
                wave_results = [
                    cls._result(store, config, judge_pack, session, wave_index, index, manifest)
                    for index, manifest in enumerate(reversed(batch))
                ]
                if config.fault_wave == wave_index:
                    bad = dict(wave_results[0].payload)
                    bad["confidence_basis_points"] = 10_001
                    wave_results[0] = store.put("FencedJudgeResult", "1.0.0", bad)
                turn_manifest_hashes = {manifest.content_hash for manifest in batch}
                for item in wave_results:
                    cls._validate_result(
                        item,
                        config,
                        judge_pack,
                        session,
                        {manifest.content_hash for manifest in manifests},
                        turn_manifest_hashes,
                    )
                wave = store.put(
                    "RouterWave",
                    "1.0.0",
                    {
                        "active_subthreads": min(5, config.capacity.max_global_subthreads, len(batch)),
                        "admission_policy": "queue_only_no_predicted_delay_rejection",
                        "capacity_limit": config.capacity.max_global_subthreads,
                        "queued_item_count": max(0, len(batch) - config.capacity.max_global_subthreads),
                        "result_hashes": [item.content_hash for item in wave_results],
                        "session_hash": session.content_hash,
                        "wave_index": wave_index,
                    },
                )
                ArtifactStore.durable_mkdir(wave_ref.parent)
                cls._publish_once(wave_ref, wave)
                if crash_after_wave == wave_index:
                    raise InjectedRouterCrash("injected crash after committed RouterWave")
            waves.append(wave)
            results.extend(wave_results)
        report = store.put(
            "TraceRouterReport",
            "1.0.0",
            {
                "aggregation": "calibrated_scalar",
                "expected_set_hash": expected_set.content_hash,
                "judge_pack_hash": judge_pack.content_hash,
                "prompt_optimized_during_training": False,
                "result_count": len(results),
                "result_set_hash": sha256_hex(canonical_json_bytes(sorted(item.content_hash for item in results))),
                "session_hash": session.content_hash,
                "sol_shadow_called": False,
                "status": "resolved",
                "wave_hashes": [item.content_hash for item in waves],
            },
        )
        cls._publish_once(namespace / "report.ref", report)
        return cls.resume(root, config.run_id, config.global_step, config.uid)

    @classmethod
    def resume(cls, root: str | Path, run_id: str, global_step: int, uid: str) -> TraceRouterSnapshot:
        base_namespace = Path(root) / "trace-router-runs" / run_id / str(global_step) / uid
        store = ArtifactStore(root)
        try:
            current_epoch = RouterFenceAuthority(root, run_id).current()
            resolver_epoch = current_epoch.payload["epoch"]
            if type(resolver_epoch) is not int:
                raise TraceRouterError("router resolver epoch artifact is invalid")
            namespace = base_namespace / "epochs" / str(resolver_epoch)
            identity = store.read(
                (base_namespace / "identity.ref").read_text().strip(), expected_schema_name="TraceRouterIdentity"
            )
            input_artifact = store.read(
                (namespace / "input.ref").read_text().strip(), expected_schema_name="TraceRouterInput"
            )
            report = store.read(
                (namespace / "report.ref").read_text().strip(), expected_schema_name="TraceRouterReport"
            )
            session = store.read(cast(str, report.payload["session_hash"]), expected_schema_name="ScoringSession")
        except (ArtifactCorruption, KeyError, OSError, TypeError, ValueError) as error:
            raise TraceRouterError("router cannot be recovered") from error
        expected_set = store.read(
            cast(str, input_artifact.payload["expected_set_hash"]), expected_schema_name="ExpectedTrajectorySet"
        )
        pack = store.read(cast(str, input_artifact.payload["judge_pack_hash"]), expected_schema_name="JudgePack")
        config = TraceRouterConfig(
            run_id,
            global_step,
            uid,
            cast(int, input_artifact.payload["resolver_epoch"]),
            RouterCapacityConfig(cast(int, input_artifact.payload["capacity"])),
            cast(int | None, input_artifact.payload["fault_wave"]),
        )
        authorization = cls._expected_gate(root, config, expected_set)
        manifests = cls._manifests(root, expected_set, uid)
        pack_payload = cls._pack(pack, uid, expected_set)
        RouterFenceAuthority(root, run_id).ensure(config.resolver_epoch)
        expected_input_fields = {
            "authorization_hash",
            "capacity",
            "expected_set_hash",
            "fault_wave",
            "global_step",
            "identity_hash",
            "judge_pack_hash",
            "resolver_epoch",
            "run_id",
            "uid",
        }
        if (
            set(input_artifact.payload) != expected_input_fields
            or input_artifact.payload.get("authorization_hash") != authorization.content_hash
            or input_artifact.payload.get("expected_set_hash") != expected_set.content_hash
            or input_artifact.payload.get("judge_pack_hash") != pack.content_hash
            or input_artifact.payload.get("identity_hash") != identity.content_hash
            or input_artifact.payload.get("run_id") != run_id
            or input_artifact.payload.get("global_step") != global_step
            or input_artifact.payload.get("uid") != uid
        ):
            raise TraceRouterError("router input lineage is invalid")
        expected_identity = {
            "authorization_hash",
            "capacity",
            "expected_set_hash",
            "global_step",
            "judge_pack_hash",
            "run_id",
            "uid",
        }
        if (
            identity.schema_version != "1.0.0"
            or set(identity.payload) != expected_identity
            or identity.payload.get("authorization_hash") != authorization.content_hash
            or identity.payload.get("capacity") != config.capacity.max_global_subthreads
            or identity.payload.get("expected_set_hash") != expected_set.content_hash
            or identity.payload.get("global_step") != global_step
            or identity.payload.get("judge_pack_hash") != pack.content_hash
            or identity.payload.get("run_id") != run_id
            or identity.payload.get("uid") != uid
        ):
            raise TraceRouterError("router identity lineage is invalid")
        cls._validate_session(session, config, pack, cast(int, pack_payload["items_per_turn"]))
        items_per_turn = cast(int, pack_payload["items_per_turn"])
        wave_hashes_value = report.payload.get("wave_hashes")
        expected_wave_count = math.ceil(len(manifests) / items_per_turn)
        if (
            not isinstance(wave_hashes_value, list)
            or len(wave_hashes_value) != expected_wave_count
            or any(type(item) is not str or _HASH.fullmatch(item) is None for item in wave_hashes_value)
        ):
            raise TraceRouterError("router wave cardinality is invalid")
        wave_hashes = cast(list[str], wave_hashes_value)
        waves = tuple(store.read(item, expected_schema_name="RouterWave") for item in wave_hashes)
        results_list: list[Artifact] = []
        manifest_hashes = {item.content_hash for item in manifests}
        for expected_index, wave in enumerate(waves):
            if (
                set(wave.payload)
                != {
                    "active_subthreads",
                    "admission_policy",
                    "capacity_limit",
                    "queued_item_count",
                    "result_hashes",
                    "session_hash",
                    "wave_index",
                }
                or wave.payload.get("wave_index") != expected_index
                or wave.payload.get("capacity_limit") != config.capacity.max_global_subthreads
                or wave.payload.get("admission_policy") != "queue_only_no_predicted_delay_rejection"
            ):
                raise TraceRouterError("router wave lineage is invalid")
            wave_session_hash = wave.payload.get("session_hash")
            if type(wave_session_hash) is not str or _HASH.fullmatch(wave_session_hash) is None:
                raise TraceRouterError("router wave session hash is invalid")
            wave_session = store.read(wave_session_hash, expected_schema_name="ScoringSession")
            result_hashes_value = wave.payload.get("result_hashes")
            expected_result_count = min(items_per_turn, len(manifests) - expected_index * items_per_turn)
            if (
                not isinstance(result_hashes_value, list)
                or len(result_hashes_value) != expected_result_count
                or len(set(cast(list[object], result_hashes_value))) != expected_result_count
                or any(type(item) is not str or _HASH.fullmatch(item) is None for item in result_hashes_value)
            ):
                raise TraceRouterError("router wave result cardinality is invalid")
            result_hashes = cast(list[str], result_hashes_value)
            wave_results = [store.read(item, expected_schema_name="FencedJudgeResult") for item in result_hashes]
            expected_active = min(5, config.capacity.max_global_subthreads, len(result_hashes))
            turn_manifest_hashes = {cast(str, item.payload.get("trajectory_manifest_hash")) for item in wave_results}
            for item in wave_results:
                cls._validate_result(
                    item,
                    config,
                    pack,
                    wave_session,
                    manifest_hashes,
                    turn_manifest_hashes,
                )
            cls._validate_session(
                wave_session,
                config,
                pack,
                cast(int, pack_payload["items_per_turn"]),
            )
            if wave.payload.get("active_subthreads") != expected_active or wave.payload.get("queued_item_count") != max(
                0, len(result_hashes) - config.capacity.max_global_subthreads
            ):
                raise TraceRouterError("router wave capacity evidence is invalid")
            results_list.extend(wave_results)
        results = tuple(results_list)
        if len(results) != 128 or report.payload.get("result_count") != 128:
            raise TraceRouterError("router result cardinality is invalid")
        if {item.payload.get("trajectory_manifest_hash") for item in results} != manifest_hashes:
            raise TraceRouterError("router result TrajectoryManifest coverage is invalid")
        expected_report = {
            "aggregation",
            "expected_set_hash",
            "judge_pack_hash",
            "prompt_optimized_during_training",
            "result_count",
            "result_set_hash",
            "session_hash",
            "sol_shadow_called",
            "status",
            "wave_hashes",
        }
        if (
            set(report.payload) != expected_report
            or report.payload.get("expected_set_hash") != expected_set.content_hash
            or report.payload.get("judge_pack_hash") != pack.content_hash
            or report.payload.get("wave_hashes") != [item.content_hash for item in waves]
            or report.payload.get("result_set_hash")
            != sha256_hex(canonical_json_bytes(sorted(item.content_hash for item in results)))
            or report.payload.get("aggregation") != "calibrated_scalar"
            or report.payload.get("status") != "resolved"
            or report.payload.get("sol_shadow_called") is not False
            or report.payload.get("prompt_optimized_during_training") is not False
        ):
            raise TraceRouterError("router report lineage is invalid")
        return TraceRouterSnapshot(report, session, waves, results)

    @staticmethod
    def _validate_session(
        session: Artifact,
        config: TraceRouterConfig,
        pack: Artifact,
        items_per_turn: int,
    ) -> None:
        expected = {
            "attempt",
            "global_step",
            "hidden_memory_restored",
            "items_per_turn",
            "judge_pack_hash",
            "run_id",
            "session_id",
            "sol_shadow_called",
            "uid",
        }
        if (
            session.schema_version != "1.0.0"
            or set(session.payload) != expected
            or session.payload.get("run_id") != config.run_id
            or session.payload.get("global_step") != config.global_step
            or session.payload.get("uid") != config.uid
            or session.payload.get("judge_pack_hash") != pack.content_hash
            or session.payload.get("items_per_turn") != items_per_turn
            or session.payload.get("hidden_memory_restored") is not False
            or session.payload.get("sol_shadow_called") is not False
            or type(session.payload.get("attempt")) is not int
            or cast(int, session.payload["attempt"]) <= 0
            or type(session.payload.get("session_id")) is not str
            or _ID.fullmatch(cast(str, session.payload["session_id"])) is None
        ):
            raise TraceRouterError("ScoringSession lineage is invalid")

    @staticmethod
    def _expected_gate(root: str | Path, config: TraceRouterConfig, expected_set: Artifact) -> Artifact:
        try:
            snapshot = ExpectedTrajectorySetWorkflow.resume(root, config.run_id, config.global_step)
        except ExpectedTrajectorySetError as error:
            raise TraceRouterError("EXPECTED_TRAJECTORY_SET_INCOMPLETE") from error
        if snapshot.expected_set.content_hash != expected_set.content_hash:
            raise TraceRouterError("ExpectedTrajectorySet identity changed")
        return snapshot.authorization

    @staticmethod
    def _manifests(root: str | Path, expected_set: Artifact, uid: str) -> tuple[Artifact, ...]:
        snapshot = ExpectedTrajectorySetWorkflow.resume(
            root, cast(str, expected_set.payload["run_id"]), cast(int, expected_set.payload["global_step"])
        )
        selected = tuple(
            item for item in snapshot.manifests if cast(dict[str, JsonValue], item.payload["slot_key"])["uid"] == uid
        )
        if len(selected) != 128:
            raise TraceRouterError("single Trace Router requires exactly 128 rollout")
        return selected

    @staticmethod
    def _pack(pack: Artifact, uid: str, expected_set: Artifact) -> dict[str, JsonValue]:
        expected = {
            "aggregation",
            "algorithm_contract",
            "items_per_turn",
            "judge_pack_id",
            "judge_pack_version",
            "prompt_hash",
            "scalarizer_hash",
            "scalarizer_version",
            "scorer_tier",
            "status",
            "trace_id",
            "uid",
        }
        if (
            pack.schema_name != "JudgePack"
            or set(pack.payload) != expected
            or pack.payload.get("uid") != uid
            or pack.payload.get("judge_pack_version") != pack.schema_version
            or pack.payload.get("scalarizer_version") != "1.0.0"
            or pack.payload.get("status") != "certified"
            or pack.payload.get("aggregation") != "calibrated_scalar"
            or pack.payload.get("items_per_turn") not in {4, 8, 16, 32}
            or pack.payload.get("scorer_tier") not in {"luna", "sol"}
            or pack.payload.get("scorer_tier") == "sol"
            and pack.payload.get("items_per_turn") != 4
        ):
            raise TraceRouterError("JudgePack routing contract is invalid")
        mapped = [
            item for item in cast(list[dict[str, JsonValue]], expected_set.payload["slots"]) if item.get("uid") == uid
        ]
        if (
            not mapped
            or {item.get("trace_id") for item in mapped} != {pack.payload.get("trace_id")}
            or {item.get("judge_pack_id") for item in mapped} != {pack.payload.get("judge_pack_id")}
        ):
            raise TraceRouterError("JudgePack does not match ExpectedTrajectorySet mapping")
        return pack.payload

    @staticmethod
    def _result(
        store: ArtifactStore,
        config: TraceRouterConfig,
        pack: Artifact,
        session: Artifact,
        wave: int,
        turn: int,
        manifest: Artifact,
    ) -> Artifact:
        seed = int(manifest.content_hash[:8], 16)
        dimensions = {name: 40 + (seed >> offset) % 61 for offset, name in enumerate(_DIMS)}
        scalar = sum(dimensions.values()) * 250_000
        pack_payload = pack.payload
        return store.put(
            "FencedJudgeResult",
            "1.0.0",
            {
                "confidence_basis_points": 7000 + seed % 3001,
                "dimensions": dimensions,
                "evidence": [manifest.content_hash],
                "failure_tags": [],
                "judge_pack_hash": pack.content_hash,
                "judge_pack_version": pack.schema_version,
                "resolver_epoch": config.resolver_epoch,
                "scalar_micros": scalar,
                "scalarizer_hash": pack_payload["scalarizer_hash"],
                "scalarizer_version": pack_payload["scalarizer_version"],
                "session_hash": session.content_hash,
                "thread_ref": f"thread-{turn % min(5, config.capacity.max_global_subthreads)}",
                "trajectory_manifest_hash": manifest.content_hash,
                "turn_local_tie_groups": [[manifest.content_hash]],
                "turn_ref": f"wave-{wave}-turn-{turn}",
            },
        )

    @staticmethod
    def _validate_result(
        result: Artifact,
        config: TraceRouterConfig,
        pack: Artifact,
        session: Artifact,
        manifest_hashes: set[str],
        turn_manifest_hashes: set[str],
    ) -> None:
        payload = result.payload
        dimensions = payload.get("dimensions")
        expected = {
            "confidence_basis_points",
            "dimensions",
            "evidence",
            "failure_tags",
            "judge_pack_hash",
            "judge_pack_version",
            "resolver_epoch",
            "scalar_micros",
            "scalarizer_hash",
            "scalarizer_version",
            "session_hash",
            "thread_ref",
            "trajectory_manifest_hash",
            "turn_local_tie_groups",
            "turn_ref",
        }
        pack_payload = pack.payload
        evidence = payload.get("evidence")
        failure_tags = payload.get("failure_tags")
        tie_groups = payload.get("turn_local_tie_groups")
        refs = [
            payload.get("judge_pack_hash"),
            payload.get("scalarizer_hash"),
            payload.get("session_hash"),
            payload.get("trajectory_manifest_hash"),
        ]
        if (
            result.schema_version != "1.0.0"
            or set(payload) != expected
            or payload.get("resolver_epoch") != config.resolver_epoch
            or not isinstance(dimensions, dict)
            or set(dimensions) != set(_DIMS)
            or any(type(item) is not int or not 0 <= item <= 100 for item in dimensions.values())
            or type(payload.get("scalar_micros")) is not int
            or payload.get("scalar_micros") != sum(cast(dict[str, int], dimensions).values()) * 250_000
            or type(payload.get("confidence_basis_points")) is not int
            or not 0 <= cast(int, payload["confidence_basis_points"]) <= 10_000
            or payload.get("scalarizer_hash") != pack_payload.get("scalarizer_hash")
            or payload.get("scalarizer_version") != pack_payload.get("scalarizer_version")
            or payload.get("judge_pack_hash") != pack.content_hash
            or payload.get("judge_pack_version") != pack.schema_version
            or payload.get("session_hash") != session.content_hash
            or payload.get("trajectory_manifest_hash") not in manifest_hashes
            or any(type(item) is not str or _HASH.fullmatch(item) is None for item in refs)
            or not isinstance(failure_tags, list)
            or any(type(item) is not str for item in failure_tags)
            or not isinstance(evidence, list)
            or not evidence
            or any(type(item) is not str or _HASH.fullmatch(item) is None for item in evidence)
            or not isinstance(tie_groups, list)
            or not tie_groups
            or any(
                not isinstance(group, list)
                or not group
                or any(type(item) is not str or _HASH.fullmatch(item) is None for item in group)
                for group in tie_groups
            )
            or any(
                cast(str, item) not in turn_manifest_hashes
                for group in cast(list[list[JsonValue]], tie_groups)
                for item in group
            )
            or type(payload.get("thread_ref")) is not str
            or _ID.fullmatch(cast(str, payload["thread_ref"])) is None
            or type(payload.get("turn_ref")) is not str
            or _ID.fullmatch(cast(str, payload["turn_ref"])) is None
        ):
            raise TraceRouterError("INVALID_JUDGE_RESULT")

    @staticmethod
    def _publish_once(ref: Path, artifact: Artifact) -> None:
        try:
            ArtifactStore._publish(ref, f"{artifact.content_hash}\n".encode())
        except ImmutableArtifactConflict:
            if ref.read_text().strip() != artifact.content_hash:
                raise TraceRouterError("router immutable ref conflict") from None
