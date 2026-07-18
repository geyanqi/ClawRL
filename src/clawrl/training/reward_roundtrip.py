"""Production-shaped synthetic CFS probe and fenced reward roundtrip."""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from clawrl.artifacts import (
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    ImmutableArtifactConflict,
    JsonValue,
    canonical_json_bytes,
    sha256_hex,
)

_HASH = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_ATTEMPT_LEASE_TICKS = 4
_MAX_ATTEMPTS = 3


class RewardRoundtripError(RuntimeError):
    """The CFS/reward state machine failed closed."""


class RewardIntegrityError(RewardRoundtripError):
    """A stable identity was reused with conflicting immutable content."""


@dataclass(frozen=True, slots=True)
class RewardSlotKey:
    run_id: str
    global_step: int
    uid: str
    rollout_index: int
    judge_pack_id: str

    def __post_init__(self) -> None:
        if (
            any(_ID.fullmatch(item) is None for item in (self.run_id, self.uid, self.judge_pack_id))
            or type(self.global_step) is not int
            or self.global_step < 0
            or type(self.rollout_index) is not int
            or self.rollout_index < 0
        ):
            raise ValueError("reward slot key is invalid")

    def payload(self) -> dict[str, JsonValue]:
        return {
            "global_step": self.global_step,
            "judge_pack_id": self.judge_pack_id,
            "rollout_index": self.rollout_index,
            "run_id": self.run_id,
            "uid": self.uid,
        }

    @property
    def content_hash(self) -> str:
        return sha256_hex(canonical_json_bytes({"domain": "reward-slot-key/1.0.0", "key": self.payload()}))


@dataclass(frozen=True, slots=True)
class FixtureCfsConfig:
    backend_id: str
    fault: Literal["valid", "overwrite", "partial_visibility", "cleanup_failure"] = "valid"
    visibility_deadline_ticks: int = 4

    def __post_init__(self) -> None:
        if _ID.fullmatch(self.backend_id) is None or self.visibility_deadline_ticks != 4:
            raise ValueError("fixture CFS config is invalid")


@dataclass(frozen=True, slots=True)
class ProductionCfsRewardConfig:
    cfs_approval_hash: str | None = None
    fence_authority_approval_hash: str | None = None
    reward_provider_approval_hash: str | None = None
    attempt_policy_approval_hash: str | None = None


@dataclass(frozen=True, slots=True)
class RewardSnapshot:
    trajectory_manifest: Artifact
    reward_request: Artifact
    resolved_reward: Artifact | None
    attempt_ordinals: tuple[int, ...]


class FixtureCfsBackend:
    """Stateful deterministic cross-client CFS simulator with injected faults."""

    def __init__(self, root: str | Path, config: FixtureCfsConfig) -> None:
        self.root = Path(root)
        self.config = config
        self.store = ArtifactStore(root)
        self.namespace = self.root / "cfs-probes" / config.backend_id

    def probe(self) -> Artifact:
        evidence_ref = self.namespace / "evidence.ref"
        if evidence_ref.exists():
            existing = self.store.read(
                evidence_ref.read_text(encoding="ascii").strip(),
                expected_schema_name="CfsCapabilityProbeEvidence",
            )
            if existing.payload.get("fixture_fault_injection") != self.config.fault:
                raise RewardIntegrityError("CFS probe namespace was reused with a different fixture fault")
            return existing

        target = self.namespace / "winner.manifest"
        ArtifactStore.durable_mkdir(self.namespace)
        payloads = {
            "client-a": b"cfs-probe-payload-v1/client-a",
            "client-b": b"cfs-probe-payload-v1/client-b",
        }
        manifests: dict[str, bytes] = {}
        candidate_payloads: list[dict[str, JsonValue]] = []
        for client_id, payload in payloads.items():
            digest = sha256_hex(payload)
            size = len(payload)
            ArtifactStore._publish(self.namespace / "payloads" / f"{digest}.blob", payload)
            candidate_payloads.append({"client_id": client_id, "payload_hash": digest, "payload_size": size})
            manifests[client_id] = canonical_json_bytes(
                {"client_id": client_id, "payload_hash": digest, "payload_size": size}
            )

        ready = threading.Barrier(2)
        first_publication_finished = threading.Event()

        def publish(client_id: str) -> dict[str, JsonValue]:
            temporary = self.namespace / f".{client_id}.manifest.tmp"
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(manifests[client_id])
                stream.flush()
                os.fsync(stream.fileno())
            ready.wait()
            if client_id == "client-b":
                first_publication_finished.wait()
            publication_status = "unattempted"
            try:
                if self.config.fault == "overwrite":
                    os.replace(temporary, target)
                    publication_status = "published_overwrite"
                elif self.config.fault == "partial_visibility" and client_id == "client-a":
                    partial = manifests[client_id][: max(1, len(manifests[client_id]) // 2)]
                    target_descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(target_descriptor, "wb") as stream:
                        stream.write(partial)
                        stream.flush()
                        os.fsync(stream.fileno())
                    publication_status = "published_partial"
                else:
                    os.link(temporary, target)
                    publication_status = "published_no_replace"
                self._sync_namespace()
            except FileExistsError:
                publication_status = "file_exists_no_replace"
            finally:
                if client_id == "client-a":
                    first_publication_finished.set()

            cleanup_status = "not_required"
            retained_after_cleanup = False
            if temporary.exists():
                try:
                    if self.config.fault == "cleanup_failure" and client_id == "client-b":
                        raise OSError("injected cleanup failure")
                    temporary.unlink()
                    self._sync_namespace()
                    cleanup_status = "succeeded"
                except OSError:
                    cleanup_status = "failed"
                    retained_after_cleanup = temporary.exists()
            return {
                "cleanup_status": cleanup_status,
                "client_id": client_id,
                "mount_view_id": f"fixture-mount-{client_id[-1]}",
                "publication_status": publication_status,
                "retained_after_cleanup": retained_after_cleanup,
            }

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cfs-probe-client") as executor:
            futures = [executor.submit(publish, client_id) for client_id in payloads]
            client_results = sorted(
                (future.result() for future in futures),
                key=lambda item: cast(str, item["client_id"]),
            )

        winner_statuses = {"published_no_replace", "published_partial", "published_overwrite"}
        winner_count = sum(item["publication_status"] in winner_statuses for item in client_results)
        loser_results = [item for item in client_results if item["publication_status"] not in winner_statuses]
        loser_behavior = (
            cast(str, loser_results[0]["publication_status"]) if len(loser_results) == 1 else "no_unique_loser"
        )
        cleanup_status = "failed" if any(item["cleanup_status"] == "failed" for item in client_results) else "succeeded"

        consumer_status = "manifest_unavailable"
        payload_hash = "0" * 64
        payload_size = 0
        partial_visible = False
        checksum_valid = False
        visibility_tick = 0
        for tick in range(1, self.config.visibility_deadline_ticks + 1):
            if target.exists():
                visibility_tick = tick
                break
        try:
            if visibility_tick == 0:
                raise OSError("manifest did not become visible before its deadline")
            raw_manifest = target.read_bytes()
            decoded = json.loads(raw_manifest.decode("utf-8"))
            if not isinstance(decoded, dict) or set(decoded) != {"client_id", "payload_hash", "payload_size"}:
                raise ValueError("manifest fields are invalid")
            payload_hash_value = decoded["payload_hash"]
            payload_size_value = decoded["payload_size"]
            if (
                type(payload_hash_value) is not str
                or _HASH.fullmatch(payload_hash_value) is None
                or type(payload_size_value) is not int
                or payload_size_value <= 0
            ):
                raise ValueError("manifest hash/size types are invalid")
            payload_hash = payload_hash_value
            payload_size = payload_size_value
            consumer_payload = (self.namespace / "payloads" / f"{payload_hash}.blob").read_bytes()
            if len(consumer_payload) != payload_size or sha256_hex(consumer_payload) != payload_hash:
                raise ValueError("consumer payload hash/size verification failed")
            checksum_valid = True
            consumer_status = "verified"
        except (ArtifactCorruption, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            partial_visible = target.exists()
            consumer_status = "partial_or_invalid_payload_visible" if partial_visible else "manifest_unavailable"

        qualified = (
            winner_count == 1
            and not partial_visible
            and checksum_valid
            and loser_behavior == "file_exists_no_replace"
            and 0 < visibility_tick <= self.config.visibility_deadline_ticks
        )
        evidence = self.store.put(
            "CfsCapabilityProbeEvidence",
            "1.0.0",
            {
                "backend_id": self.config.backend_id,
                "candidate_payloads": candidate_payloads,
                "client_results": client_results,
                "checksum_valid": checksum_valid,
                "cleanup_status": cleanup_status,
                "consumer_status": consumer_status,
                "dedicated_namespace": str(self.namespace.relative_to(self.root)),
                "directory_listing_used": False,
                "fixture_fault_injection": self.config.fault,
                "loser_behavior": loser_behavior,
                "partial_payload_visible": partial_visible,
                "payload_hash": payload_hash,
                "payload_size": payload_size,
                "status": "qualified" if qualified else "blocked",
                "visibility_deadline_ticks": self.config.visibility_deadline_ticks,
                "visibility_tick": visibility_tick,
                "winner_count": winner_count,
            },
        )
        ArtifactStore._publish(evidence_ref, f"{evidence.content_hash}\n".encode("ascii"))
        return evidence

    def _sync_namespace(self) -> None:
        descriptor = os.open(self.namespace, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class FenceAuthority:
    """External current-epoch authority persisted independently of a reward slot."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.store = ArtifactStore(root)
        self.ref = self.root / "fence-authority" / "current.ref"

    def advance(self, epoch: int) -> Artifact:
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("resolver epoch is invalid")
        if self.ref.exists():
            current = self.current()
            if cast(int, current.payload["epoch"]) >= epoch:
                raise RewardRoundtripError("fence epoch must advance monotonically")
        artifact = self.store.put("FenceEpoch", "1.0.0", {"epoch": epoch, "status": "current"})
        ArtifactStore.durable_mkdir(self.ref.parent)
        temporary = self.ref.parent / f".current-{epoch}.tmp"
        with temporary.open("wb") as stream:
            stream.write(f"{artifact.content_hash}\n".encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.ref)
        return artifact

    def current(self) -> Artifact:
        if not self.ref.exists():
            raise RewardRoundtripError("fence authority has no current epoch")
        current = self.store.read(self.ref.read_text(encoding="ascii").strip(), expected_schema_name="FenceEpoch")
        if (
            set(current.payload) != {"epoch", "status"}
            or type(current.payload.get("epoch")) is not int
            or cast(int, current.payload.get("epoch")) <= 0
            or current.payload.get("status") != "current"
        ):
            raise RewardIntegrityError("fence authority current epoch is invalid")
        return current


class RewardRoundtripWorkflow:
    @staticmethod
    def production_readiness(root: str | Path, config: ProductionCfsRewardConfig) -> Artifact:
        checks = []
        for code, value in (
            ("CFS_PUBLISH_IF_ABSENT_UNAVAILABLE", config.cfs_approval_hash),
            ("FENCE_AUTHORITY_UNAVAILABLE", config.fence_authority_approval_hash),
            ("REWARD_PROVIDER_UNAVAILABLE", config.reward_provider_approval_hash),
            ("REWARD_ATTEMPT_POLICY_UNAVAILABLE", config.attempt_policy_approval_hash),
        ):
            if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
                checks.append({"code": code, "status": "blocked"})
        if not checks:
            checks.append({"code": "PRODUCTION_REWARD_ROUNDTRIP_NOT_CONFIGURED", "status": "blocked"})
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": checks,
                "execution_profile": "production",
                "phase": "TRAIN_35B",
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )

    @classmethod
    def start(
        cls,
        root: str | Path,
        *,
        key: RewardSlotKey,
        probe_evidence_hash: str,
        trajectory_payload: dict[str, JsonValue],
        reward_schema_hash: str,
        scalarizer_hash: str,
    ) -> RewardSnapshot:
        store = ArtifactStore(root)
        probe = store.read(probe_evidence_hash, expected_schema_name="CfsCapabilityProbeEvidence")
        cls._validate_probe(probe)
        if probe.payload.get("status") != "qualified":
            blocked = store.put(
                "RewardRoundtripBlockedEvidence",
                "1.0.0",
                {
                    "probe_evidence_hash": probe.content_hash,
                    "reason_code": "CFS_BACKEND_UNQUALIFIED",
                    "status": "blocked",
                },
            )
            raise RewardRoundtripError(f"CFS_BACKEND_UNQUALIFIED:{blocked.content_hash}")
        cls._contracts(store, reward_schema_hash, scalarizer_hash)
        attempt_policy = store.put(
            "RewardAttemptPolicy",
            "1.0.0",
            {
                "lease_ticks": _ATTEMPT_LEASE_TICKS,
                "max_attempts": _MAX_ATTEMPTS,
                "retry_after": "terminal_result_or_lease_expired",
                "schema_version": "reward-attempt-policy/1.0.0",
            },
        )
        decision = store.put(
            "DecisionRecord",
            "1.0.0",
            {
                "attempt_policy_hash": attempt_policy.content_hash,
                "decision": "ADOPT_FIXTURE_REWARD_ATTEMPT_POLICY_V1",
                "rationale": "Ticket10 fixture freezes max-attempt and lease defaults; production remains blocked.",
                "scope": "fixture_only",
            },
        )
        slot = Path(root) / "reward-slots" / key.content_hash
        ArtifactStore.durable_mkdir(slot)
        trajectory = store.put(
            "TrajectoryManifest",
            "1.0.0",
            {"slot_key": key.payload(), "slot_key_hash": key.content_hash, "trajectory": trajectory_payload},
        )
        cls._publish_ref(store, slot / "trajectory.ref", trajectory, "TRAJECTORY_STABLE_KEY_CONFLICT")
        request = store.put(
            "RewardRequest",
            "1.0.0",
            {
                "attempt_policy_hash": attempt_policy.content_hash,
                "decision_record_hash": decision.content_hash,
                "max_attempts": _MAX_ATTEMPTS,
                "reward_schema_hash": reward_schema_hash,
                "scalarizer_hash": scalarizer_hash,
                "slot_key": key.payload(),
                "slot_key_hash": key.content_hash,
                "trajectory_manifest_hash": trajectory.content_hash,
            },
        )
        cls._publish_ref(store, slot / "request.ref", request, "REWARD_REQUEST_STABLE_KEY_CONFLICT")
        return cls.resume(root, key=key)

    @classmethod
    def claim_attempt(
        cls, root: str | Path, *, key: RewardSlotKey, resolver_epoch: int, clock_tick: int, lease_ticks: int = 4
    ) -> Artifact:
        if (
            type(resolver_epoch) is not int
            or resolver_epoch <= 0
            or type(clock_tick) is not int
            or clock_tick < 0
            or lease_ticks != _ATTEMPT_LEASE_TICKS
        ):
            raise RewardRoundtripError("attempt lease policy is invalid")
        snapshot = cls.resume(root, key=key)
        store = ArtifactStore(root)
        slot = Path(root) / "reward-slots" / key.content_hash
        ordinals = snapshot.attempt_ordinals
        if len(ordinals) >= _MAX_ATTEMPTS:
            raise RewardRoundtripError("REWARD_ATTEMPTS_EXHAUSTED")
        if ordinals:
            previous = cls._attempt(
                store,
                slot,
                ordinals[-1],
                attempt_policy_hash=cast(str, snapshot.reward_request.payload["attempt_policy_hash"]),
                request_hash=snapshot.reward_request.content_hash,
                slot_key_hash=key.content_hash,
            )
            result_ref = slot / "attempts" / str(ordinals[-1]) / "result.ref"
            if not result_ref.exists() and clock_tick <= cast(int, previous.payload["lease_expires_tick"]):
                raise RewardRoundtripError("PREVIOUS_ATTEMPT_NOT_TERMINAL_OR_EXPIRED")
        ordinal = len(ordinals) + 1
        attempt = store.put(
            "RewardAttemptLease",
            "1.0.0",
            {
                "attempt_ordinal": ordinal,
                "lease_expires_tick": clock_tick + lease_ticks,
                "lease_start_tick": clock_tick,
                "max_attempts": _MAX_ATTEMPTS,
                "attempt_policy_hash": snapshot.reward_request.payload["attempt_policy_hash"],
                "request_hash": snapshot.reward_request.content_hash,
                "resolver_epoch": resolver_epoch,
                "slot_key_hash": key.content_hash,
                "status": "leased",
            },
        )
        attempt_ref = slot / "attempts" / str(ordinal) / "attempt.ref"
        cls._publish_ref(store, attempt_ref, attempt, "ATTEMPT_LEASE_CONFLICT")
        return attempt

    @classmethod
    def commit_result(
        cls, root: str | Path, *, key: RewardSlotKey, attempt_ordinal: int, result_payload: dict[str, JsonValue]
    ) -> Artifact:
        snapshot = cls.resume(root, key=key)
        if snapshot.resolved_reward is not None:
            submitted_result_hash = sha256_hex(
                canonical_json_bytes(
                    {
                        "attempt_ordinal": attempt_ordinal,
                        "result_payload": result_payload,
                        "slot_key_hash": key.content_hash,
                    }
                )
            )
            quarantine = ArtifactStore(root).put(
                "LateRewardResultQuarantine",
                "1.0.0",
                {
                    "attempt_ordinal": attempt_ordinal,
                    "reason_code": "RESULT_AFTER_RESOLUTION",
                    "slot_key_hash": key.content_hash,
                    "submitted_result_hash": submitted_result_hash,
                },
            )
            raise RewardRoundtripError(f"LATE_RESULT_QUARANTINED:{quarantine.content_hash}")
        store = ArtifactStore(root)
        slot = Path(root) / "reward-slots" / key.content_hash
        attempt = cls._attempt(
            store,
            slot,
            attempt_ordinal,
            attempt_policy_hash=cast(str, snapshot.reward_request.payload["attempt_policy_hash"]),
            request_hash=snapshot.reward_request.content_hash,
            slot_key_hash=key.content_hash,
        )
        result = store.put(
            "RewardAttemptResult",
            "1.0.0",
            {
                **result_payload,
                "attempt_ordinal": attempt_ordinal,
                "attempt_hash": attempt.content_hash,
                "request_hash": snapshot.reward_request.content_hash,
                "slot_key_hash": key.content_hash,
            },
        )
        cls._publish_ref(
            store,
            slot / "attempts" / str(attempt_ordinal) / "result.ref",
            result,
            "ATTEMPT_RESULT_HASH_CONFLICT",
        )
        return result

    @classmethod
    def resolve(cls, root: str | Path, *, key: RewardSlotKey, resolver_epoch: int) -> Artifact:
        snapshot = cls.resume(root, key=key)
        if snapshot.resolved_reward is not None:
            return snapshot.resolved_reward
        store = ArtifactStore(root)
        current = FenceAuthority(root).current()
        if current.payload.get("epoch") != resolver_epoch:
            raise RewardRoundtripError("STALE_RESOLVER_EPOCH")
        slot = Path(root) / "reward-slots" / key.content_hash
        eligible: list[tuple[int, Artifact]] = []
        for ordinal in snapshot.attempt_ordinals:
            ref = slot / "attempts" / str(ordinal) / "result.ref"
            if ref.exists():
                attempt = cls._attempt(
                    store,
                    slot,
                    ordinal,
                    attempt_policy_hash=cast(str, snapshot.reward_request.payload["attempt_policy_hash"]),
                    request_hash=snapshot.reward_request.content_hash,
                    slot_key_hash=key.content_hash,
                )
                result = store.read(ref.read_text(encoding="ascii").strip(), expected_schema_name="RewardAttemptResult")
                if result.payload.get("attempt_hash") != attempt.content_hash:
                    raise RewardIntegrityError("attempt result does not echo its attempt hash")
                if result.payload.get("attempt_ordinal") != ordinal:
                    raise RewardIntegrityError("attempt result ordinal changed")
                try:
                    cls._validate_result(store, snapshot.reward_request, result)
                except RewardRoundtripError:
                    continue
                eligible.append((ordinal, result))
        if not eligible:
            raise RewardRoundtripError("NO_ELIGIBLE_VALID_ATTEMPT_RESULT")
        ordinal, result = min(eligible, key=lambda item: item[0])
        resolved = store.put(
            "ResolvedReward",
            "1.0.0",
            {
                "attempt_ordinal": ordinal,
                "confidence_basis_points": result.payload["confidence_basis_points"],
                "request_hash": snapshot.reward_request.content_hash,
                "resolver_epoch": resolver_epoch,
                "reward_micros": result.payload["reward_micros"],
                "result_hash": result.content_hash,
                "slot_key_hash": key.content_hash,
                "status": "resolved",
            },
        )
        cls._publish_ref(store, slot / "resolved.ref", resolved, "RESOLVED_REWARD_HASH_CONFLICT")
        return resolved

    @classmethod
    def resume(cls, root: str | Path, *, key: RewardSlotKey) -> RewardSnapshot:
        store = ArtifactStore(root)
        slot = Path(root) / "reward-slots" / key.content_hash
        try:
            trajectory = store.read(
                (slot / "trajectory.ref").read_text().strip(), expected_schema_name="TrajectoryManifest"
            )
            request = store.read((slot / "request.ref").read_text().strip(), expected_schema_name="RewardRequest")
        except (ArtifactCorruption, OSError) as error:
            raise RewardRoundtripError("committed reward slot cannot be recovered") from error
        if (
            set(trajectory.payload) != {"slot_key", "slot_key_hash", "trajectory"}
            or trajectory.payload.get("slot_key") != key.payload()
            or trajectory.payload.get("slot_key_hash") != key.content_hash
            or set(request.payload)
            != {
                "attempt_policy_hash",
                "decision_record_hash",
                "max_attempts",
                "reward_schema_hash",
                "scalarizer_hash",
                "slot_key",
                "slot_key_hash",
                "trajectory_manifest_hash",
            }
            or request.payload.get("slot_key") != key.payload()
            or request.payload.get("slot_key_hash") != key.content_hash
            or request.payload.get("trajectory_manifest_hash") != trajectory.content_hash
            or request.payload.get("max_attempts") != _MAX_ATTEMPTS
        ):
            raise RewardIntegrityError("reward slot lineage changed")
        attempt_policy = store.read(
            cast(str, request.payload["attempt_policy_hash"]), expected_schema_name="RewardAttemptPolicy"
        )
        decision = store.read(cast(str, request.payload["decision_record_hash"]), expected_schema_name="DecisionRecord")
        if (
            set(attempt_policy.payload) != {"lease_ticks", "max_attempts", "retry_after", "schema_version"}
            or attempt_policy.payload.get("lease_ticks") != _ATTEMPT_LEASE_TICKS
            or attempt_policy.payload.get("max_attempts") != _MAX_ATTEMPTS
            or attempt_policy.payload.get("retry_after") != "terminal_result_or_lease_expired"
            or attempt_policy.payload.get("schema_version") != "reward-attempt-policy/1.0.0"
            or set(decision.payload) != {"attempt_policy_hash", "decision", "rationale", "scope"}
            or decision.payload.get("attempt_policy_hash") != attempt_policy.content_hash
            or decision.payload.get("decision") != "ADOPT_FIXTURE_REWARD_ATTEMPT_POLICY_V1"
            or decision.payload.get("scope") != "fixture_only"
        ):
            raise RewardIntegrityError("reward attempt policy lineage changed")
        cls._contracts(
            store,
            cast(str, request.payload["reward_schema_hash"]),
            cast(str, request.payload["scalarizer_hash"]),
        )
        ordinals: list[int] = []
        missing_seen = False
        for ordinal in range(1, _MAX_ATTEMPTS + 1):
            attempt_ref = slot / "attempts" / str(ordinal) / "attempt.ref"
            result_ref = slot / "attempts" / str(ordinal) / "result.ref"
            if not attempt_ref.exists():
                missing_seen = True
                if result_ref.exists():
                    raise RewardIntegrityError("attempt result exists without its lease")
                continue
            if missing_seen:
                raise RewardIntegrityError("attempt ordinals are not contiguous")
            ordinals.append(ordinal)
            cls._attempt(
                store,
                slot,
                ordinal,
                attempt_policy_hash=attempt_policy.content_hash,
                request_hash=request.content_hash,
                slot_key_hash=key.content_hash,
            )
        resolved: Artifact | None = None
        if (slot / "resolved.ref").exists():
            resolved = store.read((slot / "resolved.ref").read_text().strip(), expected_schema_name="ResolvedReward")
            if (
                set(resolved.payload)
                != {
                    "attempt_ordinal",
                    "confidence_basis_points",
                    "request_hash",
                    "resolver_epoch",
                    "reward_micros",
                    "result_hash",
                    "slot_key_hash",
                    "status",
                }
                or resolved.payload.get("request_hash") != request.content_hash
                or resolved.payload.get("slot_key_hash") != key.content_hash
                or resolved.payload.get("reward_micros") is None
                or resolved.payload.get("status") != "resolved"
            ):
                raise RewardIntegrityError("resolved reward lineage is invalid")
            resolved_ordinal = resolved.payload.get("attempt_ordinal")
            result_hash = resolved.payload.get("result_hash")
            if (
                type(resolved_ordinal) is not int
                or cast(int, resolved_ordinal) not in ordinals
                or type(result_hash) is not str
            ):
                raise RewardIntegrityError("resolved reward attempt identity is invalid")
            result_ref = slot / "attempts" / str(resolved_ordinal) / "result.ref"
            try:
                committed_hash = result_ref.read_text(encoding="ascii").strip()
                result = store.read(committed_hash, expected_schema_name="RewardAttemptResult")
            except (ArtifactCorruption, OSError) as error:
                raise RewardIntegrityError("resolved reward result is unavailable") from error
            attempt = cls._attempt(
                store,
                slot,
                cast(int, resolved_ordinal),
                attempt_policy_hash=attempt_policy.content_hash,
                request_hash=request.content_hash,
                slot_key_hash=key.content_hash,
            )
            if (
                committed_hash != result_hash
                or result.payload.get("attempt_hash") != attempt.content_hash
                or result.payload.get("attempt_ordinal") != resolved_ordinal
                or result.payload.get("reward_micros") != resolved.payload.get("reward_micros")
                or result.payload.get("confidence_basis_points") != resolved.payload.get("confidence_basis_points")
            ):
                raise RewardIntegrityError("resolved reward result lineage changed")
            cls._validate_result(store, request, result)
        return RewardSnapshot(trajectory, request, resolved, tuple(ordinals))

    @staticmethod
    def _validate_probe(probe: Artifact) -> None:
        expected = {
            "backend_id",
            "candidate_payloads",
            "checksum_valid",
            "cleanup_status",
            "client_results",
            "consumer_status",
            "dedicated_namespace",
            "directory_listing_used",
            "fixture_fault_injection",
            "loser_behavior",
            "partial_payload_visible",
            "payload_hash",
            "payload_size",
            "status",
            "visibility_deadline_ticks",
            "visibility_tick",
            "winner_count",
        }
        payload = probe.payload
        if set(payload) != expected or payload.get("status") not in {"qualified", "blocked"}:
            raise RewardIntegrityError("CFS probe evidence fields are invalid")
        if payload.get("status") == "blocked":
            return
        clients = payload.get("client_results")
        candidates = payload.get("candidate_payloads")
        if not isinstance(clients, list) or not isinstance(candidates, list):
            raise RewardIntegrityError("CFS qualified probe evidence is malformed")
        if len(clients) != 2 or len(candidates) != 2:
            raise RewardIntegrityError("CFS probe did not execute exactly two clients")
        if not all(isinstance(item, dict) for item in clients + candidates):
            raise RewardIntegrityError("CFS probe client evidence is malformed")
        client_rows = cast(list[dict[str, JsonValue]], clients)
        candidate_rows = cast(list[dict[str, JsonValue]], candidates)
        statuses = [item.get("publication_status") for item in client_rows]
        mount_views = [item.get("mount_view_id") for item in client_rows]
        observed_candidate = any(
            item.get("payload_hash") == payload.get("payload_hash")
            and item.get("payload_size") == payload.get("payload_size")
            for item in candidate_rows
        )
        visibility_tick = payload.get("visibility_tick")
        visibility_deadline = payload.get("visibility_deadline_ticks")
        backend_id = payload.get("backend_id")
        candidate_rows_valid = all(
            set(item) == {"client_id", "payload_hash", "payload_size"}
            and type(item.get("client_id")) is str
            and type(item.get("payload_hash")) is str
            and _HASH.fullmatch(cast(str, item.get("payload_hash"))) is not None
            and type(item.get("payload_size")) is int
            and cast(int, item.get("payload_size")) > 0
            for item in candidate_rows
        )
        client_rows_valid = all(
            set(item)
            == {
                "cleanup_status",
                "client_id",
                "mount_view_id",
                "publication_status",
                "retained_after_cleanup",
            }
            and type(item.get("client_id")) is str
            and item.get("cleanup_status") in {"failed", "not_required", "succeeded"}
            and type(item.get("retained_after_cleanup")) is bool
            for item in client_rows
        )
        if (
            type(backend_id) is not str
            or _ID.fullmatch(cast(str, backend_id)) is None
            or payload.get("dedicated_namespace") != f"cfs-probes/{backend_id}"
            or payload.get("fixture_fault_injection") not in {"valid", "cleanup_failure"}
            or payload.get("cleanup_status") not in {"failed", "succeeded"}
            or not candidate_rows_valid
            or not client_rows_valid
            or any(type(status) is not str for status in statuses)
            or sorted(cast(list[str], statuses)) != ["file_exists_no_replace", "published_no_replace"]
            or any(type(view) is not str for view in mount_views)
            or len(set(cast(list[str], mount_views))) != 2
            or payload.get("winner_count") != 1
            or payload.get("loser_behavior") != "file_exists_no_replace"
            or payload.get("partial_payload_visible") is not False
            or payload.get("checksum_valid") is not True
            or payload.get("consumer_status") != "verified"
            or payload.get("directory_listing_used") is not False
            or not observed_candidate
            or type(visibility_tick) is not int
            or type(visibility_deadline) is not int
            or not 0 < cast(int, visibility_tick) <= cast(int, visibility_deadline)
        ):
            raise RewardIntegrityError("CFS qualified probe did not prove required capabilities")

    @staticmethod
    def _contracts(store: ArtifactStore, reward_hash: str, scalarizer_hash: str) -> None:
        reward = store.read(reward_hash, expected_schema_name="RewardSchema")
        scalarizer = store.read(scalarizer_hash, expected_schema_name="Scalarizer")
        reward_expected = {
            "aggregation",
            "maximum_micros",
            "minimum_micros",
            "schema_id",
            "schema_version",
        }
        scalarizer_expected = {"dimension_weights_micros", "scalarizer_id", "schema_version"}
        weights = scalarizer.payload.get("dimension_weights_micros")
        expected_dimensions = {"correctness", "reasoning_quality", "task_completion", "tool_discipline"}
        if (
            reward.schema_version != "1.0.0"
            or set(reward.payload) != reward_expected
            or reward.payload.get("schema_version") != "reward-schema/1.0.0"
            or reward.payload.get("aggregation") != "calibrated_scalar"
            or reward.payload.get("minimum_micros") != 0
            or reward.payload.get("maximum_micros") != 100_000_000
            or type(reward.payload.get("schema_id")) is not str
            or scalarizer.schema_version != "1.0.0"
            or set(scalarizer.payload) != scalarizer_expected
            or scalarizer.payload.get("schema_version") != "scalarizer/1.0.0"
            or type(scalarizer.payload.get("scalarizer_id")) is not str
            or not isinstance(weights, dict)
            or set(weights) != expected_dimensions
            or any(type(value) is not int or cast(int, value) <= 0 for value in weights.values())
            or sum(cast(int, value) for value in weights.values()) != 1_000_000
        ):
            raise RewardRoundtripError("reward contracts are incompatible")

    @classmethod
    def _validate_result(cls, store: ArtifactStore, request: Artifact, result: Artifact) -> None:
        expected = {
            "attempt_hash",
            "attempt_ordinal",
            "confidence_basis_points",
            "failure_tags",
            "request_hash",
            "reward_micros",
            "reward_schema_hash",
            "scalarizer_hash",
            "slot_key_hash",
            "turn_local_tie_groups",
        }
        confidence = result.payload.get("confidence_basis_points")
        reward = result.payload.get("reward_micros")
        ties = result.payload.get("turn_local_tie_groups")
        failure_tags = result.payload.get("failure_tags")
        flattened_ties: list[str] = []
        if isinstance(ties, list):
            for group in ties:
                if isinstance(group, list) and all(type(member) is str for member in group):
                    flattened_ties.extend(cast(list[str], group))
        if (
            set(result.payload) != expected
            or result.payload.get("request_hash") != request.content_hash
            or result.payload.get("slot_key_hash") != request.payload.get("slot_key_hash")
            or result.payload.get("reward_schema_hash") != request.payload.get("reward_schema_hash")
            or result.payload.get("scalarizer_hash") != request.payload.get("scalarizer_hash")
            or type(result.payload.get("attempt_ordinal")) is not int
            or cast(int, result.payload.get("attempt_ordinal")) <= 0
            or type(result.payload.get("attempt_hash")) is not str
            or _HASH.fullmatch(cast(str, result.payload.get("attempt_hash"))) is None
            or type(reward) is not int
            or not 0 <= cast(int, reward) <= 100_000_000
            or type(confidence) is not int
            or not 0 <= cast(int, confidence) <= 10_000
            or not isinstance(failure_tags, list)
            or any(type(tag) is not str or _ID.fullmatch(cast(str, tag)) is None for tag in failure_tags)
            or not isinstance(ties, list)
            or not ties
            or any(
                not isinstance(group, list)
                or not group
                or any(type(member) is not str or _ID.fullmatch(cast(str, member)) is None for member in group)
                or len(group) != len(set(cast(list[str], group)))
                for group in ties
            )
            or len(flattened_ties) != len(set(flattened_ties))
        ):
            raise RewardRoundtripError("attempt result is invalid")
        cls._contracts(
            store, cast(str, request.payload["reward_schema_hash"]), cast(str, request.payload["scalarizer_hash"])
        )

    @staticmethod
    def _attempt(
        store: ArtifactStore,
        slot: Path,
        ordinal: int,
        *,
        attempt_policy_hash: str,
        request_hash: str,
        slot_key_hash: str,
    ) -> Artifact:
        ref = slot / "attempts" / str(ordinal) / "attempt.ref"
        try:
            attempt = store.read(ref.read_text().strip(), expected_schema_name="RewardAttemptLease")
        except (ArtifactCorruption, OSError) as error:
            raise RewardRoundtripError("attempt lease is unavailable") from error
        expected = {
            "attempt_policy_hash",
            "attempt_ordinal",
            "lease_expires_tick",
            "lease_start_tick",
            "max_attempts",
            "request_hash",
            "resolver_epoch",
            "slot_key_hash",
            "status",
        }
        lease_start = attempt.payload.get("lease_start_tick")
        lease_expires = attempt.payload.get("lease_expires_tick")
        if (
            set(attempt.payload) != expected
            or attempt.payload.get("attempt_ordinal") != ordinal
            or attempt.payload.get("request_hash") != request_hash
            or attempt.payload.get("slot_key_hash") != slot_key_hash
            or attempt.payload.get("max_attempts") != _MAX_ATTEMPTS
            or attempt.payload.get("attempt_policy_hash") != attempt_policy_hash
            or attempt.payload.get("status") != "leased"
            or type(attempt.payload.get("resolver_epoch")) is not int
            or cast(int, attempt.payload.get("resolver_epoch")) <= 0
            or type(lease_start) is not int
            or type(lease_expires) is not int
            or cast(int, lease_expires) - cast(int, lease_start) != _ATTEMPT_LEASE_TICKS
        ):
            raise RewardIntegrityError("attempt lease lineage or policy changed")
        return attempt

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
            raise RewardIntegrityError(f"{code}:existing ref cannot be verified") from error
        if existing == artifact.content_hash:
            return
        conflict = store.put(
            "RewardIntegrityConflict",
            "1.0.0",
            {
                "conflicting_hash": artifact.content_hash,
                "existing_hash": existing,
                "reason_code": code,
                "status": "corruption",
            },
        )
        raise RewardIntegrityError(f"{code}:{conflict.content_hash}")
