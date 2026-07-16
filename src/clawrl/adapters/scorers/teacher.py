"""Private ingress and strict normalization for isolated Sol TeacherScorer output."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from clawrl.artifacts import Artifact, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex

_HASH = re.compile(r"^[0-9a-f]{64}$")
_LINEAGE = "/root/ticket04_teacher_scorer_retry"
_TOP_FIELDS = {
    "input_packet_content_hash",
    "input_payload_canonical_hash",
    "labels",
    "packet_id",
    "schema_version",
    "scoring_session_id",
    "seed",
    "thread_sessions",
    "turns",
}
_LABEL_FIELDS = {
    "dimension_scores",
    "evidence",
    "failure_tags",
    "input_content_hash",
    "item_index",
    "prompt_hash",
    "response_hash",
    "scalar_micros",
    "thread_id",
    "trajectory_id",
    "turn_index",
    "wave_index",
}
_THREAD_FIELDS = {"scoring_session_id", "thread_id", "turn_indices"}
_TURN_FIELDS = {"items", "thread_id", "turn_index", "wave_index"}
_TURN_ITEM_FIELDS = {"content_hash", "prompt_hash", "response_hash", "trajectory_id"}
_TAG_FIELDS = {"dimension", "severity", "type"}
_DIMENSIONS = {"correctness", "reasoning_quality", "task_completion", "tool_discipline"}


class TeacherOutputError(ValueError):
    """TeacherScorer raw output or private ingress state is invalid."""


def teacher_output_schema_contract() -> dict[str, JsonValue]:
    """Return the exact closed-world output shape enforced by TeacherScorerIngress."""

    return cast(
        dict[str, JsonValue],
        {
            "dimension_fields": sorted(_DIMENSIONS),
            "failure_tag_required": sorted(_TAG_FIELDS),
            "label_required": sorted(_LABEL_FIELDS),
            "required": sorted(_TOP_FIELDS),
            "schema_version": "teacher-scorer-output/1.0.0",
            "thread_session_required": sorted(_THREAD_FIELDS),
            "turn_item_required": sorted(_TURN_ITEM_FIELDS),
            "turn_required": sorted(_TURN_FIELDS),
        },
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object key")
        result[key] = value
    return result


def _bounded_json(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes or not 1 <= len(raw) <= 1_048_576:
        raise TeacherOutputError("TeacherScorer raw output byte size is invalid")
    try:
        decoded = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise TeacherOutputError("TeacherScorer raw output is not strict UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise TeacherOutputError("TeacherScorer raw output must be an object")
    pending: list[tuple[object, int]] = [(decoded, 0)]
    nodes = 0
    strings = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > 20_000 or depth > 32:
            raise TeacherOutputError("TeacherScorer raw output exceeds structural limits")
        if isinstance(value, dict):
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
        elif isinstance(value, str):
            strings += len(value)
            if len(value) > 16_384 or strings > 262_144:
                raise TeacherOutputError("TeacherScorer raw output string limits are exceeded")
        elif value is not None and type(value) not in {bool, int}:
            raise TeacherOutputError("TeacherScorer raw output contains an unsupported value")
    return cast(dict[str, object], decoded)


@dataclass(frozen=True, slots=True)
class TeacherRoleIngressReceipt:
    """Hash-only public handle to exact private TeacherScorer bytes."""

    ingress_hash: str
    input_packet_hash: str
    raw_output_hash: str
    normalized_output_hash: str
    role_invocation_audit_hash: str

    def __post_init__(self) -> None:
        for value in (
            self.ingress_hash,
            self.input_packet_hash,
            self.raw_output_hash,
            self.normalized_output_hash,
            self.role_invocation_audit_hash,
        ):
            if type(value) is not str or _HASH.fullmatch(value) is None:
                raise TeacherOutputError("TeacherScorer ingress receipt contains an invalid hash")

    def artifact_payload(self) -> dict[str, str]:
        return {
            "ingress_hash": self.ingress_hash,
            "input_packet_hash": self.input_packet_hash,
            "normalized_output_hash": self.normalized_output_hash,
            "raw_output_hash": self.raw_output_hash,
            "role_invocation_audit_hash": self.role_invocation_audit_hash,
        }


@dataclass(frozen=True, slots=True)
class TeacherPublicContract:
    normalized_output: Artifact
    role_invocation_audit: Artifact


class TeacherScorerIngress:
    """Normalize one isolated TeacherScorer exchange behind a private boundary."""

    @classmethod
    def validate_input_packet(cls, root: str | Path, packet_hash: str) -> Artifact:
        """Apply the actual TeacherScorer ingress contract to a published packet."""

        return cls._read_packet(ArtifactStore(root), packet_hash)

    @classmethod
    def stage_private_exchange(
        cls,
        root: str | Path,
        *,
        input_packet_hash: str,
        raw_role_output: bytes,
        role_session_lineage: str,
    ) -> TeacherRoleIngressReceipt:
        store = ArtifactStore(root)
        packet = cls._read_packet(store, input_packet_hash)
        normalized = cls._normalize(packet, raw_role_output, role_session_lineage)
        normalized_bytes = canonical_json_bytes(normalized)
        normalized_artifact = store.put("NormalizedTeacherScorerOutput", "1.0.0", normalized)
        raw_hash = sha256_hex(raw_role_output)
        audit = store.put(
            "RoleInvocationAudit",
            "1.0.0",
            {
                "input_hash": input_packet_hash,
                "input_packet_hash": input_packet_hash,
                "normalized_output_hash": normalized_artifact.content_hash,
                "normalized_output_payload_hash": sha256_hex(normalized_bytes),
                "normalized_output_size": len(normalized_bytes),
                "output_hash": raw_hash,
                "packet_id": packet.payload["packet_id"],
                "raw_output_hash": raw_hash,
                "raw_output_size": len(raw_role_output),
                "role_session_lineage": role_session_lineage,
                "role_type": "TeacherScorer",
                "seed": packet.payload["seed"],
            },
        )
        identity = {
            "input_packet_hash": input_packet_hash,
            "normalized_output_hash": normalized_artifact.content_hash,
            "raw_output_hash": raw_hash,
            "role_invocation_audit_hash": audit.content_hash,
        }
        ingress_hash = sha256_hex(
            canonical_json_bytes({"domain": "teacher-scorer-private-ingress/1.0.0", "identity": identity})
        )
        receipt = TeacherRoleIngressReceipt(ingress_hash=ingress_hash, **identity)
        private_root = Path(root) / "private-boundary" / "teacher-scorer-ingress" / ingress_hash
        ArtifactStore.durable_mkdir(private_root)
        ArtifactStore._publish(private_root / "role-output.raw.json", raw_role_output)
        ArtifactStore._publish(private_root / "role-output.normalized.canonical.json", normalized_bytes)
        ArtifactStore._publish(private_root / "role-input.canonical.json", canonical_json_bytes(packet.payload))
        ArtifactStore._publish(
            private_root / "ingress-receipt.canonical.json",
            canonical_json_bytes(
                {
                    **receipt.artifact_payload(),
                    "normalized_output_size": len(normalized_bytes),
                    "raw_output_size": len(raw_role_output),
                    "role_session_lineage": role_session_lineage,
                    "schema_version": "teacher-scorer-ingress/1.0.0",
                }
            ),
        )
        return receipt

    @classmethod
    def load_public_contract(
        cls,
        root: str | Path,
        receipt: TeacherRoleIngressReceipt,
    ) -> TeacherPublicContract:
        """Reparse private bytes, then expose only already-normalized public artifacts."""

        if type(receipt) is not TeacherRoleIngressReceipt:
            raise TeacherOutputError("opaque TeacherScorer ingress receipt is required")
        store = ArtifactStore(root)
        packet = cls._read_packet(store, receipt.input_packet_hash)
        private_root = Path(root) / "private-boundary" / "teacher-scorer-ingress" / receipt.ingress_hash
        try:
            raw = (private_root / "role-output.raw.json").read_bytes()
            normalized_bytes = (private_root / "role-output.normalized.canonical.json").read_bytes()
            input_bytes = (private_root / "role-input.canonical.json").read_bytes()
            manifest_bytes = (private_root / "ingress-receipt.canonical.json").read_bytes()
            manifest = json.loads(manifest_bytes, object_pairs_hook=_strict_object)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise TeacherOutputError("TeacherScorer private ingress state is unavailable") from error
        expected_manifest = {
            **receipt.artifact_payload(),
            "normalized_output_size": len(normalized_bytes),
            "raw_output_size": len(raw),
            "role_session_lineage": _LINEAGE,
            "schema_version": "teacher-scorer-ingress/1.0.0",
        }
        if (
            not isinstance(manifest, dict)
            or canonical_json_bytes(manifest) != manifest_bytes
            or manifest != expected_manifest
            or input_bytes != canonical_json_bytes(packet.payload)
            or sha256_hex(raw) != receipt.raw_output_hash
        ):
            raise TeacherOutputError("TeacherScorer private ingress state is corrupt")
        normalized = cls._normalize(packet, raw, _LINEAGE)
        if canonical_json_bytes(normalized) != normalized_bytes:
            raise TeacherOutputError("TeacherScorer normalization cannot be reproduced")
        normalized_artifact = store.read(
            receipt.normalized_output_hash,
            expected_schema_name="NormalizedTeacherScorerOutput",
        )
        audit = store.read(receipt.role_invocation_audit_hash, expected_schema_name="RoleInvocationAudit")
        if (
            normalized_artifact.schema_version != "1.0.0"
            or normalized_artifact.payload != normalized
            or audit.schema_version != "1.0.0"
            or audit.payload
            != {
                "input_hash": receipt.input_packet_hash,
                "input_packet_hash": receipt.input_packet_hash,
                "normalized_output_hash": receipt.normalized_output_hash,
                "normalized_output_payload_hash": sha256_hex(normalized_bytes),
                "normalized_output_size": len(normalized_bytes),
                "output_hash": receipt.raw_output_hash,
                "packet_id": packet.payload["packet_id"],
                "raw_output_hash": receipt.raw_output_hash,
                "raw_output_size": len(raw),
                "role_session_lineage": _LINEAGE,
                "role_type": "TeacherScorer",
                "seed": packet.payload["seed"],
            }
        ):
            raise TeacherOutputError("TeacherScorer public audit artifacts are invalid")
        return TeacherPublicContract(normalized_artifact, audit)

    @staticmethod
    def _read_packet(store: ArtifactStore, packet_hash: str) -> Artifact:
        if type(packet_hash) is not str or _HASH.fullmatch(packet_hash) is None:
            raise TeacherOutputError("TeacherScorer input packet hash is invalid")
        packet = store.read(packet_hash, expected_schema_name="TeacherScorerInputPacket")
        payload = packet.payload
        if (
            packet.schema_version != "1.0.0"
            or payload.get("schema_version") != "teacher-scorer-input/1.0.0"
            or payload.get("role") != "teacher_scorer"
            or payload.get("allowlisted_fields") != sorted(payload)
            or payload.get("output_schema") != teacher_output_schema_contract()
        ):
            raise TeacherOutputError("TeacherScorer input packet contract is invalid")
        return packet

    @classmethod
    def _normalize(
        cls,
        packet: Artifact,
        raw: bytes,
        role_session_lineage: str,
    ) -> dict[str, JsonValue]:
        if role_session_lineage != _LINEAGE:
            raise TeacherOutputError("TeacherScorer role lineage is not the isolated approved task")
        decoded = _bounded_json(raw)
        session = packet.payload.get("scoring_session")
        if not isinstance(session, dict):
            raise TeacherOutputError("TeacherScorer packet session is invalid")
        if (
            set(decoded) != _TOP_FIELDS
            or decoded.get("schema_version") != "teacher-scorer-output/1.0.0"
            or decoded.get("packet_id") != packet.payload.get("packet_id")
            or type(decoded.get("seed")) is not int
            or decoded.get("seed") != packet.payload.get("seed")
            or decoded.get("scoring_session_id") != session.get("scoring_session_id")
            or decoded.get("input_packet_content_hash") != packet.content_hash
            or decoded.get("input_payload_canonical_hash") != sha256_hex(canonical_json_bytes(packet.payload))
        ):
            raise TeacherOutputError("TeacherScorer output envelope is invalid")
        expected_turns = session.get("turns")
        expected_threads = session.get("resident_thread_ids")
        raw_threads = decoded.get("thread_sessions")
        raw_turns = decoded.get("turns")
        raw_labels = decoded.get("labels")
        if (
            not isinstance(expected_turns, list)
            or len(expected_turns) != 8
            or not isinstance(expected_threads, list)
            or len(expected_threads) != 5
            or not isinstance(raw_threads, list)
            or len(raw_threads) != 5
            or not isinstance(raw_turns, list)
            or len(raw_turns) != 8
            or not isinstance(raw_labels, list)
            or len(raw_labels) != 32
        ):
            raise TeacherOutputError("TeacherScorer session cardinality is invalid")
        normalized_threads = cls._normalize_threads(
            raw_threads,
            expected_threads,
            expected_turns,
            session.get("scoring_session_id"),
        )
        expected_items: list[dict[str, object]] = []
        normalized_turns: list[dict[str, JsonValue]] = []
        for expected, actual in zip(expected_turns, raw_turns, strict=True):
            if not isinstance(expected, dict) or not isinstance(actual, dict) or set(actual) != _TURN_FIELDS:
                raise TeacherOutputError("TeacherScorer turn trace fields are invalid")
            items = expected.get("items")
            if not isinstance(items, list) or len(items) != 4 or not all(isinstance(item, dict) for item in items):
                raise TeacherOutputError("TeacherScorer packet turn items are invalid")
            item_dicts = cast(list[dict[str, object]], items)
            actual_items = actual.get("items")
            expected_actual_items = [
                {
                    "content_hash": item.get("content_hash"),
                    "prompt_hash": item.get("prompt_hash"),
                    "response_hash": item.get("response_hash"),
                    "trajectory_id": item.get("trajectory_id"),
                }
                for item in item_dicts
            ]
            if (
                actual
                != {
                    "items": expected_actual_items,
                    "thread_id": expected.get("thread_id"),
                    "turn_index": expected.get("turn_index"),
                    "wave_index": expected.get("wave_index"),
                }
                or type(actual.get("turn_index")) is not int
                or type(actual.get("wave_index")) is not int
                or not isinstance(actual_items, list)
                or any(not isinstance(item, dict) or set(item) != _TURN_ITEM_FIELDS for item in actual_items)
            ):
                raise TeacherOutputError("TeacherScorer turn trace does not echo the frozen schedule")
            for item_index, item in enumerate(item_dicts, start=1):
                expected_items.append(
                    {
                        **item,
                        "item_index": item_index,
                        "thread_id": expected.get("thread_id"),
                        "turn_index": expected.get("turn_index"),
                        "wave_index": expected.get("wave_index"),
                    }
                )
            normalized_turns.append(cast(dict[str, JsonValue], actual))
        normalized_labels = cls._normalize_labels(raw_labels, expected_items)
        return {
            "input_packet_content_hash": cast(str, decoded["input_packet_content_hash"]),
            "input_payload_canonical_hash": cast(str, decoded["input_payload_canonical_hash"]),
            "labels": cast(JsonValue, normalized_labels),
            "packet_id": cast(str, decoded["packet_id"]),
            "schema_version": "normalized-teacher-scorer-output/1.0.0",
            "scoring_session_id": cast(str, decoded["scoring_session_id"]),
            "seed": cast(int, decoded["seed"]),
            "thread_sessions": cast(JsonValue, normalized_threads),
            "turns": cast(JsonValue, normalized_turns),
        }

    @staticmethod
    def _normalize_threads(
        raw_threads: Sequence[object],
        expected_threads: Sequence[object],
        expected_turns: Sequence[object],
        scoring_session_id: object,
    ) -> list[dict[str, JsonValue]]:
        normalized: list[dict[str, JsonValue]] = []
        if type(scoring_session_id) is not str:
            raise TeacherOutputError("TeacherScorer scoring session identity is invalid")
        for thread_index, (thread_id, actual) in enumerate(zip(expected_threads, raw_threads, strict=True), start=1):
            if not isinstance(actual, dict) or set(actual) != _THREAD_FIELDS:
                raise TeacherOutputError("TeacherScorer thread session fields are invalid")
            expected_indices = [
                cast(dict[str, object], turn).get("turn_index")
                for turn in expected_turns
                if isinstance(turn, dict) and turn.get("thread_id") == thread_id
            ]
            if (
                actual.get("thread_id") != thread_id
                or actual.get("turn_indices") != expected_indices
                or actual.get("scoring_session_id") != scoring_session_id
                or thread_id != f"sol-thread-{thread_index:02d}"
            ):
                raise TeacherOutputError("TeacherScorer thread session lineage is invalid")
            normalized.append(cast(dict[str, JsonValue], actual))
        return normalized

    @staticmethod
    def _normalize_labels(
        raw_labels: list[object],
        expected_items: list[dict[str, object]],
    ) -> list[dict[str, JsonValue]]:
        normalized: list[dict[str, JsonValue]] = []
        seen_ids: set[str] = set()
        for expected, actual in zip(expected_items, raw_labels, strict=True):
            if not isinstance(actual, dict) or set(actual) != _LABEL_FIELDS:
                raise TeacherOutputError("TeacherScorer label fields are invalid")
            scores = actual.get("dimension_scores")
            tags = actual.get("failure_tags")
            evidence = actual.get("evidence")
            scalar = actual.get("scalar_micros")
            trajectory_id = actual.get("trajectory_id")
            content_hash = actual.get("input_content_hash")
            if (
                not isinstance(scores, dict)
                or set(scores) != _DIMENSIONS
                or any(type(score) is not int or not 0 <= score <= 100 for score in scores.values())
                or type(scalar) is not int
                or scalar != sum(cast(dict[str, int], scores).values()) * 250_000
                or not isinstance(tags, list)
                or len(tags) > 16
                or any(
                    not isinstance(tag, dict)
                    or set(tag) != _TAG_FIELDS
                    or tag.get("dimension") not in _DIMENSIONS
                    or tag.get("severity") not in {"minor", "major", "critical"}
                    or type(tag.get("type")) is not str
                    or not cast(str, tag.get("type"))
                    or len(cast(str, tag.get("type"))) > 128
                    for tag in tags
                )
                or len(
                    {
                        (tag.get("dimension"), tag.get("severity"), tag.get("type"))
                        for tag in tags
                        if isinstance(tag, dict)
                    }
                )
                != len(tags)
                or not isinstance(evidence, dict)
                or set(evidence) != _DIMENSIONS
                or any(type(item) is not str or not 1 <= len(item) <= 2_048 for item in evidence.values())
                or trajectory_id != expected.get("trajectory_id")
                or content_hash != expected.get("content_hash")
                or actual.get("prompt_hash") != expected.get("prompt_hash")
                or actual.get("response_hash") != expected.get("response_hash")
                or actual.get("item_index") != expected.get("item_index")
                or type(actual.get("item_index")) is not int
                or actual.get("thread_id") != expected.get("thread_id")
                or actual.get("turn_index") != expected.get("turn_index")
                or type(actual.get("turn_index")) is not int
                or actual.get("wave_index") != expected.get("wave_index")
                or type(actual.get("wave_index")) is not int
                or not isinstance(trajectory_id, str)
                or trajectory_id in seen_ids
            ):
                raise TeacherOutputError("TeacherScorer label semantics are invalid")
            seen_ids.add(trajectory_id)
            normalized.append(cast(dict[str, JsonValue], actual))
        if len(seen_ids) != 32:
            raise TeacherOutputError("TeacherScorer labels are incomplete")
        return normalized


__all__ = [
    "TeacherOutputError",
    "TeacherPublicContract",
    "TeacherRoleIngressReceipt",
    "TeacherScorerIngress",
    "teacher_output_schema_contract",
]
