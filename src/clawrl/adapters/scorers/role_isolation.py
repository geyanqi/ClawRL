"""Strict private ingress for Ticket 04 PromptOptimizer and AlignmentAuditor outputs."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, cast

from clawrl.artifacts import Artifact, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex

_HASH = re.compile(r"^[0-9a-f]{64}$")
_ROLES = {"PromptOptimizer", "AlignmentAuditor"}
_LINEAGES = {
    "PromptOptimizer": "/root/ticket04_prompt_optimizer",
    "AlignmentAuditor": "/root/ticket04_alignment_auditor",
}
_OPTIMIZER_PACKET_FIELDS = {
    "allowlisted_fields",
    "base_judge_prompt",
    "fit_items",
    "mode",
    "output_schema",
    "packet_id",
    "role",
    "role_isolation_policy_hash",
    "schema_version",
    "seed",
    "session",
    "teacher_label_set_hash",
    "trace_id",
}
_AUDITOR_PACKET_FIELDS = {
    "allowlisted_fields",
    "immutable_artifacts",
    "inspected_packet_contracts",
    "mode",
    "output_schema",
    "packet_id",
    "read_only_authority",
    "role",
    "schema_version",
    "seed",
    "session",
    "trace_id",
}
_FIT_ITEM_FIELDS = {
    "content_hash",
    "prompt",
    "prompt_hash",
    "response",
    "response_hash",
    "teacher_label",
    "trajectory_id",
}


class RoleIsolationOutputError(ValueError):
    """A Ticket 04 role output or private ingress is invalid."""


def prompt_optimizer_output_schema_contract() -> dict[str, JsonValue]:
    """Return the exact PromptOptimizer output envelope accepted by ingress."""

    return cast(
        dict[str, JsonValue],
        {
            "required": [
                "schema_version",
                "packet_id",
                "seed",
                "optimizer_session_id",
                "candidate_prompt",
                "aggregate_diagnostics",
                "input_packet_content_hash",
            ],
            "schema_version": "prompt-optimizer-output/1.0.0",
        },
    )


def alignment_auditor_output_schema_contract() -> dict[str, JsonValue]:
    """Return the exact AlignmentAuditor output envelope accepted by ingress."""

    return cast(
        dict[str, JsonValue],
        {
            "required": [
                "schema_version",
                "packet_id",
                "seed",
                "auditor_session_id",
                "isolation_verdict",
                "violations",
                "input_packet_content_hash",
            ],
            "schema_version": "alignment-auditor-isolation-output/1.0.0",
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
    if type(raw) is not bytes or not 1 <= len(raw) <= 262_144:
        raise RoleIsolationOutputError("role output byte size is invalid")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_float=str,
            parse_int=int,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RoleIsolationOutputError("role output is not strict UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise RoleIsolationOutputError("role output must be an object")
    pending: list[tuple[object, int]] = [(decoded, 0)]
    nodes = 0
    total_chars = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > 20_000 or depth > 32:
            raise RoleIsolationOutputError("role output exceeds structural limits")
        if isinstance(value, dict):
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
        elif isinstance(value, str):
            total_chars += len(value)
            if len(value) > 32_768 or total_chars > 131_072:
                raise RoleIsolationOutputError("role output exceeds string limits")
        elif value is not None and type(value) not in {bool, int}:
            raise RoleIsolationOutputError("role output contains an unsupported value")
    return cast(dict[str, object], decoded)


@dataclass(frozen=True, slots=True)
class RoleIsolationIngressReceipt:
    role_type: Literal["PromptOptimizer", "AlignmentAuditor"]
    ingress_hash: str
    input_packet_hash: str
    raw_output_hash: str
    normalized_output_hash: str
    role_invocation_audit_hash: str

    def __post_init__(self) -> None:
        if self.role_type not in _ROLES:
            raise RoleIsolationOutputError("role ingress type is invalid")
        for value in (
            self.ingress_hash,
            self.input_packet_hash,
            self.raw_output_hash,
            self.normalized_output_hash,
            self.role_invocation_audit_hash,
        ):
            if type(value) is not str or _HASH.fullmatch(value) is None:
                raise RoleIsolationOutputError("role ingress receipt contains an invalid hash")

    def artifact_payload(self) -> dict[str, str]:
        return {
            "ingress_hash": self.ingress_hash,
            "input_packet_hash": self.input_packet_hash,
            "normalized_output_hash": self.normalized_output_hash,
            "raw_output_hash": self.raw_output_hash,
            "role_invocation_audit_hash": self.role_invocation_audit_hash,
            "role_type": self.role_type,
        }


@dataclass(frozen=True, slots=True)
class RoleIsolationPublicContract:
    role_type: Literal["PromptOptimizer", "AlignmentAuditor"]
    normalized_output: Artifact
    role_invocation_audit: Artifact


class RoleIsolationIngress:
    """Persist exact role bytes privately and expose only recertified public artifacts."""

    @classmethod
    def validate_input_packet(
        cls,
        root: str | Path,
        *,
        role_type: Literal["PromptOptimizer", "AlignmentAuditor"],
        input_packet_hash: str,
    ) -> Artifact:
        """Apply the actual ingress consumer contract to a newly published role packet."""

        return cls._read_packet(ArtifactStore(root), role_type, input_packet_hash)

    @classmethod
    def stage_private_exchange(
        cls,
        root: str | Path,
        *,
        role_type: Literal["PromptOptimizer", "AlignmentAuditor"],
        input_packet_hash: str,
        raw_role_output: bytes,
        role_session_lineage: str,
    ) -> RoleIsolationIngressReceipt:
        store = ArtifactStore(root)
        packet = cls._read_packet(store, role_type, input_packet_hash)
        normalized = cls._normalize(store, role_type, packet, raw_role_output, role_session_lineage)
        normalized_bytes = canonical_json_bytes(normalized)
        normalized_artifact = store.put(f"Normalized{role_type}Output", "1.0.0", normalized)
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
                "role_type": role_type,
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
            canonical_json_bytes({"domain": f"{role_type.lower()}-private-ingress/1.0.0", "identity": identity})
        )
        receipt = RoleIsolationIngressReceipt(role_type=role_type, ingress_hash=ingress_hash, **identity)
        private_root = Path(root) / "private-boundary" / "role-isolation-ingress" / role_type / ingress_hash
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
                    "schema_version": "role-isolation-ingress/1.0.0",
                }
            ),
        )
        return receipt

    @classmethod
    def load_public_contract(
        cls,
        root: str | Path,
        receipt: RoleIsolationIngressReceipt,
    ) -> RoleIsolationPublicContract:
        if type(receipt) is not RoleIsolationIngressReceipt:
            raise RoleIsolationOutputError("opaque role ingress receipt is required")
        store = ArtifactStore(root)
        packet = cls._read_packet(store, receipt.role_type, receipt.input_packet_hash)
        lineage = _LINEAGES[receipt.role_type]
        private_root = (
            Path(root) / "private-boundary" / "role-isolation-ingress" / receipt.role_type / receipt.ingress_hash
        )
        try:
            raw = (private_root / "role-output.raw.json").read_bytes()
            normalized_bytes = (private_root / "role-output.normalized.canonical.json").read_bytes()
            input_bytes = (private_root / "role-input.canonical.json").read_bytes()
            manifest_bytes = (private_root / "ingress-receipt.canonical.json").read_bytes()
            manifest = json.loads(manifest_bytes, object_pairs_hook=_strict_object)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise RoleIsolationOutputError("private role ingress state is unavailable") from error
        expected_manifest = {
            **receipt.artifact_payload(),
            "normalized_output_size": len(normalized_bytes),
            "raw_output_size": len(raw),
            "role_session_lineage": lineage,
            "schema_version": "role-isolation-ingress/1.0.0",
        }
        if (
            not isinstance(manifest, dict)
            or canonical_json_bytes(manifest) != manifest_bytes
            or manifest != expected_manifest
            or input_bytes != canonical_json_bytes(packet.payload)
            or sha256_hex(raw) != receipt.raw_output_hash
        ):
            raise RoleIsolationOutputError("private role ingress state is corrupt")
        normalized = cls._normalize(store, receipt.role_type, packet, raw, lineage)
        if canonical_json_bytes(normalized) != normalized_bytes:
            raise RoleIsolationOutputError("role output normalization cannot be reproduced")
        normalized_artifact = store.read(
            receipt.normalized_output_hash,
            expected_schema_name=f"Normalized{receipt.role_type}Output",
        )
        audit = store.read(receipt.role_invocation_audit_hash, expected_schema_name="RoleInvocationAudit")
        if normalized_artifact.payload != normalized or audit.payload != {
            "input_hash": receipt.input_packet_hash,
            "input_packet_hash": receipt.input_packet_hash,
            "normalized_output_hash": receipt.normalized_output_hash,
            "normalized_output_payload_hash": sha256_hex(normalized_bytes),
            "normalized_output_size": len(normalized_bytes),
            "output_hash": receipt.raw_output_hash,
            "packet_id": packet.payload["packet_id"],
            "raw_output_hash": receipt.raw_output_hash,
            "raw_output_size": len(raw),
            "role_session_lineage": lineage,
            "role_type": receipt.role_type,
            "seed": packet.payload["seed"],
        }:
            raise RoleIsolationOutputError("public role audit artifacts are invalid")
        return RoleIsolationPublicContract(receipt.role_type, normalized_artifact, audit)

    @staticmethod
    def _read_packet(store: ArtifactStore, role_type: str, packet_hash: str) -> Artifact:
        if role_type not in _ROLES or type(packet_hash) is not str or _HASH.fullmatch(packet_hash) is None:
            raise RoleIsolationOutputError("role input packet identity is invalid")
        schema = "PromptOptimizerInputPacket" if role_type == "PromptOptimizer" else "AlignmentAuditorInputPacket"
        packet = store.read(packet_hash, expected_schema_name=schema)
        if packet.schema_version != "1.0.0" or packet.payload.get("allowlisted_fields") != sorted(packet.payload):
            raise RoleIsolationOutputError("role input packet allowlist is invalid")
        payload = packet.payload
        session = payload.get("session")
        if role_type == "PromptOptimizer":
            fit_items = payload.get("fit_items")
            base_prompt = payload.get("base_judge_prompt")
            if (
                set(payload) != _OPTIMIZER_PACKET_FIELDS
                or payload.get("role") != "prompt_optimizer"
                or payload.get("mode") != "fit_prompt_optimization"
                or payload.get("schema_version") != "prompt-optimizer-input/1.0.0"
                or payload.get("output_schema") != prompt_optimizer_output_schema_contract()
                or not isinstance(session, dict)
                or set(session) != {"optimizer_session_id", "permissions"}
                or session.get("permissions")
                != ["read_fit_items", "read_normalized_teacher_labels", "write_candidate_prompt"]
                or not isinstance(base_prompt, dict)
                or set(base_prompt) != {"artifact_hash", "contract"}
                or not isinstance(fit_items, list)
                or len(fit_items) != 32
                or any(not isinstance(item, dict) or set(item) != _FIT_ITEM_FIELDS for item in fit_items)
                or any(
                    type(cast(dict[str, object], item).get("content_hash")) is not str
                    or _HASH.fullmatch(cast(str, cast(dict[str, object], item).get("content_hash"))) is None
                    or type(cast(dict[str, object], item).get("trajectory_id")) is not str
                    for item in fit_items
                )
                or len({cast(str, cast(dict[str, object], item)["content_hash"]) for item in fit_items}) != 32
                or len({cast(str, cast(dict[str, object], item)["trajectory_id"]) for item in fit_items}) != 32
            ):
                raise RoleIsolationOutputError("PromptOptimizer input packet contract is invalid")
        else:
            if (
                set(payload) != _AUDITOR_PACKET_FIELDS
                or payload.get("role") != "alignment_auditor"
                or payload.get("mode") != "role_input_isolation_attestation"
                or payload.get("schema_version") != "alignment-auditor-isolation-input/1.0.0"
                or payload.get("output_schema") != alignment_auditor_output_schema_contract()
                or not isinstance(session, dict)
                or set(session) != {"auditor_session_id", "permissions"}
                or session.get("permissions") != ["read_hashes", "write_attestation"]
            ):
                raise RoleIsolationOutputError("AlignmentAuditor input packet contract is invalid")
        encoded = canonical_json_bytes(packet.payload).decode("utf-8").lower()
        if "generator" in encoded or (role_type == "PromptOptimizer" and "audit" in encoded):
            raise RoleIsolationOutputError("role input packet exposes prohibited cross-role state")
        if role_type == "AlignmentAuditor":
            authority = packet.payload.get("read_only_authority")
            if (
                not isinstance(authority, dict)
                or authority.get("may_modify_prompt") is not False
                or authority.get("may_modify_policy") is not False
                or authority.get("mutable_artifact_hashes") != []
            ):
                raise RoleIsolationOutputError("AlignmentAuditor packet is not read-only")
        return packet

    @classmethod
    def _normalize(
        cls,
        store: ArtifactStore,
        role_type: str,
        packet: Artifact,
        raw: bytes,
        lineage: str,
    ) -> dict[str, JsonValue]:
        if lineage != _LINEAGES.get(role_type):
            raise RoleIsolationOutputError("role session lineage is not the isolated approved task")
        decoded = _bounded_json(raw)
        if role_type == "PromptOptimizer":
            return cls._normalize_optimizer(store, packet, decoded)
        return cls._normalize_auditor(packet, decoded)

    @classmethod
    def _normalize_optimizer(
        cls,
        store: ArtifactStore,
        packet: Artifact,
        decoded: dict[str, object],
    ) -> dict[str, JsonValue]:
        fields = {
            "aggregate_diagnostics",
            "candidate_prompt",
            "input_packet_content_hash",
            "optimizer_session_id",
            "packet_id",
            "schema_version",
            "seed",
        }
        session = packet.payload.get("session")
        diagnostics = decoded.get("aggregate_diagnostics")
        candidate = decoded.get("candidate_prompt")
        if (
            set(decoded) != fields
            or decoded.get("schema_version") != "prompt-optimizer-output/1.0.0"
            or decoded.get("packet_id") != packet.payload.get("packet_id")
            or type(decoded.get("seed")) is not int
            or decoded.get("seed") != packet.payload.get("seed")
            or decoded.get("input_packet_content_hash") != packet.content_hash
            or not isinstance(session, dict)
            or decoded.get("optimizer_session_id") != session.get("optimizer_session_id")
            or not isinstance(candidate, str)
            or not 64 <= len(candidate) <= 16_384
            or not isinstance(diagnostics, dict)
        ):
            raise RoleIsolationOutputError("PromptOptimizer output envelope is invalid")
        fit_items = packet.payload.get("fit_items")
        if not isinstance(fit_items, list) or len(fit_items) != 32:
            raise RoleIsolationOutputError("PromptOptimizer fit inputs are invalid")
        expected = cls._expected_diagnostics(cast(list[object], fit_items))
        normalized_diagnostics = cls._validate_diagnostics(diagnostics, expected)
        return {
            "aggregate_diagnostics": normalized_diagnostics,
            "candidate_prompt": candidate,
            "input_packet_content_hash": packet.content_hash,
            "optimizer_session_id": cast(str, decoded["optimizer_session_id"]),
            "packet_id": cast(str, decoded["packet_id"]),
            "schema_version": "normalized-prompt-optimizer-output/1.0.0",
            "seed": cast(int, decoded["seed"]),
        }

    @staticmethod
    def _expected_diagnostics(fit_items: list[object]) -> dict[str, object]:
        dimensions = ("correctness", "reasoning_quality", "task_completion", "tool_discipline")
        score_counters: dict[str, Counter[int]] = {name: Counter() for name in dimensions}
        tag_counter: Counter[tuple[str, str, str]] = Counter()
        scalar_counter: Counter[int] = Counter()
        for item in fit_items:
            if not isinstance(item, dict) or not isinstance(item.get("teacher_label"), dict):
                raise RoleIsolationOutputError("PromptOptimizer normalized label input is invalid")
            label = cast(dict[str, object], item["teacher_label"])
            scores = label.get("dimension_scores")
            tags = label.get("failure_tags")
            scalar = label.get("scalar_micros")
            if not isinstance(scores, dict) or not isinstance(tags, list) or type(scalar) is not int:
                raise RoleIsolationOutputError("PromptOptimizer normalized label input is invalid")
            for name in dimensions:
                score = scores.get(name)
                if type(score) is not int:
                    raise RoleIsolationOutputError("PromptOptimizer normalized score is invalid")
                score_counters[name][cast(int, score)] += 1
            scalar_counter[cast(int, scalar)] += 1
            for tag in tags:
                if not isinstance(tag, dict):
                    raise RoleIsolationOutputError("PromptOptimizer normalized failure tag is invalid")
                key = (cast(str, tag.get("dimension")), cast(str, tag.get("severity")), cast(str, tag.get("type")))
                tag_counter[key] += 1
        means = {
            name: Decimal(sum(score * count for score, count in counter.items())) / Decimal(32)
            for name, counter in score_counters.items()
        }
        scalar_mean = Decimal(sum(value * count for value, count in scalar_counter.items())) / Decimal(32)
        return {
            "failure_tag_counts": [
                {"count": count, "dimension": key[0], "severity": key[1], "type": key[2]}
                for key, count in sorted(tag_counter.items())
            ],
            "fit_item_count": 32,
            "mean_dimension_scores": means,
            "scalar_micros": {"distribution": scalar_counter, "mean": scalar_mean},
            "score_distributions": score_counters,
        }

    @classmethod
    def _validate_diagnostics(
        cls,
        diagnostics: dict[str, object],
        expected: dict[str, object],
    ) -> dict[str, JsonValue]:
        if (
            set(diagnostics)
            != {
                "failure_tag_counts",
                "fit_item_count",
                "mean_dimension_scores",
                "observations",
                "scalar_micros",
                "score_distributions",
            }
            or diagnostics.get("fit_item_count") != 32
            or type(diagnostics.get("fit_item_count")) is not int
        ):
            raise RoleIsolationOutputError("PromptOptimizer aggregate diagnostic fields are invalid")
        failure_counts = diagnostics.get("failure_tag_counts")
        means = diagnostics.get("mean_dimension_scores")
        scalar = diagnostics.get("scalar_micros")
        distributions = diagnostics.get("score_distributions")
        observations = diagnostics.get("observations")
        if (
            failure_counts != expected["failure_tag_counts"]
            or not isinstance(means, dict)
            or not isinstance(scalar, dict)
            or set(scalar) != {"distribution", "mean"}
            or not isinstance(distributions, dict)
            or not isinstance(observations, list)
            or not 1 <= len(observations) <= 16
            or len(observations) != len(set(cast(list[object], observations)))
            or any(type(item) is not str or not item or len(item) > 2_048 for item in observations)
        ):
            raise RoleIsolationOutputError("PromptOptimizer aggregate diagnostics are invalid")
        expected_means = cast(dict[str, Decimal], expected["mean_dimension_scores"])
        if set(means) != set(expected_means) or any(
            cls._decimal(means.get(name)) != value for name, value in expected_means.items()
        ):
            raise RoleIsolationOutputError("PromptOptimizer dimension means are invalid")
        expected_scalar = cast(dict[str, object], expected["scalar_micros"])
        if cls._decimal(scalar.get("mean")) != expected_scalar["mean"] or scalar.get("distribution") != {
            str(key): value for key, value in cast(Counter[int], expected_scalar["distribution"]).items()
        }:
            raise RoleIsolationOutputError("PromptOptimizer scalar distribution is invalid")
        expected_distributions = cast(dict[str, Counter[int]], expected["score_distributions"])
        if distributions != {
            name: {str(key): value for key, value in counter.items()}
            for name, counter in expected_distributions.items()
        }:
            raise RoleIsolationOutputError("PromptOptimizer score distributions are invalid")
        return {
            "failure_tag_counts": cast(JsonValue, failure_counts),
            "fit_item_count": 32,
            "mean_dimension_scores": {name: cls._decimal_text(value) for name, value in expected_means.items()},
            "observations": cast(JsonValue, observations),
            "scalar_micros": {
                "distribution": cast(JsonValue, scalar["distribution"]),
                "mean": cls._decimal_text(cast(Decimal, expected_scalar["mean"])),
            },
            "score_distributions": cast(JsonValue, distributions),
        }

    @staticmethod
    def _normalize_auditor(packet: Artifact, decoded: dict[str, object]) -> dict[str, JsonValue]:
        fields = {
            "auditor_session_id",
            "input_packet_content_hash",
            "isolation_verdict",
            "packet_id",
            "schema_version",
            "seed",
            "violations",
        }
        session = packet.payload.get("session")
        if (
            set(decoded) != fields
            or decoded.get("schema_version") != "alignment-auditor-isolation-output/1.0.0"
            or decoded.get("packet_id") != packet.payload.get("packet_id")
            or type(decoded.get("seed")) is not int
            or decoded.get("seed") != packet.payload.get("seed")
            or decoded.get("input_packet_content_hash") != packet.content_hash
            or not isinstance(session, dict)
            or decoded.get("auditor_session_id") != session.get("auditor_session_id")
            or decoded.get("isolation_verdict") != "pass"
            or decoded.get("violations") != []
        ):
            raise RoleIsolationOutputError("AlignmentAuditor isolation output is invalid")
        return {
            "auditor_session_id": cast(str, decoded["auditor_session_id"]),
            "input_packet_content_hash": packet.content_hash,
            "isolation_verdict": "pass",
            "packet_id": cast(str, decoded["packet_id"]),
            "schema_version": "normalized-alignment-auditor-output/1.0.0",
            "seed": cast(int, decoded["seed"]),
            "violations": [],
        }

    @staticmethod
    def _decimal(value: object) -> Decimal:
        if type(value) not in {int, str}:
            raise RoleIsolationOutputError("diagnostic decimal is invalid")
        try:
            result = Decimal(str(value))
        except InvalidOperation as error:
            raise RoleIsolationOutputError("diagnostic decimal is invalid") from error
        if not result.is_finite():
            raise RoleIsolationOutputError("diagnostic decimal is invalid")
        return result

    @staticmethod
    def _decimal_text(value: Decimal) -> str:
        text = format(value, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text


__all__ = [
    "alignment_auditor_output_schema_contract",
    "prompt_optimizer_output_schema_contract",
    "RoleIsolationIngress",
    "RoleIsolationIngressReceipt",
    "RoleIsolationOutputError",
    "RoleIsolationPublicContract",
]
