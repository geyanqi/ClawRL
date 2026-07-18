"""Strict private ingress for isolated Ticket 05 certification roles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex

CertificationRole = Literal["TeacherScorer", "StudentJudge", "AlignmentAuditor", "PromptOptimizer"]
CertificationExecutionProfile = Literal["isolated_subagent", "fixture_contract_simulator"]
_HASH = re.compile(r"^[0-9a-f]{64}$")
_ROLE_LINEAGES: dict[CertificationRole, str] = {
    "TeacherScorer": "/root/ticket05_teacher_scorer",
    "StudentJudge": "/root/ticket05_student_judge",
    "AlignmentAuditor": "/root/ticket05_alignment_auditor",
    "PromptOptimizer": "/root/ticket05_prompt_optimizer",
}
_ROLE_LINEAGE_ALLOWLISTS: dict[CertificationRole, frozenset[str]] = {
    role: frozenset(
        {
            lineage,
            f"{lineage}_p1",
            f"{lineage}_p1a2",
            f"{lineage}_rep2a2",
            f"{lineage}_rep2a3",
        }
    )
    for role, lineage in _ROLE_LINEAGES.items()
}
_FIXTURE_ROLE_LINEAGES: dict[CertificationRole, str] = {
    role: f"/fixture/ticket05_contract_example_{role.casefold()}" for role in _ROLE_LINEAGES
}
_TICKET06_FIXTURE_ROLE_LINEAGES: dict[CertificationRole, str] = {
    role: f"/fixture/ticket06_contract_example_{role.casefold()}" for role in _ROLE_LINEAGES
}
_PROFILE_LINEAGE_ALLOWLISTS: dict[CertificationExecutionProfile, dict[CertificationRole, frozenset[str]]] = {
    "isolated_subagent": {role: _ROLE_LINEAGE_ALLOWLISTS[role] for role in _ROLE_LINEAGES},
    "fixture_contract_simulator": {
        role: frozenset({_FIXTURE_ROLE_LINEAGES[role], _TICKET06_FIXTURE_ROLE_LINEAGES[role]})
        for role in _ROLE_LINEAGES
    },
}
_PACKET_SCHEMAS: dict[CertificationRole, str] = {
    "TeacherScorer": "HoldoutTeacherInputPacket",
    "StudentJudge": "HoldoutStudentInputPacket",
    "AlignmentAuditor": "CertificationAuditorInputPacket",
    "PromptOptimizer": "CertificationOptimizerInputPacket",
}
_OUTPUT_SCHEMA_VERSIONS: dict[CertificationRole, str] = {
    "TeacherScorer": "holdout-teacher-output/1.0.0",
    "StudentJudge": "holdout-student-output/1.0.0",
    "AlignmentAuditor": "certification-auditor-output/1.0.0",
    "PromptOptimizer": "certification-optimizer-output/1.0.0",
}
_TICKET06_PACKET_ID_PREFIXES: dict[CertificationRole, str] = {
    "TeacherScorer": "t06-teacher-",
    "StudentJudge": "t06-student-",
    "AlignmentAuditor": "t06-auditor-",
    "PromptOptimizer": "t06-optimizer-",
}


class CertificationRoleOutputError(ValueError):
    """A role output or its durable private ingress is invalid."""


def required_role_lineage(role: CertificationRole) -> str:
    return _ROLE_LINEAGES[role]


def fixture_role_lineage(role: CertificationRole) -> str:
    """Return lineage that can only identify local contract-example simulation."""

    return _FIXTURE_ROLE_LINEAGES[role]


def ticket06_fixture_role_lineage(role: CertificationRole) -> str:
    """Return a simulator-only Ticket06 lineage with no isolated-role authority."""

    return _TICKET06_FIXTURE_ROLE_LINEAGES[role]


def _is_ticket06_packet(role: CertificationRole, packet: Artifact) -> bool:
    packet_id = packet.payload.get("packet_id")
    return type(packet_id) is str and cast(str, packet_id).startswith(_TICKET06_PACKET_ID_PREFIXES[role])


def _execution_profile_for_lineage(
    role: CertificationRole, role_session_lineage: object
) -> CertificationExecutionProfile:
    matches = [
        profile
        for profile, lineages_by_role in _PROFILE_LINEAGE_ALLOWLISTS.items()
        if role_session_lineage in lineages_by_role[role]
    ]
    if len(matches) != 1:
        raise CertificationRoleOutputError("role session lineage has no unique execution profile")
    return matches[0]


def output_schema_contract(role: CertificationRole, *, items_per_turn: int = 8) -> dict[str, JsonValue]:
    if role in {"TeacherScorer", "StudentJudge"}:
        if items_per_turn not in {4, 8}:
            raise CertificationRoleOutputError("scoring items_per_turn is unsupported")
        return {
            "dimension_fields": ["correctness", "reasoning_quality", "task_completion", "tool_discipline"],
            "dimension_score_range": {"maximum": 100, "minimum": 0, "unit": "integer_points"},
            "evidence_semantics": {"maximum_characters": 4096, "minimum_characters": 32, "wire_type": "string"},
            "label_required": ["dimension_scores", "evidence", "item_hash", "scalar_micros"],
            "required": [
                "input_packet_content_hash",
                "input_payload_canonical_hash",
                "labels",
                "packet_id",
                "schema_version",
                "seed",
                "session_id",
                "turns",
            ],
            "scalar_range": {"maximum_micros": 100_000_000, "minimum_micros": 0},
            "schema_version": _OUTPUT_SCHEMA_VERSIONS[role],
            "turns_semantics": (
                {"exact_turn_count": 4, "wire_type": "integer"}
                if items_per_turn == 8
                else {
                    "exact_item_count": 32,
                    "turn_count": "derived_from_packet_scoring_session",
                    "wire_type": "integer",
                }
            ),
        }
    if role == "AlignmentAuditor":
        return {
            "diagnostic_required": [
                "absolute_bias_micros",
                "failure_rate_micros",
                "mean_absolute_error_micros",
                "p95_absolute_error_micros",
                "pairwise_agreement_micros",
                "student_variance_micros",
            ],
            "required": [
                "aggregate_diagnostics",
                "auditor_session_id",
                "input_packet_content_hash",
                "input_payload_canonical_hash",
                "packet_id",
                "schema_version",
                "seed",
                "verdict",
            ],
            "schema_version": _OUTPUT_SCHEMA_VERSIONS[role],
        }
    return {
        "candidate_prompt_semantics": {
            "maximum_characters": 16_384,
            "minimum_characters": 32,
            "wire_type": "string",
        },
        "optimizer_session_id_semantics": {"wire_type": "string"},
        "required": [
            "candidate_prompt",
            "input_packet_content_hash",
            "input_payload_canonical_hash",
            "optimizer_session_id",
            "packet_id",
            "schema_version",
            "seed",
        ],
        "schema_version": _OUTPUT_SCHEMA_VERSIONS[role],
    }


def compute_aggregate_diagnostics(packet: Artifact) -> dict[str, int]:
    """Recompute the preregistered calibrated-scalar diagnostics from auditor input."""

    if packet.schema_name != "CertificationAuditorInputPacket":
        raise CertificationRoleOutputError("aggregate diagnostics require an auditor packet")
    comparison = packet.payload.get("comparison")
    policy = packet.payload.get("alignment_policy")
    if not isinstance(comparison, dict) or not isinstance(policy, dict):
        raise CertificationRoleOutputError("auditor comparison inputs are invalid")
    item_hashes = comparison.get("item_hashes")
    teacher_labels = comparison.get("teacher_labels")
    student_labels = comparison.get("student_labels")
    if (
        not isinstance(item_hashes, list)
        or len(item_hashes) != 32
        or len(set(cast(list[str], item_hashes))) != 32
        or not isinstance(teacher_labels, list)
        or not isinstance(student_labels, list)
        or len(teacher_labels) != 32
        or len(student_labels) != 32
    ):
        raise CertificationRoleOutputError("auditor comparison cardinality is invalid")
    teacher: list[int] = []
    student: list[int] = []
    for expected_hash, teacher_label, student_label in zip(item_hashes, teacher_labels, student_labels, strict=True):
        if (
            not isinstance(teacher_label, dict)
            or not isinstance(student_label, dict)
            or teacher_label.get("item_hash") != expected_hash
            or student_label.get("item_hash") != expected_hash
            or type(teacher_label.get("scalar_micros")) is not int
            or type(student_label.get("scalar_micros")) is not int
        ):
            raise CertificationRoleOutputError("auditor labels do not align by item hash")
        teacher.append(cast(int, teacher_label["scalar_micros"]))
        student.append(cast(int, student_label["scalar_micros"]))
    differences = [student_value - teacher_value for teacher_value, student_value in zip(teacher, student, strict=True)]
    absolute = sorted(abs(value) for value in differences)
    pair_count = 0
    agreeing_pairs = 0
    for left in range(32):
        for right in range(left + 1, 32):
            teacher_order = (teacher[left] > teacher[right]) - (teacher[left] < teacher[right])
            student_order = (student[left] > student[right]) - (student[left] < student[right])
            pair_count += 1
            agreeing_pairs += teacher_order == student_order
    student_mean = sum(student) // 32
    max_item_error = cast(int, policy.get("max_p95_absolute_error_micros"))
    return {
        "absolute_bias_micros": abs(sum(differences)) // 32,
        "failure_rate_micros": sum(value > max_item_error for value in absolute) * 1_000_000 // 32,
        "mean_absolute_error_micros": sum(absolute) // 32,
        "p95_absolute_error_micros": absolute[30],
        "pairwise_agreement_micros": agreeing_pairs * 1_000_000 // pair_count,
        "student_variance_micros": sum(abs(value - student_mean) for value in student) // 32,
    }


def aggregate_diagnostic_contract() -> dict[str, JsonValue]:
    """Return the exact all-integer algorithm preregistered before audit."""

    return {
        "absolute_bias_micros": "abs(sum(student-teacher)) // 32",
        "failure_rate_micros": ("count(abs(student-teacher) > policy.max_p95_absolute_error_micros) * 1000000 // 32"),
        "mean_absolute_error_micros": "sum(abs(student-teacher)) // 32",
        "p95_absolute_error_micros": "sorted(abs(student-teacher))[30] for exact n=32",
        "pairwise_agreement_micros": (
            "for all 496 unordered pairs, compare sign(left-right) including ties; equal signs count as agreement; "
            "agreement_count * 1000000 // 496"
        ),
        "rounding": "nonnegative integer floor division only; no binary floating point",
        "student_variance_micros": (
            "mean absolute dispersion: student_mean=floor(sum(student)/32), then sum(abs(student-student_mean)) // 32"
        ),
        "version": "calibrated-scalar-diagnostics/1.0.0",
    }


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _bounded_json(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes or not 1 <= len(raw) <= 1_048_576:
        raise CertificationRoleOutputError("role output size is invalid")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CertificationRoleOutputError("role output is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CertificationRoleOutputError("role output must be an object")
    pending: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    chars = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > 30_000 or depth > 32:
            raise CertificationRoleOutputError("role output structural limits exceeded")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            chars += len(item)
            if len(item) > 32_768 or chars > 400_000:
                raise CertificationRoleOutputError("role output string limits exceeded")
        elif item is not None and type(item) not in {bool, int}:
            raise CertificationRoleOutputError("role output has an unsupported scalar")
    return cast(dict[str, object], value)


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            keys.update(cast(dict[str, object], item))
            pending.extend(cast(dict[str, object], item).values())
        elif isinstance(item, list):
            pending.extend(item)
    return keys


@dataclass(frozen=True, slots=True)
class CertificationRoleIngressReceipt:
    role: CertificationRole
    ingress_hash: str
    input_packet_hash: str
    raw_output_hash: str
    normalized_output_hash: str
    role_invocation_audit_hash: str

    def artifact_payload(self) -> dict[str, str]:
        return {
            "ingress_hash": self.ingress_hash,
            "input_packet_hash": self.input_packet_hash,
            "normalized_output_hash": self.normalized_output_hash,
            "raw_output_hash": self.raw_output_hash,
            "role": self.role,
            "role_invocation_audit_hash": self.role_invocation_audit_hash,
        }


@dataclass(frozen=True, slots=True)
class CertificationRolePublicContract:
    role: CertificationRole
    normalized_output: Artifact
    role_invocation_audit: Artifact


class CertificationRoleIngress:
    """Persist exact role bytes privately and deterministically recertify them."""

    @classmethod
    def validate_input_packet(cls, root: str | Path, role: CertificationRole, packet_hash: str) -> Artifact:
        return cls._read_packet(ArtifactStore(root), role, packet_hash)

    @classmethod
    def configure_isolated_lineage(
        cls,
        root: str | Path,
        *,
        role: CertificationRole,
        input_packet_hash: str,
        role_session_lineage: str,
        run_id: str,
        attempt_index: int,
        retry_index: int,
    ) -> Artifact:
        """Freeze one orchestrator-approved isolated invocation identity before dispatch."""

        if (
            type(role_session_lineage) is not str
            or not role_session_lineage.startswith("/")
            or role_session_lineage.startswith("/fixture/")
            or not 4 <= len(role_session_lineage) <= 512
            or type(run_id) is not str
            or not 1 <= len(run_id) <= 256
            or attempt_index not in {1, 2, 3}
            or retry_index not in {0, 1, 2, 3}
        ):
            raise CertificationRoleOutputError("isolated role lineage grant identity is invalid")
        store = ArtifactStore(root)
        packet = cls._read_packet(store, role, input_packet_hash)
        payload: dict[str, JsonValue] = {
            "attempt_index": attempt_index,
            "execution_profile": "isolated_subagent",
            "input_packet_hash": packet.content_hash,
            "packet_session_identity": cls._packet_session_identity(packet),
            "retry_index": retry_index,
            "role": role,
            "role_session_lineage": role_session_lineage,
            "run_id": run_id,
            "schema_version": "certification-role-lineage-grant/1.0.0",
        }
        grant = store.put("CertificationRoleLineageGrant", "1.0.0", payload)
        ref = (
            Path(root)
            / "private-boundary"
            / "certification-role-lineage-grants"
            / packet.content_hash
            / role
            / f"{attempt_index}-{retry_index}.ref"
        )
        ArtifactStore.durable_mkdir(ref.parent)
        expected = f"{grant.content_hash}\n".encode("ascii")
        if ref.exists():
            try:
                committed = ref.read_bytes()
            except OSError as error:
                raise CertificationRoleOutputError("isolated role lineage grant ref is unavailable") from error
            if committed != expected:
                raise CertificationRoleOutputError("isolated role lineage grant conflicts with frozen invocation")
        else:
            ArtifactStore._publish(ref, expected)
        return grant

    @classmethod
    def validate_isolated_lineage(
        cls,
        root: str | Path,
        *,
        role: CertificationRole,
        input_packet_hash: str,
        role_session_lineage: str,
        run_id: str,
        attempt_index: int,
        retry_index: int,
    ) -> Artifact:
        store = ArtifactStore(root)
        packet = cls._read_packet(store, role, input_packet_hash)
        ref = (
            Path(root)
            / "private-boundary"
            / "certification-role-lineage-grants"
            / packet.content_hash
            / role
            / f"{attempt_index}-{retry_index}.ref"
        )
        try:
            ref_bytes = ref.read_bytes()
        except OSError as error:
            raise CertificationRoleOutputError("isolated role lineage is not configured") from error
        if len(ref_bytes) != 65 or not ref_bytes.endswith(b"\n"):
            raise CertificationRoleOutputError("isolated role lineage grant ref is invalid")
        try:
            grant_hash = ref_bytes[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise CertificationRoleOutputError("isolated role lineage grant ref is invalid") from error
        try:
            grant = store.read(grant_hash, expected_schema_name="CertificationRoleLineageGrant")
        except ArtifactCorruption as error:
            raise CertificationRoleOutputError("isolated role lineage grant cannot be freshly recertified") from error
        expected: dict[str, JsonValue] = {
            "attempt_index": attempt_index,
            "execution_profile": "isolated_subagent",
            "input_packet_hash": packet.content_hash,
            "packet_session_identity": cls._packet_session_identity(packet),
            "retry_index": retry_index,
            "role": role,
            "role_session_lineage": role_session_lineage,
            "run_id": run_id,
            "schema_version": "certification-role-lineage-grant/1.0.0",
        }
        if grant.payload != expected:
            raise CertificationRoleOutputError("isolated role lineage grant cannot be freshly recertified")
        return grant

    @staticmethod
    def _packet_session_identity(packet: Artifact) -> dict[str, JsonValue]:
        session = packet.payload.get("scoring_session", packet.payload.get("session"))
        if not isinstance(session, dict):
            raise CertificationRoleOutputError("role packet session identity is unavailable")
        if "session_id" in session:
            identity: dict[str, JsonValue] = {"session_id": cast(JsonValue, session["session_id"])}
        else:
            identity = {key: value for key, value in session.items() if key.endswith("_session_id")}
        if len(identity) != 1 or type(next(iter(identity.values()))) is not str:
            raise CertificationRoleOutputError("role packet session identity is invalid")
        return identity

    @classmethod
    def stage_private_exchange(
        cls,
        root: str | Path,
        *,
        role: CertificationRole,
        input_packet_hash: str,
        raw_role_output: bytes,
        role_session_lineage: str,
        execution_profile: CertificationExecutionProfile,
        run_id: str | None = None,
        attempt_index: int = 1,
        retry_index: int = 0,
    ) -> CertificationRoleIngressReceipt:
        store = ArtifactStore(root)
        packet = cls._read_packet(store, role, input_packet_hash)
        ticket06_packet = _is_ticket06_packet(role, packet)
        lineage_grant: Artifact | None = None
        if execution_profile == "fixture_contract_simulator":
            accepted_fixture_lineages = (
                frozenset({_TICKET06_FIXTURE_ROLE_LINEAGES[role]})
                if ticket06_packet
                else _PROFILE_LINEAGE_ALLOWLISTS[execution_profile][role]
            )
            if role_session_lineage not in accepted_fixture_lineages:
                raise CertificationRoleOutputError("fixture role lineage does not match its closed-world profile")
        elif execution_profile == "isolated_subagent":
            if ticket06_packet and role_session_lineage in _ROLE_LINEAGE_ALLOWLISTS[role]:
                raise CertificationRoleOutputError("Ticket05 legacy isolated lineage cannot authorize Ticket06")
            if ticket06_packet or role_session_lineage not in _ROLE_LINEAGE_ALLOWLISTS[role]:
                lineage_grant = cls.validate_isolated_lineage(
                    root,
                    role=role,
                    input_packet_hash=input_packet_hash,
                    role_session_lineage=role_session_lineage,
                    run_id=cast(str, run_id),
                    attempt_index=attempt_index,
                    retry_index=retry_index,
                )
        else:
            raise CertificationRoleOutputError("role execution profile is invalid")
        try:
            normalized = cls._normalize(role, packet, raw_role_output)
        except CertificationRoleOutputError as error:
            cls._persist_rejected_output(
                store,
                role=role,
                packet=packet,
                raw_role_output=raw_role_output,
                role_session_lineage=role_session_lineage,
                execution_profile=execution_profile,
                validation_error=str(error),
            )
            raise
        normalized_bytes = canonical_json_bytes(normalized)
        normalized_artifact = store.put(f"NormalizedCertification{role}Output", "1.0.0", normalized)
        raw_hash = sha256_hex(raw_role_output)
        audit_payload: dict[str, JsonValue] = {
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
            "role_type": role,
            "seed": packet.payload["seed"],
        }
        if lineage_grant is not None:
            audit_payload.update(
                {
                    "attempt_index": attempt_index,
                    "retry_index": retry_index,
                    "role_lineage_grant_hash": lineage_grant.content_hash,
                    "run_id": cast(str, run_id),
                }
            )
        audit = store.put(
            "RoleInvocationAudit",
            "1.0.0",
            audit_payload,
        )
        identity = {
            "input_packet_hash": input_packet_hash,
            "normalized_output_hash": normalized_artifact.content_hash,
            "raw_output_hash": raw_hash,
            "role_invocation_audit_hash": audit.content_hash,
        }
        ingress_hash = sha256_hex(
            canonical_json_bytes({"domain": "certification-role-ingress/1.0.0", "identity": identity, "role": role})
        )
        receipt = CertificationRoleIngressReceipt(role=role, ingress_hash=ingress_hash, **identity)
        private = Path(root) / "private-boundary" / "certification-role-ingress" / role / ingress_hash
        ArtifactStore.durable_mkdir(private)
        ArtifactStore._publish(private / "role-output.raw.json", raw_role_output)
        ArtifactStore._publish(private / "role-output.normalized.canonical.json", normalized_bytes)
        ArtifactStore._publish(private / "role-input.canonical.json", canonical_json_bytes(packet.payload))
        ingress_manifest: dict[str, JsonValue] = {
            **receipt.artifact_payload(),
            "normalized_output_size": len(normalized_bytes),
            "raw_output_size": len(raw_role_output),
            "execution_profile": execution_profile,
            "role_session_lineage": role_session_lineage,
            "schema_version": "certification-role-ingress/1.0.0",
        }
        if lineage_grant is not None:
            ingress_manifest.update(
                {
                    "attempt_index": attempt_index,
                    "retry_index": retry_index,
                    "role_lineage_grant_hash": lineage_grant.content_hash,
                    "run_id": cast(str, run_id),
                }
            )
        ArtifactStore._publish(
            private / "ingress-receipt.canonical.json",
            canonical_json_bytes(ingress_manifest),
        )
        return receipt

    @classmethod
    def _persist_rejected_output(
        cls,
        store: ArtifactStore,
        *,
        role: CertificationRole,
        packet: Artifact,
        raw_role_output: bytes,
        role_session_lineage: str,
        execution_profile: CertificationExecutionProfile,
        validation_error: str,
    ) -> None:
        """Quarantine bounded invalid role bytes without advancing workflow state."""

        if type(raw_role_output) is not bytes or not 1 <= len(raw_role_output) <= 1_048_576:
            return
        raw_hash = sha256_hex(raw_role_output)
        rejection_identity = sha256_hex(
            canonical_json_bytes(
                {
                    "domain": "rejected-certification-role-output/1.0.0",
                    "input_packet_hash": packet.content_hash,
                    "raw_output_hash": raw_hash,
                    "role": role,
                }
            )
        )
        private = store.root / "private-boundary" / "certification-role-rejections" / role / rejection_identity
        ArtifactStore.durable_mkdir(private)
        ArtifactStore._publish(private / "role-output.rejected.raw.json", raw_role_output)
        manifest = {
            "input_packet_hash": packet.content_hash,
            "raw_output_hash": raw_hash,
            "raw_output_size": len(raw_role_output),
            "reason_code": "ROLE_OUTPUT_CONTRACT_REJECTED",
            "rejection_identity": rejection_identity,
            "role": role,
            "execution_profile": execution_profile,
            "role_session_lineage": role_session_lineage,
            "schema_version": "rejected-certification-role-ingress/1.0.0",
            "validation_error": validation_error,
        }
        ArtifactStore._publish(private / "rejection-manifest.canonical.json", canonical_json_bytes(manifest))
        store.put("RejectedRoleInvocationAudit", "1.0.0", manifest)

    @classmethod
    def load_public_contract(
        cls, root: str | Path, receipt: CertificationRoleIngressReceipt
    ) -> CertificationRolePublicContract:
        if type(receipt) is not CertificationRoleIngressReceipt:
            raise CertificationRoleOutputError("opaque certification role receipt is required")
        store = ArtifactStore(root)
        packet = cls._read_packet(store, receipt.role, receipt.input_packet_hash)
        ticket06_packet = _is_ticket06_packet(receipt.role, packet)
        private = Path(root) / "private-boundary" / "certification-role-ingress" / receipt.role / receipt.ingress_hash
        try:
            raw = (private / "role-output.raw.json").read_bytes()
            normalized_bytes = (private / "role-output.normalized.canonical.json").read_bytes()
            input_bytes = (private / "role-input.canonical.json").read_bytes()
            manifest_bytes = (private / "ingress-receipt.canonical.json").read_bytes()
            manifest = json.loads(manifest_bytes, object_pairs_hook=_strict_object)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise CertificationRoleOutputError("private certification role ingress is unavailable") from error
        normalized = cls._normalize(receipt.role, packet, raw)
        manifest_lineage = manifest.get("role_session_lineage") if isinstance(manifest, dict) else None
        manifest_profile = manifest.get("execution_profile") if isinstance(manifest, dict) else None
        legacy_isolated_manifest = (
            not ticket06_packet
            and manifest_profile is None
            and manifest_lineage in _ROLE_LINEAGE_ALLOWLISTS[receipt.role]
        )
        effective_profile: object = "isolated_subagent" if legacy_isolated_manifest else manifest_profile
        configured_grant: Artifact | None = None
        if (
            effective_profile == "isolated_subagent"
            and (ticket06_packet or manifest_lineage not in _ROLE_LINEAGE_ALLOWLISTS[receipt.role])
            and isinstance(manifest, dict)
        ):
            configured_grant = cls.validate_isolated_lineage(
                root,
                role=receipt.role,
                input_packet_hash=receipt.input_packet_hash,
                role_session_lineage=cast(str, manifest_lineage),
                run_id=cast(str, manifest.get("run_id")),
                attempt_index=cast(int, manifest.get("attempt_index")),
                retry_index=cast(int, manifest.get("retry_index")),
            )
        expected_manifest = {
            **receipt.artifact_payload(),
            "normalized_output_size": len(normalized_bytes),
            "raw_output_size": len(raw),
            "role_session_lineage": manifest_lineage,
            "schema_version": "certification-role-ingress/1.0.0",
        }
        if not legacy_isolated_manifest:
            expected_manifest["execution_profile"] = cast(JsonValue, manifest_profile)
        if configured_grant is not None:
            expected_manifest.update(
                {
                    "attempt_index": cast(JsonValue, manifest.get("attempt_index")),
                    "retry_index": cast(JsonValue, manifest.get("retry_index")),
                    "role_lineage_grant_hash": configured_grant.content_hash,
                    "run_id": cast(JsonValue, manifest.get("run_id")),
                }
            )
        static_lineage = (
            effective_profile in _PROFILE_LINEAGE_ALLOWLISTS
            and manifest_lineage
            in _PROFILE_LINEAGE_ALLOWLISTS[cast(CertificationExecutionProfile, effective_profile)][receipt.role]
        )
        if ticket06_packet:
            static_lineage = (
                effective_profile == "fixture_contract_simulator"
                and manifest_lineage == _TICKET06_FIXTURE_ROLE_LINEAGES[receipt.role]
            )
            if manifest_lineage in _ROLE_LINEAGE_ALLOWLISTS[receipt.role]:
                raise CertificationRoleOutputError("Ticket05 legacy isolated lineage cannot authorize Ticket06")
        if (
            not isinstance(manifest, dict)
            or effective_profile not in _PROFILE_LINEAGE_ALLOWLISTS
            or not (static_lineage or configured_grant is not None)
            or canonical_json_bytes(manifest) != manifest_bytes
            or manifest != expected_manifest
            or input_bytes != canonical_json_bytes(packet.payload)
            or sha256_hex(raw) != receipt.raw_output_hash
            or canonical_json_bytes(normalized) != normalized_bytes
        ):
            raise CertificationRoleOutputError("private certification role ingress is corrupt")
        normalized_artifact = store.read(
            receipt.normalized_output_hash, expected_schema_name=f"NormalizedCertification{receipt.role}Output"
        )
        audit = store.read(receipt.role_invocation_audit_hash, expected_schema_name="RoleInvocationAudit")
        if (
            normalized_artifact.payload != normalized
            or audit.payload.get("role_session_lineage") != manifest_lineage
            or audit.payload.get("role_lineage_grant_hash")
            != (configured_grant.content_hash if configured_grant is not None else None)
            or audit.payload.get("attempt_index") != (manifest.get("attempt_index") if configured_grant else None)
            or audit.payload.get("retry_index") != (manifest.get("retry_index") if configured_grant else None)
            or audit.payload.get("run_id") != (manifest.get("run_id") if configured_grant else None)
        ):
            raise CertificationRoleOutputError("public certification role contract is invalid")
        return CertificationRolePublicContract(receipt.role, normalized_artifact, audit)

    @classmethod
    def _read_packet(cls, store: ArtifactStore, role: CertificationRole, packet_hash: str) -> Artifact:
        if type(packet_hash) is not str or _HASH.fullmatch(packet_hash) is None:
            raise CertificationRoleOutputError("role packet hash is invalid")
        packet = store.read(packet_hash, expected_schema_name=_PACKET_SCHEMAS[role])
        payload = packet.payload
        items_per_turn = (
            cast(int, payload.get("items_per_turn"))
            if role in {"TeacherScorer", "StudentJudge"} and payload.get("items_per_turn") in {4, 8}
            else 8
        )
        expected_output_schema = output_schema_contract(role, items_per_turn=items_per_turn)
        compatible_output_schemas = [expected_output_schema]
        if role in {"TeacherScorer", "StudentJudge"} and items_per_turn == 8:
            # Ticket05 briefly persisted a packet-derived turn contract. The
            # strict session validator below still enforces exact 4x8 cardinality.
            compatible_output_schemas.append(output_schema_contract(role, items_per_turn=4))
        if (
            packet.schema_version != "1.0.0"
            or payload.get("allowlisted_fields") != sorted(payload)
            or payload.get("output_schema") not in compatible_output_schemas
            or payload.get("seed") is None
            or type(payload.get("packet_id")) is not str
        ):
            raise CertificationRoleOutputError("role packet envelope is invalid")
        encoded = canonical_json_bytes(payload).decode("utf-8").lower()
        if "generator_model" in encoded or "generator_profile" in encoded or "fit_items" in encoded:
            raise CertificationRoleOutputError("role packet exposes prohibited generator or fit state")
        keys = _all_keys(payload)
        if role == "StudentJudge" and keys.intersection(
            {
                "alignment_policy",
                "alignment_policy_hash",
                "initial_eval_rubric",
                "sol_inference",
                "teacher_labels",
                "teacher_label_set_hash",
            }
        ):
            raise CertificationRoleOutputError("Student Judge packet exposes teacher-only state")
        if role == "AlignmentAuditor" and payload.get("diagnostic_contract") != aggregate_diagnostic_contract():
            raise CertificationRoleOutputError("AlignmentAuditor packet lacks the frozen diagnostic algorithm")
        if role == "PromptOptimizer" and keys.intersection(
            {
                "holdout_items",
                "holdout_set_hash",
                "item_hash",
                "item_hashes",
                "student_labels",
                "teacher_labels",
            }
        ):
            raise CertificationRoleOutputError("PromptOptimizer packet exposes item-level holdout state")
        return packet

    @classmethod
    def _normalize(cls, role: CertificationRole, packet: Artifact, raw: bytes) -> dict[str, JsonValue]:
        decoded = _bounded_json(raw)
        required = cast(list[str], output_schema_contract(role)["required"])
        if (
            set(decoded) != set(required)
            or decoded.get("schema_version") != _OUTPUT_SCHEMA_VERSIONS[role]
            or decoded.get("packet_id") != packet.payload.get("packet_id")
            or decoded.get("seed") != packet.payload.get("seed")
            or decoded.get("input_packet_content_hash") != packet.content_hash
            or decoded.get("input_payload_canonical_hash") != sha256_hex(canonical_json_bytes(packet.payload))
        ):
            raise CertificationRoleOutputError("role output envelope is invalid")
        if role in {"TeacherScorer", "StudentJudge"}:
            return cls._normalize_scores(role, packet, decoded)
        if role == "AlignmentAuditor":
            diagnostics = decoded.get("aggregate_diagnostics")
            policy = packet.payload.get("alignment_policy")
            session = packet.payload.get("session")
            if (
                not isinstance(diagnostics, dict)
                or set(diagnostics) != set(cast(list[str], output_schema_contract(role)["diagnostic_required"]))
                or any(type(value) is not int or value < 0 or value > 100_000_000 for value in diagnostics.values())
                or decoded.get("verdict") not in {"pass", "fail"}
                or not isinstance(session, dict)
                or type(decoded.get("auditor_session_id")) is not str
                or decoded.get("auditor_session_id") != session.get("auditor_session_id")
                or not isinstance(policy, dict)
            ):
                raise CertificationRoleOutputError("auditor output is invalid")
            expected_diagnostics = compute_aggregate_diagnostics(packet)
            if diagnostics != expected_diagnostics:
                raise CertificationRoleOutputError("auditor diagnostics do not match calibrated scalar inputs")
            expected_pass = (
                cast(int, diagnostics["mean_absolute_error_micros"])
                <= cast(int, policy["max_mean_absolute_error_micros"])
                and cast(int, diagnostics["p95_absolute_error_micros"])
                <= cast(int, policy["max_p95_absolute_error_micros"])
                and cast(int, diagnostics["absolute_bias_micros"]) <= cast(int, policy["max_absolute_bias_micros"])
                and cast(int, diagnostics["pairwise_agreement_micros"])
                >= cast(int, policy["min_pairwise_agreement_micros"])
                and cast(int, diagnostics["student_variance_micros"])
                >= cast(int, policy["min_student_variance_micros"])
                and cast(int, diagnostics["failure_rate_micros"]) <= cast(int, policy["max_failure_rate_micros"])
            )
            if decoded["verdict"] != ("pass" if expected_pass else "fail"):
                raise CertificationRoleOutputError("auditor verdict conflicts with frozen numeric policy")
        else:
            session = packet.payload.get("session")
            if (
                not isinstance(session, dict)
                or type(decoded.get("candidate_prompt")) is not str
                or not 32 <= len(cast(str, decoded.get("candidate_prompt"))) <= 16_384
                or type(decoded.get("optimizer_session_id")) is not str
                or decoded.get("optimizer_session_id") != session.get("optimizer_session_id")
            ):
                raise CertificationRoleOutputError("optimizer output is invalid")
        normalized = dict(decoded)
        normalized["schema_version"] = f"normalized-{_OUTPUT_SCHEMA_VERSIONS[role]}"
        return cast(dict[str, JsonValue], normalized)

    @classmethod
    def _normalize_scores(
        cls, role: CertificationRole, packet: Artifact, decoded: dict[str, object]
    ) -> dict[str, JsonValue]:
        session = packet.payload.get("scoring_session")
        labels = decoded.get("labels")
        if (
            not isinstance(session, dict)
            or type(decoded.get("session_id")) is not str
            or decoded.get("session_id") != session.get("session_id")
            or not isinstance(labels, list)
            or len(labels) != 32
            or decoded.get("turns") != len(cast(list[object], session.get("turns", [])))
            or type(decoded.get("turns")) is not int
        ):
            raise CertificationRoleOutputError("scoring role session is invalid")
        expected_hashes = [
            cast(str, item["item_hash"])
            for turn in cast(list[dict[str, object]], session["turns"])
            for item in cast(list[dict[str, object]], turn["items"])
        ]
        turns = session.get("turns")
        items_per_turn = session.get("items_per_turn")
        if (
            not isinstance(turns, list)
            or type(items_per_turn) is not int
            or items_per_turn not in {4, 8}
            or session.get("turn_count") != len(turns)
            or len(turns) != 32 // items_per_turn
            or any(
                not isinstance(turn, dict)
                or not isinstance(turn.get("items"), list)
                or len(cast(list[object], turn["items"])) != items_per_turn
                for turn in turns
            )
            or len(expected_hashes) != 32
            or len(set(expected_hashes)) != 32
        ):
            raise CertificationRoleOutputError("scoring packet cardinality is invalid")
        normalized_labels: list[dict[str, JsonValue]] = []
        for expected, label in zip(expected_hashes, labels, strict=True):
            if not isinstance(label, dict) or set(label) != {
                "dimension_scores",
                "evidence",
                "item_hash",
                "scalar_micros",
            }:
                raise CertificationRoleOutputError("scoring label fields are invalid")
            dimensions = label.get("dimension_scores")
            dimension_names = {"correctness", "reasoning_quality", "task_completion", "tool_discipline"}
            if (
                label.get("item_hash") != expected
                or not isinstance(dimensions, dict)
                or set(dimensions) != dimension_names
                or any(type(value) is not int or not 0 <= value <= 100 for value in dimensions.values())
                or type(label.get("evidence")) is not str
                or not 32 <= len(cast(str, label.get("evidence"))) <= 4096
                or type(label.get("scalar_micros")) is not int
                or label.get("scalar_micros") != sum(cast(dict[str, int], dimensions).values()) * 250_000
            ):
                raise CertificationRoleOutputError("scoring label does not match the frozen item/scalar contract")
            normalized_labels.append(cast(dict[str, JsonValue], label))
        if [cast(str, label["item_hash"]) for label in normalized_labels] != expected_hashes:
            raise CertificationRoleOutputError("scoring labels reordered or duplicated holdout items")
        scalar_values = {cast(int, label["scalar_micros"]) for label in normalized_labels}
        dimension_vectors = {
            canonical_json_bytes(cast(dict[str, JsonValue], label["dimension_scores"])) for label in normalized_labels
        }
        evidence_values = {cast(str, label["evidence"]) for label in normalized_labels}
        if len(scalar_values) < 2 or len(dimension_vectors) < 2 or len(evidence_values) != 32:
            raise CertificationRoleOutputError(
                "scoring output must contain nonconstant scores and item-specific evidence"
            )
        normalized = dict(decoded)
        normalized["labels"] = normalized_labels
        normalized["schema_version"] = f"normalized-{_OUTPUT_SCHEMA_VERSIONS[role]}"
        return cast(dict[str, JsonValue], normalized)
