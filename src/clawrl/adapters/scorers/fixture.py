"""Persistent deterministic Student Judge fixture boundary."""

from __future__ import annotations

import fcntl
import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from decimal import Decimal, DecimalException, InvalidOperation
from pathlib import Path
from typing import cast

from clawrl.adapters.faults import PersistentFaultAttemptLedger
from clawrl.artifacts import (
    MAX_SAFE_INTEGER,
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    CanonicalizationError,
    JsonValue,
    canonical_json_bytes,
    sha256_hex,
)
from clawrl.judge.student_packet import (
    StudentPacketValidationError,
    sanitized_rejection_payload,
    validate_student_packet_structure,
)

_KEY = re.compile(r"^[0-9a-f]{64}$")
_SAFE_LINEAGE = re.compile(r"^/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+$")
_PROHIBITED_PACKET_TERMS = ("teacher", "holdout", "generator", "credential", "production")
_PROHIBITED_ROLE_VALUE = re.compile(
    r"(?:^|[^a-z0-9])(?:teacher[\s._-]*(?:only|label|secret)|holdout[\s._-]*only)(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)
_STUDENT_PACKET_FIELDS = (
    "schema_version",
    "role",
    "seed",
    "packet_id",
    "run_id",
    "global_step",
    "trace_id",
    "trajectory",
    "judge_pack",
    "rubric",
    "output_schema",
    "allowlisted_fields",
)
_TRAJECTORY_FIELDS = {"trajectory_id", "prompt", "response"}
_JUDGE_PACK_FIELDS = {
    "judge_pack_id",
    "items_per_turn",
    "reward_schema_version",
    "scalarizer_version",
    "dimensions",
    "scalarizer",
}
_OUTPUT_SCHEMA_FIELDS = {"schema_version", "required"}
_OUTPUT_FIELDS = (
    "packet_id",
    "seed",
    "dimensions",
    "overall_scalar",
    "confidence",
    "failure_tags",
    "evidence",
    "turn_local_tie_groups",
)
_VALIDATION_LIMITS: dict[str, int] = {
    "max_bytes": 262_144,
    "max_depth": 64,
    "max_nodes": 10_000,
    "max_string_characters": 32_768,
    "max_total_string_characters": 131_072,
}


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


class _InvalidScoringExchange(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class PersistentFixtureScorer:
    """A stateful, content-derived scorer with durable idempotent outcomes."""

    def __init__(
        self,
        root: str | Path,
        store: ArtifactStore,
        failure_code: str | None,
        fault_schedule: tuple[str, ...] | None = None,
    ) -> None:
        self.root = Path(root)
        self.store = store
        self.failure_code = failure_code
        self.fault_schedule = fault_schedule or (("permanent_failure",) if failure_code is not None else ("success",))
        self.attempt_ledger = PersistentFaultAttemptLedger(
            self.root,
            self.store,
            "scorer",
            self.fault_schedule,
        )
        self.outcomes_dir = self.root / "boundaries" / "scorer" / "outcomes"
        ArtifactStore.durable_mkdir(self.outcomes_dir)
        self.lock_path = self.outcomes_dir.parent / "adapter.lock"
        ArtifactStore.durable_touch(self.lock_path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self.lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def query(
        self,
        idempotency_key: str,
        expected_request_hash: str,
    ) -> Artifact | None:
        self._validate_key(idempotency_key)
        self._validate_request_hash(expected_request_hash)
        ref = self.outcomes_dir / f"{idempotency_key}.ref"
        if not ref.exists():
            return None
        try:
            outcome = self.store.read(
                self._read_ref(ref),
                expected_schema_name="Observation",
            )
        except Exception as error:
            if isinstance(error, ArtifactCorruption):
                raise
            raise ArtifactCorruption("scorer outcome ref cannot be resolved") from error
        if (
            outcome.payload.get("producer") != "fixture_student_judge"
            or outcome.payload.get("idempotency_key") != idempotency_key
            or outcome.payload.get("request_hash") != expected_request_hash
        ):
            raise ArtifactCorruption("scorer outcome request binding mismatch")
        return outcome

    def execute(
        self,
        idempotency_key: str,
        request: Artifact,
        input_packet: Mapping[str, object],
        raw_output: bytes,
        session_lineage: str,
        attempt_sequence: int = 1,
    ) -> Artifact:
        self._validate_key(idempotency_key)
        with self._locked():
            directive = self.attempt_ledger.directive(attempt_sequence)
            existing = self.query(idempotency_key, request.content_hash)
            if existing is not None:
                return existing
            existing_attempt = self.attempt_ledger.existing_observation(
                idempotency_key,
                request.content_hash,
                attempt_sequence,
            )
            if existing_attempt is not None:
                if existing_attempt.payload.get("status") != "retryable":
                    self._publish_outcome(idempotency_key, existing_attempt)
                return existing_attempt
            if directive in {"timeout", "delayed"}:
                failure_code = "SCORER_TIMEOUT" if directive == "timeout" else "SCORER_RESULT_DELAYED"
                detail = (
                    "deterministic scheduled scorer timeout"
                    if directive == "timeout"
                    else "deterministic scheduled scorer result delay"
                )
                outcome = self._retryable(
                    idempotency_key,
                    request,
                    attempt_sequence,
                    directive,
                    failure_code,
                    detail,
                )
            elif directive == "permanent_failure":
                if self.failure_code is None:
                    raise ArtifactCorruption("scorer permanent failure directive has no failure code")
                outcome = self._failure(
                    idempotency_key,
                    request,
                    self.failure_code,
                    "deterministic injected scorer failure",
                    None,
                    attempt_sequence,
                    directive,
                )
            elif directive == "success":
                try:
                    outcome = self._score(
                        idempotency_key,
                        request,
                        input_packet,
                        raw_output,
                        session_lineage,
                        attempt_sequence,
                        directive,
                    )
                except _InvalidScoringExchange as error:
                    role_invocation = self._persist_failed_invocation(
                        request,
                        input_packet,
                        raw_output,
                        session_lineage,
                        error.code,
                        error.detail,
                    )
                    outcome = self._failure(
                        idempotency_key,
                        request,
                        error.code,
                        error.detail,
                        role_invocation.content_hash,
                        attempt_sequence,
                        directive,
                    )
                except (
                    CanonicalizationError,
                    DecimalException,
                    KeyError,
                    MemoryError,
                    RecursionError,
                    TypeError,
                    ValueError,
                ):
                    code = "STUDENT_OUTPUT_INVALID"
                    detail = "role output contains a value outside the normalized schema"
                    role_invocation = self._persist_failed_invocation(
                        request,
                        input_packet,
                        raw_output,
                        session_lineage,
                        code,
                        detail,
                    )
                    outcome = self._failure(
                        idempotency_key,
                        request,
                        code,
                        detail,
                        role_invocation.content_hash,
                        attempt_sequence,
                        directive,
                    )
            else:
                raise ArtifactCorruption("scorer fault schedule directive is invalid")
            self.attempt_ledger.commit(
                idempotency_key,
                request.content_hash,
                attempt_sequence,
                outcome,
            )
            if outcome.payload.get("status") != "retryable":
                self._publish_outcome(idempotency_key, outcome)
            return outcome

    def execution_count(self) -> int:
        return len(list(self.outcomes_dir.glob("*.ref")))

    def read_attempt(
        self,
        idempotency_key: str,
        request_hash: str,
        attempt_sequence: int,
    ) -> Artifact:
        return self.attempt_ledger.read_attempt(
            idempotency_key,
            request_hash,
            attempt_sequence,
        )

    def _score(
        self,
        idempotency_key: str,
        request: Artifact,
        input_packet: Mapping[str, object],
        raw_output: bytes,
        session_lineage: str,
        attempt_sequence: int,
        directive: str,
    ) -> Artifact:
        packet = dict(input_packet)
        try:
            validate_student_packet_structure(packet)
        except StudentPacketValidationError as error:
            raise _InvalidScoringExchange(error.code, error.detail) from error
        allowlisted = packet.get("allowlisted_fields")
        if not isinstance(allowlisted, list) or not all(isinstance(item, str) for item in allowlisted):
            raise _InvalidScoringExchange("ROLE_PACKET_NOT_ALLOWLISTED", "student packet has no valid field allowlist")
        if allowlisted != list(_STUDENT_PACKET_FIELDS) or set(packet) != set(_STUDENT_PACKET_FIELDS):
            raise _InvalidScoringExchange(
                "ROLE_PACKET_NOT_ALLOWLISTED",
                "student packet does not match the versioned adapter allowlist",
            )
        if self._contains_prohibited_key(packet):
            raise _InvalidScoringExchange(
                "ROLE_PACKET_NOT_ALLOWLISTED", "student packet exposes a prohibited role field"
            )
        if _SAFE_LINEAGE.fullmatch(session_lineage) is None:
            raise _InvalidScoringExchange(
                "ROLE_SESSION_LINEAGE_INVALID", "student role session lineage is not a safe task path"
            )
        request_payload = request.payload
        trajectory = packet.get("trajectory")
        judge_pack = packet.get("judge_pack")
        if not isinstance(trajectory, dict) or not isinstance(judge_pack, dict):
            raise _InvalidScoringExchange("STUDENT_INPUT_INVALID", "trajectory or judge pack is absent")
        output_schema = packet.get("output_schema")
        rubric = packet.get("rubric")
        dimensions = judge_pack.get("dimensions")
        scalarizer = judge_pack.get("scalarizer")
        if (
            packet.get("schema_version") != "student-judge-input/1.0.0"
            or set(trajectory) != _TRAJECTORY_FIELDS
            or set(judge_pack) != _JUDGE_PACK_FIELDS
            or not isinstance(output_schema, dict)
            or set(output_schema) != _OUTPUT_SCHEMA_FIELDS
            or output_schema.get("schema_version") != "student-judge-output/1.0.0"
            or output_schema.get("required") != list(_OUTPUT_FIELDS)
            or not isinstance(dimensions, dict)
            or not isinstance(rubric, dict)
            or set(rubric) != set(dimensions)
            or not all(isinstance(value, str) and value for value in rubric.values())
            or not isinstance(scalarizer, dict)
            or set(scalarizer) != {"formula", "min", "max"}
            or any(
                not isinstance(contract, dict) or set(contract) != {"min", "max"} for contract in dimensions.values()
            )
        ):
            raise _InvalidScoringExchange(
                "ROLE_PACKET_NOT_ALLOWLISTED",
                "student packet nested fields do not match the versioned schema",
            )
        committed_judge_pack_hash = request_payload.get("judge_pack_hash")
        if not isinstance(committed_judge_pack_hash, str):
            raise _InvalidScoringExchange("STUDENT_INPUT_INVALID", "request has no JudgePack hash")
        committed_judge_pack = self.store.read(committed_judge_pack_hash, expected_schema_name="JudgePack")
        if committed_judge_pack.payload != judge_pack:
            raise _InvalidScoringExchange(
                "STUDENT_INPUT_MISMATCH", "student packet JudgePack is not the committed JudgePack"
            )
        expected_request_fields = {
            "run_id": packet.get("run_id"),
            "trace_id": packet.get("trace_id"),
            "trajectory_id": trajectory.get("trajectory_id"),
            "prompt": trajectory.get("prompt"),
            "response": trajectory.get("response"),
            "judge_pack_id": judge_pack.get("judge_pack_id"),
            "packet_id": packet.get("packet_id"),
        }
        if any(request_payload.get(key) != value for key, value in expected_request_fields.items()):
            raise _InvalidScoringExchange(
                "STUDENT_INPUT_MISMATCH", "role packet does not match the committed scoring request"
            )
        seed = packet.get("seed")
        global_step = packet.get("global_step")
        if (
            packet.get("role") != "student_judge"
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not 0 <= seed <= MAX_SAFE_INTEGER
            or not isinstance(global_step, int)
            or isinstance(global_step, bool)
            or not 0 <= global_step <= MAX_SAFE_INTEGER
        ):
            raise _InvalidScoringExchange("STUDENT_INPUT_INVALID", "student role or deterministic seed is invalid")
        normalized, reward_basis_points = self.recompute_output_contract(
            raw_output,
            packet,
            judge_pack,
        )

        input_artifact = self.store.put("StudentJudgeInputPacket", "1.0.0", packet)
        blob_hash, byte_size = self.store.put_blob(raw_output)
        raw_manifest = self.store.put(
            "RawRoleOutput",
            "1.0.0",
            {
                "blob_hash": blob_hash,
                "byte_size": byte_size,
                "media_type": "application/json; charset=utf-8",
            },
        )
        normalized_artifact = self.store.put("NormalizedRoleOutput", "1.0.0", normalized)
        role_invocation = self.store.put(
            "RoleInvocation",
            "1.0.0",
            {
                "input_hash": sha256_hex(canonical_json_bytes(packet)),
                "input_packet_hash": input_artifact.content_hash,
                "invocation": "student-judge",
                "normalized_output_hash": normalized_artifact.content_hash,
                "output_hash": blob_hash,
                "packet_id": cast(str, packet["packet_id"]),
                "producer": "fixture_student_judge",
                "raw_output_hash": blob_hash,
                "raw_output_manifest_hash": raw_manifest.content_hash,
                "request_hash": request.content_hash,
                "role": "student_judge",
                "seed": seed,
                "session_lineage": session_lineage,
                "status": "succeeded",
            },
        )
        evidence = self.store.put(
            "ScoringEvidence",
            "1.0.0",
            {
                "dimension_scores": normalized["dimensions"],
                "evidence": normalized["evidence"],
                "normalized_output_hash": normalized_artifact.content_hash,
                "request_hash": request.content_hash,
                "role_invocation_hash": role_invocation.content_hash,
            },
        )
        reward = self.store.put(
            "EvidenceLinkedReward",
            "1.0.0",
            {
                "confidence_basis_points": normalized["confidence_basis_points"],
                "evidence_hash": evidence.content_hash,
                "judge_pack_id": cast(str, judge_pack["judge_pack_id"]),
                "normalized_output_hash": normalized_artifact.content_hash,
                "request_hash": request.content_hash,
                "reward_basis_points": reward_basis_points,
                "role_invocation_hash": role_invocation.content_hash,
                "scalarizer_version": cast(str, judge_pack["scalarizer_version"]),
            },
        )
        return self.store.put(
            "Observation",
            "1.0.0",
            {
                "attempt_sequence": attempt_sequence,
                "directive": directive,
                "evidence_hash": evidence.content_hash,
                "idempotency_key": idempotency_key,
                "producer": "fixture_student_judge",
                "request_hash": request.content_hash,
                "reward_hash": reward.content_hash,
                "role_invocation_hash": role_invocation.content_hash,
                "status": "succeeded",
            },
        )

    def recompute_output_contract(
        self,
        raw_output: bytes,
        packet: Mapping[str, object],
        judge_pack: Mapping[str, object],
    ) -> tuple[dict[str, JsonValue], int]:
        """Recompute the normalized Student output using the scorer's frozen contract."""

        output_schema = packet.get("output_schema")
        if (
            not isinstance(output_schema, dict)
            or set(output_schema) != _OUTPUT_SCHEMA_FIELDS
            or output_schema.get("schema_version") != "student-judge-output/1.0.0"
            or output_schema.get("required") != list(_OUTPUT_FIELDS)
        ):
            raise _InvalidScoringExchange(
                "ROLE_PACKET_NOT_ALLOWLISTED",
                "student packet nested fields do not match the versioned schema",
            )
        if len(raw_output) > _VALIDATION_LIMITS["max_bytes"]:
            raise _InvalidScoringExchange("STUDENT_OUTPUT_INVALID", "raw role output exceeds the frozen byte limit")
        try:
            raw_text = raw_output.decode("utf-8")
            self._validate_json_text_limits(raw_text)
            decoded = json.loads(
                raw_text,
                parse_float=Decimal,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                object_pairs_hook=_reject_duplicate_object_pairs,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
            raise _InvalidScoringExchange(
                "STUDENT_OUTPUT_INVALID", "raw role output is not strict UTF-8 JSON"
            ) from error
        self._validate_decoded_json_limits(decoded)
        if not isinstance(decoded, dict):
            raise _InvalidScoringExchange("STUDENT_OUTPUT_INVALID", "role output must be an object")
        required = output_schema["required"]
        if set(decoded) != set(required):
            raise _InvalidScoringExchange("STUDENT_OUTPUT_INVALID", "role output fields do not match the frozen schema")
        if decoded.get("packet_id") != packet.get("packet_id") or decoded.get("seed") != packet.get("seed"):
            raise _InvalidScoringExchange(
                "STUDENT_OUTPUT_INVALID", "role output does not echo packet identity and seed"
            )
        judge_pack_value = dict(judge_pack)
        normalized = self._normalize(decoded, judge_pack_value)
        return normalized, self._reward_basis_points(normalized, judge_pack_value)

    def _normalize(self, decoded: dict[str, object], judge_pack: dict[str, object]) -> dict[str, JsonValue]:
        if any(self._contains_decimal(value) for key, value in decoded.items() if key != "confidence"):
            raise _InvalidScoringExchange(
                "STUDENT_OUTPUT_INVALID",
                "floating-point values are not allowed outside confidence",
            )
        dimensions = decoded.get("dimensions")
        dimension_contract = judge_pack.get("dimensions")
        if not isinstance(dimensions, dict) or not isinstance(dimension_contract, dict):
            raise _InvalidScoringExchange("STUDENT_OUTPUT_INVALID", "dimension vector is invalid")
        if set(dimensions) != set(dimension_contract):
            raise _InvalidScoringExchange("STUDENT_OUTPUT_INVALID", "dimension vector does not match JudgePack")
        normalized_dimensions: dict[str, JsonValue] = {}
        for name, contract in dimension_contract.items():
            score = dimensions.get(name)
            if (
                not isinstance(name, str)
                or not isinstance(contract, dict)
                or not isinstance(score, int)
                or isinstance(score, bool)
                or not isinstance(contract.get("min"), int)
                or isinstance(contract.get("min"), bool)
                or not isinstance(contract.get("max"), int)
                or isinstance(contract.get("max"), bool)
                or cast(int, contract["max"]) < cast(int, contract["min"])
                or not cast(int, contract["min"]) <= score <= cast(int, contract["max"])
            ):
                raise _InvalidScoringExchange("STUDENT_OUTPUT_INVALID", "dimension score is outside the frozen range")
            normalized_dimensions[name] = score
        scalarizer = judge_pack.get("scalarizer")
        if not isinstance(scalarizer, dict) or not isinstance(scalarizer.get("formula"), str):
            raise _InvalidScoringExchange("STUDENT_INPUT_INVALID", "scalarizer formula is invalid")
        formula_names = [name.strip() for name in cast(str, scalarizer["formula"]).split("+")]
        if (
            not formula_names
            or len(formula_names) != len(set(formula_names))
            or set(formula_names) != set(normalized_dimensions)
        ):
            raise _InvalidScoringExchange("STUDENT_INPUT_INVALID", "unsupported scalarizer formula")
        derived_overall = sum(cast(int, normalized_dimensions[name]) for name in formula_names)
        supplied_overall = decoded.get("overall_scalar")
        if (
            not isinstance(supplied_overall, int)
            or isinstance(supplied_overall, bool)
            or supplied_overall != derived_overall
        ):
            raise _InvalidScoringExchange("STUDENT_OUTPUT_INVALID", "overall scalar does not match dimension evidence")
        scalar_min = scalarizer.get("min")
        scalar_max = scalarizer.get("max")
        contract_min = sum(
            cast(int, cast(dict[str, object], dimension_contract[name])["min"]) for name in formula_names
        )
        contract_max = sum(
            cast(int, cast(dict[str, object], dimension_contract[name])["max"]) for name in formula_names
        )
        if (
            not isinstance(scalar_min, int)
            or isinstance(scalar_min, bool)
            or not isinstance(scalar_max, int)
            or isinstance(scalar_max, bool)
            or scalar_max <= scalar_min
            or scalar_min != contract_min
            or scalar_max != contract_max
            or not scalar_min <= derived_overall <= scalar_max
        ):
            raise _InvalidScoringExchange(
                "REWARD_RANGE_INVALID",
                "dimension domain and frozen scalarizer range are inconsistent",
            )
        confidence = self._confidence_basis_points(decoded.get("confidence"))
        failure_tags = decoded.get("failure_tags")
        evidence = decoded.get("evidence")
        tie_groups = decoded.get("turn_local_tie_groups")
        if (
            not isinstance(failure_tags, list)
            or not all(isinstance(tag, str) for tag in failure_tags)
            or not isinstance(evidence, list)
            or not evidence
            or not all(isinstance(item, str) and item for item in evidence)
            or not isinstance(tie_groups, list)
            or not all(
                isinstance(group, list) and all(isinstance(member, str) and member for member in group)
                for group in tie_groups
            )
            or not isinstance(decoded.get("packet_id"), str)
            or not isinstance(decoded.get("seed"), int)
            or isinstance(decoded.get("seed"), bool)
        ):
            raise _InvalidScoringExchange("STUDENT_OUTPUT_INVALID", "diagnostic output fields are invalid")
        return {
            "confidence_basis_points": confidence,
            "dimensions": normalized_dimensions,
            "evidence": cast(list[JsonValue], evidence),
            "failure_tags": cast(list[JsonValue], failure_tags),
            "overall_scalar": derived_overall,
            "packet_id": cast(str, decoded["packet_id"]),
            "seed": cast(int, decoded["seed"]),
            "turn_local_tie_groups": cast(list[JsonValue], tie_groups),
        }

    @staticmethod
    def _reward_basis_points(normalized: dict[str, JsonValue], judge_pack: dict[str, object]) -> int:
        scalarizer = judge_pack.get("scalarizer")
        overall = normalized.get("overall_scalar")
        if not isinstance(scalarizer, dict) or not isinstance(overall, int):
            raise _InvalidScoringExchange("REWARD_RANGE_INVALID", "scalarizer is absent")
        scalar_min = scalarizer.get("min")
        scalar_max = scalarizer.get("max")
        if (
            not isinstance(scalar_min, int)
            or isinstance(scalar_min, bool)
            or not isinstance(scalar_max, int)
            or isinstance(scalar_max, bool)
            or scalar_max <= scalar_min
            or not scalar_min <= overall <= scalar_max
        ):
            raise _InvalidScoringExchange("REWARD_RANGE_INVALID", "overall scalar is outside the frozen range")
        reward = (overall - scalar_min) * 10_000 // (scalar_max - scalar_min)
        if not 0 <= reward <= 10_000:
            raise _InvalidScoringExchange("REWARD_RANGE_INVALID", "normalized reward is outside [0,10000]")
        return reward

    @staticmethod
    def _confidence_basis_points(value: object) -> int:
        try:
            if isinstance(value, Decimal):
                confidence = value
            elif isinstance(value, int) and not isinstance(value, bool):
                confidence = Decimal(value)
            elif isinstance(value, str):
                confidence = Decimal(value)
            else:
                raise InvalidOperation
            if not confidence.is_finite():
                raise InvalidOperation
            if not Decimal(0) <= confidence <= Decimal(1):
                raise InvalidOperation
            exponent = confidence.as_tuple().exponent
            if not isinstance(exponent, int) or not -4 <= exponent <= 0:
                raise InvalidOperation
            scaled = confidence * 10_000
            if scaled != scaled.to_integral_value():
                raise InvalidOperation
        except (DecimalException, ValueError) as error:
            raise _InvalidScoringExchange("STUDENT_OUTPUT_INVALID", "confidence is not a canonical decimal") from error
        return int(scaled)

    def _persist_failed_invocation(
        self,
        request: Artifact,
        input_packet: Mapping[str, object],
        raw_output: bytes,
        session_lineage: str,
        failure_code: str,
        detail: str,
    ) -> Artifact:
        packet = dict(input_packet)
        try:
            validate_student_packet_structure(packet)
        except StudentPacketValidationError as packet_error:
            rejection = sanitized_rejection_payload(packet, packet_error)
            input_artifact = self.store.put("StudentPacketRejection", "1.0.0", rejection)
            input_schema_name = "StudentPacketRejection"
            rejected_hash = rejection.get("input_hash")
            input_hash = rejected_hash if isinstance(rejected_hash, str) else None
        else:
            input_bytes = canonical_json_bytes(packet)
            input_artifact = self.store.put("StudentJudgeInputPacket", "1.0.0", packet)
            input_schema_name = "StudentJudgeInputPacket"
            input_hash = sha256_hex(input_bytes)
        blob_hash, byte_size = self.store.put_blob(raw_output)
        raw_manifest = self.store.put(
            "RawRoleOutput",
            "1.0.0",
            {
                "blob_hash": blob_hash,
                "byte_size": byte_size,
                "media_type": "application/json; charset=utf-8",
            },
        )
        normalized_failure = self.store.put(
            "NormalizedRoleOutput",
            "1.0.0",
            {
                "failure_code": failure_code,
                "failure_detail": detail,
                "status": "invalid",
                "validation_limits": dict(_VALIDATION_LIMITS),
            },
        )
        seed = packet.get("seed")
        safe_seed = (
            seed if isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed <= MAX_SAFE_INTEGER else None
        )
        packet_id = packet.get("packet_id")
        return self.store.put(
            "RoleInvocation",
            "1.0.0",
            {
                "failure_code": failure_code,
                "input_hash": input_hash,
                "input_packet_hash": input_artifact.content_hash,
                "input_schema_name": input_schema_name,
                "invocation": "student-judge",
                "normalized_output_hash": normalized_failure.content_hash,
                "output_hash": blob_hash,
                "packet_id": packet_id if isinstance(packet_id, str) else None,
                "producer": "fixture_student_judge",
                "raw_output_hash": blob_hash,
                "raw_output_manifest_hash": raw_manifest.content_hash,
                "request_hash": request.content_hash,
                "role": "student_judge",
                "seed": safe_seed,
                "session_lineage": session_lineage,
                "status": "failed",
            },
        )

    def _failure(
        self,
        idempotency_key: str,
        request: Artifact,
        failure_code: str,
        detail: str,
        role_invocation_hash: str | None,
        attempt_sequence: int,
        directive: str,
    ) -> Artifact:
        return self.store.put(
            "Observation",
            "1.0.0",
            {
                "attempt_sequence": attempt_sequence,
                "detail": detail,
                "directive": directive,
                "failure_code": failure_code,
                "idempotency_key": idempotency_key,
                "producer": "fixture_student_judge",
                "request_hash": request.content_hash,
                "role_invocation_hash": role_invocation_hash,
                "status": "failed",
            },
        )

    def _retryable(
        self,
        idempotency_key: str,
        request: Artifact,
        attempt_sequence: int,
        directive: str,
        failure_code: str,
        detail: str,
    ) -> Artifact:
        return self.store.put(
            "Observation",
            "1.0.0",
            {
                "attempt_sequence": attempt_sequence,
                "detail": detail,
                "directive": directive,
                "failure_code": failure_code,
                "idempotency_key": idempotency_key,
                "producer": "fixture_student_judge",
                "request_hash": request.content_hash,
                "role_invocation_hash": None,
                "status": "retryable",
            },
        )

    def _publish_outcome(self, idempotency_key: str, outcome: Artifact) -> None:
        ArtifactStore._publish(
            self.outcomes_dir / f"{idempotency_key}.ref",
            f"{outcome.content_hash}\n".encode("ascii"),
        )

    @classmethod
    def _contains_prohibited_key(cls, value: object) -> bool:
        pending = [value]
        while pending:
            current = pending.pop()
            if isinstance(current, dict):
                for key, nested in current.items():
                    if isinstance(key, str) and any(term in key.lower() for term in _PROHIBITED_PACKET_TERMS):
                        return True
                    pending.append(nested)
            elif isinstance(current, list):
                pending.extend(current)
            elif isinstance(current, str) and _PROHIBITED_ROLE_VALUE.search(current) is not None:
                return True
        return False

    @classmethod
    def _contains_decimal(cls, value: object) -> bool:
        pending = [value]
        while pending:
            current = pending.pop()
            if isinstance(current, Decimal):
                return True
            if isinstance(current, dict):
                pending.extend(current.values())
            elif isinstance(current, list):
                pending.extend(current)
        return False

    @staticmethod
    def _validate_json_text_limits(raw_text: str) -> None:
        depth = 0
        nodes = 1
        in_string = False
        escaped = False
        string_characters = 0
        for character in raw_text:
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                    string_characters = 0
                else:
                    string_characters += 1
                    if string_characters > _VALIDATION_LIMITS["max_string_characters"]:
                        raise ValueError("JSON string exceeds frozen role-output limit")
                continue
            if character == '"':
                in_string = True
            elif character in "[{":
                depth += 1
                nodes += 1
                if depth > _VALIDATION_LIMITS["max_depth"]:
                    raise ValueError("JSON nesting exceeds frozen role-output limit")
            elif character in "]}":
                depth -= 1
            elif character == ",":
                nodes += 1
            if nodes > _VALIDATION_LIMITS["max_nodes"]:
                raise ValueError("JSON node count exceeds frozen role-output limit")

    @staticmethod
    def _validate_decoded_json_limits(value: object) -> None:
        pending: list[tuple[object, int]] = [(value, 1)]
        nodes = 0
        total_string_characters = 0
        while pending:
            current, depth = pending.pop()
            nodes += 1
            if nodes > _VALIDATION_LIMITS["max_nodes"]:
                raise _InvalidScoringExchange(
                    "STUDENT_OUTPUT_INVALID",
                    "decoded role output exceeds the frozen node limit",
                )
            if depth > _VALIDATION_LIMITS["max_depth"]:
                raise _InvalidScoringExchange(
                    "STUDENT_OUTPUT_INVALID",
                    "decoded role output exceeds the frozen depth limit",
                )
            if isinstance(current, str):
                characters = len(current)
                if characters > _VALIDATION_LIMITS["max_string_characters"]:
                    raise _InvalidScoringExchange(
                        "STUDENT_OUTPUT_INVALID",
                        "decoded role output string exceeds the frozen limit",
                    )
                total_string_characters += characters
            elif isinstance(current, dict):
                for key, item in current.items():
                    nodes += 1
                    if nodes > _VALIDATION_LIMITS["max_nodes"]:
                        raise _InvalidScoringExchange(
                            "STUDENT_OUTPUT_INVALID",
                            "decoded role output exceeds the frozen node limit",
                        )
                    key_characters = len(key)
                    if key_characters > _VALIDATION_LIMITS["max_string_characters"]:
                        raise _InvalidScoringExchange(
                            "STUDENT_OUTPUT_INVALID",
                            "decoded role output string exceeds the frozen limit",
                        )
                    total_string_characters += key_characters
                    pending.append((item, depth + 1))
            elif isinstance(current, list):
                pending.extend((item, depth + 1) for item in current)
            if total_string_characters > _VALIDATION_LIMITS["max_total_string_characters"]:
                raise _InvalidScoringExchange(
                    "STUDENT_OUTPUT_INVALID",
                    "decoded role output exceeds the frozen aggregate string limit",
                )

    @staticmethod
    def _validate_key(idempotency_key: str) -> None:
        if _KEY.fullmatch(idempotency_key) is None:
            raise ValueError("scorer idempotency key must be a SHA-256 hex digest")

    @staticmethod
    def _validate_request_hash(request_hash: str) -> None:
        if _KEY.fullmatch(request_hash) is None:
            raise ValueError("scorer request hash must be a SHA-256 hex digest")

    @staticmethod
    def _read_ref(path: Path) -> str:
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ArtifactCorruption("scorer outcome ref is unavailable") from error
        if len(content) != 65 or not content.endswith(b"\n"):
            raise ArtifactCorruption("scorer outcome ref is corrupt")
        try:
            content_hash = content[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise ArtifactCorruption("scorer outcome ref is not ASCII") from error
        if _KEY.fullmatch(content_hash) is None:
            raise ArtifactCorruption("scorer outcome ref has an invalid content hash")
        return content_hash
