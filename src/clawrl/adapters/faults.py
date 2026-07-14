"""Durable deterministic fixture-boundary attempt schedules."""

from __future__ import annotations

import re
from pathlib import Path

from clawrl.artifacts import (
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    canonical_json_bytes,
    sha256_hex,
)

FAULT_SCHEDULE_VERSION = "fixture-fault-schedule/1.0.0"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_DIRECTIVES = {"timeout", "delayed", "success", "permanent_failure"}
_TRANSIENT_DIRECTIVES = {"timeout", "delayed"}
_TERMINAL_DIRECTIVES = {"success", "permanent_failure"}
_ATTEMPT_FIELDS = {
    "attempt_sequence",
    "available_after_attempt",
    "boundary",
    "directive",
    "idempotency_key",
    "late_completion_hashes",
    "observation_hash",
    "previous_attempt_hash",
    "request_hash",
    "schedule_hash",
    "schedule_version",
    "status",
}
_LATE_COMPLETION_FIELDS = {
    "boundary",
    "idempotency_key",
    "observed_at_attempt",
    "origin_attempt_hash",
    "origin_attempt_sequence",
    "request_hash",
    "schedule_hash",
    "status",
}


class PersistentFaultAttemptLedger:
    """Append-only per-request attempts with deterministic schedule lineage."""

    def __init__(
        self,
        root: str | Path,
        store: ArtifactStore,
        boundary: str,
        schedule: tuple[str, ...],
    ) -> None:
        self.root = Path(root)
        self.store = store
        self.boundary = boundary
        self.schedule = schedule
        if boundary not in {"scorer", "cluster"}:
            raise ArtifactCorruption("fixture fault schedule boundary is invalid")
        if (
            not isinstance(schedule, tuple)
            or not 1 <= len(schedule) <= 16
            or any(not isinstance(directive, str) or directive not in _DIRECTIVES for directive in schedule)
            or schedule[-1] not in _TERMINAL_DIRECTIVES
            or any(directive in _TERMINAL_DIRECTIVES for directive in schedule[:-1])
        ):
            raise ArtifactCorruption("fixture fault schedule is invalid")
        self.schedule_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "boundary": boundary,
                    "directives": list(schedule),
                    "version": FAULT_SCHEDULE_VERSION,
                }
            )
        )
        self.attempts_root = self.root / "boundaries" / boundary / "attempts"
        ArtifactStore.durable_mkdir(self.attempts_root)

    def directive(self, attempt_sequence: int) -> str:
        if (
            not isinstance(attempt_sequence, int)
            or isinstance(attempt_sequence, bool)
            or not 1 <= attempt_sequence <= len(self.schedule)
        ):
            raise ArtifactCorruption("fixture fault schedule attempt is exhausted or invalid")
        return self.schedule[attempt_sequence - 1]

    def existing_observation(
        self,
        idempotency_key: str,
        request_hash: str,
        attempt_sequence: int,
    ) -> Artifact | None:
        ref = self._attempt_ref(idempotency_key, attempt_sequence)
        if not ref.exists():
            attempt_dir = ref.parent
            refs = sorted(attempt_dir.glob("*.ref")) if attempt_dir.exists() else []
            if len(refs) != attempt_sequence - 1:
                raise ArtifactCorruption("fixture attempt predecessor refs are missing or out of order")
            for sequence, predecessor_ref in enumerate(refs, start=1):
                if predecessor_ref.name != f"{sequence:020d}.ref":
                    raise ArtifactCorruption("fixture attempt predecessor ref sequence is invalid")
                self._read_attempt(
                    predecessor_ref,
                    idempotency_key,
                    request_hash,
                    sequence,
                )
            return None
        attempt = self._read_attempt(ref, idempotency_key, request_hash, attempt_sequence)
        return self.store.read(
            self._required_hash(attempt, "observation_hash"),
            expected_schema_name="Observation",
        )

    def commit(
        self,
        idempotency_key: str,
        request_hash: str,
        attempt_sequence: int,
        observation: Artifact,
    ) -> Artifact:
        directive = self.directive(attempt_sequence)
        attempt_dir = self.attempts_root / idempotency_key
        ArtifactStore.durable_mkdir(attempt_dir)
        current_ref = self._attempt_ref(idempotency_key, attempt_sequence)
        if current_ref.exists():
            existing = self._read_attempt(
                current_ref,
                idempotency_key,
                request_hash,
                attempt_sequence,
            )
            if existing.payload.get("observation_hash") != observation.content_hash:
                raise ArtifactCorruption("fixture attempt outcome conflicts with committed attempt")
            return existing

        refs = sorted(attempt_dir.glob("*.ref"))
        if len(refs) != attempt_sequence - 1:
            raise ArtifactCorruption("fixture attempt sequence has a gap or concurrent advance")
        previous: Artifact | None = None
        for sequence, ref in enumerate(refs, start=1):
            if ref.name != f"{sequence:020d}.ref":
                raise ArtifactCorruption("fixture attempt ref sequence is invalid")
            previous = self._read_attempt(
                ref,
                idempotency_key,
                request_hash,
                sequence,
            )

        late_completion_hashes: list[str] = []
        if (
            previous is not None
            and previous.payload.get("directive") == "delayed"
            and previous.payload.get("available_after_attempt") == attempt_sequence
        ):
            late = self.store.put(
                "FixtureLateBoundaryCompletion",
                "1.0.0",
                {
                    "boundary": self.boundary,
                    "idempotency_key": idempotency_key,
                    "observed_at_attempt": attempt_sequence,
                    "origin_attempt_hash": previous.content_hash,
                    "origin_attempt_sequence": attempt_sequence - 1,
                    "request_hash": request_hash,
                    "schedule_hash": self.schedule_hash,
                    "status": "available_late_ignored",
                },
            )
            late_completion_hashes.append(late.content_hash)

        retryable = observation.payload.get("status") == "retryable"
        attempt = self.store.put(
            "FixtureBoundaryAttempt",
            "1.0.0",
            {
                "attempt_sequence": attempt_sequence,
                "available_after_attempt": (attempt_sequence + 1 if directive == "delayed" else None),
                "boundary": self.boundary,
                "directive": directive,
                "idempotency_key": idempotency_key,
                "late_completion_hashes": late_completion_hashes,
                "observation_hash": observation.content_hash,
                "previous_attempt_hash": (previous.content_hash if previous is not None else None),
                "request_hash": request_hash,
                "schedule_hash": self.schedule_hash,
                "schedule_version": FAULT_SCHEDULE_VERSION,
                "status": "retryable" if retryable else "final",
            },
        )
        ArtifactStore._publish(
            current_ref,
            f"{attempt.content_hash}\n".encode("ascii"),
        )
        return attempt

    def read_attempt(
        self,
        idempotency_key: str,
        request_hash: str,
        attempt_sequence: int,
    ) -> Artifact:
        ref = self._attempt_ref(idempotency_key, attempt_sequence)
        if not ref.exists():
            raise ArtifactCorruption("fixture attempt ref is missing")
        return self._read_attempt(ref, idempotency_key, request_hash, attempt_sequence)

    def _read_attempt(
        self,
        ref: Path,
        idempotency_key: str,
        request_hash: str,
        attempt_sequence: int,
    ) -> Artifact:
        attempt = self.store.read(
            self._read_ref(ref),
            expected_schema_name="FixtureBoundaryAttempt",
        )
        payload = attempt.payload
        directive = self.directive(attempt_sequence)
        if (
            attempt.schema_version != "1.0.0"
            or set(payload) != _ATTEMPT_FIELDS
            or payload.get("attempt_sequence") != attempt_sequence
            or payload.get("boundary") != self.boundary
            or payload.get("directive") != directive
            or payload.get("idempotency_key") != idempotency_key
            or payload.get("request_hash") != request_hash
            or payload.get("schedule_hash") != self.schedule_hash
            or payload.get("schedule_version") != FAULT_SCHEDULE_VERSION
            or payload.get("status") not in {"retryable", "final"}
            or not isinstance(payload.get("late_completion_hashes"), list)
        ):
            raise ArtifactCorruption("fixture attempt lineage is invalid")
        observation = self.store.read(
            self._required_hash(attempt, "observation_hash"),
            expected_schema_name="Observation",
        )
        expected_producer = "fixture_student_judge" if self.boundary == "scorer" else "fixture_cluster"
        observation_status = observation.payload.get("status")
        expected_attempt_status = "retryable" if directive in _TRANSIENT_DIRECTIVES else "final"
        if (
            observation.schema_version != "1.0.0"
            or observation.payload.get("attempt_sequence") != attempt_sequence
            or observation.payload.get("directive") != directive
            or observation.payload.get("idempotency_key") != idempotency_key
            or observation.payload.get("producer") != expected_producer
            or observation.payload.get("request_hash") != request_hash
            or payload.get("status") != expected_attempt_status
            or (directive in _TRANSIENT_DIRECTIVES and observation_status != "retryable")
            or (directive == "success" and observation_status not in {"failed", "succeeded"})
            or (directive == "permanent_failure" and observation_status != "failed")
            or (observation_status == "succeeded" and directive != "success")
        ):
            raise ArtifactCorruption("fixture attempt observation binding is invalid")
        expected_previous = None
        previous: Artifact | None = None
        if attempt_sequence > 1:
            previous = self._read_attempt(
                self._attempt_ref(idempotency_key, attempt_sequence - 1),
                idempotency_key,
                request_hash,
                attempt_sequence - 1,
            )
            expected_previous = previous.content_hash
            if previous.payload.get("status") != "retryable":
                raise ArtifactCorruption("fixture attempt follows a terminal boundary attempt")
        if payload.get("previous_attempt_hash") != expected_previous:
            raise ArtifactCorruption("fixture attempt previous hash is invalid")
        expected_availability = attempt_sequence + 1 if payload.get("directive") == "delayed" else None
        if payload.get("available_after_attempt") != expected_availability:
            raise ArtifactCorruption("fixture attempt availability is invalid")
        late_hashes = payload.get("late_completion_hashes")
        expected_late_count = (
            1
            if previous is not None
            and previous.payload.get("directive") == "delayed"
            and previous.payload.get("available_after_attempt") == attempt_sequence
            else 0
        )
        if (
            not isinstance(late_hashes, list)
            or len(late_hashes) != expected_late_count
            or any(not isinstance(value, str) or _HASH.fullmatch(value) is None for value in late_hashes)
        ):
            raise ArtifactCorruption("fixture late completion references are invalid")
        if expected_late_count == 1:
            if previous is None:
                raise ArtifactCorruption("fixture late completion has no origin")
            late_hash = late_hashes[0]
            if not isinstance(late_hash, str):
                raise ArtifactCorruption("fixture late completion hash is invalid")
            late = self.store.read(
                late_hash,
                expected_schema_name="FixtureLateBoundaryCompletion",
            )
            if (
                late.schema_version != "1.0.0"
                or set(late.payload) != _LATE_COMPLETION_FIELDS
                or late.payload.get("boundary") != self.boundary
                or late.payload.get("idempotency_key") != idempotency_key
                or late.payload.get("observed_at_attempt") != attempt_sequence
                or late.payload.get("origin_attempt_hash") != previous.content_hash
                or late.payload.get("origin_attempt_sequence") != attempt_sequence - 1
                or late.payload.get("request_hash") != request_hash
                or late.payload.get("schedule_hash") != self.schedule_hash
                or late.payload.get("status") != "available_late_ignored"
            ):
                raise ArtifactCorruption("fixture late completion lineage is invalid")
        return attempt

    def _attempt_ref(self, idempotency_key: str, attempt_sequence: int) -> Path:
        if _HASH.fullmatch(idempotency_key) is None:
            raise ArtifactCorruption("fixture attempt idempotency key is invalid")
        return self.attempts_root / idempotency_key / f"{attempt_sequence:020d}.ref"

    @staticmethod
    def _required_hash(artifact: Artifact, field: str) -> str:
        value = artifact.payload.get(field)
        if not isinstance(value, str) or _HASH.fullmatch(value) is None:
            raise ArtifactCorruption(f"fixture attempt is missing {field}")
        return value

    @staticmethod
    def _read_ref(path: Path) -> str:
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ArtifactCorruption("fixture attempt ref is unavailable") from error
        if len(content) != 65 or not content.endswith(b"\n"):
            raise ArtifactCorruption("fixture attempt ref framing is invalid")
        try:
            value = content[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise ArtifactCorruption("fixture attempt ref is not ASCII") from error
        if _HASH.fullmatch(value) is None:
            raise ArtifactCorruption("fixture attempt ref hash is invalid")
        return value
