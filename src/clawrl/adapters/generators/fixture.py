"""Stateful deterministic and fault-injectable fit generator boundary."""

from __future__ import annotations

import fcntl
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from clawrl.artifacts import Artifact, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.judge.fit_models import FixtureFitConfig, GeneratorPlan

_HASH = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_FIELDS = {
    "dataset_version_hash",
    "generator_plan_hash",
    "online_response",
    "output_fault",
    "prompt",
    "schema_version",
    "trace_id",
    "training_trace_hash",
}
_ATTEMPT_FIELDS = {
    "attempt_sequence",
    "directive",
    "late_completion_hashes",
    "observation_hash",
    "previous_attempt_hash",
    "request_hash",
    "schedule_hash",
}


class GeneratorBoundaryError(RuntimeError):
    """Generator boundary state is malformed, conflicted, or out of order."""


class FixtureFitGenerator:
    """Durable simulator replacing only the external generator system boundary."""

    def __init__(self, root: str | Path, store: ArtifactStore, config: FixtureFitConfig) -> None:
        self.root = Path(root)
        self.store = store
        self.config = config
        self.schedule_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "boundary": "fit_generator",
                    "directives": list(config.fault_schedule),
                    "version": "fixture-generator-fault-schedule/1.0.0",
                }
            )
        )
        self.boundary_root = self.root / "boundaries" / "generator"
        self.attempts_root = self.boundary_root / "attempts"
        ArtifactStore.durable_mkdir(self.attempts_root)
        self.lock_path = self.boundary_root / "adapter.lock"
        ArtifactStore.durable_touch(self.lock_path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self.lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def execute(self, request: Artifact, *, attempt_sequence: int) -> Artifact:
        """Commit one durable attempt before returning its observation."""

        if type(attempt_sequence) is not int or not 1 <= attempt_sequence <= len(self.config.fault_schedule):
            raise GeneratorBoundaryError("generator attempt sequence is invalid or exhausted")
        plan = self._validate_request(request)
        with self._locked():
            attempt_dir = self.attempts_root / request.content_hash
            ArtifactStore.durable_mkdir(attempt_dir)
            current_ref = attempt_dir / f"{attempt_sequence:020d}.ref"
            existing_refs = sorted(attempt_dir.glob("*.ref"))
            if current_ref.exists():
                attempt = self._read_attempt(current_ref, request.content_hash, attempt_sequence)
                return self.store.read(
                    cast(str, attempt.payload["observation_hash"]),
                    expected_schema_name="GeneratorBoundaryObservation",
                )
            if len(existing_refs) != attempt_sequence - 1:
                raise GeneratorBoundaryError("generator attempt predecessor is missing or out of order")
            previous: Artifact | None = None
            for sequence, ref in enumerate(existing_refs, start=1):
                if ref.name != f"{sequence:020d}.ref":
                    raise GeneratorBoundaryError("generator attempt ref order is invalid")
                previous = self._read_attempt(ref, request.content_hash, sequence)

            directive = self.config.fault_schedule[attempt_sequence - 1]
            late_hashes: list[str] = []
            if previous is not None and previous.payload.get("directive") == "delayed":
                late_raw = self._raw_batch(request, plan, ignored_late=True)
                late = self.store.put(
                    "FixtureGeneratorLateCompletion",
                    "1.0.0",
                    {
                        "arrived_at_attempt": attempt_sequence,
                        "origin_attempt_hash": previous.content_hash,
                        "raw_batch_hash": late_raw.content_hash,
                        "reason_code": "LATE_GENERATOR_RESULT_QUARANTINED",
                        "request_hash": request.content_hash,
                        "status": "ignored",
                    },
                )
                late_hashes.append(late.content_hash)

            if directive in {"timeout", "delayed"}:
                observation = self.store.put(
                    "GeneratorBoundaryObservation",
                    "1.0.0",
                    {
                        "attempt_sequence": attempt_sequence,
                        "failure_code": ("GENERATOR_TIMEOUT" if directive == "timeout" else "GENERATOR_RESULT_DELAYED"),
                        "raw_batch_hash": None,
                        "request_hash": request.content_hash,
                        "schedule_hash": self.schedule_hash,
                        "status": "retryable",
                    },
                )
            elif directive == "permanent_failure":
                observation = self.store.put(
                    "GeneratorBoundaryObservation",
                    "1.0.0",
                    {
                        "attempt_sequence": attempt_sequence,
                        "failure_code": "GENERATOR_PERMANENT_FAILURE",
                        "raw_batch_hash": None,
                        "request_hash": request.content_hash,
                        "schedule_hash": self.schedule_hash,
                        "status": "failed",
                    },
                )
            else:
                raw = self._raw_batch(request, plan, ignored_late=False)
                observation = self.store.put(
                    "GeneratorBoundaryObservation",
                    "1.0.0",
                    {
                        "attempt_sequence": attempt_sequence,
                        "failure_code": None,
                        "raw_batch_hash": raw.content_hash,
                        "request_hash": request.content_hash,
                        "schedule_hash": self.schedule_hash,
                        "status": "succeeded",
                    },
                )
            attempt = self.store.put(
                "FixtureGeneratorAttempt",
                "1.0.0",
                {
                    "attempt_sequence": attempt_sequence,
                    "directive": directive,
                    "late_completion_hashes": late_hashes,
                    "observation_hash": observation.content_hash,
                    "previous_attempt_hash": previous.content_hash if previous is not None else None,
                    "request_hash": request.content_hash,
                    "schedule_hash": self.schedule_hash,
                },
            )
            ArtifactStore._publish(current_ref, f"{attempt.content_hash}\n".encode("ascii"))
            return observation

    def verify_attempt_chain(self, request: Artifact, expected_count: int) -> tuple[Artifact, ...]:
        """Re-read every attempt, observation, and late result without writing."""

        if type(expected_count) is not int or expected_count < 1:
            raise GeneratorBoundaryError("expected attempt count is invalid")
        self._validate_request(request)
        attempt_dir = self.attempts_root / request.content_hash
        refs = sorted(attempt_dir.glob("*.ref"))
        if len(refs) != expected_count:
            raise GeneratorBoundaryError("generator attempt chain cardinality is invalid")
        attempts: list[Artifact] = []
        for sequence, ref in enumerate(refs, start=1):
            attempt = self._read_attempt(ref, request.content_hash, sequence)
            observation = self.store.read(
                cast(str, attempt.payload["observation_hash"]),
                expected_schema_name="GeneratorBoundaryObservation",
            )
            self._validate_observation(observation, request.content_hash, sequence)
            late_hashes = attempt.payload.get("late_completion_hashes")
            if not isinstance(late_hashes, list):
                raise GeneratorBoundaryError("generator late completion lineage is invalid")
            for late_hash in late_hashes:
                if type(late_hash) is not str or _HASH.fullmatch(cast(str, late_hash)) is None:
                    raise GeneratorBoundaryError("generator late completion hash is invalid")
                late = self.store.read(cast(str, late_hash), expected_schema_name="FixtureGeneratorLateCompletion")
                raw_hash = late.payload.get("raw_batch_hash")
                if (
                    set(late.payload)
                    != {
                        "arrived_at_attempt",
                        "origin_attempt_hash",
                        "raw_batch_hash",
                        "reason_code",
                        "request_hash",
                        "status",
                    }
                    or late.payload.get("request_hash") != request.content_hash
                    or late.payload.get("arrived_at_attempt") != sequence
                    or type(raw_hash) is not str
                ):
                    raise GeneratorBoundaryError("generator late completion is invalid")
                self.store.read(cast(str, raw_hash), expected_schema_name="FixtureGeneratorRawBatch")
            attempts.append(attempt)
        return tuple(attempts)

    def read_attempt(self, request: Artifact, attempt_sequence: int) -> Artifact:
        """Verify one historical prefix member while later committed attempts may exist."""

        if type(attempt_sequence) is not int or attempt_sequence < 1:
            raise GeneratorBoundaryError("generator attempt sequence is invalid")
        self._validate_request(request)
        ref = self.attempts_root / request.content_hash / f"{attempt_sequence:020d}.ref"
        if not ref.exists():
            raise GeneratorBoundaryError("generator attempt ref is missing")
        attempt = self._read_attempt(ref, request.content_hash, attempt_sequence)
        observation = self.store.read(
            cast(str, attempt.payload["observation_hash"]),
            expected_schema_name="GeneratorBoundaryObservation",
        )
        self._validate_observation(observation, request.content_hash, attempt_sequence)
        return attempt

    def committed_attempt_count(self, request: Artifact) -> int:
        """Return total durable attempts after validating contiguous ref names."""

        self._validate_request(request)
        refs = sorted((self.attempts_root / request.content_hash).glob("*.ref"))
        for sequence, ref in enumerate(refs, start=1):
            if ref.name != f"{sequence:020d}.ref":
                raise GeneratorBoundaryError("generator attempt refs have a gap")
        return len(refs)

    def read_raw_batch(self, observation: Artifact, request: Artifact) -> Artifact:
        self._validate_observation(
            observation, request.content_hash, cast(int, observation.payload.get("attempt_sequence"))
        )
        if observation.payload.get("status") != "succeeded":
            raise GeneratorBoundaryError("generator observation has no successful raw batch")
        raw_hash = observation.payload.get("raw_batch_hash")
        if type(raw_hash) is not str:
            raise GeneratorBoundaryError("generator raw batch hash is invalid")
        raw = self.store.read(cast(str, raw_hash), expected_schema_name="FixtureGeneratorRawBatch")
        if (
            raw.schema_version != "1.0.0"
            or set(raw.payload) != {"ignored_late", "items", "request_hash", "schema_version"}
            or raw.payload.get("request_hash") != request.content_hash
            or raw.payload.get("ignored_late") is not False
            or raw.payload.get("schema_version") != "fixture-generator-raw-batch/1.0.0"
            or not isinstance(raw.payload.get("items"), list)
        ):
            raise GeneratorBoundaryError("generator raw batch is invalid")
        return raw

    def _validate_request(self, request: Artifact) -> GeneratorPlan:
        if request.schema_name != "GeneratorRequest" or request.schema_version != "1.0.0":
            raise GeneratorBoundaryError("generator request schema is invalid")
        payload = request.payload
        if set(payload) != _REQUEST_FIELDS or payload.get("schema_version") != "generator-request/1.0.0":
            raise GeneratorBoundaryError("generator request fields are invalid")
        plan_hash = payload.get("generator_plan_hash")
        if type(plan_hash) is not str:
            raise GeneratorBoundaryError("generator request plan hash is invalid")
        try:
            plan_artifact = self.store.read(cast(str, plan_hash), expected_schema_name="GeneratorPlan")
            plan = GeneratorPlan.from_mapping(cast(dict[str, object], plan_artifact.payload))
        except Exception as error:
            raise GeneratorBoundaryError("generator plan cannot be verified") from error
        if (
            plan_artifact.schema_version != "1.0.0"
            or plan.artifact_payload() != plan_artifact.payload
            or payload.get("dataset_version_hash") != plan.dataset_version_hash
            or payload.get("training_trace_hash") != plan.training_trace_hash
            or payload.get("trace_id") != plan.trace_id
            or payload.get("output_fault") != self.config.output_fault
            or type(payload.get("prompt")) is not str
            or not cast(str, payload.get("prompt"))
            or type(payload.get("online_response")) is not str
            or not cast(str, payload.get("online_response"))
        ):
            raise GeneratorBoundaryError("generator request does not match its frozen plan")
        return plan

    def _read_attempt(self, ref: Path, request_hash: str, sequence: int) -> Artifact:
        try:
            ref_bytes = ref.read_bytes()
        except OSError as error:
            raise GeneratorBoundaryError("generator attempt ref is unavailable") from error
        if len(ref_bytes) != 65 or not ref_bytes.endswith(b"\n"):
            raise GeneratorBoundaryError("generator attempt ref bytes are invalid")
        try:
            attempt_hash = ref_bytes[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise GeneratorBoundaryError("generator attempt ref is not ASCII") from error
        if _HASH.fullmatch(attempt_hash) is None:
            raise GeneratorBoundaryError("generator attempt ref hash is invalid")
        attempt = self.store.read(attempt_hash, expected_schema_name="FixtureGeneratorAttempt")
        expected_previous = None
        if sequence > 1:
            prior_ref = ref.parent / f"{sequence - 1:020d}.ref"
            prior_bytes = prior_ref.read_bytes()
            expected_previous = prior_bytes[:-1].decode("ascii") if prior_bytes.endswith(b"\n") else None
        directive = self.config.fault_schedule[sequence - 1]
        if (
            attempt.schema_version != "1.0.0"
            or set(attempt.payload) != _ATTEMPT_FIELDS
            or attempt.payload.get("attempt_sequence") != sequence
            or type(attempt.payload.get("attempt_sequence")) is not int
            or attempt.payload.get("directive") != directive
            or attempt.payload.get("request_hash") != request_hash
            or attempt.payload.get("schedule_hash") != self.schedule_hash
            or attempt.payload.get("previous_attempt_hash") != expected_previous
            or type(attempt.payload.get("observation_hash")) is not str
        ):
            raise GeneratorBoundaryError("generator attempt artifact is invalid")
        return attempt

    def _validate_observation(self, observation: Artifact, request_hash: str, sequence: int) -> None:
        expected_status = (
            "retryable"
            if self.config.fault_schedule[sequence - 1] in {"timeout", "delayed"}
            else "failed"
            if self.config.fault_schedule[sequence - 1] == "permanent_failure"
            else "succeeded"
        )
        if (
            observation.schema_name != "GeneratorBoundaryObservation"
            or observation.schema_version != "1.0.0"
            or set(observation.payload)
            != {"attempt_sequence", "failure_code", "raw_batch_hash", "request_hash", "schedule_hash", "status"}
            or observation.payload.get("attempt_sequence") != sequence
            or type(observation.payload.get("attempt_sequence")) is not int
            or observation.payload.get("request_hash") != request_hash
            or observation.payload.get("schedule_hash") != self.schedule_hash
            or observation.payload.get("status") != expected_status
        ):
            raise GeneratorBoundaryError("generator boundary observation is invalid")

    def _raw_batch(self, request: Artifact, plan: GeneratorPlan, *, ignored_late: bool) -> Artifact:
        prompt = cast(str, request.payload["prompt"])
        online_response = cast(str, request.payload["online_response"])
        items: list[dict[str, JsonValue]] = []
        for slot in plan.slots:
            if slot.origin == "online_response":
                response = online_response
            else:
                reasoning_hash = sha256_hex(
                    canonical_json_bytes(
                        {
                            "prompt": prompt,
                            "slot_id": slot.slot_id,
                            "slot_seed": slot.slot_seed,
                            "trace_id": plan.trace_id,
                        }
                    )
                )
                response = (
                    f"Fit candidate {slot.slot_index + 1} derives its answer from prompt fingerprint "
                    f"{sha256_hex(prompt.encode())[:16]}. It checks constraints in order, records the "
                    f"content-derived reasoning token {reasoning_hash[:24]}, and concludes with verification "
                    f"token {reasoning_hash[24:48]}."
                )
            items.append(
                {
                    "origin": slot.origin,
                    "prompt": prompt,
                    "response": response,
                    "slot_id": slot.slot_id,
                    "slot_index": slot.slot_index,
                }
            )
        if not ignored_late:
            fault = self.config.output_fault
            if fault == "short_response":
                items[5]["response"] = "short"
            elif fault == "duplicate_response":
                items[-1]["response"] = items[-2]["response"]
            elif fault == "copied_online_response":
                items[1]["response"] = online_response
            elif fault == "wrong_count":
                items.pop()
            elif fault == "out_of_order":
                items.reverse()
        return self.store.put(
            "FixtureGeneratorRawBatch",
            "1.0.0",
            {
                "ignored_late": ignored_late,
                "items": items,
                "request_hash": request.content_hash,
                "schema_version": "fixture-generator-raw-batch/1.0.0",
            },
        )
