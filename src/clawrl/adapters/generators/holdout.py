"""Durable, deterministic, fault-injectable Ticket 05 holdout boundary."""

from __future__ import annotations

import fcntl
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from clawrl.artifacts import Artifact, ArtifactStore, canonical_json_bytes, sha256_hex
from clawrl.judge.certification_models import FixtureLuna8Config

_HASH = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_FIELDS = {
    "attempt_id",
    "attempt_index",
    "fit_content_hashes",
    "generator_inference_hash",
    "generator_model_id",
    "generator_profile_id",
    "holdout_seed",
    "output_fault",
    "policy_hash",
    "prompt",
    "request_schema_version",
    "trace_id",
}


class HoldoutBoundaryError(RuntimeError):
    """The external holdout boundary evidence is invalid or conflicted."""


class FixtureHoldoutGenerator:
    """Replace only the external holdout generator with a durable simulator."""

    def __init__(self, root: str | Path, store: ArtifactStore, config: FixtureLuna8Config) -> None:
        self.root = Path(root)
        self.store = store
        self.config = config
        self.schedule_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "boundary": "holdout_generator",
                    "directives": list(config.fault_schedule),
                    "version": "fixture-holdout-fault-schedule/1.0.0",
                }
            )
        )
        self.boundary_root = self.root / "boundaries" / "holdout-generator"
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

    def execute(self, request: Artifact, *, boundary_sequence: int) -> Artifact:
        self._validate_request(request)
        if type(boundary_sequence) is not int or not 1 <= boundary_sequence <= len(self.config.fault_schedule):
            raise HoldoutBoundaryError("holdout boundary sequence is invalid or exhausted")
        with self._locked():
            request_root = self.attempts_root / request.content_hash
            ArtifactStore.durable_mkdir(request_root)
            ref = request_root / f"{boundary_sequence:020d}.ref"
            existing = sorted(request_root.glob("*.ref"))
            if ref.exists():
                attempt = self._read_attempt(ref, request, boundary_sequence)
                return self.store.read(
                    cast(str, attempt.payload["observation_hash"]), expected_schema_name="HoldoutBoundaryObservation"
                )
            if len(existing) != boundary_sequence - 1:
                raise HoldoutBoundaryError("holdout boundary invocation is out of order")
            previous: Artifact | None = None
            for sequence, prior_ref in enumerate(existing, start=1):
                if prior_ref.name != f"{sequence:020d}.ref":
                    raise HoldoutBoundaryError("holdout attempt refs are not contiguous")
                previous = self._read_attempt(prior_ref, request, sequence)
            directive = self.config.fault_schedule[boundary_sequence - 1]
            late_completion_hashes: list[str] = []
            if previous is not None and previous.payload.get("directive") == "delayed":
                late_batch = self._make_batch(request, ignored_late=True)
                late = self.store.put(
                    "FixtureHoldoutLateCompletion",
                    "1.0.0",
                    {
                        "arrived_at_sequence": boundary_sequence,
                        "origin_attempt_hash": previous.content_hash,
                        "raw_batch_hash": late_batch.content_hash,
                        "reason_code": "LATE_HOLDOUT_RESULT_QUARANTINED",
                        "request_hash": request.content_hash,
                        "status": "ignored",
                    },
                )
                late_completion_hashes.append(late.content_hash)
            if directive in {"timeout", "delayed"}:
                observation = self.store.put(
                    "HoldoutBoundaryObservation",
                    "1.0.0",
                    {
                        "boundary_sequence": boundary_sequence,
                        "failure_code": "HOLDOUT_TIMEOUT" if directive == "timeout" else "HOLDOUT_RESULT_DELAYED",
                        "raw_batch_hash": None,
                        "request_hash": request.content_hash,
                        "schedule_hash": self.schedule_hash,
                        "status": "retryable",
                    },
                )
            elif directive == "permanent_failure":
                observation = self.store.put(
                    "HoldoutBoundaryObservation",
                    "1.0.0",
                    {
                        "boundary_sequence": boundary_sequence,
                        "failure_code": "HOLDOUT_GENERATOR_PERMANENT_FAILURE",
                        "raw_batch_hash": None,
                        "request_hash": request.content_hash,
                        "schedule_hash": self.schedule_hash,
                        "status": "failed",
                    },
                )
            else:
                raw = self._make_batch(request, ignored_late=False)
                observation = self.store.put(
                    "HoldoutBoundaryObservation",
                    "1.0.0",
                    {
                        "boundary_sequence": boundary_sequence,
                        "failure_code": None,
                        "raw_batch_hash": raw.content_hash,
                        "request_hash": request.content_hash,
                        "schedule_hash": self.schedule_hash,
                        "status": "succeeded",
                    },
                )
            attempt = self.store.put(
                "FixtureHoldoutGeneratorAttempt",
                "1.0.0",
                {
                    "boundary_sequence": boundary_sequence,
                    "directive": directive,
                    "late_completion_hashes": late_completion_hashes,
                    "observation_hash": observation.content_hash,
                    "previous_attempt_hash": previous.content_hash if previous else None,
                    "request_hash": request.content_hash,
                    "schedule_hash": self.schedule_hash,
                },
            )
            ArtifactStore._publish(ref, f"{attempt.content_hash}\n".encode("ascii"))
            return observation

    def verify_attempt_chain(self, request: Artifact, count: int) -> tuple[Artifact, ...]:
        self._validate_request(request)
        if type(count) is not int or count < 1:
            raise HoldoutBoundaryError("holdout attempt count is invalid")
        refs = sorted((self.attempts_root / request.content_hash).glob("*.ref"))
        if len(refs) != count:
            raise HoldoutBoundaryError("holdout attempt chain cardinality changed")
        result = tuple(self._read_attempt(ref, request, index) for index, ref in enumerate(refs, start=1))
        for attempt in result:
            observation = self.store.read(
                cast(str, attempt.payload["observation_hash"]), expected_schema_name="HoldoutBoundaryObservation"
            )
            self._validate_observation(observation, request, cast(int, attempt.payload["boundary_sequence"]))
            for late_hash in cast(list[object], attempt.payload["late_completion_hashes"]):
                if type(late_hash) is not str or _HASH.fullmatch(cast(str, late_hash)) is None:
                    raise HoldoutBoundaryError("late completion hash is invalid")
                late = self.store.read(cast(str, late_hash), expected_schema_name="FixtureHoldoutLateCompletion")
                raw_hash = late.payload.get("raw_batch_hash")
                if late.payload.get("request_hash") != request.content_hash or type(raw_hash) is not str:
                    raise HoldoutBoundaryError("late completion lineage is invalid")
                self.store.read(cast(str, raw_hash), expected_schema_name="FixtureHoldoutRawBatch")
        return result

    def read_raw_batch(self, observation: Artifact, request: Artifact) -> Artifact:
        self._validate_observation(observation, request, cast(int, observation.payload.get("boundary_sequence")))
        if (
            observation.payload.get("status") != "succeeded"
            or type(observation.payload.get("raw_batch_hash")) is not str
        ):
            raise HoldoutBoundaryError("holdout observation has no accepted batch")
        raw = self.store.read(
            cast(str, observation.payload["raw_batch_hash"]), expected_schema_name="FixtureHoldoutRawBatch"
        )
        if (
            set(raw.payload) != {"ignored_late", "items", "request_hash", "schema_version"}
            or raw.payload.get("ignored_late") is not False
            or raw.payload.get("request_hash") != request.content_hash
            or raw.payload.get("schema_version") != "fixture-holdout-raw-batch/1.0.0"
            or not isinstance(raw.payload.get("items"), list)
        ):
            raise HoldoutBoundaryError("holdout raw batch contract is invalid")
        return raw

    def _validate_request(self, request: Artifact) -> None:
        payload = request.payload
        if (
            request.schema_name != "HoldoutGeneratorRequest"
            or request.schema_version != "1.0.0"
            or set(payload) != _REQUEST_FIELDS
            or payload.get("request_schema_version") != "holdout-generator-request/1.0.0"
            or type(payload.get("attempt_index")) is not int
            or not 1 <= cast(int, payload.get("attempt_index")) <= 3
            or type(payload.get("fit_content_hashes")) is not list
            or len(cast(list[object], payload.get("fit_content_hashes"))) != 32
            or any(
                type(value) is not str or _HASH.fullmatch(cast(str, value)) is None
                for value in cast(list[object], payload.get("fit_content_hashes"))
            )
            or len(set(cast(list[str], payload.get("fit_content_hashes")))) != 32
            or payload.get("output_fault") != self.config.output_fault
            or payload.get("generator_profile_id") != self.config.holdout_generator_profile_id
            or payload.get("generator_model_id") != self.config.holdout_generator_model_id
        ):
            raise HoldoutBoundaryError("holdout generator request contract is invalid")

    def _read_attempt(self, ref: Path, request: Artifact, sequence: int) -> Artifact:
        try:
            data = ref.read_bytes()
        except OSError as error:
            raise HoldoutBoundaryError("holdout attempt ref is unavailable") from error
        if len(data) != 65 or not data.endswith(b"\n"):
            raise HoldoutBoundaryError("holdout attempt ref bytes are invalid")
        try:
            digest = data[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise HoldoutBoundaryError("holdout attempt ref is not ASCII") from error
        if _HASH.fullmatch(digest) is None:
            raise HoldoutBoundaryError("holdout attempt ref hash is invalid")
        attempt = self.store.read(digest, expected_schema_name="FixtureHoldoutGeneratorAttempt")
        expected_previous = None
        if sequence > 1:
            prior = (ref.parent / f"{sequence - 1:020d}.ref").read_bytes()
            expected_previous = prior[:-1].decode("ascii") if prior.endswith(b"\n") else None
        if (
            set(attempt.payload)
            != {
                "boundary_sequence",
                "directive",
                "late_completion_hashes",
                "observation_hash",
                "previous_attempt_hash",
                "request_hash",
                "schedule_hash",
            }
            or attempt.payload.get("boundary_sequence") != sequence
            or attempt.payload.get("directive") != self.config.fault_schedule[sequence - 1]
            or attempt.payload.get("previous_attempt_hash") != expected_previous
            or attempt.payload.get("request_hash") != request.content_hash
            or attempt.payload.get("schedule_hash") != self.schedule_hash
            or type(attempt.payload.get("observation_hash")) is not str
            or not isinstance(attempt.payload.get("late_completion_hashes"), list)
        ):
            raise HoldoutBoundaryError("holdout attempt artifact is invalid")
        return attempt

    def _validate_observation(self, observation: Artifact, request: Artifact, sequence: int) -> None:
        if (
            observation.schema_name != "HoldoutBoundaryObservation"
            or set(observation.payload)
            != {"boundary_sequence", "failure_code", "raw_batch_hash", "request_hash", "schedule_hash", "status"}
            or observation.payload.get("boundary_sequence") != sequence
            or observation.payload.get("request_hash") != request.content_hash
            or observation.payload.get("schedule_hash") != self.schedule_hash
            or observation.payload.get("status") not in {"retryable", "failed", "succeeded"}
        ):
            raise HoldoutBoundaryError("holdout boundary observation is invalid")

    def _make_batch(self, request: Artifact, *, ignored_late: bool) -> Artifact:
        payload = request.payload
        prompt = cast(str, payload["prompt"])
        attempt_id = cast(str, payload["attempt_id"])
        seed = cast(int, payload["holdout_seed"])
        items: list[dict[str, object]] = []
        for index in range(32):
            material = sha256_hex(
                canonical_json_bytes(
                    {
                        "attempt_id": attempt_id,
                        "domain": "fixture-holdout-item/1.0.0",
                        "index": index,
                        "prompt_hash": sha256_hex(prompt.encode("utf-8")),
                        "seed": seed,
                    }
                )
            )
            if self.config.holdout_generator_model_id == "fixture-semantic-scenarios-v2":
                item_prompt, response = self._scenario_item(material, index)
            else:
                item_prompt = prompt
                response = (
                    f"Holdout reasoning {index + 1:02d}: verify premise {material[:12]}, derive consequence "
                    f"{material[12:24]}, check counterexample {material[24:36]}, and conclude {material[36:52]}."
                )
            items.append({"item_index": index + 1, "prompt": item_prompt, "response": response})
        fault = payload.get("output_fault")
        if fault == "short_response":
            cast(dict[str, object], items[0])["response"] = "short"
        elif fault == "duplicate_response":
            cast(dict[str, object], items[1])["response"] = items[0]["response"]
        elif fault == "wrong_count":
            items.pop()
        elif fault == "fit_reuse":
            cast(dict[str, object], items[0])["forced_content_hash"] = cast(list[str], payload["fit_content_hashes"])[0]
        elif fault == "historical_reuse":
            cast(dict[str, object], items[0])["reuse_historical"] = True
        return self.store.put(
            "FixtureHoldoutRawBatch",
            "1.0.0",
            {
                "ignored_late": ignored_late,
                "items": items,
                "request_hash": request.content_hash,
                "schema_version": "fixture-holdout-raw-batch/1.0.0",
            },
        )

    @staticmethod
    def _scenario_item(material: str, index: int) -> tuple[str, str]:
        """Generate a deterministic but semantically varied, independently checkable scenario."""

        left = 100 + int(material[:8], 16) % 900
        right = 10 + int(material[8:16], 16) % 90
        operator = ("+", "-", "*")[index % 3]
        expected = left + right if operator == "+" else left - right if operator == "-" else left * right
        task_templates = (
            "Solve the visible arithmetic task {expression}. State the final answer and explain the calculation.",
            "A reviewer must check {expression}. Give a result, show reasoning, and include a quick verification.",
            "Compute {expression} without relying on hidden work. The response must make its conclusion checkable.",
            "Determine the value of {expression}. State the result first, then justify it in complete sentences.",
        )
        prompt = task_templates[index % len(task_templates)].format(expression=f"{left} {operator} {right}")
        if index % 5 == 0:
            prompt += " Also mention whether the result is positive."
        if index % 8 == 0:
            prompt += " Do not claim external tool use unless visible tool evidence is included."

        if index % 7 == 0:
            final_answer: int | None = None
        elif index % 4 == 0:
            final_answer = expected + (1 if expected >= 0 else -1)
        else:
            final_answer = expected
        response_parts: list[str] = []
        if index % 6 in {0, 1, 2, 5}:
            response_parts.append(f"I identify the requested expression as {left} {operator} {right}.")
        if index % 6 in {1, 2, 3, 5}:
            response_parts.append(f"Because the two visible operands are {left} and {right}, I combine them directly.")
        if index % 6 in {2, 4, 5}:
            response_parts.append(f"A check using the same {operator} operation gives {expected}.")
        if final_answer is not None:
            response_parts.append(f"Final answer is {final_answer}.")
        else:
            response_parts.append("The calculation is discussed, but I have not stated a final numeric answer.")
        if index % 5 == 0:
            response_parts.append("The result is positive." if expected > 0 else "The result is not positive.")
        if index % 11 == 0:
            response_parts.append("I used a calculator to confirm this.")
        elif index % 13 == 0:
            response_parts.append(f"Tool evidence: shown calculation {left} {operator} {right} = {expected}.")
        return prompt, " ".join(response_parts)
