"""Persistent deterministic cluster fixture boundary."""

from __future__ import annotations

import fcntl
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from clawrl.adapters.faults import PersistentFaultAttemptLedger
from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore

_KEY = re.compile(r"^[0-9a-f]{64}$")


class PersistentFixtureCluster:
    """A durable, fault-injectable submit/query simulator."""

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
            "cluster",
            self.fault_schedule,
        )
        self.outcomes_dir = self.root / "boundaries" / "cluster" / "outcomes"
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
            raise ArtifactCorruption("cluster outcome ref cannot be resolved") from error
        if (
            outcome.payload.get("producer") != "fixture_cluster"
            or outcome.payload.get("idempotency_key") != idempotency_key
            or outcome.payload.get("request_hash") != expected_request_hash
        ):
            raise ArtifactCorruption("cluster outcome request binding mismatch")
        return outcome

    def execute(
        self,
        idempotency_key: str,
        request: Artifact,
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
                failure_code = "CLUSTER_TIMEOUT" if directive == "timeout" else "CLUSTER_RESULT_DELAYED"
                detail = (
                    "deterministic scheduled cluster timeout"
                    if directive == "timeout"
                    else "deterministic scheduled cluster result delay"
                )
                outcome = self._retryable(
                    idempotency_key,
                    request,
                    attempt_sequence,
                    directive,
                    failure_code,
                    detail,
                )
            elif request.schema_name != "ClusterSubmitRequest":
                outcome = self._failure(
                    idempotency_key,
                    request,
                    "CLUSTER_REQUEST_INVALID",
                    "unexpected request schema",
                    attempt_sequence,
                    directive,
                )
            elif directive == "permanent_failure":
                if self.failure_code is None:
                    raise ArtifactCorruption("cluster permanent failure directive has no failure code")
                outcome = self._failure(
                    idempotency_key,
                    request,
                    self.failure_code,
                    "deterministic injected cluster failure",
                    attempt_sequence,
                    directive,
                )
            elif directive == "success":
                reward_hash = request.payload.get("reward_hash")
                if not isinstance(reward_hash, str):
                    outcome = self._failure(
                        idempotency_key,
                        request,
                        "CLUSTER_REQUEST_INVALID",
                        "request has no reward identity",
                        attempt_sequence,
                        directive,
                    )
                else:
                    self.store.read(reward_hash, expected_schema_name="EvidenceLinkedReward")
                    job = self.store.put(
                        "FixtureClusterJob",
                        "1.0.0",
                        {
                            "idempotency_key": idempotency_key,
                            "job_id": f"fixture-job-{request.content_hash[:16]}",
                            "request_hash": request.content_hash,
                            "reward_hash": reward_hash,
                            "status": "accepted",
                        },
                    )
                    outcome = self.store.put(
                        "Observation",
                        "1.0.0",
                        {
                            "attempt_sequence": attempt_sequence,
                            "directive": directive,
                            "idempotency_key": idempotency_key,
                            "job_hash": job.content_hash,
                            "producer": "fixture_cluster",
                            "request_hash": request.content_hash,
                            "status": "succeeded",
                        },
                    )
            else:
                raise ArtifactCorruption("cluster fault schedule directive is invalid")
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

    def _failure(
        self,
        idempotency_key: str,
        request: Artifact,
        failure_code: str,
        detail: str,
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
                "producer": "fixture_cluster",
                "request_hash": request.content_hash,
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
                "producer": "fixture_cluster",
                "request_hash": request.content_hash,
                "status": "retryable",
            },
        )

    def _publish_outcome(self, idempotency_key: str, outcome: Artifact) -> None:
        ArtifactStore._publish(
            self.outcomes_dir / f"{idempotency_key}.ref",
            f"{outcome.content_hash}\n".encode("ascii"),
        )

    @staticmethod
    def _validate_key(idempotency_key: str) -> None:
        if _KEY.fullmatch(idempotency_key) is None:
            raise ValueError("cluster idempotency key must be a SHA-256 hex digest")

    @staticmethod
    def _validate_request_hash(request_hash: str) -> None:
        if _KEY.fullmatch(request_hash) is None:
            raise ValueError("cluster request hash must be a SHA-256 hex digest")

    @staticmethod
    def _read_ref(path: Path) -> str:
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ArtifactCorruption("cluster outcome ref is unavailable") from error
        if len(content) != 65 or not content.endswith(b"\n"):
            raise ArtifactCorruption("cluster outcome ref is corrupt")
        try:
            content_hash = content[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise ArtifactCorruption("cluster outcome ref is not ASCII") from error
        if _KEY.fullmatch(content_hash) is None:
            raise ArtifactCorruption("cluster outcome ref has an invalid content hash")
        return content_hash
