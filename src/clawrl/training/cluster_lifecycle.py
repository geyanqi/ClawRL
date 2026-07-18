"""Production-shaped synthetic ClusterAdapter lifecycle and readiness gate."""

from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
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

CLUSTER_OPERATIONS = (
    "submit",
    "status",
    "logs",
    "artifact",
    "checkpoint",
    "graceful_stop",
    "force_cancel",
)
_PHASES = {"TRAIN_35B", "TRAIN_122B"}
_CLOSE_MODES = {"graceful_stop", "force_cancel"}
_FAULT_DIRECTIVES = {None, "provider_error"}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


class ClusterLifecycleError(RuntimeError):
    """The persisted cluster lifecycle failed closed."""


class InjectedClusterControllerCrash(RuntimeError):
    """Controller crash after a provider outcome but before workflow commit."""


def _safe_id(value: object, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(cast(str, value)) is None:
        raise ClusterLifecycleError(f"{field} is invalid")
    return cast(str, value)


@dataclass(frozen=True, slots=True)
class FixtureClusterLifecycleConfig:
    run_id: str
    phase: Literal["TRAIN_35B", "TRAIN_122B"]
    close_mode: str = "graceful_stop"
    fault_operation: str | None = None
    fault_directive: str | None = None

    def __post_init__(self) -> None:
        _safe_id(self.run_id, "run_id")
        if self.phase not in _PHASES or self.close_mode not in _CLOSE_MODES:
            raise ClusterLifecycleError("fixture cluster phase or close mode is invalid")
        if (self.fault_operation is None) != (self.fault_directive is None):
            raise ClusterLifecycleError("cluster fault operation and directive must be configured together")
        if self.fault_operation is not None and self.fault_operation not in CLUSTER_OPERATIONS:
            raise ClusterLifecycleError("cluster fault operation is invalid")
        if self.fault_directive not in _FAULT_DIRECTIVES:
            raise ClusterLifecycleError("cluster fault directive is invalid")

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "close_mode": self.close_mode,
            "fault_directive": self.fault_directive,
            "fault_operation": self.fault_operation,
            "phase": self.phase,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class ProductionClusterLifecycleConfig:
    phase: Literal["TRAIN_35B", "TRAIN_122B"]
    jobbuilder: str | None = None
    image: str | None = None
    cfs_mount: str | None = None
    nas_mount: str | None = None
    queue: str | None = None
    resource_spec: str | None = None
    credential_ref: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ClusterLifecycleError("production cluster phase is invalid")
        for field in (
            "jobbuilder",
            "image",
            "cfs_mount",
            "nas_mount",
            "queue",
            "resource_spec",
            "credential_ref",
        ):
            value = getattr(self, field)
            if value is not None and type(value) is not str:
                raise ClusterLifecycleError(f"production cluster {field} is invalid")


@dataclass(frozen=True, slots=True)
class ClusterActionContract:
    run_id: str
    phase: str
    spec_hash: str
    operation: str
    sequence: int
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class ClusterLifecycleSnapshot:
    run_record: Artifact
    outcomes: tuple[Artifact, ...]
    recovery_evidence: tuple[Artifact, ...]


class PersistentFixtureClusterLifecycle:
    """Durable, stateful provider simulator for every typed cluster operation."""

    def __init__(self, root: str | Path, store: ArtifactStore, config: FixtureClusterLifecycleConfig) -> None:
        self.root = Path(root)
        self.store = store
        self.config = config
        self.boundary = self.root / "boundaries" / "cluster-lifecycle"
        self.outcomes = self.boundary / "outcomes"
        self.job_root = self.boundary / "jobs" / config.run_id
        ArtifactStore.durable_mkdir(self.outcomes)
        ArtifactStore.durable_mkdir(self.job_root)
        self.lock_path = self.boundary / "provider.lock"
        ArtifactStore.durable_touch(self.lock_path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self.lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def query(self, idempotency_key: str, request_hash: str) -> Artifact | None:
        self._key(idempotency_key)
        self._key(request_hash)
        ref = self.outcomes / f"{idempotency_key}.ref"
        if not ref.exists():
            return None
        outcome = self.store.read(
            ref.read_text(encoding="ascii").strip(), expected_schema_name="ClusterOperationOutcome"
        )
        if (
            outcome.payload.get("idempotency_key") != idempotency_key
            or outcome.payload.get("request_hash") != request_hash
            or outcome.payload.get("run_id") != self.config.run_id
        ):
            raise ArtifactCorruption("cluster lifecycle outcome binding changed")
        return outcome

    def execute(self, contract: ClusterActionContract, request: Artifact) -> Artifact:
        self._validate_request(contract, request)
        with self._locked():
            existing = self.query(contract.idempotency_key, request.content_hash)
            if existing is not None:
                return existing
            self.store.put(
                "ClusterProviderInvocation",
                "1.0.0",
                {
                    "idempotency_key": contract.idempotency_key,
                    "operation": contract.operation,
                    "request_hash": request.content_hash,
                    "run_id": contract.run_id,
                    "sequence": contract.sequence,
                },
            )
            if self.config.fault_operation == contract.operation and self.config.fault_directive == "provider_error":
                outcome = self._outcome(
                    contract,
                    request,
                    status="failed",
                    result_hash=None,
                    error_code=f"FIXTURE_{contract.operation.upper()}_PROVIDER_ERROR",
                )
            else:
                result = self._apply_success(contract, request)
                outcome = self._outcome(
                    contract,
                    request,
                    status="succeeded",
                    result_hash=result.content_hash,
                    error_code=None,
                )
            ArtifactStore._publish(
                self.outcomes / f"{contract.idempotency_key}.ref",
                f"{outcome.content_hash}\n".encode("ascii"),
            )
            return outcome

    def _apply_success(self, contract: ClusterActionContract, request: Artifact) -> Artifact:
        state = self._current_state(required=contract.operation != "submit")
        if contract.operation == "submit":
            if state is not None:
                raise ArtifactCorruption("cluster job was submitted twice with distinct actions")
            job = self.store.put(
                "FixtureClusterJob",
                "1.0.0",
                {
                    "job_id": f"fixture-job-{contract.idempotency_key[:16]}",
                    "phase": contract.phase,
                    "run_id": contract.run_id,
                    "spec_hash": contract.spec_hash,
                    "status": "submitted",
                },
            )
            self._publish_state(job, job.content_hash, "submitted")
            return job
        if state is None:
            raise ArtifactCorruption("cluster job state is unavailable")
        job_hash = cast(str, state.payload["job_hash"])
        current_status = cast(str, state.payload["status"])
        if current_status in {"stopped_gracefully", "canceled"}:
            raise ArtifactCorruption("cluster operation attempted after terminal provider state")
        if contract.operation == "status":
            result = self.store.put(
                "ClusterStatusSnapshot",
                "1.0.0",
                {"job_hash": job_hash, "run_id": contract.run_id, "status": "running"},
            )
            self._publish_state(result, job_hash, "running")
            return result
        if contract.operation == "logs":
            data = f"fixture cluster log run={contract.run_id} spec={contract.spec_hash}\nstatus=running\n".encode()
            blob_hash, byte_size = self.store.put_blob(data)
            return self.store.put(
                "ClusterLogBundle",
                "1.0.0",
                {
                    "blob_hash": blob_hash,
                    "byte_size": byte_size,
                    "job_hash": job_hash,
                    "redacted": True,
                    "run_id": contract.run_id,
                },
            )
        if contract.operation == "artifact":
            data = canonical_json_bytes(
                {"kind": "fixture-model-artifact", "run_id": contract.run_id, "spec_hash": contract.spec_hash}
            )
            blob_hash, byte_size = self.store.put_blob(data)
            return self.store.put(
                "ClusterArtifactManifest",
                "1.0.0",
                {
                    "blob_hash": blob_hash,
                    "byte_size": byte_size,
                    "job_hash": job_hash,
                    "run_id": contract.run_id,
                    "spec_hash": contract.spec_hash,
                },
            )
        if contract.operation == "checkpoint":
            model_hash, model_size = self.store.put_blob(f"model:{contract.run_id}:{contract.spec_hash}".encode())
            optimizer_hash, optimizer_size = self.store.put_blob(
                f"optimizer:{contract.run_id}:{contract.spec_hash}".encode()
            )
            return self.store.put(
                "ClusterCheckpointManifest",
                "1.0.0",
                {
                    "durable_sync_completed": True,
                    "job_hash": job_hash,
                    "model_blob_hash": model_hash,
                    "model_byte_size": model_size,
                    "optimizer_blob_hash": optimizer_hash,
                    "optimizer_byte_size": optimizer_size,
                    "run_id": contract.run_id,
                    "spec_hash": contract.spec_hash,
                },
            )
        if contract.operation in {"graceful_stop", "force_cancel"}:
            terminal_status = "stopped_gracefully" if contract.operation == "graceful_stop" else "canceled"
            result = self.store.put(
                "ClusterJobTerminalState",
                "1.0.0",
                {
                    "job_hash": job_hash,
                    "operation": contract.operation,
                    "run_id": contract.run_id,
                    "status": terminal_status,
                },
            )
            self._publish_state(result, job_hash, terminal_status)
            return result
        raise ArtifactCorruption("unsupported cluster lifecycle operation")

    def _current_state(self, *, required: bool) -> Artifact | None:
        ref = self.job_root / "state.ref"
        if not ref.exists():
            if required:
                raise ArtifactCorruption("cluster provider state is missing")
            return None
        state = self.store.read(ref.read_text(encoding="ascii").strip(), expected_schema_name="FixtureClusterJobState")
        if (
            set(state.payload) != {"job_hash", "last_result_hash", "run_id", "status"}
            or state.payload.get("run_id") != self.config.run_id
            or type(state.payload.get("job_hash")) is not str
        ):
            raise ArtifactCorruption("cluster provider state is corrupt")
        return state

    def _publish_state(self, result: Artifact, job_hash: str, status: str) -> None:
        state = self.store.put(
            "FixtureClusterJobState",
            "1.0.0",
            {
                "job_hash": job_hash,
                "last_result_hash": result.content_hash,
                "run_id": self.config.run_id,
                "status": status,
            },
        )
        _replace_ref(self.job_root / "state.ref", state.content_hash)

    def _outcome(
        self,
        contract: ClusterActionContract,
        request: Artifact,
        *,
        status: str,
        result_hash: str | None,
        error_code: str | None,
    ) -> Artifact:
        return self.store.put(
            "ClusterOperationOutcome",
            "1.0.0",
            {
                "error_code": error_code,
                "idempotency_key": contract.idempotency_key,
                "operation": contract.operation,
                "provider_execution_ordinal": 1,
                "request_hash": request.content_hash,
                "result_hash": result_hash,
                "run_id": contract.run_id,
                "spec_hash": contract.spec_hash,
                "status": status,
            },
        )

    @staticmethod
    def _validate_request(contract: ClusterActionContract, request: Artifact) -> None:
        if (
            request.schema_name != "ClusterOperationRequest"
            or set(request.payload)
            != {
                "idempotency_key",
                "operation",
                "phase",
                "previous_outcome_hash",
                "run_id",
                "sequence",
                "spec_hash",
            }
            or request.payload.get("idempotency_key") != contract.idempotency_key
            or request.payload.get("operation") != contract.operation
            or request.payload.get("phase") != contract.phase
            or request.payload.get("run_id") != contract.run_id
            or request.payload.get("sequence") != contract.sequence
            or request.payload.get("spec_hash") != contract.spec_hash
        ):
            raise ArtifactCorruption("cluster lifecycle request contract is invalid")

    @staticmethod
    def _key(value: str) -> None:
        if _HASH.fullmatch(value) is None:
            raise ValueError("cluster lifecycle key must be a SHA-256 digest")


class ClusterLifecycleWorkflow:
    @staticmethod
    def action_contracts(config: FixtureClusterLifecycleConfig, spec_hash: str) -> tuple[ClusterActionContract, ...]:
        if _HASH.fullmatch(spec_hash) is None:
            raise ClusterLifecycleError("ExperimentSpec hash is invalid")
        plan = (
            CLUSTER_OPERATIONS[:-1]
            if config.close_mode == "graceful_stop"
            else (*CLUSTER_OPERATIONS[:5], "force_cancel")
        )
        return tuple(
            ClusterActionContract(
                run_id=config.run_id,
                phase=config.phase,
                spec_hash=spec_hash,
                operation=operation,
                sequence=sequence,
                idempotency_key=sha256_hex(
                    canonical_json_bytes(
                        {
                            "domain": "cluster-action-idempotency/1.0.0",
                            "operation": operation,
                            "phase": config.phase,
                            "run_id": config.run_id,
                            "sequence": sequence,
                            "spec_hash": spec_hash,
                        }
                    )
                ),
            )
            for sequence, operation in enumerate(plan, start=1)
        )

    @staticmethod
    def production_readiness(root: str | Path, config: ProductionClusterLifecycleConfig) -> Artifact:
        checks: list[dict[str, str]] = []
        for code, value in (
            ("MISSING_JOBBUILDER", config.jobbuilder),
            ("MISSING_IMAGE", config.image),
            ("MISSING_CFS_MOUNT", config.cfs_mount),
            ("MISSING_NAS_MOUNT", config.nas_mount),
            ("MISSING_QUEUE", config.queue),
            ("MISSING_RESOURCE_SPEC", config.resource_spec),
            ("MISSING_CLUSTER_CREDENTIAL", config.credential_ref),
        ):
            if type(value) is not str or not cast(str, value).strip():
                checks.append({"code": code, "status": "blocked"})
        checks.append({"code": "PRODUCTION_CLUSTER_ADAPTER_UNVERIFIED", "status": "blocked"})
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": checks,
                "execution_profile": "production",
                "phase": config.phase,
                "side_effects_permitted": False,
                "status": "blocked",
                "submit_attempted": False,
            },
        )

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: FixtureClusterLifecycleConfig,
        spec_hash: str,
        crash_after_operation: str | None = None,
    ) -> ClusterLifecycleSnapshot:
        store = ArtifactStore(root)
        store.read(spec_hash, expected_schema_name="ExperimentSpec")
        run_root = Path(root) / "cluster-lifecycle-runs" / config.run_id
        ArtifactStore.durable_mkdir(run_root)
        input_artifact = store.put(
            "ClusterLifecycleRunInput",
            "1.0.0",
            {"config": config.artifact_payload(), "spec_hash": spec_hash},
        )
        cls._publish_ref(store, run_root / "input.ref", input_artifact, "CLUSTER_RUN_INPUT_CONFLICT")
        if not (run_root / "state.ref").exists():
            initial = cls._state_artifact(
                store,
                run_id=config.run_id,
                input_hash=input_artifact.content_hash,
                spec_hash=spec_hash,
                next_index=0,
                outcome_hashes=[],
                recovery_hashes=[],
                run_record_hash=None,
                status="active",
            )
            _replace_ref(run_root / "state.ref", initial.content_hash)
        return cls._drive(root, config, spec_hash, input_artifact, crash_after_operation)

    @classmethod
    def resume(cls, root: str | Path, run_id: str) -> ClusterLifecycleSnapshot:
        _safe_id(run_id, "run_id")
        store = ArtifactStore(root)
        run_root = Path(root) / "cluster-lifecycle-runs" / run_id
        try:
            input_artifact = store.read(
                (run_root / "input.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="ClusterLifecycleRunInput",
            )
        except (ArtifactCorruption, OSError) as error:
            raise ClusterLifecycleError("cluster lifecycle input cannot be recovered") from error
        config, spec_hash = cls._input_contract(input_artifact)
        if config.run_id != run_id:
            raise ClusterLifecycleError("cluster lifecycle run identity changed")
        store.read(spec_hash, expected_schema_name="ExperimentSpec")
        return cls._drive(root, config, spec_hash, input_artifact, None)

    @classmethod
    def _drive(
        cls,
        root: str | Path,
        config: FixtureClusterLifecycleConfig,
        spec_hash: str,
        input_artifact: Artifact,
        crash_after_operation: str | None,
    ) -> ClusterLifecycleSnapshot:
        run_root = Path(root) / "cluster-lifecycle-runs" / config.run_id
        lock_path = run_root / "controller.lock"
        ArtifactStore.durable_touch(lock_path)
        crash_armed = crash_after_operation
        while True:
            with lock_path.open("rb") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    snapshot = cls._advance_once(
                        root,
                        config,
                        spec_hash,
                        input_artifact,
                        crash_armed,
                    )
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            if snapshot is not None:
                return snapshot

    @classmethod
    def _advance_once(
        cls,
        root: str | Path,
        config: FixtureClusterLifecycleConfig,
        spec_hash: str,
        input_artifact: Artifact,
        crash_after_operation: str | None,
    ) -> ClusterLifecycleSnapshot | None:
        store = ArtifactStore(root)
        run_root = Path(root) / "cluster-lifecycle-runs" / config.run_id
        state = cls._read_state(store, run_root, input_artifact.content_hash, spec_hash)
        if state.payload["status"] == "terminal":
            return cls._snapshot(store, state)
        contracts = cls.action_contracts(config, spec_hash)
        next_index = cast(int, state.payload["next_operation_index"])
        if not 0 <= next_index < len(contracts):
            raise ClusterLifecycleError("cluster workflow operation index is invalid")
        contract = contracts[next_index]
        outcome_hashes = cast(list[str], state.payload["outcome_hashes"])
        recovery_hashes = cast(list[str], state.payload["recovery_evidence_hashes"])
        previous_hash = outcome_hashes[-1] if outcome_hashes else None
        request = store.put(
            "ClusterOperationRequest",
            "1.0.0",
            {
                "idempotency_key": contract.idempotency_key,
                "operation": contract.operation,
                "phase": contract.phase,
                "previous_outcome_hash": previous_hash,
                "run_id": contract.run_id,
                "sequence": contract.sequence,
                "spec_hash": contract.spec_hash,
            },
        )
        proposal = store.put(
            "ClusterActionProposal",
            "1.0.0",
            {
                "idempotency_key": contract.idempotency_key,
                "operation": contract.operation,
                "request_hash": request.content_hash,
                "run_id": contract.run_id,
                "status": "authorized_fixture_action",
            },
        )
        provider = PersistentFixtureClusterLifecycle(root, store, config)
        outcome = provider.query(contract.idempotency_key, request.content_hash)
        if outcome is not None:
            recovery = store.put(
                "ClusterRecoveryEvidence",
                "1.0.0",
                {
                    "idempotency_key": contract.idempotency_key,
                    "operation": contract.operation,
                    "outcome_hash": outcome.content_hash,
                    "provider_query_found": True,
                    "request_hash": request.content_hash,
                },
            )
            recovery_hashes = [*recovery_hashes, recovery.content_hash]
        else:
            outcome = provider.execute(contract, request)
        cls._validate_outcome(outcome, contract, request)
        if crash_after_operation == contract.operation:
            raise InjectedClusterControllerCrash(f"injected crash after provider {contract.operation}")
        outcome_hashes = [*outcome_hashes, outcome.content_hash]
        if outcome.payload["status"] == "failed":
            evidence = store.put(
                "ClusterProviderFailureEvidence",
                "1.0.0",
                {
                    "action_proposal_hash": proposal.content_hash,
                    "error_code": outcome.payload["error_code"],
                    "operation": contract.operation,
                    "outcome_hash": outcome.content_hash,
                    "request_hash": request.content_hash,
                    "run_id": config.run_id,
                    "status": "failed",
                },
            )
            record = cls._run_record(
                store,
                config,
                spec_hash,
                outcome_hashes,
                status="failed",
                closure_operation=None,
                failed_operation=contract.operation,
                failure_evidence_hash=evidence.content_hash,
            )
            terminal = cls._state_artifact(
                store,
                run_id=config.run_id,
                input_hash=input_artifact.content_hash,
                spec_hash=spec_hash,
                next_index=next_index,
                outcome_hashes=outcome_hashes,
                recovery_hashes=recovery_hashes,
                run_record_hash=record.content_hash,
                status="terminal",
            )
            _replace_ref(run_root / "state.ref", terminal.content_hash)
            return cls._snapshot(store, terminal)
        if next_index == len(contracts) - 1:
            record_status = "succeeded" if contract.operation == "graceful_stop" else "canceled"
            record = cls._run_record(
                store,
                config,
                spec_hash,
                outcome_hashes,
                status=record_status,
                closure_operation=contract.operation,
                failed_operation=None,
                failure_evidence_hash=None,
            )
            terminal = cls._state_artifact(
                store,
                run_id=config.run_id,
                input_hash=input_artifact.content_hash,
                spec_hash=spec_hash,
                next_index=len(contracts),
                outcome_hashes=outcome_hashes,
                recovery_hashes=recovery_hashes,
                run_record_hash=record.content_hash,
                status="terminal",
            )
            _replace_ref(run_root / "state.ref", terminal.content_hash)
            return cls._snapshot(store, terminal)
        advanced = cls._state_artifact(
            store,
            run_id=config.run_id,
            input_hash=input_artifact.content_hash,
            spec_hash=spec_hash,
            next_index=next_index + 1,
            outcome_hashes=outcome_hashes,
            recovery_hashes=recovery_hashes,
            run_record_hash=None,
            status="active",
        )
        _replace_ref(run_root / "state.ref", advanced.content_hash)
        return None

    @staticmethod
    def _run_record(
        store: ArtifactStore,
        config: FixtureClusterLifecycleConfig,
        spec_hash: str,
        outcome_hashes: list[str],
        *,
        status: str,
        closure_operation: str | None,
        failed_operation: str | None,
        failure_evidence_hash: str | None,
    ) -> Artifact:
        outcomes = [store.read(item, expected_schema_name="ClusterOperationOutcome") for item in outcome_hashes]
        result_by_operation = {
            cast(str, item.payload["operation"]): item.payload.get("result_hash") for item in outcomes
        }
        submit_hash = result_by_operation.get("submit")
        return store.put(
            "RunRecord",
            "1.0.0",
            {
                "artifact_hash": result_by_operation.get("artifact"),
                "checkpoint_hash": result_by_operation.get("checkpoint"),
                "closure_operation": closure_operation,
                "failed_operation": failed_operation,
                "failure_evidence_hash": failure_evidence_hash,
                "fixture_evidence": True,
                "log_hash": result_by_operation.get("logs"),
                "operation_outcome_hashes": outcome_hashes,
                "phase": config.phase,
                "production_smoke": False,
                "provider_job_hash": submit_hash,
                "run_id": config.run_id,
                "spec_hash": spec_hash,
                "status": status,
            },
        )

    @staticmethod
    def _state_artifact(
        store: ArtifactStore,
        *,
        run_id: str,
        input_hash: str,
        spec_hash: str,
        next_index: int,
        outcome_hashes: list[str],
        recovery_hashes: list[str],
        run_record_hash: str | None,
        status: str,
    ) -> Artifact:
        return store.put(
            "ClusterWorkflowState",
            "1.0.0",
            {
                "input_hash": input_hash,
                "next_operation_index": next_index,
                "outcome_hashes": outcome_hashes,
                "recovery_evidence_hashes": recovery_hashes,
                "run_id": run_id,
                "run_record_hash": run_record_hash,
                "spec_hash": spec_hash,
                "status": status,
            },
        )

    @staticmethod
    def _read_state(store: ArtifactStore, run_root: Path, input_hash: str, spec_hash: str) -> Artifact:
        try:
            state = store.read(
                (run_root / "state.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="ClusterWorkflowState",
            )
        except (ArtifactCorruption, OSError) as error:
            raise ClusterLifecycleError("cluster workflow state cannot be recovered") from error
        if (
            set(state.payload)
            != {
                "input_hash",
                "next_operation_index",
                "outcome_hashes",
                "recovery_evidence_hashes",
                "run_id",
                "run_record_hash",
                "spec_hash",
                "status",
            }
            or state.payload.get("input_hash") != input_hash
            or state.payload.get("spec_hash") != spec_hash
            or state.payload.get("run_id") != run_root.name
            or state.payload.get("status") not in {"active", "terminal"}
            or type(state.payload.get("next_operation_index")) is not int
            or not isinstance(state.payload.get("outcome_hashes"), list)
            or not isinstance(state.payload.get("recovery_evidence_hashes"), list)
        ):
            raise ClusterLifecycleError("cluster workflow state lineage is invalid")
        return state

    @staticmethod
    def _snapshot(store: ArtifactStore, state: Artifact) -> ClusterLifecycleSnapshot:
        record_hash = state.payload.get("run_record_hash")
        if type(record_hash) is not str:
            raise ClusterLifecycleError("terminal cluster state has no RunRecord")
        record = store.read(record_hash, expected_schema_name="RunRecord")
        outcomes = tuple(
            store.read(item, expected_schema_name="ClusterOperationOutcome")
            for item in cast(list[str], state.payload["outcome_hashes"])
        )
        recovery = tuple(
            store.read(item, expected_schema_name="ClusterRecoveryEvidence")
            for item in cast(list[str], state.payload["recovery_evidence_hashes"])
        )
        input_artifact = store.read(
            cast(str, state.payload["input_hash"]), expected_schema_name="ClusterLifecycleRunInput"
        )
        config, spec_hash = ClusterLifecycleWorkflow._input_contract(input_artifact)
        if config.run_id != state.payload.get("run_id") or spec_hash != state.payload.get("spec_hash"):
            raise ClusterLifecycleError("cluster workflow input identity changed")
        contracts = ClusterLifecycleWorkflow.action_contracts(config, spec_hash)
        if not outcomes or len(outcomes) > len(contracts):
            raise ClusterLifecycleError("terminal cluster outcome cardinality is invalid")
        previous_hash: str | None = None
        for index, outcome in enumerate(outcomes):
            contract = contracts[index]
            request_hash = outcome.payload.get("request_hash")
            if type(request_hash) is not str:
                raise ClusterLifecycleError("cluster outcome request identity is invalid")
            request = store.read(request_hash, expected_schema_name="ClusterOperationRequest")
            if request.payload.get("previous_outcome_hash") != previous_hash:
                raise ClusterLifecycleError("cluster operation request chain changed")
            ClusterLifecycleWorkflow._validate_outcome(outcome, contract, request)
            ClusterLifecycleWorkflow._validate_provider_result(store, contract.operation, outcome, spec_hash)
            if outcome.payload["status"] == "failed" and index != len(outcomes) - 1:
                raise ClusterLifecycleError("cluster workflow continued after a provider failure")
            previous_hash = outcome.content_hash
        outcomes_by_hash = {item.content_hash: item for item in outcomes}
        for evidence in recovery:
            outcome_hash = evidence.payload.get("outcome_hash")
            matching_outcome = outcomes_by_hash.get(outcome_hash) if type(outcome_hash) is str else None
            if (
                set(evidence.payload)
                != {"idempotency_key", "operation", "outcome_hash", "provider_query_found", "request_hash"}
                or evidence.payload.get("provider_query_found") is not True
                or matching_outcome is None
                or evidence.payload.get("idempotency_key") != matching_outcome.payload.get("idempotency_key")
                or evidence.payload.get("operation") != matching_outcome.payload.get("operation")
                or evidence.payload.get("request_hash") != matching_outcome.payload.get("request_hash")
            ):
                raise ClusterLifecycleError("cluster recovery evidence is invalid")
        ClusterLifecycleWorkflow._validate_run_record(store, record, outcomes, state, contracts)
        return ClusterLifecycleSnapshot(record, outcomes, recovery)

    @staticmethod
    def _validate_run_record(
        store: ArtifactStore,
        record: Artifact,
        outcomes: tuple[Artifact, ...],
        state: Artifact,
        contracts: tuple[ClusterActionContract, ...],
    ) -> None:
        expected = {
            "artifact_hash",
            "checkpoint_hash",
            "closure_operation",
            "failed_operation",
            "failure_evidence_hash",
            "fixture_evidence",
            "log_hash",
            "operation_outcome_hashes",
            "phase",
            "production_smoke",
            "provider_job_hash",
            "run_id",
            "spec_hash",
            "status",
        }
        if (
            set(record.payload) != expected
            or record.payload.get("run_id") != state.payload.get("run_id")
            or record.payload.get("spec_hash") != state.payload.get("spec_hash")
            or record.payload.get("phase") != contracts[0].phase
            or record.payload.get("operation_outcome_hashes") != [item.content_hash for item in outcomes]
            or record.payload.get("fixture_evidence") is not True
            or record.payload.get("production_smoke") is not False
            or record.payload.get("status") not in {"succeeded", "canceled", "failed"}
        ):
            raise ClusterLifecycleError("cluster RunRecord lineage is invalid")
        record_status = record.payload["status"]
        if record_status in {"succeeded", "canceled"}:
            expected_closure = "graceful_stop" if record_status == "succeeded" else "force_cancel"
            if (
                len(outcomes) != len(contracts)
                or record.payload.get("closure_operation") != expected_closure
                or record.payload.get("failed_operation") is not None
                or record.payload.get("failure_evidence_hash") is not None
                or any(
                    record.payload.get(field) is None
                    for field in ("provider_job_hash", "log_hash", "artifact_hash", "checkpoint_hash")
                )
            ):
                raise ClusterLifecycleError("successful cluster RunRecord is incomplete")
        else:
            last_outcome = outcomes[-1]
            if (
                last_outcome.payload.get("status") != "failed"
                or record.payload.get("failed_operation") != last_outcome.payload.get("operation")
                or type(record.payload.get("failure_evidence_hash")) is not str
                or record.payload.get("closure_operation") is not None
            ):
                raise ClusterLifecycleError("failed cluster RunRecord has invalid evidence")
            evidence = store.read(
                cast(str, record.payload["failure_evidence_hash"]),
                expected_schema_name="ClusterProviderFailureEvidence",
            )
            action_proposal_hash = evidence.payload.get("action_proposal_hash")
            if (
                set(evidence.payload)
                != {
                    "action_proposal_hash",
                    "error_code",
                    "operation",
                    "outcome_hash",
                    "request_hash",
                    "run_id",
                    "status",
                }
                or evidence.payload.get("run_id") != record.payload.get("run_id")
                or evidence.payload.get("operation") != last_outcome.payload.get("operation")
                or evidence.payload.get("outcome_hash") != last_outcome.content_hash
                or evidence.payload.get("request_hash") != last_outcome.payload.get("request_hash")
                or evidence.payload.get("error_code") != last_outcome.payload.get("error_code")
                or type(action_proposal_hash) is not str
                or evidence.payload.get("status") != "failed"
            ):
                raise ClusterLifecycleError("cluster provider failure evidence lineage is invalid")
            proposal = store.read(action_proposal_hash, expected_schema_name="ClusterActionProposal")
            if (
                set(proposal.payload) != {"idempotency_key", "operation", "request_hash", "run_id", "status"}
                or proposal.payload.get("idempotency_key") != last_outcome.payload.get("idempotency_key")
                or proposal.payload.get("operation") != last_outcome.payload.get("operation")
                or proposal.payload.get("request_hash") != last_outcome.payload.get("request_hash")
                or proposal.payload.get("run_id") != last_outcome.payload.get("run_id")
                or proposal.payload.get("status") != "authorized_fixture_action"
            ):
                raise ClusterLifecycleError("cluster failure action proposal lineage is invalid")
        for field, schema in (
            ("provider_job_hash", "FixtureClusterJob"),
            ("log_hash", "ClusterLogBundle"),
            ("artifact_hash", "ClusterArtifactManifest"),
            ("checkpoint_hash", "ClusterCheckpointManifest"),
            ("failure_evidence_hash", "ClusterProviderFailureEvidence"),
        ):
            value = record.payload.get(field)
            if value is not None:
                if type(value) is not str:
                    raise ClusterLifecycleError(f"RunRecord {field} is invalid")
                store.read(value, expected_schema_name=schema)

    @staticmethod
    def _validate_provider_result(store: ArtifactStore, operation: str, outcome: Artifact, spec_hash: str) -> None:
        if outcome.payload["status"] == "failed":
            if outcome.payload.get("result_hash") is not None or type(outcome.payload.get("error_code")) is not str:
                raise ClusterLifecycleError("failed provider result fields are invalid")
            return
        result_hash = outcome.payload.get("result_hash")
        schema_by_operation = {
            "submit": "FixtureClusterJob",
            "status": "ClusterStatusSnapshot",
            "logs": "ClusterLogBundle",
            "artifact": "ClusterArtifactManifest",
            "checkpoint": "ClusterCheckpointManifest",
            "graceful_stop": "ClusterJobTerminalState",
            "force_cancel": "ClusterJobTerminalState",
        }
        if type(result_hash) is not str:
            raise ClusterLifecycleError("successful provider result hash is invalid")
        result = store.read(result_hash, expected_schema_name=schema_by_operation[operation])
        expected_fields_by_operation = {
            "submit": {"job_id", "phase", "run_id", "spec_hash", "status"},
            "status": {"job_hash", "run_id", "status"},
            "logs": {"blob_hash", "byte_size", "job_hash", "redacted", "run_id"},
            "artifact": {"blob_hash", "byte_size", "job_hash", "run_id", "spec_hash"},
            "checkpoint": {
                "durable_sync_completed",
                "job_hash",
                "model_blob_hash",
                "model_byte_size",
                "optimizer_blob_hash",
                "optimizer_byte_size",
                "run_id",
                "spec_hash",
            },
            "graceful_stop": {"job_hash", "operation", "run_id", "status"},
            "force_cancel": {"job_hash", "operation", "run_id", "status"},
        }
        if set(result.payload) != expected_fields_by_operation[operation]:
            raise ClusterLifecycleError("provider result fields are invalid")
        if result.payload.get("run_id") != outcome.payload.get("run_id"):
            raise ClusterLifecycleError("provider result run identity changed")
        if operation in {"artifact", "checkpoint", "submit"} and result.payload.get("spec_hash") != spec_hash:
            raise ClusterLifecycleError("provider result ExperimentSpec identity changed")
        if operation == "submit" and result.payload.get("status") != "submitted":
            raise ClusterLifecycleError("cluster submit result status is invalid")
        if operation == "status" and result.payload.get("status") != "running":
            raise ClusterLifecycleError("cluster status result is invalid")
        if operation in {"graceful_stop", "force_cancel"}:
            expected_status = "stopped_gracefully" if operation == "graceful_stop" else "canceled"
            if result.payload.get("operation") != operation or result.payload.get("status") != expected_status:
                raise ClusterLifecycleError("cluster terminal result is invalid")
        if operation == "logs":
            blob_hash = result.payload.get("blob_hash")
            byte_size = result.payload.get("byte_size")
            if type(blob_hash) is not str or type(byte_size) is not int or result.payload.get("redacted") is not True:
                raise ClusterLifecycleError("cluster log bundle fields are invalid")
            store.read_blob(blob_hash, expected_size=byte_size)
        if operation == "artifact":
            blob_hash = result.payload.get("blob_hash")
            byte_size = result.payload.get("byte_size")
            if type(blob_hash) is not str or type(byte_size) is not int:
                raise ClusterLifecycleError("cluster artifact manifest fields are invalid")
            store.read_blob(blob_hash, expected_size=byte_size)
        if operation == "checkpoint":
            model_hash = result.payload.get("model_blob_hash")
            model_size = result.payload.get("model_byte_size")
            optimizer_hash = result.payload.get("optimizer_blob_hash")
            optimizer_size = result.payload.get("optimizer_byte_size")
            if (
                result.payload.get("durable_sync_completed") is not True
                or type(model_hash) is not str
                or type(model_size) is not int
                or type(optimizer_hash) is not str
                or type(optimizer_size) is not int
            ):
                raise ClusterLifecycleError("cluster checkpoint manifest fields are invalid")
            store.read_blob(model_hash, expected_size=model_size)
            store.read_blob(optimizer_hash, expected_size=optimizer_size)

    @staticmethod
    def _validate_outcome(outcome: Artifact, contract: ClusterActionContract, request: Artifact) -> None:
        if (
            set(outcome.payload)
            != {
                "error_code",
                "idempotency_key",
                "operation",
                "provider_execution_ordinal",
                "request_hash",
                "result_hash",
                "run_id",
                "spec_hash",
                "status",
            }
            or outcome.payload.get("idempotency_key") != contract.idempotency_key
            or outcome.payload.get("operation") != contract.operation
            or outcome.payload.get("request_hash") != request.content_hash
            or outcome.payload.get("run_id") != contract.run_id
            or outcome.payload.get("spec_hash") != contract.spec_hash
            or outcome.payload.get("provider_execution_ordinal") != 1
            or outcome.payload.get("status") not in {"succeeded", "failed"}
        ):
            raise ClusterLifecycleError("cluster provider outcome is invalid")
        succeeded = outcome.payload["status"] == "succeeded"
        if succeeded != (type(outcome.payload.get("result_hash")) is str) or succeeded == (
            type(outcome.payload.get("error_code")) is str
        ):
            raise ClusterLifecycleError("cluster provider outcome success/error fields are invalid")

    @staticmethod
    def _input_contract(input_artifact: Artifact) -> tuple[FixtureClusterLifecycleConfig, str]:
        if set(input_artifact.payload) != {"config", "spec_hash"}:
            raise ClusterLifecycleError("cluster run input fields are invalid")
        value = input_artifact.payload.get("config")
        spec_hash = input_artifact.payload.get("spec_hash")
        if not isinstance(value, dict) or set(value) != {
            "close_mode",
            "fault_directive",
            "fault_operation",
            "phase",
            "run_id",
        }:
            raise ClusterLifecycleError("cluster fixture config artifact is invalid")
        if type(spec_hash) is not str or _HASH.fullmatch(spec_hash) is None:
            raise ClusterLifecycleError("cluster ExperimentSpec identity is invalid")
        return (
            FixtureClusterLifecycleConfig(
                run_id=cast(str, value.get("run_id")),
                phase=cast(Literal["TRAIN_35B", "TRAIN_122B"], value.get("phase")),
                close_mode=cast(str, value.get("close_mode")),
                fault_operation=cast(str | None, value.get("fault_operation")),
                fault_directive=cast(str | None, value.get("fault_directive")),
            ),
            spec_hash,
        )

    @staticmethod
    def _publish_ref(store: ArtifactStore, ref: Path, artifact: Artifact, code: str) -> None:
        try:
            ArtifactStore._publish(ref, f"{artifact.content_hash}\n".encode("ascii"))
            return
        except ImmutableArtifactConflict:
            pass
        existing = ref.read_text(encoding="ascii").strip()
        if existing == artifact.content_hash:
            return
        conflict = store.put(
            "ClusterLifecycleConflict",
            "1.0.0",
            {
                "conflicting_hash": artifact.content_hash,
                "existing_hash": existing,
                "reason_code": code,
                "status": "corruption",
            },
        )
        raise ClusterLifecycleError(f"{code}:{conflict.content_hash}")


def _replace_ref(ref: Path, content_hash: str) -> None:
    ArtifactStore.durable_mkdir(ref.parent)
    temporary = ref.parent / f".{ref.name}.tmp"
    with temporary.open("wb") as stream:
        stream.write(f"{content_hash}\n".encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, ref)
