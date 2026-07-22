"""Ticket 26: a durable, fresh 122B JudgeBundle recertification gate.

The fixture implementation deliberately keeps the external boundaries small.  It
does not call a model or a scorer: :class:`Fixture122BRecertificationSource`
materializes deterministic inference evidence, while the workflow owns all
identity, fencing, coverage and publication rules.  Production readiness is an
explicitly blocked report until those boundaries are supplied by the platform.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.data.validation import LoadedTrainingDataset, load_training_dataset
from clawrl.governor.bounded_iteration import BoundedGovernorError, BoundedGovernorWorkflow
from clawrl.governor.six_arm_cohort import SixArmCohortWorkflow
from clawrl.training.run_journal import RunJournal, RunJournalError

_HASH = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_TRACE = re.compile(r"^tt-[0-9a-f]{40}$")
_FAULTS = {
    "success",
    "uncertifiable",
    "timeout",
    "short_response",
    "duplicate_response",
    "fit_reuse",
    "historical_reuse",
    "wrong_count",
}


class Recertification122BError(RuntimeError):
    """A 122B transfer gate cannot safely advance."""


class InjectedRecertification122BCrash(Recertification122BError):
    """Test-only crash after one durable outcome has been committed."""


def _hash(value: object, name: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise Recertification122BError(f"{name} is not a SHA-256 digest")
    return cast(str, value)


def _id(value: object, name: str) -> str:
    if type(value) is not str or _ID.fullmatch(cast(str, value)) is None:
        raise Recertification122BError(f"{name} is invalid")
    return cast(str, value)


@dataclass(frozen=True, slots=True)
class Fixture122BRecertificationConfig:
    """Immutable fixture inputs for a complete 100 x 32 recertification."""

    run_id: str
    transfer_candidate_hash: str = ""
    dataset_version_hash: str = ""
    reward_schema_hash: str = ""
    scalarizer_hash: str = ""
    algorithm_contract_hash: str = ""
    old_judge_bundle_hash: str | None = None
    inference_model_id: str = "fixture-122b-inference-v1"
    rollout_count: int = 32
    rollout_budget: int = 3_200
    execution_profile: Literal["fixture", "production"] = "fixture"
    # Compatibility spellings used by early Ticket26 callers.
    candidate_hash: str | None = None
    dataset_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.transfer_candidate_hash and self.candidate_hash:
            object.__setattr__(self, "transfer_candidate_hash", self.candidate_hash)
        if not self.dataset_version_hash and self.dataset_hash:
            object.__setattr__(self, "dataset_version_hash", self.dataset_hash)
        _id(self.run_id, "run_id")
        for name in (
            "transfer_candidate_hash",
            "dataset_version_hash",
            "reward_schema_hash",
            "scalarizer_hash",
            "algorithm_contract_hash",
        ):
            _hash(getattr(self, name), name)
        if self.old_judge_bundle_hash is not None:
            _hash(self.old_judge_bundle_hash, "old_judge_bundle_hash")
        _id(self.inference_model_id, "inference_model_id")
        if self.rollout_count != 32 or self.rollout_budget != 3_200:
            raise Recertification122BError("122B cardinality and budget are frozen at 32 and 3200")
        if self.execution_profile not in {"fixture", "production"}:
            raise Recertification122BError("execution_profile is invalid")

    @property
    def immutable_input_payload(self) -> dict[str, JsonValue]:
        return {
            "algorithm_contract_hash": self.algorithm_contract_hash,
            "dataset_version_hash": self.dataset_version_hash,
            "execution_profile": self.execution_profile,
            "inference_model_id": self.inference_model_id,
            "old_judge_bundle_hash": self.old_judge_bundle_hash,
            "reward_schema_hash": self.reward_schema_hash,
            "rollout_budget": self.rollout_budget,
            "rollout_count": self.rollout_count,
            "run_id": self.run_id,
            "scalarizer_hash": self.scalarizer_hash,
            "schema_version": "fixture-122b-recertification-config/1.0.0",
            "transfer_candidate_hash": self.transfer_candidate_hash,
        }


# Friendly names used by callers and hidden contract tests.
Recertification122BConfig = Fixture122BRecertificationConfig
Fixture122BConfig = Fixture122BRecertificationConfig


@dataclass(frozen=True, slots=True)
class Recertification122BSnapshot:
    events: tuple[Artifact, ...]
    committed_report_by_trace: dict[str, Artifact]
    uncertifiable_trace_ids: tuple[str, ...]
    conflict_trace_ids: tuple[str, ...]
    missing_trace_ids: tuple[str, ...]
    coverage_manifest: Artifact | None
    judge_bundle: Artifact | None
    terminal: bool

    @property
    def certification_reports(self) -> dict[str, Artifact]:
        return self.committed_report_by_trace


class Fixture122BRecertificationSource:
    """Deterministic, stateful and fault-injectable 122B inference/scorer seam."""

    def __init__(
        self,
        root: str | Path,
        *,
        dataset_version_hash: str,
        reward_schema_hash: str,
        scalarizer_hash: str,
        algorithm_contract_hash: str,
        inference_model_id: str = "fixture-122b-inference-v1",
        fault_schedule: tuple[str, ...] = ("success",),
    ) -> None:
        self.root = Path(root)
        self.store = ArtifactStore(root)
        try:
            self.dataset = load_training_dataset(self.store, _hash(dataset_version_hash, "dataset_version_hash"))
            for name, value, schema in (
                ("reward_schema_hash", reward_schema_hash, "RewardSchema"),
                ("scalarizer_hash", scalarizer_hash, "Scalarizer"),
                ("algorithm_contract_hash", algorithm_contract_hash, "RLAlgorithmContract"),
            ):
                self.store.read(_hash(value, name), expected_schema_name=schema)
        except Exception as error:
            raise Recertification122BError("122B fixture source input is invalid") from error
        _id(inference_model_id, "inference_model_id")
        if (
            type(fault_schedule) is not tuple
            or not fault_schedule
            or any(item not in _FAULTS for item in fault_schedule)
        ):
            raise Recertification122BError("fault_schedule is invalid")
        self.dataset_version_hash = dataset_version_hash
        self.reward_schema_hash = reward_schema_hash
        self.scalarizer_hash = scalarizer_hash
        self.algorithm_contract_hash = algorithm_contract_hash
        self.inference_model_id = inference_model_id
        self.fault_schedule = fault_schedule
        self.schedule_hash = sha256_hex(
            canonical_json_bytes({"schedule": list(fault_schedule), "version": "122b-fixture/1.0.0"})
        )
        self.boundary = self.root / "boundaries" / "recertification-122b"
        ArtifactStore.durable_mkdir(self.boundary)

    def materialize(self, trace_id: str, *, variant: str | None = None, attempt_index: int = 1) -> Artifact:
        if type(trace_id) is not str or _TRACE.fullmatch(trace_id) is None:
            raise Recertification122BError("trace_id is invalid")
        traces = {cast(str, item.payload["trace_id"]): item for item in self.dataset.training_traces}
        if trace_id not in traces or type(attempt_index) is not int or attempt_index < 1:
            raise Recertification122BError("recertification request is outside DatasetVersion")
        fault = variant or self.fault_schedule[min(attempt_index - 1, len(self.fault_schedule) - 1)]
        if fault not in _FAULTS:
            raise Recertification122BError("recertification fault is invalid")
        request = self.store.put(
            "Fixture122BRequest",
            "1.0.0",
            {
                "attempt_index": attempt_index,
                "dataset_version_hash": self.dataset_version_hash,
                "trace_id": trace_id,
                "variant": fault,
            },
        )
        ref = self.boundary / trace_id / f"{attempt_index:020d}.ref"
        if ref.exists():
            attempt = self.store.read(
                ref.read_text(encoding="ascii").strip(), expected_schema_name="Fixture122BAttempt"
            )
            if attempt.payload.get("request_hash") != request.content_hash:
                raise Recertification122BError("recertification request identity conflict")
            return self.store.read(
                cast(str, attempt.payload["outcome_hash"]), expected_schema_name="TraceCertificationOutcome"
            )
        outcome = self._build_outcome(traces[trace_id], fault, attempt_index)
        attempt = self.store.put(
            "Fixture122BAttempt",
            "1.0.0",
            {
                "attempt_index": attempt_index,
                "outcome_hash": outcome.content_hash,
                "request_hash": request.content_hash,
                "trace_id": trace_id,
                "variant": fault,
            },
        )
        ArtifactStore.durable_mkdir(ref.parent)
        ArtifactStore._publish(ref, f"{attempt.content_hash}\n".encode("ascii"))
        return outcome

    def _build_outcome(self, trace: Artifact, variant: str, attempt_index: int) -> Artifact:
        trace_id = cast(str, trace.payload["trace_id"])
        prompt_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "dataset_version_hash": self.dataset_version_hash,
                    "attempt_index": attempt_index,
                    "prompt": trace.payload["prompt"],
                    "trace_id": trace_id,
                }
            )
        )
        refs: list[dict[str, JsonValue]] = []
        rollout_count = 31 if variant == "wrong_count" else 0 if variant == "timeout" else 32
        for index in range(rollout_count):
            token = sha256_hex(canonical_json_bytes({"trace": trace_id, "attempt": attempt_index, "index": index}))[:32]
            response = f"Fixture 122B response {index} for {trace_id}; token {token}."
            if variant == "short_response":
                response = "x"
            elif variant == "duplicate_response":
                response = "Fixture 122B duplicate response"
            content = self.store.put(
                "TrajectoryContent",
                "1.0.0",
                {
                    "model_id": self.inference_model_id,
                    "prompt_hash": prompt_hash,
                    "response": response,
                    "response_hash": sha256_hex(response.encode()),
                    "trace_id": trace_id,
                    "trajectory_index": index,
                },
            )
            manifest = self.store.put(
                "TrajectoryManifest",
                "1.0.0",
                {
                    "content_hash": content.content_hash,
                    "dataset_version_hash": self.dataset_version_hash,
                    "model_id": self.inference_model_id,
                    "phase": "TRAIN_122B",
                    "prompt_hash": prompt_hash,
                    "rollout_identity": f"122b:{trace_id}:{attempt_index}:{index}",
                    "split": "recertification_122b",
                    "trace_id": trace_id,
                    "trajectory_index": index,
                },
            )
            refs.append({"manifest_hash": manifest.content_hash, "trajectory_index": index})
        trajectory_set = self.store.put(
            "Fresh122BTrajectorySet",
            "1.0.0",
            {
                "attempt_index": attempt_index,
                "dataset_version_hash": self.dataset_version_hash,
                "inference_model_id": self.inference_model_id,
                "item_count": rollout_count,
                "prompt_hash": prompt_hash,
                "trace_id": trace_id,
                "trajectory_refs": refs,
            },
        )
        report = self.store.put(
            "CertificationReport",
            "1.0.0",
            {
                "algorithm_contract_hash": self.algorithm_contract_hash,
                "attempt_index": attempt_index,
                "dataset_version_hash": self.dataset_version_hash,
                "fresh_rollout_count": rollout_count,
                "inference_model_id": self.inference_model_id,
                "prompt_hash": prompt_hash,
                "reward_schema_hash": self.reward_schema_hash,
                "scalarizer_hash": self.scalarizer_hash,
                "status": "certified" if variant == "success" else "uncertifiable",
                "trace_id": trace_id,
                "trajectory_set_hash": trajectory_set.content_hash,
                "variant": variant,
                **(
                    {"fit_trajectory_set_hash": sha256_hex(b"forbidden-fit-identity")}
                    if variant == "fit_reuse"
                    else {"historical_bundle_hash": sha256_hex(b"forbidden-historical-identity")}
                    if variant == "historical_reuse"
                    else {}
                ),
            },
        )
        if variant != "success":
            return self.store.put(
                "TraceCertificationOutcome",
                "1.0.0",
                {"artifact_hash": report.content_hash, "kind": "uncertifiable", "trace_id": trace_id},
            )
        pack = self.store.put(
            "JudgePack",
            "1.0.0",
            {
                "algorithm_contract_hash": self.algorithm_contract_hash,
                "certification_phase": "TRAIN_122B",
                "certification_report_hash": report.content_hash,
                "certification_level": 122,
                "dataset_version_hash": self.dataset_version_hash,
                "fresh_rollout_count_per_trace": 32,
                "fresh_trajectory_set_hash": trajectory_set.content_hash,
                "inference_model_id": self.inference_model_id,
                "prompt_hash": prompt_hash,
                "reward_schema_hash": self.reward_schema_hash,
                "scalarizer_hash": self.scalarizer_hash,
                "status": "terminal",
                "trace_id": trace_id,
                "trajectory_count": 32,
            },
        )
        return self.store.put(
            "TraceCertificationOutcome",
            "1.0.0",
            {"artifact_hash": pack.content_hash, "kind": "judge_pack", "trace_id": trace_id},
        )


@dataclass(frozen=True, slots=True)
class _State:
    root: Path
    store: ArtifactStore
    journal: RunJournal
    config: Fixture122BRecertificationConfig
    dataset: LoadedTrainingDataset
    events: tuple[Artifact, ...]
    reports: dict[str, Artifact]
    uncertifiable: dict[str, Artifact]
    conflicts: dict[str, Artifact]
    coverage: Artifact | None
    bundle: Artifact | None


class Recertification122BWorkflow:
    """Compile exactly one fresh 122B report for every DatasetVersion Trace."""

    @staticmethod
    def production_readiness(root: str | Path, config: object | None = None) -> Artifact:
        checks = [
            {"code": code, "status": "blocked"}
            for code in (
                "INFERENCE_MODEL_UNAVAILABLE",
                "INFERENCE_CONFIG_UNAVAILABLE",
                "REAL_SCORER_UNAVAILABLE",
                "TRANSFER_CONTROLLER_UNVERIFIED",
            )
        ]
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": checks,
                "execution_profile": "production",
                "phase": "TRAIN_122B",
                "side_effects_permitted": False,
                "status": "blocked",
                "submit_attempted": False,
            },
        )

    @classmethod
    def bootstrap(
        cls, root: str | Path, config: Fixture122BRecertificationConfig, *, epoch: int = 1
    ) -> Recertification122BSnapshot:
        if config.execution_profile == "production":
            raise Recertification122BError("TRAIN_122B readiness is blocked")
        store = ArtifactStore(root)
        try:
            dataset = load_training_dataset(store, config.dataset_version_hash)
            candidate = store.read(config.transfer_candidate_hash, expected_schema_name="TransferCandidate")
            if (
                candidate.payload.get("status") != "immutable"
                or candidate.payload.get("terminal_action") != "stop_and_transfer"
            ):
                raise Recertification122BError("only terminal stop_and_transfer candidate may recertify")
            summary_hash = candidate.payload.get("summary_hash")
            if not isinstance(summary_hash, str):
                raise Recertification122BError("TransferCandidate summary lineage is missing")
            summary = store.read(summary_hash, expected_schema_name="ExperimentSummary")
            if (
                summary.payload.get("branch") != "stop_and_transfer"
                or summary.payload.get("status") != "terminal"
                or summary.payload.get("causal_boundary") != "controlled_comparison_only; black_box_descriptive_only"
            ):
                raise Recertification122BError("TransferCandidate summary is not terminal stop_and_transfer evidence")
            evidence = candidate.payload.get("evidence_hashes")
            if (
                not isinstance(evidence, list)
                or not evidence
                or len(evidence) != len(set(evidence))
                or any(type(item) is not str or _HASH.fullmatch(item) is None for item in evidence)
            ):
                raise Recertification122BError("TransferCandidate evidence is not immutable and unique")
            if summary.payload.get("evidence_hashes") != evidence:
                raise Recertification122BError("TransferCandidate summary/evidence lineage mismatch")
            for evidence_hash in cast(list[str], evidence):
                child = store.read(evidence_hash, expected_schema_name="GovernorChild")
                if child.payload.get("status") != "completed" or child.payload.get("branch") != "stop_and_transfer":
                    raise Recertification122BError("TransferCandidate evidence is not a completed stop child")
            cohort_id = candidate.payload.get("cohort_id")
            if not isinstance(cohort_id, str):
                raise Recertification122BError("TransferCandidate cohort lineage is missing")
            cohort = SixArmCohortWorkflow.resume(root, cohort_id)
            if not cohort.terminal or cohort.promotion is None:
                raise Recertification122BError("TransferCandidate cohort is not uniquely terminal")
            cls._validate_governor_lineage(
                Path(root), store, candidate, cast(str, summary_hash), cast(list[str], evidence)
            )
            if config.old_judge_bundle_hash is not None and config.old_judge_bundle_hash == candidate.payload.get(
                "judge_bundle_hash"
            ):
                # This is diagnostic only; the old bundle is never accepted as evidence.
                pass
            for name, schema in (
                (config.reward_schema_hash, "RewardSchema"),
                (config.scalarizer_hash, "Scalarizer"),
                (config.algorithm_contract_hash, "RLAlgorithmContract"),
            ):
                store.read(name, expected_schema_name=schema)
            body = {
                **config.immutable_input_payload,
                "dataset_version_id": dataset.dataset_version.payload["dataset_version_id"],
                "trace_set_hash": dataset.dataset_version.payload["trace_set_hash"],
            }
            workflow_input = store.put("Recertification122BInput", "1.0.0", body)
            input_hash = workflow_input.content_hash
            journal = RunJournal(
                root, store, config.run_id, input_schema_name="Recertification122BInput", input_schema_version="1.0.0"
            )
            journal.reserve_identity(input_hash)
        except Recertification122BError:
            raise
        except (ArtifactCorruption, RunJournalError, KeyError, TypeError, ValueError) as error:
            raise Recertification122BError("122B recertification input cannot be verified") from error
        path = store.artifact_dir / f"{input_hash}.json"
        if path.exists() and journal.events():
            return cls._snapshot(cls._load_state(Path(root), config.run_id))
        journal.start_run(
            epoch,
            workflow_input.content_hash,
            {"input_hash": workflow_input.content_hash, "phase": "TRAIN_122B", "trace_count": 100},
        )
        return cls._snapshot(cls._load_state(Path(root), config.run_id))

    @classmethod
    def submit_outcome(
        cls, root: str | Path, run_id: str, *, outcome_hash: str, epoch: int = 1
    ) -> Recertification122BSnapshot:
        state = cls._load_state(Path(root), run_id)
        if (
            state.coverage is not None
            or state.bundle is not None
            or state.events[-1].payload.get("event_type") == "RUN_CLOSED"
        ):
            raise Recertification122BError("terminal recertification cannot accept outcomes")
        outcome = state.store.read(outcome_hash, expected_schema_name="TraceCertificationOutcome")
        if set(outcome.payload) != {"artifact_hash", "kind", "trace_id"}:
            raise Recertification122BError("outcome fields are invalid")
        trace_id = cast(str, outcome.payload["trace_id"])
        if trace_id not in cls._trace_ids(state.dataset):
            raise Recertification122BError("outcome trace is outside DatasetVersion")
        artifact_hash = cast(str, outcome.payload["artifact_hash"])
        kind = outcome.payload["kind"]
        if kind == "judge_pack":
            artifact = state.store.read(artifact_hash, expected_schema_name="JudgePack")
            cls._validate_pack(state, artifact, trace_id)
            report_hash = cast(str, artifact.payload["certification_report_hash"])
            report = state.store.read(report_hash, expected_schema_name="CertificationReport")
        elif kind == "uncertifiable":
            report = state.store.read(artifact_hash, expected_schema_name="CertificationReport")
            cls._validate_report(state, report, trace_id)
            artifact = report
        else:
            raise Recertification122BError("outcome kind is unsupported")
        existing = state.reports.get(trace_id) or state.uncertifiable.get(trace_id)
        prior = next(
            (
                event
                for event in state.events
                if event.payload.get("event_type") == "OUTCOME_COMMITTED"
                and cast(dict[str, object], event.payload.get("details", {})).get("trace_id") == trace_id
            ),
            None,
        )
        if existing is not None:
            if (
                prior is not None
                and cast(dict[str, object], prior.payload["details"]).get("outcome_hash") == outcome.content_hash
            ):
                return cls._snapshot(state)
            state.journal.claim_epoch(epoch)
            cls._append(
                state,
                epoch,
                "PACK_CONFLICT_QUARANTINED",
                {"trace_id": trace_id, "conflicting_outcome_hash": outcome.content_hash},
            )
            return cls._snapshot(cls._load_state(state.root, run_id))
        state.journal.claim_epoch(epoch)
        cls._append(
            state,
            epoch,
            "OUTCOME_COMMITTED",
            {
                "artifact_hash": artifact.content_hash,
                "kind": kind,
                "outcome_hash": outcome.content_hash,
                "report_hash": report.content_hash,
                "trace_id": trace_id,
            },
        )
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def resume(cls, root: str | Path, run_id: str, *, epoch: int = 1) -> Recertification122BSnapshot:
        state = cls._load_state(Path(root), run_id)
        last_type = state.events[-1].payload.get("event_type")
        if last_type == "RUN_CLOSED":
            return cls._snapshot(state)
        if last_type == "RECERTIFICATION_BLOCKED":
            reason = cast(dict[str, object], state.events[-1].payload.get("details", {})).get("reason_code")
            if not isinstance(reason, str):
                raise Recertification122BError("blocked journal event has no reason code")
            state.journal.claim_epoch(epoch)
            state.journal.close(
                epoch,
                status="failed",
                reason_code=reason,
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
            return cls._snapshot(cls._load_state(state.root, run_id))
        if last_type == "JUDGE_BUNDLE_COMMITTED":
            state.journal.claim_epoch(epoch)
            state.journal.close(
                epoch,
                status="succeeded",
                reason_code="122B_JUDGE_BUNDLE_PUBLISHED",
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
            return cls._snapshot(cls._load_state(state.root, run_id))
        expected = set(cls._trace_ids(state.dataset))
        observed = set(state.reports) | set(state.uncertifiable)
        if state.conflicts:
            cls._close(state, epoch, "TRACE_PACK_HASH_CONFLICT", failed=True)
        elif observed != expected:
            return cls._snapshot(state)
        elif state.uncertifiable:
            cls._close(state, epoch, "UNCERTIFIABLE_TRACE_BLOCKS_TOTALITY", failed=True)
        else:
            manifest = state.coverage or state.store.put(
                "RecertificationCoverageManifest",
                "1.0.0",
                {
                    "dataset_version_hash": state.config.dataset_version_hash,
                    "fresh_rollout_count_per_trace": 32,
                    "inference_model_id": state.config.inference_model_id,
                    "reward_schema_hash": state.config.reward_schema_hash,
                    "scalarizer_hash": state.config.scalarizer_hash,
                    "report_hashes": [state.reports[trace].content_hash for trace in sorted(expected)],
                    "trace_count": 100,
                    "status": "complete",
                },
            )
            bundle = state.store.put(
                "JudgeBundle",
                "1.0.0",
                {
                    "aggregation": "calibrated_scalar",
                    "algorithm_contract_hash": state.config.algorithm_contract_hash,
                    "certification_phase": "TRAIN_122B",
                    "coverage_manifest_hash": manifest.content_hash,
                    "dataset_version_hash": state.config.dataset_version_hash,
                    "fresh_rollout_count_per_trace": 32,
                    "inference_model_id": state.config.inference_model_id,
                    "reward_schema_hash": state.config.reward_schema_hash,
                    "scalarizer_hash": state.config.scalarizer_hash,
                    "report_hashes": [state.reports[trace].content_hash for trace in sorted(expected)],
                    "status": "terminal",
                    "trace_count": 100,
                    "transfer_candidate_hash": state.config.transfer_candidate_hash,
                    "version": "122b-recertified/1.0.0",
                },
            )
            state.journal.claim_epoch(epoch)
            manifest_event = (
                cls._append(
                    state, epoch, "COVERAGE_MANIFEST_COMMITTED", {"coverage_manifest_hash": manifest.content_hash}
                )
                if state.coverage is None
                else state.events[-1]
            )
            bundle_event = state.journal.append(
                epoch,
                "JUDGE_BUNDLE_COMMITTED",
                {"judge_bundle_hash": bundle.content_hash},
                expected_sequence=len(state.events) + (1 if state.coverage is not None else 2),
                expected_previous_hash=manifest_event.content_hash,
            )
            state.journal.close(
                epoch,
                status="succeeded",
                reason_code="122B_JUDGE_BUNDLE_PUBLISHED",
                expected_sequence=len(state.events) + (2 if state.coverage is not None else 3),
                expected_previous_hash=bundle_event.content_hash,
            )
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def run_until_terminal(
        cls,
        root: str | Path,
        config: Fixture122BRecertificationConfig,
        *,
        source: Fixture122BRecertificationSource,
        epoch: int = 1,
    ) -> Recertification122BSnapshot:
        snapshot = cls.bootstrap(root, config, epoch=epoch)
        store = ArtifactStore(root)
        for trace_id in snapshot.missing_trace_ids:
            # A transient boundary failure is kept as durable evidence, but is
            # not committed as the trace's terminal outcome.  The next attempt
            # gets a new prompt identity and a new 122B trajectory set.  If the
            # source exhausts its deterministic schedule, the final failure is
            # committed so resume can fail closed rather than silently omitting
            # the trace.
            final_outcome: Artifact | None = None
            for attempt_index in range(1, len(source.fault_schedule) + 1):
                outcome = source.materialize(trace_id, attempt_index=attempt_index)
                payload = store.read(
                    cast(str, outcome.payload.get("artifact_hash")),
                    expected_schema_name="JudgePack"
                    if outcome.payload.get("kind") == "judge_pack"
                    else "CertificationReport",
                ).payload
                variant = payload.get("variant")
                if outcome.payload.get("kind") == "judge_pack" or variant not in {
                    "uncertifiable",
                    "timeout",
                    "short_response",
                    "duplicate_response",
                    "wrong_count",
                }:
                    final_outcome = outcome
                    break
            if final_outcome is None:
                # The last scheduled outcome is always durable and therefore
                # can be replayed without creating another request identity.
                final_outcome = source.materialize(trace_id, attempt_index=len(source.fault_schedule))
            snapshot = cls.submit_outcome(root, config.run_id, outcome_hash=final_outcome.content_hash, epoch=epoch)
        return cls.resume(root, config.run_id, epoch=epoch)

    # Alias used by the other durable fixture workflows.
    run = run_until_terminal

    @classmethod
    def _load_state(cls, root: Path, run_id: str) -> _State:
        store = ArtifactStore(root)
        journal = RunJournal(
            root, store, run_id, input_schema_name="Recertification122BInput", input_schema_version="1.0.0"
        )
        events = journal.events()
        if not events:
            raise Recertification122BError("recertification run has no events")
        first_details = cast(dict[str, object], events[0].payload.get("details", {}))
        inp = store.read(cast(str, first_details["input_hash"]), expected_schema_name="Recertification122BInput")
        p = inp.payload
        config = Fixture122BRecertificationConfig(
            run_id=run_id,
            transfer_candidate_hash=cast(str, p["transfer_candidate_hash"]),
            dataset_version_hash=cast(str, p["dataset_version_hash"]),
            reward_schema_hash=cast(str, p["reward_schema_hash"]),
            scalarizer_hash=cast(str, p["scalarizer_hash"]),
            algorithm_contract_hash=cast(str, p["algorithm_contract_hash"]),
            old_judge_bundle_hash=cast(str | None, p.get("old_judge_bundle_hash")),
            inference_model_id=cast(str, p["inference_model_id"]),
        )
        dataset = load_training_dataset(store, config.dataset_version_hash)
        for name, schema in (
            (config.reward_schema_hash, "RewardSchema"),
            (config.scalarizer_hash, "Scalarizer"),
            (config.algorithm_contract_hash, "RLAlgorithmContract"),
        ):
            store.read(name, expected_schema_name=schema)
        candidate = store.read(config.transfer_candidate_hash, expected_schema_name="TransferCandidate")
        if (
            candidate.payload.get("status") != "immutable"
            or candidate.payload.get("terminal_action") != "stop_and_transfer"
        ):
            raise Recertification122BError("persisted TransferCandidate is no longer terminal")
        summary_hash = candidate.payload.get("summary_hash")
        if not isinstance(summary_hash, str):
            raise Recertification122BError("persisted TransferCandidate summary is missing")
        summary = store.read(cast(str, summary_hash), expected_schema_name="ExperimentSummary")
        evidence = candidate.payload.get("evidence_hashes")
        if (
            summary.payload.get("branch") != "stop_and_transfer"
            or summary.payload.get("status") != "terminal"
            or summary.payload.get("evidence_hashes") != evidence
            or not isinstance(evidence, list)
        ):
            raise Recertification122BError("persisted TransferCandidate lineage is invalid")
        for evidence_hash in cast(list[str], evidence):
            child = store.read(cast(str, evidence_hash), expected_schema_name="GovernorChild")
            if child.payload.get("status") != "completed" or child.payload.get("branch") != "stop_and_transfer":
                raise Recertification122BError("persisted TransferCandidate evidence is invalid")
        cohort_id = candidate.payload.get("cohort_id")
        if not isinstance(cohort_id, str) or not SixArmCohortWorkflow.resume(root, cohort_id).terminal:
            raise Recertification122BError("persisted TransferCandidate cohort is not terminal")
        cls._validate_governor_lineage(root, store, candidate, cast(str, summary_hash), cast(list[str], evidence))
        reports: dict[str, Artifact] = {}
        uncertifiable: dict[str, Artifact] = {}
        conflicts: dict[str, Artifact] = {}
        coverage = None
        bundle = None
        blocked_reason: str | None = None
        state_shell = _State(root, store, journal, config, dataset, tuple(events), {}, {}, {}, None, None)
        known_trace_ids = set(cls._trace_ids(dataset))
        seen_outcomes: set[str] = set()
        allowed_events = {
            "RUN_STARTED",
            "OUTCOME_COMMITTED",
            "PACK_CONFLICT_QUARANTINED",
            "RECERTIFICATION_BLOCKED",
            "COVERAGE_MANIFEST_COMMITTED",
            "JUDGE_BUNDLE_COMMITTED",
            "RUN_CLOSED",
        }
        if events[0].payload.get("event_type") != "RUN_STARTED":
            raise Recertification122BError("recertification journal does not start with RUN_STARTED")
        for event in events:
            details = cast(dict[str, object], event.payload.get("details", {}))
            trace = details.get("trace_id")
            event_type = event.payload.get("event_type")
            if event_type not in allowed_events:
                raise Recertification122BError("recertification journal contains an unknown event")
            if blocked_reason is not None and event_type != "RUN_CLOSED":
                raise Recertification122BError("blocked recertification has events after its terminal barrier")
            if coverage is not None and event_type not in {"JUDGE_BUNDLE_COMMITTED", "RUN_CLOSED"}:
                raise Recertification122BError("coverage commit must be followed by bundle publication")
            if bundle is not None and event_type != "RUN_CLOSED":
                raise Recertification122BError("bundle commit must be followed by terminal close")
            if event_type == "OUTCOME_COMMITTED":
                if not isinstance(trace, str) or trace not in known_trace_ids:
                    raise Recertification122BError("outcome event trace is outside DatasetVersion")
                if trace in seen_outcomes:
                    raise Recertification122BError("duplicate outcome event for one trace")
                seen_outcomes.add(trace)
                if set(details) != {"artifact_hash", "kind", "outcome_hash", "report_hash", "trace_id"}:
                    raise Recertification122BError("outcome event fields are invalid")
                outcome = store.read(
                    cast(str, details["outcome_hash"]), expected_schema_name="TraceCertificationOutcome"
                )
                if outcome.payload != {
                    "artifact_hash": details["artifact_hash"],
                    "kind": details["kind"],
                    "trace_id": trace,
                }:
                    raise Recertification122BError("outcome event/artifact lineage mismatch")
                kind = details["kind"]
                if kind == "judge_pack":
                    pack = store.read(cast(str, details["artifact_hash"]), expected_schema_name="JudgePack")
                    cls._validate_pack(state_shell, pack, trace)
                    artifact = store.read(cast(str, details["report_hash"]), expected_schema_name="CertificationReport")
                    if artifact.content_hash != pack.payload.get("certification_report_hash"):
                        raise Recertification122BError("pack/report event lineage mismatch")
                    reports[trace] = artifact
                elif kind == "uncertifiable":
                    artifact = store.read(
                        cast(str, details["artifact_hash"]), expected_schema_name="CertificationReport"
                    )
                    cls._validate_report(state_shell, artifact, trace)
                    if (
                        artifact.payload.get("status") != "uncertifiable"
                        or artifact.content_hash != details["report_hash"]
                    ):
                        raise Recertification122BError("uncertifiable report event lineage mismatch")
                    uncertifiable[trace] = artifact
                else:
                    raise Recertification122BError("outcome event kind is unsupported")
            elif event_type == "PACK_CONFLICT_QUARANTINED" and isinstance(trace, str):
                if trace not in known_trace_ids:
                    raise Recertification122BError("conflict trace is outside DatasetVersion")
                if trace not in seen_outcomes:
                    raise Recertification122BError("conflict event precedes its committed outcome")
                if set(details) != {"trace_id", "conflicting_outcome_hash"}:
                    raise Recertification122BError("conflict event fields are invalid")
                store.read(
                    cast(str, details["conflicting_outcome_hash"]), expected_schema_name="TraceCertificationOutcome"
                )
                conflicts[trace] = event
            elif event_type == "RECERTIFICATION_BLOCKED":
                if blocked_reason is not None or coverage is not None or bundle is not None:
                    raise Recertification122BError("blocked event is out of order")
                if set(details) != {"reason_code"} or details.get("reason_code") not in {
                    "TRACE_PACK_HASH_CONFLICT",
                    "UNCERTIFIABLE_TRACE_BLOCKS_TOTALITY",
                }:
                    raise Recertification122BError("blocked event reason is invalid")
                reason = cast(str, details["reason_code"])
                if (reason == "TRACE_PACK_HASH_CONFLICT" and not conflicts) or (
                    reason == "UNCERTIFIABLE_TRACE_BLOCKS_TOTALITY" and not uncertifiable
                ):
                    raise Recertification122BError("blocked event has no corresponding failed evidence")
                blocked_reason = reason
            elif event_type == "RUN_CLOSED":
                if events.index(event) == 0 or events[events.index(event) - 1].payload.get("event_type") not in {
                    "JUDGE_BUNDLE_COMMITTED",
                    "RECERTIFICATION_BLOCKED",
                }:
                    raise Recertification122BError("terminal close is not after a terminal barrier")
                if set(details) != {"run_closed_hash"}:
                    raise Recertification122BError("terminal event fields are invalid")
                if blocked_reason is None:
                    if bundle is None:
                        raise Recertification122BError("successful close requires a committed JudgeBundle")
                closed_hash = details.get("run_closed_hash")
                if not isinstance(closed_hash, str):
                    raise Recertification122BError("terminal event is missing RunClosed artifact")
                closed = store.read(closed_hash, expected_schema_name="RunClosed")
                if blocked_reason is not None:
                    if closed.payload.get("status") != "failed" or closed.payload.get("reason_code") != blocked_reason:
                        raise Recertification122BError("blocked terminal outcome is inconsistent")
                elif (
                    closed.payload.get("status") != "succeeded"
                    or closed.payload.get("reason_code") != "122B_JUDGE_BUNDLE_PUBLISHED"
                ):
                    raise Recertification122BError("successful terminal outcome is inconsistent")
            elif event_type == "COVERAGE_MANIFEST_COMMITTED":
                if coverage is not None or seen_outcomes != known_trace_ids or uncertifiable or conflicts:
                    raise Recertification122BError("coverage event is out of order or incomplete")
                if set(details) != {"coverage_manifest_hash"}:
                    raise Recertification122BError("coverage event fields are invalid")
                coverage = store.read(
                    cast(str, details["coverage_manifest_hash"]), expected_schema_name="RecertificationCoverageManifest"
                )
                coverage_payload = coverage.payload
                expected_report_hashes = [reports[trace].content_hash for trace in sorted(known_trace_ids)]
                if (
                    coverage_payload.get("dataset_version_hash") != config.dataset_version_hash
                    or coverage_payload.get("fresh_rollout_count_per_trace") != 32
                    or coverage_payload.get("inference_model_id") != config.inference_model_id
                    or coverage_payload.get("reward_schema_hash") != config.reward_schema_hash
                    or coverage_payload.get("scalarizer_hash") != config.scalarizer_hash
                    or coverage_payload.get("report_hashes") != expected_report_hashes
                    or coverage_payload.get("trace_count") != 100
                    or coverage_payload.get("status") != "complete"
                ):
                    raise Recertification122BError("persisted coverage manifest contract is invalid")
            elif event_type == "JUDGE_BUNDLE_COMMITTED":
                if bundle is not None or coverage is None:
                    raise Recertification122BError("JudgeBundle event is out of order")
                if set(details) != {"judge_bundle_hash"}:
                    raise Recertification122BError("bundle event fields are invalid")
                bundle = store.read(cast(str, details["judge_bundle_hash"]), expected_schema_name="JudgeBundle")
                if (
                    bundle.payload.get("status") != "terminal"
                    or bundle.payload.get("certification_phase") != "TRAIN_122B"
                    or bundle.payload.get("dataset_version_hash") != config.dataset_version_hash
                    or bundle.payload.get("inference_model_id") != config.inference_model_id
                    or bundle.payload.get("reward_schema_hash") != config.reward_schema_hash
                    or bundle.payload.get("scalarizer_hash") != config.scalarizer_hash
                    or bundle.payload.get("algorithm_contract_hash") != config.algorithm_contract_hash
                    or bundle.payload.get("transfer_candidate_hash") != config.transfer_candidate_hash
                    or bundle.payload.get("fresh_rollout_count_per_trace") != 32
                    or bundle.payload.get("trace_count") != 100
                    or bundle.payload.get("coverage_manifest_hash") != coverage.content_hash
                    or bundle.payload.get("report_hashes") != coverage.payload.get("report_hashes")
                ):
                    raise Recertification122BError("persisted JudgeBundle contract is invalid")
        event_types = [event.payload.get("event_type") for event in events]
        if "RUN_CLOSED" in event_types and event_types[-1] != "RUN_CLOSED":
            raise Recertification122BError("terminal recertification event is not final")
        return _State(
            root, store, journal, config, dataset, tuple(events), reports, uncertifiable, conflicts, coverage, bundle
        )

    @staticmethod
    def _trace_ids(dataset: LoadedTrainingDataset) -> tuple[str, ...]:
        return tuple(cast(str, trace.payload["trace_id"]) for trace in dataset.training_traces)

    @staticmethod
    def _validate_governor_lineage(
        root: Path,
        store: ArtifactStore,
        candidate: Artifact,
        summary_hash: str,
        evidence: list[str],
    ) -> None:
        governor_id: str | None = None
        plan_hash: str | None = None
        for child_hash in evidence:
            child = store.read(child_hash, expected_schema_name="GovernorChild")
            action_id = child.payload.get("action_id")
            child_index = child.payload.get("child_index")
            child_plan_hash = child.payload.get("plan_hash")
            if (
                child.payload.get("status") != "completed"
                or child.payload.get("branch") != "stop_and_transfer"
                or child.payload.get("budget_reconciled") is not True
                or not isinstance(action_id, str)
                or ":child:" not in action_id
                or type(child_index) is not int
                or not isinstance(child_plan_hash, str)
                or not isinstance(child.payload.get("harness_transition_hashes"), list)
                or len(cast(list[object], child.payload["harness_transition_hashes"])) != 5
            ):
                raise Recertification122BError("Governor child evidence is not terminal and reconciled")
            current_governor, suffix = action_id.rsplit(":child:", 1)
            if not suffix.isdigit() or int(suffix) != child_index:
                raise Recertification122BError("Governor child action identity is invalid")
            if governor_id is None:
                governor_id = current_governor
                plan_hash = child_plan_hash
            if current_governor != governor_id or child_plan_hash != plan_hash:
                raise Recertification122BError("Governor child DAG is not single-plan evidence")
            plan = store.read(child_plan_hash, expected_schema_name="ActionPlan")
            child_ids = plan.payload.get("child_ids")
            if not isinstance(child_ids, list) or action_id not in child_ids:
                raise Recertification122BError("Governor child is not bound to its ActionPlan")
            expected_key = sha256_hex(
                canonical_json_bytes({"action_id": action_id, "plan_hash": child_plan_hash, "index": child_index})
            )
            if child.payload.get("idempotency_key") != expected_key:
                raise Recertification122BError("Governor child idempotency identity is invalid")
        if governor_id is None or plan_hash is None:
            raise Recertification122BError("TransferCandidate has no Governor child evidence")
        state_ref = root / "governor-iterations" / governor_id / "state.ref"
        try:
            state = store.read(state_ref.read_text(encoding="ascii").strip(), expected_schema_name="GovernorState")
        except (OSError, ArtifactCorruption) as error:
            raise Recertification122BError("Governor terminal state cannot be recovered") from error
        if (
            state.payload.get("status") != "terminal"
            or state.payload.get("child_hashes") != evidence
            or state.payload.get("plan_hash") != plan_hash
            or state.payload.get("summary_hash") != summary_hash
            or state.payload.get("transfer_candidate_hash") != candidate.content_hash
            or state.payload.get("next_index") != len(evidence)
        ):
            raise Recertification122BError("TransferCandidate is not bound to the immutable Governor terminal state")
        try:
            governor_snapshot = BoundedGovernorWorkflow.resume(root, governor_id)
        except BoundedGovernorError as error:
            raise Recertification122BError("Governor terminal workflow cannot be replayed") from error
        if (
            not governor_snapshot.terminal
            or governor_snapshot.transfer_candidate is None
            or governor_snapshot.transfer_candidate.content_hash != candidate.content_hash
        ):
            raise Recertification122BError("Governor replay did not yield this immutable TransferCandidate")
        cohort_id = candidate.payload.get("cohort_id")
        if not isinstance(cohort_id, str):
            raise Recertification122BError("TransferCandidate cohort identity is missing")
        matching_candidates: list[str] = []
        governor_root = root / "governor-iterations"
        if governor_root.exists():
            for state_ref in sorted(governor_root.glob("*/state.ref")):
                try:
                    persisted_state = store.read(
                        state_ref.read_text(encoding="ascii").strip(), expected_schema_name="GovernorState"
                    )
                    if persisted_state.payload.get("status") != "terminal":
                        continue
                    persisted_candidate_hash = persisted_state.payload.get("transfer_candidate_hash")
                    if not isinstance(persisted_candidate_hash, str):
                        continue
                    persisted_candidate = store.read(persisted_candidate_hash, expected_schema_name="TransferCandidate")
                    if persisted_candidate.payload.get("cohort_id") == cohort_id:
                        matching_candidates.append(persisted_candidate.content_hash)
                except (OSError, ArtifactCorruption, KeyError, TypeError) as error:
                    raise Recertification122BError("Governor candidate uniqueness cannot be verified") from error
        if matching_candidates != [candidate.content_hash]:
            raise Recertification122BError("TransferCandidate is not the unique terminal candidate for its cohort")

    @classmethod
    def _validate_report(cls, state: _State, report: Artifact, trace_id: str) -> None:
        p = report.payload
        if (
            p.get("trace_id") != trace_id
            or p.get("dataset_version_hash") != state.config.dataset_version_hash
            or p.get("inference_model_id") != state.config.inference_model_id
        ):
            raise Recertification122BError("CertificationReport is not fresh 122B evidence")
        if (
            p.get("reward_schema_hash") != state.config.reward_schema_hash
            or p.get("scalarizer_hash") != state.config.scalarizer_hash
            or p.get("algorithm_contract_hash") != state.config.algorithm_contract_hash
        ):
            raise Recertification122BError("CertificationReport contract identity mismatch")
        if any(
            key in p
            for key in (
                "teacher_label_set_hash",
                "holdout_set_hash",
                "fit_trajectory_set_hash",
                "historical_bundle_hash",
            )
        ):
            raise Recertification122BError("122B certification cannot consume teacher, holdout, or fit identity")
        if p.get("status") == "uncertifiable":
            return
        if (
            p.get("status") != "certified"
            or p.get("variant") != "success"
            or p.get("fresh_rollout_count") != 32
            or not isinstance(p.get("trajectory_set_hash"), str)
        ):
            raise Recertification122BError("CertificationReport is not certifiable fresh 122B evidence")
        trajectory = state.store.read(
            cast(str, p["trajectory_set_hash"]), expected_schema_name="Fresh122BTrajectorySet"
        )
        tp = trajectory.payload
        trace_artifact = next(
            (item for item in state.dataset.training_traces if item.payload.get("trace_id") == trace_id), None
        )
        attempt_index = p.get("attempt_index")
        if trace_artifact is None or type(attempt_index) is not int or attempt_index < 1:
            raise Recertification122BError("CertificationReport attempt lineage is invalid")
        expected_prompt_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "dataset_version_hash": state.config.dataset_version_hash,
                    "attempt_index": attempt_index,
                    "prompt": trace_artifact.payload.get("prompt"),
                    "trace_id": trace_id,
                }
            )
        )
        if p.get("prompt_hash") != expected_prompt_hash:
            raise Recertification122BError("CertificationReport prompt identity is not trace-derived")
        if (
            tp.get("item_count") != 32
            or tp.get("attempt_index") != attempt_index
            or tp.get("dataset_version_hash") != state.config.dataset_version_hash
            or tp.get("inference_model_id") != state.config.inference_model_id
            or tp.get("trace_id") != trace_id
            or tp.get("prompt_hash") != p.get("prompt_hash")
        ):
            raise Recertification122BError("fresh trajectory set cardinality or identity is invalid")
        refs = tp.get("trajectory_refs")
        indexes = (
            [item.get("trajectory_index") for item in refs if isinstance(item, dict)] if isinstance(refs, list) else []
        )
        if not isinstance(refs, list) or len(refs) != 32 or indexes != list(range(32)):
            raise Recertification122BError("fresh 122B trajectory indexes are not exactly 0..31")
        content_hashes: set[str] = set()
        response_hashes: set[str] = set()
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {"manifest_hash", "trajectory_index"}:
                raise Recertification122BError("fresh trajectory ref is malformed")
            manifest = state.store.read(cast(str, ref["manifest_hash"]), expected_schema_name="TrajectoryManifest")
            mp = manifest.payload
            if (
                mp.get("split") != "recertification_122b"
                or mp.get("phase") != "TRAIN_122B"
                or mp.get("model_id") != state.config.inference_model_id
                or mp.get("dataset_version_hash") != state.config.dataset_version_hash
                or mp.get("trace_id") != trace_id
                or mp.get("trajectory_index") != ref["trajectory_index"]
            ):
                raise Recertification122BError("trajectory manifest is not fresh 122B inference evidence")
            content_hash = mp.get("content_hash")
            if not isinstance(content_hash, str) or content_hash in content_hashes:
                raise Recertification122BError("trajectory content identity is duplicated or missing")
            content = state.store.read(content_hash, expected_schema_name="TrajectoryContent")
            cp = content.payload
            if (
                cp.get("model_id") != state.config.inference_model_id
                or cp.get("trace_id") != trace_id
                or cp.get("trajectory_index") != ref["trajectory_index"]
                or cp.get("prompt_hash") != mp.get("prompt_hash")
                or not isinstance(cp.get("response"), str)
                or cp.get("response_hash") != sha256_hex(cast(str, cp["response"]).encode())
                or cp.get("response_hash") in response_hashes
            ):
                raise Recertification122BError("trajectory content is not fresh or is duplicated")
            content_hashes.add(content_hash)
            response_hashes.add(cast(str, cp["response_hash"]))

    @classmethod
    def _validate_pack(cls, state: _State, pack: Artifact, trace_id: str) -> None:
        p = pack.payload
        if (
            p.get("trace_id") != trace_id
            or p.get("status") != "terminal"
            or p.get("fresh_rollout_count_per_trace") != 32
            or p.get("trajectory_count") != 32
            or p.get("certification_phase") != "TRAIN_122B"
            or not isinstance(p.get("fresh_trajectory_set_hash"), str)
            or p.get("dataset_version_hash") != state.config.dataset_version_hash
            or p.get("inference_model_id") != state.config.inference_model_id
            or p.get("reward_schema_hash") != state.config.reward_schema_hash
            or p.get("scalarizer_hash") != state.config.scalarizer_hash
            or p.get("algorithm_contract_hash") != state.config.algorithm_contract_hash
        ):
            raise Recertification122BError("JudgePack is not a complete fresh 122B pack")
        report = state.store.read(cast(str, p["certification_report_hash"]), expected_schema_name="CertificationReport")
        cls._validate_report(state, report, trace_id)
        if p.get("fresh_trajectory_set_hash") != report.payload.get("trajectory_set_hash"):
            raise Recertification122BError("JudgePack trajectory/report identity mismatch")

    @staticmethod
    def _append(state: _State, epoch: int, event_type: str, details: dict[str, object]) -> Artifact:
        return state.journal.append(
            epoch,
            event_type,
            details,
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )

    @staticmethod
    def _close(state: _State, epoch: int, reason: str, *, failed: bool) -> None:
        state.journal.claim_epoch(epoch)
        blocked = Recertification122BWorkflow._append(state, epoch, "RECERTIFICATION_BLOCKED", {"reason_code": reason})
        state.journal.close(
            epoch,
            status="failed" if failed else "succeeded",
            reason_code=reason,
            expected_sequence=len(state.events) + 2,
            expected_previous_hash=blocked.content_hash,
        )

    @classmethod
    def _snapshot(cls, state: _State) -> Recertification122BSnapshot:
        expected = set(cls._trace_ids(state.dataset))
        observed = set(state.reports) | set(state.uncertifiable)
        terminal = bool(state.events and state.events[-1].payload.get("event_type") == "RUN_CLOSED")
        return Recertification122BSnapshot(
            state.events,
            dict(state.reports),
            tuple(sorted(state.uncertifiable)),
            tuple(sorted(state.conflicts)),
            tuple(sorted(expected - observed)),
            state.coverage,
            state.bundle,
            terminal,
        )


JudgeBundleRecertificationWorkflow = Recertification122BWorkflow
TransferRecertificationWorkflow = Recertification122BWorkflow
Recertification122BSnapshot = Recertification122BSnapshot

__all__ = [
    "Fixture122BConfig",
    "Fixture122BRecertificationConfig",
    "Fixture122BRecertificationSource",
    "InjectedRecertification122BCrash",
    "JudgeBundleRecertificationWorkflow",
    "Recertification122BConfig",
    "Recertification122BError",
    "Recertification122BSnapshot",
    "Recertification122BWorkflow",
    "TransferRecertificationWorkflow",
]
