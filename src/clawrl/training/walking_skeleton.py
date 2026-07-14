"""Production-shaped one-Trace workflow using profile-specific boundary ports."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from clawrl.adapters.cluster.fixture import PersistentFixtureCluster
from clawrl.adapters.scorers.fixture import PersistentFixtureScorer
from clawrl.artifacts import (
    MAX_SAFE_INTEGER,
    Artifact,
    ArtifactCorruption,
    ArtifactError,
    ArtifactStore,
    JsonValue,
    canonical_json_bytes,
    sha256_hex,
)
from clawrl.judge.student_packet import (
    StudentPacketValidationError,
    sanitized_rejection_payload,
    validate_judge_pack_structure,
    validate_student_packet_structure,
)
from clawrl.profiles import (
    FixtureProfileConfig,
    ProductionProfileConfig,
    ProfileConfig,
    invalid_profile_readiness_payload,
    production_readiness_payload,
    runtime_profile_is_valid,
)
from clawrl.training.run_journal import (
    JournalHeadConflict,
    RunIdentityConflict,
    RunJournal,
)

_SAFE_LINEAGE = re.compile(r"^/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+$")
_LIFECYCLE_EVENT_TYPES = {
    "RUN_STARTED",
    "SCORING_REQUESTED",
    "SCORING_RETRY_SCHEDULED",
    "SCORING_FAILED",
    "REWARD_COMMITTED",
    "CLUSTER_SUBMIT_REQUESTED",
    "CLUSTER_RETRY_SCHEDULED",
    "CLUSTER_FAILED",
    "CLUSTER_ACCEPTED",
    "RUN_RECORD_COMMITTED",
    "DECISION_OUTCOME_COMMITTED",
    "RUN_CLOSED",
}


class WorkflowIntegrityError(RuntimeError):
    """Persisted state and supplied immutable input disagree."""


class InjectedProcessCrash(RuntimeError):
    """Test-only host crash after an adapter committed a durable side effect."""


@dataclass(frozen=True, slots=True)
class TraceWorkflowInput:
    run_id: str
    dataset_version_id: str
    experiment_spec_id: str
    trace_id: str
    trajectory_id: str
    prompt: str
    response: str
    judge_pack: Mapping[str, object]
    student_input_packet: Mapping[str, object]
    raw_student_output: bytes
    role_session_lineage: str
    student_packet_rejection: Mapping[str, object] | None = None
    rejected_raw_output_hash: str | None = None

    def __post_init__(self) -> None:
        string_fields = (
            self.run_id,
            self.dataset_version_id,
            self.experiment_spec_id,
            self.trace_id,
            self.trajectory_id,
            self.prompt,
            self.response,
        )
        if any(not value for value in string_fields):
            raise ValueError("Trace workflow identity and text fields must be nonempty")
        if _SAFE_LINEAGE.fullmatch(self.role_session_lineage) is None:
            raise ValueError("role session lineage must be a nonempty canonical task path")
        if not isinstance(self.raw_student_output, bytes):
            raise ValueError("raw student output must be bytes")
        if self.student_packet_rejection is None and not self.raw_student_output:
            raise ValueError("accepted Student exchange requires nonempty raw output bytes")
        if self.student_packet_rejection is not None and self.raw_student_output:
            raise ValueError("rejected Student packet cannot retain raw role output bytes")


@dataclass(frozen=True, slots=True)
class WorkflowSnapshot:
    events: list[Artifact]
    closed: Artifact | None = None
    reward: Artifact | None = None
    run_record: Artifact | None = None
    decision_outcome: Artifact | None = None
    readiness_report: Artifact | None = None


@dataclass(frozen=True, slots=True)
class _PreparedTraceInput:
    trace_payload: dict[str, object]
    trajectory_payload: dict[str, object]
    judge_pack_payload: dict[str, object] | None
    experiment_payload: dict[str, object] | None
    fixture_config_payload: dict[str, JsonValue]
    rejection_payload: dict[str, JsonValue] | None
    student_packet_payload: dict[str, object] | None
    raw_manifest_payload: dict[str, object] | None
    fixture_exchange_payload: dict[str, object] | None
    workflow_input_payload: dict[str, object]
    input_hash: str


class FixtureAdapterFactory(Protocol):
    def build_fixture(
        self,
        root: Path,
        store: ArtifactStore,
        config: FixtureProfileConfig,
    ) -> tuple[PersistentFixtureScorer, PersistentFixtureCluster]: ...


class DefaultFixtureAdapterFactory:
    def build_fixture(
        self,
        root: Path,
        store: ArtifactStore,
        config: FixtureProfileConfig,
    ) -> tuple[PersistentFixtureScorer, PersistentFixtureCluster]:
        return (
            PersistentFixtureScorer(
                root,
                store,
                config.scorer_failure_code,
                config.effective_scorer_fault_schedule(),
            ),
            PersistentFixtureCluster(
                root,
                store,
                config.cluster_failure_code,
                config.effective_cluster_fault_schedule(),
            ),
        )


class ProfiledTraceWorkflow:
    """Advance a persisted workflow by one committed controller event."""

    def __init__(
        self,
        root: str | Path,
        config: ProfileConfig,
        *,
        adapter_factory: FixtureAdapterFactory | None = None,
    ) -> None:
        self.root = Path(root)
        self.store = ArtifactStore(self.root)
        self.config = config
        self.adapter_factory = adapter_factory or DefaultFixtureAdapterFactory()

    @classmethod
    def resume(
        cls,
        root: str | Path,
        run_id: str,
        *,
        epoch: int,
        crash_after_side_effect: str | None = None,
    ) -> WorkflowSnapshot:
        """Advance using only immutable state recovered from disk."""

        loader = cls(root, FixtureProfileConfig())
        item, config = loader._load_persisted_input(run_id)
        return cls(root, config).advance(
            item,
            epoch=epoch,
            crash_after_side_effect=crash_after_side_effect,
        )

    def advance(
        self,
        item: TraceWorkflowInput,
        *,
        epoch: int,
        crash_after_side_effect: str | None = None,
    ) -> WorkflowSnapshot:
        if crash_after_side_effect not in {
            None,
            "reservation",
            "start_event",
            "start_fence",
            "scorer",
            "cluster",
        }:
            raise ValueError("unknown crash injection boundary")
        if not runtime_profile_is_valid(self.config):
            report = self.store.put(
                "ReadinessReport",
                "1.0.0",
                invalid_profile_readiness_payload(),
            )
            return WorkflowSnapshot(events=[], readiness_report=report)
        if type(self.config) is ProductionProfileConfig:
            production_config = cast(ProductionProfileConfig, self.config)
            report = self.store.put(
                "ReadinessReport",
                "1.0.0",
                production_readiness_payload(production_config),
            )
            return WorkflowSnapshot(events=[], readiness_report=report)
        if type(self.config) is not FixtureProfileConfig:
            report = self.store.put(
                "ReadinessReport",
                "1.0.0",
                invalid_profile_readiness_payload(),
            )
            return WorkflowSnapshot(events=[], readiness_report=report)
        fixture_config = cast(FixtureProfileConfig, self.config)

        prepared = self._prepare_input(item, fixture_config)
        journal = RunJournal(self.root, self.store, item.run_id)
        try:
            journal.reserve_identity(prepared.input_hash)
        except RunIdentityConflict as error:
            raise WorkflowIntegrityError("supplied input does not match reserved run identity") from error
        events = journal.events()
        if events:
            started = self._find_event(events, "RUN_STARTED")
            if started is None:
                raise WorkflowIntegrityError("run has no committed RUN_STARTED identity")
            first_details = self._details(started)
            if first_details.get("input_hash") != prepared.input_hash:
                raise WorkflowIntegrityError("supplied input does not match committed RUN_STARTED")
        lifecycle_head = self._last_lifecycle_event(events)
        if lifecycle_head is not None and lifecycle_head.payload["event_type"] == "RUN_CLOSED":
            return self.load_result(item.run_id)
        materialized = self._materialize_input(prepared, item.raw_student_output)
        workflow_input = self._required_artifact(materialized, "input")
        if not events:
            if crash_after_side_effect == "reservation":
                raise InjectedProcessCrash("injected crash after input materialization and identity reservation")

            def interrupt_after_start_event() -> None:
                raise InjectedProcessCrash("injected crash after authoritative RUN_STARTED commit")

            try:
                journal.start_run(
                    epoch,
                    prepared.input_hash,
                    {
                        "experiment_spec_hash": prepared.workflow_input_payload.get("experiment_spec_hash"),
                        "input_hash": workflow_input.content_hash,
                        "judge_pack_hash": prepared.workflow_input_payload.get("judge_pack_hash"),
                        "trace_hash": self._required_artifact(materialized, "trace").content_hash,
                        "trajectory_hash": self._required_artifact(materialized, "trajectory").content_hash,
                    },
                    after_event_commit=(
                        interrupt_after_start_event if crash_after_side_effect == "start_event" else None
                    ),
                )
            except RunIdentityConflict as error:
                raise WorkflowIntegrityError(
                    "supplied input does not match concurrently committed RUN_STARTED"
                ) from error
            if crash_after_side_effect == "start_fence":
                raise InjectedProcessCrash("injected crash after derivative startup fence commit")
            return self.load_result(item.run_id)

        journal.claim_epoch(epoch)

        last = self._last_lifecycle_event(events)
        if last is None:
            raise WorkflowIntegrityError("run has no supported lifecycle state")
        event_type = cast(str, last.payload["event_type"])
        if event_type == "RUN_STARTED":
            packet_status = workflow_input.payload.get("packet_status")
            packet_rejection_hash = workflow_input.payload.get("packet_rejection_hash")
            if packet_status == "accepted" and packet_rejection_hash is None:
                packet_id: str | None = self._required_string(item.student_input_packet, "packet_id")
                judge_pack_hash: str | None = self._required_string(workflow_input.payload, "judge_pack_hash")
                judge_pack_id: str | None = self._required_string(item.judge_pack, "judge_pack_id")
            elif packet_status == "rejected" and isinstance(packet_rejection_hash, str):
                packet_id = None
                judge_pack_hash = None
                judge_pack_id = None
            else:
                raise WorkflowIntegrityError("TraceRunInput packet binding is invalid")
            request = self.store.put(
                "ScoringRequest",
                "1.0.0",
                {
                    "input_hash": workflow_input.content_hash,
                    "judge_pack_hash": judge_pack_hash,
                    "judge_pack_id": judge_pack_id,
                    "packet_id": packet_id,
                    "packet_rejection_hash": packet_rejection_hash,
                    "prompt": item.prompt,
                    "response": item.response,
                    "run_id": item.run_id,
                    "trace_id": item.trace_id,
                    "trajectory_id": item.trajectory_id,
                },
            )
            self._append_transition(
                journal,
                epoch,
                "SCORING_REQUESTED",
                {
                    "attempt_sequence": 1,
                    "idempotency_key": self._action_key("student-score", request.content_hash),
                    "request_hash": request.content_hash,
                },
                events,
            )
        elif event_type in {"SCORING_REQUESTED", "SCORING_RETRY_SCHEDULED"}:
            details = self._details(last)
            attempt_sequence = self._detail_positive_integer(
                details,
                "attempt_sequence",
            )
            request_hash = self._detail_string(details, "request_hash")
            key = self._detail_string(details, "idempotency_key")
            scoring_request: Artifact | None = None
            scorer_candidate_hash: str | None = None
            try:
                scoring_request = self.store.read(
                    request_hash,
                    expected_schema_name="ScoringRequest",
                )
                packet_rejection_hash = scoring_request.payload.get("packet_rejection_hash")
                if isinstance(packet_rejection_hash, str):
                    rejection = self.store.read(
                        packet_rejection_hash,
                        expected_schema_name="StudentPacketRejection",
                    )
                    rejection_payload = self._validated_rejection_payload(rejection.payload)
                    observation = self.store.put(
                        "Observation",
                        "1.0.0",
                        {
                            "detail": self._required_string(
                                rejection_payload,
                                "failure_detail",
                            ),
                            "failure_code": self._required_string(
                                rejection_payload,
                                "failure_code",
                            ),
                            "idempotency_key": key,
                            "producer": "student_packet_gate",
                            "request_hash": scoring_request.content_hash,
                            "role_invocation_hash": None,
                            "status": "failed",
                        },
                    )
                elif packet_rejection_hash is None:
                    scorer, _ = self.adapter_factory.build_fixture(
                        self.root,
                        self.store,
                        self.config,
                    )
                    queried_observation = scorer.query(
                        key,
                        scoring_request.content_hash,
                    )
                    if queried_observation is None:
                        candidate_observation = scorer.execute(
                            key,
                            scoring_request,
                            item.student_input_packet,
                            item.raw_student_output,
                            item.role_session_lineage,
                            attempt_sequence,
                        )
                    else:
                        candidate_observation = queried_observation
                    scorer_candidate_hash = candidate_observation.content_hash
                    observation = self.store.read(
                        scorer_candidate_hash,
                        expected_schema_name="Observation",
                    )
                    self._validate_scorer_outcome(
                        observation,
                        scoring_request,
                        key,
                        scorer,
                        attempt_sequence,
                    )
                else:
                    raise ArtifactCorruption("ScoringRequest packet rejection binding is invalid")
            except ArtifactError:
                observation = self._integrity_failure_observation(
                    producer="scorer_outcome_gate",
                    failure_code="SCORER_OUTCOME_CORRUPTION",
                    detail="scorer outcome failed request and lineage validation",
                    idempotency_key=key,
                    request_hash=request_hash,
                    source_event_hash=last.content_hash,
                    candidate_observation_hash=scorer_candidate_hash,
                    include_role_hash=True,
                )
            except Exception:
                observation = self._integrity_failure_observation(
                    producer="scorer_adapter_gate",
                    failure_code="SCORER_ADAPTER_FAILURE",
                    detail="scorer adapter interaction failed safely",
                    idempotency_key=key,
                    request_hash=request_hash,
                    source_event_hash=last.content_hash,
                    candidate_observation_hash=scorer_candidate_hash,
                    include_role_hash=True,
                )
            if crash_after_side_effect == "scorer":
                raise InjectedProcessCrash("injected crash after scorer commit")
            if observation.payload.get("status") == "retryable":
                self._append_transition(
                    journal,
                    epoch,
                    "SCORING_RETRY_SCHEDULED",
                    {
                        "attempt_sequence": attempt_sequence + 1,
                        "failure_code": self._required_string(
                            observation.payload,
                            "failure_code",
                        ),
                        "idempotency_key": key,
                        "observation_hash": observation.content_hash,
                        "request_hash": request_hash,
                    },
                    events,
                )
            elif observation.payload.get("status") == "failed":
                role_hash = observation.payload.get("role_invocation_hash")
                self._append_transition(
                    journal,
                    epoch,
                    "SCORING_FAILED",
                    {
                        "failure_code": self._required_string(observation.payload, "failure_code"),
                        "observation_hash": observation.content_hash,
                        "role_invocation_hash": role_hash,
                    },
                    events,
                )
            else:
                self._append_transition(
                    journal,
                    epoch,
                    "REWARD_COMMITTED",
                    {
                        "evidence_hash": self._required_string(observation.payload, "evidence_hash"),
                        "observation_hash": observation.content_hash,
                        "reward_hash": self._required_string(observation.payload, "reward_hash"),
                        "role_invocation_hash": self._required_string(observation.payload, "role_invocation_hash"),
                    },
                    events,
                )
        elif event_type == "REWARD_COMMITTED":
            details = self._details(last)
            proposal = self.store.put(
                "DecisionProposal",
                "1.0.0",
                {
                    "action_type": "submit_fixture_training_job",
                    "evidence_hash": self._detail_string(details, "evidence_hash"),
                    "reward_hash": self._detail_string(details, "reward_hash"),
                    "run_id": item.run_id,
                    "trace_id": item.trace_id,
                },
            )
            request = self.store.put(
                "ClusterSubmitRequest",
                "1.0.0",
                {
                    "decision_proposal_hash": proposal.content_hash,
                    "experiment_spec_hash": self._required_artifact(materialized, "experiment").content_hash,
                    "fixture_config_hash": self._required_string(
                        workflow_input.payload,
                        "fixture_config_hash",
                    ),
                    "reward_hash": self._detail_string(details, "reward_hash"),
                    "run_id": item.run_id,
                    "trace_id": item.trace_id,
                },
            )
            self._append_transition(
                journal,
                epoch,
                "CLUSTER_SUBMIT_REQUESTED",
                {
                    "attempt_sequence": 1,
                    "decision_proposal_hash": proposal.content_hash,
                    "idempotency_key": self._action_key("cluster-submit", proposal.content_hash),
                    "request_hash": request.content_hash,
                },
                events,
            )
        elif event_type in {
            "CLUSTER_SUBMIT_REQUESTED",
            "CLUSTER_RETRY_SCHEDULED",
        }:
            details = self._details(last)
            attempt_sequence = self._detail_positive_integer(
                details,
                "attempt_sequence",
            )
            request_hash = self._detail_string(details, "request_hash")
            key = self._detail_string(details, "idempotency_key")
            cluster_candidate_hash: str | None = None
            try:
                cluster_request = self.store.read(
                    request_hash,
                    expected_schema_name="ClusterSubmitRequest",
                )
                _, cluster = self.adapter_factory.build_fixture(
                    self.root,
                    self.store,
                    self.config,
                )
                queried_observation = cluster.query(
                    key,
                    cluster_request.content_hash,
                )
                if queried_observation is None:
                    candidate_observation = cluster.execute(
                        key,
                        cluster_request,
                        attempt_sequence,
                    )
                else:
                    candidate_observation = queried_observation
                cluster_candidate_hash = candidate_observation.content_hash
                observation = self.store.read(
                    cluster_candidate_hash,
                    expected_schema_name="Observation",
                )
                self._validate_cluster_outcome(
                    observation,
                    cluster_request,
                    key,
                    cluster,
                    attempt_sequence,
                )
            except ArtifactError:
                observation = self._integrity_failure_observation(
                    producer="cluster_outcome_gate",
                    failure_code="CLUSTER_OUTCOME_CORRUPTION",
                    detail="cluster outcome failed request and lineage validation",
                    idempotency_key=key,
                    request_hash=request_hash,
                    source_event_hash=last.content_hash,
                    candidate_observation_hash=cluster_candidate_hash,
                    include_role_hash=False,
                )
            except Exception:
                observation = self._integrity_failure_observation(
                    producer="cluster_adapter_gate",
                    failure_code="CLUSTER_ADAPTER_FAILURE",
                    detail="cluster adapter interaction failed safely",
                    idempotency_key=key,
                    request_hash=request_hash,
                    source_event_hash=last.content_hash,
                    candidate_observation_hash=cluster_candidate_hash,
                    include_role_hash=False,
                )
            if crash_after_side_effect == "cluster":
                raise InjectedProcessCrash("injected crash after cluster commit")
            if observation.payload.get("status") == "retryable":
                self._append_transition(
                    journal,
                    epoch,
                    "CLUSTER_RETRY_SCHEDULED",
                    {
                        "attempt_sequence": attempt_sequence + 1,
                        "failure_code": self._required_string(
                            observation.payload,
                            "failure_code",
                        ),
                        "idempotency_key": key,
                        "observation_hash": observation.content_hash,
                        "request_hash": request_hash,
                    },
                    events,
                )
            elif observation.payload.get("status") == "failed":
                self._append_transition(
                    journal,
                    epoch,
                    "CLUSTER_FAILED",
                    {
                        "failure_code": self._required_string(observation.payload, "failure_code"),
                        "observation_hash": observation.content_hash,
                    },
                    events,
                )
            else:
                self._append_transition(
                    journal,
                    epoch,
                    "CLUSTER_ACCEPTED",
                    {
                        "job_hash": self._required_string(observation.payload, "job_hash"),
                        "observation_hash": observation.content_hash,
                    },
                    events,
                )
        elif event_type in {"SCORING_FAILED", "CLUSTER_FAILED", "CLUSTER_ACCEPTED"}:
            run_record = self._build_run_record(item, events)
            self._append_transition(
                journal,
                epoch,
                "RUN_RECORD_COMMITTED",
                {"run_record_hash": run_record.content_hash},
                events,
            )
        elif event_type == "RUN_RECORD_COMMITTED":
            run_record = self.store.read(
                self._detail_string(self._details(last), "run_record_hash"),
                expected_schema_name="RunRecord",
            )
            decision = self._build_decision_outcome(item.run_id, run_record, events)
            self._append_transition(
                journal,
                epoch,
                "DECISION_OUTCOME_COMMITTED",
                {"decision_outcome_hash": decision.content_hash},
                events,
            )
        elif event_type == "DECISION_OUTCOME_COMMITTED":
            decision = self.store.read(
                self._detail_string(self._details(last), "decision_outcome_hash"),
                expected_schema_name="DecisionOutcome",
            )
            failed = decision.payload.get("status") == "failed"
            try:
                journal.close(
                    epoch,
                    status="failed" if failed else "succeeded",
                    reason_code=self._required_string(decision.payload, "reason_code"),
                    expected_sequence=len(events) + 1,
                    expected_previous_hash=events[-1].content_hash,
                )
            except JournalHeadConflict:
                pass
        else:
            raise WorkflowIntegrityError(f"unsupported persisted workflow transition {event_type!r}")
        return self.load_result(item.run_id)

    def load_result(self, run_id: str) -> WorkflowSnapshot:
        journal = RunJournal(self.root, self.store, run_id)
        events = journal.events()
        reward: Artifact | None = None
        run_record: Artifact | None = None
        decision: Artifact | None = None
        closed: Artifact | None = None
        for event in events:
            details = self._details(event)
            event_type = event.payload["event_type"]
            if event_type == "REWARD_COMMITTED":
                reward = self.store.read(
                    self._detail_string(details, "reward_hash"),
                    expected_schema_name="EvidenceLinkedReward",
                )
            elif event_type == "RUN_RECORD_COMMITTED":
                run_record = self.store.read(
                    self._detail_string(details, "run_record_hash"),
                    expected_schema_name="RunRecord",
                )
            elif event_type == "DECISION_OUTCOME_COMMITTED":
                decision = self.store.read(
                    self._detail_string(details, "decision_outcome_hash"),
                    expected_schema_name="DecisionOutcome",
                )
            elif event_type == "RUN_CLOSED":
                closed = journal.closed()
        return WorkflowSnapshot(
            events=events,
            closed=closed,
            reward=reward,
            run_record=run_record,
            decision_outcome=decision,
        )

    def _prepare_input(
        self,
        item: TraceWorkflowInput,
        config: FixtureProfileConfig,
    ) -> _PreparedTraceInput:
        trace_payload: dict[str, object] = {
            "dataset_version_id": item.dataset_version_id,
            "prompt": item.prompt,
            "trace_id": item.trace_id,
        }
        trace_hash = self._artifact_content_hash("TrainingTrace", trace_payload)
        trajectory_payload: dict[str, object] = {
            "response": item.response,
            "trace_hash": trace_hash,
            "trajectory_id": item.trajectory_id,
        }
        trajectory_hash = self._artifact_content_hash("Trajectory", trajectory_payload)
        fixture_config_payload = dict(config.artifact_payload())
        fixture_config_hash = self._artifact_content_hash("FixtureProfileConfig", fixture_config_payload)

        rejection_payload: dict[str, JsonValue] | None = None
        if item.student_packet_rejection is not None:
            rejection_payload = self._validated_rejection_payload(item.student_packet_rejection)
        else:
            try:
                validate_student_packet_structure(item.student_input_packet)
                validate_judge_pack_structure(item.judge_pack)
                packet_trajectory = item.student_input_packet.get("trajectory")
                if (
                    item.student_input_packet.get("run_id") != item.run_id
                    or item.student_input_packet.get("trace_id") != item.trace_id
                    or not isinstance(packet_trajectory, dict)
                    or packet_trajectory.get("trajectory_id") != item.trajectory_id
                    or packet_trajectory.get("prompt") != item.prompt
                    or packet_trajectory.get("response") != item.response
                    or item.student_input_packet.get("judge_pack") != dict(item.judge_pack)
                ):
                    raise StudentPacketValidationError("student packet does not match workflow identity or JudgePack")
            except StudentPacketValidationError as error:
                rejection_payload = sanitized_rejection_payload(
                    {
                        "judge_pack": dict(item.judge_pack),
                        "student_packet": dict(item.student_input_packet),
                    },
                    error,
                )

        judge_pack_payload: dict[str, object] | None
        experiment_payload: dict[str, object] | None
        student_packet_payload: dict[str, object] | None
        raw_manifest_payload: dict[str, object] | None
        fixture_exchange_payload: dict[str, object] | None
        if rejection_payload is None:
            judge_pack_payload = dict(item.judge_pack)
            judge_pack_hash: str | None = self._artifact_content_hash("JudgePack", judge_pack_payload)
            experiment_payload = {
                "dataset_version_id": item.dataset_version_id,
                "experiment_spec_id": item.experiment_spec_id,
                "judge_pack_hash": judge_pack_hash,
                "trace_hash": trace_hash,
            }
            experiment_spec_hash: str | None = self._artifact_content_hash("ExperimentSpec", experiment_payload)
            student_packet_payload = dict(item.student_input_packet)
            student_input_packet_hash: str | None = self._artifact_content_hash(
                "StudentJudgeInputPacket", student_packet_payload
            )
            raw_blob_hash = sha256_hex(item.raw_student_output)
            raw_manifest_payload = {
                "blob_hash": raw_blob_hash,
                "byte_size": len(item.raw_student_output),
                "media_type": "application/json; charset=utf-8",
            }
            raw_manifest_hash = self._artifact_content_hash("RawRoleOutput", raw_manifest_payload)
            fixture_exchange_payload = {
                "input_packet_hash": student_input_packet_hash,
                "raw_output_hash": raw_blob_hash,
                "raw_output_manifest_hash": raw_manifest_hash,
                "session_lineage": item.role_session_lineage,
            }
            fixture_exchange_hash: str | None = self._artifact_content_hash(
                "StudentJudgeFixtureExchange", fixture_exchange_payload
            )
            packet_rejection_hash: str | None = None
            packet_status = "accepted"
            student_input_hash: str | None = sha256_hex(canonical_json_bytes(student_packet_payload))
            rejected_raw_output_hash: str | None = None
        else:
            judge_pack_payload = None
            judge_pack_hash = None
            experiment_payload = None
            experiment_spec_hash = None
            student_packet_payload = None
            student_input_packet_hash = None
            raw_manifest_payload = None
            fixture_exchange_payload = None
            fixture_exchange_hash = None
            packet_rejection_hash = self._artifact_content_hash("StudentPacketRejection", rejection_payload)
            packet_status = "rejected"
            rejected_hash = rejection_payload.get("input_hash")
            student_input_hash = rejected_hash if isinstance(rejected_hash, str) else None
            rejected_raw_output_hash = (
                item.rejected_raw_output_hash
                if item.rejected_raw_output_hash is not None
                else sha256_hex(item.raw_student_output)
            )
        workflow_input_payload: dict[str, object] = {
            "experiment_spec_hash": experiment_spec_hash,
            "experiment_spec_id": item.experiment_spec_id,
            "fixture_config_hash": fixture_config_hash,
            "fixture_exchange_hash": fixture_exchange_hash,
            "judge_pack_hash": judge_pack_hash,
            "packet_rejection_hash": packet_rejection_hash,
            "packet_status": packet_status,
            "rejected_raw_output_hash": rejected_raw_output_hash,
            "role_session_lineage": item.role_session_lineage,
            "run_id": item.run_id,
            "student_input_hash": student_input_hash,
            "student_input_packet_hash": student_input_packet_hash,
            "trace_hash": trace_hash,
            "trajectory_hash": trajectory_hash,
        }
        input_hash = self._artifact_content_hash("TraceRunInput", workflow_input_payload)
        return _PreparedTraceInput(
            trace_payload=trace_payload,
            trajectory_payload=trajectory_payload,
            judge_pack_payload=judge_pack_payload,
            experiment_payload=experiment_payload,
            fixture_config_payload=fixture_config_payload,
            rejection_payload=rejection_payload,
            student_packet_payload=student_packet_payload,
            raw_manifest_payload=raw_manifest_payload,
            fixture_exchange_payload=fixture_exchange_payload,
            workflow_input_payload=workflow_input_payload,
            input_hash=input_hash,
        )

    def _materialize_input(
        self,
        prepared: _PreparedTraceInput,
        raw_student_output: bytes,
    ) -> dict[str, Artifact | None]:
        trace = self.store.put("TrainingTrace", "1.0.0", prepared.trace_payload)
        trajectory = self.store.put("Trajectory", "1.0.0", prepared.trajectory_payload)
        fixture_config = self.store.put(
            "FixtureProfileConfig",
            "1.0.0",
            prepared.fixture_config_payload,
        )
        if prepared.rejection_payload is None:
            if (
                prepared.judge_pack_payload is None
                or prepared.experiment_payload is None
                or prepared.student_packet_payload is None
                or prepared.raw_manifest_payload is None
                or prepared.fixture_exchange_payload is None
            ):
                raise WorkflowIntegrityError("accepted input blueprint is incomplete")
            judge_pack = self.store.put("JudgePack", "1.0.0", prepared.judge_pack_payload)
            experiment = self.store.put("ExperimentSpec", "1.0.0", prepared.experiment_payload)
            self.store.put(
                "StudentJudgeInputPacket",
                "1.0.0",
                prepared.student_packet_payload,
            )
            blob_hash, byte_size = self.store.put_blob(raw_student_output)
            if (
                prepared.raw_manifest_payload.get("blob_hash") != blob_hash
                or prepared.raw_manifest_payload.get("byte_size") != byte_size
            ):
                raise WorkflowIntegrityError("raw output changed after input preparation")
            self.store.put("RawRoleOutput", "1.0.0", prepared.raw_manifest_payload)
            self.store.put(
                "StudentJudgeFixtureExchange",
                "1.0.0",
                prepared.fixture_exchange_payload,
            )
        else:
            judge_pack = None
            experiment = None
            self.store.put(
                "StudentPacketRejection",
                "1.0.0",
                prepared.rejection_payload,
            )
        workflow_input = self.store.put("TraceRunInput", "1.0.0", prepared.workflow_input_payload)
        if workflow_input.content_hash != prepared.input_hash:
            raise WorkflowIntegrityError("materialized input identity changed")
        return {
            "experiment": experiment,
            "fixture_config": fixture_config,
            "input": workflow_input,
            "judge_pack": judge_pack,
            "trace": trace,
            "trajectory": trajectory,
        }

    def _load_persisted_input(self, run_id: str) -> tuple[TraceWorkflowInput, FixtureProfileConfig]:
        journal = RunJournal(self.root, self.store, run_id)
        events = journal.events()
        started = self._find_event(events, "RUN_STARTED")
        if started is None:
            if events:
                raise WorkflowIntegrityError("run has events without a committed RUN_STARTED input binding")
            input_hash = journal.reserved_input_hash()
        else:
            input_hash = self._detail_string(self._details(started), "input_hash")
        workflow_input = self.store.read(input_hash, expected_schema_name="TraceRunInput")
        if workflow_input.payload.get("run_id") != run_id:
            raise WorkflowIntegrityError("TraceRunInput belongs to another run")
        trace = self.store.read(
            self._required_string(workflow_input.payload, "trace_hash"),
            expected_schema_name="TrainingTrace",
        )
        trajectory = self.store.read(
            self._required_string(workflow_input.payload, "trajectory_hash"),
            expected_schema_name="Trajectory",
        )
        config_artifact = self.store.read(
            self._required_string(workflow_input.payload, "fixture_config_hash"),
            expected_schema_name="FixtureProfileConfig",
        )
        config = FixtureProfileConfig.from_mapping(config_artifact.payload)
        packet_status = workflow_input.payload.get("packet_status")
        if packet_status == "accepted":
            judge_pack = self.store.read(
                self._required_string(workflow_input.payload, "judge_pack_hash"),
                expected_schema_name="JudgePack",
            )
            experiment = self.store.read(
                self._required_string(workflow_input.payload, "experiment_spec_hash"),
                expected_schema_name="ExperimentSpec",
            )
            exchange = self.store.read(
                self._required_string(workflow_input.payload, "fixture_exchange_hash"),
                expected_schema_name="StudentJudgeFixtureExchange",
            )
            packet = self.store.read(
                self._required_string(exchange.payload, "input_packet_hash"),
                expected_schema_name="StudentJudgeInputPacket",
            )
            raw_manifest = self.store.read(
                self._required_string(exchange.payload, "raw_output_manifest_hash"),
                expected_schema_name="RawRoleOutput",
            )
            raw_size = raw_manifest.payload.get("byte_size")
            if not isinstance(raw_size, int) or isinstance(raw_size, bool) or raw_size < 0:
                raise WorkflowIntegrityError("RawRoleOutput byte_size is invalid")
            raw_output = self.store.read_blob(
                self._required_string(raw_manifest.payload, "blob_hash"),
                expected_size=raw_size,
            )
            if sha256_hex(raw_output) != self._required_string(exchange.payload, "raw_output_hash"):
                raise WorkflowIntegrityError("fixture role output does not match its exchange hash")
            packet_payload: Mapping[str, object] = packet.payload
            rejection_payload: Mapping[str, object] | None = None
            rejected_raw_output_hash = None
            session_lineage = self._required_string(exchange.payload, "session_lineage")
            judge_pack_payload: Mapping[str, object] = judge_pack.payload
            experiment_spec_id = self._required_string(experiment.payload, "experiment_spec_id")
        elif packet_status == "rejected":
            rejection = self.store.read(
                self._required_string(workflow_input.payload, "packet_rejection_hash"),
                expected_schema_name="StudentPacketRejection",
            )
            rejection_payload = self._validated_rejection_payload(rejection.payload)
            packet_payload = {}
            raw_output = b""
            rejected_raw_output_hash = self._required_string(workflow_input.payload, "rejected_raw_output_hash")
            session_lineage = self._required_string(workflow_input.payload, "role_session_lineage")
            judge_pack_payload = {}
            experiment_spec_id = self._required_string(workflow_input.payload, "experiment_spec_id")
        else:
            raise WorkflowIntegrityError("TraceRunInput packet_status is invalid")
        return (
            TraceWorkflowInput(
                run_id=run_id,
                dataset_version_id=self._required_string(trace.payload, "dataset_version_id"),
                experiment_spec_id=experiment_spec_id,
                trace_id=self._required_string(trace.payload, "trace_id"),
                trajectory_id=self._required_string(trajectory.payload, "trajectory_id"),
                prompt=self._required_string(trace.payload, "prompt"),
                response=self._required_string(trajectory.payload, "response"),
                judge_pack=judge_pack_payload,
                student_input_packet=packet_payload,
                raw_student_output=raw_output,
                role_session_lineage=session_lineage,
                student_packet_rejection=rejection_payload,
                rejected_raw_output_hash=rejected_raw_output_hash,
            ),
            config,
        )

    def _build_run_record(self, item: TraceWorkflowInput, events: list[Artifact]) -> Artifact:
        last = self._last_lifecycle_event(events)
        if last is None:
            raise WorkflowIntegrityError("cannot build RunRecord without lifecycle state")
        last_type = cast(str, last.payload["event_type"])
        last_details = self._details(last)
        reward_event = self._find_event(events, "REWARD_COMMITTED")
        reward_details = self._details(reward_event) if reward_event is not None else {}
        if last_type == "CLUSTER_ACCEPTED":
            status = "succeeded"
            reason_code = "TRACE_COMPLETE"
            evidence_hash: str | None = self._detail_string(reward_details, "evidence_hash")
            cluster_observation_hash: str | None = self._detail_string(last_details, "observation_hash")
            cluster_job_hash: str | None = self._detail_string(last_details, "job_hash")
            failure_observation_hash: str | None = None
            role_invocation_hash: str | None = self._detail_string(reward_details, "role_invocation_hash")
        else:
            status = "failed"
            reason_code = self._detail_string(last_details, "failure_code")
            evidence_hash = self._detail_string(reward_details, "evidence_hash") if reward_details else None
            cluster_observation_hash = (
                self._detail_string(last_details, "observation_hash") if last_type == "CLUSTER_FAILED" else None
            )
            cluster_job_hash = None
            failure_observation_hash = self._detail_string(last_details, "observation_hash")
            role_value = (
                reward_details.get("role_invocation_hash")
                if last_type == "CLUSTER_FAILED"
                else last_details.get("role_invocation_hash")
            )
            role_invocation_hash = role_value if isinstance(role_value, str) else None
        reward_hash = self._detail_string(reward_details, "reward_hash") if reward_details else None
        return self.store.put(
            "RunRecord",
            "1.0.0",
            {
                "cluster_job_hash": cluster_job_hash,
                "cluster_observation_hash": cluster_observation_hash,
                "event_chain_head": events[-1].content_hash,
                "evidence_hash": evidence_hash,
                "failure_observation_hash": failure_observation_hash,
                "reason_code": reason_code,
                "reward_hash": reward_hash,
                "role_invocation_hash": role_invocation_hash,
                "run_id": item.run_id,
                "status": status,
                "trace_id": item.trace_id,
            },
        )

    def _build_decision_outcome(
        self,
        run_id: str,
        run_record: Artifact,
        events: list[Artifact],
    ) -> Artifact:
        proposal_event = self._find_event(events, "CLUSTER_SUBMIT_REQUESTED")
        proposal_hash = (
            self._detail_string(self._details(proposal_event), "decision_proposal_hash")
            if proposal_event is not None
            else None
        )
        return self.store.put(
            "DecisionOutcome",
            "1.0.0",
            {
                "decision_proposal_hash": proposal_hash,
                "reason_code": self._required_string(run_record.payload, "reason_code"),
                "run_id": run_id,
                "run_record_hash": run_record.content_hash,
                "status": self._required_string(run_record.payload, "status"),
            },
        )

    @staticmethod
    def _validated_rejection_payload(
        value: Mapping[str, object],
    ) -> dict[str, JsonValue]:
        expected_fields = {
            "failure_code",
            "failure_detail",
            "input_byte_size",
            "input_hash",
            "packet_schema",
            "status",
        }
        allowed_details = {
            "student packet does not match the versioned adapter allowlist",
            "student packet exposes a prohibited role field",
            "student packet trajectory or JudgePack is invalid",
            "student packet nested fields do not match the versioned schema",
            "student packet does not match workflow identity or JudgePack",
        }
        input_hash = value.get("input_hash")
        input_byte_size = value.get("input_byte_size")
        if (
            set(value) != expected_fields
            or value.get("failure_code") != "ROLE_PACKET_NOT_ALLOWLISTED"
            or value.get("failure_detail") not in allowed_details
            or value.get("packet_schema") != "student-judge-input/1.x-rejected"
            or value.get("status") != "rejected"
            or (
                input_hash is not None
                and (not isinstance(input_hash, str) or re.fullmatch(r"[0-9a-f]{64}", input_hash) is None)
            )
            or (
                input_byte_size is not None
                and (not isinstance(input_byte_size, int) or isinstance(input_byte_size, bool) or input_byte_size < 0)
            )
        ):
            raise WorkflowIntegrityError("StudentPacketRejection payload is invalid")
        return cast(dict[str, JsonValue], dict(value))

    def _validate_scorer_outcome(
        self,
        observation: Artifact,
        request: Artifact,
        idempotency_key: str,
        scorer: PersistentFixtureScorer,
        expected_attempt_sequence: int,
    ) -> None:
        if not isinstance(self.config, FixtureProfileConfig):
            raise ArtifactCorruption("scorer outcome cannot be verified under a production profile")
        self._require_exact_artifact(
            request,
            "ScoringRequest",
            {
                "input_hash",
                "judge_pack_hash",
                "judge_pack_id",
                "packet_id",
                "packet_rejection_hash",
                "prompt",
                "response",
                "run_id",
                "trace_id",
                "trajectory_id",
            },
        )
        payload = observation.payload
        attempt = scorer.read_attempt(
            idempotency_key,
            request.content_hash,
            expected_attempt_sequence,
        )
        if attempt.payload.get("observation_hash") != observation.content_hash:
            raise ArtifactCorruption("scorer attempt does not bind the observation")
        if (
            payload.get("producer") != "fixture_student_judge"
            or payload.get("idempotency_key") != idempotency_key
            or payload.get("request_hash") != request.content_hash
        ):
            raise ArtifactCorruption("scorer observation identity is invalid")
        directive = payload.get("directive")
        if (
            payload.get("attempt_sequence") != expected_attempt_sequence
            or directive != self.config.effective_scorer_fault_schedule()[expected_attempt_sequence - 1]
        ):
            raise ArtifactCorruption("scorer observation attempt identity is invalid")
        status = payload.get("status")
        if status == "retryable":
            self._require_exact_artifact(
                observation,
                "Observation",
                {
                    "attempt_sequence",
                    "detail",
                    "directive",
                    "failure_code",
                    "idempotency_key",
                    "producer",
                    "request_hash",
                    "role_invocation_hash",
                    "status",
                },
            )
            expected_failure = {
                "timeout": (
                    "SCORER_TIMEOUT",
                    "deterministic scheduled scorer timeout",
                ),
                "delayed": (
                    "SCORER_RESULT_DELAYED",
                    "deterministic scheduled scorer result delay",
                ),
            }.get(cast(str, directive))
            if (
                expected_failure is None
                or payload.get("failure_code") != expected_failure[0]
                or payload.get("detail") != expected_failure[1]
                or payload.get("role_invocation_hash") is not None
            ):
                raise ArtifactCorruption("retryable scorer observation is invalid")
            return
        if status == "failed":
            self._require_exact_artifact(
                observation,
                "Observation",
                {
                    "attempt_sequence",
                    "detail",
                    "directive",
                    "failure_code",
                    "idempotency_key",
                    "producer",
                    "request_hash",
                    "role_invocation_hash",
                    "status",
                },
            )
            failure_code = payload.get("failure_code")
            detail = payload.get("detail")
            if not isinstance(failure_code, str) or not failure_code or not isinstance(detail, str) or not detail:
                raise ArtifactCorruption("failed scorer observation has no failure code")
            role_hash = payload.get("role_invocation_hash")
            if role_hash is None:
                if (
                    directive != "permanent_failure"
                    or failure_code != self.config.scorer_failure_code
                    or detail != "deterministic injected scorer failure"
                ):
                    raise ArtifactCorruption("unattributed scorer failure does not match injected configuration")
            else:
                if directive != "success":
                    raise ArtifactCorruption("role-attributed scorer failure has an invalid directive")
                role, _, normalized, _, _ = self._validated_role_exchange(
                    request,
                    expected_status="failed",
                    role_hash=self._lineage_hash(payload, "role_invocation_hash"),
                )
                self._require_exact_artifact(
                    normalized,
                    "NormalizedRoleOutput",
                    {
                        "failure_code",
                        "failure_detail",
                        "status",
                        "validation_limits",
                    },
                )
                if (
                    role.payload.get("failure_code") != failure_code
                    or normalized.payload.get("status") != "invalid"
                    or normalized.payload.get("failure_code") != failure_code
                    or normalized.payload.get("failure_detail") != detail
                    or normalized.payload.get("validation_limits")
                    != {
                        "max_bytes": 262_144,
                        "max_depth": 64,
                        "max_nodes": 10_000,
                        "max_string_characters": 32_768,
                        "max_total_string_characters": 131_072,
                    }
                ):
                    raise ArtifactCorruption("failed scorer role lineage is invalid")
            return
        if status != "succeeded":
            raise ArtifactCorruption("scorer observation status is invalid")
        if directive != "success":
            raise ArtifactCorruption("successful scorer observation has a non-success directive")

        self._require_exact_artifact(
            observation,
            "Observation",
            {
                "attempt_sequence",
                "directive",
                "evidence_hash",
                "idempotency_key",
                "producer",
                "request_hash",
                "reward_hash",
                "role_invocation_hash",
                "status",
            },
        )
        role, raw_output, normalized, packet, judge_pack = self._validated_role_exchange(
            request,
            expected_status="succeeded",
            role_hash=self._lineage_hash(payload, "role_invocation_hash"),
        )
        self._require_exact_artifact(
            normalized,
            "NormalizedRoleOutput",
            {
                "confidence_basis_points",
                "dimensions",
                "evidence",
                "failure_tags",
                "overall_scalar",
                "packet_id",
                "seed",
                "turn_local_tie_groups",
            },
        )
        try:
            expected_normalized, expected_reward_basis_points = scorer.recompute_output_contract(
                raw_output,
                packet.payload,
                judge_pack.payload,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ArtifactCorruption("committed raw role output no longer satisfies the scorer contract") from error
        if normalized.payload != expected_normalized:
            raise ArtifactCorruption("normalized role output does not match committed raw output")
        normalized_hash = normalized.content_hash
        evidence = self.store.read(
            self._lineage_hash(payload, "evidence_hash"),
            expected_schema_name="ScoringEvidence",
        )
        reward = self.store.read(
            self._lineage_hash(payload, "reward_hash"),
            expected_schema_name="EvidenceLinkedReward",
        )
        self._require_exact_artifact(
            evidence,
            "ScoringEvidence",
            {
                "dimension_scores",
                "evidence",
                "normalized_output_hash",
                "request_hash",
                "role_invocation_hash",
            },
        )
        self._require_exact_artifact(
            reward,
            "EvidenceLinkedReward",
            {
                "confidence_basis_points",
                "evidence_hash",
                "judge_pack_id",
                "normalized_output_hash",
                "request_hash",
                "reward_basis_points",
                "role_invocation_hash",
                "scalarizer_version",
            },
        )
        reward_basis_points = reward.payload.get("reward_basis_points")
        confidence_basis_points = reward.payload.get("confidence_basis_points")
        if (
            role.payload.get("request_hash") != request.content_hash
            or role.payload.get("status") != "succeeded"
            or evidence.payload.get("request_hash") != request.content_hash
            or evidence.payload.get("role_invocation_hash") != role.content_hash
            or evidence.payload.get("normalized_output_hash") != normalized_hash
            or evidence.payload.get("dimension_scores") != normalized.payload.get("dimensions")
            or evidence.payload.get("evidence") != normalized.payload.get("evidence")
            or reward.payload.get("request_hash") != request.content_hash
            or reward.payload.get("evidence_hash") != evidence.content_hash
            or reward.payload.get("role_invocation_hash") != role.content_hash
            or reward.payload.get("normalized_output_hash") != normalized_hash
            or reward.payload.get("judge_pack_id") != judge_pack.payload.get("judge_pack_id")
            or reward.payload.get("scalarizer_version") != judge_pack.payload.get("scalarizer_version")
            or not isinstance(reward_basis_points, int)
            or isinstance(reward_basis_points, bool)
            or not 0 <= reward_basis_points <= 10_000
            or reward_basis_points != expected_reward_basis_points
            or not isinstance(confidence_basis_points, int)
            or isinstance(confidence_basis_points, bool)
            or not 0 <= confidence_basis_points <= 10_000
            or normalized.payload.get("confidence_basis_points") != confidence_basis_points
        ):
            raise ArtifactCorruption("scorer outcome lineage is invalid")

    def _validated_role_exchange(
        self,
        request: Artifact,
        *,
        expected_status: str,
        role_hash: str,
    ) -> tuple[Artifact, bytes, Artifact, Artifact, Artifact]:
        role = self.store.read(role_hash, expected_schema_name="RoleInvocation")
        role_fields = {
            "input_hash",
            "input_packet_hash",
            "invocation",
            "normalized_output_hash",
            "output_hash",
            "packet_id",
            "producer",
            "raw_output_hash",
            "raw_output_manifest_hash",
            "request_hash",
            "role",
            "seed",
            "session_lineage",
            "status",
        }
        if expected_status == "failed":
            role_fields |= {"failure_code", "input_schema_name"}
        self._require_exact_artifact(role, "RoleInvocation", role_fields)

        workflow_input = self.store.read(
            self._lineage_hash(request.payload, "input_hash"),
            expected_schema_name="TraceRunInput",
        )
        self._require_exact_artifact(
            workflow_input,
            "TraceRunInput",
            {
                "experiment_spec_hash",
                "experiment_spec_id",
                "fixture_config_hash",
                "fixture_exchange_hash",
                "judge_pack_hash",
                "packet_rejection_hash",
                "packet_status",
                "rejected_raw_output_hash",
                "role_session_lineage",
                "run_id",
                "student_input_hash",
                "student_input_packet_hash",
                "trace_hash",
                "trajectory_hash",
            },
        )
        exchange = self.store.read(
            self._lineage_hash(workflow_input.payload, "fixture_exchange_hash"),
            expected_schema_name="StudentJudgeFixtureExchange",
        )
        self._require_exact_artifact(
            exchange,
            "StudentJudgeFixtureExchange",
            {
                "input_packet_hash",
                "raw_output_hash",
                "raw_output_manifest_hash",
                "session_lineage",
            },
        )
        packet = self.store.read(
            self._lineage_hash(exchange.payload, "input_packet_hash"),
            expected_schema_name="StudentJudgeInputPacket",
        )
        self._require_schema_version(packet, "StudentJudgeInputPacket")
        try:
            validate_student_packet_structure(packet.payload)
        except StudentPacketValidationError as error:
            raise ArtifactCorruption("committed Student packet is not allowlisted") from error
        judge_pack = self.store.read(
            self._lineage_hash(request.payload, "judge_pack_hash"),
            expected_schema_name="JudgePack",
        )
        self._require_schema_version(judge_pack, "JudgePack")
        raw_manifest = self.store.read(
            self._lineage_hash(exchange.payload, "raw_output_manifest_hash"),
            expected_schema_name="RawRoleOutput",
        )
        self._require_exact_artifact(
            raw_manifest,
            "RawRoleOutput",
            {"blob_hash", "byte_size", "media_type"},
        )
        normalized = self.store.read(
            self._lineage_hash(role.payload, "normalized_output_hash"),
            expected_schema_name="NormalizedRoleOutput",
        )

        trajectory = packet.payload.get("trajectory")
        packet_judge_pack = packet.payload.get("judge_pack")
        seed = packet.payload.get("seed")
        packet_id = packet.payload.get("packet_id")
        session_lineage = exchange.payload.get("session_lineage")
        canonical_input_hash = sha256_hex(canonical_json_bytes(packet.payload))
        raw_size = raw_manifest.payload.get("byte_size")
        raw_hash = raw_manifest.payload.get("blob_hash")
        if not isinstance(raw_size, int) or isinstance(raw_size, bool) or raw_size < 0:
            raise ArtifactCorruption("raw role output byte size is invalid")
        raw_output = self.store.read_blob(
            self._lineage_hash(raw_manifest.payload, "blob_hash"),
            expected_size=raw_size,
        )
        if (
            workflow_input.payload.get("packet_status") != "accepted"
            or workflow_input.payload.get("packet_rejection_hash") is not None
            or workflow_input.payload.get("rejected_raw_output_hash") is not None
            or workflow_input.payload.get("run_id") != request.payload.get("run_id")
            or workflow_input.payload.get("judge_pack_hash") != judge_pack.content_hash
            or workflow_input.payload.get("student_input_packet_hash") != packet.content_hash
            or workflow_input.payload.get("student_input_hash") != canonical_input_hash
            or workflow_input.payload.get("role_session_lineage") != session_lineage
            or not isinstance(trajectory, dict)
            or packet_judge_pack != judge_pack.payload
            or packet.payload.get("run_id") != request.payload.get("run_id")
            or packet.payload.get("trace_id") != request.payload.get("trace_id")
            or trajectory.get("trajectory_id") != request.payload.get("trajectory_id")
            or trajectory.get("prompt") != request.payload.get("prompt")
            or trajectory.get("response") != request.payload.get("response")
            or packet_id != request.payload.get("packet_id")
            or judge_pack.payload.get("judge_pack_id") != request.payload.get("judge_pack_id")
            or request.payload.get("packet_rejection_hash") is not None
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not 0 <= seed <= MAX_SAFE_INTEGER
            or not isinstance(packet_id, str)
            or not packet_id
            or not isinstance(session_lineage, str)
            or _SAFE_LINEAGE.fullmatch(session_lineage) is None
            or exchange.payload.get("raw_output_hash") != raw_hash
            or role.payload.get("input_hash") != canonical_input_hash
            or role.payload.get("input_packet_hash") != packet.content_hash
            or role.payload.get("invocation") != "student-judge"
            or role.payload.get("output_hash") != raw_hash
            or role.payload.get("packet_id") != packet_id
            or role.payload.get("producer") != "fixture_student_judge"
            or role.payload.get("raw_output_hash") != raw_hash
            or role.payload.get("raw_output_manifest_hash") != raw_manifest.content_hash
            or role.payload.get("request_hash") != request.content_hash
            or role.payload.get("role") != "student_judge"
            or role.payload.get("seed") != seed
            or role.payload.get("session_lineage") != session_lineage
            or role.payload.get("status") != expected_status
            or raw_manifest.payload.get("media_type") != "application/json; charset=utf-8"
        ):
            raise ArtifactCorruption("Student role invocation exchange is invalid")
        if expected_status == "failed" and role.payload.get("input_schema_name") != "StudentJudgeInputPacket":
            raise ArtifactCorruption("failed Student invocation input schema is invalid")
        return role, raw_output, normalized, packet, judge_pack

    @staticmethod
    def _require_schema_version(artifact: Artifact, schema_name: str) -> None:
        if artifact.schema_name != schema_name or artifact.schema_version != "1.0.0":
            raise ArtifactCorruption(f"{schema_name} artifact does not use the frozen 1.0.0 schema")

    @classmethod
    def _require_exact_artifact(
        cls,
        artifact: Artifact,
        schema_name: str,
        fields: set[str],
    ) -> None:
        cls._require_schema_version(artifact, schema_name)
        if set(artifact.payload) != fields:
            raise ArtifactCorruption(f"{schema_name} payload fields are invalid")

    def _validate_cluster_outcome(
        self,
        observation: Artifact,
        request: Artifact,
        idempotency_key: str,
        cluster: PersistentFixtureCluster,
        expected_attempt_sequence: int,
    ) -> None:
        if not isinstance(self.config, FixtureProfileConfig):
            raise ArtifactCorruption("cluster outcome cannot be verified under a production profile")
        self._require_exact_artifact(
            request,
            "ClusterSubmitRequest",
            {
                "decision_proposal_hash",
                "experiment_spec_hash",
                "fixture_config_hash",
                "reward_hash",
                "run_id",
                "trace_id",
            },
        )
        proposal = self.store.read(
            self._lineage_hash(request.payload, "decision_proposal_hash"),
            expected_schema_name="DecisionProposal",
        )
        fixture_config = self.store.read(
            self._lineage_hash(request.payload, "fixture_config_hash"),
            expected_schema_name="FixtureProfileConfig",
        )
        self._require_exact_artifact(
            fixture_config,
            "FixtureProfileConfig",
            {
                "cluster_failure_code",
                "cluster_fault_schedule",
                "execution_profile",
                "fault_schedule_version",
                "scorer_failure_code",
                "scorer_fault_schedule",
            },
        )
        self._require_exact_artifact(
            proposal,
            "DecisionProposal",
            {
                "action_type",
                "evidence_hash",
                "reward_hash",
                "run_id",
                "trace_id",
            },
        )
        reward_hash = self._lineage_hash(request.payload, "reward_hash")
        reward = self.store.read(
            reward_hash,
            expected_schema_name="EvidenceLinkedReward",
        )
        self._require_exact_artifact(
            reward,
            "EvidenceLinkedReward",
            {
                "confidence_basis_points",
                "evidence_hash",
                "judge_pack_id",
                "normalized_output_hash",
                "request_hash",
                "reward_basis_points",
                "role_invocation_hash",
                "scalarizer_version",
            },
        )
        evidence = self.store.read(
            self._lineage_hash(reward.payload, "evidence_hash"),
            expected_schema_name="ScoringEvidence",
        )
        self._require_exact_artifact(
            evidence,
            "ScoringEvidence",
            {
                "dimension_scores",
                "evidence",
                "normalized_output_hash",
                "request_hash",
                "role_invocation_hash",
            },
        )
        experiment = self.store.read(
            self._lineage_hash(request.payload, "experiment_spec_hash"),
            expected_schema_name="ExperimentSpec",
        )
        self._require_exact_artifact(
            experiment,
            "ExperimentSpec",
            {
                "dataset_version_id",
                "experiment_spec_id",
                "judge_pack_hash",
                "trace_hash",
            },
        )
        trace = self.store.read(
            self._lineage_hash(experiment.payload, "trace_hash"),
            expected_schema_name="TrainingTrace",
        )
        self._require_exact_artifact(
            trace,
            "TrainingTrace",
            {"dataset_version_id", "prompt", "trace_id"},
        )
        judge_pack = self.store.read(
            self._lineage_hash(experiment.payload, "judge_pack_hash"),
            expected_schema_name="JudgePack",
        )
        self._require_schema_version(judge_pack, "JudgePack")
        if (
            idempotency_key != self._action_key("cluster-submit", proposal.content_hash)
            or proposal.payload.get("action_type") != "submit_fixture_training_job"
            or fixture_config.payload != self.config.artifact_payload()
            or proposal.payload.get("reward_hash") != reward.content_hash
            or proposal.payload.get("evidence_hash") != evidence.content_hash
            or proposal.payload.get("run_id") != request.payload.get("run_id")
            or proposal.payload.get("trace_id") != request.payload.get("trace_id")
            or trace.payload.get("trace_id") != request.payload.get("trace_id")
            or trace.payload.get("dataset_version_id") != experiment.payload.get("dataset_version_id")
            or reward.payload.get("judge_pack_id") != judge_pack.payload.get("judge_pack_id")
            or reward.payload.get("scalarizer_version") != judge_pack.payload.get("scalarizer_version")
            or reward.payload.get("evidence_hash") != evidence.content_hash
            or reward.payload.get("normalized_output_hash") != evidence.payload.get("normalized_output_hash")
            or reward.payload.get("role_invocation_hash") != evidence.payload.get("role_invocation_hash")
            or not isinstance(request.payload.get("run_id"), str)
            or not request.payload.get("run_id")
            or not isinstance(request.payload.get("trace_id"), str)
            or not request.payload.get("trace_id")
        ):
            raise ArtifactCorruption("cluster request lineage is invalid")

        payload = observation.payload
        attempt = cluster.read_attempt(
            idempotency_key,
            request.content_hash,
            expected_attempt_sequence,
        )
        if attempt.payload.get("observation_hash") != observation.content_hash:
            raise ArtifactCorruption("cluster attempt does not bind the observation")
        if (
            payload.get("producer") != "fixture_cluster"
            or payload.get("idempotency_key") != idempotency_key
            or payload.get("request_hash") != request.content_hash
        ):
            raise ArtifactCorruption("cluster observation identity is invalid")
        directive = payload.get("directive")
        if (
            payload.get("attempt_sequence") != expected_attempt_sequence
            or directive != self.config.effective_cluster_fault_schedule()[expected_attempt_sequence - 1]
        ):
            raise ArtifactCorruption("cluster observation attempt identity is invalid")
        status = payload.get("status")
        if status == "retryable":
            self._require_exact_artifact(
                observation,
                "Observation",
                {
                    "attempt_sequence",
                    "detail",
                    "directive",
                    "failure_code",
                    "idempotency_key",
                    "producer",
                    "request_hash",
                    "status",
                },
            )
            expected_failure = {
                "timeout": (
                    "CLUSTER_TIMEOUT",
                    "deterministic scheduled cluster timeout",
                ),
                "delayed": (
                    "CLUSTER_RESULT_DELAYED",
                    "deterministic scheduled cluster result delay",
                ),
            }.get(cast(str, directive))
            if (
                expected_failure is None
                or payload.get("failure_code") != expected_failure[0]
                or payload.get("detail") != expected_failure[1]
            ):
                raise ArtifactCorruption("retryable cluster observation is invalid")
            return
        if status == "failed":
            self._require_exact_artifact(
                observation,
                "Observation",
                {
                    "attempt_sequence",
                    "detail",
                    "directive",
                    "failure_code",
                    "idempotency_key",
                    "producer",
                    "request_hash",
                    "status",
                },
            )
            failure_code = payload.get("failure_code")
            if (
                not isinstance(failure_code, str)
                or not failure_code
                or directive != "permanent_failure"
                or failure_code != fixture_config.payload.get("cluster_failure_code")
                or payload.get("detail") != "deterministic injected cluster failure"
            ):
                raise ArtifactCorruption("cluster failure does not match injected adapter configuration")
            return
        if status != "succeeded":
            raise ArtifactCorruption("cluster observation status is invalid")
        if directive != "success":
            raise ArtifactCorruption("successful cluster observation has a non-success directive")
        self._require_exact_artifact(
            observation,
            "Observation",
            {
                "attempt_sequence",
                "directive",
                "idempotency_key",
                "job_hash",
                "producer",
                "request_hash",
                "status",
            },
        )
        job = self.store.read(
            self._lineage_hash(payload, "job_hash"),
            expected_schema_name="FixtureClusterJob",
        )
        self._require_exact_artifact(
            job,
            "FixtureClusterJob",
            {
                "idempotency_key",
                "job_id",
                "request_hash",
                "reward_hash",
                "status",
            },
        )
        if (
            job.payload.get("idempotency_key") != idempotency_key
            or job.payload.get("job_id") != f"fixture-job-{request.content_hash[:16]}"
            or job.payload.get("request_hash") != request.content_hash
            or job.payload.get("reward_hash") != reward_hash
            or job.payload.get("status") != "accepted"
        ):
            raise ArtifactCorruption("cluster job lineage is invalid")

    def _integrity_failure_observation(
        self,
        *,
        producer: str,
        failure_code: str,
        detail: str,
        idempotency_key: str,
        request_hash: str,
        source_event_hash: str,
        candidate_observation_hash: str | None,
        include_role_hash: bool,
    ) -> Artifact:
        payload: dict[str, object] = {
            "candidate_observation_hash": candidate_observation_hash,
            "detail": detail,
            "failure_code": failure_code,
            "idempotency_key": idempotency_key,
            "producer": producer,
            "request_hash": request_hash,
            "source_event_hash": source_event_hash,
            "status": "failed",
        }
        if include_role_hash:
            payload["role_invocation_hash"] = None
        return self.store.put("Observation", "1.0.0", payload)

    @staticmethod
    def _lineage_hash(value: Mapping[str, object], field: str) -> str:
        content_hash = value.get(field)
        if not isinstance(content_hash, str) or re.fullmatch(r"[0-9a-f]{64}", content_hash) is None:
            raise ArtifactCorruption(f"outcome lineage is missing {field}")
        return content_hash

    @staticmethod
    def _action_key(action_type: str, identity_hash: str) -> str:
        return sha256_hex(canonical_json_bytes({"action_type": action_type, "identity_hash": identity_hash}))

    @staticmethod
    def _artifact_content_hash(
        schema_name: str,
        payload: Mapping[str, object],
    ) -> str:
        return sha256_hex(
            canonical_json_bytes(
                {
                    "payload": dict(payload),
                    "schema_name": schema_name,
                    "schema_version": "1.0.0",
                }
            )
        )

    @staticmethod
    def _required_artifact(
        artifacts: Mapping[str, Artifact | None],
        field: str,
    ) -> Artifact:
        artifact = artifacts.get(field)
        if artifact is None:
            raise WorkflowIntegrityError(f"materialized input is missing {field}")
        return artifact

    @staticmethod
    def _append_transition(
        journal: RunJournal,
        epoch: int,
        event_type: str,
        details: Mapping[str, object],
        events: list[Artifact],
    ) -> bool:
        try:
            journal.append(
                epoch,
                event_type,
                details,
                expected_sequence=len(events) + 1,
                expected_previous_hash=events[-1].content_hash if events else None,
            )
        except JournalHeadConflict:
            return False
        return True

    @staticmethod
    def _details(event: Artifact) -> dict[str, JsonValue]:
        details = event.payload.get("details")
        if not isinstance(details, dict):
            raise WorkflowIntegrityError("RunEvent details are corrupt")
        return details

    @staticmethod
    def _detail_string(details: Mapping[str, object], field: str) -> str:
        value = details.get(field)
        if not isinstance(value, str) or not value:
            raise WorkflowIntegrityError(f"persisted event is missing {field}")
        return value

    @staticmethod
    def _detail_positive_integer(
        details: Mapping[str, object],
        field: str,
    ) -> int:
        value = details.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > MAX_SAFE_INTEGER:
            raise WorkflowIntegrityError(f"persisted event is missing {field}")
        return value

    @staticmethod
    def _required_string(value: Mapping[str, object], field: str) -> str:
        item = value.get(field)
        if not isinstance(item, str) or not item:
            raise WorkflowIntegrityError(f"input or artifact is missing {field}")
        return item

    @staticmethod
    def _find_event(events: list[Artifact], event_type: str) -> Artifact | None:
        return next(
            (event for event in events if event.payload.get("event_type") == event_type),
            None,
        )

    @staticmethod
    def _last_lifecycle_event(events: list[Artifact]) -> Artifact | None:
        return next(
            (event for event in reversed(events) if event.payload.get("event_type") in _LIFECYCLE_EVENT_TYPES),
            None,
        )
