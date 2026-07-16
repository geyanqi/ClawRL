"""Fenced Ticket 04 workflow from one verified TrainingTrace to Phase A role input."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from clawrl.adapters.generators.fixture import FixtureFitGenerator, GeneratorBoundaryError
from clawrl.adapters.scorers.role_isolation import (
    RoleIsolationIngress,
    RoleIsolationIngressReceipt,
    RoleIsolationOutputError,
    alignment_auditor_output_schema_contract,
    prompt_optimizer_output_schema_contract,
)
from clawrl.adapters.scorers.teacher import (
    TeacherOutputError,
    TeacherRoleIngressReceipt,
    TeacherScorerIngress,
    teacher_output_schema_contract,
)
from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.data.models import DatasetValidationError
from clawrl.data.validation import LoadedTrainingDataset, load_training_dataset
from clawrl.judge.fit_models import (
    FitContractError,
    FitValidationError,
    FixtureFitConfig,
    GeneratorPlan,
    ProductionJudgeCertifyConfig,
)
from clawrl.training.run_journal import RunIdentityConflict, RunJournal, StaleFencingEpoch

_HASH = re.compile(r"^[0-9a-f]{64}$")
_INPUT_SCHEMA = "FitTraceWorkflowInput"
_INPUT_FIELDS = {
    "config",
    "dataset_version_hash",
    "generator_inference_config_hash",
    "initial_eval_rubric_hash",
    "run_id",
    "schema_version",
    "sol_inference_config_hash",
    "trace_id",
    "training_trace_hash",
}
_TRAJECTORY_CONTENT_FIELDS = {
    "prompt",
    "prompt_hash",
    "response",
    "response_hash",
    "schema_version",
    "split",
    "trace_id",
    "trajectory_id",
}
_TRAJECTORY_MANIFEST_FIELDS = {
    "content_hash",
    "dataset_version_hash",
    "lineage",
    "origin",
    "prompt_hash",
    "response_hash",
    "schema_version",
    "split",
    "trace_id",
    "training_trace_hash",
    "trajectory_id",
}


class FitWorkflowError(RuntimeError):
    """Ticket 04 workflow cannot verify or advance durable state."""


@dataclass(frozen=True, slots=True)
class FitWorkflowSnapshot:
    events: tuple[Artifact, ...]
    generator_plan: Artifact | None
    fit_trajectory_set: Artifact | None
    teacher_packet: Artifact | None
    phase_a_checkpoint: Artifact | None
    teacher_label_set: Artifact | None
    prompt_optimizer_packet: Artifact | None
    alignment_auditor_packet: Artifact | None
    prompt_optimizer_output: Artifact | None
    alignment_auditor_output: Artifact | None
    role_isolation_attestation: Artifact | None
    terminal: bool


@dataclass(frozen=True, slots=True)
class _State:
    store: ArtifactStore
    journal: RunJournal
    workflow_input: Artifact
    config: FixtureFitConfig
    loaded_dataset: LoadedTrainingDataset
    training_trace: Artifact
    events: tuple[Artifact, ...]
    plan: Artifact | None
    request: Artifact | None
    fit_set: Artifact | None
    teacher_packet: Artifact | None
    checkpoint: Artifact | None
    teacher_normalized: Artifact | None
    teacher_audit: Artifact | None
    teacher_label_set: Artifact | None
    base_judge_prompt: Artifact | None
    role_isolation_policy: Artifact | None
    prompt_optimizer_packet: Artifact | None
    alignment_auditor_packet: Artifact | None
    prompt_optimizer_output: Artifact | None
    prompt_optimizer_audit: Artifact | None
    alignment_auditor_output: Artifact | None
    alignment_auditor_audit: Artifact | None
    role_isolation_attestation: Artifact | None


class FitTrajectoryWorkflow:
    """Advance a single exact-32 fit generation run one durable transition at a time."""

    @staticmethod
    def production_readiness(root: str | Path, config: ProductionJudgeCertifyConfig) -> Artifact:
        """Fail closed before model/generator boundary construction for missing approvals."""

        checks: list[dict[str, str]] = []
        values = (
            ("SOL_MODEL_CONFIGURATION_UNAVAILABLE", config.sol_model_approval_hash),
            ("INITIAL_EVAL_RUBRIC_UNAVAILABLE", config.initial_eval_rubric_approval_hash),
            ("GENERATOR_CONFIGURATION_UNAVAILABLE", config.generator_approval_hash),
            (
                "PERMANENT_TRACE_GOVERNANCE_APPROVAL_UNAVAILABLE",
                config.permanent_trace_governance_approval_hash,
            ),
        )
        for code, value in values:
            if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
                checks.append({"code": code, "status": "blocked"})
        if not checks:
            checks.append({"code": "PRODUCTION_JUDGE_BOUNDARY_NOT_CONFIGURED", "status": "blocked"})
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": checks,
                "execution_profile": "production",
                "phase": "JUDGE_CERTIFY",
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )

    @staticmethod
    def _journal(root: Path, store: ArtifactStore, run_id: str) -> RunJournal:
        return RunJournal(root, store, run_id, input_schema_name=_INPUT_SCHEMA, input_schema_version="1.0.0")

    @classmethod
    def bootstrap(
        cls,
        root: str | Path,
        config: FixtureFitConfig,
        *,
        epoch: int,
    ) -> FitWorkflowSnapshot:
        root_path = Path(root)
        store = ArtifactStore(root_path)
        try:
            loaded = load_training_dataset(store, config.dataset_version_hash)
            trace = cls._select_trace(loaded, config.trace_id)
            input_payload = cls._input_payload(config, trace)
            input_hash = cls._artifact_hash(_INPUT_SCHEMA, input_payload)
            journal = cls._journal(root_path, store, config.run_id)
            journal.reserve_identity(input_hash)
        except (DatasetValidationError, FitContractError, ArtifactCorruption) as error:
            raise FitWorkflowError("fit workflow input cannot be verified") from error
        except RunIdentityConflict as error:
            raise FitWorkflowError("run_id is bound to a different immutable fit input") from error

        input_path = store.artifact_dir / f"{input_hash}.json"
        if input_path.exists():
            persisted = store.read(input_hash, expected_schema_name=_INPUT_SCHEMA)
            if persisted.payload != input_payload:
                raise FitWorkflowError("persisted fit input conflicts with its reserved identity")
            if journal.events():
                return cls._snapshot(cls._load_state(root_path, config.run_id))

        rubric = store.put("InitialEvalRubric", "1.0.0", config.rubric.artifact_payload())
        sol = store.put("SolInferenceConfig", "1.0.0", config.sol_inference.artifact_payload())
        generator_inference = store.put(
            "GeneratorInferenceConfig",
            "1.0.0",
            config.generator_inference_payload,
        )
        workflow_input = store.put(_INPUT_SCHEMA, "1.0.0", input_payload)
        if (
            workflow_input.content_hash != input_hash
            or rubric.content_hash != input_payload["initial_eval_rubric_hash"]
            or sol.content_hash != input_payload["sol_inference_config_hash"]
            or generator_inference.content_hash != input_payload["generator_inference_config_hash"]
        ):
            raise FitWorkflowError("fit input dependency hashes changed during publication")
        journal.start_run(
            epoch,
            workflow_input.content_hash,
            {
                "input_hash": workflow_input.content_hash,
                "phase": "fit_trajectory_generation",
                "trace_id": config.trace_id,
            },
        )
        return cls._snapshot(cls._load_state(root_path, config.run_id))

    @classmethod
    def run_once(
        cls,
        root: str | Path,
        config: FixtureFitConfig,
        *,
        epoch: int,
    ) -> FitWorkflowSnapshot:
        snapshot = cls.bootstrap(root, config, epoch=epoch)
        for _transition in range(64):
            if snapshot.phase_a_checkpoint is not None or snapshot.terminal:
                return snapshot
            snapshot = cls.resume(root, config.run_id, epoch=epoch)
        raise FitWorkflowError("fit workflow exceeded its bounded transition count")

    @classmethod
    def submit_teacher_output(
        cls,
        root: str | Path,
        run_id: str,
        *,
        role_ingress: TeacherRoleIngressReceipt,
        epoch: int,
    ) -> FitWorkflowSnapshot:
        """Bind one validated isolated TeacherScorer exchange to the Phase A checkpoint."""

        root_path = Path(root)
        state = cls._load_state(root_path, run_id)
        if state.teacher_packet is None or state.checkpoint is None:
            raise FitWorkflowError("TeacherScorer output cannot precede the Phase A checkpoint")
        try:
            contract = TeacherScorerIngress.load_public_contract(root_path, role_ingress)
        except (TeacherOutputError, ArtifactCorruption) as error:
            raise FitWorkflowError("TeacherScorer ingress cannot be recertified") from error
        if role_ingress.input_packet_hash != state.teacher_packet.content_hash:
            raise FitWorkflowError("TeacherScorer ingress belongs to a different input packet")
        accepted = next(
            (event for event in state.events if event.payload.get("event_type") == "TEACHER_OUTPUT_ACCEPTED"),
            None,
        )
        expected_details = {"teacher_ingress": role_ingress.artifact_payload()}
        if accepted is not None:
            if accepted.payload.get("details") != expected_details:
                raise FitWorkflowError("a different immutable TeacherScorer output is already committed")
            return cls._snapshot(state)
        if cls._terminal(state.events) or state.events[-1].payload.get("event_type") != "PHASE_A_CHECKPOINT":
            raise FitWorkflowError("TeacherScorer output cannot be committed in the current state")
        if (
            contract.normalized_output.content_hash != role_ingress.normalized_output_hash
            or contract.role_invocation_audit.content_hash != role_ingress.role_invocation_audit_hash
        ):
            raise FitWorkflowError("TeacherScorer public contract conflicts with its receipt")
        state.journal.claim_epoch(epoch)
        state.journal.append(
            epoch,
            "TEACHER_OUTPUT_ACCEPTED",
            expected_details,
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )
        return cls._snapshot(cls._load_state(root_path, run_id))

    @classmethod
    def submit_role_output(
        cls,
        root: str | Path,
        run_id: str,
        *,
        role_ingress: RoleIsolationIngressReceipt,
        epoch: int,
    ) -> FitWorkflowSnapshot:
        """Commit one recertified isolated optimizer or auditor exchange."""

        root_path = Path(root)
        state = cls._load_state(root_path, run_id)
        if state.prompt_optimizer_packet is None or state.alignment_auditor_packet is None:
            raise FitWorkflowError("role output cannot precede the isolation packets")
        if cls._terminal(state.events) or state.role_isolation_attestation is not None:
            raise FitWorkflowError("role output cannot be committed after isolation attestation")
        expected_packet = (
            state.prompt_optimizer_packet
            if role_ingress.role_type == "PromptOptimizer"
            else state.alignment_auditor_packet
        )
        if role_ingress.input_packet_hash != expected_packet.content_hash:
            raise FitWorkflowError("role ingress belongs to a different role input packet")
        try:
            contract = RoleIsolationIngress.load_public_contract(root_path, role_ingress)
        except (RoleIsolationOutputError, ArtifactCorruption) as error:
            raise FitWorkflowError("role ingress cannot be recertified") from error
        event_type = (
            "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED"
            if role_ingress.role_type == "PromptOptimizer"
            else "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED"
        )
        existing = next(
            (event for event in state.events if event.payload.get("event_type") == event_type),
            None,
        )
        expected_details = {"role_ingress": role_ingress.artifact_payload()}
        if existing is not None:
            if existing.payload.get("details") != expected_details:
                raise FitWorkflowError(f"a different immutable {role_ingress.role_type} output is already committed")
            return cls._snapshot(state)
        if (
            contract.role_type != role_ingress.role_type
            or contract.normalized_output.content_hash != role_ingress.normalized_output_hash
            or contract.role_invocation_audit.content_hash != role_ingress.role_invocation_audit_hash
        ):
            raise FitWorkflowError("role public contract conflicts with its receipt")
        state.journal.claim_epoch(epoch)
        state.journal.append(
            epoch,
            event_type,
            expected_details,
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )
        return cls._snapshot(cls._load_state(root_path, run_id))

    @classmethod
    def resume(cls, root: str | Path, run_id: str, *, epoch: int) -> FitWorkflowSnapshot:
        root_path = Path(root)
        state = cls._load_state(root_path, run_id)
        if cls._terminal(state.events):
            return cls._snapshot(state)
        if state.events[-1].payload.get("event_type") == "PHASE_A_CHECKPOINT":
            return cls._snapshot(state)
        if state.events[-1].payload.get("event_type") == "ROLE_ISOLATION_PACKETS_COMMITTED":
            return cls._snapshot(state)
        if state.events[-1].payload.get("event_type") in {
            "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED",
            "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED",
        } and (state.prompt_optimizer_output is None or state.alignment_auditor_output is None):
            return cls._snapshot(state)
        state.journal.claim_epoch(epoch)
        event_type = cast(str, state.events[-1].payload["event_type"])

        if event_type == "RUN_STARTED":
            plan = GeneratorPlan.create(
                dataset_version_hash=state.config.dataset_version_hash,
                training_trace_hash=state.training_trace.content_hash,
                trace_id=state.config.trace_id,
                seed=state.config.generator_seed,
                include_online_response=state.config.include_online_response,
                generator_profile_id=state.config.generator_profile_id,
                generator_model_id=state.config.generator_model_id,
                inference_config_hash=cast(str, state.workflow_input.payload["generator_inference_config_hash"]),
            )
            plan_artifact = state.store.put("GeneratorPlan", "1.0.0", plan.artifact_payload())
            state.journal.append(
                epoch,
                "GENERATOR_PLAN_FROZEN",
                {"generator_plan_hash": plan_artifact.content_hash},
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
        elif event_type == "GENERATOR_PLAN_FROZEN":
            if state.plan is None:
                raise FitWorkflowError("frozen GeneratorPlan is unavailable")
            request = state.store.put(
                "GeneratorRequest",
                "1.0.0",
                {
                    "dataset_version_hash": state.config.dataset_version_hash,
                    "generator_plan_hash": state.plan.content_hash,
                    "online_response": state.training_trace.payload["response"],
                    "output_fault": state.config.output_fault,
                    "prompt": state.training_trace.payload["prompt"],
                    "schema_version": "generator-request/1.0.0",
                    "trace_id": state.config.trace_id,
                    "training_trace_hash": state.training_trace.content_hash,
                },
            )
            state.journal.append(
                epoch,
                "GENERATION_REQUESTED",
                {"generator_request_hash": request.content_hash},
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
        elif event_type in {"GENERATION_REQUESTED", "GENERATION_RETRY_SCHEDULED"}:
            cls._execute_generator_transition(state, epoch)
        elif event_type == "TRAJECTORIES_COMMITTED":
            if state.fit_set is None:
                raise FitWorkflowError("committed fit trajectory set is unavailable")
            packet = cls._build_teacher_packet(state)
            state.journal.append(
                epoch,
                "TEACHER_PACKET_COMMITTED",
                {"teacher_packet_hash": packet.content_hash},
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
        elif event_type == "TEACHER_PACKET_COMMITTED":
            if state.fit_set is None or state.teacher_packet is None or state.plan is None:
                raise FitWorkflowError("phase A inputs are incomplete")
            decision = state.store.put(
                "DecisionRecord",
                "1.0.0",
                {
                    "decision": "freeze_fixture_fit_generation_defaults",
                    "generator_plan_hash": state.plan.content_hash,
                    "input_hash": state.workflow_input.content_hash,
                    "reason_code": "TICKET04_RECOMMENDED_DEFAULTS_ADOPTED",
                    "run_id": state.config.run_id,
                    "teacher_packet_hash": state.teacher_packet.content_hash,
                },
            )
            state.journal.append(
                epoch,
                "DECISION_RECORDED",
                {"decision_record_hash": decision.content_hash},
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
        elif event_type == "DECISION_RECORDED":
            if state.fit_set is None or state.teacher_packet is None or state.plan is None:
                raise FitWorkflowError("phase A inputs are incomplete")
            decision_details = state.events[-1].payload.get("details")
            if not isinstance(decision_details, dict) or type(decision_details.get("decision_record_hash")) is not str:
                raise FitWorkflowError("DecisionRecord event is invalid")
            checkpoint = state.store.put(
                "PhaseACheckpoint",
                "1.0.0",
                {
                    "decision_record_hash": decision_details["decision_record_hash"],
                    "fit_trajectory_set_hash": state.fit_set.content_hash,
                    "generator_plan_hash": state.plan.content_hash,
                    "next_required_role": "TeacherScorer",
                    "run_id": state.config.run_id,
                    "status": "awaiting_isolated_teacher_output",
                    "teacher_packet_hash": state.teacher_packet.content_hash,
                    "trace_id": state.config.trace_id,
                },
            )
            state.journal.append(
                epoch,
                "PHASE_A_CHECKPOINT",
                {"phase_a_checkpoint_hash": checkpoint.content_hash},
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
        elif event_type == "TEACHER_OUTPUT_ACCEPTED":
            if state.teacher_normalized is None or state.teacher_audit is None:
                raise FitWorkflowError("accepted TeacherScorer output is unavailable")
            label_set = cls._build_teacher_label_set(state)
            state.journal.append(
                epoch,
                "TEACHER_LABEL_SET_COMMITTED",
                {"teacher_label_set_hash": label_set.content_hash},
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
        elif event_type == "TEACHER_LABEL_SET_COMMITTED":
            if state.teacher_label_set is None:
                raise FitWorkflowError("TeacherLabelSet is unavailable")
            base_prompt, isolation_policy, optimizer_packet, auditor_packet, decision = cls._build_role_packets(state)
            state.journal.append(
                epoch,
                "ROLE_ISOLATION_PACKETS_COMMITTED",
                {
                    "alignment_auditor_packet_hash": auditor_packet.content_hash,
                    "base_judge_prompt_hash": base_prompt.content_hash,
                    "decision_record_hash": decision.content_hash,
                    "prompt_optimizer_packet_hash": optimizer_packet.content_hash,
                    "role_isolation_policy_hash": isolation_policy.content_hash,
                },
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
        elif event_type in {
            "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED",
            "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED",
        }:
            if state.prompt_optimizer_output is None or state.alignment_auditor_output is None:
                raise FitWorkflowError("both isolated role outputs are required for attestation")
            attestation = cls._build_role_isolation_attestation(state)
            state.journal.append(
                epoch,
                "ROLE_ISOLATION_ATTESTATION_COMMITTED",
                {"role_isolation_attestation_hash": attestation.content_hash},
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
        elif event_type == "ROLE_ISOLATION_ATTESTATION_COMMITTED":
            if state.role_isolation_attestation is None:
                raise FitWorkflowError("role-isolation attestation is unavailable")
            state.journal.close(
                epoch,
                status="succeeded",
                reason_code="TICKET04_COMPLETE",
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
        else:
            raise FitWorkflowError("fit workflow event cannot be advanced")
        return cls._snapshot(cls._load_state(root_path, run_id))

    @classmethod
    def _execute_generator_transition(cls, state: _State, epoch: int) -> None:
        if state.request is None or state.plan is None:
            raise FitWorkflowError("generator request is unavailable")
        retry_events = sum(event.payload.get("event_type") == "GENERATION_RETRY_SCHEDULED" for event in state.events)
        attempt_sequence = retry_events + 1
        adapter = FixtureFitGenerator(state.store.root, state.store, state.config)
        try:
            observation = adapter.execute(state.request, attempt_sequence=attempt_sequence)
            attempts = adapter.verify_attempt_chain(state.request, attempt_sequence)
        except (GeneratorBoundaryError, ArtifactCorruption) as error:
            raise FitWorkflowError("generator boundary evidence is invalid") from error
        status = observation.payload.get("status")
        if status == "retryable":
            if attempt_sequence >= len(state.config.fault_schedule):
                cls._record_failure(state, epoch, "GENERATOR_RETRY_EXHAUSTED")
                return
            state.journal.append(
                epoch,
                "GENERATION_RETRY_SCHEDULED",
                {
                    "attempt_count": attempt_sequence,
                    "failure_code": observation.payload["failure_code"],
                    "generator_attempt_hash": attempts[-1].content_hash,
                    "observation_hash": observation.content_hash,
                },
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
            return
        if status == "failed":
            cls._record_failure(state, epoch, "GENERATOR_PERMANENT_FAILURE")
            return
        try:
            raw = adapter.read_raw_batch(observation, state.request)
            fit_set = cls._commit_trajectories(state, raw, observation, attempts[-1])
        except (GeneratorBoundaryError, FitValidationError, ArtifactCorruption, TypeError, ValueError):
            cls._record_failure(state, epoch, "FIT_TRAJECTORY_BATCH_INVALID")
            return
        state.journal.append(
            epoch,
            "TRAJECTORIES_COMMITTED",
            {
                "attempt_count": attempt_sequence,
                "fit_trajectory_set_hash": fit_set.content_hash,
                "generator_attempt_hash": attempts[-1].content_hash,
                "observation_hash": observation.content_hash,
            },
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )

    @classmethod
    def _record_failure(cls, state: _State, epoch: int, reason_code: str) -> None:
        decision = state.store.put(
            "DecisionRecord",
            "1.0.0",
            {
                "input_hash": state.workflow_input.content_hash,
                "reason_code": reason_code,
                "run_id": state.config.run_id,
                "status": "failed",
            },
        )
        failure = state.journal.append(
            epoch,
            "FIT_FAILURE_RECORDED",
            {"decision_record_hash": decision.content_hash, "reason_code": reason_code},
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )
        state.journal.close(
            epoch,
            status="failed",
            reason_code=reason_code,
            expected_sequence=len(state.events) + 2,
            expected_previous_hash=failure.content_hash,
        )

    @classmethod
    def _commit_trajectories(
        cls,
        state: _State,
        raw: Artifact,
        observation: Artifact,
        attempt: Artifact,
    ) -> Artifact:
        if state.plan is None:
            raise FitValidationError("GeneratorPlan is unavailable")
        plan = GeneratorPlan.from_mapping(cast(dict[str, object], state.plan.payload))
        items = raw.payload.get("items")
        if not isinstance(items, list) or len(items) != 32:
            raise FitValidationError("generator batch must contain exactly 32 items")
        online_response = state.training_trace.payload.get("response")
        prompt = state.training_trace.payload.get("prompt")
        if type(online_response) is not str or type(prompt) is not str:
            raise FitValidationError("TrainingTrace content is invalid")
        response_hashes: set[str] = set()
        trajectory_ids: set[str] = set()
        refs: list[dict[str, str]] = []
        for index, (item, slot) in enumerate(zip(items, plan.slots, strict=True)):
            if not isinstance(item, dict) or set(item) != {"origin", "prompt", "response", "slot_id", "slot_index"}:
                raise FitValidationError("generator item fields are invalid")
            response = item.get("response")
            if (
                item.get("slot_index") != index
                or type(item.get("slot_index")) is not int
                or item.get("slot_id") != slot.slot_id
                or item.get("origin") != slot.origin
                or item.get("prompt") != prompt
                or type(response) is not str
                or not 24 <= len(cast(str, response)) <= 4096
            ):
                raise FitValidationError("generator item does not match its frozen slot")
            if slot.origin == "online_response":
                if not plan.include_online_response or response != online_response:
                    raise FitValidationError("online response was not explicitly and exactly admitted")
            elif response == online_response:
                raise FitValidationError("fresh rollout copied the online response")
            prompt_hash = sha256_hex(cast(str, prompt).encode("utf-8"))
            response_hash = sha256_hex(cast(str, response).encode("utf-8"))
            if response_hash in response_hashes:
                raise FitValidationError("duplicate responses cannot pad fit cardinality")
            response_hashes.add(response_hash)
            trajectory_identity = sha256_hex(
                canonical_json_bytes(
                    {
                        "domain": "fit-trajectory-identity/1.0.0",
                        "prompt_hash": prompt_hash,
                        "response_hash": response_hash,
                        "split": "fit",
                        "trace_id": state.config.trace_id,
                    }
                )
            )
            trajectory_id = f"trj-{trajectory_identity[:40]}"
            if trajectory_id in trajectory_ids:
                raise FitValidationError("fit trajectory identity is duplicated")
            trajectory_ids.add(trajectory_id)
            content = state.store.put(
                "TrajectoryContent",
                "1.0.0",
                {
                    "prompt": prompt,
                    "prompt_hash": prompt_hash,
                    "response": response,
                    "response_hash": response_hash,
                    "schema_version": "trajectory-content/1.0.0",
                    "split": "fit",
                    "trace_id": state.config.trace_id,
                    "trajectory_id": trajectory_id,
                },
            )
            manifest = state.store.put(
                "TrajectoryManifest",
                "1.0.0",
                {
                    "content_hash": content.content_hash,
                    "dataset_version_hash": state.config.dataset_version_hash,
                    "lineage": {
                        "generator_attempt_hash": attempt.content_hash,
                        "generator_observation_hash": observation.content_hash,
                        "generator_plan_hash": state.plan.content_hash,
                        "slot_id": slot.slot_id,
                        "slot_index": slot.slot_index,
                    },
                    "origin": slot.origin,
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                    "schema_version": "trajectory-manifest/1.0.0",
                    "split": "fit",
                    "trace_id": state.config.trace_id,
                    "training_trace_hash": state.training_trace.content_hash,
                    "trajectory_id": trajectory_id,
                },
            )
            refs.append({"manifest_hash": manifest.content_hash, "trajectory_id": trajectory_id})
        if len(refs) != 32 or len(response_hashes) != 32 or len(trajectory_ids) != 32:
            raise FitValidationError("fit trajectory cardinality is incomplete")
        return state.store.put(
            "FitTrajectorySet",
            "1.0.0",
            {
                "dataset_version_hash": state.config.dataset_version_hash,
                "generator_plan_hash": state.plan.content_hash,
                "split": "fit",
                "trace_id": state.config.trace_id,
                "training_trace_hash": state.training_trace.content_hash,
                "trajectory_count": 32,
                "trajectory_refs": refs,
                "trajectory_set_hash": sha256_hex(
                    canonical_json_bytes({"domain": "fit-trajectory-set/1.0.0", "trajectory_refs": refs})
                ),
            },
        )

    @classmethod
    def _build_teacher_packet(cls, state: _State) -> Artifact:
        if state.fit_set is None:
            raise FitWorkflowError("fit trajectory set is unavailable")
        contents = cls._validate_fit_set(state, state.fit_set)
        rubric_hash = cast(str, state.workflow_input.payload["initial_eval_rubric_hash"])
        sol_hash = cast(str, state.workflow_input.payload["sol_inference_config_hash"])
        rubric = state.store.read(rubric_hash, expected_schema_name="InitialEvalRubric")
        sol = state.store.read(sol_hash, expected_schema_name="SolInferenceConfig")
        turns: list[dict[str, JsonValue]] = []
        for turn_index in range(8):
            turn_contents = contents[turn_index * 4 : turn_index * 4 + 4]
            items: list[dict[str, JsonValue]] = []
            for content in turn_contents:
                items.append(
                    {
                        "content_hash": content.content_hash,
                        "prompt": content.payload["prompt"],
                        "prompt_hash": content.payload["prompt_hash"],
                        "response": content.payload["response"],
                        "response_hash": content.payload["response_hash"],
                        "trajectory_id": content.payload["trajectory_id"],
                    }
                )
            turns.append(
                {
                    "items": cast(JsonValue, items),
                    "thread_id": f"sol-thread-{turn_index % 5 + 1:02d}",
                    "turn_index": turn_index + 1,
                    "wave_index": 1 if turn_index < 5 else 2,
                }
            )
        packet_id = f"ts-ticket04-{state.config.trace_id[3:19]}"
        session_material = sha256_hex(
            canonical_json_bytes(
                {
                    "content_hashes": [content.content_hash for content in contents],
                    "packet_id": packet_id,
                    "rubric_hash": rubric_hash,
                    "sol_hash": sol_hash,
                }
            )
        )
        payload: dict[str, object] = {
            "initial_eval_rubric": {"artifact_hash": rubric_hash, "contract": rubric.payload},
            "output_schema": teacher_output_schema_contract(),
            "packet_id": packet_id,
            "role": "teacher_scorer",
            "schema_version": "teacher-scorer-input/1.0.0",
            "scoring_session": {
                "items_per_turn": 4,
                "max_resident_subthreads": 5,
                "resident_thread_ids": [f"sol-thread-{index:02d}" for index in range(1, 6)],
                "scoring_session_id": f"sol-session-{session_material[:32]}",
                "turn_count": 8,
                "turns": turns,
                "wave_count": 2,
            },
            "seed": state.config.role_seed,
            "sol_inference": {"artifact_hash": sol_hash, "contract": sol.payload},
            "trace": {
                "dataset_version_hash": state.config.dataset_version_hash,
                "prompt": state.training_trace.payload["prompt"],
                "prompt_hash": sha256_hex(cast(str, state.training_trace.payload["prompt"]).encode("utf-8")),
                "trace_id": state.config.trace_id,
                "training_trace_hash": state.training_trace.content_hash,
            },
        }
        payload["allowlisted_fields"] = sorted([*payload, "allowlisted_fields"])
        encoded = canonical_json_bytes(payload).decode("utf-8")
        if "generator" in encoded.lower() or "online_response" in encoded.lower():
            raise FitWorkflowError("TeacherScorer packet exposes generator-only identity or provenance")
        packet = state.store.put("TeacherScorerInputPacket", "1.0.0", payload)
        TeacherScorerIngress.validate_input_packet(state.store.root, packet.content_hash)
        return packet

    @classmethod
    def _build_teacher_label_set(cls, state: _State) -> Artifact:
        if (
            state.fit_set is None
            or state.teacher_packet is None
            or state.teacher_normalized is None
            or state.teacher_audit is None
        ):
            raise FitWorkflowError("TeacherLabelSet inputs are incomplete")
        normalized = state.teacher_normalized.payload
        packet = state.teacher_packet.payload
        rubric = packet.get("initial_eval_rubric")
        sol = packet.get("sol_inference")
        scoring_session = packet.get("scoring_session")
        labels = normalized.get("labels")
        threads = normalized.get("thread_sessions")
        turns = normalized.get("turns")
        if (
            not isinstance(rubric, dict)
            or type(rubric.get("artifact_hash")) is not str
            or not isinstance(sol, dict)
            or type(sol.get("artifact_hash")) is not str
            or not isinstance(scoring_session, dict)
            or not isinstance(labels, list)
            or len(labels) != 32
            or not isinstance(threads, list)
            or len(threads) != 5
            or not isinstance(turns, list)
            or len(turns) != 8
        ):
            raise FitWorkflowError("normalized TeacherScorer output is incomplete")
        input_hashes = [
            cast(str, label.get("input_content_hash"))
            for label in labels
            if isinstance(label, dict) and type(label.get("input_content_hash")) is str
        ]
        if len(input_hashes) != 32 or len(set(input_hashes)) != 32:
            raise FitWorkflowError("TeacherLabelSet input content hashes are incomplete")
        audit = state.teacher_audit.payload
        lineage: dict[str, JsonValue] = {
            "initial_eval_rubric_hash": cast(str, rubric["artifact_hash"]),
            "input_packet_hash": state.teacher_packet.content_hash,
            "normalized_output_hash": state.teacher_normalized.content_hash,
            "raw_output_hash": cast(str, audit["raw_output_hash"]),
            "role_invocation_audit_hash": state.teacher_audit.content_hash,
            "role_session_lineage": cast(str, audit["role_session_lineage"]),
            "sol_inference_config_hash": cast(str, sol["artifact_hash"]),
        }
        identity_payload: dict[str, object] = {
            "dataset_version_hash": state.config.dataset_version_hash,
            "fit_trajectory_set_hash": state.fit_set.content_hash,
            "input_content_hashes": input_hashes,
            "label_count": 32,
            "labels": labels,
            "lineage": lineage,
            "scoring_session": {
                "items_per_turn": scoring_session.get("items_per_turn"),
                "max_resident_subthreads": scoring_session.get("max_resident_subthreads"),
                "scoring_session_id": normalized.get("scoring_session_id"),
                "thread_sessions": threads,
                "turns": turns,
                "wave_count": scoring_session.get("wave_count"),
            },
            "seed": normalized.get("seed"),
            "trace_id": state.config.trace_id,
            "training_trace_hash": state.training_trace.content_hash,
        }
        label_set_id = (
            "tls-"
            + sha256_hex(
                canonical_json_bytes({"domain": "teacher-label-set-identity/1.0.0", "label_set": identity_payload})
            )[:40]
        )
        return state.store.put(
            "TeacherLabelSet",
            "1.0.0",
            {**identity_payload, "teacher_label_set_id": label_set_id},
        )

    @classmethod
    def _build_role_packets(
        cls,
        state: _State,
    ) -> tuple[Artifact, Artifact, Artifact, Artifact, Artifact]:
        if state.teacher_label_set is None or state.teacher_packet is None:
            raise FitWorkflowError("role-isolation packets require a complete TeacherLabelSet")
        base_prompt = state.store.put(
            "BaseJudgePrompt",
            "1.0.0",
            {
                "prompt_id": "fixture-base-judge-prompt-v1",
                "schema_version": "base-judge-prompt/1.0.0",
                "text": (
                    "Evaluate the response only against the supplied task and frozen rubric. "
                    "Return the required structured dimensions, calibrated scalar, failure tags, and evidence."
                ),
            },
        )
        isolation_policy = state.store.put(
            "RoleIsolationPolicy",
            "1.0.0",
            {
                "policy_id": "ticket04-role-isolation-v1",
                "rules": [
                    "prompt_optimizer_may_read_fit_and_normalized_teacher_labels",
                    "prompt_optimizer_may_not_read_role_audit_raw",
                    "alignment_auditor_is_read_only",
                    "alignment_auditor_may_not_modify_prompt_or_policy",
                    "role_sessions_must_be_distinct",
                ],
                "schema_version": "role-isolation-policy/1.0.0",
            },
        )
        teacher_session = cast(dict[str, object], state.teacher_packet.payload["scoring_session"])
        teacher_turns = cast(list[dict[str, object]], teacher_session["turns"])
        label_by_id = {
            cast(str, label["trajectory_id"]): label
            for label in cast(list[dict[str, JsonValue]], state.teacher_label_set.payload["labels"])
        }
        fit_items: list[dict[str, JsonValue]] = []
        for turn in teacher_turns:
            for item in cast(list[dict[str, JsonValue]], turn["items"]):
                trajectory_id = cast(str, item["trajectory_id"])
                label = label_by_id.get(trajectory_id)
                if label is None:
                    raise FitWorkflowError("PromptOptimizer fit item has no normalized Teacher label")
                fit_items.append(
                    {
                        "content_hash": item["content_hash"],
                        "prompt": item["prompt"],
                        "prompt_hash": item["prompt_hash"],
                        "response": item["response"],
                        "response_hash": item["response_hash"],
                        "teacher_label": label,
                        "trajectory_id": trajectory_id,
                    }
                )
        if len(fit_items) != 32:
            raise FitWorkflowError("PromptOptimizer requires exactly 32 fit items")
        optimizer_basis = {
            "base_prompt_hash": base_prompt.content_hash,
            "teacher_label_set_hash": state.teacher_label_set.content_hash,
            "trace_id": state.config.trace_id,
        }
        optimizer_session_id = "optimizer-session-" + sha256_hex(canonical_json_bytes(optimizer_basis))[:24]
        optimizer_payload: dict[str, object] = {
            "base_judge_prompt": {"artifact_hash": base_prompt.content_hash, "contract": base_prompt.payload},
            "fit_items": fit_items,
            "mode": "fit_prompt_optimization",
            "output_schema": prompt_optimizer_output_schema_contract(),
            "packet_id": f"po-ticket04-{state.config.trace_id[3:19]}",
            "role": "prompt_optimizer",
            "role_isolation_policy_hash": isolation_policy.content_hash,
            "schema_version": "prompt-optimizer-input/1.0.0",
            "seed": 4404,
            "session": {
                "optimizer_session_id": optimizer_session_id,
                "permissions": ["read_fit_items", "read_normalized_teacher_labels", "write_candidate_prompt"],
            },
            "teacher_label_set_hash": state.teacher_label_set.content_hash,
            "trace_id": state.config.trace_id,
        }
        optimizer_payload["allowlisted_fields"] = sorted([*optimizer_payload, "allowlisted_fields"])
        optimizer_encoded = canonical_json_bytes(optimizer_payload).decode("utf-8").lower()
        if "audit" in optimizer_encoded or "generator" in optimizer_encoded:
            raise FitWorkflowError("PromptOptimizer packet exposes prohibited role or generator state")
        optimizer_packet = state.store.put("PromptOptimizerInputPacket", "1.0.0", optimizer_payload)
        RoleIsolationIngress.validate_input_packet(
            state.store.root,
            role_type="PromptOptimizer",
            input_packet_hash=optimizer_packet.content_hash,
        )

        auditor_basis = {
            "optimizer_packet_hash": optimizer_packet.content_hash,
            "policy_hash": isolation_policy.content_hash,
            "teacher_packet_hash": state.teacher_packet.content_hash,
        }
        auditor_session_id = "auditor-session-" + sha256_hex(canonical_json_bytes(auditor_basis))[:24]
        auditor_payload: dict[str, object] = {
            "immutable_artifacts": {
                "prompt_optimizer_packet_hash": optimizer_packet.content_hash,
                "role_isolation_policy_hash": isolation_policy.content_hash,
                "teacher_label_set_hash": state.teacher_label_set.content_hash,
                "teacher_packet_hash": state.teacher_packet.content_hash,
            },
            "inspected_packet_contracts": {
                "prompt_optimizer_allowlisted_fields": optimizer_packet.payload["allowlisted_fields"],
                "teacher_scorer_allowlisted_fields": state.teacher_packet.payload["allowlisted_fields"],
            },
            "mode": "role_input_isolation_attestation",
            "output_schema": alignment_auditor_output_schema_contract(),
            "packet_id": f"aa-ticket04-{state.config.trace_id[3:19]}",
            "read_only_authority": {
                "may_modify_policy": False,
                "may_modify_prompt": False,
                "mutable_artifact_hashes": [],
                "scope": "ticket04_role_input_isolation_only_no_alignment_certification",
            },
            "role": "alignment_auditor",
            "schema_version": "alignment-auditor-isolation-input/1.0.0",
            "seed": 4504,
            "session": {"auditor_session_id": auditor_session_id, "permissions": ["read_hashes", "write_attestation"]},
            "trace_id": state.config.trace_id,
        }
        auditor_payload["allowlisted_fields"] = sorted([*auditor_payload, "allowlisted_fields"])
        auditor_encoded = canonical_json_bytes(auditor_payload).decode("utf-8").lower()
        if "generator" in auditor_encoded or auditor_session_id == optimizer_session_id:
            raise FitWorkflowError("AlignmentAuditor packet exposes generator state or reuses a role session")
        auditor_packet = state.store.put("AlignmentAuditorInputPacket", "1.0.0", auditor_payload)
        RoleIsolationIngress.validate_input_packet(
            state.store.root,
            role_type="AlignmentAuditor",
            input_packet_hash=auditor_packet.content_hash,
        )
        decision = state.store.put(
            "DecisionRecord",
            "1.0.0",
            {
                "alignment_auditor_packet_hash": auditor_packet.content_hash,
                "base_judge_prompt_hash": base_prompt.content_hash,
                "decision": "freeze_ticket04_role_isolation_packets",
                "prompt_optimizer_packet_hash": optimizer_packet.content_hash,
                "reason_code": "TICKET04_ROLE_ISOLATION_DEFAULTS_ADOPTED",
                "role_isolation_policy_hash": isolation_policy.content_hash,
                "run_id": state.config.run_id,
            },
        )
        return base_prompt, isolation_policy, optimizer_packet, auditor_packet, decision

    @classmethod
    def _build_role_isolation_attestation(cls, state: _State) -> Artifact:
        if (
            state.base_judge_prompt is None
            or state.role_isolation_policy is None
            or state.prompt_optimizer_packet is None
            or state.alignment_auditor_packet is None
            or state.prompt_optimizer_output is None
            or state.prompt_optimizer_audit is None
            or state.alignment_auditor_output is None
            or state.alignment_auditor_audit is None
        ):
            raise FitWorkflowError("role-isolation attestation inputs are incomplete")
        optimizer_audit = state.prompt_optimizer_audit.payload
        auditor_audit = state.alignment_auditor_audit.payload
        if (
            state.prompt_optimizer_output.payload.get("input_packet_content_hash")
            != state.prompt_optimizer_packet.content_hash
            or state.alignment_auditor_output.payload.get("input_packet_content_hash")
            != state.alignment_auditor_packet.content_hash
            or state.alignment_auditor_output.payload.get("isolation_verdict") != "pass"
            or state.alignment_auditor_output.payload.get("violations") != []
            or optimizer_audit.get("role_type") != "PromptOptimizer"
            or auditor_audit.get("role_type") != "AlignmentAuditor"
            or optimizer_audit.get("role_session_lineage") != "/root/ticket04_prompt_optimizer"
            or auditor_audit.get("role_session_lineage") != "/root/ticket04_alignment_auditor"
        ):
            raise FitWorkflowError("isolated role outputs do not satisfy the attestation contract")
        optimizer_session = state.prompt_optimizer_output.payload.get("optimizer_session_id")
        auditor_session = state.alignment_auditor_output.payload.get("auditor_session_id")
        if (
            type(optimizer_session) is not str
            or type(auditor_session) is not str
            or optimizer_session == auditor_session
        ):
            raise FitWorkflowError("role sessions are missing or not isolated")
        return state.store.put(
            "RoleIsolationAttestation",
            "1.0.0",
            {
                "alignment_auditor": {
                    "input_packet_hash": state.alignment_auditor_packet.content_hash,
                    "normalized_output_hash": state.alignment_auditor_output.content_hash,
                    "raw_output_hash": auditor_audit["raw_output_hash"],
                    "role_invocation_audit_hash": state.alignment_auditor_audit.content_hash,
                    "role_session_lineage": auditor_audit["role_session_lineage"],
                    "seed": auditor_audit["seed"],
                    "session_id": auditor_session,
                },
                "base_judge_prompt_hash": state.base_judge_prompt.content_hash,
                "prompt_optimizer": {
                    "candidate_prompt_hash": sha256_hex(
                        cast(str, state.prompt_optimizer_output.payload["candidate_prompt"]).encode("utf-8")
                    ),
                    "input_packet_hash": state.prompt_optimizer_packet.content_hash,
                    "normalized_output_hash": state.prompt_optimizer_output.content_hash,
                    "raw_output_hash": optimizer_audit["raw_output_hash"],
                    "role_invocation_audit_hash": state.prompt_optimizer_audit.content_hash,
                    "role_session_lineage": optimizer_audit["role_session_lineage"],
                    "seed": optimizer_audit["seed"],
                    "session_id": optimizer_session,
                },
                "role_isolation_policy_hash": state.role_isolation_policy.content_hash,
                "run_id": state.config.run_id,
                "schema_version": "role-isolation-attestation/1.0.0",
                "status": "passed",
                "trace_id": state.config.trace_id,
            },
        )

    @classmethod
    def _load_state(cls, root: Path, run_id: str) -> _State:
        try:
            store = ArtifactStore(root)
            journal = cls._journal(root, store, run_id)
            input_hash = journal.reserved_input_hash()
            workflow_input = store.read(input_hash, expected_schema_name=_INPUT_SCHEMA)
            if workflow_input.schema_version != "1.0.0" or set(workflow_input.payload) != _INPUT_FIELDS:
                raise FitValidationError("fit workflow input fields are invalid")
            config_payload = workflow_input.payload.get("config")
            if not isinstance(config_payload, dict):
                raise FitValidationError("fit workflow config is invalid")
            config = FixtureFitConfig.from_mapping(cast(dict[str, object], config_payload))
            if workflow_input.payload != cls._input_payload(config, None, store=store):
                raise FitValidationError("fit workflow input dependency binding is invalid")
            loaded = load_training_dataset(store, config.dataset_version_hash)
            trace = cls._select_trace(loaded, config.trace_id)
            if workflow_input.payload.get("training_trace_hash") != trace.content_hash:
                raise FitValidationError("fit workflow TrainingTrace binding is invalid")
            events = tuple(journal.events())
            if not events:
                raise FitValidationError("fit workflow has no RUN_STARTED event")
            journal.verify()
            return cls._verify_events(store, journal, workflow_input, config, loaded, trace, events)
        except (FitWorkflowError, StaleFencingEpoch):
            raise
        except Exception as error:
            raise FitWorkflowError("persisted fit workflow state cannot be recertified") from error

    @classmethod
    def _verify_events(
        cls,
        store: ArtifactStore,
        journal: RunJournal,
        workflow_input: Artifact,
        config: FixtureFitConfig,
        loaded: LoadedTrainingDataset,
        trace: Artifact,
        events: tuple[Artifact, ...],
    ) -> _State:
        first = events[0]
        if first.payload.get("event_type") != "RUN_STARTED" or first.payload.get("details") != {
            "input_hash": workflow_input.content_hash,
            "phase": "fit_trajectory_generation",
            "trace_id": config.trace_id,
        }:
            raise FitValidationError("fit RUN_STARTED event is invalid")
        phase = "started"
        plan: Artifact | None = None
        request: Artifact | None = None
        fit_set: Artifact | None = None
        teacher_packet: Artifact | None = None
        checkpoint: Artifact | None = None
        decision_hash: str | None = None
        teacher_normalized: Artifact | None = None
        teacher_audit: Artifact | None = None
        teacher_label_set: Artifact | None = None
        base_judge_prompt: Artifact | None = None
        role_isolation_policy: Artifact | None = None
        prompt_optimizer_packet: Artifact | None = None
        alignment_auditor_packet: Artifact | None = None
        prompt_optimizer_output: Artifact | None = None
        prompt_optimizer_audit: Artifact | None = None
        alignment_auditor_output: Artifact | None = None
        alignment_auditor_audit: Artifact | None = None
        role_isolation_attestation: Artifact | None = None
        retry_count = 0
        for event in events[1:]:
            event_type = event.payload.get("event_type")
            details = event.payload.get("details")
            if not isinstance(details, dict):
                raise FitValidationError("fit event details are invalid")
            if event_type == "GENERATOR_PLAN_FROZEN" and phase == "started":
                if set(details) != {"generator_plan_hash"}:
                    raise FitValidationError("GeneratorPlan event fields are invalid")
                plan = cls._event_ref(store, details, "generator_plan_hash", "GeneratorPlan")
                expected = GeneratorPlan.create(
                    dataset_version_hash=config.dataset_version_hash,
                    training_trace_hash=trace.content_hash,
                    trace_id=config.trace_id,
                    seed=config.generator_seed,
                    include_online_response=config.include_online_response,
                    generator_profile_id=config.generator_profile_id,
                    generator_model_id=config.generator_model_id,
                    inference_config_hash=cast(str, workflow_input.payload["generator_inference_config_hash"]),
                )
                if plan.payload != expected.artifact_payload():
                    raise FitValidationError("persisted GeneratorPlan conflicts with expected input")
                phase = "planned"
            elif event_type == "GENERATION_REQUESTED" and phase == "planned":
                if set(details) != {"generator_request_hash"}:
                    raise FitValidationError("GeneratorRequest event fields are invalid")
                request = cls._event_ref(store, details, "generator_request_hash", "GeneratorRequest")
                if plan is None or request.payload != {
                    "dataset_version_hash": config.dataset_version_hash,
                    "generator_plan_hash": plan.content_hash,
                    "online_response": trace.payload["response"],
                    "output_fault": config.output_fault,
                    "prompt": trace.payload["prompt"],
                    "schema_version": "generator-request/1.0.0",
                    "trace_id": config.trace_id,
                    "training_trace_hash": trace.content_hash,
                }:
                    raise FitValidationError("persisted GeneratorRequest is invalid")
                phase = "requested"
            elif event_type == "GENERATION_RETRY_SCHEDULED" and phase in {"requested", "retrying"}:
                retry_count += 1
                if (
                    request is None
                    or set(details)
                    != {
                        "attempt_count",
                        "failure_code",
                        "generator_attempt_hash",
                        "observation_hash",
                    }
                    or details.get("attempt_count") != retry_count
                    or type(details.get("attempt_count")) is not int
                ):
                    raise FitValidationError("generator retry event is invalid")
                adapter = FixtureFitGenerator(store.root, store, config)
                attempt = adapter.read_attempt(request, retry_count)
                if attempt.content_hash != details.get("generator_attempt_hash"):
                    raise FitValidationError("generator retry attempt lineage is invalid")
                observation = store.read(
                    cast(str, details.get("observation_hash")), expected_schema_name="GeneratorBoundaryObservation"
                )
                if (
                    observation.content_hash != attempt.payload.get("observation_hash")
                    or observation.payload.get("status") != "retryable"
                ):
                    raise FitValidationError("generator retry observation is invalid")
                phase = "retrying"
            elif event_type == "TRAJECTORIES_COMMITTED" and phase in {"requested", "retrying"}:
                attempt_count = details.get("attempt_count")
                if (
                    request is None
                    or type(attempt_count) is not int
                    or attempt_count != retry_count + 1
                    or set(details)
                    != {
                        "attempt_count",
                        "fit_trajectory_set_hash",
                        "generator_attempt_hash",
                        "observation_hash",
                    }
                ):
                    raise FitValidationError("trajectory commit event is invalid")
                adapter = FixtureFitGenerator(store.root, store, config)
                attempts = adapter.verify_attempt_chain(request, cast(int, attempt_count))
                if attempts[-1].content_hash != details.get("generator_attempt_hash"):
                    raise FitValidationError("trajectory commit attempt lineage is invalid")
                observation = store.read(
                    cast(str, details.get("observation_hash")), expected_schema_name="GeneratorBoundaryObservation"
                )
                if (
                    observation.content_hash != attempts[-1].payload.get("observation_hash")
                    or observation.payload.get("status") != "succeeded"
                ):
                    raise FitValidationError("trajectory commit observation is invalid")
                fit_set = cls._event_ref(store, details, "fit_trajectory_set_hash", "FitTrajectorySet")
                phase = "trajectories"
            elif event_type == "TEACHER_PACKET_COMMITTED" and phase == "trajectories":
                if set(details) != {"teacher_packet_hash"}:
                    raise FitValidationError("TeacherScorer packet event fields are invalid")
                teacher_packet = cls._event_ref(store, details, "teacher_packet_hash", "TeacherScorerInputPacket")
                phase = "teacher_packet"
            elif event_type == "DECISION_RECORDED" and phase == "teacher_packet":
                if set(details) != {"decision_record_hash"}:
                    raise FitValidationError("DecisionRecord event fields are invalid")
                decision = cls._event_ref(store, details, "decision_record_hash", "DecisionRecord")
                if (
                    plan is None
                    or teacher_packet is None
                    or decision.payload
                    != {
                        "decision": "freeze_fixture_fit_generation_defaults",
                        "generator_plan_hash": plan.content_hash,
                        "input_hash": workflow_input.content_hash,
                        "reason_code": "TICKET04_RECOMMENDED_DEFAULTS_ADOPTED",
                        "run_id": config.run_id,
                        "teacher_packet_hash": teacher_packet.content_hash,
                    }
                ):
                    raise FitValidationError("fit DecisionRecord is invalid")
                decision_hash = decision.content_hash
                phase = "decision"
            elif event_type == "PHASE_A_CHECKPOINT" and phase == "decision":
                if set(details) != {"phase_a_checkpoint_hash"}:
                    raise FitValidationError("Phase A event fields are invalid")
                checkpoint = cls._event_ref(store, details, "phase_a_checkpoint_hash", "PhaseACheckpoint")
                if (
                    plan is None
                    or fit_set is None
                    or teacher_packet is None
                    or decision_hash is None
                    or checkpoint.payload
                    != {
                        "decision_record_hash": decision_hash,
                        "fit_trajectory_set_hash": fit_set.content_hash,
                        "generator_plan_hash": plan.content_hash,
                        "next_required_role": "TeacherScorer",
                        "run_id": config.run_id,
                        "status": "awaiting_isolated_teacher_output",
                        "teacher_packet_hash": teacher_packet.content_hash,
                        "trace_id": config.trace_id,
                    }
                ):
                    raise FitValidationError("Phase A checkpoint is invalid")
                phase = "checkpoint"
            elif event_type == "TEACHER_OUTPUT_ACCEPTED" and phase == "checkpoint":
                ingress_payload = details.get("teacher_ingress")
                if (
                    not isinstance(ingress_payload, dict)
                    or set(details) != {"teacher_ingress"}
                    or set(ingress_payload)
                    != {
                        "ingress_hash",
                        "input_packet_hash",
                        "normalized_output_hash",
                        "raw_output_hash",
                        "role_invocation_audit_hash",
                    }
                ):
                    raise FitValidationError("TeacherScorer ingress event is invalid")
                try:
                    receipt = TeacherRoleIngressReceipt(
                        ingress_hash=cast(str, ingress_payload.get("ingress_hash")),
                        input_packet_hash=cast(str, ingress_payload.get("input_packet_hash")),
                        normalized_output_hash=cast(str, ingress_payload.get("normalized_output_hash")),
                        raw_output_hash=cast(str, ingress_payload.get("raw_output_hash")),
                        role_invocation_audit_hash=cast(
                            str,
                            ingress_payload.get("role_invocation_audit_hash"),
                        ),
                    )
                    contract = TeacherScorerIngress.load_public_contract(store.root, receipt)
                except (TeacherOutputError, ArtifactCorruption) as error:
                    raise FitValidationError("TeacherScorer ingress cannot be recertified") from error
                if teacher_packet is None or receipt.input_packet_hash != teacher_packet.content_hash:
                    raise FitValidationError("TeacherScorer ingress input packet binding is invalid")
                teacher_normalized = contract.normalized_output
                teacher_audit = contract.role_invocation_audit
                phase = "teacher_output"
            elif event_type == "TEACHER_LABEL_SET_COMMITTED" and phase == "teacher_output":
                if set(details) != {"teacher_label_set_hash"}:
                    raise FitValidationError("TeacherLabelSet event fields are invalid")
                teacher_label_set = cls._event_ref(
                    store,
                    details,
                    "teacher_label_set_hash",
                    "TeacherLabelSet",
                )
                phase = "labels"
            elif event_type == "ROLE_ISOLATION_PACKETS_COMMITTED" and phase == "labels":
                expected_fields = {
                    "alignment_auditor_packet_hash",
                    "base_judge_prompt_hash",
                    "decision_record_hash",
                    "prompt_optimizer_packet_hash",
                    "role_isolation_policy_hash",
                }
                if set(details) != expected_fields:
                    raise FitValidationError("role-isolation packet event fields are invalid")
                base_judge_prompt = cls._event_ref(
                    store,
                    details,
                    "base_judge_prompt_hash",
                    "BaseJudgePrompt",
                )
                role_isolation_policy = cls._event_ref(
                    store,
                    details,
                    "role_isolation_policy_hash",
                    "RoleIsolationPolicy",
                )
                prompt_optimizer_packet = cls._event_ref(
                    store,
                    details,
                    "prompt_optimizer_packet_hash",
                    "PromptOptimizerInputPacket",
                )
                alignment_auditor_packet = cls._event_ref(
                    store,
                    details,
                    "alignment_auditor_packet_hash",
                    "AlignmentAuditorInputPacket",
                )
                decision = cls._event_ref(store, details, "decision_record_hash", "DecisionRecord")
                if decision.payload != {
                    "alignment_auditor_packet_hash": alignment_auditor_packet.content_hash,
                    "base_judge_prompt_hash": base_judge_prompt.content_hash,
                    "decision": "freeze_ticket04_role_isolation_packets",
                    "prompt_optimizer_packet_hash": prompt_optimizer_packet.content_hash,
                    "reason_code": "TICKET04_ROLE_ISOLATION_DEFAULTS_ADOPTED",
                    "role_isolation_policy_hash": role_isolation_policy.content_hash,
                    "run_id": config.run_id,
                }:
                    raise FitValidationError("role-isolation DecisionRecord is invalid")
                phase = "role_packets"
            elif event_type in {
                "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED",
                "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED",
            } and phase in {"role_packets", "role_partial"}:
                role_type = (
                    "PromptOptimizer" if event_type == "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED" else "AlignmentAuditor"
                )
                ingress_payload = details.get("role_ingress")
                if (
                    set(details) != {"role_ingress"}
                    or not isinstance(ingress_payload, dict)
                    or set(ingress_payload)
                    != {
                        "ingress_hash",
                        "input_packet_hash",
                        "normalized_output_hash",
                        "raw_output_hash",
                        "role_invocation_audit_hash",
                        "role_type",
                    }
                    or ingress_payload.get("role_type") != role_type
                ):
                    raise FitValidationError(f"{role_type} ingress event is invalid")
                if (role_type == "PromptOptimizer" and prompt_optimizer_output is not None) or (
                    role_type == "AlignmentAuditor" and alignment_auditor_output is not None
                ):
                    raise FitValidationError(f"duplicate {role_type} ingress event is invalid")
                try:
                    role_receipt = RoleIsolationIngressReceipt(
                        role_type=cast(
                            Literal["PromptOptimizer", "AlignmentAuditor"],
                            ingress_payload.get("role_type"),
                        ),
                        ingress_hash=cast(str, ingress_payload.get("ingress_hash")),
                        input_packet_hash=cast(str, ingress_payload.get("input_packet_hash")),
                        normalized_output_hash=cast(str, ingress_payload.get("normalized_output_hash")),
                        raw_output_hash=cast(str, ingress_payload.get("raw_output_hash")),
                        role_invocation_audit_hash=cast(
                            str,
                            ingress_payload.get("role_invocation_audit_hash"),
                        ),
                    )
                    role_contract = RoleIsolationIngress.load_public_contract(store.root, role_receipt)
                except (RoleIsolationOutputError, ArtifactCorruption) as error:
                    raise FitValidationError(f"{role_type} ingress cannot be recertified") from error
                expected_role_packet = (
                    prompt_optimizer_packet if role_type == "PromptOptimizer" else alignment_auditor_packet
                )
                if (
                    expected_role_packet is None
                    or role_receipt.input_packet_hash != expected_role_packet.content_hash
                    or role_contract.role_type != role_type
                ):
                    raise FitValidationError(f"{role_type} input packet binding is invalid")
                if role_type == "PromptOptimizer":
                    prompt_optimizer_output = role_contract.normalized_output
                    prompt_optimizer_audit = role_contract.role_invocation_audit
                else:
                    alignment_auditor_output = role_contract.normalized_output
                    alignment_auditor_audit = role_contract.role_invocation_audit
                phase = (
                    "role_outputs"
                    if prompt_optimizer_output is not None and alignment_auditor_output is not None
                    else "role_partial"
                )
            elif event_type == "ROLE_ISOLATION_ATTESTATION_COMMITTED" and phase == "role_outputs":
                if set(details) != {"role_isolation_attestation_hash"}:
                    raise FitValidationError("role-isolation attestation event fields are invalid")
                role_isolation_attestation = cls._event_ref(
                    store,
                    details,
                    "role_isolation_attestation_hash",
                    "RoleIsolationAttestation",
                )
                phase = "attested"
            elif event_type == "FIT_FAILURE_RECORDED" and phase in {"requested", "retrying"}:
                if set(details) != {"decision_record_hash", "reason_code"}:
                    raise FitValidationError("fit failure event fields are invalid")
                decision = cls._event_ref(store, details, "decision_record_hash", "DecisionRecord")
                reason = details.get("reason_code")
                if reason not in {
                    "GENERATOR_RETRY_EXHAUSTED",
                    "GENERATOR_PERMANENT_FAILURE",
                    "FIT_TRAJECTORY_BATCH_INVALID",
                } or decision.payload != {
                    "input_hash": workflow_input.content_hash,
                    "reason_code": reason,
                    "run_id": config.run_id,
                    "status": "failed",
                }:
                    raise FitValidationError("fit failure DecisionRecord is invalid")
                phase = "failed"
            elif event_type == "RUN_CLOSED" and phase in {"failed", "attested"}:
                closed = journal.closed()
                if phase == "failed":
                    if closed.payload.get("status") != "failed":
                        raise FitValidationError("failed fit run has an invalid terminal outcome")
                    phase = "closed_failed"
                else:
                    if (
                        closed.payload.get("status") != "succeeded"
                        or closed.payload.get("reason_code") != "TICKET04_COMPLETE"
                    ):
                        raise FitValidationError("Ticket04 run has an invalid terminal outcome")
                    phase = "closed_succeeded"
            else:
                raise FitValidationError("fit application event order is invalid")

        state = _State(
            store=store,
            journal=journal,
            workflow_input=workflow_input,
            config=config,
            loaded_dataset=loaded,
            training_trace=trace,
            events=events,
            plan=plan,
            request=request,
            fit_set=fit_set,
            teacher_packet=teacher_packet,
            checkpoint=checkpoint,
            teacher_normalized=teacher_normalized,
            teacher_audit=teacher_audit,
            teacher_label_set=teacher_label_set,
            base_judge_prompt=base_judge_prompt,
            role_isolation_policy=role_isolation_policy,
            prompt_optimizer_packet=prompt_optimizer_packet,
            alignment_auditor_packet=alignment_auditor_packet,
            prompt_optimizer_output=prompt_optimizer_output,
            prompt_optimizer_audit=prompt_optimizer_audit,
            alignment_auditor_output=alignment_auditor_output,
            alignment_auditor_audit=alignment_auditor_audit,
            role_isolation_attestation=role_isolation_attestation,
        )
        if request is not None:
            adapter = FixtureFitGenerator(store.root, store, config)
            committed_count = adapter.committed_attempt_count(request)
            if phase in {"requested", "retrying"}:
                if committed_count not in {retry_count, retry_count + 1}:
                    raise FitValidationError("generator has more than one unjournaled crash-recovery attempt")
            elif phase in {"failed", "closed_failed"}:
                if committed_count != retry_count + 1:
                    raise FitValidationError("failed generator attempt total is invalid")
                adapter.verify_attempt_chain(request, committed_count)
            elif phase in {
                "trajectories",
                "teacher_packet",
                "decision",
                "checkpoint",
                "teacher_output",
                "labels",
                "closed_succeeded",
                "role_packets",
                "role_partial",
                "role_outputs",
                "attested",
            }:
                if committed_count != retry_count + 1:
                    raise FitValidationError("successful generator attempt total is invalid")
                adapter.verify_attempt_chain(request, committed_count)
        if fit_set is not None:
            cls._validate_fit_set(state, fit_set)
        if teacher_packet is not None:
            expected_packet = cls._build_teacher_packet(state)
            if (
                expected_packet.content_hash != teacher_packet.content_hash
                or expected_packet.payload != teacher_packet.payload
            ):
                raise FitValidationError("TeacherScorer packet cannot be reproduced")
        if teacher_label_set is not None:
            expected_label_set = cls._build_teacher_label_set(state)
            if (
                expected_label_set.content_hash != teacher_label_set.content_hash
                or expected_label_set.payload != teacher_label_set.payload
            ):
                raise FitValidationError("TeacherLabelSet cannot be reproduced")
        if prompt_optimizer_packet is not None or alignment_auditor_packet is not None:
            expected_role_packets = cls._build_role_packets(state)
            if (
                base_judge_prompt is None
                or role_isolation_policy is None
                or prompt_optimizer_packet is None
                or alignment_auditor_packet is None
                or base_judge_prompt.content_hash != expected_role_packets[0].content_hash
                or role_isolation_policy.content_hash != expected_role_packets[1].content_hash
                or prompt_optimizer_packet.content_hash != expected_role_packets[2].content_hash
                or alignment_auditor_packet.content_hash != expected_role_packets[3].content_hash
            ):
                raise FitValidationError("role-isolation packets cannot be reproduced")
        if role_isolation_attestation is not None:
            expected_attestation = cls._build_role_isolation_attestation(state)
            if (
                expected_attestation.content_hash != role_isolation_attestation.content_hash
                or expected_attestation.payload != role_isolation_attestation.payload
            ):
                raise FitValidationError("role-isolation attestation cannot be reproduced")
        return state

    @classmethod
    def _validate_fit_set(cls, state: _State, fit_set: Artifact) -> tuple[Artifact, ...]:
        payload = fit_set.payload
        refs = payload.get("trajectory_refs")
        if (
            fit_set.schema_version != "1.0.0"
            or set(payload)
            != {
                "dataset_version_hash",
                "generator_plan_hash",
                "split",
                "trace_id",
                "training_trace_hash",
                "trajectory_count",
                "trajectory_refs",
                "trajectory_set_hash",
            }
            or payload.get("dataset_version_hash") != state.config.dataset_version_hash
            or state.plan is None
            or payload.get("generator_plan_hash") != state.plan.content_hash
            or payload.get("split") != "fit"
            or payload.get("trace_id") != state.config.trace_id
            or payload.get("training_trace_hash") != state.training_trace.content_hash
            or type(payload.get("trajectory_count")) is not int
            or payload.get("trajectory_count") != 32
            or not isinstance(refs, list)
            or len(refs) != 32
            or payload.get("trajectory_set_hash")
            != sha256_hex(canonical_json_bytes({"domain": "fit-trajectory-set/1.0.0", "trajectory_refs": refs}))
        ):
            raise FitValidationError("FitTrajectorySet fields are invalid")
        plan = GeneratorPlan.from_mapping(cast(dict[str, object], state.plan.payload))
        response_hashes: set[str] = set()
        trajectory_ids: set[str] = set()
        contents: list[Artifact] = []
        online_response = state.training_trace.payload["response"]
        for index, (ref, slot) in enumerate(zip(refs, plan.slots, strict=True)):
            if not isinstance(ref, dict) or set(ref) != {"manifest_hash", "trajectory_id"}:
                raise FitValidationError("fit trajectory ref fields are invalid")
            manifest_hash = ref.get("manifest_hash")
            trajectory_id = ref.get("trajectory_id")
            if type(manifest_hash) is not str or type(trajectory_id) is not str:
                raise FitValidationError("fit trajectory ref identity is invalid")
            manifest = state.store.read(cast(str, manifest_hash), expected_schema_name="TrajectoryManifest")
            if manifest.schema_version != "1.0.0" or set(manifest.payload) != _TRAJECTORY_MANIFEST_FIELDS:
                raise FitValidationError("TrajectoryManifest fields are invalid")
            lineage = manifest.payload.get("lineage")
            if (
                not isinstance(lineage, dict)
                or set(lineage)
                != {
                    "generator_attempt_hash",
                    "generator_observation_hash",
                    "generator_plan_hash",
                    "slot_id",
                    "slot_index",
                }
                or lineage.get("generator_plan_hash") != state.plan.content_hash
                or lineage.get("slot_id") != slot.slot_id
                or lineage.get("slot_index") != index
                or type(lineage.get("slot_index")) is not int
            ):
                raise FitValidationError("TrajectoryManifest generator lineage is invalid")
            content_hash = manifest.payload.get("content_hash")
            if type(content_hash) is not str:
                raise FitValidationError("TrajectoryManifest content hash is invalid")
            content = state.store.read(cast(str, content_hash), expected_schema_name="TrajectoryContent")
            if content.schema_version != "1.0.0" or set(content.payload) != _TRAJECTORY_CONTENT_FIELDS:
                raise FitValidationError("TrajectoryContent fields are invalid")
            prompt = content.payload.get("prompt")
            response = content.payload.get("response")
            prompt_hash = content.payload.get("prompt_hash")
            response_hash = content.payload.get("response_hash")
            expected_identity = sha256_hex(
                canonical_json_bytes(
                    {
                        "domain": "fit-trajectory-identity/1.0.0",
                        "prompt_hash": prompt_hash,
                        "response_hash": response_hash,
                        "split": "fit",
                        "trace_id": state.config.trace_id,
                    }
                )
            )
            expected_trajectory_id = f"trj-{expected_identity[:40]}"
            if (
                type(prompt) is not str
                or prompt != state.training_trace.payload["prompt"]
                or type(response) is not str
                or not 24 <= len(cast(str, response)) <= 4096
                or prompt_hash != sha256_hex(cast(str, prompt).encode("utf-8"))
                or response_hash != sha256_hex(cast(str, response).encode("utf-8"))
                or content.payload.get("schema_version") != "trajectory-content/1.0.0"
                or content.payload.get("split") != "fit"
                or content.payload.get("trace_id") != state.config.trace_id
                or content.payload.get("trajectory_id") != expected_trajectory_id
                or trajectory_id != expected_trajectory_id
                or manifest.payload
                != {
                    "content_hash": content.content_hash,
                    "dataset_version_hash": state.config.dataset_version_hash,
                    "lineage": lineage,
                    "origin": slot.origin,
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                    "schema_version": "trajectory-manifest/1.0.0",
                    "split": "fit",
                    "trace_id": state.config.trace_id,
                    "training_trace_hash": state.training_trace.content_hash,
                    "trajectory_id": expected_trajectory_id,
                }
                or (slot.origin == "online_response" and response != online_response)
                or (slot.origin == "fresh_rollout" and response == online_response)
            ):
                raise FitValidationError("Trajectory content/manifest semantic binding is invalid")
            if cast(str, response_hash) in response_hashes or expected_trajectory_id in trajectory_ids:
                raise FitValidationError("fit trajectories are duplicated")
            response_hashes.add(cast(str, response_hash))
            trajectory_ids.add(expected_trajectory_id)
            contents.append(content)
        if len(contents) != 32 or len(response_hashes) != 32 or len(trajectory_ids) != 32:
            raise FitValidationError("fit trajectory set is incomplete")
        return tuple(contents)

    @classmethod
    def _input_payload(
        cls,
        config: FixtureFitConfig,
        trace: Artifact | None,
        *,
        store: ArtifactStore | None = None,
    ) -> dict[str, object]:
        if trace is not None:
            trace_hash = trace.content_hash
        elif store is not None:
            loaded = load_training_dataset(store, config.dataset_version_hash)
            trace_hash = cls._select_trace(loaded, config.trace_id).content_hash
        else:
            raise FitContractError("TrainingTrace is required for fit input identity")
        rubric_hash = cls._artifact_hash("InitialEvalRubric", config.rubric.artifact_payload())
        sol_hash = cls._artifact_hash("SolInferenceConfig", config.sol_inference.artifact_payload())
        generator_hash = cls._artifact_hash("GeneratorInferenceConfig", config.generator_inference_payload)
        return {
            "config": config.artifact_payload(),
            "dataset_version_hash": config.dataset_version_hash,
            "generator_inference_config_hash": generator_hash,
            "initial_eval_rubric_hash": rubric_hash,
            "run_id": config.run_id,
            "schema_version": "fit-trace-workflow-input/1.0.0",
            "sol_inference_config_hash": sol_hash,
            "trace_id": config.trace_id,
            "training_trace_hash": trace_hash,
        }

    @staticmethod
    def _select_trace(loaded: LoadedTrainingDataset, trace_id: str) -> Artifact:
        matches = [trace for trace in loaded.training_traces if trace.payload.get("trace_id") == trace_id]
        if len(matches) != 1:
            raise FitWorkflowError("exactly one verified TrainingTrace must be selected")
        return matches[0]

    @staticmethod
    def _artifact_hash(schema_name: str, payload: dict[str, JsonValue] | dict[str, object]) -> str:
        return sha256_hex(
            canonical_json_bytes({"payload": dict(payload), "schema_name": schema_name, "schema_version": "1.0.0"})
        )

    @staticmethod
    def _event_ref(store: ArtifactStore, details: Mapping[str, object], key: str, schema: str) -> Artifact:
        if type(details.get(key)) is not str:
            raise FitValidationError(f"{schema} event ref is invalid")
        return store.read(cast(str, details[key]), expected_schema_name=schema)

    @staticmethod
    def _terminal(events: tuple[Artifact, ...]) -> bool:
        return bool(events and events[-1].payload.get("event_type") == "RUN_CLOSED")

    @classmethod
    def _snapshot(cls, state: _State) -> FitWorkflowSnapshot:
        return FitWorkflowSnapshot(
            events=state.events,
            generator_plan=state.plan,
            fit_trajectory_set=state.fit_set,
            teacher_packet=state.teacher_packet,
            phase_a_checkpoint=state.checkpoint,
            teacher_label_set=state.teacher_label_set,
            prompt_optimizer_packet=state.prompt_optimizer_packet,
            alignment_auditor_packet=state.alignment_auditor_packet,
            prompt_optimizer_output=state.prompt_optimizer_output,
            alignment_auditor_output=state.alignment_auditor_output,
            role_isolation_attestation=state.role_isolation_attestation,
            terminal=cls._terminal(state.events),
        )
