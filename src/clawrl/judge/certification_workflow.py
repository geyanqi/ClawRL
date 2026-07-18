"""Fenced Ticket 05 Luna@8 certification workflow."""

from __future__ import annotations

import fcntl
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from clawrl.adapters.generators.holdout import FixtureHoldoutGenerator, HoldoutBoundaryError
from clawrl.adapters.scorers.certification_roles import (
    CertificationRole,
    CertificationRoleIngress,
    CertificationRoleIngressReceipt,
    CertificationRoleOutputError,
    aggregate_diagnostic_contract,
    output_schema_contract,
)
from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.judge.certification_models import (
    CertificationContractError,
    FixtureLuna8Config,
    ProductionLuna8Config,
    TraceJudgePrompt,
    derive_fit_calibrated_trace_prompt,
)
from clawrl.judge.fit_workflow import FitTrajectoryWorkflow, FitWorkflowError
from clawrl.judge.semantic_diversity import (
    SemanticDiversityError,
    semantic_diversity_summary,
    semantic_profile_fingerprint,
    visible_semantic_content_fingerprint,
)
from clawrl.training.run_journal import RunIdentityConflict, RunJournal

_HASH = re.compile(r"^[0-9a-f]{64}$")
_INPUT_SCHEMA = "Luna8CertificationInput"


class Luna8WorkflowError(RuntimeError):
    """Luna@8 certification cannot safely advance or recertify."""


class HoldoutSemanticDiversityError(Luna8WorkflowError):
    """A semantic calibration run received a semantically degenerate holdout."""

    def __init__(self, report_hash: str) -> None:
        super().__init__("holdout semantic feature profiles do not satisfy the frozen gate")
        self.report_hash = report_hash


@dataclass(frozen=True, slots=True)
class Luna8WorkflowSnapshot:
    events: tuple[Artifact, ...]
    alignment_policy: Artifact | None
    base_judge_prompt: Artifact | None
    trace_judge_prompt: Artifact | None
    holdout_set: Artifact | None
    holdout_sets: tuple[Artifact, ...]
    teacher_packet: Artifact | None
    student_packet: Artifact | None
    auditor_packet: Artifact | None
    optimizer_packet: Artifact | None
    certification_report: Artifact | None
    exhaustion_report: Artifact | None
    attempt_index: int
    terminal: bool


@dataclass(frozen=True, slots=True)
class _State:
    root: Path
    store: ArtifactStore
    journal: RunJournal
    config: FixtureLuna8Config
    workflow_input: Artifact
    ticket04_terminal_hash: str
    teacher_label_set: Artifact
    fit_set: Artifact
    trace_id: str
    training_trace_hash: str
    events: tuple[Artifact, ...]
    alignment_policy: Artifact | None
    base_prompt: Artifact | None
    trace_prompt: Artifact | None
    holdout_request: Artifact | None
    holdout_sets: tuple[Artifact, ...]
    teacher_packet: Artifact | None
    student_packet: Artifact | None
    teacher_output: Artifact | None
    teacher_audit: Artifact | None
    student_output: Artifact | None
    student_audit: Artifact | None
    auditor_packet: Artifact | None
    auditor_output: Artifact | None
    auditor_audit: Artifact | None
    optimizer_packet: Artifact | None
    optimizer_output: Artifact | None
    optimizer_audit: Artifact | None
    certification_report: Artifact | None
    exhaustion_report: Artifact | None


class Luna8CertificationWorkflow:
    """Advance one certified Ticket04 trace through bounded TRY_8 attempts."""

    @staticmethod
    def production_readiness(root: str | Path, config: ProductionLuna8Config) -> Artifact:
        checks: list[dict[str, str]] = []
        values = (
            ("ALIGNMENT_POLICY_NUMERIC_CONTRACT_UNAVAILABLE", config.alignment_policy_approval_hash),
            ("SOL_MODEL_CONFIGURATION_UNAVAILABLE", config.sol_model_approval_hash),
            ("LUNA_MODEL_CONFIGURATION_UNAVAILABLE", config.luna_model_approval_hash),
            ("HOLDOUT_GENERATOR_CONFIGURATION_UNAVAILABLE", config.holdout_generator_approval_hash),
            ("REWARD_SCHEMA_UNAVAILABLE", config.reward_schema_approval_hash),
            ("SCALARIZER_UNAVAILABLE", config.scalarizer_approval_hash),
            ("RL_ALGORITHM_CONTRACT_UNAVAILABLE", config.algorithm_contract_approval_hash),
            ("BASE_JUDGE_PROMPT_UNAVAILABLE", config.base_prompt_approval_hash),
            ("TRACE_JUDGE_PROMPT_UNAVAILABLE", config.trace_prompt_approval_hash),
            ("PERMANENT_TRACE_GOVERNANCE_APPROVAL_UNAVAILABLE", config.permanent_trace_governance_approval_hash),
        )
        for code, value in values:
            if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
                checks.append({"code": code, "status": "blocked"})
        if not checks:
            checks.append({"code": "PRODUCTION_JUDGE_BOUNDARIES_NOT_CONFIGURED", "status": "blocked"})
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

    @classmethod
    def bootstrap(cls, root: str | Path, config: FixtureLuna8Config, *, epoch: int) -> Luna8WorkflowSnapshot:
        root_path = Path(root)
        store = ArtifactStore(root_path)
        try:
            ticket04 = FitTrajectoryWorkflow.resume(root_path, config.ticket04_run_id, epoch=epoch)
            if not ticket04.terminal or ticket04.teacher_label_set is None:
                raise Luna8WorkflowError("Ticket04 input is not terminal with a TeacherLabelSet")
            if ticket04.teacher_label_set.content_hash != config.teacher_label_set_hash:
                raise Luna8WorkflowError("Ticket04 TeacherLabelSet identity conflicts with config")
            terminal_details = cast(dict[str, object], ticket04.events[-1].payload["details"])
            terminal_hash = cast(str, terminal_details["run_closed_hash"])
            closed = store.read(terminal_hash, expected_schema_name="RunClosed")
            if closed.payload.get("status") != "succeeded" or closed.payload.get("reason_code") != "TICKET04_COMPLETE":
                raise Luna8WorkflowError("Ticket04 terminal outcome is not accepted")
            fit_hash = ticket04.teacher_label_set.payload.get("fit_trajectory_set_hash")
            if type(fit_hash) is not str:
                raise Luna8WorkflowError("TeacherLabelSet has no fit set lineage")
            fit_set = store.read(cast(str, fit_hash), expected_schema_name="FitTrajectorySet")
            algorithm = cls._read_algorithm_contract(store, config.algorithm_contract_hash)
            trace_id = cast(str, ticket04.teacher_label_set.payload.get("trace_id"))
            if config.trace_prompt.trace_id != trace_id:
                raise Luna8WorkflowError("TraceJudgePrompt belongs to another trace")
            payload: dict[str, object] = {
                **config.immutable_input_payload,
                "algorithm_contract_hash": algorithm.content_hash,
                "ticket04_terminal_hash": terminal_hash,
                "fit_trajectory_set_hash": fit_set.content_hash,
                "trace_id": trace_id,
                "training_trace_hash": ticket04.teacher_label_set.payload["training_trace_hash"],
            }
            input_hash = cls._artifact_hash(_INPUT_SCHEMA, payload)
            journal = RunJournal(
                root_path, store, config.run_id, input_schema_name=_INPUT_SCHEMA, input_schema_version="1.0.0"
            )
            journal.reserve_identity(input_hash)
        except (FitWorkflowError, ArtifactCorruption, CertificationContractError, RunIdentityConflict) as error:
            raise Luna8WorkflowError("Luna@8 immutable input cannot be verified") from error
        input_path = store.artifact_dir / f"{input_hash}.json"
        if input_path.exists():
            persisted = store.read(input_hash, expected_schema_name=_INPUT_SCHEMA)
            if persisted.payload != payload:
                raise Luna8WorkflowError("persisted Luna@8 input conflicts with reserved identity")
            if journal.events():
                return cls._snapshot(cls._load_state(root_path, config.run_id))
        workflow_input = store.put(_INPUT_SCHEMA, "1.0.0", payload)
        if workflow_input.content_hash != input_hash:
            raise Luna8WorkflowError("Luna@8 input publication changed identity")
        journal.start_run(
            epoch,
            workflow_input.content_hash,
            {
                "input_hash": workflow_input.content_hash,
                "phase": "luna8_certification",
                "teacher_label_set_hash": config.teacher_label_set_hash,
                "ticket04_terminal_hash": terminal_hash,
                "trace_id": trace_id,
            },
        )
        return cls._snapshot(cls._load_state(root_path, config.run_id))

    @classmethod
    def run_until_role_input(cls, root: str | Path, config: FixtureLuna8Config, *, epoch: int) -> Luna8WorkflowSnapshot:
        snapshot = cls.bootstrap(root, config, epoch=epoch)
        for _ in range(64):
            if (
                snapshot.terminal
                or snapshot.teacher_packet is not None
                or snapshot.auditor_packet is not None
                or snapshot.optimizer_packet is not None
            ):
                return snapshot
            snapshot = cls.resume(root, config.run_id, epoch=epoch)
        raise Luna8WorkflowError("Luna@8 phase exceeded bounded transitions")

    @classmethod
    def submit_role_output(
        cls,
        root: str | Path,
        run_id: str,
        *,
        role_ingress: CertificationRoleIngressReceipt,
        epoch: int,
    ) -> Luna8WorkflowSnapshot:
        state = cls._load_state(Path(root), run_id)
        packet_by_role: dict[CertificationRole, Artifact | None] = {
            "TeacherScorer": state.teacher_packet,
            "StudentJudge": state.student_packet,
            "AlignmentAuditor": state.auditor_packet,
            "PromptOptimizer": state.optimizer_packet,
        }
        expected_packet = packet_by_role[role_ingress.role]
        if expected_packet is None or expected_packet.content_hash != role_ingress.input_packet_hash:
            raise Luna8WorkflowError("role ingress does not match the currently committed packet")
        try:
            contract = CertificationRoleIngress.load_public_contract(state.root, role_ingress)
        except (CertificationRoleOutputError, ArtifactCorruption) as error:
            raise Luna8WorkflowError("certification role ingress cannot be recertified") from error
        event_type = {
            "TeacherScorer": "HOLDOUT_TEACHER_OUTPUT_ACCEPTED",
            "StudentJudge": "HOLDOUT_STUDENT_OUTPUT_ACCEPTED",
            "AlignmentAuditor": "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED",
            "PromptOptimizer": "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED",
        }[role_ingress.role]
        current_attempt = cls._attempt_index(state.events)
        existing = next(
            (
                event
                for event in state.events
                if event.payload.get("event_type") == event_type
                and isinstance(event.payload.get("details"), dict)
                and cast(dict[str, object], event.payload["details"]).get("attempt_index") == current_attempt
            ),
            None,
        )
        details = {"attempt_index": current_attempt, "role_ingress": role_ingress.artifact_payload()}
        if existing is not None:
            if existing.payload.get("details") != details:
                raise Luna8WorkflowError("a different role output is already committed for this attempt")
            return cls._snapshot(state)
        if cls._terminal(state.events):
            raise Luna8WorkflowError("role output cannot follow a terminal outcome")
        if contract.normalized_output.content_hash != role_ingress.normalized_output_hash:
            raise Luna8WorkflowError("role normalized output conflicts with receipt")
        state.journal.claim_epoch(epoch)
        state.journal.append(
            epoch,
            event_type,
            details,
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def close_rejected_semantic_batch(
        cls,
        root: str | Path,
        run_id: str,
        *,
        rejected_role_audit_hash: str,
        epoch: int,
    ) -> Luna8WorkflowSnapshot:
        """Close a semantically degenerate batch after strict scorer ingress rejects it."""

        state = cls._load_state(Path(root), run_id)
        if cls._terminal(state.events) or state.student_packet is None or not state.holdout_sets:
            raise Luna8WorkflowError("semantic batch rejection is not available in the current state")
        if (
            state.student_output is not None
            or state.events[-1].payload.get("event_type") != "SCORING_PACKETS_COMMITTED"
        ):
            raise Luna8WorkflowError("semantic batch rejection must precede any accepted Student output")
        audit = state.store.read(rejected_role_audit_hash, expected_schema_name="RejectedRoleInvocationAudit")
        if (
            audit.payload.get("role") != "StudentJudge"
            or audit.payload.get("input_packet_hash") != state.student_packet.content_hash
            or audit.payload.get("reason_code") != "ROLE_OUTPUT_CONTRACT_REJECTED"
            or audit.payload.get("validation_error")
            != "scoring output must contain nonconstant scores and item-specific evidence"
        ):
            raise Luna8WorkflowError("rejected Student output does not prove semantic degeneracy")
        cls._validate_rejected_role_audit_private_bytes(state.root, audit)
        holdout = state.holdout_sets[-1]
        items = cls._holdout_items(state.store, holdout)
        visible_items = [{"prompt": item.payload["prompt"], "response": item.payload["response"]} for item in items]
        try:
            diversity = semantic_diversity_summary(cast(list[dict[str, object]], visible_items))
        except SemanticDiversityError as error:
            raise Luna8WorkflowError("rejected holdout semantic structure cannot be measured") from error
        if diversity["status"] != "failed":
            raise Luna8WorkflowError("holdout has sufficient visible semantic diversity")
        report = state.store.put(
            "HoldoutSemanticDiversityFailureReport",
            "1.0.0",
            {
                "attempt_index": cls._attempt_index(state.events),
                "diversity_summary": diversity,
                "feature_extractor": "structural-visible-semantic-profile/2.0.0",
                "generator_lineage": holdout.payload["generator_lineage"],
                "holdout_set_hash": holdout.content_hash,
                "reason_code": "HOLDOUT_SEMANTIC_DIVERSITY_INSUFFICIENT",
                "rejected_raw_output_hash": audit.payload["raw_output_hash"],
                "rejected_role_audit_hash": audit.content_hash,
                "status": "failed",
                "student_packet_hash": state.student_packet.content_hash,
            },
        )
        state.journal.claim_epoch(epoch)
        event = state.journal.append(
            epoch,
            "HOLDOUT_SEMANTIC_DIVERSITY_REJECTED",
            {
                "attempt_index": cls._attempt_index(state.events),
                "failure_report_hash": report.content_hash,
            },
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )
        state.journal.close(
            epoch,
            status="failed",
            reason_code="HOLDOUT_SEMANTIC_DIVERSITY_INSUFFICIENT",
            expected_sequence=len(state.events) + 2,
            expected_previous_hash=event.content_hash,
        )
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def supersede_ambiguous_auditor_packet(cls, root: str | Path, run_id: str, *, epoch: int) -> Luna8WorkflowSnapshot:
        """Replace an unconsumed legacy auditor packet with the frozen algorithm contract.

        This recovery transition is intentionally unavailable after any auditor output
        has been accepted. It preserves the old packet as immutable evidence and records
        the reason and exact predecessor/successor identities in the journal.
        """

        state = cls._load_state(Path(root), run_id)
        if state.auditor_packet is None or state.teacher_output is None or state.student_output is None:
            raise Luna8WorkflowError("auditor packet supersession requires complete scorer outputs")
        if state.auditor_output is not None or cls._terminal(state.events):
            raise Luna8WorkflowError("an accepted or terminal auditor packet cannot be superseded")
        if state.auditor_packet.payload.get("diagnostic_contract") == aggregate_diagnostic_contract():
            CertificationRoleIngress.validate_input_packet(
                state.root, "AlignmentAuditor", state.auditor_packet.content_hash
            )
            return cls._snapshot(state)
        if state.events[-1].payload.get("event_type") != "AUDITOR_PACKET_COMMITTED":
            raise Luna8WorkflowError("auditor packet can only be superseded while awaiting its first output")
        replacement = cls._build_auditor_packet(state, cls._attempt_index(state.events))
        decision = state.store.put(
            "DecisionRecord",
            "1.0.0",
            {
                "decision": "supersede_ambiguous_unconsumed_auditor_packet",
                "new_auditor_packet_hash": replacement.content_hash,
                "old_auditor_packet_hash": state.auditor_packet.content_hash,
                "reason_code": "AUDITOR_DIAGNOSTIC_ALGORITHM_WAS_NOT_CLOSED_WORLD",
                "run_id": run_id,
            },
        )
        state.journal.claim_epoch(epoch)
        state.journal.append(
            epoch,
            "AUDITOR_PACKET_COMMITTED",
            {
                "attempt_index": cls._attempt_index(state.events),
                "auditor_packet_hash": replacement.content_hash,
                "decision_record_hash": decision.content_hash,
                "supersedes_auditor_packet_hash": state.auditor_packet.content_hash,
            },
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def supersede_ambiguous_optimizer_packet(
        cls, root: str | Path, run_id: str, *, epoch: int
    ) -> Luna8WorkflowSnapshot:
        """Replace an unconsumed optimizer packet that omitted prompt wire semantics."""

        state = cls._load_state(Path(root), run_id)
        if state.optimizer_packet is None or state.auditor_output is None:
            raise Luna8WorkflowError("optimizer packet supersession requires a failed audited attempt")
        if state.optimizer_output is not None or cls._terminal(state.events):
            raise Luna8WorkflowError("an accepted or terminal optimizer packet cannot be superseded")
        replacement = cls._build_optimizer_packet(state, cls._attempt_index(state.events))
        output_schema_complete = state.optimizer_packet.payload.get("output_schema") == output_schema_contract(
            "PromptOptimizer"
        )
        identity_complete = state.optimizer_packet.payload.get("packet_id") == replacement.payload.get(
            "packet_id"
        ) and state.optimizer_packet.payload.get("session") == replacement.payload.get("session")
        if output_schema_complete and identity_complete:
            CertificationRoleIngress.validate_input_packet(
                state.root, "PromptOptimizer", state.optimizer_packet.content_hash
            )
            return cls._snapshot(state)
        if state.events[-1].payload.get("event_type") != "OPTIMIZER_PACKET_COMMITTED":
            raise Luna8WorkflowError("optimizer packet can only be superseded while awaiting its first output")
        reason_code = (
            "OPTIMIZER_PACKET_SESSION_IDENTITY_WAS_NOT_RUN_BOUND"
            if output_schema_complete
            else "OPTIMIZER_CANDIDATE_PROMPT_WIRE_CONTRACT_WAS_AMBIGUOUS"
        )
        decision = state.store.put(
            "DecisionRecord",
            "1.0.0",
            {
                "decision": "supersede_ambiguous_unconsumed_optimizer_packet",
                "new_optimizer_packet_hash": replacement.content_hash,
                "old_optimizer_packet_hash": state.optimizer_packet.content_hash,
                "reason_code": reason_code,
                "run_id": run_id,
            },
        )
        state.journal.claim_epoch(epoch)
        state.journal.append(
            epoch,
            "OPTIMIZER_PACKET_COMMITTED",
            {
                "attempt_index": cls._attempt_index(state.events),
                "decision_record_hash": decision.content_hash,
                "optimizer_packet_hash": replacement.content_hash,
                "supersedes_optimizer_packet_hash": state.optimizer_packet.content_hash,
            },
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def resume(cls, root: str | Path, run_id: str, *, epoch: int) -> Luna8WorkflowSnapshot:
        state = cls._load_state(Path(root), run_id)
        if cls._terminal(state.events):
            return cls._snapshot(state)
        event_type = cast(str, state.events[-1].payload["event_type"])
        if (
            event_type == "SCORING_PACKETS_COMMITTED"
            or (
                event_type in {"HOLDOUT_TEACHER_OUTPUT_ACCEPTED", "HOLDOUT_STUDENT_OUTPUT_ACCEPTED"}
                and (state.teacher_output is None or state.student_output is None)
            )
            or event_type in {"AUDITOR_PACKET_COMMITTED", "OPTIMIZER_PACKET_COMMITTED"}
        ):
            return cls._snapshot(state)
        state.journal.claim_epoch(epoch)
        attempt_index = cls._attempt_index(state.events)
        if event_type == "RUN_STARTED":
            cls._freeze_foundation(state, epoch)
        elif event_type in {"FOUNDATION_FROZEN", "NEXT_CANDIDATE_FROZEN"}:
            cls._freeze_holdout_plan(state, epoch, attempt_index)
        elif event_type == "HOLDOUT_PLAN_FROZEN":
            cls._request_holdout(state, epoch, attempt_index)
        elif event_type in {"HOLDOUT_REQUESTED", "HOLDOUT_RETRY_SCHEDULED"}:
            cls._execute_holdout(state, epoch, attempt_index)
        elif event_type == "HOLDOUT_COMMITTED":
            teacher, student = cls._build_scoring_packets(state, attempt_index)
            state.journal.append(
                epoch,
                "SCORING_PACKETS_COMMITTED",
                {
                    "attempt_index": attempt_index,
                    "student_packet_hash": student.content_hash,
                    "teacher_packet_hash": teacher.content_hash,
                },
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
        elif event_type in {"HOLDOUT_TEACHER_OUTPUT_ACCEPTED", "HOLDOUT_STUDENT_OUTPUT_ACCEPTED"}:
            if state.teacher_output is None or state.student_output is None:
                return cls._snapshot(state)
            auditor = cls._build_auditor_packet(state, attempt_index)
            state.journal.append(
                epoch,
                "AUDITOR_PACKET_COMMITTED",
                {"attempt_index": attempt_index, "auditor_packet_hash": auditor.content_hash},
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
        elif event_type == "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED":
            if state.auditor_output is None:
                raise Luna8WorkflowError("accepted auditor output is unavailable")
            verdict = state.auditor_output.payload.get("verdict")
            if verdict == "pass":
                report = cls._build_certification_report(state, attempt_index)
                event = state.journal.append(
                    epoch,
                    "LUNA8_CERTIFIED",
                    {"attempt_index": attempt_index, "certification_report_hash": report.content_hash},
                    expected_sequence=len(state.events) + 1,
                    expected_previous_hash=state.events[-1].content_hash,
                )
                state.journal.close(
                    epoch,
                    status="succeeded",
                    reason_code="LUNA8_CERTIFIED",
                    expected_sequence=len(state.events) + 2,
                    expected_previous_hash=event.content_hash,
                )
            elif attempt_index == 3:
                exhausted = cls._build_exhaustion_report(state)
                event = state.journal.append(
                    epoch,
                    "TRY_8_EXHAUSTED",
                    {"attempt_count": 3, "exhaustion_report_hash": exhausted.content_hash},
                    expected_sequence=len(state.events) + 1,
                    expected_previous_hash=state.events[-1].content_hash,
                )
                state.journal.close(
                    epoch,
                    status="succeeded",
                    reason_code="TRY_8_EXHAUSTED",
                    expected_sequence=len(state.events) + 2,
                    expected_previous_hash=event.content_hash,
                )
            elif verdict == "fail":
                optimizer = cls._build_optimizer_packet(state, attempt_index)
                state.journal.append(
                    epoch,
                    "OPTIMIZER_PACKET_COMMITTED",
                    {"attempt_index": attempt_index, "optimizer_packet_hash": optimizer.content_hash},
                    expected_sequence=len(state.events) + 1,
                    expected_previous_hash=state.events[-1].content_hash,
                )
            else:
                raise Luna8WorkflowError("auditor verdict is invalid")
        elif event_type == "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED":
            prompt = cls._freeze_next_candidate(state, attempt_index + 1)
            state.journal.append(
                epoch,
                "NEXT_CANDIDATE_FROZEN",
                {"attempt_index": attempt_index + 1, "trace_judge_prompt_hash": prompt.content_hash},
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
        else:
            raise Luna8WorkflowError(f"Luna@8 event cannot be advanced: {event_type}")
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def _freeze_foundation(cls, state: _State, epoch: int) -> None:
        base = state.store.put("BaseJudgePrompt", "1.0.0", state.config.base_prompt.artifact_payload())
        if state.config.trace_prompt.parent_prompt_hash != base.content_hash:
            raise Luna8WorkflowError("initial TraceJudgePrompt does not bind the BaseJudgePrompt artifact")
        trace = state.store.put("TraceJudgePrompt", "1.0.0", state.config.trace_prompt.artifact_payload())
        derivation = state.store.read(
            cast(str, state.config.trace_prompt_derivation_hash),
            expected_schema_name="TracePromptDerivationRecord",
        )
        policy = state.store.put("AlignmentPolicy", "1.0.0", state.config.policy.artifact_payload())
        cls._validate_trace_derivation(
            state.store,
            state.fit_set,
            state.teacher_label_set,
            base,
            trace,
            derivation,
            target_run_id=state.config.run_id,
            target_alignment_policy_hash=policy.content_hash,
        )
        calibration = cls._build_teacher_calibration_profile(state)
        luna = state.store.put("LunaInferenceConfig", "1.0.0", state.config.luna_inference.artifact_payload())
        reward = state.store.put("RewardSchema", "1.0.0", state.config.reward_schema.artifact_payload())
        scalarizer = state.store.put("Scalarizer", "1.0.0", state.config.scalarizer.artifact_payload())
        holdout_inference = state.store.put(
            "HoldoutGeneratorInferenceConfig", "1.0.0", state.config.holdout_inference_payload
        )
        decision_payload: dict[str, object] = {
            "algorithm_contract_hash": state.config.algorithm_contract_hash,
            "alignment_policy_hash": policy.content_hash,
            "base_judge_prompt_hash": base.content_hash,
            "decision": "freeze_luna8_numeric_policy_and_fixture_contracts_before_holdout",
            "holdout_generator_inference_hash": holdout_inference.content_hash,
            "luna_inference_config_hash": luna.content_hash,
            "reason_code": "TICKET05_RECOMMENDED_DEFAULTS_ADOPTED",
            "reward_schema_hash": reward.content_hash,
            "scalarizer_hash": scalarizer.content_hash,
            "teacher_calibration_profile_hash": calibration.content_hash,
            "trace_judge_prompt_hash": trace.content_hash,
            "trace_prompt_derivation_hash": derivation.content_hash,
        }
        decision = state.store.put(
            "DecisionRecord",
            "1.0.0",
            decision_payload,
        )
        foundation_details: dict[str, object] = {
            "algorithm_contract_hash": state.config.algorithm_contract_hash,
            "alignment_policy_hash": policy.content_hash,
            "base_judge_prompt_hash": base.content_hash,
            "decision_record_hash": decision.content_hash,
            "holdout_generator_inference_hash": holdout_inference.content_hash,
            "luna_inference_config_hash": luna.content_hash,
            "reward_schema_hash": reward.content_hash,
            "scalarizer_hash": scalarizer.content_hash,
            "teacher_calibration_profile_hash": calibration.content_hash,
            "trace_judge_prompt_hash": trace.content_hash,
            "trace_prompt_derivation_hash": derivation.content_hash,
        }
        state.journal.append(
            epoch,
            "FOUNDATION_FROZEN",
            foundation_details,
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )

    @classmethod
    def _validate_trace_derivation(
        cls,
        store: ArtifactStore,
        fit_set: Artifact,
        teacher_label_set: Artifact,
        base: Artifact,
        trace: Artifact,
        derivation: Artifact,
        *,
        target_run_id: str,
        target_alignment_policy_hash: str,
    ) -> None:
        common = {
            "fit_trajectory_set_hash",
            "holdout_refs",
            "schema_version",
            "source_base_judge_prompt_hash",
            "source_prompt_optimizer_output_hash",
            "source_scope",
            "teacher_label_set_hash",
            "trace_judge_prompt_hash",
            "transformation",
        }
        transformation = derivation.payload.get("transformation")
        if (
            derivation.payload.get("holdout_refs") != []
            or derivation.payload.get("trace_judge_prompt_hash") != trace.content_hash
            or derivation.payload.get("source_base_judge_prompt_hash") != base.content_hash
            or derivation.payload.get("teacher_label_set_hash") != teacher_label_set.content_hash
            or derivation.payload.get("fit_trajectory_set_hash") != fit_set.content_hash
            or type(derivation.payload.get("source_prompt_optimizer_output_hash")) is not str
        ):
            raise Luna8WorkflowError("TraceJudgePrompt derivation is not holdout-blind")
        if transformation == "append_fit_aggregate_calibration_to_isolated_optimizer_candidate":
            if (
                derivation.payload.get("schema_version") != "trace-prompt-derivation/1.0.0"
                or derivation.payload.get("source_scope") != "ticket04_fit_only"
            ):
                raise Luna8WorkflowError("legacy fit aggregate prompt derivation scope is invalid")
            source_optimizer = store.read(
                cast(str, derivation.payload["source_prompt_optimizer_output_hash"]),
                expected_schema_name="NormalizedPromptOptimizerOutput",
            )
            source_candidate = source_optimizer.payload.get("candidate_prompt")
            source_diagnostics = source_optimizer.payload.get("aggregate_diagnostics")
            if (
                set(derivation.payload) != common | {"source_fit_aggregate_hash"}
                or type(source_candidate) is not str
                or not isinstance(source_diagnostics, dict)
                or derivation.payload.get("source_fit_aggregate_hash")
                != sha256_hex(canonical_json_bytes(cast(dict[str, JsonValue], source_diagnostics)))
                or trace.payload.get("text")
                != derive_fit_calibrated_trace_prompt(cast(str, source_candidate), source_diagnostics)
            ):
                raise Luna8WorkflowError("legacy fit aggregate prompt derivation cannot be reproduced")
            return
        if transformation == "reuse_prior_aggregate_only_optimizer_candidate":
            cls._validate_prior_optimizer_derivation(
                store,
                base,
                trace,
                derivation,
                common,
                target_run_id=target_run_id,
                target_alignment_policy_hash=target_alignment_policy_hash,
            )
            return
        raise Luna8WorkflowError("TraceJudgePrompt derivation transformation is unsupported")

    @classmethod
    def _validate_prior_optimizer_derivation(
        cls,
        store: ArtifactStore,
        base: Artifact,
        trace: Artifact,
        derivation: Artifact,
        common: set[str],
        *,
        target_run_id: str,
        target_alignment_policy_hash: str,
    ) -> None:
        provenance_fields = {
            "source_attempt_index",
            "source_optimizer_acceptance_event_hash",
            "source_optimizer_packet_hash",
            "source_optimizer_role_audit_hash",
            "source_alignment_policy_hash",
            "source_run_id",
        }
        payload = derivation.payload
        if (
            set(payload) != common | provenance_fields
            or payload.get("schema_version") != "trace-prompt-derivation/1.1.0"
            or payload.get("source_scope") != "prior_certification_aggregate_only"
            or type(payload.get("source_run_id")) is not str
            or payload.get("source_run_id") == target_run_id
            or payload.get("source_attempt_index") != 1
            or payload.get("source_alignment_policy_hash") != target_alignment_policy_hash
            or trace.payload.get("candidate_index") != 1
        ):
            raise Luna8WorkflowError("prior optimizer prompt provenance is invalid")
        event = store.read(
            cast(str, payload["source_optimizer_acceptance_event_hash"]),
            expected_schema_name="RunEvent",
        )
        details = event.payload.get("details")
        receipt = details.get("role_ingress") if isinstance(details, dict) else None
        if (
            event.payload.get("event_type") != "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED"
            or event.payload.get("run_id") != payload["source_run_id"]
            or not isinstance(details, dict)
            or details.get("attempt_index") != 1
            or not isinstance(receipt, dict)
            or receipt.get("role") != "PromptOptimizer"
            or receipt.get("input_packet_hash") != payload["source_optimizer_packet_hash"]
            or receipt.get("normalized_output_hash") != payload["source_prompt_optimizer_output_hash"]
            or receipt.get("role_invocation_audit_hash") != payload["source_optimizer_role_audit_hash"]
        ):
            raise Luna8WorkflowError("prior optimizer acceptance event does not bind the declared provenance")
        previous_hash = event.payload.get("previous_event_hash")
        previous = store.read(cast(str, previous_hash), expected_schema_name="RunEvent")
        previous_details = previous.payload.get("details")
        if (
            previous.payload.get("event_type") != "OPTIMIZER_PACKET_COMMITTED"
            or previous.payload.get("run_id") != payload["source_run_id"]
            or not isinstance(previous_details, dict)
            or previous_details.get("attempt_index") != 1
            or previous_details.get("optimizer_packet_hash") != payload["source_optimizer_packet_hash"]
        ):
            raise Luna8WorkflowError("prior optimizer packet commit is not the accepted event predecessor")
        cursor = previous
        source_foundation: Artifact | None = None
        for _ in range(64):
            if cursor.payload.get("event_type") == "FOUNDATION_FROZEN":
                source_foundation = cursor
                break
            cursor_previous_hash = cursor.payload.get("previous_event_hash")
            if type(cursor_previous_hash) is not str:
                break
            cursor = store.read(cast(str, cursor_previous_hash), expected_schema_name="RunEvent")
            if cursor.payload.get("run_id") != payload["source_run_id"]:
                raise Luna8WorkflowError("prior optimizer event chain crosses a run identity")
        source_foundation_details = source_foundation.payload.get("details") if source_foundation is not None else None
        if (
            not isinstance(source_foundation_details, dict)
            or source_foundation_details.get("alignment_policy_hash") != payload["source_alignment_policy_hash"]
        ):
            raise Luna8WorkflowError("prior optimizer policy is not byte-identical to the target policy")
        packet = store.read(
            cast(str, payload["source_optimizer_packet_hash"]),
            expected_schema_name="CertificationOptimizerInputPacket",
        )
        output = store.read(
            cast(str, payload["source_prompt_optimizer_output_hash"]),
            expected_schema_name="NormalizedCertificationPromptOptimizerOutput",
        )
        audit = store.read(
            cast(str, payload["source_optimizer_role_audit_hash"]),
            expected_schema_name="RoleInvocationAudit",
        )
        forbidden = {
            "holdout_refs",
            "holdout_set_hash",
            "item_hash",
            "item_hashes",
            "student_label",
            "student_labels",
            "teacher_label",
            "teacher_labels",
        }
        if (
            cls._nested_field_names(packet.payload) & forbidden
            or packet.payload.get("attempt_index") != 1
            or packet.payload.get("next_candidate_index") != 2
            or packet.payload.get("verdict") != "fail"
            or not isinstance(packet.payload.get("aggregate_diagnostics"), dict)
            or cast(dict[str, object], packet.payload.get("base_judge_prompt", {})).get("artifact_hash")
            != base.content_hash
            or output.payload.get("input_packet_content_hash") != packet.content_hash
            or output.payload.get("candidate_prompt") != trace.payload.get("text")
            or audit.payload.get("input_packet_hash") != packet.content_hash
            or audit.payload.get("normalized_output_hash") != output.content_hash
            or audit.payload.get("raw_output_hash") != receipt.get("raw_output_hash")
            or audit.payload.get("role_type") != "PromptOptimizer"
        ):
            raise Luna8WorkflowError("prior optimizer provenance cannot be replayed without item-level data")

    @classmethod
    def _nested_field_names(cls, value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {nested for child in value.values() for nested in cls._nested_field_names(child)}
        if isinstance(value, list):
            return {nested for child in value for nested in cls._nested_field_names(child)}
        return set()

    @classmethod
    def _build_teacher_calibration_profile(cls, state: _State) -> Artifact:
        return state.store.put(
            "TeacherCalibrationProfile",
            "1.0.0",
            cls._teacher_calibration_profile_payload(state.teacher_label_set, state.fit_set),
        )

    @staticmethod
    def _teacher_calibration_profile_payload(teacher_label_set: Artifact, fit_set: Artifact) -> dict[str, JsonValue]:
        labels = teacher_label_set.payload.get("labels")
        names = ("correctness", "reasoning_quality", "task_completion", "tool_discipline")
        if not isinstance(labels, list) or len(labels) != 32:
            raise Luna8WorkflowError("fit calibration requires exactly 32 TeacherLabelSet labels")
        values: dict[str, list[int]] = {name: [] for name in names}
        scalars: list[int] = []
        for label in labels:
            if not isinstance(label, dict) or not isinstance(label.get("dimension_scores"), dict):
                raise Luna8WorkflowError("fit calibration label is invalid")
            dimensions = cast(dict[str, object], label["dimension_scores"])
            scalar = label.get("scalar_micros")
            if (
                set(dimensions) != set(names)
                or any(
                    type(dimensions[name]) is not int or not 0 <= cast(int, dimensions[name]) <= 100 for name in names
                )
                or type(scalar) is not int
                or not 0 <= cast(int, scalar) <= 100_000_000
            ):
                raise Luna8WorkflowError("fit calibration scores are invalid")
            for name in names:
                values[name].append(cast(int, dimensions[name]))
            scalars.append(cast(int, scalar))
        payload: dict[str, JsonValue] = {
            "dimension_statistics": {
                name: {
                    "maximum_points": max(values[name]),
                    "mean_points_micros": sum(values[name]) * 1_000_000 // 32,
                    "minimum_points": min(values[name]),
                }
                for name in names
            },
            "fit_item_count": 32,
            "fit_trajectory_set_hash": fit_set.content_hash,
            "guidance": (
                "Use these fit-only aggregates as numeric anchors for semantically similar visible responses; "
                "score new evidence independently and never inspect holdout history or student output."
            ),
            "scalar_statistics": {
                "maximum_micros": max(scalars),
                "mean_micros": sum(scalars) // 32,
                "minimum_micros": min(scalars),
            },
            "schema_version": "teacher-calibration-profile/1.0.0",
            "teacher_label_set_hash": teacher_label_set.content_hash,
        }
        return payload

    @classmethod
    def _freeze_holdout_plan(cls, state: _State, epoch: int, attempt_index: int) -> None:
        if state.alignment_policy is None or state.trace_prompt is None:
            raise Luna8WorkflowError("policy and candidate prompt must precede holdout plan")
        history = cls._historical_holdout_set_hashes(state)
        attempt_identity = sha256_hex(
            canonical_json_bytes(
                {
                    "attempt_index": attempt_index,
                    "domain": "luna8-alignment-attempt/1.0.0",
                    "input_hash": state.workflow_input.content_hash,
                    "prior_holdouts": history,
                    "trace_prompt_hash": state.trace_prompt.content_hash,
                }
            )
        )
        attempt_id = f"la-{attempt_identity[:40]}"
        plan = state.store.put(
            "HoldoutPlan",
            "1.0.0",
            {
                "alignment_policy_hash": state.alignment_policy.content_hash,
                "attempt_id": attempt_id,
                "attempt_index": attempt_index,
                "fit_trajectory_set_hash": state.fit_set.content_hash,
                "historical_holdout_hashes": history,
                "required_count": 32,
                "seed": cls._attempt_seed(state.config.holdout_seed, attempt_index),
                "trace_id": state.trace_id,
                "trace_judge_prompt_hash": state.trace_prompt.content_hash,
            },
        )
        state.journal.append(
            epoch,
            "HOLDOUT_PLAN_FROZEN",
            {"attempt_id": attempt_id, "attempt_index": attempt_index, "holdout_plan_hash": plan.content_hash},
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )

    @classmethod
    def _request_holdout(cls, state: _State, epoch: int, attempt_index: int) -> None:
        details = cast(dict[str, object], state.events[-1].payload["details"])
        plan_hash = cast(str, details["holdout_plan_hash"])
        plan = state.store.read(plan_hash, expected_schema_name="HoldoutPlan")
        foundation = cls._event_details(state.events, "FOUNDATION_FROZEN")
        fit_hashes = cls._fit_content_hashes(state)
        request = state.store.put(
            "HoldoutGeneratorRequest",
            "1.0.0",
            {
                "attempt_id": plan.payload["attempt_id"],
                "attempt_index": attempt_index,
                "fit_content_hashes": fit_hashes,
                "generator_inference_hash": foundation["holdout_generator_inference_hash"],
                "generator_model_id": state.config.holdout_generator_model_id,
                "generator_profile_id": state.config.holdout_generator_profile_id,
                "holdout_seed": plan.payload["seed"],
                "output_fault": state.config.output_fault,
                "policy_hash": state.alignment_policy.content_hash if state.alignment_policy else None,
                "prompt": cls._task_prompt(state),
                "request_schema_version": "holdout-generator-request/1.0.0",
                "trace_id": state.trace_id,
            },
        )
        state.journal.append(
            epoch,
            "HOLDOUT_REQUESTED",
            {"attempt_index": attempt_index, "holdout_request_hash": request.content_hash},
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )

    @classmethod
    def _execute_holdout(cls, state: _State, epoch: int, attempt_index: int) -> None:
        if state.holdout_request is None:
            raise Luna8WorkflowError("holdout request is unavailable")
        retries = sum(
            event.payload.get("event_type") == "HOLDOUT_RETRY_SCHEDULED"
            and isinstance(event.payload.get("details"), dict)
            and cast(dict[str, object], event.payload["details"]).get("attempt_index") == attempt_index
            for event in state.events
        )
        sequence = retries + 1
        adapter = FixtureHoldoutGenerator(state.root, state.store, state.config)
        try:
            observation = adapter.execute(state.holdout_request, boundary_sequence=sequence)
            attempts = adapter.verify_attempt_chain(state.holdout_request, sequence)
        except (HoldoutBoundaryError, ArtifactCorruption) as error:
            raise Luna8WorkflowError("holdout boundary evidence is invalid") from error
        if observation.payload.get("status") == "retryable":
            if sequence >= len(state.config.fault_schedule):
                cls._close_failed(state, epoch, "HOLDOUT_RETRY_EXHAUSTED")
                return
            state.journal.append(
                epoch,
                "HOLDOUT_RETRY_SCHEDULED",
                {
                    "attempt_index": attempt_index,
                    "boundary_sequence": sequence,
                    "failure_code": observation.payload["failure_code"],
                    "generator_attempt_hash": attempts[-1].content_hash,
                    "observation_hash": observation.content_hash,
                },
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
            return
        if observation.payload.get("status") == "failed":
            cls._close_failed(state, epoch, "HOLDOUT_GENERATOR_PERMANENT_FAILURE")
            return
        try:
            raw = adapter.read_raw_batch(observation, state.holdout_request)
            holdout, ledger_root = cls._commit_holdout(state, raw, observation, attempts[-1], attempt_index)
        except HoldoutSemanticDiversityError as error:
            cls._close_failed(
                state,
                epoch,
                "HOLDOUT_SEMANTIC_DIVERSITY_INSUFFICIENT",
                {"semantic_diversity_report_hash": error.report_hash},
            )
            return
        except (HoldoutBoundaryError, ArtifactCorruption, Luna8WorkflowError, OSError, ValueError, TypeError):
            cls._close_failed(state, epoch, "HOLDOUT_BATCH_INVALID")
            return
        state.journal.append(
            epoch,
            "HOLDOUT_COMMITTED",
            {
                "attempt_index": attempt_index,
                "generator_attempt_hash": attempts[-1].content_hash,
                "holdout_set_hash": holdout.content_hash,
                "holdout_ledger_root_hash": ledger_root.content_hash,
                "observation_hash": observation.content_hash,
            },
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )

    @classmethod
    def _commit_holdout(
        cls, state: _State, raw: Artifact, observation: Artifact, generator_attempt: Artifact, attempt_index: int
    ) -> tuple[Artifact, Artifact]:
        if state.holdout_request is None:
            raise Luna8WorkflowError("holdout request disappeared before commit")
        request = state.holdout_request
        plan_event = cls._event_details(state.events, "HOLDOUT_PLAN_FROZEN")
        plan = state.store.read(cast(str, plan_event["holdout_plan_hash"]), expected_schema_name="HoldoutPlan")
        plan_history = plan.payload.get("historical_holdout_hashes")
        if not isinstance(plan_history, list) or any(
            type(value) is not str or _HASH.fullmatch(cast(str, value)) is None for value in plan_history
        ):
            raise Luna8WorkflowError("holdout plan history is invalid")
        items = raw.payload.get("items")
        if not isinstance(items, list) or len(items) != 32:
            raise Luna8WorkflowError("holdout batch must contain exactly 32 items")
        semantic_diversity_report = cls._semantic_diversity_report(
            state, raw, generator_attempt, cast(list[object], items)
        )
        if semantic_diversity_report is not None and semantic_diversity_report.payload.get("status") != "passed":
            raise HoldoutSemanticDiversityError(semantic_diversity_report.content_hash)
        fit_response_hashes = set(cls._fit_content_hashes(state))
        refs: list[dict[str, JsonValue]] = []
        response_hashes: set[str] = set()
        semantic_item_hashes: list[str] = []
        visible_content_hashes: list[str] = []
        for index, value in enumerate(items, start=1):
            if not isinstance(value, dict) or set(value) != {"item_index", "prompt", "response"}:
                raise Luna8WorkflowError("holdout item fields are invalid")
            prompt = value.get("prompt")
            response = value.get("response")
            if (
                value.get("item_index") != index
                or type(prompt) is not str
                or type(response) is not str
                or not 32 <= len(cast(str, response)) <= 4096
            ):
                raise Luna8WorkflowError("holdout item does not satisfy its ordered slot")
            response_hash = sha256_hex(cast(str, response).encode("utf-8"))
            if response_hash in response_hashes:
                raise Luna8WorkflowError("duplicate holdout response cannot pad cardinality")
            if response_hash in fit_response_hashes:
                raise Luna8WorkflowError("holdout response overlaps a fit trajectory")
            response_hashes.add(response_hash)
            item_identity = sha256_hex(
                canonical_json_bytes(
                    {
                        "domain": "holdout-item-identity/1.0.0",
                        "prompt_hash": sha256_hex(cast(str, prompt).encode("utf-8")),
                        "response_hash": response_hash,
                        "trace_id": state.trace_id,
                    }
                )
            )
            try:
                semantic_item_hashes.append(semantic_profile_fingerprint(cast(str, prompt), cast(str, response)))
                visible_content_hashes.append(
                    visible_semantic_content_fingerprint(cast(str, prompt), cast(str, response))
                )
            except SemanticDiversityError as error:
                raise Luna8WorkflowError("holdout semantic identity cannot be derived") from error
            content = state.store.put(
                "HoldoutItemContent",
                "1.0.0",
                {
                    "item_id": f"hoi-{item_identity[:40]}",
                    "prompt": prompt,
                    "prompt_hash": sha256_hex(cast(str, prompt).encode("utf-8")),
                    "response": response,
                    "response_hash": response_hash,
                    "schema_version": "holdout-item-content/1.0.0",
                    "split": "holdout",
                    "trace_id": state.trace_id,
                },
            )
            refs.append({"item_hash": content.content_hash, "item_id": content.payload["item_id"]})
        item_hashes = [cast(str, ref["item_hash"]) for ref in refs]
        if len(set(item_hashes)) != 32:
            raise Luna8WorkflowError("holdout item content is not unique")
        with cls._holdout_ledger_lock(state.root, state.trace_id) as ledger:
            previous_root = cls._read_holdout_ledger_root(
                state.root,
                state.store,
                state.trace_id,
                allow_legacy=state.config.holdout_generator_model_id != "fixture-semantic-scenarios-v2",
            )
            historical_entries = (
                cast(list[dict[str, JsonValue]], previous_root.payload["entries"]) if previous_root is not None else []
            )
            historical = {
                fingerprint
                for entry in historical_entries
                for fingerprint in cast(list[str], entry["visible_content_fingerprints"])
            }
            overlap = historical.intersection(visible_content_hashes)
            if overlap:
                raise Luna8WorkflowError("holdout item overlaps a historical audited holdout")
            for visible_hash, item_hash in zip(visible_content_hashes, item_hashes, strict=True):
                ArtifactStore._publish(ledger / f"{visible_hash}.ref", f"{item_hash}\n".encode("ascii"))
            holdout = state.store.put(
                "HoldoutSet",
                "1.0.0",
                {
                    "alignment_policy_hash": state.alignment_policy.content_hash if state.alignment_policy else None,
                    "attempt_id": cast(str, request.payload["attempt_id"]),
                    "attempt_index": attempt_index,
                    "fit_trajectory_set_hash": state.fit_set.content_hash,
                    "generator_lineage": {
                        "generator_attempt_hash": generator_attempt.content_hash,
                        "generator_observation_hash": observation.content_hash,
                        "generator_request_hash": request.content_hash,
                        **(
                            {"semantic_diversity_report_hash": semantic_diversity_report.content_hash}
                            if semantic_diversity_report is not None
                            else {}
                        ),
                    },
                    "historical_holdout_set_hashes": plan_history,
                    "item_count": 32,
                    "item_refs": refs,
                    "split": "holdout",
                    "trace_id": state.trace_id,
                },
            )
            ArtifactStore._publish(
                ledger.parent / f"attempt-{cast(str, request.payload['attempt_id'])}.ref",
                f"{holdout.content_hash}\n".encode("ascii"),
            )
            entry: dict[str, JsonValue] = {
                "attempt_id": request.payload["attempt_id"],
                "holdout_set_hash": holdout.content_hash,
                "item_hashes": cast(list[JsonValue], item_hashes),
                "semantic_fingerprints": cast(list[JsonValue], semantic_item_hashes),
                "visible_content_fingerprints": cast(list[JsonValue], visible_content_hashes),
            }
            ledger_root = state.store.put(
                "HoldoutLedgerRoot",
                "1.0.0",
                {
                    "entries": [*historical_entries, entry],
                    "entry_count": len(historical_entries) + 1,
                    "previous_root_hash": previous_root.content_hash if previous_root is not None else None,
                    "schema_version": "holdout-ledger-root/1.0.0",
                    "trace_id": state.trace_id,
                },
            )
            ArtifactStore._publish(
                ledger.parent / "roots" / f"{len(historical_entries) + 1:08d}.ref",
                f"{ledger_root.content_hash}\n".encode("ascii"),
            )
        return holdout, ledger_root

    @classmethod
    def _semantic_diversity_report(
        cls, state: _State, raw: Artifact, generator_attempt: Artifact, items: list[object]
    ) -> Artifact | None:
        if state.config.holdout_generator_model_id != "fixture-semantic-scenarios-v2":
            return None
        visible_items: list[dict[str, object]] = []
        for item in items:
            if (
                not isinstance(item, dict)
                or type(item.get("prompt")) is not str
                or type(item.get("response")) is not str
            ):
                raise Luna8WorkflowError("semantic diversity input item is invalid")
            visible_items.append({"prompt": item["prompt"], "response": item["response"]})
        try:
            summary = semantic_diversity_summary(visible_items)
        except SemanticDiversityError as error:
            raise Luna8WorkflowError("semantic diversity summary cannot be computed") from error
        return state.store.put(
            "HoldoutSemanticDiversityReport",
            "1.0.0",
            {
                "generator_attempt_hash": generator_attempt.content_hash,
                "input_raw_batch_hash": raw.content_hash,
                "profile_algorithm": "structural-visible-semantic-profile/2.0.0",
                "status": summary["status"],
                "summary": summary,
            },
        )

    @classmethod
    def _build_scoring_packets(cls, state: _State, attempt_index: int) -> tuple[Artifact, Artifact]:
        holdout = state.holdout_sets[-1]
        items = cls._holdout_items(state.store, holdout)
        foundation = cls._event_details(state.events, "FOUNDATION_FROZEN")
        sol_hash = cast(
            str, cast(dict[str, object], state.teacher_label_set.payload["lineage"])["sol_inference_config_hash"]
        )
        rubric_hash = cast(
            str, cast(dict[str, object], state.teacher_label_set.payload["lineage"])["initial_eval_rubric_hash"]
        )
        sol = state.store.read(sol_hash, expected_schema_name="SolInferenceConfig")
        rubric = state.store.read(rubric_hash, expected_schema_name="InitialEvalRubric")
        reward = state.store.read(cast(str, foundation["reward_schema_hash"]), expected_schema_name="RewardSchema")
        scalarizer = state.store.read(cast(str, foundation["scalarizer_hash"]), expected_schema_name="Scalarizer")
        turns = cls._turns(items)
        attempt_id = cast(str, holdout.payload["attempt_id"])
        teacher_session = (
            "ticket05-sol-"
            + sha256_hex(canonical_json_bytes({"attempt": attempt_id, "items": [item.content_hash for item in items]}))[
                :24
            ]
        )
        student_session = (
            "ticket05-luna-"
            + sha256_hex(
                canonical_json_bytes(
                    {"attempt": attempt_id, "prompt": state.trace_prompt.content_hash if state.trace_prompt else ""}
                )
            )[:24]
        )
        common = {
            "aggregation": "calibrated_scalar",
            "attempt_id": attempt_id,
            "attempt_index": attempt_index,
            "holdout_set_hash": holdout.content_hash,
            "items_per_turn": 8,
            "reward_schema": {"artifact_hash": reward.content_hash, "contract": reward.payload},
            "scalarizer": {"artifact_hash": scalarizer.content_hash, "contract": scalarizer.payload},
            "trace_id": state.trace_id,
        }
        teacher_payload: dict[str, object] = {
            **common,
            "initial_eval_rubric": {"artifact_hash": rubric_hash, "contract": rubric.payload},
            "output_schema": output_schema_contract("TeacherScorer"),
            "packet_id": f"t05-teacher-{attempt_id[3:19]}",
            "role": "teacher_scorer",
            "schema_version": "holdout-teacher-input/1.0.0",
            "scoring_session": {"items_per_turn": 8, "session_id": teacher_session, "turn_count": 4, "turns": turns},
            "seed": state.config.role_seed + attempt_index * 10 + 1,
            "sol_inference": {"artifact_hash": sol_hash, "contract": sol.payload},
        }
        teacher_payload["allowlisted_fields"] = sorted([*teacher_payload, "allowlisted_fields"])
        teacher = state.store.put("HoldoutTeacherInputPacket", "1.0.0", teacher_payload)
        student_payload: dict[str, object] = {
            **common,
            "candidate_prompt": {
                "artifact_hash": state.trace_prompt.content_hash if state.trace_prompt else "",
                "contract": state.trace_prompt.payload if state.trace_prompt else {},
            },
            "luna_inference": {
                "artifact_hash": foundation["luna_inference_config_hash"],
                "contract": state.config.luna_inference.artifact_payload(),
            },
            "output_schema": output_schema_contract("StudentJudge"),
            "packet_id": f"t05-student-{attempt_id[3:19]}",
            "role": "student_judge",
            "schema_version": "holdout-student-input/1.0.0",
            "scoring_session": {"items_per_turn": 8, "session_id": student_session, "turn_count": 4, "turns": turns},
            "seed": state.config.role_seed + attempt_index * 10 + 2,
        }
        student_payload["allowlisted_fields"] = sorted([*student_payload, "allowlisted_fields"])
        student = state.store.put("HoldoutStudentInputPacket", "1.0.0", student_payload)
        CertificationRoleIngress.validate_input_packet(state.root, "TeacherScorer", teacher.content_hash)
        CertificationRoleIngress.validate_input_packet(state.root, "StudentJudge", student.content_hash)
        teacher_hashes = cls._packet_item_hashes(teacher)
        student_hashes = cls._packet_item_hashes(student)
        if teacher_hashes != student_hashes or len(teacher_hashes) != 32 or teacher_session == student_session:
            raise Luna8WorkflowError("Sol and Luna scoring packets do not bind the same isolated holdout")
        return teacher, student

    @classmethod
    def _build_auditor_packet(cls, state: _State, attempt_index: int) -> Artifact:
        if None in (
            state.teacher_packet,
            state.student_packet,
            state.teacher_output,
            state.student_output,
            state.alignment_policy,
        ):
            raise Luna8WorkflowError("auditor inputs are incomplete")
        teacher_packet = cast(Artifact, state.teacher_packet)
        student_packet = cast(Artifact, state.student_packet)
        if cls._packet_item_hashes(teacher_packet) != cls._packet_item_hashes(student_packet):
            raise Luna8WorkflowError("auditor cannot compare different holdout item sets")
        teacher_labels = cast(list[dict[str, JsonValue]], cast(Artifact, state.teacher_output).payload["labels"])
        student_labels = cast(list[dict[str, JsonValue]], cast(Artifact, state.student_output).payload["labels"])
        payload: dict[str, object] = {
            "aggregation": "calibrated_scalar",
            "alignment_policy": cast(Artifact, state.alignment_policy).payload,
            "alignment_policy_hash": cast(Artifact, state.alignment_policy).content_hash,
            "attempt_id": state.holdout_sets[-1].payload["attempt_id"],
            "attempt_index": attempt_index,
            "comparison": {
                "item_hashes": cls._packet_item_hashes(teacher_packet),
                "student_labels": student_labels,
                "teacher_labels": teacher_labels,
            },
            "diagnostic_contract": aggregate_diagnostic_contract(),
            "holdout_set_hash": state.holdout_sets[-1].content_hash,
            "output_schema": output_schema_contract("AlignmentAuditor"),
            "packet_id": f"t05-auditor-{cast(str, state.holdout_sets[-1].payload['attempt_id'])[3:19]}",
            "preregistered_diagnostics": cast(
                list[str], output_schema_contract("AlignmentAuditor")["diagnostic_required"]
            ),
            "role": "alignment_auditor",
            "schema_version": "certification-auditor-input/1.0.0",
            "seed": state.config.role_seed + attempt_index * 10 + 3,
            "session": {
                "auditor_session_id": "ticket05-auditor-"
                + sha256_hex(
                    canonical_json_bytes(
                        {
                            "attempt": attempt_index,
                            "teacher": cast(Artifact, state.teacher_output).content_hash,
                            "student": cast(Artifact, state.student_output).content_hash,
                        }
                    )
                )[:24]
            },
        }
        payload["allowlisted_fields"] = sorted([*payload, "allowlisted_fields"])
        packet = state.store.put("CertificationAuditorInputPacket", "1.0.0", payload)
        CertificationRoleIngress.validate_input_packet(state.root, "AlignmentAuditor", packet.content_hash)
        return packet

    @classmethod
    def _build_optimizer_packet(cls, state: _State, attempt_index: int) -> Artifact:
        if state.auditor_output is None or state.base_prompt is None or state.trace_prompt is None:
            raise Luna8WorkflowError("optimizer aggregate inputs are incomplete")
        run_token = sha256_hex(
            canonical_json_bytes(
                {
                    "domain": "certification-optimizer-session/1.0.0",
                    "run_id": state.config.run_id,
                }
            )
        )[:12]
        payload: dict[str, object] = {
            "aggregate_diagnostics": state.auditor_output.payload["aggregate_diagnostics"],
            "attempt_index": attempt_index,
            "base_judge_prompt": {
                "artifact_hash": state.base_prompt.content_hash,
                "contract": state.base_prompt.payload,
            },
            "current_trace_judge_prompt": {
                "artifact_hash": state.trace_prompt.content_hash,
                "contract": state.trace_prompt.payload,
            },
            "next_candidate_index": attempt_index + 1,
            "output_schema": output_schema_contract("PromptOptimizer"),
            "packet_id": f"t05-optimizer-{attempt_index}-{state.trace_id[-8:]}-{run_token}",
            "role": "prompt_optimizer",
            "schema_version": "certification-optimizer-input/1.0.0",
            "seed": state.config.role_seed + attempt_index * 10 + 4,
            "session": {
                "optimizer_session_id": (f"ticket05-optimizer-{state.trace_id[-12:]}-{attempt_index}-{run_token}")
            },
            "verdict": "fail",
        }
        payload["allowlisted_fields"] = sorted([*payload, "allowlisted_fields"])
        packet = state.store.put("CertificationOptimizerInputPacket", "1.0.0", payload)
        CertificationRoleIngress.validate_input_packet(state.root, "PromptOptimizer", packet.content_hash)
        return packet

    @classmethod
    def _freeze_next_candidate(cls, state: _State, candidate_index: int) -> Artifact:
        if state.optimizer_output is None or state.trace_prompt is None:
            raise Luna8WorkflowError("next candidate requires isolated optimizer output")
        prompt = TraceJudgePrompt(
            prompt_id=f"ticket05-trace-prompt-{state.trace_id[-12:]}-{candidate_index}",
            trace_id=state.trace_id,
            candidate_index=candidate_index,
            parent_prompt_hash=state.trace_prompt.content_hash,
            text=cast(str, state.optimizer_output.payload["candidate_prompt"]),
        )
        return state.store.put("TraceJudgePrompt", "1.0.0", prompt.artifact_payload())

    @classmethod
    def _build_certification_report(cls, state: _State, attempt_index: int) -> Artifact:
        if None in (
            state.auditor_output,
            state.auditor_audit,
            state.teacher_audit,
            state.student_audit,
            state.trace_prompt,
            state.base_prompt,
            state.alignment_policy,
        ):
            raise Luna8WorkflowError("certification identity is incomplete")
        foundation = cls._event_details(state.events, "FOUNDATION_FROZEN")
        holdout = state.holdout_sets[-1]
        identity: dict[str, object] = {
            "aggregation": "calibrated_scalar",
            "algorithm_contract_hash": state.config.algorithm_contract_hash,
            "alignment_policy_hash": cast(Artifact, state.alignment_policy).content_hash,
            "attempt_id": holdout.payload["attempt_id"],
            "attempt_index": attempt_index,
            "base_judge_prompt_hash": cast(Artifact, state.base_prompt).content_hash,
            "dataset_version_hash": state.fit_set.payload["dataset_version_hash"],
            "fit_trajectory_set_hash": state.fit_set.content_hash,
            "holdout_set_hash": holdout.content_hash,
            "items_per_turn": 8,
            "luna_inference_config_hash": foundation["luna_inference_config_hash"],
            "reward_schema_hash": foundation["reward_schema_hash"],
            "scalarizer_hash": foundation["scalarizer_hash"],
            "sol_inference_config_hash": cast(dict[str, object], state.teacher_label_set.payload["lineage"])[
                "sol_inference_config_hash"
            ],
            "teacher_label_set_hash": state.teacher_label_set.content_hash,
            "teacher_calibration_profile_hash": foundation["teacher_calibration_profile_hash"],
            "trace_id": state.trace_id,
            "trace_judge_prompt_hash": cast(Artifact, state.trace_prompt).content_hash,
            "trace_prompt_derivation_hash": foundation["trace_prompt_derivation_hash"],
            "training_trace_hash": state.training_trace_hash,
        }
        report_id = (
            "cr-"
            + sha256_hex(canonical_json_bytes({"domain": "luna8-certification-identity/1.0.0", "identity": identity}))[
                :40
            ]
        )
        return state.store.put(
            "CertificationReport",
            "1.0.0",
            {
                **identity,
                "alignment_diagnostics": cast(Artifact, state.auditor_output).payload["aggregate_diagnostics"],
                "certification_report_id": report_id,
                "role_lineage": {
                    "alignment_auditor_audit_hash": cast(Artifact, state.auditor_audit).content_hash,
                    "student_judge_audit_hash": cast(Artifact, state.student_audit).content_hash,
                    "teacher_scorer_audit_hash": cast(Artifact, state.teacher_audit).content_hash,
                },
                "scorer_tier": "luna",
                "status": "certified",
            },
        )

    @classmethod
    def _build_exhaustion_report(cls, state: _State) -> Artifact:
        auditor_events = [
            event for event in state.events if event.payload.get("event_type") == "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED"
        ]
        if (
            len(auditor_events) != 3
            or len(state.holdout_sets) != 3
            or len({value.content_hash for value in state.holdout_sets}) != 3
        ):
            raise Luna8WorkflowError("TRY_8 exhaustion requires exactly three distinct failed audited attempts")
        attempts: list[dict[str, JsonValue]] = []
        for event, holdout in zip(auditor_events, state.holdout_sets, strict=True):
            details = cast(dict[str, object], event.payload["details"])
            receipt = cast(dict[str, object], details["role_ingress"])
            normalized = state.store.read(
                cast(str, receipt["normalized_output_hash"]),
                expected_schema_name="NormalizedCertificationAlignmentAuditorOutput",
            )
            if normalized.payload.get("verdict") != "fail":
                raise Luna8WorkflowError("TRY_8 exhaustion contains a passing attempt")
            attempts.append(
                {
                    "attempt_id": holdout.payload["attempt_id"],
                    "attempt_index": holdout.payload["attempt_index"],
                    "auditor_normalized_output_hash": normalized.content_hash,
                    "holdout_set_hash": holdout.content_hash,
                }
            )
        return state.store.put(
            "Try8ExhaustionReport",
            "1.0.0",
            {
                "alignment_policy_hash": state.alignment_policy.content_hash if state.alignment_policy else None,
                "attempt_count": 3,
                "attempts": attempts,
                "next_state": "TRY_4",
                "reason_code": "TRY_8_EXHAUSTED",
                "status": "exhausted",
                "trace_id": state.trace_id,
            },
        )

    @classmethod
    def _load_state(cls, root: Path, run_id: str) -> _State:
        store = ArtifactStore(root)
        journal = RunJournal(root, store, run_id, input_schema_name=_INPUT_SCHEMA, input_schema_version="1.0.0")
        try:
            journal.verify()
            input_hash = journal.reserved_input_hash()
            workflow_input = store.read(input_hash, expected_schema_name=_INPUT_SCHEMA)
            config_fields = set(FixtureLuna8Config.__dataclass_fields__)  # type: ignore[attr-defined]
            config_payload = {
                key: value
                for key, value in workflow_input.payload.items()
                if key in config_fields
                or key
                in {
                    "base_prompt",
                    "trace_prompt",
                    "policy",
                    "luna_inference",
                    "reward_schema",
                    "scalarizer",
                    "holdout_generator_inference",
                }
            }
            config = FixtureLuna8Config.from_mapping(cast(dict[str, object], config_payload))
            cls._read_algorithm_contract(store, config.algorithm_contract_hash)
            ticket04 = FitTrajectoryWorkflow.resume(root, config.ticket04_run_id, epoch=0)
            if (
                not ticket04.terminal
                or ticket04.teacher_label_set is None
                or ticket04.teacher_label_set.content_hash != config.teacher_label_set_hash
            ):
                raise Luna8WorkflowError("Ticket04 dependency no longer recertifies")
            terminal_hash = cast(
                str, cast(dict[str, object], ticket04.events[-1].payload["details"])["run_closed_hash"]
            )
            if terminal_hash != workflow_input.payload.get("ticket04_terminal_hash"):
                raise Luna8WorkflowError("Ticket04 terminal lineage changed")
            teacher_label_set = ticket04.teacher_label_set
            fit_set = store.read(
                cast(str, workflow_input.payload["fit_trajectory_set_hash"]), expected_schema_name="FitTrajectorySet"
            )
            events = tuple(journal.events())
            if not events:
                raise Luna8WorkflowError("Luna@8 journal has not started")
            cls._validate_event_order(events)
            cls._recertify_all_role_ingresses(store, events)
            by_type: dict[str, list[Artifact]] = {}
            for event in events:
                by_type.setdefault(cast(str, event.payload["event_type"]), []).append(event)
            foundation = cls._optional_event_details(events, "FOUNDATION_FROZEN")
            alignment_policy = cls._read_detail_artifact(store, foundation, "alignment_policy_hash", "AlignmentPolicy")
            base_prompt = cls._read_detail_artifact(store, foundation, "base_judge_prompt_hash", "BaseJudgePrompt")
            trace_prompt = cls._current_trace_prompt(store, events, foundation)
            if foundation is not None and config.schema_version == "fixture-luna8-config/1.1.0":
                if base_prompt is None:
                    raise Luna8WorkflowError("fit-derived foundation has no BaseJudgePrompt")
                initial_trace = cls._read_detail_artifact(
                    store, foundation, "trace_judge_prompt_hash", "TraceJudgePrompt"
                )
                derivation = cls._read_detail_artifact(
                    store, foundation, "trace_prompt_derivation_hash", "TracePromptDerivationRecord"
                )
                calibration = cls._read_detail_artifact(
                    store,
                    foundation,
                    "teacher_calibration_profile_hash",
                    "TeacherCalibrationProfile",
                )
                if initial_trace is None or derivation is None or calibration is None:
                    raise Luna8WorkflowError("fit-derived foundation artifacts are incomplete")
                cls._validate_trace_derivation(
                    store,
                    fit_set,
                    teacher_label_set,
                    base_prompt,
                    initial_trace,
                    derivation,
                    target_run_id=config.run_id,
                    target_alignment_policy_hash=cast(Artifact, alignment_policy).content_hash,
                )
                if derivation.content_hash != config.trace_prompt_derivation_hash or calibration.payload != (
                    cls._teacher_calibration_profile_payload(teacher_label_set, fit_set)
                ):
                    raise Luna8WorkflowError("fit-derived foundation cannot be freshly recertified")
            holdout_sets = tuple(
                store.read(
                    cast(str, cast(dict[str, object], event.payload["details"])["holdout_set_hash"]),
                    expected_schema_name="HoldoutSet",
                )
                for event in by_type.get("HOLDOUT_COMMITTED", [])
            )
            if config.holdout_generator_model_id == "fixture-semantic-scenarios-v2" and holdout_sets:
                with cls._holdout_ledger_lock(root, cast(str, workflow_input.payload["trace_id"])):
                    latest_ledger = cls._read_holdout_ledger_root(
                        root,
                        store,
                        cast(str, workflow_input.payload["trace_id"]),
                        allow_legacy=False,
                    )
                if latest_ledger is None:
                    raise Luna8WorkflowError("committed holdouts have no ledger root")
                latest_entries = cast(list[dict[str, JsonValue]], latest_ledger.payload["entries"])
                for event, holdout in zip(by_type.get("HOLDOUT_COMMITTED", []), holdout_sets, strict=True):
                    details = cast(dict[str, object], event.payload["details"])
                    ledger_hash = details.get("holdout_ledger_root_hash")
                    if type(ledger_hash) is not str:
                        raise Luna8WorkflowError("HOLDOUT_COMMITTED does not bind its ledger root")
                    event_ledger = store.read(cast(str, ledger_hash), expected_schema_name="HoldoutLedgerRoot")
                    event_entries = cast(list[dict[str, JsonValue]], event_ledger.payload.get("entries"))
                    if (
                        not event_entries
                        or event_entries[-1].get("holdout_set_hash") != holdout.content_hash
                        or latest_entries[: len(event_entries)] != event_entries
                    ):
                        raise Luna8WorkflowError("HOLDOUT_COMMITTED ledger root is not in the committed chain")
            current_attempt = cls._attempt_index(events)
            holdout_request = cls._current_artifact(
                store, events, "HOLDOUT_REQUESTED", "holdout_request_hash", "HoldoutGeneratorRequest", current_attempt
            )
            teacher_packet = cls._current_artifact(
                store,
                events,
                "SCORING_PACKETS_COMMITTED",
                "teacher_packet_hash",
                "HoldoutTeacherInputPacket",
                current_attempt,
            )
            student_packet = cls._current_artifact(
                store,
                events,
                "SCORING_PACKETS_COMMITTED",
                "student_packet_hash",
                "HoldoutStudentInputPacket",
                current_attempt,
            )
            semantic_failure = cls._last_detail_artifact(
                store,
                events,
                "HOLDOUT_SEMANTIC_DIVERSITY_REJECTED",
                "failure_report_hash",
                "HoldoutSemanticDiversityFailureReport",
            )
            if semantic_failure is not None:
                audit = store.read(
                    cast(str, semantic_failure.payload["rejected_role_audit_hash"]),
                    expected_schema_name="RejectedRoleInvocationAudit",
                )
                cls._validate_rejected_role_audit_private_bytes(root, audit)
                matching_holdout = next(
                    (
                        holdout
                        for holdout in holdout_sets
                        if holdout.content_hash == semantic_failure.payload.get("holdout_set_hash")
                    ),
                    None,
                )
                if matching_holdout is None or student_packet is None:
                    raise Luna8WorkflowError("semantic diversity failure lineage is incomplete")
                visible_items = [
                    {"prompt": item.payload["prompt"], "response": item.payload["response"]}
                    for item in cls._holdout_items(store, matching_holdout)
                ]
                summary = semantic_diversity_summary(cast(list[dict[str, object]], visible_items))
                if (
                    semantic_failure.payload.get("reason_code") != "HOLDOUT_SEMANTIC_DIVERSITY_INSUFFICIENT"
                    or semantic_failure.payload.get("status") != "failed"
                    or semantic_failure.payload.get("student_packet_hash") != student_packet.content_hash
                    or semantic_failure.payload.get("rejected_raw_output_hash") != audit.payload.get("raw_output_hash")
                    or semantic_failure.payload.get("diversity_summary") != summary
                    or summary.get("status") != "failed"
                ):
                    raise Luna8WorkflowError("semantic diversity failure cannot be freshly recertified")
            teacher_output, teacher_audit = cls._role_artifacts(
                store, events, "HOLDOUT_TEACHER_OUTPUT_ACCEPTED", current_attempt, "TeacherScorer"
            )
            student_output, student_audit = cls._role_artifacts(
                store, events, "HOLDOUT_STUDENT_OUTPUT_ACCEPTED", current_attempt, "StudentJudge"
            )
            auditor_packet = cls._current_artifact(
                store,
                events,
                "AUDITOR_PACKET_COMMITTED",
                "auditor_packet_hash",
                "CertificationAuditorInputPacket",
                current_attempt,
            )
            auditor_output, auditor_audit = cls._role_artifacts(
                store, events, "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED", current_attempt, "AlignmentAuditor"
            )
            optimizer_packet = cls._current_artifact(
                store,
                events,
                "OPTIMIZER_PACKET_COMMITTED",
                "optimizer_packet_hash",
                "CertificationOptimizerInputPacket",
                current_attempt,
            )
            optimizer_output, optimizer_audit = cls._role_artifacts(
                store, events, "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED", current_attempt, "PromptOptimizer"
            )
            certification = cls._last_detail_artifact(
                store, events, "LUNA8_CERTIFIED", "certification_report_hash", "CertificationReport"
            )
            exhaustion = cls._last_detail_artifact(
                store, events, "TRY_8_EXHAUSTED", "exhaustion_report_hash", "Try8ExhaustionReport"
            )
        except (
            ArtifactCorruption,
            CertificationContractError,
            CertificationRoleOutputError,
            FitWorkflowError,
            KeyError,
            SemanticDiversityError,
            TypeError,
            ValueError,
        ) as error:
            raise Luna8WorkflowError("persisted Luna@8 graph cannot be recertified") from error
        return _State(
            root=root,
            store=store,
            journal=journal,
            config=config,
            workflow_input=workflow_input,
            ticket04_terminal_hash=terminal_hash,
            teacher_label_set=teacher_label_set,
            fit_set=fit_set,
            trace_id=cast(str, workflow_input.payload["trace_id"]),
            training_trace_hash=cast(str, workflow_input.payload["training_trace_hash"]),
            events=events,
            alignment_policy=alignment_policy,
            base_prompt=base_prompt,
            trace_prompt=trace_prompt,
            holdout_request=holdout_request,
            holdout_sets=holdout_sets,
            teacher_packet=teacher_packet,
            student_packet=student_packet,
            teacher_output=teacher_output,
            teacher_audit=teacher_audit,
            student_output=student_output,
            student_audit=student_audit,
            auditor_packet=auditor_packet,
            auditor_output=auditor_output,
            auditor_audit=auditor_audit,
            optimizer_packet=optimizer_packet,
            optimizer_output=optimizer_output,
            optimizer_audit=optimizer_audit,
            certification_report=certification,
            exhaustion_report=exhaustion,
        )

    @staticmethod
    def _snapshot(state: _State) -> Luna8WorkflowSnapshot:
        return Luna8WorkflowSnapshot(
            events=state.events,
            alignment_policy=state.alignment_policy,
            base_judge_prompt=state.base_prompt,
            trace_judge_prompt=state.trace_prompt,
            holdout_set=state.holdout_sets[-1] if state.holdout_sets else None,
            holdout_sets=state.holdout_sets,
            teacher_packet=state.teacher_packet,
            student_packet=state.student_packet,
            auditor_packet=state.auditor_packet,
            optimizer_packet=state.optimizer_packet,
            certification_report=state.certification_report,
            exhaustion_report=state.exhaustion_report,
            attempt_index=Luna8CertificationWorkflow._attempt_index(state.events),
            terminal=Luna8CertificationWorkflow._terminal(state.events),
        )

    @staticmethod
    def _fit_content_hashes(state: _State) -> list[str]:
        """Return exact fit response hashes used for boundary and semantic disjointness."""
        refs = state.fit_set.payload.get("trajectory_refs")
        if not isinstance(refs, list) or len(refs) != 32:
            raise Luna8WorkflowError("fit set cardinality is invalid")
        result: list[str] = []
        for ref in refs:
            if not isinstance(ref, dict) or type(ref.get("manifest_hash")) is not str:
                raise Luna8WorkflowError("fit set ref is invalid")
            manifest = state.store.read(cast(str, ref["manifest_hash"]), expected_schema_name="TrajectoryManifest")
            content_hash = manifest.payload.get("content_hash")
            if type(content_hash) is not str:
                raise Luna8WorkflowError("fit manifest content lineage is invalid")
            content = state.store.read(cast(str, content_hash), expected_schema_name="TrajectoryContent")
            response_hash = content.payload.get("response_hash")
            if type(response_hash) is not str or _HASH.fullmatch(cast(str, response_hash)) is None:
                raise Luna8WorkflowError("fit response identity is invalid")
            result.append(cast(str, response_hash))
        if len(set(result)) != 32:
            raise Luna8WorkflowError("fit content hashes are not unique")
        return result

    @staticmethod
    def _holdout_items(store: ArtifactStore, holdout: Artifact) -> list[Artifact]:
        refs = holdout.payload.get("item_refs")
        if not isinstance(refs, list) or len(refs) != 32:
            raise Luna8WorkflowError("HoldoutSet cardinality is invalid")
        result: list[Artifact] = []
        for ref in refs:
            if (
                not isinstance(ref, dict)
                or set(ref) != {"item_hash", "item_id"}
                or type(ref.get("item_hash")) is not str
            ):
                raise Luna8WorkflowError("HoldoutSet item ref is invalid")
            item = store.read(cast(str, ref["item_hash"]), expected_schema_name="HoldoutItemContent")
            if item.payload.get("item_id") != ref.get("item_id"):
                raise Luna8WorkflowError("HoldoutSet item identity conflicts")
            result.append(item)
        if len({value.content_hash for value in result}) != 32:
            raise Luna8WorkflowError("HoldoutSet items are not unique")
        return result

    @staticmethod
    def _turns(items: list[Artifact]) -> list[dict[str, JsonValue]]:
        turns: list[dict[str, JsonValue]] = []
        for turn_index in range(4):
            turn_items = items[turn_index * 8 : turn_index * 8 + 8]
            turns.append(
                {
                    "items": [
                        {
                            "item_hash": item.content_hash,
                            "item_id": item.payload["item_id"],
                            "prompt": item.payload["prompt"],
                            "prompt_hash": item.payload["prompt_hash"],
                            "response": item.payload["response"],
                            "response_hash": item.payload["response_hash"],
                        }
                        for item in turn_items
                    ],
                    "turn_index": turn_index + 1,
                }
            )
        return turns

    @staticmethod
    def _packet_item_hashes(packet: Artifact) -> list[str]:
        session = cast(dict[str, object], packet.payload["scoring_session"])
        return [
            cast(str, item["item_hash"])
            for turn in cast(list[dict[str, object]], session["turns"])
            for item in cast(list[dict[str, object]], turn["items"])
        ]

    @classmethod
    def _historical_holdout_set_hashes(cls, state: _State) -> list[str]:
        """Snapshot every globally committed holdout set for this trace under the ledger lock."""

        with cls._holdout_ledger_lock(state.root, state.trace_id) as items:
            root = cls._read_holdout_ledger_root(
                state.root,
                state.store,
                state.trace_id,
                allow_legacy=state.config.holdout_generator_model_id != "fixture-semantic-scenarios-v2",
            )
            if root is not None:
                return [
                    cast(str, entry["holdout_set_hash"])
                    for entry in cast(list[dict[str, JsonValue]], root.payload["entries"])
                ]
            result: list[str] = []
            for ref in sorted(items.parent.glob("attempt-*.ref")):
                try:
                    raw = ref.read_bytes()
                except OSError as error:
                    raise Luna8WorkflowError("historical holdout set ref is unavailable") from error
                if len(raw) != 65 or not raw.endswith(b"\n"):
                    raise Luna8WorkflowError("historical holdout set ref bytes are invalid")
                try:
                    holdout_hash = raw[:-1].decode("ascii")
                except UnicodeDecodeError as error:
                    raise Luna8WorkflowError("historical holdout set ref is not ASCII") from error
                if _HASH.fullmatch(holdout_hash) is None:
                    raise Luna8WorkflowError("historical holdout set ref hash is invalid")
                holdout = state.store.read(holdout_hash, expected_schema_name="HoldoutSet")
                if holdout.payload.get("trace_id") != state.trace_id:
                    raise Luna8WorkflowError("historical holdout set belongs to another trace")
                result.append(holdout.content_hash)
        if len(result) != len(set(result)):
            raise Luna8WorkflowError("historical holdout set refs contain duplicate identities")
        return sorted(result)

    @classmethod
    def _read_holdout_ledger_root(
        cls,
        root: Path,
        store: ArtifactStore,
        trace_id: str,
        *,
        allow_legacy: bool,
    ) -> Artifact | None:
        ledger_root = root / "boundaries" / "holdout-ledger" / trace_id
        items_dir = ledger_root / "items"
        root_refs = sorted((ledger_root / "roots").glob("*.ref"))
        if not root_refs:
            has_refs = any(items_dir.glob("*.ref")) or any(ledger_root.glob("attempt-*.ref"))
            if has_refs and not allow_legacy:
                raise Luna8WorkflowError("holdout ledger refs exist without a committed root manifest")
            return None
        roots: list[Artifact] = []
        for position, root_ref in enumerate(root_refs, start=1):
            if root_ref.name != f"{position:08d}.ref":
                raise Luna8WorkflowError("holdout ledger root sequence is incomplete")
            root_hash = cls._read_ascii_hash_ref(root_ref, "holdout ledger root")
            candidate = store.read(root_hash, expected_schema_name="HoldoutLedgerRoot")
            candidate_entries = candidate.payload.get("entries")
            expected_previous = roots[-1].content_hash if roots else None
            if (
                set(candidate.payload) != {"entries", "entry_count", "previous_root_hash", "schema_version", "trace_id"}
                or candidate.payload.get("schema_version") != "holdout-ledger-root/1.0.0"
                or candidate.payload.get("trace_id") != trace_id
                or not isinstance(candidate_entries, list)
                or candidate.payload.get("entry_count") != position
                or len(candidate_entries) != position
                or candidate.payload.get("previous_root_hash") != expected_previous
                or (roots and candidate_entries[:-1] != roots[-1].payload.get("entries"))
            ):
                raise Luna8WorkflowError("holdout ledger root chain is invalid")
            roots.append(candidate)
        current = roots[-1]
        entries = current.payload.get("entries")
        if (
            set(current.payload) != {"entries", "entry_count", "previous_root_hash", "schema_version", "trace_id"}
            or current.payload.get("schema_version") != "holdout-ledger-root/1.0.0"
            or current.payload.get("trace_id") != trace_id
            or not isinstance(entries, list)
            or current.payload.get("entry_count") != len(entries)
        ):
            raise Luna8WorkflowError("holdout ledger root contract is invalid")
        expected_item_refs: dict[str, str] = {}
        expected_attempt_refs: dict[str, str] = {}
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or set(entry)
                != {
                    "attempt_id",
                    "holdout_set_hash",
                    "item_hashes",
                    "semantic_fingerprints",
                    "visible_content_fingerprints",
                }
                or type(entry.get("attempt_id")) is not str
                or type(entry.get("holdout_set_hash")) is not str
                or not isinstance(entry.get("item_hashes"), list)
                or not isinstance(entry.get("semantic_fingerprints"), list)
                or not isinstance(entry.get("visible_content_fingerprints"), list)
            ):
                raise Luna8WorkflowError("holdout ledger entry is invalid")
            item_hashes = cast(list[str], entry["item_hashes"])
            fingerprints = cast(list[str], entry["semantic_fingerprints"])
            visible_fingerprints = cast(list[str], entry["visible_content_fingerprints"])
            if (
                len(item_hashes) != 32
                or len(fingerprints) != 32
                or len(visible_fingerprints) != 32
                or len(set(visible_fingerprints)) != 32
            ):
                raise Luna8WorkflowError("holdout ledger entry cardinality is invalid")
            holdout = store.read(cast(str, entry["holdout_set_hash"]), expected_schema_name="HoldoutSet")
            refs = cast(list[dict[str, JsonValue]], holdout.payload.get("item_refs"))
            if (
                holdout.payload.get("trace_id") != trace_id
                or holdout.payload.get("attempt_id") != entry["attempt_id"]
                or [ref.get("item_hash") for ref in refs] != item_hashes
            ):
                raise Luna8WorkflowError("holdout ledger entry conflicts with HoldoutSet")
            for fingerprint, visible_fingerprint, item_hash in zip(
                fingerprints, visible_fingerprints, item_hashes, strict=True
            ):
                item = store.read(item_hash, expected_schema_name="HoldoutItemContent")
                if (
                    semantic_profile_fingerprint(cast(str, item.payload["prompt"]), cast(str, item.payload["response"]))
                    != fingerprint
                ):
                    raise Luna8WorkflowError("holdout ledger semantic fingerprint changed")
                if (
                    visible_semantic_content_fingerprint(
                        cast(str, item.payload["prompt"]), cast(str, item.payload["response"])
                    )
                    != visible_fingerprint
                ):
                    raise Luna8WorkflowError("holdout ledger visible content fingerprint changed")
                expected_item_refs[visible_fingerprint] = item_hash
            expected_attempt_refs[cast(str, entry["attempt_id"])] = holdout.content_hash
        if len(expected_item_refs) != len(entries) * 32 or len(expected_attempt_refs) != len(entries):
            raise Luna8WorkflowError("holdout ledger contains duplicate global identities")
        actual_item_refs = {
            path.stem: cls._read_ascii_hash_ref(path, "holdout item") for path in items_dir.glob("*.ref")
        }
        actual_attempt_refs = {
            path.stem.removeprefix("attempt-"): cls._read_ascii_hash_ref(path, "holdout attempt")
            for path in ledger_root.glob("attempt-*.ref")
        }
        if actual_item_refs != expected_item_refs or actual_attempt_refs != expected_attempt_refs:
            raise Luna8WorkflowError("holdout ledger physical refs do not match the committed root")
        return current

    @staticmethod
    def _read_ascii_hash_ref(path: Path, label: str) -> str:
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise Luna8WorkflowError(f"{label} ref is unavailable") from error
        if len(raw) != 65 or not raw.endswith(b"\n"):
            raise Luna8WorkflowError(f"{label} ref bytes are invalid")
        try:
            digest = raw[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise Luna8WorkflowError(f"{label} ref is not ASCII") from error
        if _HASH.fullmatch(digest) is None:
            raise Luna8WorkflowError(f"{label} ref hash is invalid")
        return digest

    @staticmethod
    @contextmanager
    def _holdout_ledger_lock(root: Path, trace_id: str) -> Iterator[Path]:
        ledger_root = root / "boundaries" / "holdout-ledger" / trace_id
        items = ledger_root / "items"
        ArtifactStore.durable_mkdir(items)
        lock_path = ledger_root / "ledger.lock"
        ArtifactStore.durable_touch(lock_path)
        with lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield items
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _task_prompt(state: _State) -> str:
        refs = state.fit_set.payload.get("trajectory_refs")
        if not isinstance(refs, list) or not refs or not isinstance(refs[0], dict):
            raise Luna8WorkflowError("fit task prompt lineage is unavailable")
        manifest_hash = refs[0].get("manifest_hash")
        if type(manifest_hash) is not str:
            raise Luna8WorkflowError("fit task manifest is unavailable")
        manifest = state.store.read(cast(str, manifest_hash), expected_schema_name="TrajectoryManifest")
        content_hash = manifest.payload.get("content_hash")
        if type(content_hash) is not str:
            raise Luna8WorkflowError("fit task content is unavailable")
        content = state.store.read(cast(str, content_hash), expected_schema_name="TrajectoryContent")
        prompt = content.payload.get("prompt")
        if type(prompt) is not str or not prompt:
            raise Luna8WorkflowError("fit task prompt is invalid")
        return cast(str, prompt)

    @staticmethod
    def _attempt_seed(seed: int, attempt_index: int) -> int:
        return int(sha256_hex(canonical_json_bytes({"attempt_index": attempt_index, "seed": seed}))[:13], 16)

    @staticmethod
    def _attempt_index(events: tuple[Artifact, ...]) -> int:
        candidate_events = [event for event in events if event.payload.get("event_type") == "NEXT_CANDIDATE_FROZEN"]
        return len(candidate_events) + 1

    @staticmethod
    def _terminal(events: tuple[Artifact, ...]) -> bool:
        return bool(events and events[-1].payload.get("event_type") == "RUN_CLOSED")

    @staticmethod
    def _artifact_hash(schema_name: str, payload: dict[str, object]) -> str:
        return sha256_hex(
            canonical_json_bytes({"payload": payload, "schema_name": schema_name, "schema_version": "1.0.0"})
        )

    @staticmethod
    def _read_algorithm_contract(store: ArtifactStore, content_hash: str) -> Artifact:
        algorithm = store.read(content_hash, expected_schema_name="RLAlgorithmContract")
        if (
            algorithm.schema_version != "1.0.0"
            or set(algorithm.payload) != {"advantage_estimator", "algorithm_id", "reward_aggregation", "schema_version"}
            or algorithm.payload.get("schema_version") != "rl-algorithm-contract/1.0.0"
            or algorithm.payload.get("advantage_estimator") != "grpo"
            or algorithm.payload.get("reward_aggregation") != "calibrated_scalar"
            or type(algorithm.payload.get("algorithm_id")) is not str
            or not cast(str, algorithm.payload["algorithm_id"]).startswith("fixture-")
        ):
            raise Luna8WorkflowError("RLAlgorithmContract is unavailable or incompatible")
        return algorithm

    @staticmethod
    def _event_details(events: tuple[Artifact, ...], event_type: str) -> dict[str, JsonValue]:
        details = Luna8CertificationWorkflow._optional_event_details(events, event_type)
        if details is None:
            raise Luna8WorkflowError(f"required {event_type} event is missing")
        return details

    @staticmethod
    def _optional_event_details(events: tuple[Artifact, ...], event_type: str) -> dict[str, JsonValue] | None:
        matches = [event for event in events if event.payload.get("event_type") == event_type]
        if not matches:
            return None
        details = matches[-1].payload.get("details")
        if not isinstance(details, dict):
            raise Luna8WorkflowError(f"{event_type} details are invalid")
        return details

    @staticmethod
    def _read_detail_artifact(
        store: ArtifactStore, details: dict[str, JsonValue] | None, key: str, schema: str
    ) -> Artifact | None:
        if details is None:
            return None
        value = details.get(key)
        if type(value) is not str:
            raise Luna8WorkflowError(f"{key} is invalid")
        return store.read(cast(str, value), expected_schema_name=schema)

    @classmethod
    def _current_trace_prompt(
        cls, store: ArtifactStore, events: tuple[Artifact, ...], foundation: dict[str, JsonValue] | None
    ) -> Artifact | None:
        candidates = [event for event in events if event.payload.get("event_type") == "NEXT_CANDIDATE_FROZEN"]
        details = cast(dict[str, JsonValue], candidates[-1].payload["details"]) if candidates else foundation
        return cls._read_detail_artifact(store, details, "trace_judge_prompt_hash", "TraceJudgePrompt")

    @staticmethod
    def _current_artifact(
        store: ArtifactStore,
        events: tuple[Artifact, ...],
        event_type: str,
        key: str,
        schema: str,
        attempt_index: int,
    ) -> Artifact | None:
        for event in reversed(events):
            if event.payload.get("event_type") != event_type or not isinstance(event.payload.get("details"), dict):
                continue
            details = cast(dict[str, object], event.payload["details"])
            if details.get("attempt_index") == attempt_index:
                value = details.get(key)
                if type(value) is not str:
                    raise Luna8WorkflowError(f"{event_type} artifact hash is invalid")
                return store.read(cast(str, value), expected_schema_name=schema)
        return None

    @staticmethod
    def _last_detail_artifact(
        store: ArtifactStore, events: tuple[Artifact, ...], event_type: str, key: str, schema: str
    ) -> Artifact | None:
        for event in reversed(events):
            if event.payload.get("event_type") == event_type and isinstance(event.payload.get("details"), dict):
                value = cast(dict[str, object], event.payload["details"]).get(key)
                if type(value) is not str:
                    raise Luna8WorkflowError(f"{event_type} artifact hash is invalid")
                return store.read(cast(str, value), expected_schema_name=schema)
        return None

    @staticmethod
    def _role_artifacts(
        store: ArtifactStore,
        events: tuple[Artifact, ...],
        event_type: str,
        attempt_index: int,
        role: CertificationRole,
    ) -> tuple[Artifact | None, Artifact | None]:
        for event in reversed(events):
            details = event.payload.get("details")
            if (
                event.payload.get("event_type") != event_type
                or not isinstance(details, dict)
                or details.get("attempt_index") != attempt_index
            ):
                continue
            receipt_payload = details.get("role_ingress")
            if not isinstance(receipt_payload, dict):
                raise Luna8WorkflowError("role ingress event is invalid")
            receipt = CertificationRoleIngressReceipt(
                role=role,
                ingress_hash=cast(str, receipt_payload["ingress_hash"]),
                input_packet_hash=cast(str, receipt_payload["input_packet_hash"]),
                raw_output_hash=cast(str, receipt_payload["raw_output_hash"]),
                normalized_output_hash=cast(str, receipt_payload["normalized_output_hash"]),
                role_invocation_audit_hash=cast(str, receipt_payload["role_invocation_audit_hash"]),
            )
            contract = CertificationRoleIngress.load_public_contract(store.root, receipt)
            return contract.normalized_output, contract.role_invocation_audit
        return None, None

    @staticmethod
    def _recertify_all_role_ingresses(store: ArtifactStore, events: tuple[Artifact, ...]) -> None:
        roles_by_event: dict[str, CertificationRole] = {
            "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED": "AlignmentAuditor",
            "HOLDOUT_STUDENT_OUTPUT_ACCEPTED": "StudentJudge",
            "HOLDOUT_TEACHER_OUTPUT_ACCEPTED": "TeacherScorer",
            "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED": "PromptOptimizer",
        }
        seen: set[tuple[CertificationRole, int]] = set()
        for event in events:
            event_type = event.payload.get("event_type")
            if event_type not in roles_by_event:
                continue
            role = roles_by_event[cast(str, event_type)]
            details = event.payload.get("details")
            receipt_payload = details.get("role_ingress") if isinstance(details, dict) else None
            attempt_index = details.get("attempt_index") if isinstance(details, dict) else None
            if not isinstance(receipt_payload, dict) or type(attempt_index) is not int:
                raise Luna8WorkflowError("accepted role event has incomplete ingress identity")
            identity = (role, cast(int, attempt_index))
            if identity in seen:
                raise Luna8WorkflowError("a role has multiple accepted outputs for one attempt")
            seen.add(identity)
            receipt = CertificationRoleIngressReceipt(
                role=role,
                ingress_hash=cast(str, receipt_payload["ingress_hash"]),
                input_packet_hash=cast(str, receipt_payload["input_packet_hash"]),
                raw_output_hash=cast(str, receipt_payload["raw_output_hash"]),
                normalized_output_hash=cast(str, receipt_payload["normalized_output_hash"]),
                role_invocation_audit_hash=cast(str, receipt_payload["role_invocation_audit_hash"]),
            )
            CertificationRoleIngress.load_public_contract(store.root, receipt)

    @staticmethod
    def _validate_rejected_role_audit_private_bytes(root: Path, audit: Artifact) -> None:
        rejection_identity = audit.payload.get("rejection_identity")
        role = audit.payload.get("role")
        raw_hash = audit.payload.get("raw_output_hash")
        if type(rejection_identity) is not str or type(role) is not str or type(raw_hash) is not str:
            raise Luna8WorkflowError("rejected role audit identity is invalid")
        private = (
            root
            / "private-boundary"
            / "certification-role-rejections"
            / cast(str, role)
            / cast(str, rejection_identity)
        )
        try:
            raw = (private / "role-output.rejected.raw.json").read_bytes()
            manifest_bytes = (private / "rejection-manifest.canonical.json").read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise Luna8WorkflowError("rejected role private evidence is unavailable") from error
        if (
            sha256_hex(raw) != raw_hash
            or manifest != audit.payload
            or canonical_json_bytes(cast(dict[str, JsonValue], manifest)) != manifest_bytes
        ):
            raise Luna8WorkflowError("rejected role private evidence is corrupt")

    @staticmethod
    def _validate_event_order(events: tuple[Artifact, ...]) -> None:
        if events[0].payload.get("event_type") != "RUN_STARTED":
            raise Luna8WorkflowError("Luna@8 journal does not begin with RUN_STARTED")
        foundations = [
            index for index, event in enumerate(events) if event.payload.get("event_type") == "FOUNDATION_FROZEN"
        ]
        requests = [
            index for index, event in enumerate(events) if event.payload.get("event_type") == "HOLDOUT_REQUESTED"
        ]
        if len(foundations) > 1 or (requests and (not foundations or foundations[0] > requests[0])):
            raise Luna8WorkflowError("AlignmentPolicy was not durably frozen before holdout generation")
        if len(requests) > 3:
            raise Luna8WorkflowError("TRY_8 exceeded three prompt attempts")
        if any(event.payload.get("event_type") == "RUN_CLOSED" for event in events[:-1]):
            raise Luna8WorkflowError("terminal event is not last")

    @staticmethod
    def _close_failed(
        state: _State, epoch: int, reason_code: str, evidence: dict[str, JsonValue] | None = None
    ) -> None:
        details: dict[str, JsonValue] = {"reason_code": reason_code}
        if evidence is not None:
            details["evidence"] = evidence
        failure = state.journal.append(
            epoch,
            "LUNA8_WORKFLOW_FAILED",
            details,
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
