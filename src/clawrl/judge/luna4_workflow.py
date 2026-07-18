"""Ticket 06 bounded Luna@4 certification and evidence-driven Sol fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from clawrl.adapters.generators.holdout import FixtureHoldoutGenerator, HoldoutBoundaryError
from clawrl.adapters.scorers.certification_roles import (
    CertificationRole,
    CertificationRoleIngress,
    CertificationRoleIngressReceipt,
    CertificationRoleOutputError,
    aggregate_diagnostic_contract,
    output_schema_contract,
)
from clawrl.adapters.scorers.sol_fallback import FixtureSolFallbackEvidenceAdapter, SolFallbackBoundaryError
from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.judge.certification_models import TraceJudgePrompt
from clawrl.judge.certification_workflow import Luna8CertificationWorkflow, Luna8WorkflowError
from clawrl.judge.luna4_models import (
    FixtureLuna4Config,
    Luna4ContractError,
    ProductionLuna4Config,
)
from clawrl.training.run_journal import RunIdentityConflict, RunJournal

_HASH = re.compile(r"^[0-9a-f]{64}$")
_INPUT_SCHEMA = "Luna4CertificationInput"


class Luna4WorkflowError(RuntimeError):
    """TRY_4 cannot safely advance or be freshly recertified."""


@dataclass(frozen=True, slots=True)
class Luna4WorkflowSnapshot:
    events: tuple[Artifact, ...]
    attempt_index: int
    teacher_packet: Artifact | None
    student_packet: Artifact | None
    auditor_packet: Artifact | None
    optimizer_packet: Artifact | None
    holdout_sets: tuple[Artifact, ...]
    judge_pack: Artifact | None
    golden_hard_entry: Artifact | None
    human_escalation: Artifact | None
    uncertifiable_report: Artifact | None
    terminal: bool


@dataclass(frozen=True, slots=True)
class _State:
    root: Path
    store: ArtifactStore
    journal: RunJournal
    config: FixtureLuna4Config
    workflow_input: Artifact
    events: tuple[Artifact, ...]
    source_snapshot: Any
    source_input: Artifact
    source_foundation: dict[str, JsonValue]
    trace_id: str
    training_trace_hash: str
    fit_set: Artifact
    teacher_label_set: Artifact
    alignment_policy: Artifact
    base_prompt: Artifact
    trace_prompt: Artifact
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
    judge_pack: Artifact | None
    golden_hard_entry: Artifact | None
    human_escalation: Artifact | None
    uncertifiable_report: Artifact | None


class Luna4CertificationWorkflow:
    """Consume one valid TRY_8 exhaustion and produce exactly one TRY_4 terminal."""

    @staticmethod
    def production_readiness(root: str | Path, config: ProductionLuna4Config) -> Artifact:
        checks: list[dict[str, str]] = []
        for code, value in (
            ("ALIGNMENT_POLICY_APPROVAL_UNAVAILABLE", config.alignment_policy_approval_hash),
            ("SOL_MODEL_CONFIGURATION_UNAVAILABLE", config.sol_model_approval_hash),
            ("LUNA_MODEL_CONFIGURATION_UNAVAILABLE", config.luna_model_approval_hash),
            ("HOLDOUT_GENERATOR_CONFIGURATION_UNAVAILABLE", config.holdout_generator_approval_hash),
            ("REWARD_SCHEMA_UNAVAILABLE", config.reward_schema_approval_hash),
            ("SCALARIZER_UNAVAILABLE", config.scalarizer_approval_hash),
            ("RL_ALGORITHM_CONTRACT_UNAVAILABLE", config.algorithm_contract_approval_hash),
            ("SOL_FALLBACK_POLICY_UNAVAILABLE", config.fallback_policy_approval_hash),
            ("HUMAN_ESCALATION_SINK_UNAVAILABLE", config.human_escalation_sink_approval_hash),
            ("BASE_JUDGE_PROMPT_UNAVAILABLE", config.base_prompt_approval_hash),
            ("TRACE_JUDGE_PROMPT_UNAVAILABLE", config.trace_prompt_approval_hash),
            (
                "PERMANENT_TRACE_GOVERNANCE_APPROVAL_UNAVAILABLE",
                config.permanent_trace_governance_approval_hash,
            ),
        ):
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
    def bootstrap(cls, root: str | Path, config: FixtureLuna4Config, *, epoch: int) -> Luna4WorkflowSnapshot:
        root_path = Path(root)
        store = ArtifactStore(root_path)
        try:
            source, source_input, source_foundation = cls._load_source(root_path, config)
            fit_set = store.read(
                cast(str, source_input.payload["fit_trajectory_set_hash"]), expected_schema_name="FitTrajectorySet"
            )
            teacher = store.read(
                cast(str, source_input.payload["teacher_label_set_hash"]), expected_schema_name="TeacherLabelSet"
            )
            payload: dict[str, object] = {
                **config.immutable_input_payload,
                "alignment_policy_hash": cast(Artifact, source.alignment_policy).content_hash,
                "algorithm_contract_hash": source_foundation["algorithm_contract_hash"],
                "base_judge_prompt_hash": cast(Artifact, source.base_judge_prompt).content_hash,
                "fit_trajectory_set_hash": fit_set.content_hash,
                "source_terminal_event_hash": source.events[-1].content_hash,
                "teacher_label_set_hash": teacher.content_hash,
                "trace_id": source.exhaustion_report.payload["trace_id"],
                "training_trace_hash": teacher.payload["training_trace_hash"],
            }
            input_hash = cls._artifact_hash(_INPUT_SCHEMA, payload)
            journal = RunJournal(
                root_path, store, config.run_id, input_schema_name=_INPUT_SCHEMA, input_schema_version="1.0.0"
            )
            journal.reserve_identity(input_hash)
        except (ArtifactCorruption, Luna4ContractError, Luna8WorkflowError, RunIdentityConflict) as error:
            raise Luna4WorkflowError("TRY_4 immutable input cannot be verified") from error
        input_path = store.artifact_dir / f"{input_hash}.json"
        if input_path.exists():
            persisted = store.read(input_hash, expected_schema_name=_INPUT_SCHEMA)
            if persisted.payload != payload:
                raise Luna4WorkflowError("persisted TRY_4 input conflicts with reserved identity")
            if journal.events():
                return cls._snapshot(cls._load_state(root_path, config.run_id))
        workflow_input = store.put(_INPUT_SCHEMA, "1.0.0", payload)
        journal.start_run(
            epoch,
            workflow_input.content_hash,
            {
                "input_hash": workflow_input.content_hash,
                "phase": "luna4_certification",
                "trace_id": payload["trace_id"],
                "try8_exhaustion_hash": config.try8_exhaustion_hash,
                "try8_run_id": config.try8_run_id,
            },
        )
        return cls._snapshot(cls._load_state(root_path, config.run_id))

    @classmethod
    def run_until_role_input(cls, root: str | Path, config: FixtureLuna4Config, *, epoch: int) -> Luna4WorkflowSnapshot:
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
        raise Luna4WorkflowError("TRY_4 exceeded bounded transitions")

    @classmethod
    def submit_role_output(
        cls,
        root: str | Path,
        run_id: str,
        *,
        role_ingress: CertificationRoleIngressReceipt,
        epoch: int,
    ) -> Luna4WorkflowSnapshot:
        state = cls._load_state(Path(root), run_id)
        expected: dict[CertificationRole, Artifact | None] = {
            "TeacherScorer": state.teacher_packet,
            "StudentJudge": state.student_packet,
            "AlignmentAuditor": state.auditor_packet,
            "PromptOptimizer": state.optimizer_packet,
        }
        packet = expected[role_ingress.role]
        if packet is None or packet.content_hash != role_ingress.input_packet_hash or cls._terminal(state.events):
            raise Luna4WorkflowError("role ingress does not match the open TRY_4 state")
        try:
            public = CertificationRoleIngress.load_public_contract(state.root, role_ingress)
        except (ArtifactCorruption, CertificationRoleOutputError) as error:
            raise Luna4WorkflowError("TRY_4 role ingress cannot be recertified") from error
        if public.normalized_output.content_hash != role_ingress.normalized_output_hash:
            raise Luna4WorkflowError("TRY_4 normalized role output conflicts with receipt")
        role_audit = public.role_invocation_audit
        attempt = cls._attempt_index(state.events)
        if role_audit.payload.get("role_lineage_grant_hash") is not None:
            timeout_count = sum(
                event.payload.get("event_type") == "ROLE_INVOCATION_TIMEOUT_OBSERVED"
                and isinstance(event.payload.get("details"), dict)
                and cast(dict[str, object], event.payload["details"]).get("attempt_index") == attempt
                and cast(dict[str, object], event.payload["details"]).get("role") == role_ingress.role
                for event in state.events
            )
            if (
                role_audit.payload.get("attempt_index") != attempt
                or role_audit.payload.get("retry_index") != timeout_count
                or role_audit.payload.get("run_id") != run_id
            ):
                raise Luna4WorkflowError("isolated role output does not match its frozen invocation slot")
        event_type = {
            "TeacherScorer": "HOLDOUT_TEACHER_OUTPUT_ACCEPTED",
            "StudentJudge": "HOLDOUT_STUDENT_OUTPUT_ACCEPTED",
            "AlignmentAuditor": "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED",
            "PromptOptimizer": "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED",
        }[role_ingress.role]
        details = {"attempt_index": attempt, "role_ingress": role_ingress.artifact_payload()}
        prior = cls._event_for_attempt(state.events, event_type, attempt)
        if prior is not None:
            if prior.payload.get("details") != details:
                raise Luna4WorkflowError("a conflicting role output is already committed")
            return cls._snapshot(state)
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
    def record_role_timeout(
        cls,
        root: str | Path,
        run_id: str,
        *,
        role: CertificationRole,
        input_packet_hash: str,
        role_session_lineage: str,
        retry_index: int | None = None,
        epoch: int,
    ) -> Luna4WorkflowSnapshot:
        """Persist a retryable isolated-role timeout without inventing an output."""

        state = cls._load_state(Path(root), run_id)
        expected: dict[CertificationRole, Artifact | None] = {
            "TeacherScorer": state.teacher_packet,
            "StudentJudge": state.student_packet,
            "AlignmentAuditor": state.auditor_packet,
            "PromptOptimizer": state.optimizer_packet,
        }
        accepted: dict[CertificationRole, Artifact | None] = {
            "TeacherScorer": state.teacher_output,
            "StudentJudge": state.student_output,
            "AlignmentAuditor": state.auditor_output,
            "PromptOptimizer": state.optimizer_output,
        }
        packet = expected[role]
        attempt = cls._attempt_index(state.events)
        if (
            cls._terminal(state.events)
            or packet is None
            or packet.content_hash != input_packet_hash
            or accepted[role] is not None
        ):
            raise Luna4WorkflowError("role timeout does not match the current isolated invocation")
        observed_retry_indices: set[int] = set()
        for event in state.events:
            details_value = event.payload.get("details")
            if (
                event.payload.get("event_type") != "ROLE_INVOCATION_TIMEOUT_OBSERVED"
                or not isinstance(details_value, dict)
                or details_value.get("attempt_index") != attempt
                or details_value.get("role") != role
            ):
                continue
            prior_audit = state.store.read(
                cast(str, details_value["timeout_audit_hash"]), expected_schema_name="RoleInvocationTimeoutAudit"
            )
            prior_budget = prior_audit.payload.get("retry_budget")
            prior_index = (
                cast(dict[str, object], prior_budget).get("retry_index")
                if isinstance(prior_budget, dict)
                else prior_audit.payload.get("retry_index", 0)
            )
            if type(prior_index) is not int:
                raise Luna4WorkflowError("prior role timeout retry index is invalid")
            observed_retry_indices.add(prior_index)
        requested_retry_index = len(observed_retry_indices) if retry_index is None else retry_index
        if requested_retry_index not in {0, 1, 2, 3}:
            raise Luna4WorkflowError("role timeout retry index is outside the frozen budget")
        try:
            lineage_grant = CertificationRoleIngress.validate_isolated_lineage(
                state.root,
                role=role,
                input_packet_hash=input_packet_hash,
                role_session_lineage=role_session_lineage,
                run_id=run_id,
                attempt_index=attempt,
                retry_index=requested_retry_index,
            )
        except CertificationRoleOutputError as error:
            raise Luna4WorkflowError("role timeout lineage was not frozen before dispatch") from error
        prior: list[Artifact] = []
        for event in state.events:
            details_value = event.payload.get("details")
            if (
                event.payload.get("event_type") != "ROLE_INVOCATION_TIMEOUT_OBSERVED"
                or not isinstance(details_value, dict)
                or details_value.get("attempt_index") != attempt
                or details_value.get("role") != role
            ):
                continue
            prior_audit = state.store.read(
                cast(str, details_value["timeout_audit_hash"]), expected_schema_name="RoleInvocationTimeoutAudit"
            )
            duplicate_budget = cast(dict[str, object], prior_audit.payload["retry_budget"])
            prior_index = cast(int, duplicate_budget["retry_index"])
            if prior_index == requested_retry_index:
                prior.append(event)
        if requested_retry_index not in observed_retry_indices and observed_retry_indices != set(
            range(requested_retry_index)
        ):
            raise Luna4WorkflowError("role timeout retry evidence is out of sequence")
        audit = state.store.put(
            "RoleInvocationTimeoutAudit",
            "1.0.0",
            {
                "attempt_index": attempt,
                "input_packet_hash": input_packet_hash,
                "reason_code": "ISOLATED_ROLE_TIMEOUT",
                "retry_budget": {"max_retries": 3, "retry_index": requested_retry_index},
                "retry_scheduled": requested_retry_index < 3,
                "role": role,
                "role_lineage_grant_hash": lineage_grant.content_hash,
                "role_session_lineage": role_session_lineage,
                "run_id": run_id,
                "status": "retryable_failure",
            },
        )
        details: dict[str, object] = {
            "attempt_index": attempt,
            "retry_index": requested_retry_index,
            "role": role,
            "role_session_lineage": role_session_lineage,
            "timeout_audit_hash": audit.content_hash,
        }
        if prior:
            if len(prior) != 1 or prior[0].payload.get("details") != details:
                raise Luna4WorkflowError("role timeout conflicts with the committed retry evidence")
            return cls._snapshot(state)
        state.journal.claim_epoch(epoch)
        state.journal.append(
            epoch,
            "ROLE_INVOCATION_TIMEOUT_OBSERVED",
            details,
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def close_role_boundary_exhausted(
        cls,
        root: str | Path,
        run_id: str,
        *,
        role: CertificationRole,
        epoch: int,
    ) -> Luna4WorkflowSnapshot:
        """Close after the frozen three-retry role budget, with no hidden retry."""

        state = cls._load_state(Path(root), run_id)
        if cls._terminal(state.events):
            raise Luna4WorkflowError("role-boundary exhaustion is unavailable")
        accepted_by_role: dict[CertificationRole, Artifact | None] = {
            "TeacherScorer": state.teacher_output,
            "StudentJudge": state.student_output,
            "AlignmentAuditor": state.auditor_output,
            "PromptOptimizer": state.optimizer_output,
        }
        packet_by_role: dict[CertificationRole, Artifact | None] = {
            "TeacherScorer": state.teacher_packet,
            "StudentJudge": state.student_packet,
            "AlignmentAuditor": state.auditor_packet,
            "PromptOptimizer": state.optimizer_packet,
        }
        accepted = accepted_by_role[role]
        packet = packet_by_role[role]
        audits: list[Artifact] = []
        for event in state.events:
            if event.payload.get("event_type") != "ROLE_INVOCATION_TIMEOUT_OBSERVED":
                continue
            details = cast(dict[str, object], event.payload["details"])
            if details.get("role") == role and details.get("attempt_index") == cls._attempt_index(state.events):
                audits.append(
                    state.store.read(
                        cast(str, details["timeout_audit_hash"]), expected_schema_name="RoleInvocationTimeoutAudit"
                    )
                )
        retry_indices = {
            cast(dict[str, object], audit.payload["retry_budget"]).get("retry_index")
            if isinstance(audit.payload.get("retry_budget"), dict)
            else audit.payload.get("retry_index", 0)
            for audit in audits
        }
        if packet is None or accepted is not None or len(audits) != 4 or retry_indices != {0, 1, 2, 3}:
            raise Luna4WorkflowError("role boundary has not exhausted its exact retry budget")
        report = state.store.put(
            "RoleBoundaryExhaustionReport",
            "1.0.0",
            {
                "attempt_index": cls._attempt_index(state.events),
                "input_packet_hash": packet.content_hash,
                "max_retries": 3,
                "reason_code": "ISOLATED_ROLE_BOUNDARY_EXHAUSTED",
                "role": role,
                "status": "failed_closed",
                "timeout_audit_hashes": [audit.content_hash for audit in audits],
                "total_invocations": 4,
            },
        )
        event = cls._append(
            state,
            epoch,
            "TRY4_FAILED_CLOSED",
            {"failure_report_hash": report.content_hash, "reason_code": "ISOLATED_ROLE_BOUNDARY_EXHAUSTED"},
        )
        state.journal.close(
            epoch,
            status="failed",
            reason_code="ISOLATED_ROLE_BOUNDARY_EXHAUSTED",
            expected_sequence=len(state.events) + 2,
            expected_previous_hash=event.content_hash,
        )
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def resume(cls, root: str | Path, run_id: str, *, epoch: int) -> Luna4WorkflowSnapshot:
        state = cls._load_state(Path(root), run_id)
        if cls._terminal(state.events):
            return cls._snapshot(state)
        last = cast(str, state.events[-1].payload["event_type"])
        if (
            last == "SCORING_PACKETS_COMMITTED"
            or last == "ROLE_INVOCATION_TIMEOUT_OBSERVED"
            or (
                last in {"HOLDOUT_TEACHER_OUTPUT_ACCEPTED", "HOLDOUT_STUDENT_OUTPUT_ACCEPTED"}
                and (state.teacher_output is None or state.student_output is None)
            )
            or last in {"AUDITOR_PACKET_COMMITTED", "OPTIMIZER_PACKET_COMMITTED"}
        ):
            return cls._snapshot(state)
        state.journal.claim_epoch(epoch)
        attempt = cls._attempt_index(state.events)
        if last == "RUN_STARTED":
            decision = state.store.put(
                "DecisionRecord",
                "1.0.0",
                {
                    "decision": "enter_bounded_try4_only_after_verified_try8_exhaustion",
                    "items_per_turn": 4,
                    "max_attempts": 3,
                    "reason_code": "TICKET06_RECOMMENDED_DEFAULTS_ADOPTED",
                    "try8_exhaustion_hash": state.config.try8_exhaustion_hash,
                },
            )
            cls._append(state, epoch, "TRY4_FOUNDATION_FROZEN", {"decision_record_hash": decision.content_hash})
        elif last in {"TRY4_FOUNDATION_FROZEN", "NEXT_CANDIDATE_FROZEN"}:
            cls._freeze_holdout_plan(state, epoch, attempt)
        elif last == "HOLDOUT_PLAN_FROZEN":
            cls._request_holdout(state, epoch, attempt)
        elif last in {"HOLDOUT_REQUESTED", "HOLDOUT_RETRY_SCHEDULED"}:
            cls._execute_holdout(state, epoch, attempt)
        elif last == "HOLDOUT_COMMITTED":
            teacher, student = cls._build_scoring_packets(state, attempt)
            cls._append(
                state,
                epoch,
                "SCORING_PACKETS_COMMITTED",
                {
                    "attempt_index": attempt,
                    "student_packet_hash": student.content_hash,
                    "teacher_packet_hash": teacher.content_hash,
                },
            )
        elif last in {"HOLDOUT_TEACHER_OUTPUT_ACCEPTED", "HOLDOUT_STUDENT_OUTPUT_ACCEPTED"}:
            if state.teacher_output is None or state.student_output is None:
                return cls._snapshot(state)
            auditor = cls._build_auditor_packet(state, attempt)
            cls._append(
                state,
                epoch,
                "AUDITOR_PACKET_COMMITTED",
                {"attempt_index": attempt, "auditor_packet_hash": auditor.content_hash},
            )
        elif last == "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED":
            if state.auditor_output is None:
                raise Luna4WorkflowError("accepted auditor output is missing")
            if state.auditor_output.payload.get("verdict") == "pass":
                pack = cls._build_luna4_pack(state, attempt)
                event = cls._append(
                    state,
                    epoch,
                    "LUNA4_TERMINAL_PACK_COMMITTED",
                    {"attempt_index": attempt, "judge_pack_hash": pack.content_hash},
                )
                state.journal.close(
                    epoch,
                    status="succeeded",
                    reason_code="LUNA4_CERTIFIED_TERMINAL",
                    expected_sequence=len(state.events) + 2,
                    expected_previous_hash=event.content_hash,
                )
            elif attempt < 3:
                optimizer = cls._build_optimizer_packet(state, attempt)
                cls._append(
                    state,
                    epoch,
                    "OPTIMIZER_PACKET_COMMITTED",
                    {"attempt_index": attempt, "optimizer_packet_hash": optimizer.content_hash},
                )
            else:
                cls._close_fallback_or_uncertifiable(state, epoch)
        elif last == "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED":
            prompt = cls._freeze_next_candidate(state, attempt + 1)
            cls._append(
                state,
                epoch,
                "NEXT_CANDIDATE_FROZEN",
                {"attempt_index": attempt + 1, "trace_judge_prompt_hash": prompt.content_hash},
            )
        else:
            raise Luna4WorkflowError(f"TRY_4 event cannot be advanced: {last}")
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def _freeze_holdout_plan(cls, state: _State, epoch: int, attempt: int) -> None:
        history = [item.content_hash for item in state.source_snapshot.holdout_sets]
        history.extend(item.content_hash for item in state.holdout_sets)
        plan = state.store.put(
            "HoldoutPlan",
            "1.0.0",
            {
                "alignment_policy_hash": state.alignment_policy.content_hash,
                "attempt_id": cls._attempt_id(state.config.run_id, attempt),
                "attempt_index": attempt,
                "fit_trajectory_set_hash": state.fit_set.content_hash,
                "generator_inference_hash": state.source_foundation["holdout_generator_inference_hash"],
                "generator_model_id": state.config.holdout_generator_model_id,
                "generator_profile_id": state.config.holdout_generator_profile_id,
                "historical_holdout_hashes": history,
                "holdout_seed": state.config.holdout_seed + attempt * 10_003,
                "item_count": 32,
                "items_per_turn": 4,
                "output_fault": state.config.output_fault,
                "phase": "TRY_4",
                "policy_frozen_before_generation": True,
                "trace_id": state.trace_id,
            },
        )
        cls._append(
            state,
            epoch,
            "HOLDOUT_PLAN_FROZEN",
            {"attempt_index": attempt, "holdout_plan_hash": plan.content_hash},
        )

    @classmethod
    def _request_holdout(cls, state: _State, epoch: int, attempt: int) -> None:
        plan = cls._artifact_for_attempt(state, "HOLDOUT_PLAN_FROZEN", "holdout_plan_hash", "HoldoutPlan", attempt)
        inference_hash = cast(str, state.source_foundation["holdout_generator_inference_hash"])
        request = state.store.put(
            "HoldoutGeneratorRequest",
            "1.0.0",
            {
                "attempt_id": plan.payload["attempt_id"],
                "attempt_index": attempt,
                "fit_content_hashes": Luna8CertificationWorkflow._fit_content_hashes(cast(Any, state)),
                "generator_inference_hash": inference_hash,
                "generator_model_id": state.config.holdout_generator_model_id,
                "generator_profile_id": state.config.holdout_generator_profile_id,
                "holdout_seed": state.config.holdout_seed + attempt * 10_003,
                "output_fault": state.config.output_fault,
                "policy_hash": state.alignment_policy.content_hash,
                "prompt": Luna8CertificationWorkflow._task_prompt(cast(Any, state)),
                "request_schema_version": "holdout-generator-request/1.0.0",
                "trace_id": state.trace_id,
            },
        )
        cls._append(
            state,
            epoch,
            "HOLDOUT_REQUESTED",
            {"attempt_index": attempt, "boundary_sequence": 1, "holdout_request_hash": request.content_hash},
        )

    @classmethod
    def _execute_holdout(cls, state: _State, epoch: int, attempt: int) -> None:
        request = cls._artifact_for_attempt(
            state, "HOLDOUT_REQUESTED", "holdout_request_hash", "HoldoutGeneratorRequest", attempt
        )
        retries = [
            event
            for event in state.events
            if event.payload.get("event_type") == "HOLDOUT_RETRY_SCHEDULED"
            and cast(dict[str, object], event.payload.get("details", {})).get("attempt_index") == attempt
        ]
        sequence = len(retries) + 1
        generator = FixtureHoldoutGenerator(state.root, state.store, cast(Any, state.config))
        try:
            observation = generator.execute(request, boundary_sequence=sequence)
            attempts = generator.verify_attempt_chain(request, sequence)
        except HoldoutBoundaryError as error:
            raise Luna4WorkflowError("holdout boundary evidence is invalid") from error
        generator_attempt = attempts[-1]
        status = observation.payload.get("status")
        if status == "retryable":
            if sequence >= len(state.config.fault_schedule):
                raise Luna4WorkflowError("holdout retry policy exhausted without terminal boundary evidence")
            cls._append(
                state,
                epoch,
                "HOLDOUT_RETRY_SCHEDULED",
                {
                    "attempt_index": attempt,
                    "boundary_observation_hash": observation.content_hash,
                    "boundary_sequence": sequence + 1,
                    "generator_attempt_hash": generator_attempt.content_hash,
                },
            )
            return
        if status != "succeeded":
            cls._close_failed(
                state,
                epoch,
                "HOLDOUT_GENERATOR_PERMANENT_FAILURE",
                {
                    "attempt_index": attempt,
                    "boundary_observation_hash": observation.content_hash,
                    "boundary_sequence": sequence,
                    "generator_attempt_hash": generator_attempt.content_hash,
                    "holdout_request_hash": request.content_hash,
                },
            )
            return
        raw = generator.read_raw_batch(observation, request)
        # The shared commit routine owns exact-32 validation, diversity, fit/history
        # disjointness, immutable item publication, and the cumulative global ledger.
        current_plan_event = cls._event_for_attempt(state.events, "HOLDOUT_PLAN_FROZEN", attempt)
        if current_plan_event is None:
            raise Luna4WorkflowError("current holdout plan is unavailable")
        commit_state = SimpleNamespace(**{field: getattr(state, field) for field in state.__dataclass_fields__})
        commit_state.events = (current_plan_event,)
        try:
            holdout, ledger = Luna8CertificationWorkflow._commit_holdout(
                cast(Any, commit_state), raw, observation, generator_attempt, attempt
            )
        except (Luna8WorkflowError, ValueError) as error:
            raise Luna4WorkflowError("TRY_4 holdout failed shared validation") from error
        cls._append(
            state,
            epoch,
            "HOLDOUT_COMMITTED",
            {
                "attempt_index": attempt,
                "holdout_ledger_root_hash": ledger.content_hash,
                "holdout_set_hash": holdout.content_hash,
            },
        )

    @classmethod
    def _build_scoring_packets(cls, state: _State, attempt: int) -> tuple[Artifact, Artifact]:
        holdout = state.holdout_sets[-1]
        items = Luna8CertificationWorkflow._holdout_items(state.store, holdout)
        turns = cls._turns(items, 4)
        attempt_id = cast(str, holdout.payload["attempt_id"])
        sol_hash = cast(
            str, cast(dict[str, object], state.teacher_label_set.payload["lineage"])["sol_inference_config_hash"]
        )
        rubric_hash = cast(
            str, cast(dict[str, object], state.teacher_label_set.payload["lineage"])["initial_eval_rubric_hash"]
        )
        sol = state.store.read(sol_hash, expected_schema_name="SolInferenceConfig")
        rubric = state.store.read(rubric_hash, expected_schema_name="InitialEvalRubric")
        reward = state.store.read(
            cast(str, state.source_foundation["reward_schema_hash"]), expected_schema_name="RewardSchema"
        )
        scalarizer = state.store.read(
            cast(str, state.source_foundation["scalarizer_hash"]), expected_schema_name="Scalarizer"
        )
        common: dict[str, object] = {
            "aggregation": "calibrated_scalar",
            "attempt_id": attempt_id,
            "attempt_index": attempt,
            "holdout_set_hash": holdout.content_hash,
            "items_per_turn": 4,
            "reward_schema": {"artifact_hash": reward.content_hash, "contract": reward.payload},
            "scalarizer": {"artifact_hash": scalarizer.content_hash, "contract": scalarizer.payload},
            "trace_id": state.trace_id,
        }
        teacher_session = (
            f"ticket06-sol-{sha256_hex(canonical_json_bytes({'attempt': attempt_id, 'role': 'teacher'}))[:24]}"
        )
        teacher_payload: dict[str, object] = {
            **common,
            "initial_eval_rubric": {"artifact_hash": rubric_hash, "contract": rubric.payload},
            "output_schema": output_schema_contract("TeacherScorer", items_per_turn=4),
            "packet_id": f"t06-teacher-{attempt_id[-16:]}",
            "role": "teacher_scorer",
            "schema_version": "holdout-teacher-input/1.0.0",
            "scoring_session": {"items_per_turn": 4, "session_id": teacher_session, "turn_count": 8, "turns": turns},
            "seed": state.config.role_seed + attempt * 10 + 1,
            "sol_inference": {"artifact_hash": sol_hash, "contract": sol.payload},
        }
        teacher_payload["allowlisted_fields"] = sorted([*teacher_payload, "allowlisted_fields"])
        teacher = state.store.put("HoldoutTeacherInputPacket", "1.0.0", teacher_payload)
        student_session_hash = sha256_hex(
            canonical_json_bytes({"attempt": attempt_id, "prompt": state.trace_prompt.content_hash})
        )
        student_session = f"ticket06-luna-{student_session_hash[:24]}"
        student_payload: dict[str, object] = {
            **common,
            "candidate_prompt": {
                "artifact_hash": state.trace_prompt.content_hash,
                "contract": state.trace_prompt.payload,
            },
            "luna_inference": {
                "artifact_hash": state.source_foundation["luna_inference_config_hash"],
                "contract": state.store.read(
                    cast(str, state.source_foundation["luna_inference_config_hash"]),
                    expected_schema_name="LunaInferenceConfig",
                ).payload,
            },
            "output_schema": output_schema_contract("StudentJudge", items_per_turn=4),
            "packet_id": f"t06-student-{attempt_id[-16:]}",
            "role": "student_judge",
            "schema_version": "holdout-student-input/1.0.0",
            "scoring_session": {"items_per_turn": 4, "session_id": student_session, "turn_count": 8, "turns": turns},
            "seed": state.config.role_seed + attempt * 10 + 2,
        }
        student_payload["allowlisted_fields"] = sorted([*student_payload, "allowlisted_fields"])
        student = state.store.put("HoldoutStudentInputPacket", "1.0.0", student_payload)
        CertificationRoleIngress.validate_input_packet(state.root, "TeacherScorer", teacher.content_hash)
        CertificationRoleIngress.validate_input_packet(state.root, "StudentJudge", student.content_hash)
        return teacher, student

    @classmethod
    def _build_auditor_packet(cls, state: _State, attempt: int) -> Artifact:
        if None in (state.teacher_packet, state.student_packet, state.teacher_output, state.student_output):
            raise Luna4WorkflowError("auditor inputs are incomplete")
        teacher = cast(Artifact, state.teacher_output)
        student = cast(Artifact, state.student_output)
        item_hashes = Luna8CertificationWorkflow._packet_item_hashes(cast(Artifact, state.teacher_packet))
        if item_hashes != Luna8CertificationWorkflow._packet_item_hashes(cast(Artifact, state.student_packet)):
            raise Luna4WorkflowError("Sol and Luna packets do not bind the same holdout")
        auditor_session_hash = sha256_hex(
            canonical_json_bytes({"student": student.content_hash, "teacher": teacher.content_hash})
        )
        payload: dict[str, object] = {
            "aggregation": "calibrated_scalar",
            "alignment_policy": state.alignment_policy.payload,
            "alignment_policy_hash": state.alignment_policy.content_hash,
            "attempt_id": state.holdout_sets[-1].payload["attempt_id"],
            "attempt_index": attempt,
            "comparison": {
                "item_hashes": item_hashes,
                "student_labels": student.payload["labels"],
                "teacher_labels": teacher.payload["labels"],
            },
            "diagnostic_contract": aggregate_diagnostic_contract(),
            "holdout_set_hash": state.holdout_sets[-1].content_hash,
            "output_schema": output_schema_contract("AlignmentAuditor"),
            "packet_id": f"t06-auditor-{cast(str, state.holdout_sets[-1].payload['attempt_id'])[-16:]}",
            "preregistered_diagnostics": output_schema_contract("AlignmentAuditor")["diagnostic_required"],
            "role": "alignment_auditor",
            "schema_version": "certification-auditor-input/1.0.0",
            "seed": state.config.role_seed + attempt * 10 + 3,
            "session": {"auditor_session_id": f"ticket06-auditor-{auditor_session_hash[:24]}"},
        }
        payload["allowlisted_fields"] = sorted([*payload, "allowlisted_fields"])
        packet = state.store.put("CertificationAuditorInputPacket", "1.0.0", payload)
        CertificationRoleIngress.validate_input_packet(state.root, "AlignmentAuditor", packet.content_hash)
        return packet

    @classmethod
    def _build_optimizer_packet(cls, state: _State, attempt: int) -> Artifact:
        if state.auditor_output is None:
            raise Luna4WorkflowError("optimizer aggregate input is unavailable")
        payload: dict[str, object] = {
            "aggregate_diagnostics": state.auditor_output.payload["aggregate_diagnostics"],
            "attempt_index": attempt,
            "base_judge_prompt": {
                "artifact_hash": state.base_prompt.content_hash,
                "contract": state.base_prompt.payload,
            },
            "current_trace_judge_prompt": {
                "artifact_hash": state.trace_prompt.content_hash,
                "contract": state.trace_prompt.payload,
            },
            "next_candidate_index": attempt + 1,
            "output_schema": output_schema_contract("PromptOptimizer"),
            "packet_id": f"t06-optimizer-{attempt}-{state.config.run_id[-12:]}",
            "role": "prompt_optimizer",
            "schema_version": "certification-optimizer-input/1.0.0",
            "seed": state.config.role_seed + attempt * 10 + 4,
            "session": {"optimizer_session_id": f"ticket06-optimizer-{state.config.run_id[-16:]}-{attempt}"},
            "verdict": "fail",
        }
        payload["allowlisted_fields"] = sorted([*payload, "allowlisted_fields"])
        packet = state.store.put("CertificationOptimizerInputPacket", "1.0.0", payload)
        CertificationRoleIngress.validate_input_packet(state.root, "PromptOptimizer", packet.content_hash)
        return packet

    @classmethod
    def _freeze_next_candidate(cls, state: _State, candidate: int) -> Artifact:
        if state.optimizer_output is None:
            raise Luna4WorkflowError("next TRY_4 candidate requires an optimizer output")
        prompt = TraceJudgePrompt(
            candidate_index=candidate,
            parent_prompt_hash=state.trace_prompt.content_hash,
            prompt_id=f"ticket06-{state.trace_id[-12:]}-{candidate}",
            text=cast(str, state.optimizer_output.payload["candidate_prompt"]),
            trace_id=state.trace_id,
        )
        return state.store.put("TraceJudgePrompt", "1.0.0", prompt.artifact_payload())

    @classmethod
    def _build_luna4_pack(cls, state: _State, attempt: int) -> Artifact:
        if None in (state.auditor_output, state.auditor_audit, state.teacher_audit, state.student_audit):
            raise Luna4WorkflowError("Luna@4 pack lacks audited pass evidence")
        auditor_output = cast(Artifact, state.auditor_output)
        auditor_audit = cast(Artifact, state.auditor_audit)
        if auditor_output.payload.get("verdict") != "pass":
            raise Luna4WorkflowError("a failed attempt cannot produce a Luna@4 pack")
        identity = cls._judge_pack_identity(state, attempt)
        return state.store.put(
            "JudgePack",
            "1.0.0",
            {
                **identity,
                "alignment_diagnostics": auditor_output.payload["aggregate_diagnostics"],
                "certification_level": 4,
                "certification_mode": "student_holdout",
                "holdout_set_hash": state.holdout_sets[-1].content_hash,
                "role_lineage": {
                    "alignment_auditor_audit_hash": auditor_audit.content_hash,
                    "student_judge_audit_hash": cast(Artifact, state.student_audit).content_hash,
                    "teacher_scorer_audit_hash": cast(Artifact, state.teacher_audit).content_hash,
                },
                "scorer_tier": "luna",
                "status": "terminal",
            },
        )

    @staticmethod
    def _judge_pack_identity(state: _State, attempt: int) -> dict[str, JsonValue]:
        teacher_lineage = cast(dict[str, JsonValue], state.teacher_label_set.payload["lineage"])
        dataset_version_hash = cast(str, state.fit_set.payload["dataset_version_hash"])
        dataset_version = state.store.read(dataset_version_hash, expected_schema_name="DatasetVersion")
        return {
            "aggregation": "calibrated_scalar",
            "algorithm_contract_hash": state.source_foundation["algorithm_contract_hash"],
            "alignment_policy_hash": state.alignment_policy.content_hash,
            "attempt_id": state.holdout_sets[-1].payload["attempt_id"],
            "attempt_index": attempt,
            "base_judge_prompt_hash": state.base_prompt.content_hash,
            "dataset_version_hash": dataset_version_hash,
            "dataset_version_id": dataset_version.payload["dataset_version_id"],
            "fit_trajectory_set_hash": state.fit_set.content_hash,
            "initial_eval_rubric_hash": teacher_lineage["initial_eval_rubric_hash"],
            "items_per_turn": 4,
            "luna_inference_config_hash": state.source_foundation["luna_inference_config_hash"],
            "reward_schema_hash": state.source_foundation["reward_schema_hash"],
            "scalarizer_hash": state.source_foundation["scalarizer_hash"],
            "sol_inference_config_hash": teacher_lineage["sol_inference_config_hash"],
            "teacher_calibration_profile_hash": state.source_foundation["teacher_calibration_profile_hash"],
            "teacher_label_set_hash": state.teacher_label_set.content_hash,
            "trace_id": state.trace_id,
            "trace_judge_prompt_hash": state.trace_prompt.content_hash,
            "trace_prompt_derivation_hash": state.source_foundation["trace_prompt_derivation_hash"],
            "training_trace_hash": state.training_trace_hash,
            "try8_exhaustion_hash": state.config.try8_exhaustion_hash,
            "try8_source_terminal_event_hash": state.workflow_input.payload["source_terminal_event_hash"],
        }

    @classmethod
    def _close_fallback_or_uncertifiable(cls, state: _State, epoch: int) -> None:
        failures = cls._failed_attempt_lineage(state)
        evidence = cls._sol_fallback_evidence(state)
        if evidence.payload.get("calibrated_scalar_valid") is not True:
            report = state.store.put(
                "UncertifiableJudgeReport",
                "1.0.0",
                {
                    "blocks_bundle_publication": True,
                    "diagnostic_only": evidence.payload.get("relative_order_available") is True,
                    "evidence_tags": [
                        "TEACHER_NO_VALID_CALIBRATED_VARIANCE",
                        (
                            "SOL_GROUP_RELATIVE_ONLY"
                            if evidence.payload.get("reason_code") == "SOL_GROUP_RELATIVE_ONLY_UNSUPPORTED_V1"
                            else "SOL_CALIBRATED_VARIANCE_INVALID"
                        ),
                    ],
                    "failure_lineage": failures,
                    "reason_code": evidence.payload["reason_code"],
                    "sol_fallback_evidence_hash": evidence.content_hash,
                    "status": "uncertifiable",
                    "trace_id": state.trace_id,
                    "training_authorized": False,
                    "try8_exhaustion_hash": state.config.try8_exhaustion_hash,
                },
            )
            event = cls._append(
                state,
                epoch,
                "UNCERTIFIABLE_TERMINAL",
                {"uncertifiable_report_hash": report.content_hash},
            )
            state.journal.close(
                epoch,
                status="failed",
                reason_code="SOL_CALIBRATED_SCALAR_UNCERTIFIABLE",
                expected_sequence=len(state.events) + 2,
                expected_previous_hash=event.content_hash,
            )
            return
        golden = state.store.put(
            "GoldenHardEntry",
            "1.0.0",
            {
                "failure_lineage": failures,
                "reason_tags": cls._failure_reason_tags(state),
                "schema_version": "golden-hard-entry/1.0.0",
                "source_try8_exhaustion_hash": state.config.try8_exhaustion_hash,
                "status": "active",
                "trace_id": state.trace_id,
            },
        )
        escalation = state.store.put(
            "HumanEscalation",
            "1.0.0",
            {
                "blocking": False,
                "golden_hard_entry_hash": golden.content_hash,
                "reason_code": "LUNA8_AND_LUNA4_EXHAUSTED",
                "requested_action": "review_trace_and_future_prompt_strategy",
                "status": "open_nonblocking",
                "trace_id": state.trace_id,
            },
        )
        pack = state.store.put(
            "JudgePack",
            "1.0.0",
            {
                **cls._judge_pack_identity(state, 3),
                "attempted_holdout_set_hashes": [item.content_hash for item in state.holdout_sets],
                "certification_mode": "teacher_fallback",
                "failure_lineage": failures,
                "golden_hard_entry_hash": golden.content_hash,
                "human_escalation_hash": escalation.content_hash,
                "scorer_tier": "sol",
                "sol_fallback_evidence_hash": evidence.content_hash,
                "status": "terminal",
            },
        )
        event = cls._append(
            state,
            epoch,
            "SOL_FALLBACK_TERMINAL_PACK_COMMITTED",
            {
                "golden_hard_entry_hash": golden.content_hash,
                "human_escalation_hash": escalation.content_hash,
                "judge_pack_hash": pack.content_hash,
            },
        )
        state.journal.close(
            epoch,
            status="succeeded",
            reason_code="SOL_FALLBACK_TERMINAL",
            expected_sequence=len(state.events) + 2,
            expected_previous_hash=event.content_hash,
        )

    @classmethod
    def _sol_fallback_evidence(cls, state: _State) -> Artifact:
        request, observation = FixtureSolFallbackEvidenceAdapter(state.root, state.store).execute(
            run_id=state.config.run_id,
            teacher_label_set=state.teacher_label_set,
            mode=state.config.sol_evidence_mode,
        )
        classification = cast(dict[str, JsonValue], observation.payload["classification"])
        return state.store.put(
            "SolFallbackEvidence",
            "1.0.0",
            {
                **classification,
                "boundary_observation_hash": observation.content_hash,
                "boundary_request_hash": request.content_hash,
                "label_set_hash": state.teacher_label_set.content_hash,
            },
        )

    @staticmethod
    def _recertify_sol_fallback_boundary(
        root: Path,
        store: ArtifactStore,
        config: FixtureLuna4Config,
        teacher_label_set: Artifact,
        judge_pack: Artifact | None,
        uncertifiable: Artifact | None,
    ) -> None:
        evidence_hash: object = None
        if judge_pack is not None and judge_pack.payload.get("scorer_tier") == "sol":
            evidence_hash = judge_pack.payload.get("sol_fallback_evidence_hash")
        elif uncertifiable is not None:
            evidence_hash = uncertifiable.payload.get("sol_fallback_evidence_hash")
        else:
            return
        evidence = store.read(cast(str, evidence_hash), expected_schema_name="SolFallbackEvidence")
        request = store.read(
            cast(str, evidence.payload["boundary_request_hash"]), expected_schema_name="SolFallbackBoundaryRequest"
        )
        observation = store.read(
            cast(str, evidence.payload["boundary_observation_hash"]),
            expected_schema_name="SolFallbackBoundaryObservation",
        )
        FixtureSolFallbackEvidenceAdapter(root, store).verify(
            request=request,
            observation=observation,
            teacher_label_set=teacher_label_set,
            mode=config.sol_evidence_mode,
        )
        expected: dict[str, JsonValue] = {
            **cast(dict[str, JsonValue], observation.payload["classification"]),
            "boundary_observation_hash": observation.content_hash,
            "boundary_request_hash": request.content_hash,
            "label_set_hash": teacher_label_set.content_hash,
        }
        if evidence.payload != expected:
            raise Luna4WorkflowError("Sol fallback evidence does not match its boundary observation")

    @classmethod
    def _failure_reason_tags(cls, state: _State) -> list[str]:
        diagnostics = [
            cast(dict[str, int], output.payload["aggregate_diagnostics"])
            for output in cls._all_role_outputs(state, "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED", "AlignmentAuditor")
        ]
        tags: set[str] = set()
        minimum = cast(int, state.alignment_policy.payload["min_student_variance_micros"])
        if any(item["student_variance_micros"] < minimum for item in diagnostics):
            tags.add("TEACHER_VARIANCE_STUDENT_COLLAPSE")
        teacher_scalars = cls._teacher_fit_scalars(state)
        if (
            teacher_scalars
            and len(set(teacher_scalars)) > 1
            and sum(teacher_scalars) // len(teacher_scalars) < 60_000_000
        ):
            tags.add("LOW_SCALAR_WITH_RELATIVE_ORDER")
        if not tags:
            tags.add("STUDENT_CALIBRATION_MISMATCH")
        return sorted(tags)

    @classmethod
    def _failed_attempt_lineage(cls, state: _State) -> list[dict[str, JsonValue]]:
        outputs = cls._all_role_outputs(state, "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED", "AlignmentAuditor")
        if len(outputs) != 3 or len(state.holdout_sets) != 3:
            raise Luna4WorkflowError("Sol fallback requires exactly three audited TRY_4 failures")
        result: list[dict[str, JsonValue]] = []
        for index, (output, holdout) in enumerate(zip(outputs, state.holdout_sets, strict=True), start=1):
            if output.payload.get("verdict") != "fail":
                raise Luna4WorkflowError("Sol fallback lineage contains a passing TRY_4 attempt")
            role_audits = {
                "alignment_auditor_audit_hash": cls._role_receipt_for_attempt(
                    state.events, "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED", index
                ).role_invocation_audit_hash,
                "student_judge_audit_hash": cls._role_receipt_for_attempt(
                    state.events, "HOLDOUT_STUDENT_OUTPUT_ACCEPTED", index
                ).role_invocation_audit_hash,
                "teacher_scorer_audit_hash": cls._role_receipt_for_attempt(
                    state.events, "HOLDOUT_TEACHER_OUTPUT_ACCEPTED", index
                ).role_invocation_audit_hash,
            }
            prompt = cls._prompt_for_attempt(state, index)
            result.append(
                {
                    "attempt_id": holdout.payload["attempt_id"],
                    "attempt_index": index,
                    "auditor_output_hash": output.content_hash,
                    "holdout_set_hash": holdout.content_hash,
                    **role_audits,
                    "trace_judge_prompt_hash": prompt.content_hash,
                }
            )
        return result

    @classmethod
    def _role_receipt_for_attempt(
        cls, events: tuple[Artifact, ...], event_type: str, attempt: int
    ) -> CertificationRoleIngressReceipt:
        event = cls._event_for_attempt(events, event_type, attempt)
        if event is None:
            raise Luna4WorkflowError(f"attempt {attempt} lacks {event_type} role receipt")
        details = cast(dict[str, object], event.payload["details"])
        return cls._receipt(cast(dict[str, object], details["role_ingress"]))

    @classmethod
    def _prompt_for_attempt(cls, state: _State, attempt: int) -> Artifact:
        if attempt == 1:
            return cast(Artifact, state.source_snapshot.trace_judge_prompt)
        events = [
            event
            for event in state.events
            if event.payload.get("event_type") == "NEXT_CANDIDATE_FROZEN"
            and cast(dict[str, object], event.payload["details"]).get("attempt_index") == attempt
        ]
        if len(events) != 1:
            raise Luna4WorkflowError(f"attempt {attempt} trace prompt lineage is incomplete")
        return cls._artifact_from_event(state.store, events[0], "trace_judge_prompt_hash", "TraceJudgePrompt")

    @classmethod
    def _load_state(cls, root: Path, run_id: str) -> _State:
        store = ArtifactStore(root)
        journal = RunJournal(root, store, run_id, input_schema_name=_INPUT_SCHEMA, input_schema_version="1.0.0")
        try:
            journal.verify()
            workflow_input = store.read(journal.reserved_input_hash(), expected_schema_name=_INPUT_SCHEMA)
            config = FixtureLuna4Config.from_mapping(
                {
                    key: value
                    for key, value in workflow_input.payload.items()
                    if key in FixtureLuna4Config.__dataclass_fields__ or key == "holdout_generator_inference"
                }
            )
            source, source_input, foundation = cls._load_source(root, config)
            events = tuple(journal.events())
            if not events or events[0].payload.get("event_type") != "RUN_STARTED":
                raise Luna4WorkflowError("TRY_4 journal has not started")
            if sum(event.payload.get("event_type") == "HOLDOUT_COMMITTED" for event in events) > 3:
                raise Luna4WorkflowError("TRY_4 exceeded three holdouts")
            cls._recertify_event_artifact_refs(store, events)
            cls._validate_foundation_event(store, events, config)
            cls._recertify_role_timeouts(root, store, events, config.run_id)
            cls._recertify_role_ingresses(root, events, config.run_id)
            fit_set = store.read(
                cast(str, workflow_input.payload["fit_trajectory_set_hash"]), expected_schema_name="FitTrajectorySet"
            )
            teacher_label_set = store.read(
                cast(str, workflow_input.payload["teacher_label_set_hash"]), expected_schema_name="TeacherLabelSet"
            )
            alignment = store.read(
                cast(str, workflow_input.payload["alignment_policy_hash"]), expected_schema_name="AlignmentPolicy"
            )
            base = store.read(
                cast(str, workflow_input.payload["base_judge_prompt_hash"]), expected_schema_name="BaseJudgePrompt"
            )
            attempt = cls._attempt_index(events)
            prompt = cls._current_prompt(store, events, cast(Artifact, source.trace_judge_prompt))
            cls._validate_trace_prompt_chain(
                root,
                store,
                events,
                cast(Artifact, source.trace_judge_prompt),
                cast(str, workflow_input.payload["trace_id"]),
            )
            holdouts = tuple(
                cls._artifact_from_event(store, event, "holdout_set_hash", "HoldoutSet")
                for event in events
                if event.payload.get("event_type") == "HOLDOUT_COMMITTED"
            )
            cls._validate_holdout_lineage(root, store, config, source, foundation, fit_set, alignment, events, holdouts)
            request = cls._optional_artifact_for_attempt(
                store, events, "HOLDOUT_REQUESTED", "holdout_request_hash", "HoldoutGeneratorRequest", attempt
            )
            teacher_packet = cls._optional_artifact_for_attempt(
                store, events, "SCORING_PACKETS_COMMITTED", "teacher_packet_hash", "HoldoutTeacherInputPacket", attempt
            )
            student_packet = cls._optional_artifact_for_attempt(
                store, events, "SCORING_PACKETS_COMMITTED", "student_packet_hash", "HoldoutStudentInputPacket", attempt
            )
            teacher_output, teacher_audit = cls._role_output(
                store, events, "HOLDOUT_TEACHER_OUTPUT_ACCEPTED", attempt, "TeacherScorer"
            )
            student_output, student_audit = cls._role_output(
                store, events, "HOLDOUT_STUDENT_OUTPUT_ACCEPTED", attempt, "StudentJudge"
            )
            auditor_packet = cls._optional_artifact_for_attempt(
                store,
                events,
                "AUDITOR_PACKET_COMMITTED",
                "auditor_packet_hash",
                "CertificationAuditorInputPacket",
                attempt,
            )
            auditor_output, auditor_audit = cls._role_output(
                store, events, "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED", attempt, "AlignmentAuditor"
            )
            optimizer_packet = cls._optional_artifact_for_attempt(
                store,
                events,
                "OPTIMIZER_PACKET_COMMITTED",
                "optimizer_packet_hash",
                "CertificationOptimizerInputPacket",
                attempt,
            )
            optimizer_output, optimizer_audit = cls._role_output(
                store, events, "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED", attempt, "PromptOptimizer"
            )
            terminal_details = cls._terminal_details(events)
            judge_pack = cls._optional_detail_artifact(store, terminal_details, "judge_pack_hash", "JudgePack")
            golden = cls._optional_detail_artifact(store, terminal_details, "golden_hard_entry_hash", "GoldenHardEntry")
            escalation = cls._optional_detail_artifact(
                store, terminal_details, "human_escalation_hash", "HumanEscalation"
            )
            uncertifiable = cls._optional_detail_artifact(
                store, terminal_details, "uncertifiable_report_hash", "UncertifiableJudgeReport"
            )
            if cls._terminal(events):
                cls._recertify_sol_fallback_boundary(root, store, config, teacher_label_set, judge_pack, uncertifiable)
                validation_state = SimpleNamespace(
                    root=root,
                    store=store,
                    config=config,
                    workflow_input=workflow_input,
                    events=events,
                    source_snapshot=source,
                    source_foundation=foundation,
                    trace_id=cast(str, workflow_input.payload["trace_id"]),
                    training_trace_hash=cast(str, workflow_input.payload["training_trace_hash"]),
                    fit_set=fit_set,
                    teacher_label_set=teacher_label_set,
                    alignment_policy=alignment,
                    base_prompt=base,
                    trace_prompt=prompt,
                    holdout_sets=holdouts,
                    teacher_packet=teacher_packet,
                    student_packet=student_packet,
                    teacher_audit=teacher_audit,
                    teacher_output=teacher_output,
                    student_audit=student_audit,
                    student_output=student_output,
                    auditor_packet=auditor_packet,
                    auditor_output=auditor_output,
                    auditor_audit=auditor_audit,
                    optimizer_packet=optimizer_packet,
                    optimizer_output=optimizer_output,
                )
                cls._validate_terminal(
                    cast(Any, validation_state), judge_pack, golden, escalation, uncertifiable, events
                )
        except (
            ArtifactCorruption,
            CertificationRoleOutputError,
            HoldoutBoundaryError,
            Luna4ContractError,
            Luna8WorkflowError,
            SolFallbackBoundaryError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise Luna4WorkflowError("persisted TRY_4 graph cannot be freshly recertified") from error
        return _State(
            root=root,
            store=store,
            journal=journal,
            config=config,
            workflow_input=workflow_input,
            events=events,
            source_snapshot=source,
            source_input=source_input,
            source_foundation=foundation,
            trace_id=cast(str, workflow_input.payload["trace_id"]),
            training_trace_hash=cast(str, workflow_input.payload["training_trace_hash"]),
            fit_set=fit_set,
            teacher_label_set=teacher_label_set,
            alignment_policy=alignment,
            base_prompt=base,
            trace_prompt=prompt,
            holdout_request=request,
            holdout_sets=holdouts,
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
            judge_pack=judge_pack,
            golden_hard_entry=golden,
            human_escalation=escalation,
            uncertifiable_report=uncertifiable,
        )

    @classmethod
    def _load_source(cls, root: Path, config: FixtureLuna4Config) -> tuple[Any, Artifact, dict[str, JsonValue]]:
        source = Luna8CertificationWorkflow.resume(root, config.try8_run_id, epoch=0)
        if (
            not source.terminal
            or source.exhaustion_report is None
            or source.exhaustion_report.content_hash != config.try8_exhaustion_hash
            or source.certification_report is not None
            or len(source.holdout_sets) != 3
            or any(
                item.payload.get("event_type") in {"LUNA8_CERTIFIED", "LUNA4_TERMINAL_PACK_COMMITTED"}
                for item in source.events
            )
        ):
            raise Luna4WorkflowError("TRY_4 source is not a valid sole TRY_8_EXHAUSTED lineage")
        closed_details = cast(dict[str, object], source.events[-1].payload["details"])
        closed = ArtifactStore(root).read(
            cast(str, closed_details["run_closed_hash"]), expected_schema_name="RunClosed"
        )
        if closed.payload.get("reason_code") != "TRY_8_EXHAUSTED" or closed.payload.get("status") != "succeeded":
            raise Luna4WorkflowError("TRY_8 source terminal is not accepted")
        store = ArtifactStore(root)
        source_journal = RunJournal(
            root,
            store,
            config.try8_run_id,
            input_schema_name="Luna8CertificationInput",
            input_schema_version="1.0.0",
        )
        source_input = store.read(source_journal.reserved_input_hash(), expected_schema_name="Luna8CertificationInput")
        foundations = [event for event in source.events if event.payload.get("event_type") == "FOUNDATION_FROZEN"]
        if len(foundations) != 1 or not isinstance(foundations[0].payload.get("details"), dict):
            raise Luna4WorkflowError("TRY_8 foundation is invalid")
        return source, source_input, cast(dict[str, JsonValue], foundations[0].payload["details"])

    @classmethod
    def _validate_holdout_lineage(
        cls,
        root: Path,
        store: ArtifactStore,
        config: FixtureLuna4Config,
        source: Any,
        foundation: dict[str, JsonValue],
        fit_set: Artifact,
        alignment: Artifact,
        events: tuple[Artifact, ...],
        holdouts: tuple[Artifact, ...],
    ) -> None:
        source_hashes = [item.content_hash for item in source.holdout_sets]
        trace_id = cast(str, source.exhaustion_report.payload["trace_id"])
        with Luna8CertificationWorkflow._holdout_ledger_lock(root, trace_id):
            latest = Luna8CertificationWorkflow._read_holdout_ledger_root(root, store, trace_id, allow_legacy=False)
        if holdouts and latest is None:
            raise Luna4WorkflowError("TRY_4 committed holdouts have no cumulative ledger root")
        latest_entries = cast(list[dict[str, JsonValue]], latest.payload["entries"]) if latest else []
        generator = FixtureHoldoutGenerator(root, store, cast(Any, config))
        boundary_state = SimpleNamespace(store=store, fit_set=fit_set)
        plan_events = [event for event in events if event.payload.get("event_type") == "HOLDOUT_PLAN_FROZEN"]
        for attempt, plan_event in enumerate(plan_events, start=1):
            plan_details = cast(dict[str, object], plan_event.payload["details"])
            if (
                set(plan_details) != {"attempt_index", "holdout_plan_hash"}
                or plan_details.get("attempt_index") != attempt
            ):
                raise Luna4WorkflowError("TRY_4 HoldoutPlan event identity changed")
            plan = store.read(cast(str, plan_details["holdout_plan_hash"]), expected_schema_name="HoldoutPlan")
            expected_history = [*source_hashes, *(item.content_hash for item in holdouts[: attempt - 1])]
            expected_plan: dict[str, JsonValue] = {
                "alignment_policy_hash": alignment.content_hash,
                "attempt_id": cls._attempt_id(config.run_id, attempt),
                "attempt_index": attempt,
                "fit_trajectory_set_hash": fit_set.content_hash,
                "generator_inference_hash": foundation["holdout_generator_inference_hash"],
                "generator_model_id": config.holdout_generator_model_id,
                "generator_profile_id": config.holdout_generator_profile_id,
                "historical_holdout_hashes": expected_history,
                "holdout_seed": config.holdout_seed + attempt * 10_003,
                "item_count": 32,
                "items_per_turn": 4,
                "output_fault": config.output_fault,
                "phase": "TRY_4",
                "policy_frozen_before_generation": True,
                "trace_id": trace_id,
            }
            if plan.payload != expected_plan:
                raise Luna4WorkflowError("TRY_4 HoldoutPlan cannot be exactly recertified")
        for attempt, holdout in enumerate(holdouts, start=1):
            refs = holdout.payload.get("item_refs")
            expected_history = [*source_hashes, *(item.content_hash for item in holdouts[: attempt - 1])]
            committed_plan_event = cls._event_for_attempt(events, "HOLDOUT_PLAN_FROZEN", attempt)
            request_event = cls._event_for_attempt(events, "HOLDOUT_REQUESTED", attempt)
            if committed_plan_event is None or request_event is None:
                raise Luna4WorkflowError("TRY_4 plan/request lineage is missing")
            committed_plan_details = cast(dict[str, object], committed_plan_event.payload["details"])
            request_details = cast(dict[str, object], request_event.payload["details"])
            if set(committed_plan_details) != {"attempt_index", "holdout_plan_hash"} or set(request_details) != {
                "attempt_index",
                "boundary_sequence",
                "holdout_request_hash",
            }:
                raise Luna4WorkflowError("TRY_4 plan/request event shape changed")
            request = store.read(
                cast(str, request_details["holdout_request_hash"]), expected_schema_name="HoldoutGeneratorRequest"
            )
            expected_request: dict[str, object] = {
                "attempt_id": cls._attempt_id(config.run_id, attempt),
                "attempt_index": attempt,
                "fit_content_hashes": Luna8CertificationWorkflow._fit_content_hashes(cast(Any, boundary_state)),
                "generator_inference_hash": foundation["holdout_generator_inference_hash"],
                "generator_model_id": config.holdout_generator_model_id,
                "generator_profile_id": config.holdout_generator_profile_id,
                "holdout_seed": config.holdout_seed + attempt * 10_003,
                "output_fault": config.output_fault,
                "policy_hash": alignment.content_hash,
                "prompt": Luna8CertificationWorkflow._task_prompt(cast(Any, boundary_state)),
                "request_schema_version": "holdout-generator-request/1.0.0",
                "trace_id": trace_id,
            }
            if request.payload != expected_request or request_details["boundary_sequence"] != 1:
                raise Luna4WorkflowError("TRY_4 HoldoutGeneratorRequest cannot be exactly recertified")
            if (
                holdout.payload.get("attempt_index") != attempt
                or holdout.payload.get("item_count") != 32
                or holdout.payload.get("historical_holdout_set_hashes") != expected_history
                or not isinstance(refs, list)
                or len(refs) != 32
                or len({cast(str, item["item_hash"]) for item in cast(list[dict[str, object]], refs)}) != 32
            ):
                raise Luna4WorkflowError("TRY_4 holdout cardinality or historical lineage is invalid")
            commit_event = cls._event_for_attempt(events, "HOLDOUT_COMMITTED", attempt)
            if commit_event is None:
                raise Luna4WorkflowError("TRY_4 holdout commit event is missing")
            details = cast(dict[str, object], commit_event.payload["details"])
            ledger = store.read(
                cast(str, details["holdout_ledger_root_hash"]), expected_schema_name="HoldoutLedgerRoot"
            )
            entries = cast(list[dict[str, JsonValue]], ledger.payload["entries"])
            if (
                not entries
                or entries[-1].get("holdout_set_hash") != holdout.content_hash
                or latest_entries[: len(entries)] != entries
            ):
                raise Luna4WorkflowError("TRY_4 holdout ledger root is not in the global committed chain")
            retry_count = sum(
                event.payload.get("event_type") == "HOLDOUT_RETRY_SCHEDULED"
                and isinstance(event.payload.get("details"), dict)
                and cast(dict[str, object], event.payload["details"]).get("attempt_index") == attempt
                for event in events
            )
            chain = generator.verify_attempt_chain(request, retry_count + 1)
            retry_events = [
                event
                for event in events
                if event.payload.get("event_type") == "HOLDOUT_RETRY_SCHEDULED"
                and cast(dict[str, object], event.payload["details"]).get("attempt_index") == attempt
            ]
            for retry_index, retry_event in enumerate(retry_events):
                retry_details = cast(dict[str, object], retry_event.payload["details"])
                boundary_attempt = chain[retry_index]
                if retry_details != {
                    "attempt_index": attempt,
                    "boundary_observation_hash": boundary_attempt.payload["observation_hash"],
                    "boundary_sequence": retry_index + 2,
                    "generator_attempt_hash": boundary_attempt.content_hash,
                }:
                    raise Luna4WorkflowError("TRY_4 retry event no longer binds its boundary evidence")
            lineage = cast(dict[str, object], holdout.payload["generator_lineage"])
            if chain[-1].content_hash != lineage.get("generator_attempt_hash"):
                raise Luna4WorkflowError("TRY_4 holdout boundary attempt lineage changed")
            positions = {
                "plan": events.index(committed_plan_event),
                "request": events.index(request_event),
                "commit": events.index(commit_event),
            }
            retry_positions = [
                index
                for index, event in enumerate(events)
                if event.payload.get("event_type") == "HOLDOUT_RETRY_SCHEDULED"
                and cast(dict[str, object], event.payload["details"]).get("attempt_index") == attempt
            ]
            if not (
                positions["plan"] < positions["request"]
                and all(positions["request"] < index < positions["commit"] for index in retry_positions)
                and positions["request"] < positions["commit"]
            ):
                raise Luna4WorkflowError("TRY_4 boundary journal ordering changed")

    @staticmethod
    def _turns(items: list[Artifact], items_per_turn: int) -> list[dict[str, JsonValue]]:
        result: list[dict[str, JsonValue]] = []
        for offset in range(0, len(items), items_per_turn):
            batch = items[offset : offset + items_per_turn]
            result.append(
                {
                    "items": [
                        {
                            "item_hash": item.content_hash,
                            "prompt": item.payload["prompt"],
                            "response": item.payload["response"],
                        }
                        for item in batch
                    ],
                    "turn_index": len(result) + 1,
                }
            )
        return result

    @staticmethod
    def _teacher_fit_scalars(state: _State) -> list[int]:
        labels = state.teacher_label_set.payload.get("labels")
        if not isinstance(labels, list):
            return []
        return [
            cast(int, item["scalar_micros"])
            for item in labels
            if isinstance(item, dict) and type(item.get("scalar_micros")) is int
        ]

    @classmethod
    def _all_role_outputs(cls, state: _State, event_type: str, role: CertificationRole) -> list[Artifact]:
        result: list[Artifact] = []
        for event in state.events:
            if event.payload.get("event_type") != event_type:
                continue
            details = cast(dict[str, object], event.payload["details"])
            receipt = cls._receipt(cast(dict[str, object], details["role_ingress"]))
            result.append(CertificationRoleIngress.load_public_contract(state.root, receipt).normalized_output)
        return result

    @classmethod
    def _recertify_role_ingresses(cls, root: Path, events: tuple[Artifact, ...], expected_run_id: str) -> None:
        expected: dict[str, CertificationRole] = {
            "HOLDOUT_TEACHER_OUTPUT_ACCEPTED": "TeacherScorer",
            "HOLDOUT_STUDENT_OUTPUT_ACCEPTED": "StudentJudge",
            "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED": "AlignmentAuditor",
            "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED": "PromptOptimizer",
        }
        for event_index, event in enumerate(events):
            role = expected.get(cast(str, event.payload.get("event_type")))
            if role is None:
                continue
            details = cast(dict[str, object], event.payload["details"])
            receipt = cls._receipt(cast(dict[str, object], details["role_ingress"]))
            if receipt.role != role:
                raise Luna4WorkflowError("role event receipt type conflicts")
            public = CertificationRoleIngress.load_public_contract(root, receipt)
            audit = public.role_invocation_audit
            if audit.payload.get("role_lineage_grant_hash") is not None:
                attempt = details.get("attempt_index")
                if type(attempt) is not int or attempt not in {1, 2, 3}:
                    raise Luna4WorkflowError("isolated role ingress attempt is invalid")
                prior_events = events[:event_index]
                timeout_count = sum(
                    prior.payload.get("event_type") == "ROLE_INVOCATION_TIMEOUT_OBSERVED"
                    and isinstance(prior.payload.get("details"), dict)
                    and cast(dict[str, object], prior.payload["details"]).get("attempt_index") == attempt
                    and cast(dict[str, object], prior.payload["details"]).get("role") == role
                    for prior in prior_events
                )
                lineage = audit.payload.get("role_session_lineage")
                if type(lineage) is not str:
                    raise Luna4WorkflowError("isolated role ingress lineage is invalid")
                try:
                    grant = CertificationRoleIngress.validate_isolated_lineage(
                        root,
                        role=role,
                        input_packet_hash=receipt.input_packet_hash,
                        role_session_lineage=lineage,
                        run_id=expected_run_id,
                        attempt_index=attempt,
                        retry_index=timeout_count,
                    )
                except CertificationRoleOutputError as error:
                    raise Luna4WorkflowError(
                        "isolated role ingress lineage grant cannot be freshly recertified"
                    ) from error
                if (
                    audit.payload.get("attempt_index") != attempt
                    or audit.payload.get("retry_index") != timeout_count
                    or audit.payload.get("run_id") != expected_run_id
                    or event.payload.get("run_id") != expected_run_id
                    or audit.payload.get("role_lineage_grant_hash") != grant.content_hash
                ):
                    raise Luna4WorkflowError("isolated role ingress invocation slot cannot be recertified")

    @staticmethod
    def _recertify_event_artifact_refs(store: ArtifactStore, events: tuple[Artifact, ...]) -> None:
        refs: dict[str, tuple[tuple[str, str], ...]] = {
            "TRY4_FOUNDATION_FROZEN": (("decision_record_hash", "DecisionRecord"),),
            "NEXT_CANDIDATE_FROZEN": (("trace_judge_prompt_hash", "TraceJudgePrompt"),),
            "HOLDOUT_PLAN_FROZEN": (("holdout_plan_hash", "HoldoutPlan"),),
            "HOLDOUT_REQUESTED": (("holdout_request_hash", "HoldoutGeneratorRequest"),),
            "HOLDOUT_RETRY_SCHEDULED": (
                ("boundary_observation_hash", "HoldoutBoundaryObservation"),
                ("generator_attempt_hash", "FixtureHoldoutGeneratorAttempt"),
            ),
            "HOLDOUT_COMMITTED": (
                ("holdout_ledger_root_hash", "HoldoutLedgerRoot"),
                ("holdout_set_hash", "HoldoutSet"),
            ),
            "SCORING_PACKETS_COMMITTED": (
                ("student_packet_hash", "HoldoutStudentInputPacket"),
                ("teacher_packet_hash", "HoldoutTeacherInputPacket"),
            ),
            "AUDITOR_PACKET_COMMITTED": (("auditor_packet_hash", "CertificationAuditorInputPacket"),),
            "OPTIMIZER_PACKET_COMMITTED": (("optimizer_packet_hash", "CertificationOptimizerInputPacket"),),
            "LUNA4_TERMINAL_PACK_COMMITTED": (("judge_pack_hash", "JudgePack"),),
            "SOL_FALLBACK_TERMINAL_PACK_COMMITTED": (
                ("golden_hard_entry_hash", "GoldenHardEntry"),
                ("human_escalation_hash", "HumanEscalation"),
                ("judge_pack_hash", "JudgePack"),
            ),
            "UNCERTIFIABLE_TERMINAL": (("uncertifiable_report_hash", "UncertifiableJudgeReport"),),
            "RUN_CLOSED": (("run_closed_hash", "RunClosed"),),
        }
        for event in events:
            event_type = cast(str, event.payload.get("event_type"))
            details = event.payload.get("details")
            if not isinstance(details, dict):
                raise Luna4WorkflowError("TRY_4 event details are malformed")
            for key, schema in refs.get(event_type, ()):
                value = details.get(key)
                if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
                    raise Luna4WorkflowError(f"{event_type} lacks valid {key}")
                store.read(cast(str, value), expected_schema_name=schema)
            if event_type == "TRY4_FAILED_CLOSED" and "failure_report_hash" in details:
                store.read(
                    cast(str, details["failure_report_hash"]), expected_schema_name="RoleBoundaryExhaustionReport"
                )
            if event_type == "TRY4_FAILED_CLOSED" and "boundary_observation_hash" in details:
                store.read(
                    cast(str, details["boundary_observation_hash"]), expected_schema_name="HoldoutBoundaryObservation"
                )
                store.read(
                    cast(str, details["generator_attempt_hash"]),
                    expected_schema_name="FixtureHoldoutGeneratorAttempt",
                )
                store.read(cast(str, details["holdout_request_hash"]), expected_schema_name="HoldoutGeneratorRequest")

    @staticmethod
    def _validate_foundation_event(
        store: ArtifactStore, events: tuple[Artifact, ...], config: FixtureLuna4Config
    ) -> None:
        matches = [event for event in events if event.payload.get("event_type") == "TRY4_FOUNDATION_FROZEN"]
        if not matches:
            return
        if len(matches) != 1:
            raise Luna4WorkflowError("TRY_4 foundation event cardinality changed")
        details = matches[0].payload.get("details")
        if not isinstance(details, dict) or set(details) != {"decision_record_hash"}:
            raise Luna4WorkflowError("TRY_4 foundation event shape changed")
        decision = store.read(cast(str, details["decision_record_hash"]), expected_schema_name="DecisionRecord")
        expected: dict[str, JsonValue] = {
            "decision": "enter_bounded_try4_only_after_verified_try8_exhaustion",
            "items_per_turn": 4,
            "max_attempts": 3,
            "reason_code": "TICKET06_RECOMMENDED_DEFAULTS_ADOPTED",
            "try8_exhaustion_hash": config.try8_exhaustion_hash,
        }
        if decision.payload != expected:
            raise Luna4WorkflowError("TRY_4 DecisionRecord cannot be exactly recertified")

    @classmethod
    def _validate_trace_prompt_chain(
        cls, root: Path, store: ArtifactStore, events: tuple[Artifact, ...], source_prompt: Artifact, trace_id: str
    ) -> None:
        previous = source_prompt
        matches = [event for event in events if event.payload.get("event_type") == "NEXT_CANDIDATE_FROZEN"]
        for candidate, event in enumerate(matches, start=2):
            details = event.payload.get("details")
            if not isinstance(details, dict) or set(details) != {"attempt_index", "trace_judge_prompt_hash"}:
                raise Luna4WorkflowError("TRY_4 TraceJudgePrompt event shape changed")
            prompt = store.read(cast(str, details["trace_judge_prompt_hash"]), expected_schema_name="TraceJudgePrompt")
            optimizer_event = cls._event_for_attempt(events, "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED", candidate - 1)
            if optimizer_event is None:
                raise Luna4WorkflowError("TRY_4 TraceJudgePrompt lacks optimizer lineage")
            optimizer_details = cast(dict[str, object], optimizer_event.payload["details"])
            receipt = cls._receipt(cast(dict[str, object], optimizer_details["role_ingress"]))
            optimizer = CertificationRoleIngress.load_public_contract(root, receipt).normalized_output
            expected: dict[str, JsonValue] = {
                "candidate_index": candidate,
                "parent_prompt_hash": previous.content_hash,
                "prompt_id": f"ticket06-{trace_id[-12:]}-{candidate}",
                "schema_version": "trace-judge-prompt/1.0.0",
                "text": optimizer.payload["candidate_prompt"],
                "trace_id": trace_id,
            }
            if details.get("attempt_index") != candidate or prompt.payload != expected:
                raise Luna4WorkflowError("TRY_4 TraceJudgePrompt derivation cannot be exactly recertified")
            previous = prompt

    @staticmethod
    def _recertify_role_timeouts(
        root: Path, store: ArtifactStore, events: tuple[Artifact, ...], expected_run_id: str
    ) -> None:
        seen: set[tuple[object, object, object]] = set()
        next_retry_by_boundary: dict[tuple[object, object], int] = {}
        for event in events:
            if event.payload.get("event_type") != "ROLE_INVOCATION_TIMEOUT_OBSERVED":
                continue
            details = event.payload.get("details")
            if not isinstance(details, dict) or type(details.get("timeout_audit_hash")) is not str:
                raise Luna4WorkflowError("role timeout event is malformed")
            audit = store.read(
                cast(str, details["timeout_audit_hash"]), expected_schema_name="RoleInvocationTimeoutAudit"
            )
            lineage = audit.payload.get("role_session_lineage")
            event_attempt = details.get("attempt_index")
            timeout_role = cast(CertificationRole, details.get("role"))
            retry_budget = audit.payload.get("retry_budget")
            actual_retry = (
                cast(dict[str, object], retry_budget).get("retry_index")
                if isinstance(retry_budget, dict)
                else audit.payload.get("retry_index", 0)
            )
            boundary = (details.get("attempt_index"), details.get("role"))
            expected_retry = next_retry_by_boundary.get(boundary, 0)
            identity = (*boundary, actual_retry)
            try:
                grant = CertificationRoleIngress.validate_isolated_lineage(
                    root,
                    role=timeout_role,
                    input_packet_hash=cast(str, audit.payload.get("input_packet_hash")),
                    role_session_lineage=cast(str, lineage),
                    run_id=cast(str, audit.payload.get("run_id")),
                    attempt_index=cast(int, event_attempt),
                    retry_index=cast(int, actual_retry),
                )
            except CertificationRoleOutputError as error:
                raise Luna4WorkflowError("role timeout lineage grant cannot be freshly recertified") from error
            if (
                identity in seen
                or audit.payload.get("attempt_index") != details.get("attempt_index")
                or audit.payload.get("role") != details.get("role")
                or audit.payload.get("reason_code") != "ISOLATED_ROLE_TIMEOUT"
                or audit.payload.get("run_id") != expected_run_id
                or type(actual_retry) is not int
                or actual_retry != expected_retry
                or audit.payload.get("role_lineage_grant_hash") != grant.content_hash
                or audit.payload.get("retry_scheduled") is not (expected_retry < 3)
                or audit.payload.get("status") != "retryable_failure"
                or (
                    isinstance(retry_budget, dict) and retry_budget != {"max_retries": 3, "retry_index": expected_retry}
                )
                or details.get("role_session_lineage") not in {None, lineage}
                or details.get("retry_index") not in {None, expected_retry}
            ):
                raise Luna4WorkflowError("role timeout audit cannot be freshly recertified")
            seen.add(identity)
            next_retry_by_boundary[boundary] = expected_retry + 1

    @classmethod
    def _role_output(
        cls, store: ArtifactStore, events: tuple[Artifact, ...], event_type: str, attempt: int, role: CertificationRole
    ) -> tuple[Artifact | None, Artifact | None]:
        event = cls._event_for_attempt(events, event_type, attempt)
        if event is None:
            return None, None
        details = cast(dict[str, object], event.payload["details"])
        receipt = cls._receipt(cast(dict[str, object], details["role_ingress"]))
        return (
            store.read(receipt.normalized_output_hash, expected_schema_name=f"NormalizedCertification{role}Output"),
            store.read(receipt.role_invocation_audit_hash, expected_schema_name="RoleInvocationAudit"),
        )

    @staticmethod
    def _receipt(value: dict[str, object]) -> CertificationRoleIngressReceipt:
        return CertificationRoleIngressReceipt(
            role=cast(CertificationRole, value["role"]),
            ingress_hash=cast(str, value["ingress_hash"]),
            input_packet_hash=cast(str, value["input_packet_hash"]),
            raw_output_hash=cast(str, value["raw_output_hash"]),
            normalized_output_hash=cast(str, value["normalized_output_hash"]),
            role_invocation_audit_hash=cast(str, value["role_invocation_audit_hash"]),
        )

    @classmethod
    def _validate_terminal(
        cls,
        state: _State,
        pack: Artifact | None,
        golden: Artifact | None,
        escalation: Artifact | None,
        uncertifiable: Artifact | None,
        events: tuple[Artifact, ...],
    ) -> None:
        event_types = {cast(str, item.payload["event_type"]) for item in events}
        terminal_markers = [
            item
            for item in events
            if item.payload.get("event_type")
            in {
                "LUNA4_TERMINAL_PACK_COMMITTED",
                "SOL_FALLBACK_TERMINAL_PACK_COMMITTED",
                "UNCERTIFIABLE_TERMINAL",
                "TRY4_FAILED_CLOSED",
            }
        ]
        if len(terminal_markers) != 1:
            raise Luna4WorkflowError("TRY_4 must have exactly one pre-close terminal marker")
        if terminal_markers[0].payload.get("event_type") == "TRY4_FAILED_CLOSED":
            if any(item is not None for item in (pack, golden, escalation, uncertifiable)):
                raise Luna4WorkflowError("failed-closed TRY_4 cannot publish a certification outcome")
            terminal_details = cast(dict[str, object], terminal_markers[0].payload["details"])
            reason = terminal_details.get("reason_code")
            if reason == "ISOLATED_ROLE_BOUNDARY_EXHAUSTED":
                cls._validate_role_boundary_exhaustion(state, terminal_markers[0])
            elif reason == "HOLDOUT_GENERATOR_PERMANENT_FAILURE":
                cls._validate_permanent_holdout_failure(state, terminal_markers[0])
            else:
                raise Luna4WorkflowError("failed-closed TRY_4 terminal reason is unsupported")
            return
        if uncertifiable is not None:
            if (
                pack is not None
                or golden is not None
                or escalation is not None
                or uncertifiable.payload.get("blocks_bundle_publication") is not True
            ):
                raise Luna4WorkflowError("uncertifiable terminal is not fail closed")
            cls._validate_uncertifiable_report(state, uncertifiable)
        elif pack is None:
            raise Luna4WorkflowError("successful TRY_4 terminal has no JudgePack")
        if pack is not None and pack.payload.get("scorer_tier") == "luna":
            if len([item for item in events if item.payload.get("event_type") == "HOLDOUT_COMMITTED"]) > 3:
                raise Luna4WorkflowError("Luna@4 terminal contains hidden continuation")
            if event_types.intersection({"TRY_16", "TRY_32", "LUNA8_CERTIFIED"}):
                raise Luna4WorkflowError("Luna@4 terminal continued to another certification level")
        if pack is not None and pack.payload.get("scorer_tier") == "sol" and (golden is None or escalation is None):
            raise Luna4WorkflowError("Sol fallback lacks Golden Hard or escalation lineage")
        if pack is not None:
            cls._validate_judge_pack_identity(state, pack, golden, escalation)

    @classmethod
    def _validate_uncertifiable_report(cls, state: _State, report: Artifact) -> None:
        evidence = state.store.read(
            cast(str, report.payload.get("sol_fallback_evidence_hash")), expected_schema_name="SolFallbackEvidence"
        )
        reason = evidence.payload.get("reason_code")
        expected: dict[str, object] = {
            "blocks_bundle_publication": True,
            "diagnostic_only": evidence.payload.get("relative_order_available") is True,
            "evidence_tags": [
                "TEACHER_NO_VALID_CALIBRATED_VARIANCE",
                (
                    "SOL_GROUP_RELATIVE_ONLY"
                    if reason == "SOL_GROUP_RELATIVE_ONLY_UNSUPPORTED_V1"
                    else "SOL_CALIBRATED_VARIANCE_INVALID"
                ),
            ],
            "failure_lineage": cls._failed_attempt_lineage(state),
            "reason_code": cast(JsonValue, reason),
            "sol_fallback_evidence_hash": evidence.content_hash,
            "status": "uncertifiable",
            "trace_id": state.trace_id,
            "training_authorized": False,
            "try8_exhaustion_hash": state.config.try8_exhaustion_hash,
        }
        if evidence.payload.get("calibrated_scalar_valid") is not False or report.payload != expected:
            raise Luna4WorkflowError("UncertifiableJudgeReport cannot be exactly recertified")

    @classmethod
    def _validate_judge_pack_identity(
        cls, state: _State, pack: Artifact, golden: Artifact | None, escalation: Artifact | None
    ) -> None:
        attempt = cast(int, pack.payload.get("attempt_index"))
        expected_attempt = cls._attempt_index(state.events)
        if (
            type(attempt) is not int
            or attempt not in {1, 2, 3}
            or attempt != expected_attempt
            or attempt != len(state.holdout_sets)
        ):
            raise Luna4WorkflowError("JudgePack terminal attempt identity is invalid")
        common = cls._judge_pack_identity(state, attempt)
        tier = pack.payload.get("scorer_tier")
        if tier == "luna":
            if None in (state.auditor_output, state.auditor_audit, state.teacher_audit, state.student_audit):
                raise Luna4WorkflowError("Luna@4 terminal role identity is incomplete")
            expected: dict[str, object] = {
                **common,
                "alignment_diagnostics": cast(Artifact, state.auditor_output).payload["aggregate_diagnostics"],
                "certification_level": 4,
                "certification_mode": "student_holdout",
                "holdout_set_hash": state.holdout_sets[-1].content_hash,
                "role_lineage": {
                    "alignment_auditor_audit_hash": cast(Artifact, state.auditor_audit).content_hash,
                    "student_judge_audit_hash": cast(Artifact, state.student_audit).content_hash,
                    "teacher_scorer_audit_hash": cast(Artifact, state.teacher_audit).content_hash,
                },
                "scorer_tier": "luna",
                "status": "terminal",
            }
        elif tier == "sol":
            if golden is None or escalation is None:
                raise Luna4WorkflowError("Sol fallback terminal references are incomplete")
            expected = {
                **common,
                "attempted_holdout_set_hashes": [item.content_hash for item in state.holdout_sets],
                "certification_mode": "teacher_fallback",
                "failure_lineage": cls._failed_attempt_lineage(state),
                "golden_hard_entry_hash": golden.content_hash,
                "human_escalation_hash": escalation.content_hash,
                "scorer_tier": "sol",
                "sol_fallback_evidence_hash": pack.payload["sol_fallback_evidence_hash"],
                "status": "terminal",
            }
        else:
            raise Luna4WorkflowError("JudgePack scorer tier is unsupported")
        if pack.payload != expected:
            raise Luna4WorkflowError("JudgePack immutable identity does not exactly recertify")
        direct_refs = {
            "algorithm_contract_hash": "RLAlgorithmContract",
            "alignment_policy_hash": "AlignmentPolicy",
            "base_judge_prompt_hash": "BaseJudgePrompt",
            "dataset_version_hash": "DatasetVersion",
            "fit_trajectory_set_hash": "FitTrajectorySet",
            "initial_eval_rubric_hash": "InitialEvalRubric",
            "luna_inference_config_hash": "LunaInferenceConfig",
            "reward_schema_hash": "RewardSchema",
            "scalarizer_hash": "Scalarizer",
            "sol_inference_config_hash": "SolInferenceConfig",
            "teacher_calibration_profile_hash": "TeacherCalibrationProfile",
            "teacher_label_set_hash": "TeacherLabelSet",
            "trace_judge_prompt_hash": "TraceJudgePrompt",
            "trace_prompt_derivation_hash": "TracePromptDerivationRecord",
            "training_trace_hash": "TrainingTrace",
            "try8_exhaustion_hash": "Try8ExhaustionReport",
            "try8_source_terminal_event_hash": "RunEvent",
        }
        for key, schema in direct_refs.items():
            state.store.read(cast(str, pack.payload[key]), expected_schema_name=schema)
        if tier == "luna":
            state.store.read(cast(str, pack.payload["holdout_set_hash"]), expected_schema_name="HoldoutSet")
            for audit_hash in cast(dict[str, JsonValue], pack.payload["role_lineage"]).values():
                state.store.read(cast(str, audit_hash), expected_schema_name="RoleInvocationAudit")
        else:
            for holdout_hash in cast(list[str], pack.payload["attempted_holdout_set_hashes"]):
                state.store.read(holdout_hash, expected_schema_name="HoldoutSet")
            for failure in cast(list[dict[str, JsonValue]], pack.payload["failure_lineage"]):
                state.store.read(
                    cast(str, failure["auditor_output_hash"]),
                    expected_schema_name="NormalizedCertificationAlignmentAuditorOutput",
                )
                state.store.read(cast(str, failure["holdout_set_hash"]), expected_schema_name="HoldoutSet")
                state.store.read(cast(str, failure["trace_judge_prompt_hash"]), expected_schema_name="TraceJudgePrompt")
                for key in (
                    "alignment_auditor_audit_hash",
                    "student_judge_audit_hash",
                    "teacher_scorer_audit_hash",
                ):
                    state.store.read(cast(str, failure[key]), expected_schema_name="RoleInvocationAudit")
            failure_lineage = cls._failed_attempt_lineage(state)
            expected_golden: dict[str, object] = {
                "failure_lineage": failure_lineage,
                "reason_tags": cls._failure_reason_tags(state),
                "schema_version": "golden-hard-entry/1.0.0",
                "source_try8_exhaustion_hash": state.config.try8_exhaustion_hash,
                "status": "active",
                "trace_id": state.trace_id,
            }
            expected_escalation: dict[str, object] = {
                "blocking": False,
                "golden_hard_entry_hash": cast(Artifact, golden).content_hash,
                "reason_code": "LUNA8_AND_LUNA4_EXHAUSTED",
                "requested_action": "review_trace_and_future_prompt_strategy",
                "status": "open_nonblocking",
                "trace_id": state.trace_id,
            }
            if (
                cast(Artifact, golden).payload != expected_golden
                or cast(Artifact, escalation).payload != expected_escalation
            ):
                raise Luna4WorkflowError("Sol fallback GoldenHardEntry or HumanEscalation cannot recertify")

    @classmethod
    def _validate_role_boundary_exhaustion(cls, state: _State, terminal_event: Artifact) -> None:
        details = terminal_event.payload.get("details")
        if not isinstance(details, dict) or set(details) != {"failure_report_hash", "reason_code"}:
            raise Luna4WorkflowError("role-boundary terminal details are malformed")
        if details.get("reason_code") != "ISOLATED_ROLE_BOUNDARY_EXHAUSTED":
            raise Luna4WorkflowError("role-boundary terminal reason changed")
        report = state.store.read(
            cast(str, details["failure_report_hash"]), expected_schema_name="RoleBoundaryExhaustionReport"
        )
        role = report.payload.get("role")
        packet_by_role: dict[object, Artifact | None] = {
            "TeacherScorer": state.teacher_packet,
            "StudentJudge": state.student_packet,
            "AlignmentAuditor": state.auditor_packet,
            "PromptOptimizer": state.optimizer_packet,
        }
        packet = packet_by_role.get(role)
        attempt = cls._attempt_index(state.events)
        timeout_hashes: list[str] = []
        retry_indices: list[int] = []
        for event in state.events:
            if event.payload.get("event_type") != "ROLE_INVOCATION_TIMEOUT_OBSERVED":
                continue
            timeout_details = cast(dict[str, object], event.payload["details"])
            if timeout_details.get("role") != role or timeout_details.get("attempt_index") != attempt:
                continue
            timeout_hash = cast(str, timeout_details["timeout_audit_hash"])
            audit = state.store.read(timeout_hash, expected_schema_name="RoleInvocationTimeoutAudit")
            budget = cast(dict[str, object], audit.payload["retry_budget"])
            timeout_hashes.append(timeout_hash)
            retry_indices.append(cast(int, budget["retry_index"]))
        expected: dict[str, object] = {
            "attempt_index": attempt,
            "input_packet_hash": cast(Artifact, packet).content_hash if packet is not None else "",
            "max_retries": 3,
            "reason_code": "ISOLATED_ROLE_BOUNDARY_EXHAUSTED",
            "role": cast(JsonValue, role),
            "status": "failed_closed",
            "timeout_audit_hashes": timeout_hashes,
            "total_invocations": 4,
        }
        if packet is None or retry_indices != [0, 1, 2, 3] or report.payload != expected:
            raise Luna4WorkflowError("role-boundary exhaustion report cannot be freshly recertified")

    @classmethod
    def _validate_permanent_holdout_failure(cls, state: _State, terminal_event: Artifact) -> None:
        details = terminal_event.payload.get("details")
        required = {
            "attempt_index",
            "boundary_observation_hash",
            "boundary_sequence",
            "generator_attempt_hash",
            "holdout_request_hash",
            "reason_code",
        }
        if not isinstance(details, dict) or set(details) != required:
            raise Luna4WorkflowError("permanent holdout failure terminal details are malformed")
        attempt = cls._attempt_index(state.events)
        request_event = cls._event_for_attempt(state.events, "HOLDOUT_REQUESTED", attempt)
        if request_event is None:
            raise Luna4WorkflowError("permanent holdout failure request event is missing")
        request = state.store.read(
            cast(str, details["holdout_request_hash"]), expected_schema_name="HoldoutGeneratorRequest"
        )
        request_details = cast(dict[str, object], request_event.payload["details"])
        retries = [
            event
            for event in state.events
            if event.payload.get("event_type") == "HOLDOUT_RETRY_SCHEDULED"
            and cast(dict[str, object], event.payload["details"]).get("attempt_index") == attempt
        ]
        sequence = len(retries) + 1
        generator = FixtureHoldoutGenerator(state.root, state.store, cast(Any, state.config))
        chain = generator.verify_attempt_chain(request, sequence)
        boundary_attempt = chain[-1]
        observation = state.store.read(
            cast(str, details["boundary_observation_hash"]), expected_schema_name="HoldoutBoundaryObservation"
        )
        if (
            details.get("reason_code") != "HOLDOUT_GENERATOR_PERMANENT_FAILURE"
            or details.get("attempt_index") != attempt
            or details.get("boundary_sequence") != sequence
            or request_details.get("holdout_request_hash") != request.content_hash
            or boundary_attempt.content_hash != details.get("generator_attempt_hash")
            or boundary_attempt.payload.get("observation_hash") != observation.content_hash
            or observation.payload.get("status") != "failed"
            or observation.payload.get("failure_code") != "HOLDOUT_GENERATOR_PERMANENT_FAILURE"
            or observation.payload.get("request_hash") != request.content_hash
        ):
            raise Luna4WorkflowError("permanent holdout boundary failure cannot be freshly recertified")

    @staticmethod
    def _snapshot(state: _State) -> Luna4WorkflowSnapshot:
        return Luna4WorkflowSnapshot(
            events=state.events,
            attempt_index=Luna4CertificationWorkflow._attempt_index(state.events),
            teacher_packet=state.teacher_packet,
            student_packet=state.student_packet,
            auditor_packet=state.auditor_packet,
            optimizer_packet=state.optimizer_packet,
            holdout_sets=state.holdout_sets,
            judge_pack=state.judge_pack,
            golden_hard_entry=state.golden_hard_entry,
            human_escalation=state.human_escalation,
            uncertifiable_report=state.uncertifiable_report,
            terminal=Luna4CertificationWorkflow._terminal(state.events),
        )

    @staticmethod
    def _current_prompt(store: ArtifactStore, events: tuple[Artifact, ...], source: Artifact) -> Artifact:
        matches = [event for event in events if event.payload.get("event_type") == "NEXT_CANDIDATE_FROZEN"]
        if not matches:
            return source
        return Luna4CertificationWorkflow._artifact_from_event(
            store, matches[-1], "trace_judge_prompt_hash", "TraceJudgePrompt"
        )

    @staticmethod
    def _terminal_details(events: tuple[Artifact, ...]) -> dict[str, JsonValue] | None:
        for event in reversed(events):
            if event.payload.get("event_type") in {
                "LUNA4_TERMINAL_PACK_COMMITTED",
                "SOL_FALLBACK_TERMINAL_PACK_COMMITTED",
                "UNCERTIFIABLE_TERMINAL",
            }:
                return cast(dict[str, JsonValue], event.payload["details"])
        return None

    @staticmethod
    def _optional_detail_artifact(
        store: ArtifactStore, details: dict[str, JsonValue] | None, key: str, schema: str
    ) -> Artifact | None:
        if details is None or key not in details:
            return None
        return store.read(cast(str, details[key]), expected_schema_name=schema)

    @staticmethod
    def _optional_artifact_for_attempt(
        store: ArtifactStore,
        events: tuple[Artifact, ...],
        event_type: str,
        key: str,
        schema: str,
        attempt: int,
    ) -> Artifact | None:
        event = Luna4CertificationWorkflow._event_for_attempt(events, event_type, attempt)
        return None if event is None else Luna4CertificationWorkflow._artifact_from_event(store, event, key, schema)

    @classmethod
    def _artifact_for_attempt(cls, state: _State, event_type: str, key: str, schema: str, attempt: int) -> Artifact:
        artifact = cls._optional_artifact_for_attempt(state.store, state.events, event_type, key, schema, attempt)
        if artifact is None:
            raise Luna4WorkflowError(f"{event_type} artifact is unavailable")
        return artifact

    @staticmethod
    def _artifact_from_event(store: ArtifactStore, event: Artifact, key: str, schema: str) -> Artifact:
        details = event.payload.get("details")
        if not isinstance(details, dict) or type(details.get(key)) is not str:
            raise Luna4WorkflowError("event artifact reference is invalid")
        return store.read(cast(str, details[key]), expected_schema_name=schema)

    @staticmethod
    def _event_for_attempt(events: tuple[Artifact, ...], event_type: str, attempt: int) -> Artifact | None:
        matches = [
            event
            for event in events
            if event.payload.get("event_type") == event_type
            and isinstance(event.payload.get("details"), dict)
            and cast(dict[str, object], event.payload["details"]).get("attempt_index") == attempt
        ]
        if len(matches) > 1:
            raise Luna4WorkflowError(f"duplicate {event_type} for attempt {attempt}")
        return matches[0] if matches else None

    @staticmethod
    def _attempt_index(events: tuple[Artifact, ...]) -> int:
        return 1 + sum(event.payload.get("event_type") == "NEXT_CANDIDATE_FROZEN" for event in events)

    @staticmethod
    def _attempt_id(run_id: str, attempt: int) -> str:
        return (
            "a4-"
            + sha256_hex(canonical_json_bytes({"domain": "try4-attempt/1.0.0", "run_id": run_id, "attempt": attempt}))[
                :40
            ]
        )

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
    def _close_failed(state: _State, epoch: int, reason: str, evidence: dict[str, JsonValue] | None = None) -> None:
        event = Luna4CertificationWorkflow._append(
            state,
            epoch,
            "TRY4_FAILED_CLOSED",
            {"reason_code": reason, **(evidence or {})},
        )
        state.journal.close(
            epoch,
            status="failed",
            reason_code=reason,
            expected_sequence=len(state.events) + 2,
            expected_previous_hash=event.content_hash,
        )

    @staticmethod
    def _terminal(events: tuple[Artifact, ...]) -> bool:
        return bool(events and events[-1].payload.get("event_type") == "RUN_CLOSED")

    @staticmethod
    def _artifact_hash(schema_name: str, payload: dict[str, object]) -> str:
        return sha256_hex(
            canonical_json_bytes({"payload": payload, "schema_name": schema_name, "schema_version": "1.0.0"})
        )
