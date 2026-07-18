"""Fenced Ticket 07 Luna@16/@32 expansion workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
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
from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.judge.certification_workflow import Luna8CertificationWorkflow, Luna8WorkflowError
from clawrl.judge.luna_expansion_models import (
    FixtureLunaExpansionConfig,
    LunaExpansionContractError,
    ProductionLunaExpansionConfig,
)
from clawrl.training.run_journal import RunJournal, RunJournalError

_INPUT_SCHEMA = "LunaExpansionInput"
_LEVELS = (16, 32)
_HASH = re.compile(r"^[0-9a-f]{64}$")


class LunaExpansionWorkflowError(RuntimeError):
    """Luna expansion cannot safely advance or recertify."""


@dataclass(frozen=True, slots=True)
class LunaExpansionSnapshot:
    events: tuple[Artifact, ...]
    source_judge_pack: Artifact
    retained_judge_pack: Artifact
    holdout_set: Artifact | None
    holdout_sets: tuple[Artifact, ...]
    teacher_packet: Artifact | None
    student_packet: Artifact | None
    auditor_packet: Artifact | None
    expansion_result: Artifact | None
    level: int
    terminal: bool


@dataclass(frozen=True, slots=True)
class _State:
    root: Path
    store: ArtifactStore
    journal: RunJournal
    config: FixtureLunaExpansionConfig
    workflow_input: Artifact
    source_snapshot: Any
    source_report: Artifact
    source_terminal_event: Artifact
    source_judge_pack: Artifact
    retained_judge_pack: Artifact
    alignment_policy: Artifact
    fit_set: Artifact
    teacher_label_set: Artifact
    base_prompt: Artifact
    trace_prompt: Artifact
    events: tuple[Artifact, ...]
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
    expansion_result: Artifact | None


class LunaExpansionWorkflow:
    """Expand one valid Luna@8 certification through independent 16 and 32 audits."""

    @staticmethod
    def production_readiness(root: str | Path, config: ProductionLunaExpansionConfig) -> Artifact:
        values = (
            ("SOURCE_LUNA8_PACK_APPROVAL_UNAVAILABLE", config.source_luna8_pack_approval_hash),
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
        checks = [
            {"code": code, "status": "blocked"}
            for code, value in values
            if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None
        ]
        if not checks:
            checks.append({"code": "PRODUCTION_EXPANSION_BOUNDARIES_NOT_CONFIGURED", "status": "blocked"})
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": checks,
                "execution_profile": "production",
                "phase": "JUDGE_EXPAND_16_32",
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )

    @classmethod
    def bootstrap(cls, root: str | Path, config: FixtureLunaExpansionConfig, *, epoch: int) -> LunaExpansionSnapshot:
        root_path = Path(root)
        store = ArtifactStore(root_path)
        try:
            source, report, terminal = cls._read_source(root_path, config)
            policy = store.read(config.alignment_policy_hash, expected_schema_name="AlignmentPolicy")
            cls._validate_policy(policy)
            fit_set = store.read(
                cast(str, report.payload["fit_trajectory_set_hash"]), expected_schema_name="FitTrajectorySet"
            )
            teacher_labels = store.read(
                cast(str, report.payload["teacher_label_set_hash"]), expected_schema_name="TeacherLabelSet"
            )
            source_pack_payload = cls._source_pack_payload(store, report, terminal, fit_set, teacher_labels)
            source_pack_hash = cls._artifact_hash("JudgePack", source_pack_payload)
            payload: dict[str, object] = {
                **config.immutable_input_payload,
                "source_judge_pack_hash": source_pack_hash,
                "source_terminal_event_hash": terminal.content_hash,
                "trace_id": report.payload["trace_id"],
            }
            input_hash = cls._artifact_hash(_INPUT_SCHEMA, payload)
            journal = RunJournal(
                root_path, store, config.run_id, input_schema_name=_INPUT_SCHEMA, input_schema_version="1.0.0"
            )
            journal.reserve_identity(input_hash)
        except (
            ArtifactCorruption,
            KeyError,
            Luna8WorkflowError,
            LunaExpansionContractError,
            LunaExpansionWorkflowError,
            RunJournalError,
            TypeError,
            ValueError,
        ) as error:
            raise LunaExpansionWorkflowError("Luna expansion immutable input cannot be verified") from error
        input_path = store.artifact_dir / f"{input_hash}.json"
        if input_path.exists():
            persisted = store.read(input_hash, expected_schema_name=_INPUT_SCHEMA)
            if persisted.payload != payload:
                raise LunaExpansionWorkflowError("persisted expansion input conflicts with reserved identity")
            if journal.events():
                return cls._snapshot(cls._load_state(root_path, config.run_id))
        source_pack = store.put("JudgePack", "1.0.0", source_pack_payload)
        if source_pack.content_hash != source_pack_hash:
            raise LunaExpansionWorkflowError("source Luna@8 pack publication changed identity")
        workflow_input = store.put(_INPUT_SCHEMA, "1.0.0", payload)
        if workflow_input.content_hash != input_hash:
            raise LunaExpansionWorkflowError("expansion input publication changed identity")
        journal.start_run(
            epoch,
            workflow_input.content_hash,
            {
                "input_hash": workflow_input.content_hash,
                "phase": "luna16_32_expansion",
                "source_judge_pack_hash": source_pack.content_hash,
                "source_run_id": config.source_run_id,
                "trace_id": report.payload["trace_id"],
            },
        )
        return cls._snapshot(cls._load_state(root_path, config.run_id))

    @classmethod
    def run_until_role_input(
        cls, root: str | Path, config: FixtureLunaExpansionConfig, *, epoch: int
    ) -> LunaExpansionSnapshot:
        snapshot = cls.bootstrap(root, config, epoch=epoch)
        for _ in range(64):
            last = snapshot.events[-1].payload.get("event_type")
            waiting_for_scores = last in {
                "SCORING_PACKETS_COMMITTED",
                "ROLE_INVOCATION_TIMEOUT_OBSERVED",
                "HOLDOUT_TEACHER_OUTPUT_ACCEPTED",
                "HOLDOUT_STUDENT_OUTPUT_ACCEPTED",
            }
            waiting_for_auditor = last == "AUDITOR_PACKET_COMMITTED"
            if snapshot.terminal or waiting_for_scores or waiting_for_auditor:
                return snapshot
            snapshot = cls.resume(root, config.run_id, epoch=epoch)
        raise LunaExpansionWorkflowError("expansion exceeded bounded transitions")

    @classmethod
    def submit_role_output(
        cls,
        root: str | Path,
        run_id: str,
        *,
        role_ingress: CertificationRoleIngressReceipt,
        epoch: int,
    ) -> LunaExpansionSnapshot:
        state = cls._load_state(Path(root), run_id)
        packet_by_role: dict[CertificationRole, Artifact | None] = {
            "TeacherScorer": state.teacher_packet,
            "StudentJudge": state.student_packet,
            "AlignmentAuditor": state.auditor_packet,
            "PromptOptimizer": None,
        }
        packet = packet_by_role[role_ingress.role]
        if cls._terminal(state.events) or packet is None or packet.content_hash != role_ingress.input_packet_hash:
            raise LunaExpansionWorkflowError("role ingress does not match the current expansion packet")
        try:
            public = CertificationRoleIngress.load_public_contract(state.root, role_ingress)
        except (CertificationRoleOutputError, ArtifactCorruption) as error:
            raise LunaExpansionWorkflowError("expansion role ingress cannot be recertified") from error
        if public.normalized_output.content_hash != role_ingress.normalized_output_hash:
            raise LunaExpansionWorkflowError("role normalized output conflicts with receipt")
        event_type = {
            "TeacherScorer": "HOLDOUT_TEACHER_OUTPUT_ACCEPTED",
            "StudentJudge": "HOLDOUT_STUDENT_OUTPUT_ACCEPTED",
            "AlignmentAuditor": "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED",
        }.get(role_ingress.role)
        if event_type is None:
            raise LunaExpansionWorkflowError("PromptOptimizer is not authorized during expansion")
        level = cls._level(state.events)
        cls._validate_submitted_role_audit(state, role_ingress.role, public.role_invocation_audit, level)
        details = {"level": level, "role_ingress": role_ingress.artifact_payload()}
        existing = cls._event_for_level(state.events, event_type, level)
        if existing is not None:
            if existing.payload.get("details") != details:
                raise LunaExpansionWorkflowError("a different role output is already committed for this level")
            return cls._snapshot(state)
        state.journal.claim_epoch(epoch)
        cls._append(state, epoch, event_type, details)
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def _validate_submitted_role_audit(
        cls, state: _State, role: CertificationRole, audit: Artifact, level: int
    ) -> None:
        grant_hash = audit.payload.get("role_lineage_grant_hash")
        if grant_hash is None:
            if any(key in audit.payload for key in ("attempt_index", "retry_index", "run_id")):
                raise LunaExpansionWorkflowError("fixture role audit contains partial isolated identity")
            return
        retry_index = sum(
            event.payload.get("event_type") == "ROLE_INVOCATION_TIMEOUT_OBSERVED"
            and isinstance(event.payload.get("details"), dict)
            and cast(dict[str, object], event.payload["details"]).get("level") == level
            and cast(dict[str, object], event.payload["details"]).get("role") == role
            for event in state.events
        )
        if (
            audit.payload.get("attempt_index") != cls._attempt(level)
            or audit.payload.get("retry_index") != retry_index
            or audit.payload.get("run_id") != state.config.run_id
            or audit.payload.get("role_type") != role
        ):
            raise LunaExpansionWorkflowError("isolated role audit does not match current fenced retry identity")

    @classmethod
    def record_role_timeout(
        cls,
        root: str | Path,
        run_id: str,
        *,
        role: CertificationRole,
        input_packet_hash: str,
        role_session_lineage: str,
        retry_index: int,
        epoch: int,
    ) -> LunaExpansionSnapshot:
        """Commit a granted isolated-role timeout without fabricating role output."""

        state = cls._load_state(Path(root), run_id)
        packets: dict[CertificationRole, Artifact | None] = {
            "TeacherScorer": state.teacher_packet,
            "StudentJudge": state.student_packet,
            "AlignmentAuditor": state.auditor_packet,
            "PromptOptimizer": None,
        }
        outputs: dict[CertificationRole, Artifact | None] = {
            "TeacherScorer": state.teacher_output,
            "StudentJudge": state.student_output,
            "AlignmentAuditor": state.auditor_output,
            "PromptOptimizer": None,
        }
        packet = packets[role]
        level = cls._level(state.events)
        if (
            cls._terminal(state.events)
            or packet is None
            or packet.content_hash != input_packet_hash
            or outputs[role] is not None
            or retry_index not in {0, 1, 2, 3}
        ):
            raise LunaExpansionWorkflowError("role timeout does not match the current expansion invocation")
        prior = [
            event
            for event in state.events
            if event.payload.get("event_type") == "ROLE_INVOCATION_TIMEOUT_OBSERVED"
            and isinstance(event.payload.get("details"), dict)
            and cast(dict[str, object], event.payload["details"]).get("level") == level
            and cast(dict[str, object], event.payload["details"]).get("role") == role
        ]
        prior_indices = {cast(int, cast(dict[str, object], event.payload["details"])["retry_index"]) for event in prior}
        if prior_indices != set(range(retry_index)) and retry_index not in prior_indices:
            raise LunaExpansionWorkflowError("role timeout retry evidence is out of sequence")
        duplicate = [
            event
            for event in prior
            if cast(dict[str, object], event.payload["details"]).get("retry_index") == retry_index
        ]
        if duplicate:
            duplicate_details = cast(dict[str, object], duplicate[0].payload["details"])
            if (
                len(duplicate) != 1
                or duplicate_details.get("level") != level
                or duplicate_details.get("role") != role
                or duplicate_details.get("role_session_lineage") != role_session_lineage
            ):
                raise LunaExpansionWorkflowError("role timeout conflicts with committed retry evidence")
            return cls._snapshot(state)
        try:
            grant = CertificationRoleIngress.validate_isolated_lineage(
                state.root,
                role=role,
                input_packet_hash=input_packet_hash,
                role_session_lineage=role_session_lineage,
                run_id=run_id,
                attempt_index=cls._attempt(level),
                retry_index=retry_index,
            )
        except CertificationRoleOutputError as error:
            raise LunaExpansionWorkflowError("role timeout lineage was not frozen before dispatch") from error
        audit = state.store.put(
            "RoleInvocationTimeoutAudit",
            "1.0.0",
            {
                "attempt_index": cls._attempt(level),
                "input_packet_hash": input_packet_hash,
                "level": level,
                "reason_code": "ISOLATED_ROLE_TIMEOUT",
                "retry_budget": {"max_retries": 3, "retry_index": retry_index},
                "retry_scheduled": retry_index < 3,
                "role": role,
                "role_lineage_grant_hash": grant.content_hash,
                "role_session_lineage": role_session_lineage,
                "run_id": run_id,
                "status": "exhausted" if retry_index == 3 else "retryable_failure",
            },
        )
        details: dict[str, object] = {
            "level": level,
            "retry_index": retry_index,
            "role": role,
            "role_session_lineage": role_session_lineage,
            "timeout_audit_hash": audit.content_hash,
        }
        state.journal.claim_epoch(epoch)
        cls._append(state, epoch, "ROLE_INVOCATION_TIMEOUT_OBSERVED", details)
        if retry_index == 3:
            exhausted = dataclass_replace(state, events=tuple(state.journal.events()))
            cls._close_failure(exhausted, epoch, "ISOLATED_ROLE_BOUNDARY_EXHAUSTED")
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def resume(cls, root: str | Path, run_id: str, *, epoch: int) -> LunaExpansionSnapshot:
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
            or last == "AUDITOR_PACKET_COMMITTED"
        ):
            return cls._snapshot(state)
        state.journal.claim_epoch(epoch)
        level = cls._level(state.events)
        if last == "RUN_STARTED":
            holdout_config = state.store.put(
                "HoldoutGeneratorInferenceConfig", "1.0.0", state.config.holdout_inference_payload
            )
            decision = state.store.put(
                "DecisionRecord",
                "1.0.0",
                {
                    "decision": "expand_verified_luna8_sequentially_at_16_then_32",
                    "failure_retention": "exact_previous_pack_hash",
                    "levels": [16, 32],
                    "reason_code": "TICKET07_RECOMMENDED_DEFAULTS_ADOPTED",
                    "sol_items_per_turn": 4,
                },
            )
            cls._append(
                state,
                epoch,
                "EXPANSION_FOUNDATION_FROZEN",
                {
                    "alignment_policy_hash": state.alignment_policy.content_hash,
                    "decision_record_hash": decision.content_hash,
                    "holdout_generator_inference_hash": holdout_config.content_hash,
                    "source_judge_pack_hash": state.source_judge_pack.content_hash,
                },
            )
        elif last in {"EXPANSION_FOUNDATION_FROZEN", "LEVEL_CERTIFIED"}:
            cls._freeze_holdout_plan(state, epoch, level)
        elif last == "HOLDOUT_PLAN_FROZEN":
            cls._request_holdout(state, epoch, level)
        elif last in {"HOLDOUT_REQUESTED", "HOLDOUT_RETRY_SCHEDULED"}:
            cls._execute_holdout(state, epoch, level)
        elif last == "HOLDOUT_COMMITTED":
            teacher, student = cls._build_scoring_packets(state, level)
            cls._append(
                state,
                epoch,
                "SCORING_PACKETS_COMMITTED",
                {
                    "level": level,
                    "student_packet_hash": student.content_hash,
                    "teacher_packet_hash": teacher.content_hash,
                },
            )
        elif last in {"HOLDOUT_TEACHER_OUTPUT_ACCEPTED", "HOLDOUT_STUDENT_OUTPUT_ACCEPTED"}:
            if state.teacher_output is None or state.student_output is None:
                return cls._snapshot(state)
            auditor = cls._build_auditor_packet(state, level)
            cls._append(
                state, epoch, "AUDITOR_PACKET_COMMITTED", {"auditor_packet_hash": auditor.content_hash, "level": level}
            )
        elif last == "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED":
            if state.auditor_output is None:
                raise LunaExpansionWorkflowError("accepted auditor output is missing")
            if state.auditor_output.payload.get("verdict") == "pass":
                pack = cls._build_level_pack(state, level)
                event = cls._append(
                    state,
                    epoch,
                    "LEVEL_CERTIFIED",
                    {"judge_pack_hash": pack.content_hash, "level": level},
                )
                if level == 32:
                    cls._close_retained(state, epoch, pack, "LUNA32_CERTIFIED", event.content_hash)
            else:
                cls._close_retained(
                    state,
                    epoch,
                    state.retained_judge_pack,
                    f"LUNA{level}_EXPANSION_FAILED_RETAIN_PREVIOUS",
                    state.events[-1].content_hash,
                )
        else:
            raise LunaExpansionWorkflowError(f"expansion event cannot be advanced: {last}")
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def _freeze_holdout_plan(cls, state: _State, epoch: int, level: int) -> None:
        attempt = cls._attempt(level)
        history = [item.content_hash for item in state.source_snapshot.holdout_sets]
        history.extend(item.content_hash for item in state.holdout_sets)
        attempt_id = (
            "le-"
            + sha256_hex(
                canonical_json_bytes(
                    {
                        "alignment_policy_hash": state.alignment_policy.content_hash,
                        "domain": "luna-expansion-attempt/1.0.0",
                        "level": level,
                        "run_id": state.config.run_id,
                        "source_judge_pack_hash": state.source_judge_pack.content_hash,
                    }
                )
            )[:40]
        )
        plan = state.store.put(
            "HoldoutPlan",
            "1.0.0",
            {
                "alignment_policy_hash": state.alignment_policy.content_hash,
                "attempt_id": attempt_id,
                "attempt_index": attempt,
                "fit_trajectory_set_hash": state.fit_set.content_hash,
                "generator_inference_hash": cls._foundation(state.events)["holdout_generator_inference_hash"],
                "generator_model_id": state.config.holdout_generator_model_id,
                "generator_profile_id": state.config.holdout_generator_profile_id,
                "historical_holdout_hashes": history,
                "holdout_seed": state.config.holdout_seed + attempt * 10_003,
                "item_count": 32,
                "level": level,
                "luna_items_per_turn": level,
                "output_fault": state.config.output_fault,
                "phase": f"TRY_{level}",
                "policy_frozen_before_generation": True,
                "sol_items_per_turn": 4,
                "trace_id": state.source_report.payload["trace_id"],
                "trace_judge_prompt_hash": state.trace_prompt.content_hash,
            },
        )
        cls._append(state, epoch, "HOLDOUT_PLAN_FROZEN", {"holdout_plan_hash": plan.content_hash, "level": level})

    @classmethod
    def _request_holdout(cls, state: _State, epoch: int, level: int) -> None:
        plan_details = cast(dict[str, object], state.events[-1].payload["details"])
        plan = state.store.read(cast(str, plan_details["holdout_plan_hash"]), expected_schema_name="HoldoutPlan")
        request = state.store.put(
            "HoldoutGeneratorRequest",
            "1.0.0",
            {
                "attempt_id": plan.payload["attempt_id"],
                "attempt_index": cls._attempt(level),
                "fit_content_hashes": cls._fit_content_hashes(state),
                "generator_inference_hash": plan.payload["generator_inference_hash"],
                "generator_model_id": state.config.holdout_generator_model_id,
                "generator_profile_id": state.config.holdout_generator_profile_id,
                "holdout_seed": plan.payload["holdout_seed"],
                "output_fault": state.config.output_fault,
                "policy_hash": state.alignment_policy.content_hash,
                "prompt": cls._task_prompt(state),
                "request_schema_version": "holdout-generator-request/1.0.0",
                "trace_id": state.source_report.payload["trace_id"],
            },
        )
        cls._append(
            state,
            epoch,
            "HOLDOUT_REQUESTED",
            {"boundary_sequence": 1, "holdout_request_hash": request.content_hash, "level": level},
        )

    @classmethod
    def _execute_holdout(cls, state: _State, epoch: int, level: int) -> None:
        if state.holdout_request is None:
            raise LunaExpansionWorkflowError("holdout request is unavailable")
        retries = sum(
            event.payload.get("event_type") == "HOLDOUT_RETRY_SCHEDULED"
            and isinstance(event.payload.get("details"), dict)
            and cast(dict[str, object], event.payload["details"]).get("level") == level
            for event in state.events
        )
        sequence = retries + 1
        adapter = FixtureHoldoutGenerator(state.root, state.store, cast(Any, state.config))
        try:
            observation = adapter.execute(state.holdout_request, boundary_sequence=sequence)
            attempts = adapter.verify_attempt_chain(state.holdout_request, sequence)
        except (HoldoutBoundaryError, ArtifactCorruption) as error:
            raise LunaExpansionWorkflowError("holdout boundary evidence is invalid") from error
        if observation.payload.get("status") == "retryable":
            if sequence >= len(state.config.fault_schedule):
                cls._close_failure(state, epoch, "HOLDOUT_RETRY_EXHAUSTED")
                return
            cls._append(
                state,
                epoch,
                "HOLDOUT_RETRY_SCHEDULED",
                {
                    "boundary_sequence": sequence,
                    "failure_code": observation.payload["failure_code"],
                    "generator_attempt_hash": attempts[-1].content_hash,
                    "level": level,
                    "observation_hash": observation.content_hash,
                },
            )
            return
        if observation.payload.get("status") == "failed":
            cls._close_failure(state, epoch, "HOLDOUT_GENERATOR_PERMANENT_FAILURE")
            return
        try:
            raw = adapter.read_raw_batch(observation, state.holdout_request)
            current_plan = cls._event_for_level(state.events, "HOLDOUT_PLAN_FROZEN", level)
            if current_plan is None:
                raise LunaExpansionWorkflowError("current holdout plan disappeared")
            bridge = SimpleNamespace(
                root=state.root,
                store=state.store,
                config=state.config,
                events=(current_plan,),
                alignment_policy=state.alignment_policy,
                fit_set=state.fit_set,
                holdout_request=state.holdout_request,
                trace_id=state.source_report.payload["trace_id"],
            )
            holdout, ledger = Luna8CertificationWorkflow._commit_holdout(
                cast(Any, bridge), raw, observation, attempts[-1], cls._attempt(level)
            )
        except (ArtifactCorruption, HoldoutBoundaryError, Luna8WorkflowError, OSError, TypeError, ValueError):
            cls._close_failure(state, epoch, "HOLDOUT_BATCH_INVALID")
            return
        cls._append(
            state,
            epoch,
            "HOLDOUT_COMMITTED",
            {
                "generator_attempt_hash": attempts[-1].content_hash,
                "holdout_ledger_root_hash": ledger.content_hash,
                "holdout_set_hash": holdout.content_hash,
                "level": level,
                "observation_hash": observation.content_hash,
            },
        )

    @classmethod
    def _build_scoring_packets(cls, state: _State, level: int) -> tuple[Artifact, Artifact]:
        holdout = state.holdout_sets[-1]
        items = Luna8CertificationWorkflow._holdout_items(state.store, holdout)
        teacher_turns = cls._turns(items, 4)
        student_turns = cls._turns(items, level)
        attempt_id = cast(str, holdout.payload["attempt_id"])
        common: dict[str, object] = {
            "aggregation": "calibrated_scalar",
            "attempt_id": attempt_id,
            "attempt_index": cls._attempt(level),
            "holdout_set_hash": holdout.content_hash,
            "reward_schema": cls._artifact_ref(state.store, state.source_report, "reward_schema_hash", "RewardSchema"),
            "scalarizer": cls._artifact_ref(state.store, state.source_report, "scalarizer_hash", "Scalarizer"),
            "trace_id": state.source_report.payload["trace_id"],
        }
        teacher_lineage = cast(dict[str, object], state.teacher_label_set.payload["lineage"])
        rubric = state.store.read(
            cast(str, teacher_lineage["initial_eval_rubric_hash"]), expected_schema_name="InitialEvalRubric"
        )
        sol = state.store.read(
            cast(str, teacher_lineage["sol_inference_config_hash"]), expected_schema_name="SolInferenceConfig"
        )
        teacher_payload: dict[str, object] = {
            **common,
            "initial_eval_rubric": {"artifact_hash": rubric.content_hash, "contract": rubric.payload},
            "items_per_turn": 4,
            "output_schema": output_schema_contract("TeacherScorer", items_per_turn=4),
            "packet_id": f"t07-teacher-{level}-{attempt_id[-16:]}",
            "role": "teacher_scorer",
            "schema_version": "holdout-teacher-input/1.0.0",
            "scoring_session": {
                "items_per_turn": 4,
                "session_id": f"ticket07-sol-{level}-{attempt_id[-20:]}",
                "turn_count": 8,
                "turns": teacher_turns,
            },
            "seed": state.config.role_seed + cls._attempt(level) * 10 + 1,
            "sol_inference": {"artifact_hash": sol.content_hash, "contract": sol.payload},
        }
        teacher_payload["allowlisted_fields"] = sorted([*teacher_payload, "allowlisted_fields"])
        teacher = state.store.put("HoldoutTeacherInputPacket", "1.0.0", teacher_payload)
        luna = state.store.read(
            cast(str, state.source_report.payload["luna_inference_config_hash"]),
            expected_schema_name="LunaInferenceConfig",
        )
        student_payload: dict[str, object] = {
            **common,
            "candidate_prompt": {
                "artifact_hash": state.trace_prompt.content_hash,
                "contract": state.trace_prompt.payload,
            },
            "items_per_turn": level,
            "luna_inference": {"artifact_hash": luna.content_hash, "contract": luna.payload},
            "output_schema": output_schema_contract("StudentJudge", items_per_turn=level),
            "packet_id": f"t07-student-{level}-{attempt_id[-16:]}",
            "role": "student_judge",
            "schema_version": "holdout-student-input/1.0.0",
            "scoring_session": {
                "items_per_turn": level,
                "session_id": f"ticket07-luna-{level}-{attempt_id[-20:]}",
                "turn_count": 32 // level,
                "turns": student_turns,
            },
            "seed": state.config.role_seed + cls._attempt(level) * 10 + 2,
        }
        student_payload["allowlisted_fields"] = sorted([*student_payload, "allowlisted_fields"])
        student = state.store.put("HoldoutStudentInputPacket", "1.0.0", student_payload)
        CertificationRoleIngress.validate_input_packet(state.root, "TeacherScorer", teacher.content_hash)
        CertificationRoleIngress.validate_input_packet(state.root, "StudentJudge", student.content_hash)
        return teacher, student

    @classmethod
    def _build_auditor_packet(cls, state: _State, level: int) -> Artifact:
        if None in (state.teacher_packet, state.student_packet, state.teacher_output, state.student_output):
            raise LunaExpansionWorkflowError("auditor inputs are incomplete")
        teacher_packet = cast(Artifact, state.teacher_packet)
        student_packet = cast(Artifact, state.student_packet)
        item_hashes = cls._packet_item_hashes(teacher_packet)
        if item_hashes != cls._packet_item_hashes(student_packet):
            raise LunaExpansionWorkflowError("Sol and Luna did not score the same holdout item set")
        teacher = cast(Artifact, state.teacher_output)
        student = cast(Artifact, state.student_output)
        attempt_id = cast(str, state.holdout_sets[-1].payload["attempt_id"])
        payload: dict[str, object] = {
            "aggregation": "calibrated_scalar",
            "alignment_policy": state.alignment_policy.payload,
            "alignment_policy_hash": state.alignment_policy.content_hash,
            "attempt_id": attempt_id,
            "attempt_index": cls._attempt(level),
            "comparison": {
                "item_hashes": item_hashes,
                "student_labels": student.payload["labels"],
                "teacher_labels": teacher.payload["labels"],
            },
            "diagnostic_contract": aggregate_diagnostic_contract(),
            "holdout_set_hash": state.holdout_sets[-1].content_hash,
            "level": level,
            "output_schema": output_schema_contract("AlignmentAuditor"),
            "packet_id": f"t07-auditor-{level}-{attempt_id[-16:]}",
            "preregistered_diagnostics": output_schema_contract("AlignmentAuditor")["diagnostic_required"],
            "role": "alignment_auditor",
            "schema_version": "certification-auditor-input/1.0.0",
            "seed": state.config.role_seed + cls._attempt(level) * 10 + 3,
            "session": {"auditor_session_id": f"ticket07-auditor-{level}-{attempt_id[-20:]}"},
        }
        payload["allowlisted_fields"] = sorted([*payload, "allowlisted_fields"])
        packet = state.store.put("CertificationAuditorInputPacket", "1.0.0", payload)
        CertificationRoleIngress.validate_input_packet(state.root, "AlignmentAuditor", packet.content_hash)
        return packet

    @classmethod
    def _build_level_pack(cls, state: _State, level: int) -> Artifact:
        if None in (state.auditor_output, state.auditor_audit, state.teacher_audit, state.student_audit):
            raise LunaExpansionWorkflowError("certified expansion level lacks role evidence")
        if cast(Artifact, state.auditor_output).payload.get("verdict") != "pass":
            raise LunaExpansionWorkflowError("failed expansion cannot publish a JudgePack")
        prior_pack = state.retained_judge_pack
        source = state.source_judge_pack.payload
        payload: dict[str, object] = {
            **{
                key: value
                for key, value in source.items()
                if key
                not in {
                    "alignment_diagnostics",
                    "attempt_id",
                    "attempt_index",
                    "certification_level",
                    "holdout_set_hash",
                    "items_per_turn",
                    "role_lineage",
                    "source_certification_report_hash",
                    "source_terminal_event_hash",
                    "status",
                }
            },
            "alignment_diagnostics": cast(Artifact, state.auditor_output).payload["aggregate_diagnostics"],
            "alignment_policy_hash": state.alignment_policy.content_hash,
            "attempt_id": state.holdout_sets[-1].payload["attempt_id"],
            "attempt_index": cls._attempt(level),
            "certification_level": level,
            "certification_mode": "student_holdout",
            "holdout_set_hash": state.holdout_sets[-1].content_hash,
            "items_per_turn": level,
            "prior_certified_judge_pack_hash": prior_pack.content_hash,
            "role_lineage": {
                "alignment_auditor_audit_hash": cast(Artifact, state.auditor_audit).content_hash,
                "student_judge_audit_hash": cast(Artifact, state.student_audit).content_hash,
                "teacher_scorer_audit_hash": cast(Artifact, state.teacher_audit).content_hash,
            },
            "source_luna8_judge_pack_hash": state.source_judge_pack.content_hash,
            "status": "certified",
        }
        return state.store.put("JudgePack", "1.0.0", payload)

    @classmethod
    def _close_retained(
        cls, state: _State, epoch: int, retained: Artifact, reason_code: str, previous_hash: str
    ) -> None:
        level = cls._level(state.events)
        failed = reason_code.endswith("FAILED_RETAIN_PREVIOUS")
        result = state.store.put(
            "LunaExpansionResult",
            "1.0.0",
            {
                "attempted_levels": [16] if level == 16 else [16, 32],
                "failed_level": level if failed else None,
                "retained_certification_level": retained.payload["certification_level"],
                "retained_judge_pack_hash": retained.content_hash,
                "source_judge_pack_hash": state.source_judge_pack.content_hash,
                "status": "retained_after_failure" if failed else "expanded",
                "terminal_reason_code": reason_code,
                "trace_id": state.source_report.payload["trace_id"],
            },
        )
        sequence = len(state.journal.events()) + 1
        event = state.journal.append(
            epoch,
            "EXPANSION_TERMINAL",
            {
                "expansion_result_hash": result.content_hash,
                "level": level,
                "retained_judge_pack_hash": retained.content_hash,
            },
            expected_sequence=sequence,
            expected_previous_hash=previous_hash,
        )
        state.journal.close(
            epoch,
            status="succeeded",
            reason_code=reason_code,
            expected_sequence=sequence + 1,
            expected_previous_hash=event.content_hash,
        )

    @classmethod
    def _close_failure(cls, state: _State, epoch: int, reason_code: str) -> None:
        report = state.store.put(
            "LunaExpansionFailureReport",
            "1.0.0",
            {
                "level": cls._level(state.events),
                "reason_code": reason_code,
                "retained_judge_pack_hash": state.retained_judge_pack.content_hash,
                "side_effects_permitted": False,
                "status": "failed_closed",
                "trace_id": state.source_report.payload["trace_id"],
            },
        )
        event = cls._append(
            state,
            epoch,
            "EXPANSION_FAILED_CLOSED",
            {"failure_report_hash": report.content_hash, "level": cls._level(state.events), "reason_code": reason_code},
        )
        state.journal.close(
            epoch,
            status="failed",
            reason_code=reason_code,
            expected_sequence=len(state.events) + 2,
            expected_previous_hash=event.content_hash,
        )

    @classmethod
    def _read_source(cls, root: Path, config: FixtureLunaExpansionConfig) -> tuple[Any, Artifact, Artifact]:
        source = Luna8CertificationWorkflow.resume(root, config.source_run_id, epoch=0)
        if (
            not source.terminal
            or source.certification_report is None
            or source.certification_report.content_hash != config.source_certification_report_hash
            or source.certification_report.payload.get("status") != "certified"
            or source.certification_report.payload.get("scorer_tier") != "luna"
            or source.certification_report.payload.get("items_per_turn") != 8
        ):
            raise LunaExpansionWorkflowError("source is not a valid terminal Luna@8 certification")
        if any(
            event.payload.get("event_type")
            in {"TRY_4", "LUNA4_TERMINAL_PACK_COMMITTED", "SOL_FALLBACK_TERMINAL_PACK_COMMITTED"}
            for event in source.events
        ):
            raise LunaExpansionWorkflowError("Luna@4 or Sol fallback cannot enter expansion")
        terminal = source.events[-1]
        if terminal.payload.get("event_type") != "RUN_CLOSED":
            raise LunaExpansionWorkflowError("source terminal event is missing")
        closed_hash = cast(str, cast(dict[str, object], terminal.payload["details"])["run_closed_hash"])
        closed = ArtifactStore(root).read(closed_hash, expected_schema_name="RunClosed")
        if closed.payload.get("status") != "succeeded" or closed.payload.get("reason_code") != "LUNA8_CERTIFIED":
            raise LunaExpansionWorkflowError("source terminal outcome is not accepted")
        return source, source.certification_report, terminal

    @classmethod
    def _source_pack_payload(
        cls, store: ArtifactStore, report: Artifact, terminal: Artifact, fit_set: Artifact, teacher_labels: Artifact
    ) -> dict[str, object]:
        dataset = store.read(cast(str, report.payload["dataset_version_hash"]), expected_schema_name="DatasetVersion")
        teacher_lineage = cast(dict[str, object], teacher_labels.payload["lineage"])
        return {
            **report.payload,
            "certification_level": 8,
            "certification_mode": "student_holdout",
            "dataset_version_id": dataset.payload["dataset_version_id"],
            "initial_eval_rubric_hash": teacher_lineage["initial_eval_rubric_hash"],
            "source_certification_report_hash": report.content_hash,
            "source_terminal_event_hash": terminal.content_hash,
            "status": "certified",
        }

    @classmethod
    def _load_state(cls, root: Path, run_id: str) -> _State:
        store = ArtifactStore(root)
        journal = RunJournal(root, store, run_id, input_schema_name=_INPUT_SCHEMA, input_schema_version="1.0.0")
        try:
            journal.verify()
            workflow_input = store.read(journal.reserved_input_hash(), expected_schema_name=_INPUT_SCHEMA)
            config_fields = set(FixtureLunaExpansionConfig.__dataclass_fields__)  # type: ignore[attr-defined]
            config = FixtureLunaExpansionConfig.from_mapping(
                cast(
                    dict[str, object],
                    {
                        key: value
                        for key, value in workflow_input.payload.items()
                        if key in config_fields or key == "holdout_generator_inference"
                    },
                )
            )
            source, report, terminal = cls._read_source(root, config)
            if terminal.content_hash != workflow_input.payload.get("source_terminal_event_hash"):
                raise LunaExpansionWorkflowError("source terminal identity changed")
            alignment = store.read(config.alignment_policy_hash, expected_schema_name="AlignmentPolicy")
            cls._validate_policy(alignment)
            fit_set = store.read(
                cast(str, report.payload["fit_trajectory_set_hash"]), expected_schema_name="FitTrajectorySet"
            )
            teacher_labels = store.read(
                cast(str, report.payload["teacher_label_set_hash"]), expected_schema_name="TeacherLabelSet"
            )
            base = store.read(
                cast(str, report.payload["base_judge_prompt_hash"]), expected_schema_name="BaseJudgePrompt"
            )
            trace = store.read(
                cast(str, report.payload["trace_judge_prompt_hash"]), expected_schema_name="TraceJudgePrompt"
            )
            source_pack = store.read(
                cast(str, workflow_input.payload["source_judge_pack_hash"]), expected_schema_name="JudgePack"
            )
            if source_pack.payload != cls._source_pack_payload(store, report, terminal, fit_set, teacher_labels):
                raise LunaExpansionWorkflowError("source Luna@8 JudgePack cannot be exactly recertified")
            cls._validate_pack_direct_refs(store, source_pack)
            events = tuple(journal.events())
            if not events:
                raise LunaExpansionWorkflowError("expansion journal has not started")
            cls._validate_event_order(events)
            cls._recertify_event_refs(store, events)
            cls._recertify_timeout_audits(root, store, events, run_id)
            cls._recertify_role_ingresses(root, events, run_id)
            holdouts = tuple(
                store.read(
                    cast(str, cast(dict[str, object], event.payload["details"])["holdout_set_hash"]),
                    expected_schema_name="HoldoutSet",
                )
                for event in events
                if event.payload.get("event_type") == "HOLDOUT_COMMITTED"
            )
            certified_packs = tuple(
                store.read(
                    cast(str, cast(dict[str, object], event.payload["details"])["judge_pack_hash"]),
                    expected_schema_name="JudgePack",
                )
                for event in events
                if event.payload.get("event_type") == "LEVEL_CERTIFIED"
            )
            retained = certified_packs[-1] if certified_packs else source_pack
            for index, pack in enumerate(certified_packs):
                cls._validate_level_pack(
                    store,
                    pack,
                    source_pack=source_pack,
                    prior=source_pack if index == 0 else certified_packs[index - 1],
                    holdout=holdouts[index],
                )
            level = cls._level(events)
            holdout_request = cls._current_artifact(
                store, events, "HOLDOUT_REQUESTED", "holdout_request_hash", "HoldoutGeneratorRequest", level
            )
            teacher_packet = cls._current_artifact(
                store, events, "SCORING_PACKETS_COMMITTED", "teacher_packet_hash", "HoldoutTeacherInputPacket", level
            )
            student_packet = cls._current_artifact(
                store, events, "SCORING_PACKETS_COMMITTED", "student_packet_hash", "HoldoutStudentInputPacket", level
            )
            teacher_output, teacher_audit = cls._role_artifacts(
                store, events, "HOLDOUT_TEACHER_OUTPUT_ACCEPTED", level, "TeacherScorer"
            )
            student_output, student_audit = cls._role_artifacts(
                store, events, "HOLDOUT_STUDENT_OUTPUT_ACCEPTED", level, "StudentJudge"
            )
            auditor_packet = cls._current_artifact(
                store,
                events,
                "AUDITOR_PACKET_COMMITTED",
                "auditor_packet_hash",
                "CertificationAuditorInputPacket",
                level,
            )
            auditor_output, auditor_audit = cls._role_artifacts(
                store, events, "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED", level, "AlignmentAuditor"
            )
            result = cls._last_detail_artifact(
                store, events, "EXPANSION_TERMINAL", "expansion_result_hash", "LunaExpansionResult"
            )
            failure = cls._last_detail_artifact(
                store,
                events,
                "EXPANSION_FAILED_CLOSED",
                "failure_report_hash",
                "LunaExpansionFailureReport",
            )
            if result is not None and failure is not None:
                raise LunaExpansionWorkflowError("expansion has conflicting terminal outcomes")
            if result is not None:
                result_retained = store.read(
                    cast(str, result.payload["retained_judge_pack_hash"]), expected_schema_name="JudgePack"
                )
                if result_retained.content_hash != retained.content_hash:
                    raise LunaExpansionWorkflowError("terminal retention does not preserve the last certified pack")
                cls._validate_result(result, source_pack, retained, events)
            if failure is not None:
                cls._validate_failure(failure, retained, events)
            cls._validate_holdouts(store=store, source=source, events=events, holdouts=holdouts)
        except (
            ArtifactCorruption,
            CertificationRoleOutputError,
            KeyError,
            Luna8WorkflowError,
            LunaExpansionContractError,
            LunaExpansionWorkflowError,
            RunJournalError,
            TypeError,
            ValueError,
        ) as error:
            if isinstance(error, LunaExpansionWorkflowError):
                raise
            raise LunaExpansionWorkflowError("persisted expansion graph cannot be recertified") from error
        return _State(
            root=root,
            store=store,
            journal=journal,
            config=config,
            workflow_input=workflow_input,
            source_snapshot=source,
            source_report=report,
            source_terminal_event=terminal,
            source_judge_pack=source_pack,
            retained_judge_pack=retained,
            alignment_policy=alignment,
            fit_set=fit_set,
            teacher_label_set=teacher_labels,
            base_prompt=base,
            trace_prompt=trace,
            events=events,
            holdout_request=holdout_request,
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
            expansion_result=result,
        )

    @staticmethod
    def _validate_policy(policy: Artifact) -> None:
        required = {
            "max_absolute_bias_micros",
            "max_failure_rate_micros",
            "max_mean_absolute_error_micros",
            "max_p95_absolute_error_micros",
            "max_prompt_attempts",
            "min_pairwise_agreement_micros",
            "min_student_variance_micros",
            "policy_id",
            "required_item_count",
            "schema_version",
        }
        if (
            set(policy.payload) != required
            or any(
                type(value) is not int or not 0 <= cast(int, value) <= 100_000_000
                for key, value in policy.payload.items()
                if key.endswith("_micros")
            )
            or policy.payload.get("required_item_count") != 32
            or policy.payload.get("max_prompt_attempts") != 3
        ):
            raise LunaExpansionWorkflowError("AlignmentPolicy numeric contract is invalid")

    @classmethod
    def _validate_holdouts(
        cls,
        *,
        store: ArtifactStore,
        source: Any,
        events: tuple[Artifact, ...],
        holdouts: tuple[Artifact, ...],
    ) -> None:
        source_hashes = [item.content_hash for item in source.holdout_sets]
        all_hashes = [*source_hashes, *(item.content_hash for item in holdouts)]
        if len(all_hashes) != len(set(all_hashes)):
            raise LunaExpansionWorkflowError("expansion reused a historical holdout")
        historical_items = {
            cast(str, ref["item_hash"])
            for item in source.holdout_sets
            for ref in cast(list[dict[str, object]], item.payload["item_refs"])
        }
        for index, holdout in enumerate(holdouts):
            refs = cast(list[dict[str, object]], holdout.payload.get("item_refs"))
            hashes = [cast(str, ref["item_hash"]) for ref in refs]
            if (
                holdout.payload.get("item_count") != 32
                or len(hashes) != 32
                or len(set(hashes)) != 32
                or historical_items.intersection(hashes)
                or holdout.payload.get("historical_holdout_set_hashes")
                != [*source_hashes, *(item.content_hash for item in holdouts[:index])]
            ):
                raise LunaExpansionWorkflowError("expansion holdout is not fresh, exact, and disjoint")
            historical_items.update(hashes)
            level = _LEVELS[index]
            teacher = cls._current_artifact(
                store, events, "SCORING_PACKETS_COMMITTED", "teacher_packet_hash", "HoldoutTeacherInputPacket", level
            )
            student = cls._current_artifact(
                store, events, "SCORING_PACKETS_COMMITTED", "student_packet_hash", "HoldoutStudentInputPacket", level
            )
            if teacher is not None and student is not None:
                if (
                    teacher.payload.get("items_per_turn") != 4
                    or student.payload.get("items_per_turn") != level
                    or cls._packet_item_hashes(teacher) != hashes
                    or cls._packet_item_hashes(student) != hashes
                ):
                    raise LunaExpansionWorkflowError("expansion scoring packets changed item or batch identity")

    @classmethod
    def _validate_level_pack(
        cls, store: ArtifactStore, pack: Artifact, *, source_pack: Artifact, prior: Artifact, holdout: Artifact
    ) -> None:
        level = pack.payload.get("certification_level")
        if level not in _LEVELS:
            raise LunaExpansionWorkflowError("expanded JudgePack level is invalid")
        if (
            pack.payload.get("prior_certified_judge_pack_hash") != prior.content_hash
            or pack.payload.get("source_luna8_judge_pack_hash") != source_pack.content_hash
            or pack.payload.get("holdout_set_hash") != holdout.content_hash
            or pack.payload.get("items_per_turn") != level
            or pack.payload.get("trace_judge_prompt_hash") != source_pack.payload.get("trace_judge_prompt_hash")
            or pack.payload.get("scorer_tier") != "luna"
            or pack.payload.get("certification_mode") != "student_holdout"
            or pack.payload.get("status") != "certified"
        ):
            raise LunaExpansionWorkflowError("expanded JudgePack immutable identity changed")
        cls._validate_pack_direct_refs(store, pack)
        store.read(cast(str, pack.payload["prior_certified_judge_pack_hash"]), expected_schema_name="JudgePack")
        store.read(cast(str, pack.payload["source_luna8_judge_pack_hash"]), expected_schema_name="JudgePack")
        store.read(cast(str, pack.payload["holdout_set_hash"]), expected_schema_name="HoldoutSet")
        for audit_hash in cast(dict[str, JsonValue], pack.payload["role_lineage"]).values():
            store.read(cast(str, audit_hash), expected_schema_name="RoleInvocationAudit")

    @staticmethod
    def _validate_pack_direct_refs(store: ArtifactStore, pack: Artifact) -> None:
        refs = {
            "algorithm_contract_hash": "RLAlgorithmContract",
            "alignment_policy_hash": "AlignmentPolicy",
            "base_judge_prompt_hash": "BaseJudgePrompt",
            "dataset_version_hash": "DatasetVersion",
            "fit_trajectory_set_hash": "FitTrajectorySet",
            "holdout_set_hash": "HoldoutSet",
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
        }
        for key, schema in refs.items():
            store.read(cast(str, pack.payload[key]), expected_schema_name=schema)

    @classmethod
    def _validate_result(
        cls, result: Artifact, source: Artifact, retained: Artifact, events: tuple[Artifact, ...]
    ) -> None:
        failed_level = result.payload.get("failed_level")
        event_types = [event.payload.get("event_type") for event in events]
        if (
            result.payload.get("source_judge_pack_hash") != source.content_hash
            or result.payload.get("retained_judge_pack_hash") != retained.content_hash
            or result.payload.get("retained_certification_level") != retained.payload.get("certification_level")
            or (failed_level == 16 and event_types.count("HOLDOUT_PLAN_FROZEN") != 1)
            or (failed_level == 16 and retained.content_hash != source.content_hash)
            or (failed_level == 32 and retained.payload.get("certification_level") != 16)
        ):
            raise LunaExpansionWorkflowError("terminal expansion retention cannot be freshly recertified")

    @classmethod
    def _validate_failure(cls, report: Artifact, retained: Artifact, events: tuple[Artifact, ...]) -> None:
        marker = next(event for event in events if event.payload.get("event_type") == "EXPANSION_FAILED_CLOSED")
        details = cast(dict[str, object], marker.payload["details"])
        level = cast(int, details["level"])
        expected: dict[str, object] = {
            "level": level,
            "reason_code": details["reason_code"],
            "retained_judge_pack_hash": retained.content_hash,
            "side_effects_permitted": False,
            "status": "failed_closed",
            "trace_id": retained.payload["trace_id"],
        }
        if report.payload != expected:
            raise LunaExpansionWorkflowError("expansion failure report cannot be freshly recertified")

    @classmethod
    def _validate_event_order(cls, events: tuple[Artifact, ...]) -> None:
        if events[0].payload.get("event_type") != "RUN_STARTED":
            raise LunaExpansionWorkflowError("expansion journal does not start correctly")
        phase = "foundation"
        level = 16
        accepted: set[CertificationRole] = set()
        role_retries: dict[CertificationRole, int] = {
            "TeacherScorer": 0,
            "StudentJudge": 0,
            "AlignmentAuditor": 0,
            "PromptOptimizer": 0,
        }
        boundary_retry = 0
        for index, event in enumerate(events[1:], start=1):
            event_type = event.payload.get("event_type")
            details_value = event.payload.get("details")
            if type(event_type) is not str or not isinstance(details_value, dict):
                raise LunaExpansionWorkflowError("expansion event envelope is invalid")
            details = cast(dict[str, object], details_value)
            if event_type == "RUN_CLOSED":
                if phase != "close" or index != len(events) - 1:
                    raise LunaExpansionWorkflowError("RUN_CLOSED is out of order")
                phase = "closed"
                continue
            if phase in {"close", "closed"}:
                raise LunaExpansionWorkflowError("event follows a terminal expansion marker")
            if phase == "foundation":
                if event_type != "EXPANSION_FOUNDATION_FROZEN":
                    raise LunaExpansionWorkflowError("expansion foundation transition is invalid")
                phase = "plan"
                continue
            if event_type not in {"EXPANSION_FOUNDATION_FROZEN", "RUN_STARTED"} and details.get("level") != level:
                raise LunaExpansionWorkflowError("expansion event belongs to the wrong level")
            if phase == "plan":
                if event_type != "HOLDOUT_PLAN_FROZEN":
                    raise LunaExpansionWorkflowError("holdout plan transition is invalid")
                phase = "request"
            elif phase == "request":
                if event_type != "HOLDOUT_REQUESTED":
                    raise LunaExpansionWorkflowError("holdout request transition is invalid")
                phase = "boundary"
            elif phase == "boundary":
                if event_type == "HOLDOUT_RETRY_SCHEDULED":
                    if details.get("boundary_sequence") != boundary_retry + 1:
                        raise LunaExpansionWorkflowError("holdout retry sequence is invalid")
                    boundary_retry += 1
                elif event_type == "HOLDOUT_COMMITTED":
                    phase = "packets"
                elif event_type == "EXPANSION_FAILED_CLOSED":
                    phase = "close"
                else:
                    raise LunaExpansionWorkflowError("holdout boundary transition is invalid")
            elif phase == "packets":
                if event_type != "SCORING_PACKETS_COMMITTED":
                    raise LunaExpansionWorkflowError("scoring packet transition is invalid")
                phase = "scoring"
            elif phase == "scoring":
                if event_type == "ROLE_INVOCATION_TIMEOUT_OBSERVED":
                    role = cast(CertificationRole, details.get("role"))
                    if role not in {"TeacherScorer", "StudentJudge"} or role in accepted:
                        raise LunaExpansionWorkflowError("scoring timeout role is invalid")
                    if details.get("retry_index") != role_retries[role]:
                        raise LunaExpansionWorkflowError("scoring timeout retry sequence is invalid")
                    role_retries[role] += 1
                    if details.get("retry_index") == 3:
                        phase = "timeout_exhausted"
                elif event_type in {"HOLDOUT_TEACHER_OUTPUT_ACCEPTED", "HOLDOUT_STUDENT_OUTPUT_ACCEPTED"}:
                    role = "TeacherScorer" if event_type.startswith("HOLDOUT_TEACHER") else "StudentJudge"
                    if role in accepted:
                        raise LunaExpansionWorkflowError("scoring role output is duplicated")
                    accepted.add(cast(CertificationRole, role))
                elif event_type == "AUDITOR_PACKET_COMMITTED" and accepted == {
                    "TeacherScorer",
                    "StudentJudge",
                }:
                    phase = "auditor"
                else:
                    raise LunaExpansionWorkflowError("scoring role transition is invalid")
            elif phase == "auditor":
                if event_type == "ROLE_INVOCATION_TIMEOUT_OBSERVED":
                    role = cast(CertificationRole, details.get("role"))
                    if role != "AlignmentAuditor" or role in accepted:
                        raise LunaExpansionWorkflowError("auditor timeout role is invalid")
                    if details.get("retry_index") != role_retries[role]:
                        raise LunaExpansionWorkflowError("auditor timeout retry sequence is invalid")
                    role_retries[role] += 1
                    if details.get("retry_index") == 3:
                        phase = "timeout_exhausted"
                elif event_type == "ALIGNMENT_AUDITOR_OUTPUT_ACCEPTED":
                    if "AlignmentAuditor" in accepted:
                        raise LunaExpansionWorkflowError("auditor output is duplicated")
                    accepted.add("AlignmentAuditor")
                    phase = "outcome"
                else:
                    raise LunaExpansionWorkflowError("auditor transition is invalid")
            elif phase == "timeout_exhausted":
                if event_type != "EXPANSION_FAILED_CLOSED" or details.get("reason_code") != (
                    "ISOLATED_ROLE_BOUNDARY_EXHAUSTED"
                ):
                    raise LunaExpansionWorkflowError("exhausted role boundary did not fail closed")
                phase = "close"
            elif phase == "outcome":
                if event_type == "EXPANSION_TERMINAL":
                    phase = "close"
                elif event_type == "LEVEL_CERTIFIED":
                    if level == 16:
                        level = 32
                        phase = "plan"
                        accepted = set()
                        role_retries = {role: 0 for role in role_retries}
                        boundary_retry = 0
                    else:
                        phase = "expanded_terminal"
                else:
                    raise LunaExpansionWorkflowError("expansion outcome transition is invalid")
            elif phase == "expanded_terminal":
                if event_type != "EXPANSION_TERMINAL":
                    raise LunaExpansionWorkflowError("Luna@32 pass lacks its terminal retained result")
                phase = "close"
            else:
                raise LunaExpansionWorkflowError("unknown expansion state")
        if phase in {"close", "expanded_terminal", "timeout_exhausted"}:
            raise LunaExpansionWorkflowError("expansion journal ends in an unclosed terminal transition")

    @staticmethod
    def _recertify_event_refs(store: ArtifactStore, events: tuple[Artifact, ...]) -> None:
        refs: dict[str, tuple[tuple[str, str], ...]] = {
            "EXPANSION_FOUNDATION_FROZEN": (
                ("alignment_policy_hash", "AlignmentPolicy"),
                ("decision_record_hash", "DecisionRecord"),
                ("holdout_generator_inference_hash", "HoldoutGeneratorInferenceConfig"),
                ("source_judge_pack_hash", "JudgePack"),
            ),
            "HOLDOUT_PLAN_FROZEN": (("holdout_plan_hash", "HoldoutPlan"),),
            "HOLDOUT_REQUESTED": (("holdout_request_hash", "HoldoutGeneratorRequest"),),
            "HOLDOUT_RETRY_SCHEDULED": (
                ("generator_attempt_hash", "FixtureHoldoutGeneratorAttempt"),
                ("observation_hash", "HoldoutBoundaryObservation"),
            ),
            "HOLDOUT_COMMITTED": (
                ("generator_attempt_hash", "FixtureHoldoutGeneratorAttempt"),
                ("holdout_ledger_root_hash", "HoldoutLedgerRoot"),
                ("holdout_set_hash", "HoldoutSet"),
                ("observation_hash", "HoldoutBoundaryObservation"),
            ),
            "SCORING_PACKETS_COMMITTED": (
                ("student_packet_hash", "HoldoutStudentInputPacket"),
                ("teacher_packet_hash", "HoldoutTeacherInputPacket"),
            ),
            "AUDITOR_PACKET_COMMITTED": (("auditor_packet_hash", "CertificationAuditorInputPacket"),),
            "ROLE_INVOCATION_TIMEOUT_OBSERVED": (("timeout_audit_hash", "RoleInvocationTimeoutAudit"),),
            "LEVEL_CERTIFIED": (("judge_pack_hash", "JudgePack"),),
            "EXPANSION_TERMINAL": (
                ("expansion_result_hash", "LunaExpansionResult"),
                ("retained_judge_pack_hash", "JudgePack"),
            ),
            "EXPANSION_FAILED_CLOSED": (("failure_report_hash", "LunaExpansionFailureReport"),),
        }
        for event in events:
            event_type = cast(str, event.payload.get("event_type"))
            details = event.payload.get("details")
            if event_type not in refs:
                continue
            if not isinstance(details, dict):
                raise LunaExpansionWorkflowError("expansion event details are invalid")
            for key, schema in refs[event_type]:
                artifact = store.read(cast(str, details[key]), expected_schema_name=schema)
                if schema == "RoleInvocationTimeoutAudit":
                    store.read(
                        cast(str, artifact.payload["role_lineage_grant_hash"]),
                        expected_schema_name="CertificationRoleLineageGrant",
                    )

    @classmethod
    def _recertify_timeout_audits(
        cls, root: Path, store: ArtifactStore, events: tuple[Artifact, ...], run_id: str
    ) -> None:
        for event in events:
            if event.payload.get("event_type") != "ROLE_INVOCATION_TIMEOUT_OBSERVED":
                continue
            details = cast(dict[str, object], event.payload["details"])
            role = cast(CertificationRole, details["role"])
            level = cast(int, details["level"])
            retry_index = cast(int, details["retry_index"])
            packet_event_type = (
                "AUDITOR_PACKET_COMMITTED" if role == "AlignmentAuditor" else "SCORING_PACKETS_COMMITTED"
            )
            packet_key = {
                "TeacherScorer": "teacher_packet_hash",
                "StudentJudge": "student_packet_hash",
                "AlignmentAuditor": "auditor_packet_hash",
            }.get(role)
            packet_event = cls._event_for_level(events, packet_event_type, level)
            if packet_key is None or packet_event is None:
                raise LunaExpansionWorkflowError("timeout has no current role packet")
            packet_hash = cast(str, cast(dict[str, object], packet_event.payload["details"])[packet_key])
            audit = store.read(
                cast(str, details["timeout_audit_hash"]), expected_schema_name="RoleInvocationTimeoutAudit"
            )
            try:
                grant = CertificationRoleIngress.validate_isolated_lineage(
                    root,
                    role=role,
                    input_packet_hash=packet_hash,
                    role_session_lineage=cast(str, details["role_session_lineage"]),
                    run_id=run_id,
                    attempt_index=cls._attempt(level),
                    retry_index=retry_index,
                )
            except CertificationRoleOutputError as error:
                raise LunaExpansionWorkflowError("timeout lineage grant cannot be freshly recertified") from error
            expected: dict[str, object] = {
                "attempt_index": cls._attempt(level),
                "input_packet_hash": packet_hash,
                "level": level,
                "reason_code": "ISOLATED_ROLE_TIMEOUT",
                "retry_budget": {"max_retries": 3, "retry_index": retry_index},
                "retry_scheduled": retry_index < 3,
                "role": role,
                "role_lineage_grant_hash": grant.content_hash,
                "role_session_lineage": details["role_session_lineage"],
                "run_id": run_id,
                "status": "exhausted" if retry_index == 3 else "retryable_failure",
            }
            if audit.payload != expected:
                raise LunaExpansionWorkflowError("role timeout audit identity changed")

    @classmethod
    def _recertify_role_ingresses(cls, root: Path, events: tuple[Artifact, ...], run_id: str) -> None:
        timeout_counts: dict[tuple[int, CertificationRole], int] = {}
        for event in events:
            details = event.payload.get("details")
            if event.payload.get("event_type") == "ROLE_INVOCATION_TIMEOUT_OBSERVED" and isinstance(details, dict):
                timeout_key = (cast(int, details["level"]), cast(CertificationRole, details["role"]))
                timeout_counts[timeout_key] = timeout_counts.get(timeout_key, 0) + 1
                continue
            if not isinstance(details, dict) or "role_ingress" not in details:
                continue
            payload = cast(dict[str, object], details["role_ingress"])
            role = cast(CertificationRole, payload["role"])
            level = cast(int, details["level"])
            receipt = CertificationRoleIngressReceipt(
                role=role,
                ingress_hash=cast(str, payload["ingress_hash"]),
                input_packet_hash=cast(str, payload["input_packet_hash"]),
                raw_output_hash=cast(str, payload["raw_output_hash"]),
                normalized_output_hash=cast(str, payload["normalized_output_hash"]),
                role_invocation_audit_hash=cast(str, payload["role_invocation_audit_hash"]),
            )
            public = CertificationRoleIngress.load_public_contract(root, receipt)
            audit = public.role_invocation_audit
            grant_hash = audit.payload.get("role_lineage_grant_hash")
            if grant_hash is None:
                if any(key in audit.payload for key in ("attempt_index", "retry_index", "run_id")):
                    raise LunaExpansionWorkflowError("fixture role audit contains partial isolated identity")
                continue
            if (
                audit.payload.get("attempt_index") != cls._attempt(level)
                or audit.payload.get("retry_index") != timeout_counts.get((level, role), 0)
                or audit.payload.get("run_id") != run_id
                or audit.payload.get("role_type") != role
                or audit.payload.get("input_packet_hash") != receipt.input_packet_hash
            ):
                raise LunaExpansionWorkflowError("persisted isolated role audit is not in its fenced retry slot")

    @classmethod
    def _role_artifacts(
        cls,
        store: ArtifactStore,
        events: tuple[Artifact, ...],
        event_type: str,
        level: int,
        role: CertificationRole,
    ) -> tuple[Artifact | None, Artifact | None]:
        event = cls._event_for_level(events, event_type, level)
        if event is None:
            return None, None
        details = cast(dict[str, object], event.payload["details"])
        receipt = cast(dict[str, object], details["role_ingress"])
        if receipt.get("role") != role:
            raise LunaExpansionWorkflowError("role event has the wrong role")
        return (
            store.read(
                cast(str, receipt["normalized_output_hash"]),
                expected_schema_name=f"NormalizedCertification{role}Output",
            ),
            store.read(cast(str, receipt["role_invocation_audit_hash"]), expected_schema_name="RoleInvocationAudit"),
        )

    @staticmethod
    def _snapshot(state: _State) -> LunaExpansionSnapshot:
        return LunaExpansionSnapshot(
            events=state.events,
            source_judge_pack=state.source_judge_pack,
            retained_judge_pack=state.retained_judge_pack,
            holdout_set=state.holdout_sets[-1] if state.holdout_sets else None,
            holdout_sets=state.holdout_sets,
            teacher_packet=state.teacher_packet,
            student_packet=state.student_packet,
            auditor_packet=state.auditor_packet,
            expansion_result=state.expansion_result,
            level=LunaExpansionWorkflow._level(state.events),
            terminal=LunaExpansionWorkflow._terminal(state.events),
        )

    @staticmethod
    def _append(state: _State, epoch: int, event_type: str, details: dict[str, object]) -> Artifact:
        return state.journal.append(
            epoch,
            event_type,
            cast(dict[str, JsonValue], details),
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )

    @staticmethod
    def _terminal(events: tuple[Artifact, ...]) -> bool:
        return bool(events and events[-1].payload.get("event_type") == "RUN_CLOSED")

    @staticmethod
    def _level(events: tuple[Artifact, ...]) -> int:
        certified = sum(event.payload.get("event_type") == "LEVEL_CERTIFIED" for event in events)
        return 32 if certified else 16

    @staticmethod
    def _attempt(level: int) -> int:
        if level not in _LEVELS:
            raise LunaExpansionWorkflowError("expansion level is unsupported")
        return 1 if level == 16 else 2

    @staticmethod
    def _event_for_level(events: tuple[Artifact, ...], event_type: str, level: int) -> Artifact | None:
        matches = [
            event
            for event in events
            if event.payload.get("event_type") == event_type
            and isinstance(event.payload.get("details"), dict)
            and cast(dict[str, object], event.payload["details"]).get("level") == level
        ]
        if len(matches) > 1:
            raise LunaExpansionWorkflowError(f"multiple {event_type} events exist for level {level}")
        return matches[0] if matches else None

    @classmethod
    def _current_artifact(
        cls,
        store: ArtifactStore,
        events: tuple[Artifact, ...],
        event_type: str,
        key: str,
        schema: str,
        level: int,
    ) -> Artifact | None:
        event = cls._event_for_level(events, event_type, level)
        if event is None:
            return None
        details = cast(dict[str, object], event.payload["details"])
        return store.read(cast(str, details[key]), expected_schema_name=schema)

    @staticmethod
    def _last_detail_artifact(
        store: ArtifactStore, events: tuple[Artifact, ...], event_type: str, key: str, schema: str
    ) -> Artifact | None:
        matches = [event for event in events if event.payload.get("event_type") == event_type]
        if len(matches) > 1:
            raise LunaExpansionWorkflowError(f"multiple {event_type} artifacts exist")
        if not matches:
            return None
        return store.read(
            cast(str, cast(dict[str, object], matches[0].payload["details"])[key]), expected_schema_name=schema
        )

    @staticmethod
    def _foundation(events: tuple[Artifact, ...]) -> dict[str, object]:
        matches = [event for event in events if event.payload.get("event_type") == "EXPANSION_FOUNDATION_FROZEN"]
        if len(matches) != 1 or not isinstance(matches[0].payload.get("details"), dict):
            raise LunaExpansionWorkflowError("expansion foundation is unavailable")
        return cast(dict[str, object], matches[0].payload["details"])

    @staticmethod
    def _fit_content_hashes(state: _State) -> list[str]:
        try:
            return Luna8CertificationWorkflow._fit_content_hashes(cast(Any, state))
        except Luna8WorkflowError as error:
            raise LunaExpansionWorkflowError("fit response identities cannot be recertified") from error

    @staticmethod
    def _task_prompt(state: _State) -> str:
        try:
            return Luna8CertificationWorkflow._task_prompt(cast(Any, state))
        except Luna8WorkflowError as error:
            raise LunaExpansionWorkflowError("fit task prompt is unavailable") from error

    @staticmethod
    def _turns(items: list[Artifact], items_per_turn: int) -> list[dict[str, JsonValue]]:
        if len(items) != 32 or items_per_turn not in {4, 16, 32}:
            raise LunaExpansionWorkflowError("scoring turn cardinality is invalid")
        return [
            {
                "items": [
                    {
                        "item_hash": item.content_hash,
                        "item_id": item.payload["item_id"],
                        "prompt": item.payload["prompt"],
                        "response": item.payload["response"],
                    }
                    for item in items[offset : offset + items_per_turn]
                ],
                "turn_index": offset // items_per_turn + 1,
            }
            for offset in range(0, 32, items_per_turn)
        ]

    @staticmethod
    def _packet_item_hashes(packet: Artifact) -> list[str]:
        session = cast(dict[str, object], packet.payload["scoring_session"])
        return [
            cast(str, item["item_hash"])
            for turn in cast(list[dict[str, object]], session["turns"])
            for item in cast(list[dict[str, object]], turn["items"])
        ]

    @staticmethod
    def _artifact_ref(store: ArtifactStore, source: Artifact, key: str, schema: str) -> dict[str, object]:
        artifact = store.read(cast(str, source.payload[key]), expected_schema_name=schema)
        return {"artifact_hash": artifact.content_hash, "contract": artifact.payload}

    @staticmethod
    def _artifact_hash(schema_name: str, payload: dict[str, object]) -> str:
        return sha256_hex(
            canonical_json_bytes({"payload": payload, "schema_name": schema_name, "schema_version": "1.0.0"})
        )
