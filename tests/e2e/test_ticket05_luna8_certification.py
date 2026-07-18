"""Ticket 05 production-shaped Luna@8 certification and explicit exhaustion."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from clawrl.adapters.scorers.certification_roles import (
    CertificationRoleIngress,
    CertificationRoleOutputError,
    fixture_role_lineage,
    required_role_lineage,
)
from clawrl.adapters.scorers.role_isolation import RoleIsolationIngress
from clawrl.adapters.scorers.teacher import TeacherScorerIngress
from clawrl.artifacts import ArtifactStore, canonical_json_bytes, sha256_hex
from clawrl.judge.certification_models import (
    AlignmentPolicy,
    BaseJudgePrompt,
    FixtureLuna8Config,
    LunaInferenceConfig,
    ProductionLuna8Config,
    RewardSchema,
    Scalarizer,
    TraceJudgePrompt,
    derive_fit_calibrated_trace_prompt,
)
from clawrl.judge.certification_workflow import Luna8CertificationWorkflow, Luna8WorkflowError
from clawrl.judge.fit_workflow import FitTrajectoryWorkflow
from clawrl.judge.semantic_diversity import semantic_diversity_summary
from clawrl.training.run_journal import RunJournal, StaleFencingEpoch
from tests.e2e.test_ticket04_teacher_labels import (
    AUDITOR_LINEAGE,
    AUDITOR_RAW_PATH,
    OPTIMIZER_LINEAGE,
    OPTIMIZER_RAW_PATH,
    REAL_RAW_PATH,
    ROLE_LINEAGE,
    _exact_role_raw,
    _phase_a,
)
from tests.fixtures.ticket05_roles import (
    contract_example_auditor_raw,
    contract_example_optimizer_raw,
    contract_example_student_raw,
    contract_example_teacher_raw,
)


def _nested_keys(value: object) -> set[str]:
    keys: set[str] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            keys.update(cast(dict[str, object], item))
            pending.extend(cast(dict[str, object], item).values())
        elif isinstance(item, list):
            pending.extend(item)
    return keys


def _ticket04_terminal(root: Path) -> tuple[str, str, str]:
    config, snapshot = _phase_a(root, "ticket04-for-ticket05")
    assert snapshot.teacher_packet is not None
    teacher = TeacherScorerIngress.stage_private_exchange(
        root,
        input_packet_hash=snapshot.teacher_packet.content_hash,
        raw_role_output=_exact_role_raw(
            REAL_RAW_PATH,
            size=57_040,
            digest="c2bd859c267d6c6e7d21c27d5922709c3c6413f00c5f6e10717378349d90f4cd",
        ),
        role_session_lineage=ROLE_LINEAGE,
    )
    snapshot = FitTrajectoryWorkflow.submit_teacher_output(root, config.run_id, role_ingress=teacher, epoch=4004)
    while snapshot.prompt_optimizer_packet is None:
        snapshot = FitTrajectoryWorkflow.resume(root, config.run_id, epoch=4004)
    assert snapshot.prompt_optimizer_packet is not None and snapshot.alignment_auditor_packet is not None
    optimizer = RoleIsolationIngress.stage_private_exchange(
        root,
        role_type="PromptOptimizer",
        input_packet_hash=snapshot.prompt_optimizer_packet.content_hash,
        raw_role_output=_exact_role_raw(
            OPTIMIZER_RAW_PATH,
            size=2_822,
            digest="7592d1d32e472edd76c1e26ce11d82ee5296546d41f491fc1a65d6a0f02119bc",
        ),
        role_session_lineage=OPTIMIZER_LINEAGE,
    )
    auditor = RoleIsolationIngress.stage_private_exchange(
        root,
        role_type="AlignmentAuditor",
        input_packet_hash=snapshot.alignment_auditor_packet.content_hash,
        raw_role_output=_exact_role_raw(
            AUDITOR_RAW_PATH,
            size=318,
            digest="5329f6646e4d9ab41649c26eeabbca56c87e36a06a64443c28da2b248ac5ede4",
        ),
        role_session_lineage=AUDITOR_LINEAGE,
    )
    FitTrajectoryWorkflow.submit_role_output(root, config.run_id, role_ingress=optimizer, epoch=4004)
    snapshot = FitTrajectoryWorkflow.submit_role_output(root, config.run_id, role_ingress=auditor, epoch=4004)
    while not snapshot.terminal:
        snapshot = FitTrajectoryWorkflow.resume(root, config.run_id, epoch=4004)
    assert snapshot.teacher_label_set is not None
    return config.run_id, snapshot.teacher_label_set.content_hash, config.trace_id


def _config(
    root: Path,
    *,
    run_id: str,
    ticket04_run_id: str,
    teacher_label_set_hash: str,
    trace_id: str,
    fault_schedule: tuple[str, ...] = ("success",),
    output_fault: str | None = None,
) -> FixtureLuna8Config:
    store = ArtifactStore(root)
    base = BaseJudgePrompt(
        prompt_id="fixture-base-judge-prompt-ticket05-v1",
        text=(
            "Evaluate only the supplied task and response using calibrated scalar dimensions; "
            "return concrete response-specific evidence and never infer hidden provenance."
        ),
    )
    base_hash = store.put("BaseJudgePrompt", "1.0.0", base.artifact_payload()).content_hash
    algorithm = store.put(
        "RLAlgorithmContract",
        "1.0.0",
        {
            "advantage_estimator": "grpo",
            "algorithm_id": "fixture-grpo-calibrated-scalar-v1",
            "reward_aggregation": "calibrated_scalar",
            "schema_version": "rl-algorithm-contract/1.0.0",
        },
    )
    label_set = store.read(teacher_label_set_hash, expected_schema_name="TeacherLabelSet")
    optimizer_outputs = [
        store.read(path.stem, expected_schema_name="NormalizedPromptOptimizerOutput")
        for path in store.artifact_dir.glob("*.json")
        if json.loads(path.read_text())["schema_name"] == "NormalizedPromptOptimizerOutput"
    ]
    assert len(optimizer_outputs) == 1
    source_optimizer = optimizer_outputs[0]
    source_diagnostics = cast(dict[str, object], source_optimizer.payload["aggregate_diagnostics"])
    trace_prompt = TraceJudgePrompt(
        prompt_id=f"ticket05-{run_id}-candidate-1",
        trace_id=trace_id,
        candidate_index=1,
        parent_prompt_hash=base_hash,
        text=derive_fit_calibrated_trace_prompt(
            cast(str, source_optimizer.payload["candidate_prompt"]), source_diagnostics
        ),
    )
    trace_prompt_artifact = store.put("TraceJudgePrompt", "1.0.0", trace_prompt.artifact_payload())
    derivation = store.put(
        "TracePromptDerivationRecord",
        "1.0.0",
        {
            "fit_trajectory_set_hash": label_set.payload["fit_trajectory_set_hash"],
            "holdout_refs": [],
            "schema_version": "trace-prompt-derivation/1.0.0",
            "source_base_judge_prompt_hash": base_hash,
            "source_fit_aggregate_hash": sha256_hex(canonical_json_bytes(cast(dict[str, Any], source_diagnostics))),
            "source_prompt_optimizer_output_hash": source_optimizer.content_hash,
            "source_scope": "ticket04_fit_only",
            "teacher_label_set_hash": teacher_label_set_hash,
            "trace_judge_prompt_hash": trace_prompt_artifact.content_hash,
            "transformation": "append_fit_aggregate_calibration_to_isolated_optimizer_candidate",
        },
    )
    return FixtureLuna8Config(
        run_id=run_id,
        ticket04_run_id=ticket04_run_id,
        teacher_label_set_hash=teacher_label_set_hash,
        base_prompt=base,
        trace_prompt=trace_prompt,
        policy=AlignmentPolicy.fixture_default(),
        luna_inference=LunaInferenceConfig(),
        reward_schema=RewardSchema(),
        scalarizer=Scalarizer(),
        algorithm_contract_hash=algorithm.content_hash,
        trace_prompt_derivation_hash=derivation.content_hash,
        holdout_seed=5005,
        role_seed=5505,
        fault_schedule=fault_schedule,
        output_fault=output_fault,
    )


def _submit_scores(root: Path, run_id: str, snapshot: Any, *, student_fault: str | None) -> Any:
    assert snapshot.teacher_packet is not None and snapshot.student_packet is not None
    with pytest.raises(ValueError):
        contract_example_teacher_raw(snapshot.student_packet)
    with pytest.raises(ValueError):
        contract_example_student_raw(snapshot.teacher_packet, calibration_fault=student_fault)
    teacher_raw = contract_example_teacher_raw(snapshot.teacher_packet)
    malformed = json.loads(teacher_raw)
    malformed["labels"][0]["scalar_micros"] += 1
    malformed_raw = canonical_json_bytes(malformed)
    with pytest.raises(CertificationRoleOutputError):
        CertificationRoleIngress.stage_private_exchange(
            root,
            role="TeacherScorer",
            input_packet_hash=snapshot.teacher_packet.content_hash,
            raw_role_output=malformed_raw,
            role_session_lineage=fixture_role_lineage("TeacherScorer"),
            execution_profile="fixture_contract_simulator",
        )
    rejected = list(
        (root / "private-boundary" / "certification-role-rejections" / "TeacherScorer").glob(
            "*/role-output.rejected.raw.json"
        )
    )
    assert any(path.read_bytes() == malformed_raw for path in rejected)
    constant = json.loads(teacher_raw)
    for label in constant["labels"]:
        label["dimension_scores"] = {
            "correctness": 70,
            "reasoning_quality": 70,
            "task_completion": 70,
            "tool_discipline": 70,
        }
        label["scalar_micros"] = 70_000_000
    with pytest.raises(CertificationRoleOutputError):
        CertificationRoleIngress.stage_private_exchange(
            root,
            role="TeacherScorer",
            input_packet_hash=snapshot.teacher_packet.content_hash,
            raw_role_output=canonical_json_bytes(constant),
            role_session_lineage=fixture_role_lineage("TeacherScorer"),
            execution_profile="fixture_contract_simulator",
        )
    copied_evidence = json.loads(teacher_raw)
    repeated = copied_evidence["labels"][0]["evidence"]
    for label in copied_evidence["labels"]:
        label["evidence"] = repeated
    with pytest.raises(CertificationRoleOutputError):
        CertificationRoleIngress.stage_private_exchange(
            root,
            role="TeacherScorer",
            input_packet_hash=snapshot.teacher_packet.content_hash,
            raw_role_output=canonical_json_bytes(copied_evidence),
            role_session_lineage=fixture_role_lineage("TeacherScorer"),
            execution_profile="fixture_contract_simulator",
        )
    with pytest.raises(CertificationRoleOutputError):
        CertificationRoleIngress.stage_private_exchange(
            root,
            role="TeacherScorer",
            input_packet_hash=snapshot.teacher_packet.content_hash,
            raw_role_output=teacher_raw,
            role_session_lineage="/root/shared_scorer",
            execution_profile="isolated_subagent",
        )
    with pytest.raises(CertificationRoleOutputError):
        CertificationRoleIngress.stage_private_exchange(
            root,
            role="TeacherScorer",
            input_packet_hash=snapshot.teacher_packet.content_hash,
            raw_role_output=teacher_raw,
            role_session_lineage=required_role_lineage("TeacherScorer"),
            execution_profile="fixture_contract_simulator",
        )
    with pytest.raises(CertificationRoleOutputError):
        CertificationRoleIngress.stage_private_exchange(
            root,
            role="TeacherScorer",
            input_packet_hash=snapshot.teacher_packet.content_hash,
            raw_role_output=teacher_raw,
            role_session_lineage=fixture_role_lineage("TeacherScorer"),
            execution_profile="isolated_subagent",
        )
    teacher = CertificationRoleIngress.stage_private_exchange(
        root,
        role="TeacherScorer",
        input_packet_hash=snapshot.teacher_packet.content_hash,
        raw_role_output=teacher_raw,
        role_session_lineage=fixture_role_lineage("TeacherScorer"),
        execution_profile="fixture_contract_simulator",
    )
    student = CertificationRoleIngress.stage_private_exchange(
        root,
        role="StudentJudge",
        input_packet_hash=snapshot.student_packet.content_hash,
        raw_role_output=contract_example_student_raw(snapshot.student_packet, calibration_fault=student_fault),
        role_session_lineage=fixture_role_lineage("StudentJudge"),
        execution_profile="fixture_contract_simulator",
    )
    # Deliberately commit out of order; the journal waits for both exact packet identities.
    Luna8CertificationWorkflow.submit_role_output(root, run_id, role_ingress=student, epoch=5005)
    snapshot = Luna8CertificationWorkflow.submit_role_output(root, run_id, role_ingress=teacher, epoch=5005)
    snapshot = Luna8CertificationWorkflow.resume(root, run_id, epoch=5005)
    assert snapshot.auditor_packet is not None
    auditor_raw = contract_example_auditor_raw(snapshot.auditor_packet)
    invalid_auditor = json.loads(auditor_raw)
    invalid_auditor["aggregate_diagnostics"]["mean_absolute_error_micros"] += 1
    with pytest.raises(CertificationRoleOutputError):
        CertificationRoleIngress.stage_private_exchange(
            root,
            role="AlignmentAuditor",
            input_packet_hash=snapshot.auditor_packet.content_hash,
            raw_role_output=canonical_json_bytes(invalid_auditor),
            role_session_lineage=fixture_role_lineage("AlignmentAuditor"),
            execution_profile="fixture_contract_simulator",
        )
    auditor = CertificationRoleIngress.stage_private_exchange(
        root,
        role="AlignmentAuditor",
        input_packet_hash=snapshot.auditor_packet.content_hash,
        raw_role_output=auditor_raw,
        role_session_lineage=fixture_role_lineage("AlignmentAuditor"),
        execution_profile="fixture_contract_simulator",
    )
    return Luna8CertificationWorkflow.submit_role_output(root, run_id, role_ingress=auditor, epoch=5005)


def test_first_candidate_certifies_and_three_failed_candidates_exhaust_try8(tmp_path: Path) -> None:
    ticket04_run, label_hash, trace_id = _ticket04_terminal(tmp_path)

    success_config = _config(
        tmp_path,
        run_id="ticket05-first-success",
        ticket04_run_id=ticket04_run,
        teacher_label_set_hash=label_hash,
        trace_id=trace_id,
        fault_schedule=("timeout", "delayed", "success"),
    )
    store = ArtifactStore(tmp_path)
    derivation = store.read(
        cast(str, success_config.trace_prompt_derivation_hash),
        expected_schema_name="TracePromptDerivationRecord",
    )
    leaked_payload = dict(derivation.payload)
    leaked_payload["holdout_refs"] = [sha256_hex(b"forbidden-holdout-ref")]
    leaked = store.put("TracePromptDerivationRecord", "1.0.0", leaked_payload)
    leaked_run_id = "ticket05-holdout-leaking-prompt-rejected"
    with pytest.raises(Luna8WorkflowError):
        Luna8CertificationWorkflow.run_until_role_input(
            tmp_path,
            replace(
                success_config,
                run_id=leaked_run_id,
                trace_prompt_derivation_hash=leaked.content_hash,
            ),
            epoch=5005,
        )
    leaked_events = RunJournal(
        tmp_path,
        store,
        leaked_run_id,
        input_schema_name="Luna8CertificationInput",
        input_schema_version="1.0.0",
    ).events()
    assert [event.payload["event_type"] for event in leaked_events] == ["RUN_STARTED"]

    incompatible_algorithm = store.put(
        "RLAlgorithmContract",
        "1.0.0",
        {
            "advantage_estimator": "grpo",
            "algorithm_id": "fixture-incompatible-hierarchical-v1",
            "reward_aggregation": "hierarchical",
            "schema_version": "rl-algorithm-contract/1.0.0",
        },
    )
    incompatible_run_id = "ticket05-incompatible-algorithm-rejected"
    with pytest.raises(Luna8WorkflowError):
        Luna8CertificationWorkflow.bootstrap(
            tmp_path,
            replace(
                success_config,
                run_id=incompatible_run_id,
                algorithm_contract_hash=incompatible_algorithm.content_hash,
            ),
            epoch=5005,
        )
    assert not (tmp_path / "runs" / incompatible_run_id).exists()

    success = Luna8CertificationWorkflow.run_until_role_input(tmp_path, success_config, epoch=5005)
    assert success.attempt_index == 1
    assert success.teacher_packet is not None and success.student_packet is not None
    assert success.holdout_set is not None and success.holdout_set.payload["item_count"] == 32
    event_types = [event.payload["event_type"] for event in success.events]
    assert event_types.index("FOUNDATION_FROZEN") < event_types.index("HOLDOUT_REQUESTED")
    teacher_hashes = _packet_item_hashes(success.teacher_packet)
    student_hashes = _packet_item_hashes(success.student_packet)
    assert teacher_hashes == student_hashes and len(set(teacher_hashes)) == 32
    assert success.teacher_packet.payload["items_per_turn"] == 8
    assert success.student_packet.payload["items_per_turn"] == 8
    assert success.teacher_packet.payload["aggregation"] == "calibrated_scalar"
    assert success.student_packet.payload["aggregation"] == "calibrated_scalar"
    assert "generator" not in canonical_json_bytes(success.teacher_packet.payload).decode().lower()
    student_text = canonical_json_bytes(success.student_packet.payload).decode().lower()
    assert all(
        token not in student_text
        for token in ("teacher_label_set", "teacher_labels", "sol_inference", "alignment_policy")
    )
    success = _submit_scores(tmp_path, success_config.run_id, success, student_fault=None)
    success = Luna8CertificationWorkflow.resume(tmp_path, success_config.run_id, epoch=5005)
    assert success.terminal and success.certification_report is not None
    report = success.certification_report.payload
    expected_identity_fields = {
        "aggregation",
        "algorithm_contract_hash",
        "alignment_policy_hash",
        "attempt_id",
        "attempt_index",
        "base_judge_prompt_hash",
        "dataset_version_hash",
        "fit_trajectory_set_hash",
        "holdout_set_hash",
        "items_per_turn",
        "luna_inference_config_hash",
        "reward_schema_hash",
        "scalarizer_hash",
        "sol_inference_config_hash",
        "teacher_calibration_profile_hash",
        "teacher_label_set_hash",
        "trace_id",
        "trace_judge_prompt_hash",
        "trace_prompt_derivation_hash",
        "training_trace_hash",
    }
    assert expected_identity_fields <= set(report)
    assert report["status"] == "certified" and report["scorer_tier"] == "luna"

    with pytest.raises(Luna8WorkflowError):
        Luna8CertificationWorkflow.bootstrap(
            tmp_path,
            replace(success_config, holdout_seed=success_config.holdout_seed + 1),
            epoch=5005,
        )

    fenced_config = _config(
        tmp_path,
        run_id="ticket05-fencing",
        ticket04_run_id=ticket04_run,
        teacher_label_set_hash=label_hash,
        trace_id=trace_id,
    )
    Luna8CertificationWorkflow.bootstrap(tmp_path, fenced_config, epoch=5005)
    Luna8CertificationWorkflow.resume(tmp_path, fenced_config.run_id, epoch=6000)
    with pytest.raises(StaleFencingEpoch):
        Luna8CertificationWorkflow.resume(tmp_path, fenced_config.run_id, epoch=5005)

    exhausted_config = _config(
        tmp_path,
        run_id="ticket05-three-fail",
        ticket04_run_id=ticket04_run,
        teacher_label_set_hash=label_hash,
        trace_id=trace_id,
    )
    exhausted = Luna8CertificationWorkflow.run_until_role_input(tmp_path, exhausted_config, epoch=5005)
    assert success.holdout_set is not None and exhausted.holdout_set is not None
    assert success.holdout_set.content_hash in cast(
        list[str], exhausted.holdout_set.payload["historical_holdout_set_hashes"]
    )
    for attempt_index in range(1, 4):
        assert exhausted.attempt_index == attempt_index
        exhausted = _submit_scores(
            tmp_path,
            exhausted_config.run_id,
            exhausted,
            student_fault="polarity_inversion",
        )
        exhausted = Luna8CertificationWorkflow.resume(tmp_path, exhausted_config.run_id, epoch=5005)
        if attempt_index < 3:
            assert exhausted.optimizer_packet is not None
            optimizer_run_token = sha256_hex(
                canonical_json_bytes(
                    {
                        "domain": "certification-optimizer-session/1.0.0",
                        "run_id": exhausted_config.run_id,
                    }
                )
            )[:12]
            assert cast(str, exhausted.optimizer_packet.payload["packet_id"]).endswith(f"-{optimizer_run_token}")
            assert cast(
                str,
                cast(dict[str, object], exhausted.optimizer_packet.payload["session"])["optimizer_session_id"],
            ).endswith(f"-{optimizer_run_token}")
            optimizer_keys = _nested_keys(exhausted.optimizer_packet.payload)
            assert optimizer_keys.isdisjoint(
                {
                    "holdout_set_hash",
                    "item_hash",
                    "item_hashes",
                    "student_label",
                    "student_labels",
                    "teacher_label",
                    "teacher_labels",
                }
            )
            optimizer = CertificationRoleIngress.stage_private_exchange(
                tmp_path,
                role="PromptOptimizer",
                input_packet_hash=exhausted.optimizer_packet.content_hash,
                raw_role_output=contract_example_optimizer_raw(exhausted.optimizer_packet),
                role_session_lineage=fixture_role_lineage("PromptOptimizer"),
                execution_profile="fixture_contract_simulator",
            )
            exhausted = Luna8CertificationWorkflow.submit_role_output(
                tmp_path, exhausted_config.run_id, role_ingress=optimizer, epoch=5005
            )
            exhausted = Luna8CertificationWorkflow.resume(tmp_path, exhausted_config.run_id, epoch=5005)
            exhausted = Luna8CertificationWorkflow.run_until_role_input(tmp_path, exhausted_config, epoch=5005)
    assert exhausted.terminal and exhausted.exhaustion_report is not None
    assert exhausted.exhaustion_report.payload["reason_code"] == "TRY_8_EXHAUSTED"
    assert exhausted.exhaustion_report.payload["attempt_count"] == 3
    assert len(exhausted.holdout_sets) == 3
    all_holdout_hashes = [
        cast(str, ref["item_hash"])
        for holdout in (success.holdout_sets + exhausted.holdout_sets)
        for ref in cast(list[dict[str, object]], holdout.payload["item_refs"])
    ]
    assert len(all_holdout_hashes) == 4 * 32
    assert len(set(all_holdout_hashes)) == 4 * 32
    closed = [event for event in exhausted.events if event.payload["event_type"] == "RUN_CLOSED"]
    assert len(closed) == 1
    assert sum(event.payload["event_type"] == "HOLDOUT_REQUESTED" for event in exhausted.events) == 3

    first_teacher_event = next(
        event
        for event in exhausted.events
        if event.payload["event_type"] == "HOLDOUT_TEACHER_OUTPUT_ACCEPTED"
        and cast(dict[str, object], event.payload["details"])["attempt_index"] == 1
    )
    first_teacher_receipt = cast(
        dict[str, object], cast(dict[str, object], first_teacher_event.payload["details"])["role_ingress"]
    )
    first_teacher_raw_path = (
        tmp_path
        / "private-boundary"
        / "certification-role-ingress"
        / "TeacherScorer"
        / cast(str, first_teacher_receipt["ingress_hash"])
        / "role-output.raw.json"
    )
    first_teacher_raw = first_teacher_raw_path.read_bytes()
    first_teacher_raw_path.write_bytes(b"{}")
    try:
        with pytest.raises(Luna8WorkflowError):
            Luna8CertificationWorkflow.resume(tmp_path, exhausted_config.run_id, epoch=9999)
    finally:
        first_teacher_raw_path.write_bytes(first_teacher_raw)
    assert Luna8CertificationWorkflow.resume(tmp_path, exhausted_config.run_id, epoch=9999).terminal

    source_event = next(
        event for event in exhausted.events if event.payload["event_type"] == "PROMPT_OPTIMIZER_OUTPUT_ACCEPTED"
    )
    source_receipt = cast(dict[str, object], cast(dict[str, object], source_event.payload["details"])["role_ingress"])
    source_foundation = next(event for event in exhausted.events if event.payload["event_type"] == "FOUNDATION_FROZEN")
    source_foundation_details = cast(dict[str, object], source_foundation.payload["details"])
    source_output = store.read(
        cast(str, source_receipt["normalized_output_hash"]),
        expected_schema_name="NormalizedCertificationPromptOptimizerOutput",
    )
    base_hash = store.put("BaseJudgePrompt", "1.0.0", exhausted_config.base_prompt.artifact_payload()).content_hash
    replication_trace = TraceJudgePrompt(
        prompt_id="ticket05-preoptimized-replication-candidate-1",
        trace_id=trace_id,
        candidate_index=1,
        parent_prompt_hash=base_hash,
        text=cast(str, source_output.payload["candidate_prompt"]),
    )
    replication_trace_artifact = store.put("TraceJudgePrompt", "1.0.0", replication_trace.artifact_payload())
    replication_derivation_payload = {
        "fit_trajectory_set_hash": store.read(label_hash, expected_schema_name="TeacherLabelSet").payload[
            "fit_trajectory_set_hash"
        ],
        "holdout_refs": [],
        "schema_version": "trace-prompt-derivation/1.1.0",
        "source_attempt_index": 1,
        "source_alignment_policy_hash": source_foundation_details["alignment_policy_hash"],
        "source_base_judge_prompt_hash": base_hash,
        "source_optimizer_acceptance_event_hash": source_event.content_hash,
        "source_optimizer_packet_hash": source_receipt["input_packet_hash"],
        "source_optimizer_role_audit_hash": source_receipt["role_invocation_audit_hash"],
        "source_prompt_optimizer_output_hash": source_output.content_hash,
        "source_run_id": exhausted_config.run_id,
        "source_scope": "prior_certification_aggregate_only",
        "teacher_label_set_hash": label_hash,
        "trace_judge_prompt_hash": replication_trace_artifact.content_hash,
        "transformation": "reuse_prior_aggregate_only_optimizer_candidate",
    }
    replication_derivation = store.put("TracePromptDerivationRecord", "1.1.0", replication_derivation_payload)
    tampered_derivation = store.put(
        "TracePromptDerivationRecord",
        "1.1.0",
        {
            **replication_derivation_payload,
            "source_optimizer_acceptance_event_hash": exhausted.events[0].content_hash,
        },
    )
    with pytest.raises(Luna8WorkflowError):
        Luna8CertificationWorkflow.run_until_role_input(
            tmp_path,
            replace(
                exhausted_config,
                run_id="ticket05-preoptimized-tampered-provenance",
                trace_prompt=replication_trace,
                trace_prompt_derivation_hash=tampered_derivation.content_hash,
                holdout_seed=6006,
                role_seed=6505,
            ),
            epoch=5005,
        )
    replication_config = replace(
        exhausted_config,
        run_id="ticket05-preoptimized-replication",
        trace_prompt=replication_trace,
        trace_prompt_derivation_hash=replication_derivation.content_hash,
        holdout_seed=6006,
        role_seed=6505,
    )
    replication = Luna8CertificationWorkflow.run_until_role_input(tmp_path, replication_config, epoch=5005)
    assert replication.holdout_set is not None
    replication_item_hashes = [
        cast(str, ref["item_hash"])
        for ref in cast(list[dict[str, object]], replication.holdout_set.payload["item_refs"])
    ]
    assert set(replication_item_hashes).isdisjoint(all_holdout_hashes)
    replication = _submit_scores(tmp_path, replication_config.run_id, replication, student_fault=None)
    replication = Luna8CertificationWorkflow.resume(tmp_path, replication_config.run_id, epoch=5005)
    assert replication.terminal and replication.certification_report is not None
    assert replication.attempt_index == 1
    assert replication.certification_report.payload["trace_judge_prompt_hash"] == (
        replication_trace_artifact.content_hash
    )

    # A genuinely fresh interpreter must recertify the terminal graph without writing.
    script = """
import json,sys
from clawrl.judge.certification_workflow import Luna8CertificationWorkflow
success=Luna8CertificationWorkflow.resume(sys.argv[1],sys.argv[2],epoch=9999)
exhausted=Luna8CertificationWorkflow.resume(sys.argv[1],sys.argv[3],epoch=9999)
replication=Luna8CertificationWorkflow.resume(sys.argv[1],sys.argv[4],epoch=9999)
print(json.dumps({
    'success': {
        'terminal': success.terminal,
        'attempts': success.attempt_index,
        'holdouts': len(success.holdout_sets),
        'status': success.certification_report.payload['status'],
        'report_hash': success.certification_report.content_hash,
    },
    'exhausted': {
        'terminal': exhausted.terminal,
        'attempts': exhausted.attempt_index,
        'holdouts': len(exhausted.holdout_sets),
        'reason': exhausted.exhaustion_report.payload['reason_code'],
    },
    'replication': {
        'terminal': replication.terminal,
        'attempts': replication.attempt_index,
        'holdouts': len(replication.holdout_sets),
        'status': replication.certification_report.payload['status'],
        'report_hash': replication.certification_report.content_hash,
    },
}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            success_config.run_id,
            exhausted_config.run_id,
            replication_config.run_id,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout) == {
        "success": {
            "terminal": True,
            "attempts": 1,
            "holdouts": 1,
            "status": "certified",
            "report_hash": success.certification_report.content_hash,
        },
        "exhausted": {
            "terminal": True,
            "attempts": 3,
            "holdouts": 3,
            "reason": "TRY_8_EXHAUSTED",
        },
        "replication": {
            "terminal": True,
            "attempts": 1,
            "holdouts": 1,
            "status": "certified",
            "report_hash": replication.certification_report.content_hash,
        },
    }


def test_visible_semantic_diversity_independent_roles_and_ledger_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticket04_run, label_hash, trace_id = _ticket04_terminal(tmp_path)
    degenerate_config = _config(
        tmp_path,
        run_id="ticket05-degenerate-semantic-boundary",
        ticket04_run_id=ticket04_run,
        teacher_label_set_hash=label_hash,
        trace_id=trace_id,
    )
    monkeypatch.setattr(
        "clawrl.adapters.generators.holdout.FixtureHoldoutGenerator._scenario_item",
        staticmethod(
            lambda material, index: (
                f"Compute {index + 2} + 1 and state the answer.",
                f"Because addition is direct, final answer is {index + 3}.",
            )
        ),
    )
    degenerate = Luna8CertificationWorkflow.run_until_role_input(tmp_path, degenerate_config, epoch=7006)
    assert degenerate.terminal and degenerate.holdout_set is None
    failed = next(event for event in degenerate.events if event.payload["event_type"] == "LUNA8_WORKFLOW_FAILED")
    assert cast(dict[str, Any], failed.payload["details"])["reason_code"] == ("HOLDOUT_SEMANTIC_DIVERSITY_INSUFFICIENT")
    monkeypatch.undo()

    config = _config(
        tmp_path,
        run_id="ticket05-independent-visible-success",
        ticket04_run_id=ticket04_run,
        teacher_label_set_hash=label_hash,
        trace_id=trace_id,
    )
    store = ArtifactStore(tmp_path)
    snapshot = Luna8CertificationWorkflow.run_until_role_input(tmp_path, config, epoch=7007)
    assert snapshot.teacher_packet is not None and snapshot.student_packet is not None
    assert "initial_eval_rubric" in snapshot.teacher_packet.payload
    assert "fit_calibration_profile" not in snapshot.teacher_packet.payload
    assert "candidate_prompt" not in snapshot.teacher_packet.payload
    assert "candidate_prompt" in snapshot.student_packet.payload
    assert _nested_keys(snapshot.student_packet.payload).isdisjoint(
        {"initial_eval_rubric", "teacher_label_set_hash", "fit_calibration_profile"}
    )
    assert snapshot.holdout_set is not None
    items = [
        store.read(cast(str, ref["item_hash"]), expected_schema_name="HoldoutItemContent")
        for ref in cast(list[dict[str, object]], snapshot.holdout_set.payload["item_refs"])
    ]
    visible = [{"prompt": item.payload["prompt"], "response": item.payload["response"]} for item in items]
    diversity = semantic_diversity_summary(cast(list[dict[str, object]], visible))
    assert diversity["status"] == "passed"
    assert cast(int, diversity["profile_count"]) >= 20
    assert all("provenance" not in cast(str, item.payload["response"]).casefold() for item in items)

    teacher = CertificationRoleIngress.stage_private_exchange(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=snapshot.teacher_packet.content_hash,
        raw_role_output=contract_example_teacher_raw(snapshot.teacher_packet),
        role_session_lineage=fixture_role_lineage("TeacherScorer"),
        execution_profile="fixture_contract_simulator",
    )
    student = CertificationRoleIngress.stage_private_exchange(
        tmp_path,
        role="StudentJudge",
        input_packet_hash=snapshot.student_packet.content_hash,
        raw_role_output=contract_example_student_raw(snapshot.student_packet),
        role_session_lineage=fixture_role_lineage("StudentJudge"),
        execution_profile="fixture_contract_simulator",
    )
    teacher_labels = CertificationRoleIngress.load_public_contract(tmp_path, teacher).normalized_output.payload[
        "labels"
    ]
    student_labels = CertificationRoleIngress.load_public_contract(tmp_path, student).normalized_output.payload[
        "labels"
    ]
    assert teacher_labels != student_labels
    Luna8CertificationWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=teacher, epoch=7007)
    snapshot = Luna8CertificationWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=student, epoch=7007)
    snapshot = Luna8CertificationWorkflow.resume(tmp_path, config.run_id, epoch=7007)
    assert snapshot.auditor_packet is not None
    auditor = CertificationRoleIngress.stage_private_exchange(
        tmp_path,
        role="AlignmentAuditor",
        input_packet_hash=snapshot.auditor_packet.content_hash,
        raw_role_output=contract_example_auditor_raw(snapshot.auditor_packet),
        role_session_lineage=fixture_role_lineage("AlignmentAuditor"),
        execution_profile="fixture_contract_simulator",
    )
    snapshot = Luna8CertificationWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=auditor, epoch=7007)
    snapshot = Luna8CertificationWorkflow.resume(tmp_path, config.run_id, epoch=7007)
    assert snapshot.terminal and snapshot.certification_report is not None
    assert snapshot.certification_report.payload["algorithm_contract_hash"] == config.algorithm_contract_hash
    holdout_event = next(event for event in snapshot.events if event.payload["event_type"] == "HOLDOUT_COMMITTED")
    ledger_hash = cast(str, cast(dict[str, object], holdout_event.payload["details"])["holdout_ledger_root_hash"])
    ledger = store.read(ledger_hash, expected_schema_name="HoldoutLedgerRoot")
    assert ledger.payload["entry_count"] == 1
    fresh = Luna8CertificationWorkflow.resume(tmp_path, config.run_id, epoch=9999)
    assert fresh.terminal
    fingerprint = cast(
        str,
        cast(list[dict[str, Any]], ledger.payload["entries"])[0]["visible_content_fingerprints"][0],
    )
    (tmp_path / "boundaries" / "holdout-ledger" / trace_id / "items" / f"{fingerprint}.ref").unlink()
    with pytest.raises(Luna8WorkflowError):
        Luna8CertificationWorkflow.resume(tmp_path, config.run_id, epoch=9999)


def test_unconsumed_ambiguous_auditor_packet_is_immutably_superseded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticket04_run, label_hash, trace_id = _ticket04_terminal(tmp_path)
    config = _config(
        tmp_path,
        run_id="ticket05-auditor-packet-supersession",
        ticket04_run_id=ticket04_run,
        teacher_label_set_hash=label_hash,
        trace_id=trace_id,
    )
    snapshot = Luna8CertificationWorkflow.run_until_role_input(tmp_path, config, epoch=5005)
    assert snapshot.teacher_packet is not None and snapshot.student_packet is not None
    teacher = CertificationRoleIngress.stage_private_exchange(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=snapshot.teacher_packet.content_hash,
        raw_role_output=contract_example_teacher_raw(snapshot.teacher_packet),
        role_session_lineage=fixture_role_lineage("TeacherScorer"),
        execution_profile="fixture_contract_simulator",
    )
    student = CertificationRoleIngress.stage_private_exchange(
        tmp_path,
        role="StudentJudge",
        input_packet_hash=snapshot.student_packet.content_hash,
        raw_role_output=contract_example_student_raw(snapshot.student_packet, calibration_fault="polarity_inversion"),
        role_session_lineage=fixture_role_lineage("StudentJudge"),
        execution_profile="fixture_contract_simulator",
    )
    Luna8CertificationWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=teacher, epoch=5005)
    Luna8CertificationWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=student, epoch=5005)

    original_descriptor = Luna8CertificationWorkflow.__dict__["_build_auditor_packet"]
    original_builder = Luna8CertificationWorkflow._build_auditor_packet

    def legacy_builder(cls: type[Luna8CertificationWorkflow], state: Any, attempt_index: int) -> Any:
        complete = original_builder(state, attempt_index)
        legacy_payload = dict(complete.payload)
        legacy_payload.pop("diagnostic_contract")
        legacy_payload["allowlisted_fields"] = cast(Any, sorted(legacy_payload))
        return state.store.put("CertificationAuditorInputPacket", "1.0.0", legacy_payload)

    monkeypatch.setattr(Luna8CertificationWorkflow, "_build_auditor_packet", classmethod(legacy_builder))
    snapshot = Luna8CertificationWorkflow.resume(tmp_path, config.run_id, epoch=5005)
    assert snapshot.auditor_packet is not None
    legacy = snapshot.auditor_packet
    with pytest.raises(CertificationRoleOutputError):
        CertificationRoleIngress.validate_input_packet(tmp_path, "AlignmentAuditor", legacy.content_hash)

    monkeypatch.setattr(Luna8CertificationWorkflow, "_build_auditor_packet", original_descriptor)
    snapshot = Luna8CertificationWorkflow.supersede_ambiguous_auditor_packet(tmp_path, config.run_id, epoch=5005)
    assert snapshot.auditor_packet is not None
    replacement = snapshot.auditor_packet
    assert replacement.content_hash != legacy.content_hash
    assert ArtifactStore(tmp_path).read(legacy.content_hash).payload == legacy.payload
    assert (
        CertificationRoleIngress.validate_input_packet(
            tmp_path, "AlignmentAuditor", replacement.content_hash
        ).content_hash
        == replacement.content_hash
    )
    details = cast(dict[str, object], snapshot.events[-1].payload["details"])
    assert details["supersedes_auditor_packet_hash"] == legacy.content_hash
    decision = ArtifactStore(tmp_path).read(
        cast(str, details["decision_record_hash"]), expected_schema_name="DecisionRecord"
    )
    assert decision.payload["reason_code"] == "AUDITOR_DIAGNOSTIC_ALGORITHM_WAS_NOT_CLOSED_WORLD"

    auditor = CertificationRoleIngress.stage_private_exchange(
        tmp_path,
        role="AlignmentAuditor",
        input_packet_hash=replacement.content_hash,
        raw_role_output=contract_example_auditor_raw(replacement),
        role_session_lineage=fixture_role_lineage("AlignmentAuditor"),
        execution_profile="fixture_contract_simulator",
    )
    Luna8CertificationWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=auditor, epoch=5005)
    with pytest.raises(Luna8WorkflowError):
        Luna8CertificationWorkflow.supersede_ambiguous_auditor_packet(tmp_path, config.run_id, epoch=5005)

    original_optimizer_descriptor = Luna8CertificationWorkflow.__dict__["_build_optimizer_packet"]
    original_optimizer_builder = Luna8CertificationWorkflow._build_optimizer_packet

    def legacy_optimizer_builder(cls: type[Luna8CertificationWorkflow], state: Any, attempt_index: int) -> Any:
        complete = original_optimizer_builder(state, attempt_index)
        legacy_payload = dict(complete.payload)
        legacy_schema = dict(cast(dict[str, Any], legacy_payload["output_schema"]))
        legacy_schema.pop("candidate_prompt_semantics")
        legacy_schema.pop("optimizer_session_id_semantics")
        legacy_payload["output_schema"] = cast(Any, legacy_schema)
        return state.store.put("CertificationOptimizerInputPacket", "1.0.0", legacy_payload)

    monkeypatch.setattr(Luna8CertificationWorkflow, "_build_optimizer_packet", classmethod(legacy_optimizer_builder))
    snapshot = Luna8CertificationWorkflow.resume(tmp_path, config.run_id, epoch=5005)
    assert snapshot.optimizer_packet is not None
    legacy_optimizer = snapshot.optimizer_packet
    with pytest.raises(CertificationRoleOutputError):
        CertificationRoleIngress.validate_input_packet(tmp_path, "PromptOptimizer", legacy_optimizer.content_hash)

    monkeypatch.setattr(Luna8CertificationWorkflow, "_build_optimizer_packet", original_optimizer_descriptor)
    snapshot = Luna8CertificationWorkflow.supersede_ambiguous_optimizer_packet(tmp_path, config.run_id, epoch=5005)
    assert snapshot.optimizer_packet is not None
    optimizer = snapshot.optimizer_packet
    assert optimizer.content_hash != legacy_optimizer.content_hash
    assert ArtifactStore(tmp_path).read(legacy_optimizer.content_hash).payload == legacy_optimizer.payload
    assert (
        CertificationRoleIngress.validate_input_packet(tmp_path, "PromptOptimizer", optimizer.content_hash).content_hash
        == optimizer.content_hash
    )
    details = cast(dict[str, object], snapshot.events[-1].payload["details"])
    assert details["supersedes_optimizer_packet_hash"] == legacy_optimizer.content_hash
    decision = ArtifactStore(tmp_path).read(
        cast(str, details["decision_record_hash"]), expected_schema_name="DecisionRecord"
    )
    assert decision.payload["reason_code"] == "OPTIMIZER_CANDIDATE_PROMPT_WIRE_CONTRACT_WAS_AMBIGUOUS"

    optimizer_output = CertificationRoleIngress.stage_private_exchange(
        tmp_path,
        role="PromptOptimizer",
        input_packet_hash=optimizer.content_hash,
        raw_role_output=contract_example_optimizer_raw(optimizer),
        role_session_lineage=fixture_role_lineage("PromptOptimizer"),
        execution_profile="fixture_contract_simulator",
    )
    Luna8CertificationWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=optimizer_output, epoch=5005)
    with pytest.raises(Luna8WorkflowError):
        Luna8CertificationWorkflow.supersede_ambiguous_optimizer_packet(tmp_path, config.run_id, epoch=5005)


def test_production_gate_blocks_before_constructing_any_boundary(tmp_path: Path) -> None:
    report = Luna8CertificationWorkflow.production_readiness(tmp_path, ProductionLuna8Config())
    assert report.payload["status"] == "blocked"
    assert report.payload["phase"] == "JUDGE_CERTIFY"
    assert report.payload["side_effects_permitted"] is False
    codes = {cast(dict[str, str], item)["code"] for item in cast(list[object], report.payload["checks"])}
    assert {
        "ALIGNMENT_POLICY_NUMERIC_CONTRACT_UNAVAILABLE",
        "SOL_MODEL_CONFIGURATION_UNAVAILABLE",
        "LUNA_MODEL_CONFIGURATION_UNAVAILABLE",
        "HOLDOUT_GENERATOR_CONFIGURATION_UNAVAILABLE",
        "REWARD_SCHEMA_UNAVAILABLE",
        "SCALARIZER_UNAVAILABLE",
        "RL_ALGORITHM_CONTRACT_UNAVAILABLE",
    } <= codes
    assert not (tmp_path / "boundaries").exists()
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "private-boundary").exists()


def _packet_item_hashes(packet: Any) -> list[str]:
    session = cast(dict[str, Any], packet.payload["scoring_session"])
    return [
        cast(str, item["item_hash"])
        for turn in cast(list[dict[str, Any]], session["turns"])
        for item in cast(list[dict[str, Any]], turn["items"])
    ]
