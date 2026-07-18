"""Ticket 06 exact TRY_4 success, fallback, and uncertifiable terminals."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from clawrl.adapters.scorers.certification_roles import (
    CertificationRoleIngress,
    fixture_role_lineage,
    ticket06_fixture_role_lineage,
)
from clawrl.artifacts import ArtifactStore
from clawrl.judge.certification_workflow import Luna8CertificationWorkflow
from clawrl.judge.luna4_models import FixtureLuna4Config, ProductionLuna4Config
from clawrl.judge.luna4_workflow import Luna4CertificationWorkflow, Luna4WorkflowError
from clawrl.training.run_journal import RunJournal, StaleFencingEpoch
from tests.e2e.test_ticket05_luna8_certification import _config, _submit_scores, _ticket04_terminal
from tests.fixtures.ticket05_roles import (
    contract_example_auditor_raw,
    contract_example_optimizer_raw,
    contract_example_student_raw,
    contract_example_teacher_raw,
)

_JUDGE_PACK_IDENTITY_KEYS = {
    "aggregation",
    "algorithm_contract_hash",
    "alignment_policy_hash",
    "attempt_id",
    "attempt_index",
    "base_judge_prompt_hash",
    "dataset_version_hash",
    "dataset_version_id",
    "fit_trajectory_set_hash",
    "initial_eval_rubric_hash",
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
    "try8_exhaustion_hash",
    "try8_source_terminal_event_hash",
}


def _try8_exhausted(root: Path) -> tuple[str, str]:
    ticket04_run, label_hash, trace_id = _ticket04_terminal(root)
    config = _config(
        root,
        run_id="ticket05-source-exhausted-for-ticket06",
        ticket04_run_id=ticket04_run,
        teacher_label_set_hash=label_hash,
        trace_id=trace_id,
    )
    snapshot = Luna8CertificationWorkflow.run_until_role_input(root, config, epoch=5005)
    for attempt in range(1, 4):
        snapshot = _submit_scores(root, config.run_id, snapshot, student_fault="polarity_inversion")
        snapshot = Luna8CertificationWorkflow.resume(root, config.run_id, epoch=5005)
        if attempt < 3:
            assert snapshot.optimizer_packet is not None
            ingress = CertificationRoleIngress.stage_private_exchange(
                root,
                role="PromptOptimizer",
                input_packet_hash=snapshot.optimizer_packet.content_hash,
                raw_role_output=contract_example_optimizer_raw(snapshot.optimizer_packet),
                role_session_lineage=fixture_role_lineage("PromptOptimizer"),
                execution_profile="fixture_contract_simulator",
            )
            Luna8CertificationWorkflow.submit_role_output(root, config.run_id, role_ingress=ingress, epoch=5005)
            Luna8CertificationWorkflow.resume(root, config.run_id, epoch=5005)
            snapshot = Luna8CertificationWorkflow.run_until_role_input(root, config, epoch=5005)
    assert snapshot.terminal and snapshot.exhaustion_report is not None
    return config.run_id, snapshot.exhaustion_report.content_hash


def _luna4_config(
    *, run_id: str, source_run: str, exhaustion_hash: str, mode: str = "calibrated_scalar"
) -> FixtureLuna4Config:
    return FixtureLuna4Config(
        run_id=run_id,
        try8_run_id=source_run,
        try8_exhaustion_hash=exhaustion_hash,
        holdout_seed=6006,
        role_seed=6606,
        sol_evidence_mode=cast(Any, mode),
        fault_schedule=("timeout", "delayed", "success"),
    )


def _stage(root: Path, role: str, packet: Any, raw: bytes) -> Any:
    return CertificationRoleIngress.stage_private_exchange(
        root,
        role=cast(Any, role),
        input_packet_hash=packet.content_hash,
        raw_role_output=raw,
        role_session_lineage=ticket06_fixture_role_lineage(cast(Any, role)),
        execution_profile="fixture_contract_simulator",
    )


def _drive_attempt(
    root: Path,
    config: FixtureLuna4Config,
    snapshot: Any,
    *,
    student_fault: str | None,
    epoch: int,
) -> Any:
    assert snapshot.teacher_packet is not None and snapshot.student_packet is not None
    assert snapshot.teacher_packet.payload["items_per_turn"] == 4
    assert snapshot.student_packet.payload["items_per_turn"] == 4
    assert cast(dict[str, Any], snapshot.teacher_packet.payload["scoring_session"])["turn_count"] == 8
    teacher = _stage(
        root, "TeacherScorer", snapshot.teacher_packet, contract_example_teacher_raw(snapshot.teacher_packet)
    )
    student = _stage(
        root,
        "StudentJudge",
        snapshot.student_packet,
        contract_example_student_raw(snapshot.student_packet, calibration_fault=student_fault),
    )
    # Out-of-order role completion is accepted without relaxing packet identity.
    Luna4CertificationWorkflow.submit_role_output(root, config.run_id, role_ingress=student, epoch=epoch)
    snapshot = Luna4CertificationWorkflow.submit_role_output(root, config.run_id, role_ingress=teacher, epoch=epoch)
    snapshot = Luna4CertificationWorkflow.resume(root, config.run_id, epoch=epoch)
    assert snapshot.auditor_packet is not None
    auditor = _stage(
        root,
        "AlignmentAuditor",
        snapshot.auditor_packet,
        contract_example_auditor_raw(snapshot.auditor_packet),
    )
    snapshot = Luna4CertificationWorkflow.submit_role_output(root, config.run_id, role_ingress=auditor, epoch=epoch)
    return Luna4CertificationWorkflow.resume(root, config.run_id, epoch=epoch)


def _drive_terminal_failures(root: Path, config: FixtureLuna4Config, *, epoch: int) -> Any:
    snapshot = Luna4CertificationWorkflow.run_until_role_input(root, config, epoch=epoch)
    for attempt in range(1, 4):
        assert snapshot.attempt_index == attempt
        snapshot = _drive_attempt(
            root,
            config,
            snapshot,
            student_fault="variance_collapse" if attempt == 1 else "polarity_inversion",
            epoch=epoch,
        )
        if attempt < 3:
            assert snapshot.optimizer_packet is not None
            optimizer = _stage(
                root,
                "PromptOptimizer",
                snapshot.optimizer_packet,
                contract_example_optimizer_raw(snapshot.optimizer_packet),
            )
            Luna4CertificationWorkflow.submit_role_output(root, config.run_id, role_ingress=optimizer, epoch=epoch)
            Luna4CertificationWorkflow.resume(root, config.run_id, epoch=epoch)
            snapshot = Luna4CertificationWorkflow.run_until_role_input(root, config, epoch=epoch)
    return snapshot


def test_try4_success_sol_fallback_uncertifiable_and_fresh_replay(tmp_path: Path) -> None:
    source_run, exhaustion_hash = _try8_exhausted(tmp_path)

    success_config = _luna4_config(
        run_id="ticket06-first-luna4-success", source_run=source_run, exhaustion_hash=exhaustion_hash
    )
    success = Luna4CertificationWorkflow.run_until_role_input(tmp_path, success_config, epoch=6006)
    success = _drive_attempt(tmp_path, success_config, success, student_fault=None, epoch=6006)
    assert success.terminal and success.judge_pack is not None
    assert success.judge_pack.payload["scorer_tier"] == "luna"
    assert success.judge_pack.payload["certification_level"] == 4
    assert set(success.judge_pack.payload) == _JUDGE_PACK_IDENTITY_KEYS | {
        "alignment_diagnostics",
        "certification_level",
        "certification_mode",
        "holdout_set_hash",
        "role_lineage",
        "scorer_tier",
        "status",
    }
    assert (
        ArtifactStore(tmp_path)
        .read(cast(str, success.judge_pack.payload["trace_judge_prompt_hash"]), expected_schema_name="TraceJudgePrompt")
        .content_hash
        == success.judge_pack.payload["trace_judge_prompt_hash"]
    )
    assert len(success.holdout_sets) == 1
    assert not {"TRY_16", "TRY_32", "LUNA8_CERTIFIED"}.intersection(
        event.payload["event_type"] for event in success.events
    )

    fallback_config = _luna4_config(
        run_id="ticket06-three-fail-sol-fallback", source_run=source_run, exhaustion_hash=exhaustion_hash
    )
    fallback = _drive_terminal_failures(tmp_path, fallback_config, epoch=6016)
    assert fallback.terminal and fallback.judge_pack is not None
    assert fallback.judge_pack.payload["scorer_tier"] == "sol"
    assert fallback.judge_pack.payload["certification_mode"] == "teacher_fallback"
    assert set(fallback.judge_pack.payload) == _JUDGE_PACK_IDENTITY_KEYS | {
        "attempted_holdout_set_hashes",
        "certification_mode",
        "failure_lineage",
        "golden_hard_entry_hash",
        "human_escalation_hash",
        "scorer_tier",
        "sol_fallback_evidence_hash",
        "status",
    }
    assert fallback.judge_pack.payload["attempted_holdout_set_hashes"] == [
        holdout.content_hash for holdout in fallback.holdout_sets
    ]
    assert all(
        set(cast(dict[str, object], failure))
        == {
            "alignment_auditor_audit_hash",
            "attempt_id",
            "attempt_index",
            "auditor_output_hash",
            "holdout_set_hash",
            "student_judge_audit_hash",
            "teacher_scorer_audit_hash",
            "trace_judge_prompt_hash",
        }
        for failure in cast(list[dict[str, object]], fallback.judge_pack.payload["failure_lineage"])
    )
    assert fallback.golden_hard_entry is not None and fallback.human_escalation is not None
    assert fallback.human_escalation.payload["blocking"] is False
    assert len(cast(list[object], fallback.judge_pack.payload["failure_lineage"])) == 3
    assert len(fallback.holdout_sets) == 3
    assert set(cast(list[str], fallback.golden_hard_entry.payload["reason_tags"])) == {
        "LOW_SCALAR_WITH_RELATIVE_ORDER",
        "TEACHER_VARIANCE_STUDENT_COLLAPSE",
    }
    fallback_evidence = ArtifactStore(tmp_path).read(
        cast(str, fallback.judge_pack.payload["sol_fallback_evidence_hash"]),
        expected_schema_name="SolFallbackEvidence",
    )
    assert fallback_evidence.payload["boundary_request_hash"]
    assert fallback_evidence.payload["boundary_observation_hash"]
    all_sets = cast(Any, Luna8CertificationWorkflow.resume(tmp_path, source_run, epoch=0)).holdout_sets
    all_sets = tuple(all_sets) + success.holdout_sets + fallback.holdout_sets
    item_hashes = [
        cast(str, ref["item_hash"])
        for holdout in all_sets
        for ref in cast(list[dict[str, object]], holdout.payload["item_refs"])
    ]
    assert len(item_hashes) == 7 * 32 and len(set(item_hashes)) == 7 * 32

    relative_config = _luna4_config(
        run_id="ticket06-relative-only-uncertifiable",
        source_run=source_run,
        exhaustion_hash=exhaustion_hash,
        mode="group_relative_only",
    )
    relative = _drive_terminal_failures(tmp_path, relative_config, epoch=6026)
    assert relative.terminal and relative.judge_pack is None and relative.uncertifiable_report is not None
    assert relative.uncertifiable_report.payload["blocks_bundle_publication"] is True
    assert relative.uncertifiable_report.payload["diagnostic_only"] is True
    assert relative.uncertifiable_report.payload["training_authorized"] is False
    assert relative.uncertifiable_report.payload["reason_code"] == "SOL_GROUP_RELATIVE_ONLY_UNSUPPORTED_V1"
    assert relative.uncertifiable_report.payload["evidence_tags"] == [
        "TEACHER_NO_VALID_CALIBRATED_VARIANCE",
        "SOL_GROUP_RELATIVE_ONLY",
    ]

    invalid_config = _luna4_config(
        run_id="ticket06-invalid-calibrated-variance-uncertifiable",
        source_run=source_run,
        exhaustion_hash=exhaustion_hash,
        mode="invalid_calibrated_variance",
    )
    invalid = _drive_terminal_failures(tmp_path, invalid_config, epoch=6036)
    assert invalid.terminal and invalid.judge_pack is None and invalid.uncertifiable_report is not None
    assert invalid.golden_hard_entry is None and invalid.human_escalation is None
    assert len(invalid.holdout_sets) == 3
    invalid_item_hashes = [
        cast(str, ref["item_hash"])
        for holdout in invalid.holdout_sets
        for ref in cast(list[dict[str, object]], holdout.payload["item_refs"])
    ]
    assert len(invalid_item_hashes) == 3 * 32
    assert len(set(invalid_item_hashes)) == 3 * 32
    assert set(invalid_item_hashes).isdisjoint(item_hashes)
    assert [event.payload["event_type"] for event in invalid.events][-2:] == [
        "UNCERTIFIABLE_TERMINAL",
        "RUN_CLOSED",
    ]
    invalid_payload = invalid.uncertifiable_report.payload
    assert invalid_payload["reason_code"] == "SOL_CALIBRATED_VARIANCE_INVALID"
    assert invalid_payload["evidence_tags"] == [
        "TEACHER_NO_VALID_CALIBRATED_VARIANCE",
        "SOL_CALIBRATED_VARIANCE_INVALID",
    ]
    assert invalid_payload["blocks_bundle_publication"] is True
    assert invalid_payload["training_authorized"] is False
    assert invalid_payload["diagnostic_only"] is True
    invalid_evidence = ArtifactStore(tmp_path).read(
        cast(str, invalid_payload["sol_fallback_evidence_hash"]), expected_schema_name="SolFallbackEvidence"
    )
    assert invalid_evidence.payload["calibrated_label_count"] == 32
    assert invalid_evidence.payload["scalar_variance_micros"] is None
    assert invalid_evidence.payload["relative_order_available"] is True
    assert invalid_evidence.payload["source_mode"] == "invalid_calibrated_variance"
    assert invalid_payload["evidence_tags"] != relative.uncertifiable_report.payload["evidence_tags"]
    assert invalid_payload["evidence_tags"] != fallback.golden_hard_entry.payload["reason_tags"]

    def inventory() -> dict[str, tuple[int, int, str]]:
        return {
            str(path.relative_to(tmp_path)): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in tmp_path.rglob("*")
            if path.is_file()
        }

    before_replay = inventory()
    invalid_command = (
        "from clawrl.judge.luna4_workflow import Luna4CertificationWorkflow as W; "
        f"s=W.resume({str(tmp_path)!r}, {invalid_config.run_id!r}, epoch=9999); "
        "print(s.terminal, s.uncertifiable_report.content_hash, len(s.holdout_sets), len(s.events))"
    )
    invalid_replay = subprocess.run(
        [sys.executable, "-c", invalid_command],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    assert invalid_replay.stdout.strip().startswith(f"True {invalid.uncertifiable_report.content_hash} 3 ")
    assert inventory() == before_replay

    invalid_report_bytes = invalid.uncertifiable_report.path.read_bytes()
    invalid.uncertifiable_report.path.write_bytes(b"{}")
    with pytest.raises(Luna4WorkflowError, match="cannot be freshly recertified"):
        Luna4CertificationWorkflow.resume(tmp_path, invalid_config.run_id, epoch=9999)
    invalid.uncertifiable_report.path.write_bytes(invalid_report_bytes)
    assert Luna4CertificationWorkflow.resume(tmp_path, invalid_config.run_id, epoch=9999).terminal

    invalid_ref = tmp_path / "boundaries" / "sol-fallback-evidence" / invalid_config.run_id / "observation.ref"
    committed_ref = invalid_ref.read_bytes()
    invalid_ref.write_bytes(("0" * 64 + "\n").encode("ascii"))
    with pytest.raises(Luna4WorkflowError, match="cannot be freshly recertified"):
        Luna4CertificationWorkflow.resume(tmp_path, invalid_config.run_id, epoch=9999)
    invalid_ref.write_bytes(committed_ref)
    assert Luna4CertificationWorkflow.resume(tmp_path, invalid_config.run_id, epoch=9999).terminal
    invalid_ref.unlink()
    with pytest.raises(Luna4WorkflowError, match="cannot be freshly recertified"):
        Luna4CertificationWorkflow.resume(tmp_path, invalid_config.run_id, epoch=9999)

    command = (
        "from clawrl.judge.luna4_workflow import Luna4CertificationWorkflow as W; "
        f"s=W.resume({str(tmp_path)!r}, {fallback_config.run_id!r}, epoch=9999); "
        "print(s.terminal, s.judge_pack.content_hash, len(s.holdout_sets), len(s.events))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip().startswith(f"True {fallback.judge_pack.content_hash} 3 ")

    success_pack_bytes = success.judge_pack.path.read_bytes()
    success.judge_pack.path.write_bytes(b"{}")
    with pytest.raises(Luna4WorkflowError, match="cannot be freshly recertified"):
        Luna4CertificationWorkflow.resume(tmp_path, success_config.run_id, epoch=9999)
    success.judge_pack.path.write_bytes(success_pack_bytes)
    assert Luna4CertificationWorkflow.resume(tmp_path, success_config.run_id, epoch=9999).terminal
    success_holdout_path = success.holdout_sets[-1].path
    success_holdout_bytes = success_holdout_path.read_bytes()
    success_holdout_path.unlink()
    with pytest.raises(Luna4WorkflowError, match="cannot be freshly recertified"):
        Luna4CertificationWorkflow.resume(tmp_path, success_config.run_id, epoch=9999)
    success_holdout_path.write_bytes(success_holdout_bytes)

    golden_bytes = fallback.golden_hard_entry.path.read_bytes()
    fallback.golden_hard_entry.path.write_bytes(b"{}")
    with pytest.raises(Luna4WorkflowError, match="cannot be freshly recertified"):
        Luna4CertificationWorkflow.resume(tmp_path, fallback_config.run_id, epoch=9999)
    fallback.golden_hard_entry.path.write_bytes(golden_bytes)
    assert Luna4CertificationWorkflow.resume(tmp_path, fallback_config.run_id, epoch=9999).terminal
    second_prompt_hash = cast(
        str,
        cast(list[dict[str, object]], fallback.judge_pack.payload["failure_lineage"])[1]["trace_judge_prompt_hash"],
    )
    second_prompt_path = ArtifactStore(tmp_path).read(second_prompt_hash, expected_schema_name="TraceJudgePrompt").path
    second_prompt_bytes = second_prompt_path.read_bytes()
    second_prompt_path.unlink()
    with pytest.raises(Luna4WorkflowError, match="cannot be freshly recertified"):
        Luna4CertificationWorkflow.resume(tmp_path, fallback_config.run_id, epoch=9999)
    second_prompt_path.write_bytes(second_prompt_bytes)
    escalation_path = fallback.human_escalation.path
    escalation_path.unlink()
    with pytest.raises(Luna4WorkflowError, match="cannot be freshly recertified"):
        Luna4CertificationWorkflow.resume(tmp_path, fallback_config.run_id, epoch=9999)

    with pytest.raises(Luna4WorkflowError):
        Luna4CertificationWorkflow.bootstrap(
            tmp_path,
            FixtureLuna4Config(
                run_id="ticket06-invalid-source",
                try8_run_id=source_run,
                try8_exhaustion_hash="f" * 64,
                holdout_seed=1,
                role_seed=2,
            ),
            epoch=1,
        )


def test_try4_fencing_and_production_gate_precede_boundary_calls(tmp_path: Path) -> None:
    source_run, exhaustion_hash = _try8_exhausted(tmp_path)
    config = _luna4_config(run_id="ticket06-fencing", source_run=source_run, exhaustion_hash=exhaustion_hash)
    Luna4CertificationWorkflow.bootstrap(tmp_path, config, epoch=7006)
    Luna4CertificationWorkflow.resume(tmp_path, config.run_id, epoch=8006)
    with pytest.raises(StaleFencingEpoch):
        Luna4CertificationWorkflow.resume(tmp_path, config.run_id, epoch=7006)
    foundation = Luna4CertificationWorkflow.resume(tmp_path, config.run_id, epoch=8006)
    decision_event = next(
        event for event in foundation.events if event.payload["event_type"] == "TRY4_FOUNDATION_FROZEN"
    )
    decision_hash = cast(str, cast(dict[str, object], decision_event.payload["details"])["decision_record_hash"])
    decision_path = ArtifactStore(tmp_path).read(decision_hash, expected_schema_name="DecisionRecord").path
    decision_bytes = decision_path.read_bytes()
    decision_path.unlink()
    with pytest.raises(Luna4WorkflowError, match="cannot be freshly recertified"):
        Luna4CertificationWorkflow.resume(tmp_path, config.run_id, epoch=8006)
    decision_path.write_bytes(decision_bytes)

    failed_config = FixtureLuna4Config(
        run_id="ticket06-holdout-boundary-failed-closed",
        try8_run_id=source_run,
        try8_exhaustion_hash=exhaustion_hash,
        holdout_seed=6106,
        role_seed=6616,
        fault_schedule=("permanent_failure",),
    )
    failed = Luna4CertificationWorkflow.run_until_role_input(tmp_path, failed_config, epoch=9006)
    assert failed.terminal and not failed.holdout_sets and failed.judge_pack is None
    assert [event.payload["event_type"] for event in failed.events][-2:] == ["TRY4_FAILED_CLOSED", "RUN_CLOSED"]
    assert Luna4CertificationWorkflow.resume(tmp_path, failed_config.run_id, epoch=9999).terminal
    failed_plan_event = next(event for event in failed.events if event.payload["event_type"] == "HOLDOUT_PLAN_FROZEN")
    failed_plan_hash = cast(str, cast(dict[str, object], failed_plan_event.payload["details"])["holdout_plan_hash"])
    failed_plan_path = ArtifactStore(tmp_path).read(failed_plan_hash, expected_schema_name="HoldoutPlan").path
    failed_plan_bytes = failed_plan_path.read_bytes()
    failed_plan_path.write_bytes(b"{}")
    with pytest.raises(Luna4WorkflowError, match="cannot be freshly recertified"):
        Luna4CertificationWorkflow.resume(tmp_path, failed_config.run_id, epoch=9999)
    failed_plan_path.write_bytes(failed_plan_bytes)
    assert Luna4CertificationWorkflow.resume(tmp_path, failed_config.run_id, epoch=9999).terminal

    ingress_config = FixtureLuna4Config(
        run_id="ticket06-configured-arbitrary-isolated-lineage",
        try8_run_id=source_run,
        try8_exhaustion_hash=exhaustion_hash,
        holdout_seed=6156,
        role_seed=6618,
    )
    ingress_snapshot = Luna4CertificationWorkflow.run_until_role_input(tmp_path, ingress_config, epoch=9056)
    assert ingress_snapshot.teacher_packet is not None
    ingress_lineage = "/orchestrator/runtime-42/arbitrary-teacher-session"
    teacher_raw = contract_example_teacher_raw(ingress_snapshot.teacher_packet)

    def ingress_inventory() -> dict[str, tuple[int, int, str]]:
        return {
            str(path.relative_to(tmp_path)): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in tmp_path.rglob("*")
            if path.is_file()
        }

    before_legacy_impersonation = ingress_inventory()
    with pytest.raises(ValueError, match="Ticket05 legacy isolated lineage"):
        CertificationRoleIngress.stage_private_exchange(
            tmp_path,
            role="TeacherScorer",
            input_packet_hash=ingress_snapshot.teacher_packet.content_hash,
            raw_role_output=teacher_raw,
            role_session_lineage="/root/ticket05_teacher_scorer",
            execution_profile="isolated_subagent",
            run_id=ingress_config.run_id,
        )
    with pytest.raises(ValueError, match="fixture role lineage"):
        CertificationRoleIngress.stage_private_exchange(
            tmp_path,
            role="TeacherScorer",
            input_packet_hash=ingress_snapshot.teacher_packet.content_hash,
            raw_role_output=teacher_raw,
            role_session_lineage=fixture_role_lineage("TeacherScorer"),
            execution_profile="fixture_contract_simulator",
        )
    assert ingress_inventory() == before_legacy_impersonation

    ingress_grant = CertificationRoleIngress.configure_isolated_lineage(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=ingress_snapshot.teacher_packet.content_hash,
        role_session_lineage=ingress_lineage,
        run_id=ingress_config.run_id,
        attempt_index=1,
        retry_index=0,
    )
    with pytest.raises(ValueError, match="isolated role lineage"):
        CertificationRoleIngress.stage_private_exchange(
            tmp_path,
            role="TeacherScorer",
            input_packet_hash=ingress_snapshot.teacher_packet.content_hash,
            raw_role_output=teacher_raw,
            role_session_lineage="/unconfigured/task-name-does-not-confer-authority",
            execution_profile="isolated_subagent",
            run_id=ingress_config.run_id,
        )
    with pytest.raises(ValueError, match="cannot be freshly recertified"):
        CertificationRoleIngress.stage_private_exchange(
            tmp_path,
            role="TeacherScorer",
            input_packet_hash=ingress_snapshot.teacher_packet.content_hash,
            raw_role_output=teacher_raw,
            role_session_lineage=ingress_lineage,
            execution_profile="isolated_subagent",
            run_id="wrong-run-identity",
        )
    with pytest.raises(ValueError, match="fixture role lineage"):
        CertificationRoleIngress.stage_private_exchange(
            tmp_path,
            role="TeacherScorer",
            input_packet_hash=ingress_snapshot.teacher_packet.content_hash,
            raw_role_output=teacher_raw,
            role_session_lineage=ingress_lineage,
            execution_profile="fixture_contract_simulator",
        )
    ingress_receipt = CertificationRoleIngress.stage_private_exchange(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=ingress_snapshot.teacher_packet.content_hash,
        raw_role_output=teacher_raw,
        role_session_lineage=ingress_lineage,
        execution_profile="isolated_subagent",
        run_id=ingress_config.run_id,
    )
    assert (
        CertificationRoleIngress.load_public_contract(tmp_path, ingress_receipt).role_invocation_audit.payload[
            "role_lineage_grant_hash"
        ]
        == ingress_grant.content_hash
    )
    ingress_grant_bytes = ingress_grant.path.read_bytes()
    ingress_grant.path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="cannot be freshly recertified"):
        CertificationRoleIngress.load_public_contract(tmp_path, ingress_receipt)
    ingress_grant.path.write_bytes(ingress_grant_bytes)
    assert CertificationRoleIngress.load_public_contract(tmp_path, ingress_receipt).role == "TeacherScorer"

    wrong_run_config = FixtureLuna4Config(
        run_id="ticket06-wrong-run-first-lineage-grant",
        try8_run_id=source_run,
        try8_exhaustion_hash=exhaustion_hash,
        holdout_seed=6166,
        role_seed=6628,
    )
    wrong_run_snapshot = Luna4CertificationWorkflow.run_until_role_input(tmp_path, wrong_run_config, epoch=9066)
    assert wrong_run_snapshot.teacher_packet is not None
    wrong_run_lineage = "/orchestrator/runtime-43/wrong-run-teacher-session"
    CertificationRoleIngress.configure_isolated_lineage(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=wrong_run_snapshot.teacher_packet.content_hash,
        role_session_lineage=wrong_run_lineage,
        run_id="different-owning-workflow-run",
        attempt_index=1,
        retry_index=0,
    )
    wrong_run_receipt = CertificationRoleIngress.stage_private_exchange(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=wrong_run_snapshot.teacher_packet.content_hash,
        raw_role_output=contract_example_teacher_raw(wrong_run_snapshot.teacher_packet),
        role_session_lineage=wrong_run_lineage,
        execution_profile="isolated_subagent",
        run_id="different-owning-workflow-run",
    )
    with pytest.raises(Luna4WorkflowError, match="frozen invocation slot"):
        Luna4CertificationWorkflow.submit_role_output(
            tmp_path, wrong_run_config.run_id, role_ingress=wrong_run_receipt, epoch=9066
        )

    wrong_attempt_config = FixtureLuna4Config(
        run_id="ticket06-wrong-attempt-first-lineage-grant",
        try8_run_id=source_run,
        try8_exhaustion_hash=exhaustion_hash,
        holdout_seed=6176,
        role_seed=6638,
    )
    wrong_attempt_snapshot = Luna4CertificationWorkflow.run_until_role_input(tmp_path, wrong_attempt_config, epoch=9076)
    assert wrong_attempt_snapshot.teacher_packet is not None
    wrong_attempt_lineage = "/orchestrator/runtime-44/wrong-attempt-teacher-session"
    CertificationRoleIngress.configure_isolated_lineage(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=wrong_attempt_snapshot.teacher_packet.content_hash,
        role_session_lineage=wrong_attempt_lineage,
        run_id=wrong_attempt_config.run_id,
        attempt_index=2,
        retry_index=0,
    )
    wrong_attempt_receipt = CertificationRoleIngress.stage_private_exchange(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=wrong_attempt_snapshot.teacher_packet.content_hash,
        raw_role_output=contract_example_teacher_raw(wrong_attempt_snapshot.teacher_packet),
        role_session_lineage=wrong_attempt_lineage,
        execution_profile="isolated_subagent",
        run_id=wrong_attempt_config.run_id,
        attempt_index=2,
        retry_index=0,
    )
    with pytest.raises(Luna4WorkflowError, match="frozen invocation slot"):
        Luna4CertificationWorkflow.submit_role_output(
            tmp_path, wrong_attempt_config.run_id, role_ingress=wrong_attempt_receipt, epoch=9076
        )

    role_timeout_config = FixtureLuna4Config(
        run_id="ticket06-isolated-role-retries-fail-closed",
        try8_run_id=source_run,
        try8_exhaustion_hash=exhaustion_hash,
        holdout_seed=6206,
        role_seed=6626,
    )
    role_timeout = Luna4CertificationWorkflow.run_until_role_input(tmp_path, role_timeout_config, epoch=9106)
    assert role_timeout.teacher_packet is not None
    role_timeout_teacher_packet = role_timeout.teacher_packet
    packet_hash = role_timeout_teacher_packet.content_hash
    configured_lineage = "/approved/orchestrator/session-7f91/teacher"
    with pytest.raises(Luna4WorkflowError, match="not frozen"):
        Luna4CertificationWorkflow.record_role_timeout(
            tmp_path,
            role_timeout_config.run_id,
            role="TeacherScorer",
            input_packet_hash=packet_hash,
            role_session_lineage=configured_lineage,
            epoch=9106,
        )
    with pytest.raises(ValueError, match="invalid"):
        CertificationRoleIngress.configure_isolated_lineage(
            tmp_path,
            role="TeacherScorer",
            input_packet_hash=packet_hash,
            role_session_lineage="/fixture/impersonated-isolated-role",
            run_id=role_timeout_config.run_id,
            attempt_index=1,
            retry_index=0,
        )
    lineage_grant = CertificationRoleIngress.configure_isolated_lineage(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=packet_hash,
        role_session_lineage=configured_lineage,
        run_id=role_timeout_config.run_id,
        attempt_index=1,
        retry_index=0,
    )
    grant_bytes = lineage_grant.path.read_bytes()
    lineage_grant.path.write_bytes(b"{}")
    with pytest.raises(Luna4WorkflowError, match="not frozen"):
        Luna4CertificationWorkflow.record_role_timeout(
            tmp_path,
            role_timeout_config.run_id,
            role="TeacherScorer",
            input_packet_hash=packet_hash,
            role_session_lineage=configured_lineage,
            epoch=9106,
        )
    lineage_grant.path.write_bytes(grant_bytes)
    Luna4CertificationWorkflow.record_role_timeout(
        tmp_path,
        role_timeout_config.run_id,
        role="TeacherScorer",
        input_packet_hash=packet_hash,
        role_session_lineage=configured_lineage,
        epoch=9106,
    )
    CertificationRoleIngress.configure_isolated_lineage(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=packet_hash,
        role_session_lineage=configured_lineage,
        run_id=role_timeout_config.run_id,
        attempt_index=1,
        retry_index=2,
    )
    with pytest.raises(Luna4WorkflowError, match="out of sequence"):
        Luna4CertificationWorkflow.record_role_timeout(
            tmp_path,
            role_timeout_config.run_id,
            role="TeacherScorer",
            input_packet_hash=packet_hash,
            role_session_lineage=configured_lineage,
            retry_index=2,
            epoch=9106,
        )
    for retry_index in range(1, 4):
        CertificationRoleIngress.configure_isolated_lineage(
            tmp_path,
            role="TeacherScorer",
            input_packet_hash=packet_hash,
            role_session_lineage=configured_lineage,
            run_id=role_timeout_config.run_id,
            attempt_index=1,
            retry_index=retry_index,
        )
        role_timeout = Luna4CertificationWorkflow.record_role_timeout(
            tmp_path,
            role_timeout_config.run_id,
            role="TeacherScorer",
            input_packet_hash=packet_hash,
            role_session_lineage=configured_lineage,
            retry_index=retry_index,
            epoch=9106,
        )
        if retry_index == 2:
            stale_success = CertificationRoleIngress.stage_private_exchange(
                tmp_path,
                role="TeacherScorer",
                input_packet_hash=packet_hash,
                raw_role_output=contract_example_teacher_raw(role_timeout_teacher_packet),
                role_session_lineage=configured_lineage,
                execution_profile="isolated_subagent",
                run_id=role_timeout_config.run_id,
                attempt_index=1,
                retry_index=0,
            )
            with pytest.raises(Luna4WorkflowError, match="frozen invocation slot"):
                Luna4CertificationWorkflow.submit_role_output(
                    tmp_path, role_timeout_config.run_id, role_ingress=stale_success, epoch=9106
                )
    final_timeout_audit = ArtifactStore(tmp_path).read(
        cast(str, cast(dict[str, object], role_timeout.events[-1].payload["details"])["timeout_audit_hash"]),
        expected_schema_name="RoleInvocationTimeoutAudit",
    )
    assert final_timeout_audit.payload["retry_scheduled"] is False
    role_timeout = Luna4CertificationWorkflow.close_role_boundary_exhausted(
        tmp_path, role_timeout_config.run_id, role="TeacherScorer", epoch=9106
    )
    assert role_timeout.terminal and role_timeout.judge_pack is None
    assert Luna4CertificationWorkflow.resume(tmp_path, role_timeout_config.run_id, epoch=9999).terminal
    exhaustion_event = role_timeout.events[-2]
    exhaustion_hash = cast(str, cast(dict[str, object], exhaustion_event.payload["details"])["failure_report_hash"])
    exhaustion_path = (
        ArtifactStore(tmp_path).read(exhaustion_hash, expected_schema_name="RoleBoundaryExhaustionReport").path
    )
    exhaustion_bytes = exhaustion_path.read_bytes()
    exhaustion_path.write_bytes(b"{}")
    with pytest.raises(Luna4WorkflowError, match="cannot be freshly recertified"):
        Luna4CertificationWorkflow.resume(tmp_path, role_timeout_config.run_id, epoch=9999)
    exhaustion_path.write_bytes(exhaustion_bytes)
    assert Luna4CertificationWorkflow.resume(tmp_path, role_timeout_config.run_id, epoch=9999).terminal
    exhaustion_path.unlink()
    with pytest.raises(Luna4WorkflowError, match="cannot be freshly recertified"):
        Luna4CertificationWorkflow.resume(tmp_path, role_timeout_config.run_id, epoch=9999)

    production_root = tmp_path / "production"
    report = Luna4CertificationWorkflow.production_readiness(production_root, ProductionLuna4Config())
    assert report.payload["status"] == "blocked" and report.payload["side_effects_permitted"] is False
    assert not (production_root / "boundaries").exists()
    assert not (production_root / "runs").exists()
    persisted = json.loads(report.path.read_text())
    assert persisted["content_hash"] == report.content_hash
    assert RunJournal(
        tmp_path,
        ArtifactStore(tmp_path),
        config.run_id,
        input_schema_name="Luna4CertificationInput",
        input_schema_version="1.0.0",
    ).events()
