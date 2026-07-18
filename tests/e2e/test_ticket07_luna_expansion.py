"""Ticket 07 production-shaped Luna@16/@32 expansion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from clawrl.adapters.scorers.certification_roles import CertificationRoleIngress, ticket07_fixture_role_lineage
from clawrl.artifacts import Artifact, ArtifactStore, sha256_hex
from clawrl.judge.certification_workflow import Luna8CertificationWorkflow
from clawrl.judge.luna_expansion_models import FixtureLunaExpansionConfig
from clawrl.judge.luna_expansion_workflow import LunaExpansionWorkflow, LunaExpansionWorkflowError
from clawrl.training.run_journal import RunJournal
from tests.e2e.test_ticket05_luna8_certification import _config as _luna8_config
from tests.e2e.test_ticket05_luna8_certification import _submit_scores, _ticket04_terminal
from tests.e2e.test_ticket06_luna4_sol_fallback import _drive_attempt as _drive_luna4_attempt
from tests.e2e.test_ticket06_luna4_sol_fallback import _luna4_config, _try8_exhausted
from tests.fixtures.ticket05_roles import (
    contract_example_auditor_raw,
    contract_example_student_raw,
    contract_example_teacher_raw,
)


def _source(root: Path) -> tuple[Any, FixtureLunaExpansionConfig]:
    ticket04_run, label_hash, trace_id = _ticket04_terminal(root)
    config = _luna8_config(
        root,
        run_id="ticket05-for-ticket07",
        ticket04_run_id=ticket04_run,
        teacher_label_set_hash=label_hash,
        trace_id=trace_id,
    )
    source = Luna8CertificationWorkflow.run_until_role_input(root, config, epoch=5005)
    source = _submit_scores(root, config.run_id, source, student_fault=None)
    source = Luna8CertificationWorkflow.resume(root, config.run_id, epoch=5005)
    assert source.terminal and source.certification_report is not None
    expansion = FixtureLunaExpansionConfig(
        run_id="ticket07-expand",
        source_run_id=config.run_id,
        source_certification_report_hash=source.certification_report.content_hash,
        alignment_policy_hash=cast(str, source.certification_report.payload["alignment_policy_hash"]),
        holdout_seed=7007,
        role_seed=7707,
        fault_schedule=("timeout", "delayed", "success"),
    )
    return source, expansion


def _item_hashes(packet: Artifact) -> list[str]:
    session = cast(dict[str, Any], packet.payload["scoring_session"])
    return [
        cast(str, item["item_hash"])
        for turn in cast(list[dict[str, Any]], session["turns"])
        for item in cast(list[dict[str, Any]], turn["items"])
    ]


def _submit_level(root: Path, config: FixtureLunaExpansionConfig, snapshot: Any, fault: str | None) -> Any:
    assert snapshot.teacher_packet is not None and snapshot.student_packet is not None
    teacher = CertificationRoleIngress.stage_private_exchange(
        root,
        role="TeacherScorer",
        input_packet_hash=snapshot.teacher_packet.content_hash,
        raw_role_output=contract_example_teacher_raw(snapshot.teacher_packet),
        role_session_lineage=ticket07_fixture_role_lineage("TeacherScorer"),
        execution_profile="fixture_contract_simulator",
    )
    student = CertificationRoleIngress.stage_private_exchange(
        root,
        role="StudentJudge",
        input_packet_hash=snapshot.student_packet.content_hash,
        raw_role_output=contract_example_student_raw(snapshot.student_packet, calibration_fault=fault),
        role_session_lineage=ticket07_fixture_role_lineage("StudentJudge"),
        execution_profile="fixture_contract_simulator",
    )
    LunaExpansionWorkflow.submit_role_output(root, config.run_id, role_ingress=student, epoch=7007)
    snapshot = LunaExpansionWorkflow.submit_role_output(root, config.run_id, role_ingress=teacher, epoch=7007)
    snapshot = LunaExpansionWorkflow.resume(root, config.run_id, epoch=7007)
    assert snapshot.auditor_packet is not None
    auditor = CertificationRoleIngress.stage_private_exchange(
        root,
        role="AlignmentAuditor",
        input_packet_hash=snapshot.auditor_packet.content_hash,
        raw_role_output=contract_example_auditor_raw(snapshot.auditor_packet),
        role_session_lineage=ticket07_fixture_role_lineage("AlignmentAuditor"),
        execution_profile="fixture_contract_simulator",
    )
    return LunaExpansionWorkflow.submit_role_output(root, config.run_id, role_ingress=auditor, epoch=7007)


def test_expand_16_then_32_with_fresh_disjoint_holdouts_and_fresh_resume(tmp_path: Path) -> None:
    source, config = _source(tmp_path)
    snapshot = LunaExpansionWorkflow.run_until_role_input(tmp_path, config, epoch=7007)
    source_pack_hash = snapshot.source_judge_pack.content_hash
    assert snapshot.level == 16
    assert snapshot.teacher_packet is not None and snapshot.student_packet is not None
    assert snapshot.teacher_packet.payload["items_per_turn"] == 4
    assert snapshot.student_packet.payload["items_per_turn"] == 16
    assert _item_hashes(snapshot.teacher_packet) == _item_hashes(snapshot.student_packet)
    source_items = {
        cast(str, ref["item_hash"])
        for holdout in source.holdout_sets
        for ref in cast(list[dict[str, object]], holdout.payload["item_refs"])
    }
    level16_items = set(_item_hashes(snapshot.student_packet))
    assert len(level16_items) == 32 and not source_items.intersection(level16_items)

    snapshot = _submit_level(tmp_path, config, snapshot, None)
    snapshot = LunaExpansionWorkflow.run_until_role_input(tmp_path, config, epoch=7007)
    assert snapshot.level == 32 and snapshot.retained_judge_pack.payload["certification_level"] == 16
    level16_pack_hash = snapshot.retained_judge_pack.content_hash
    assert snapshot.teacher_packet is not None and snapshot.student_packet is not None
    assert snapshot.teacher_packet.payload["items_per_turn"] == 4
    assert snapshot.student_packet.payload["items_per_turn"] == 32
    level32_items = set(_item_hashes(snapshot.student_packet))
    assert len(level32_items) == 32 and not level16_items.intersection(level32_items)
    assert _item_hashes(snapshot.teacher_packet) == _item_hashes(snapshot.student_packet)

    snapshot = _submit_level(tmp_path, config, snapshot, "polarity_inversion")
    snapshot = LunaExpansionWorkflow.resume(tmp_path, config.run_id, epoch=7007)
    assert snapshot.terminal and snapshot.expansion_result is not None
    assert snapshot.retained_judge_pack.content_hash == level16_pack_hash
    assert (
        snapshot.retained_judge_pack.payload["trace_judge_prompt_hash"]
        == (snapshot.source_judge_pack.payload["trace_judge_prompt_hash"])
    )
    assert snapshot.source_judge_pack.content_hash == source_pack_hash
    assert snapshot.expansion_result.payload["failed_level"] == 32

    artifact_count = len(list((tmp_path / "artifacts").glob("*.json")))
    fresh = LunaExpansionWorkflow.resume(tmp_path, config.run_id, epoch=0)
    assert fresh.terminal and fresh.retained_judge_pack.content_hash == level16_pack_hash
    assert len(list((tmp_path / "artifacts").glob("*.json"))) == artifact_count

    policy_payload = dict(ArtifactStore(tmp_path).read(config.alignment_policy_hash).payload)
    policy_payload["max_absolute_bias_micros"] = cast(int, policy_payload["max_absolute_bias_micros"]) - 1
    changed_policy = ArtifactStore(tmp_path).put("AlignmentPolicy", "1.0.0", policy_payload)
    with pytest.raises(LunaExpansionWorkflowError):
        LunaExpansionWorkflow.bootstrap(
            tmp_path,
            FixtureLunaExpansionConfig.from_mapping(
                {**config.immutable_input_payload, "alignment_policy_hash": changed_policy.content_hash}
            ),
            epoch=7007,
        )
    changed_run = FixtureLunaExpansionConfig.from_mapping(
        {
            **config.immutable_input_payload,
            "alignment_policy_hash": changed_policy.content_hash,
            "run_id": "ticket07-policy-change-new-attempt",
        }
    )
    changed = LunaExpansionWorkflow.run_until_role_input(tmp_path, changed_run, epoch=7007)
    assert changed.holdout_set is not None
    assert changed.holdout_set.content_hash not in {item.content_hash for item in fresh.holdout_sets}


def test_failure_at_16_retains_exact_8_and_never_attempts_32(tmp_path: Path) -> None:
    _, base = _source(tmp_path)
    config = FixtureLunaExpansionConfig.from_mapping(
        {**base.immutable_input_payload, "run_id": "ticket07-fail16", "holdout_seed": 17007}
    )
    snapshot = LunaExpansionWorkflow.run_until_role_input(tmp_path, config, epoch=7007)
    source_hash = snapshot.source_judge_pack.content_hash
    snapshot = _submit_level(tmp_path, config, snapshot, "polarity_inversion")
    snapshot = LunaExpansionWorkflow.resume(tmp_path, config.run_id, epoch=7007)
    assert snapshot.terminal
    assert snapshot.retained_judge_pack.content_hash == source_hash
    assert snapshot.retained_judge_pack.payload["certification_level"] == 8
    assert [
        cast(dict[str, object], event.payload["details"]).get("level")
        for event in snapshot.events
        if event.payload["event_type"] == "HOLDOUT_PLAN_FROZEN"
    ] == [16]
    assert snapshot.expansion_result is not None and snapshot.expansion_result.payload["failed_level"] == 16


def test_success_at_both_levels_retains_32(tmp_path: Path) -> None:
    _, config = _source(tmp_path)
    snapshot = LunaExpansionWorkflow.run_until_role_input(tmp_path, config, epoch=7007)
    snapshot = _submit_level(tmp_path, config, snapshot, None)
    snapshot = LunaExpansionWorkflow.run_until_role_input(tmp_path, config, epoch=7007)
    snapshot = _submit_level(tmp_path, config, snapshot, None)
    snapshot = LunaExpansionWorkflow.resume(tmp_path, config.run_id, epoch=7007)
    assert snapshot.terminal and snapshot.retained_judge_pack.payload["certification_level"] == 32
    assert snapshot.expansion_result is not None and snapshot.expansion_result.payload["failed_level"] is None
    assert snapshot.expansion_result.payload["attempted_levels"] == [16, 32]


def test_deleting_direct_pack_reference_fails_fresh_recertification(tmp_path: Path) -> None:
    _, config = _source(tmp_path)
    snapshot = LunaExpansionWorkflow.run_until_role_input(tmp_path, config, epoch=7007)
    direct_hash = cast(str, snapshot.source_judge_pack.payload["reward_schema_hash"])
    (ArtifactStore(tmp_path).artifact_dir / f"{direct_hash}.json").unlink()
    try:
        LunaExpansionWorkflow.resume(tmp_path, config.run_id, epoch=7007)
    except Exception as error:  # exact public type asserted without swallowing unrelated success
        assert error.__class__.__name__ == "LunaExpansionWorkflowError"
    else:
        raise AssertionError("deleted direct JudgePack reference must fail closed")


def test_isolated_role_timeout_requires_grant_and_recovers_on_fenced_retry(tmp_path: Path) -> None:
    _, config = _source(tmp_path)
    snapshot = LunaExpansionWorkflow.run_until_role_input(tmp_path, config, epoch=7007)
    assert snapshot.teacher_packet is not None
    packet = snapshot.teacher_packet
    lineage = "/root/ticket07_timeout_teacher"
    with pytest.raises(LunaExpansionWorkflowError):
        LunaExpansionWorkflow.record_role_timeout(
            tmp_path,
            config.run_id,
            role="TeacherScorer",
            input_packet_hash=packet.content_hash,
            role_session_lineage=lineage,
            retry_index=0,
            epoch=7007,
        )
    CertificationRoleIngress.configure_isolated_lineage(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=packet.content_hash,
        role_session_lineage=lineage,
        run_id=config.run_id,
        attempt_index=1,
        retry_index=0,
    )
    timed_out = LunaExpansionWorkflow.record_role_timeout(
        tmp_path,
        config.run_id,
        role="TeacherScorer",
        input_packet_hash=packet.content_hash,
        role_session_lineage=lineage,
        retry_index=0,
        epoch=7007,
    )
    assert timed_out.events[-1].payload["event_type"] == "ROLE_INVOCATION_TIMEOUT_OBSERVED"
    duplicate_artifact_count = len(list(ArtifactStore(tmp_path).artifact_dir.glob("*.json")))
    duplicate = LunaExpansionWorkflow.record_role_timeout(
        tmp_path,
        config.run_id,
        role="TeacherScorer",
        input_packet_hash=packet.content_hash,
        role_session_lineage=lineage,
        retry_index=0,
        epoch=7007,
    )
    assert duplicate.events[-1].content_hash == timed_out.events[-1].content_hash
    assert len(list(ArtifactStore(tmp_path).artifact_dir.glob("*.json"))) == duplicate_artifact_count
    CertificationRoleIngress.configure_isolated_lineage(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=packet.content_hash,
        role_session_lineage=lineage,
        run_id=config.run_id,
        attempt_index=1,
        retry_index=2,
    )
    wrong_slot = CertificationRoleIngress.stage_private_exchange(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=packet.content_hash,
        raw_role_output=contract_example_teacher_raw(packet),
        role_session_lineage=lineage,
        execution_profile="isolated_subagent",
        run_id=config.run_id,
        attempt_index=1,
        retry_index=2,
    )
    with pytest.raises(LunaExpansionWorkflowError):
        LunaExpansionWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=wrong_slot, epoch=7007)
    CertificationRoleIngress.configure_isolated_lineage(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=packet.content_hash,
        role_session_lineage=lineage,
        run_id=config.run_id,
        attempt_index=1,
        retry_index=1,
    )
    receipt = CertificationRoleIngress.stage_private_exchange(
        tmp_path,
        role="TeacherScorer",
        input_packet_hash=packet.content_hash,
        raw_role_output=contract_example_teacher_raw(packet),
        role_session_lineage=lineage,
        execution_profile="isolated_subagent",
        run_id=config.run_id,
        attempt_index=1,
        retry_index=1,
    )
    recovered = LunaExpansionWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=receipt, epoch=7007)
    assert recovered.events[-1].payload["event_type"] == "HOLDOUT_TEACHER_OUTPUT_ACCEPTED"
    fresh = LunaExpansionWorkflow.resume(tmp_path, config.run_id, epoch=0)
    assert fresh.events[-1].content_hash == recovered.events[-1].content_hash

    exhausted_config = FixtureLunaExpansionConfig.from_mapping(
        {**config.immutable_input_payload, "run_id": "ticket07-timeout-exhausted", "holdout_seed": 37007}
    )
    exhausted = LunaExpansionWorkflow.run_until_role_input(tmp_path, exhausted_config, epoch=7007)
    assert exhausted.student_packet is not None
    exhausted_packet = exhausted.student_packet
    exhausted_lineage = "/root/ticket07_timeout_exhausted_student"
    for retry_index in range(4):
        CertificationRoleIngress.configure_isolated_lineage(
            tmp_path,
            role="StudentJudge",
            input_packet_hash=exhausted_packet.content_hash,
            role_session_lineage=exhausted_lineage,
            run_id=exhausted_config.run_id,
            attempt_index=1,
            retry_index=retry_index,
        )
        exhausted = LunaExpansionWorkflow.record_role_timeout(
            tmp_path,
            exhausted_config.run_id,
            role="StudentJudge",
            input_packet_hash=exhausted_packet.content_hash,
            role_session_lineage=exhausted_lineage,
            retry_index=retry_index,
            epoch=7007,
        )
    assert exhausted.terminal
    assert exhausted.events[-2].payload["event_type"] == "EXPANSION_FAILED_CLOSED"
    closed_hash = cast(str, cast(dict[str, object], exhausted.events[-1].payload["details"])["run_closed_hash"])
    closed = ArtifactStore(tmp_path).read(closed_hash, expected_schema_name="RunClosed")
    assert closed.payload["reason_code"] == "ISOLATED_ROLE_BOUNDARY_EXHAUSTED"
    before = len(list(ArtifactStore(tmp_path).artifact_dir.glob("*.json")))
    recertified = LunaExpansionWorkflow.resume(tmp_path, exhausted_config.run_id, epoch=0)
    assert recertified.terminal and recertified.retained_judge_pack.content_hash == (
        exhausted.source_judge_pack.content_hash
    )
    assert len(list(ArtifactStore(tmp_path).artifact_dir.glob("*.json"))) == before
    exhausted_run_dir = tmp_path / "runs" / exhausted_config.run_id
    sorted((exhausted_run_dir / "events").glob("*.ref"))[-1].unlink()
    (exhausted_run_dir / "closed.ref").unlink()
    with pytest.raises(LunaExpansionWorkflowError):
        LunaExpansionWorkflow.resume(tmp_path, exhausted_config.run_id, epoch=7007)

    forged_config = FixtureLunaExpansionConfig.from_mapping(
        {**config.immutable_input_payload, "run_id": "ticket07-forged-order", "holdout_seed": 47007}
    )
    forged = LunaExpansionWorkflow.bootstrap(tmp_path, forged_config, epoch=7007)
    journal = RunJournal(
        tmp_path,
        ArtifactStore(tmp_path),
        forged_config.run_id,
        input_schema_name="LunaExpansionInput",
        input_schema_version="1.0.0",
    )
    journal.append(
        7007,
        "HOLDOUT_REQUESTED",
        {"boundary_sequence": 1, "holdout_request_hash": sha256_hex(b"forged"), "level": 16},
        expected_sequence=len(forged.events) + 1,
        expected_previous_hash=forged.events[-1].content_hash,
    )
    with pytest.raises(LunaExpansionWorkflowError):
        LunaExpansionWorkflow.resume(tmp_path, forged_config.run_id, epoch=7007)


def test_terminal_luna4_pack_is_refused_before_any_expansion_write(tmp_path: Path) -> None:
    from clawrl.judge.luna4_workflow import Luna4CertificationWorkflow

    source_run, exhaustion_hash = _try8_exhausted(tmp_path)
    luna4 = _luna4_config(
        run_id="ticket06-luna4-source-for-ticket07",
        source_run=source_run,
        exhaustion_hash=exhaustion_hash,
    )
    snapshot = Luna4CertificationWorkflow.run_until_role_input(tmp_path, luna4, epoch=6006)
    snapshot = _drive_luna4_attempt(tmp_path, luna4, snapshot, student_fault=None, epoch=6006)
    assert snapshot.terminal and snapshot.judge_pack is not None
    assert snapshot.judge_pack.payload["certification_level"] == 4
    expansion_run = "ticket07-refuse-luna4"
    with pytest.raises(LunaExpansionWorkflowError):
        LunaExpansionWorkflow.bootstrap(
            tmp_path,
            FixtureLunaExpansionConfig(
                run_id=expansion_run,
                source_run_id=luna4.run_id,
                source_certification_report_hash=snapshot.judge_pack.content_hash,
                alignment_policy_hash=cast(str, snapshot.judge_pack.payload["alignment_policy_hash"]),
                holdout_seed=27007,
                role_seed=27707,
            ),
            epoch=7007,
        )
    assert not (tmp_path / "runs" / expansion_run).exists()
