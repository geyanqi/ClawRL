"""Ticket 04 isolated TeacherScorer ingress and TeacherLabelSet completion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from clawrl.adapters.scorers.role_isolation import (
    RoleIsolationIngress,
    RoleIsolationIngressReceipt,
    RoleIsolationOutputError,
)
from clawrl.adapters.scorers.teacher import (
    TeacherOutputError,
    TeacherScorerIngress,
    teacher_output_schema_contract,
)
from clawrl.artifacts import canonical_json_bytes
from clawrl.judge.fit_workflow import FitTrajectoryWorkflow, FitWorkflowError
from tests.e2e.test_ticket04_phase_a_fit_workflow import _config, _dataset, _public_snapshot
from tests.fixtures.ticket04_roles import contract_example_teacher_raw

ROLE_LINEAGE = "/root/ticket04_teacher_scorer_retry"
REAL_RAW_PATH = Path(__file__).parents[1] / "fixtures" / "ticket04_teacher_scorer_retry_output.raw.json"
OPTIMIZER_RAW_PATH = Path(__file__).parents[1] / "fixtures" / "ticket04_prompt_optimizer_output.raw.json"
AUDITOR_RAW_PATH = Path(__file__).parents[1] / "fixtures" / "ticket04_alignment_auditor_output.raw.json"
OPTIMIZER_LINEAGE = "/root/ticket04_prompt_optimizer"
AUDITOR_LINEAGE = "/root/ticket04_alignment_auditor"


def _exact_role_raw(path: Path, *, size: int, digest: str) -> bytes:
    raw = path.read_bytes()
    assert not raw.endswith(b"\n")
    assert len(raw) == size
    assert hashlib.sha256(raw).hexdigest() == digest
    return raw


def _json_wire(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _phase_a(root: Path, run_id: str = "ticket04-teacher-labels") -> tuple[Any, Any]:
    dataset_hash, trace_id = _dataset(root)
    config = _config(dataset_hash, trace_id, run_id=run_id)
    snapshot = FitTrajectoryWorkflow.run_once(root, config, epoch=4004)
    assert snapshot.teacher_packet is not None and snapshot.phase_a_checkpoint is not None
    return config, snapshot


def test_private_role_ingress_completes_labels_and_role_packets_then_replays_read_only(tmp_path: Path) -> None:
    config, phase_a = _phase_a(tmp_path)
    packet = phase_a.teacher_packet
    assert packet is not None
    declared_output = cast(dict[str, Any], packet.payload["output_schema"])
    assert declared_output == teacher_output_schema_contract()
    assert set(cast(list[str], declared_output["required"])) == {
        "input_packet_content_hash",
        "input_payload_canonical_hash",
        "labels",
        "packet_id",
        "schema_version",
        "scoring_session_id",
        "seed",
        "thread_sessions",
        "turns",
    }
    assert set(cast(list[str], declared_output["label_required"])) == {
        "dimension_scores",
        "evidence",
        "failure_tags",
        "input_content_hash",
        "item_index",
        "prompt_hash",
        "response_hash",
        "scalar_micros",
        "thread_id",
        "trajectory_id",
        "turn_index",
        "wave_index",
    }
    raw = _exact_role_raw(
        REAL_RAW_PATH,
        size=57_040,
        digest="c2bd859c267d6c6e7d21c27d5922709c3c6413f00c5f6e10717378349d90f4cd",
    )
    receipt = TeacherScorerIngress.stage_private_exchange(
        tmp_path,
        input_packet_hash=packet.content_hash,
        raw_role_output=raw,
        role_session_lineage=ROLE_LINEAGE,
    )
    accepted = FitTrajectoryWorkflow.submit_teacher_output(
        tmp_path,
        config.run_id,
        role_ingress=receipt,
        epoch=4004,
    )
    while accepted.prompt_optimizer_packet is None or accepted.alignment_auditor_packet is None:
        accepted = FitTrajectoryWorkflow.resume(tmp_path, config.run_id, epoch=4004)
    assert accepted.teacher_label_set is not None
    label_set = accepted.teacher_label_set
    assert label_set.payload["label_count"] == 32
    labels = cast(list[dict[str, Any]], label_set.payload["labels"])
    assert len(labels) == 32
    assert len({label["trajectory_id"] for label in labels}) == 32
    assert len(set(cast(list[str], label_set.payload["input_content_hashes"]))) == 32
    scoring_session = cast(dict[str, Any], label_set.payload["scoring_session"])
    assert scoring_session["items_per_turn"] == 4
    assert scoring_session["max_resident_subthreads"] == 5
    assert len(scoring_session["thread_sessions"]) == 5
    assert len(scoring_session["turns"]) == 8
    lineage = cast(dict[str, Any], label_set.payload["lineage"])
    assert lineage["input_packet_hash"] == packet.content_hash
    assert lineage["raw_output_hash"] == receipt.raw_output_hash
    assert lineage["normalized_output_hash"] == receipt.normalized_output_hash
    assert lineage["role_invocation_audit_hash"] == receipt.role_invocation_audit_hash
    assert lineage["role_session_lineage"] == ROLE_LINEAGE
    private_raw = (
        tmp_path / "private-boundary" / "teacher-scorer-ingress" / receipt.ingress_hash / "role-output.raw.json"
    )
    assert private_raw.read_bytes() == raw
    assert not accepted.terminal
    optimizer_packet = accepted.prompt_optimizer_packet
    auditor_packet = accepted.alignment_auditor_packet
    assert optimizer_packet is not None and auditor_packet is not None
    optimizer_text = canonical_json_bytes(optimizer_packet.payload).decode().lower()
    auditor_text = canonical_json_bytes(auditor_packet.payload).decode().lower()
    assert "audit" not in optimizer_text
    assert "generator" not in optimizer_text
    assert "generator" not in auditor_text
    optimizer_session = cast(dict[str, Any], optimizer_packet.payload["session"])
    auditor_session = cast(dict[str, Any], auditor_packet.payload["session"])
    assert optimizer_session["optimizer_session_id"] != auditor_session["auditor_session_id"]
    authority = cast(dict[str, Any], auditor_packet.payload["read_only_authority"])
    assert authority["may_modify_prompt"] is False
    assert authority["may_modify_policy"] is False

    optimizer_raw = _exact_role_raw(
        OPTIMIZER_RAW_PATH,
        size=2_822,
        digest="7592d1d32e472edd76c1e26ce11d82ee5296546d41f491fc1a65d6a0f02119bc",
    )
    auditor_raw = _exact_role_raw(
        AUDITOR_RAW_PATH,
        size=318,
        digest="5329f6646e4d9ab41649c26eeabbca56c87e36a06a64443c28da2b248ac5ede4",
    )
    optimizer_receipt = RoleIsolationIngress.stage_private_exchange(
        tmp_path,
        role_type="PromptOptimizer",
        input_packet_hash=optimizer_packet.content_hash,
        raw_role_output=optimizer_raw,
        role_session_lineage=OPTIMIZER_LINEAGE,
    )
    auditor_receipt = RoleIsolationIngress.stage_private_exchange(
        tmp_path,
        role_type="AlignmentAuditor",
        input_packet_hash=auditor_packet.content_hash,
        raw_role_output=auditor_raw,
        role_session_lineage=AUDITOR_LINEAGE,
    )
    out_of_order = FitTrajectoryWorkflow.submit_role_output(
        tmp_path,
        config.run_id,
        role_ingress=auditor_receipt,
        epoch=4004,
    )
    assert out_of_order.alignment_auditor_output is not None
    assert out_of_order.prompt_optimizer_output is None
    waiting = FitTrajectoryWorkflow.resume(tmp_path, config.run_id, epoch=4004)
    assert waiting.events == out_of_order.events
    both = FitTrajectoryWorkflow.submit_role_output(
        tmp_path,
        config.run_id,
        role_ingress=optimizer_receipt,
        epoch=4004,
    )
    assert both.prompt_optimizer_output is not None
    assert both.alignment_auditor_output is not None
    attested = FitTrajectoryWorkflow.resume(tmp_path, config.run_id, epoch=4004)
    assert attested.role_isolation_attestation is not None
    attestation = attested.role_isolation_attestation
    assert attestation.payload["status"] == "passed"
    optimizer_attestation = cast(dict[str, Any], attestation.payload["prompt_optimizer"])
    auditor_attestation = cast(dict[str, Any], attestation.payload["alignment_auditor"])
    assert optimizer_attestation["raw_output_hash"] == optimizer_receipt.raw_output_hash
    assert auditor_attestation["raw_output_hash"] == auditor_receipt.raw_output_hash
    terminal = FitTrajectoryWorkflow.resume(tmp_path, config.run_id, epoch=4004)
    assert terminal.terminal
    assert terminal.events[-1].payload["event_type"] == "RUN_CLOSED"
    terminal_details = cast(dict[str, Any], terminal.events[-1].payload["details"])
    assert set(terminal_details) == {"run_closed_hash"}
    for role_type, ingress_hash, expected_raw in (
        ("PromptOptimizer", optimizer_receipt.ingress_hash, optimizer_raw),
        ("AlignmentAuditor", auditor_receipt.ingress_hash, auditor_raw),
    ):
        private = (
            tmp_path / "private-boundary" / "role-isolation-ingress" / role_type / ingress_hash / "role-output.raw.json"
        )
        assert private.read_bytes() == expected_raw

    before = _public_snapshot(tmp_path)
    replay = FitTrajectoryWorkflow.resume(tmp_path, config.run_id, epoch=9004)
    after = _public_snapshot(tmp_path)
    assert replay.teacher_label_set is not None
    assert replay.teacher_label_set.content_hash == label_set.content_hash
    assert replay.prompt_optimizer_packet is not None
    assert replay.alignment_auditor_packet is not None
    assert replay.role_isolation_attestation is not None
    assert replay.role_isolation_attestation.content_hash == attestation.content_hash
    assert replay.terminal
    assert before == after


def test_invalid_or_missing_teacher_labels_fail_closed_before_journal_commit(tmp_path: Path) -> None:
    config, phase_a = _phase_a(tmp_path, "ticket04-invalid-teacher")
    packet = phase_a.teacher_packet
    assert packet is not None
    baseline = json.loads(contract_example_teacher_raw(packet))
    assert isinstance(baseline, dict)
    invalid_values: list[dict[str, object]] = []

    missing = json.loads(canonical_json_bytes(baseline))
    missing["labels"].pop()
    invalid_values.append(missing)

    extra = json.loads(canonical_json_bytes(baseline))
    extra["unexpected"] = "closed-world"
    invalid_values.append(extra)

    bool_score = json.loads(canonical_json_bytes(baseline))
    bool_score["labels"][0]["dimension_scores"]["correctness"] = True
    invalid_values.append(bool_score)

    wrong_turn = json.loads(canonical_json_bytes(baseline))
    wrong_turn["labels"][0]["turn_index"] = 8
    invalid_values.append(wrong_turn)

    duplicate = json.loads(canonical_json_bytes(baseline))
    duplicate["labels"][1] = duplicate["labels"][0]
    invalid_values.append(duplicate)

    for value in invalid_values:
        with pytest.raises(TeacherOutputError):
            TeacherScorerIngress.stage_private_exchange(
                tmp_path,
                input_packet_hash=packet.content_hash,
                raw_role_output=canonical_json_bytes(value),
                role_session_lineage=ROLE_LINEAGE,
            )
    with pytest.raises(TeacherOutputError):
        TeacherScorerIngress.stage_private_exchange(
            tmp_path,
            input_packet_hash=packet.content_hash,
            raw_role_output=contract_example_teacher_raw(packet),
            role_session_lineage="/root/not-isolated-teacher",
        )
    current = FitTrajectoryWorkflow.resume(tmp_path, config.run_id, epoch=4004)
    assert current.phase_a_checkpoint is not None
    assert current.teacher_label_set is None
    assert current.events[-1].payload["event_type"] == "PHASE_A_CHECKPOINT"


def test_private_raw_tamper_and_receipt_substitution_fail_recertification(tmp_path: Path) -> None:
    config, phase_a = _phase_a(tmp_path, "ticket04-tamper-teacher")
    packet = phase_a.teacher_packet
    assert packet is not None
    receipt = TeacherScorerIngress.stage_private_exchange(
        tmp_path,
        input_packet_hash=packet.content_hash,
        raw_role_output=contract_example_teacher_raw(packet),
        role_session_lineage=ROLE_LINEAGE,
    )
    with pytest.raises(FitWorkflowError):
        FitTrajectoryWorkflow.submit_teacher_output(
            tmp_path,
            config.run_id,
            role_ingress=replace(receipt, raw_output_hash="f" * 64),
            epoch=4004,
        )
    raw_path = tmp_path / "private-boundary" / "teacher-scorer-ingress" / receipt.ingress_hash / "role-output.raw.json"
    raw_path.unlink()
    with pytest.raises(FitWorkflowError):
        FitTrajectoryWorkflow.submit_teacher_output(
            tmp_path,
            config.run_id,
            role_ingress=receipt,
            epoch=4004,
        )


def test_conflicting_second_teacher_output_is_rejected_after_acceptance(tmp_path: Path) -> None:
    config, phase_a = _phase_a(tmp_path, "ticket04-teacher-conflict")
    packet = phase_a.teacher_packet
    assert packet is not None
    raw = contract_example_teacher_raw(packet)
    first = TeacherScorerIngress.stage_private_exchange(
        tmp_path,
        input_packet_hash=packet.content_hash,
        raw_role_output=raw,
        role_session_lineage=ROLE_LINEAGE,
    )
    accepted = FitTrajectoryWorkflow.submit_teacher_output(
        tmp_path,
        config.run_id,
        role_ingress=first,
        epoch=4004,
    )
    assert accepted.events[-1].payload["event_type"] == "TEACHER_OUTPUT_ACCEPTED"
    decoded = json.loads(raw)
    decoded["labels"][0]["evidence"]["correctness"] += " changed"
    second = TeacherScorerIngress.stage_private_exchange(
        tmp_path,
        input_packet_hash=packet.content_hash,
        raw_role_output=canonical_json_bytes(decoded),
        role_session_lineage=ROLE_LINEAGE,
    )
    with pytest.raises(FitWorkflowError):
        FitTrajectoryWorkflow.submit_teacher_output(
            tmp_path,
            config.run_id,
            role_ingress=second,
            epoch=4004,
        )


def test_role_outputs_reject_cross_role_leaks_aggregate_tamper_and_wrong_lineage(tmp_path: Path) -> None:
    config, phase_a = _phase_a(tmp_path, "ticket04-invalid-role-isolation")
    teacher_packet = phase_a.teacher_packet
    assert teacher_packet is not None
    teacher_receipt = TeacherScorerIngress.stage_private_exchange(
        tmp_path,
        input_packet_hash=teacher_packet.content_hash,
        raw_role_output=_exact_role_raw(
            REAL_RAW_PATH,
            size=57_040,
            digest="c2bd859c267d6c6e7d21c27d5922709c3c6413f00c5f6e10717378349d90f4cd",
        ),
        role_session_lineage=ROLE_LINEAGE,
    )
    snapshot = FitTrajectoryWorkflow.submit_teacher_output(
        tmp_path,
        config.run_id,
        role_ingress=teacher_receipt,
        epoch=4004,
    )
    while snapshot.prompt_optimizer_packet is None:
        snapshot = FitTrajectoryWorkflow.resume(tmp_path, config.run_id, epoch=4004)
    optimizer_packet = snapshot.prompt_optimizer_packet
    auditor_packet = snapshot.alignment_auditor_packet
    assert optimizer_packet is not None and auditor_packet is not None
    optimizer_raw = _exact_role_raw(
        OPTIMIZER_RAW_PATH,
        size=2_822,
        digest="7592d1d32e472edd76c1e26ce11d82ee5296546d41f491fc1a65d6a0f02119bc",
    )
    auditor_raw = _exact_role_raw(
        AUDITOR_RAW_PATH,
        size=318,
        digest="5329f6646e4d9ab41649c26eeabbca56c87e36a06a64443c28da2b248ac5ede4",
    )
    altered = json.loads(optimizer_raw)
    altered["aggregate_diagnostics"]["scalar_micros"]["mean"] = 0
    invalid_exchanges = (
        ("PromptOptimizer", optimizer_packet.content_hash, _json_wire(altered), OPTIMIZER_LINEAGE),
        ("PromptOptimizer", optimizer_packet.content_hash, optimizer_raw, "/root/not-the-optimizer"),
        ("PromptOptimizer", optimizer_packet.content_hash, auditor_raw, OPTIMIZER_LINEAGE),
        ("AlignmentAuditor", auditor_packet.content_hash, optimizer_raw, AUDITOR_LINEAGE),
    )
    for role_type, packet_hash, raw, lineage in invalid_exchanges:
        with pytest.raises(RoleIsolationOutputError):
            RoleIsolationIngress.stage_private_exchange(
                tmp_path,
                role_type=cast(Any, role_type),
                input_packet_hash=packet_hash,
                raw_role_output=raw,
                role_session_lineage=lineage,
            )
    failing_auditor = json.loads(auditor_raw)
    failing_auditor["isolation_verdict"] = "fail"
    failing_auditor["violations"] = ["optimizer received a prohibited role audit"]
    with pytest.raises(RoleIsolationOutputError):
        RoleIsolationIngress.stage_private_exchange(
            tmp_path,
            role_type="AlignmentAuditor",
            input_packet_hash=auditor_packet.content_hash,
            raw_role_output=canonical_json_bytes(failing_auditor),
            role_session_lineage=AUDITOR_LINEAGE,
        )
    assert snapshot.events[-1].payload["event_type"] == "ROLE_ISOLATION_PACKETS_COMMITTED"


def test_role_output_idempotency_conflict_and_private_tamper_fail_closed(tmp_path: Path) -> None:
    config, phase_a = _phase_a(tmp_path, "ticket04-role-conflict")
    teacher_packet = phase_a.teacher_packet
    assert teacher_packet is not None
    teacher_receipt = TeacherScorerIngress.stage_private_exchange(
        tmp_path,
        input_packet_hash=teacher_packet.content_hash,
        raw_role_output=_exact_role_raw(
            REAL_RAW_PATH,
            size=57_040,
            digest="c2bd859c267d6c6e7d21c27d5922709c3c6413f00c5f6e10717378349d90f4cd",
        ),
        role_session_lineage=ROLE_LINEAGE,
    )
    snapshot = FitTrajectoryWorkflow.submit_teacher_output(
        tmp_path,
        config.run_id,
        role_ingress=teacher_receipt,
        epoch=4004,
    )
    while snapshot.prompt_optimizer_packet is None:
        snapshot = FitTrajectoryWorkflow.resume(tmp_path, config.run_id, epoch=4004)
    packet = snapshot.prompt_optimizer_packet
    assert packet is not None
    raw = _exact_role_raw(
        OPTIMIZER_RAW_PATH,
        size=2_822,
        digest="7592d1d32e472edd76c1e26ce11d82ee5296546d41f491fc1a65d6a0f02119bc",
    )
    first = RoleIsolationIngress.stage_private_exchange(
        tmp_path,
        role_type="PromptOptimizer",
        input_packet_hash=packet.content_hash,
        raw_role_output=raw,
        role_session_lineage=OPTIMIZER_LINEAGE,
    )
    accepted = FitTrajectoryWorkflow.submit_role_output(
        tmp_path,
        config.run_id,
        role_ingress=first,
        epoch=4004,
    )
    replayed = FitTrajectoryWorkflow.submit_role_output(
        tmp_path,
        config.run_id,
        role_ingress=first,
        epoch=4004,
    )
    assert replayed.events == accepted.events
    changed = json.loads(raw)
    changed["candidate_prompt"] += " Preserve the same calibrated dimensions."
    second = RoleIsolationIngress.stage_private_exchange(
        tmp_path,
        role_type="PromptOptimizer",
        input_packet_hash=packet.content_hash,
        raw_role_output=_json_wire(changed),
        role_session_lineage=OPTIMIZER_LINEAGE,
    )
    with pytest.raises(FitWorkflowError):
        FitTrajectoryWorkflow.submit_role_output(
            tmp_path,
            config.run_id,
            role_ingress=second,
            epoch=4004,
        )
    substituted = replace(first, role_type="AlignmentAuditor")
    with pytest.raises(FitWorkflowError):
        FitTrajectoryWorkflow.submit_role_output(
            tmp_path,
            config.run_id,
            role_ingress=cast(RoleIsolationIngressReceipt, substituted),
            epoch=4004,
        )
    raw_path = (
        tmp_path
        / "private-boundary"
        / "role-isolation-ingress"
        / "PromptOptimizer"
        / first.ingress_hash
        / "role-output.raw.json"
    )
    raw_path.unlink()
    with pytest.raises(FitWorkflowError):
        FitTrajectoryWorkflow.resume(tmp_path, config.run_id, epoch=4004)
