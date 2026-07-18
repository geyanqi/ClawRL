"""Ticket 05 closed-world contracts and durable holdout boundary faults."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from clawrl.adapters.generators.holdout import FixtureHoldoutGenerator, HoldoutBoundaryError
from clawrl.adapters.scorers.certification_roles import CertificationRoleIngress, CertificationRoleOutputError
from clawrl.artifacts import Artifact, ArtifactStore, canonical_json_bytes, sha256_hex
from clawrl.judge.certification_models import (
    AlignmentPolicy,
    BaseJudgePrompt,
    CertificationContractError,
    FixtureLuna8Config,
    LunaInferenceConfig,
    RewardSchema,
    Scalarizer,
    TraceJudgePrompt,
)
from clawrl.judge.certification_workflow import Luna8CertificationWorkflow, Luna8WorkflowError
from clawrl.judge.semantic_diversity import structural_semantic_profile


def _hash(label: str) -> str:
    return sha256_hex(label.encode())


def _config(*, schedule: tuple[str, ...], output_fault: str | None = None) -> FixtureLuna8Config:
    base = BaseJudgePrompt(
        prompt_id="ticket05-boundary-base",
        text="Evaluate this response against the immutable task and calibrated scalar dimensions only.",
    )
    base_hash = sha256_hex(
        canonical_json_bytes(
            {"payload": base.artifact_payload(), "schema_name": "BaseJudgePrompt", "schema_version": "1.0.0"}
        )
    )
    return FixtureLuna8Config(
        run_id=f"ticket05-boundary-{_hash(repr((schedule, output_fault)))[:12]}",
        ticket04_run_id="ticket04-boundary-contract",
        teacher_label_set_hash=_hash("labels"),
        base_prompt=base,
        trace_prompt=TraceJudgePrompt(
            prompt_id="ticket05-boundary-candidate-1",
            trace_id="tt-171805438493ea6c0d3dbd93303acbb1fbf2473c",
            candidate_index=1,
            parent_prompt_hash=base_hash,
            text="Inspect premises, intermediate reasoning, tool claims, and final completion for this trace.",
        ),
        policy=AlignmentPolicy.fixture_default(),
        luna_inference=LunaInferenceConfig(),
        reward_schema=RewardSchema(),
        scalarizer=Scalarizer(),
        algorithm_contract_hash=_hash("algorithm"),
        trace_prompt_derivation_hash=_hash("trace-prompt-derivation"),
        holdout_seed=5005,
        role_seed=5505,
        fault_schedule=schedule,
        output_fault=output_fault,
    )


def _request(store: ArtifactStore, config: FixtureLuna8Config, suffix: str = "base") -> Artifact:
    return store.put(
        "HoldoutGeneratorRequest",
        "1.0.0",
        {
            "attempt_id": f"la-{_hash(suffix)[:40]}",
            "attempt_index": 1,
            "fit_content_hashes": [_hash(f"fit-{index}") for index in range(32)],
            "generator_inference_hash": _hash("inference"),
            "generator_model_id": config.holdout_generator_model_id,
            "generator_profile_id": config.holdout_generator_profile_id,
            "holdout_seed": 12345,
            "output_fault": config.output_fault,
            "policy_hash": _hash("policy"),
            "prompt": "Determine whether the proposed bounded workflow preserves every stated invariant.",
            "request_schema_version": "holdout-generator-request/1.0.0",
            "trace_id": "tt-171805438493ea6c0d3dbd93303acbb1fbf2473c",
        },
    )


def test_holdout_boundary_retries_quarantines_late_output_and_restarts(tmp_path: Path) -> None:
    config = _config(schedule=("timeout", "delayed", "success"))
    store = ArtifactStore(tmp_path)
    request = _request(store, config)
    adapter = FixtureHoldoutGenerator(tmp_path, store, config)
    with pytest.raises(HoldoutBoundaryError):
        adapter.execute(request, boundary_sequence=2)
    first = adapter.execute(request, boundary_sequence=1)
    second = adapter.execute(request, boundary_sequence=2)
    third = adapter.execute(request, boundary_sequence=3)
    assert first.payload["status"] == second.payload["status"] == "retryable"
    assert third.payload["status"] == "succeeded"
    attempts = adapter.verify_attempt_chain(request, 3)
    assert [attempt.payload["directive"] for attempt in attempts] == ["timeout", "delayed", "success"]
    assert attempts[2].payload["late_completion_hashes"]
    raw = adapter.read_raw_batch(third, request)
    assert len(cast(list[object], raw.payload["items"])) == 32

    restarted = FixtureHoldoutGenerator(tmp_path, ArtifactStore(tmp_path), config)
    replay = restarted.execute(request, boundary_sequence=3)
    assert replay.content_hash == third.content_hash
    assert [item.content_hash for item in restarted.verify_attempt_chain(request, 3)] == [
        item.content_hash for item in attempts
    ]


@pytest.mark.parametrize(
    ("fault", "expected_count", "expected_marker"),
    [
        ("short_response", 32, "short"),
        ("duplicate_response", 32, "duplicate"),
        ("fit_reuse", 32, "forced_content_hash"),
        ("historical_reuse", 32, "reuse_historical"),
        ("wrong_count", 31, "wrong_count"),
    ],
)
def test_holdout_boundary_injects_malformed_batches_for_application_rejection(
    tmp_path: Path, fault: str, expected_count: int, expected_marker: str
) -> None:
    config = _config(schedule=("success",), output_fault=fault)
    store = ArtifactStore(tmp_path)
    request = _request(store, config, suffix=fault)
    adapter = FixtureHoldoutGenerator(tmp_path, store, config)
    observation = adapter.execute(request, boundary_sequence=1)
    raw = adapter.read_raw_batch(observation, request)
    items = cast(list[dict[str, Any]], raw.payload["items"])
    assert len(items) == expected_count
    if fault == "short_response":
        assert items[0]["response"] == "short"
    elif fault == "duplicate_response":
        assert items[0]["response"] == items[1]["response"]
    elif fault in {"fit_reuse", "historical_reuse"}:
        assert expected_marker in items[0]


def test_numeric_policy_attempt_limit_and_aggregation_are_closed_world() -> None:
    policy = AlignmentPolicy.fixture_default()
    assert policy.required_item_count == 32
    assert policy.max_prompt_attempts == 3
    assert set(policy.artifact_payload()) == {
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
    with pytest.raises(CertificationContractError):
        replace(policy, max_prompt_attempts=4)
    with pytest.raises(CertificationContractError):
        RewardSchema(aggregation=cast(Any, "hierarchical"))


def test_semantic_structure_does_not_treat_exact_operands_as_unique_provenance() -> None:
    first = structural_semantic_profile(
        "Compute 123 + 45 and state the answer.",
        "Because addition is direct, final answer is 168.",
    )
    second = structural_semantic_profile(
        "Compute 456 + 12 and state the answer.",
        "Because addition is direct, final answer is 468.",
    )
    assert first == second


def test_isolated_semantic_raw_packets_replay_through_terminal_workflow(tmp_path: Path) -> None:
    from tests.e2e.test_ticket05_luna8_certification import (
        _config as workflow_config,
    )
    from tests.e2e.test_ticket05_luna8_certification import (
        _ticket04_terminal,
    )
    from tests.fixtures.ticket05_roles import contract_example_student_raw

    fixture_root = Path(__file__).parents[1] / "fixtures"
    evidence = {
        "attempt1_teacher": {
            "file": "ticket05_semantic_attempt1_teacher_output.raw.json",
            "size": 13_164,
            "raw_hash": "df9ef5d36745a05954923da18e40b89ae203af0945006ec6339ba7a0466f0ca2",
            "role": "TeacherScorer",
            "lineage": "/root/ticket05_teacher_scorer_p1",
        },
        "attempt1_student": {
            "file": "ticket05_semantic_attempt1_student_output.raw.json",
            "size": 12_705,
            "raw_hash": "276b85cccabda96966e746d1bb31897a229269ce5ff10598eee95c08963d6d0e",
            "role": "StudentJudge",
            "lineage": "/root/ticket05_student_judge_p1",
        },
        "attempt1_auditor": {
            "file": "ticket05_semantic_attempt1_auditor_output.raw.json",
            "size": 612,
            "raw_hash": "de8e5de2f4889019b5476342f15a6944cbecfa88cf8cc8424f6032351904e147",
            "role": "AlignmentAuditor",
            "lineage": "/root/ticket05_alignment_auditor_p1",
        },
        "attempt1_optimizer": {
            "file": "ticket05_semantic_attempt1_optimizer_output.raw.json",
            "size": 1_970,
            "raw_hash": "aff1eb37cad08e8a9c7d3dcc3f113f661612d3cce124eb854d48b79c86c1ab47",
            "role": "PromptOptimizer",
            "lineage": "/root/ticket05_prompt_optimizer_p1",
        },
        "attempt2_teacher": {
            "file": "ticket05_semantic_teacher_output.raw.json",
            "size": 12_100,
            "raw_hash": "048b24e59859eeed23a9d1d5f213786c7b31ad2b971d0d530ac7a8a7a6ecf18f",
            "role": "TeacherScorer",
            "lineage": "/root/ticket05_teacher_scorer_p1a2",
        },
        "attempt2_student": {
            "file": "ticket05_semantic_student_output.raw.json",
            "size": 11_986,
            "raw_hash": "2d823ca4dd48126c12f6c409ebbb30e864dc7e947bededb5b9238451be25bd54",
            "role": "StudentJudge",
            "lineage": "/root/ticket05_student_judge_p1a2",
        },
        "attempt2_auditor": {
            "file": "ticket05_semantic_auditor_output.raw.json",
            "size": 610,
            "raw_hash": "7f0cbebf3085c45d8da28c0a18d616be9adaca52b62befe497b5ae660514ec7e",
            "role": "AlignmentAuditor",
            "lineage": "/root/ticket05_alignment_auditor_p1a2",
        },
    }

    ticket04_run, label_hash, trace_id = _ticket04_terminal(tmp_path)
    config = workflow_config(
        tmp_path,
        run_id="ticket05-independent-real-success-v2",
        ticket04_run_id=ticket04_run,
        teacher_label_set_hash=label_hash,
        trace_id=trace_id,
    )

    def stage(name: str, packet: Artifact) -> tuple[Any, Artifact]:
        item = evidence[name]
        raw = (fixture_root / cast(str, item["file"])).read_bytes()
        assert len(raw) == item["size"]
        assert sha256_hex(raw) == item["raw_hash"]
        decoded = json.loads(raw)
        assert decoded["input_packet_content_hash"] == packet.content_hash
        assert decoded["input_payload_canonical_hash"] == sha256_hex(canonical_json_bytes(packet.payload))
        assert decoded["packet_id"] == packet.payload["packet_id"]
        assert decoded["seed"] == packet.payload["seed"]
        if item["role"] == "AlignmentAuditor":
            assert (
                decoded["auditor_session_id"] == cast(dict[str, Any], packet.payload["session"])["auditor_session_id"]
            )
        elif item["role"] == "PromptOptimizer":
            assert (
                decoded["optimizer_session_id"]
                == cast(dict[str, Any], packet.payload["session"])["optimizer_session_id"]
            )
        receipt = CertificationRoleIngress.stage_private_exchange(
            tmp_path,
            role=cast(Any, item["role"]),
            input_packet_hash=packet.content_hash,
            raw_role_output=raw,
            role_session_lineage=cast(str, item["lineage"]),
            execution_profile="isolated_subagent",
        )
        contract = CertificationRoleIngress.load_public_contract(tmp_path, receipt)
        assert receipt.raw_output_hash == item["raw_hash"]
        assert contract.role_invocation_audit.payload["role_session_lineage"] == item["lineage"]
        assert cast(str, contract.role_invocation_audit.payload["role_session_lineage"]).startswith("/root/")
        assert contract.role_invocation_audit.payload["seed"] == packet.payload["seed"]
        return receipt, contract.normalized_output

    def normalized_schema_count(schema_name: str) -> int:
        return sum(
            json.loads(path.read_bytes())["schema_name"] == schema_name
            for path in (tmp_path / "artifacts").glob("*.json")
        )

    def reject_cross_session(
        name: str,
        packet: Artifact,
        *,
        session_field: str,
        normalized_schema: str,
        snapshot: Any,
    ) -> None:
        item = evidence[name]
        decoded = json.loads((fixture_root / cast(str, item["file"])).read_bytes())
        decoded[session_field] = "ticket05-cross-session-rejected"
        raw = canonical_json_bytes(decoded)
        assert decoded["input_packet_content_hash"] == packet.content_hash
        assert decoded["input_payload_canonical_hash"] == sha256_hex(canonical_json_bytes(packet.payload))
        before_events = tuple(event.content_hash for event in snapshot.events)
        before_normalized = normalized_schema_count(normalized_schema)
        with pytest.raises(CertificationRoleOutputError):
            CertificationRoleIngress.stage_private_exchange(
                tmp_path,
                role=cast(Any, item["role"]),
                input_packet_hash=packet.content_hash,
                raw_role_output=raw,
                role_session_lineage=cast(str, item["lineage"]),
                execution_profile="isolated_subagent",
            )
        resumed = Luna8CertificationWorkflow.resume(tmp_path, config.run_id, epoch=8505)
        assert tuple(event.content_hash for event in resumed.events) == before_events
        assert normalized_schema_count(normalized_schema) == before_normalized
        assert not resumed.terminal

    snapshot = Luna8CertificationWorkflow.run_until_role_input(tmp_path, config, epoch=8505)
    assert snapshot.attempt_index == 1
    assert snapshot.teacher_packet is not None and snapshot.student_packet is not None
    assert snapshot.teacher_packet.content_hash == ("2410eeae1f6924d1a0dfb5f2fdc33795074738ef805721dc699c00af07ad0463")
    assert snapshot.student_packet.content_hash == ("39bc6d723342d646be288d33dd1a8fbd115b991c0f76a949475b4a5eb3d7de82")
    baseline_example = json.loads(contract_example_student_raw(snapshot.student_packet))
    variant_payload = dict(snapshot.student_packet.payload)
    candidate_ref = dict(cast(dict[str, Any], variant_payload["candidate_prompt"]))
    candidate_contract = dict(cast(dict[str, Any], candidate_ref["contract"]))
    candidate_contract["text"] = (
        "Assess general semantic quality independently and return the required structured dimensions."
    )
    variant_candidate = ArtifactStore(tmp_path).put("TraceJudgePrompt", "1.0.0", candidate_contract)
    candidate_ref["contract"] = candidate_contract
    candidate_ref["artifact_hash"] = variant_candidate.content_hash
    variant_payload["candidate_prompt"] = candidate_ref
    variant_packet = ArtifactStore(tmp_path).put("HoldoutStudentInputPacket", "1.0.0", variant_payload)
    variant_example = json.loads(contract_example_student_raw(variant_packet))
    assert any(
        baseline["dimension_scores"] != variant["dimension_scores"]
        for baseline, variant in zip(baseline_example["labels"], variant_example["labels"], strict=True)
    )
    teacher1, teacher1_output = stage("attempt1_teacher", snapshot.teacher_packet)
    student1, student1_output = stage("attempt1_student", snapshot.student_packet)
    assert teacher1_output.payload["labels"] != student1_output.payload["labels"]
    for normalized in (teacher1_output, student1_output):
        labels = cast(list[dict[str, Any]], normalized.payload["labels"])
        assert len(labels) == 32
        assert len({label["evidence"] for label in labels}) == 32
        assert all(
            term not in cast(str, label["evidence"]).casefold()
            for label in labels
            for term in ("shared vector", "fixture shortcut", "provenance marker")
        )
    Luna8CertificationWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=student1, epoch=8505)
    Luna8CertificationWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=teacher1, epoch=8505)
    snapshot = Luna8CertificationWorkflow.resume(tmp_path, config.run_id, epoch=8505)
    assert snapshot.auditor_packet is not None
    assert snapshot.auditor_packet.content_hash == ("fc9215f12323657a099d113ce889ae73ad63fdc3dd78bc0beec9e0b091904355")
    reject_cross_session(
        "attempt1_auditor",
        snapshot.auditor_packet,
        session_field="auditor_session_id",
        normalized_schema="NormalizedCertificationAlignmentAuditorOutput",
        snapshot=snapshot,
    )
    auditor1, _ = stage("attempt1_auditor", snapshot.auditor_packet)
    Luna8CertificationWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=auditor1, epoch=8505)
    snapshot = Luna8CertificationWorkflow.resume(tmp_path, config.run_id, epoch=8505)
    assert snapshot.optimizer_packet is not None
    assert snapshot.optimizer_packet.content_hash == (
        "7a504e8ee619faa9cd0e6b265f8f0f3380ccf0f52e0e9f6ca7c7223f77883a08"
    )
    reject_cross_session(
        "attempt1_optimizer",
        snapshot.optimizer_packet,
        session_field="optimizer_session_id",
        normalized_schema="NormalizedCertificationPromptOptimizerOutput",
        snapshot=snapshot,
    )
    optimizer1, _ = stage("attempt1_optimizer", snapshot.optimizer_packet)
    Luna8CertificationWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=optimizer1, epoch=8505)
    Luna8CertificationWorkflow.resume(tmp_path, config.run_id, epoch=8505)
    snapshot = Luna8CertificationWorkflow.run_until_role_input(tmp_path, config, epoch=8505)
    assert snapshot.attempt_index == 2
    assert snapshot.teacher_packet is not None and snapshot.student_packet is not None
    assert snapshot.teacher_packet.content_hash == ("7c62ed64118a7fac700415254c29dde16da7afaa065d0720a85dadb6eac0e618")
    assert snapshot.student_packet.content_hash == ("4ef30ff796a0d38ea7057ad6391209f1e67aad27beb19844d4b588159e09eba6")
    teacher2, teacher2_output = stage("attempt2_teacher", snapshot.teacher_packet)
    student2, student2_output = stage("attempt2_student", snapshot.student_packet)
    teacher_labels = cast(list[dict[str, Any]], teacher2_output.payload["labels"])
    student_labels = cast(list[dict[str, Any]], student2_output.payload["labels"])
    assert teacher_labels != student_labels
    assert (
        sum(
            teacher["scalar_micros"] != student["scalar_micros"]
            for teacher, student in zip(teacher_labels, student_labels, strict=True)
        )
        == 32
    )
    Luna8CertificationWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=teacher2, epoch=8505)
    Luna8CertificationWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=student2, epoch=8505)
    snapshot = Luna8CertificationWorkflow.resume(tmp_path, config.run_id, epoch=8505)
    assert snapshot.auditor_packet is not None
    assert snapshot.auditor_packet.content_hash == ("a882b4f87dabb2888e992720b005e66b607ecd7a64e95a680ee506ba36a0241a")
    auditor2, _ = stage("attempt2_auditor", snapshot.auditor_packet)
    Luna8CertificationWorkflow.submit_role_output(tmp_path, config.run_id, role_ingress=auditor2, epoch=8505)
    snapshot = Luna8CertificationWorkflow.resume(tmp_path, config.run_id, epoch=8505)
    assert snapshot.terminal and snapshot.certification_report is not None
    assert snapshot.attempt_index == 2
    assert snapshot.certification_report.content_hash == (
        "9a2c1305c36ab2ba2194c3fdb757caee277d0b3178f03add1aa4065a2713fbc1"
    )

    for receipt, session_field in (
        (auditor1, "auditor_session_id"),
        (optimizer1, "optimizer_session_id"),
    ):
        raw_path = (
            tmp_path
            / "private-boundary"
            / "certification-role-ingress"
            / receipt.role
            / receipt.ingress_hash
            / "role-output.raw.json"
        )
        original = raw_path.read_bytes()
        tampered = json.loads(original)
        tampered[session_field] = "ticket05-historical-cross-session"
        raw_path.write_bytes(canonical_json_bytes(tampered))
        try:
            with pytest.raises(Luna8WorkflowError, match="persisted Luna@8 graph cannot be recertified"):
                Luna8CertificationWorkflow.resume(tmp_path, config.run_id, epoch=9999)
        finally:
            raw_path.write_bytes(original)
        assert Luna8CertificationWorkflow.resume(tmp_path, config.run_id, epoch=9999).terminal

    artifact_count = len(tuple((tmp_path / "artifacts").glob("*.json")))
    script = """
import json,sys
from clawrl.judge.certification_workflow import Luna8CertificationWorkflow
s=Luna8CertificationWorkflow.resume(sys.argv[1],sys.argv[2],epoch=9999)
print(json.dumps({'terminal':s.terminal,'attempt':s.attempt_index,'report':s.certification_report.content_hash}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), config.run_id],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout) == {
        "terminal": True,
        "attempt": 2,
        "report": snapshot.certification_report.content_hash,
    }
    assert len(tuple((tmp_path / "artifacts").glob("*.json"))) == artifact_count
