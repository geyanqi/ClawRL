"""Content-derived Ticket 05 schema examples; not substitutes for isolated roles."""

from __future__ import annotations

import re
from typing import Any, cast

from clawrl.adapters.scorers.certification_roles import compute_aggregate_diagnostics
from clawrl.artifacts import Artifact, canonical_json_bytes, sha256_hex


def contract_example_teacher_raw(packet: Artifact) -> bytes:
    """Exercise the teacher wire contract with a teacher-specific scorer."""

    if packet.payload.get("role") != "teacher_scorer":
        raise ValueError("teacher fixture requires a TeacherScorer packet")
    session = cast(dict[str, Any], packet.payload["scoring_session"])
    turns = cast(list[dict[str, Any]], session["turns"])
    labels: list[dict[str, Any]] = []
    for turn in turns:
        for item in cast(list[dict[str, Any]], turn["items"]):
            prompt = cast(str, item["prompt"])
            response = cast(str, item["response"])
            dimensions, relation = _teacher_visible_scores(prompt, response)
            labels.append(
                {
                    "dimension_scores": dimensions,
                    "evidence": (
                        "Teacher independently applied the Initial Eval Rubric: "
                        f"visible answer relation is {relation}; "
                        f"response excerpt `{response[:160]}` supports these dimension scores."
                    ),
                    "item_hash": item["item_hash"],
                    "scalar_micros": sum(dimensions.values()) * 250_000,
                }
            )
    return _score_wire(packet, labels, "holdout-teacher-output/1.0.0")


def contract_example_student_raw(packet: Artifact, *, calibration_fault: str | None = None) -> bytes:
    """Exercise a separate student boundary without reading teacher inputs."""

    if packet.payload.get("role") != "student_judge":
        raise ValueError("student fixture requires a StudentJudge packet")
    if calibration_fault not in {None, "polarity_inversion"}:
        raise ValueError("student calibration fault is invalid")
    session = cast(dict[str, Any], packet.payload["scoring_session"])
    turns = cast(list[dict[str, Any]], session["turns"])
    candidate_prompt_ref = cast(dict[str, Any], packet.payload["candidate_prompt"])
    candidate_prompt_contract = cast(dict[str, Any], candidate_prompt_ref["contract"])
    candidate_prompt = cast(str, candidate_prompt_contract["text"])
    labels: list[dict[str, Any]] = []
    for turn in turns:
        for item in cast(list[dict[str, Any]], turn["items"]):
            prompt = cast(str, item["prompt"])
            response = cast(str, item["response"])
            dimensions, relation, candidate_focus = _student_visible_scores(prompt, response, candidate_prompt)
            if calibration_fault == "polarity_inversion":
                dimensions = {
                    **dimensions,
                    "correctness": 100 - dimensions["correctness"],
                    "reasoning_quality": max(0, dimensions["reasoning_quality"] - 35),
                }
            labels.append(
                {
                    "dimension_scores": dimensions,
                    "evidence": (
                        f"Student applied candidate criteria for {candidate_focus} to visible text: "
                        f"answer relation is {relation}; "
                        f"response excerpt `{response[:160]}` yields an independent calibration."
                    ),
                    "item_hash": item["item_hash"],
                    "scalar_micros": sum(dimensions.values()) * 250_000,
                }
            )
    return _score_wire(packet, labels, "holdout-student-output/1.0.0")


def _teacher_visible_scores(prompt: str, response: str) -> tuple[dict[str, int], str]:
    expected, answer = _visible_arithmetic(prompt, response)
    relation = "missing" if answer is None or expected is None else "correct" if answer == expected else "incorrect"
    correctness = {"correct": 92, "incorrect": 25, "missing": 15}[relation]
    lowered = response.casefold()
    connectors = sum(term in lowered for term in ("because", "therefore", "check", "which gives"))
    sentences = max(1, sum(response.count(mark) for mark in ".!?"))
    reasoning = min(95, 24 + connectors * 13 + sentences * 7)
    completion = {"correct": 90, "incorrect": 46, "missing": 24}[relation]
    if "positive" in prompt.casefold() and "positive" not in lowered:
        completion -= 12
    unsupported_tool = any(term in lowered for term in ("calculator", "python", "tool")) and not any(
        term in lowered for term in ("tool evidence", "shown calculation")
    )
    return {
        "correctness": correctness,
        "reasoning_quality": reasoning,
        "task_completion": max(0, completion),
        "tool_discipline": 42 if unsupported_tool else 96,
    }, relation


def _student_visible_scores(prompt: str, response: str, candidate_prompt: str) -> tuple[dict[str, int], str, str]:
    expected, answer = _visible_arithmetic(prompt, response)
    relation = "missing" if answer is None or expected is None else "correct" if answer == expected else "incorrect"
    correctness = {"correct": 90, "incorrect": 28, "missing": 18}[relation]
    lowered = response.casefold()
    candidate = candidate_prompt.casefold()
    observable_rule = "observable" in candidate or "visible" in candidate
    checkable_rule = "checkable" in candidate or "inspectable" in candidate
    completion_rule = "deliverable" in candidate or "task completion" in candidate
    tool_rule = "tool discipline" in candidate or "tool-derived" in candidate
    reasoning_signals = sum(term in lowered for term in ("because", "check", "identify", "combine"))
    sentence_count = max(1, sum(response.count(mark) for mark in ".!?"))
    reasoning = min(94, 26 + reasoning_signals * 11 + sentence_count * 6)
    completion = {"correct": 88, "incorrect": 49, "missing": 27}[relation]
    if observable_rule and relation == "missing":
        correctness = max(0, correctness - 8)
    if checkable_rule and not any(term in lowered for term in ("because", "check", "which gives")):
        reasoning = max(0, reasoning - 14)
    if completion_rule and relation != "correct":
        completion = max(0, completion - 9)
    if "positive" in prompt.casefold() and "positive" not in lowered:
        completion -= 10
    unsupported_tool = "calculator" in lowered and "shown calculation" not in lowered
    tool_discipline = 45 if unsupported_tool and tool_rule else 94 if not unsupported_tool else 68
    focus = (
        ", ".join(
            name
            for enabled, name in (
                (observable_rule, "observable evidence"),
                (checkable_rule, "checkable reasoning"),
                (completion_rule, "explicit completion"),
                (tool_rule, "tool discipline"),
            )
            if enabled
        )
        or "general semantic quality"
    )
    return (
        {
            "correctness": correctness,
            "reasoning_quality": reasoning,
            "task_completion": max(0, completion),
            "tool_discipline": tool_discipline,
        },
        relation,
        focus,
    )


def _visible_arithmetic(prompt: str, response: str) -> tuple[int | None, int | None]:
    expression = re.search(r"(-?\d+)\s*([+\-*])\s*(-?\d+)", prompt)
    answer = re.search(r"(?:final answer|result|answer)\s*(?:is|=|:)\s*(-?\d+)", response, re.IGNORECASE)
    if expression is None:
        expected = None
    else:
        left, operator, right = int(expression.group(1)), expression.group(2), int(expression.group(3))
        expected = left + right if operator == "+" else left - right if operator == "-" else left * right
    return expected, int(answer.group(1)) if answer is not None else None


def _score_wire(packet: Artifact, labels: list[dict[str, Any]], schema: str) -> bytes:
    session = cast(dict[str, Any], packet.payload["scoring_session"])
    return canonical_json_bytes(
        {
            "input_packet_content_hash": packet.content_hash,
            "input_payload_canonical_hash": sha256_hex(canonical_json_bytes(packet.payload)),
            "labels": labels,
            "packet_id": packet.payload["packet_id"],
            "schema_version": schema,
            "seed": packet.payload["seed"],
            "session_id": session["session_id"],
            "turns": len(cast(list[object], session["turns"])),
        }
    )


def contract_example_auditor_raw(packet: Artifact) -> bytes:
    session = cast(dict[str, Any], packet.payload["session"])
    return canonical_json_bytes(
        {
            "aggregate_diagnostics": compute_aggregate_diagnostics(packet),
            "auditor_session_id": session["auditor_session_id"],
            "input_packet_content_hash": packet.content_hash,
            "input_payload_canonical_hash": sha256_hex(canonical_json_bytes(packet.payload)),
            "packet_id": packet.payload["packet_id"],
            "schema_version": "certification-auditor-output/1.0.0",
            "seed": packet.payload["seed"],
            "verdict": _verdict(packet),
        }
    )


def contract_example_optimizer_raw(packet: Artifact) -> bytes:
    session = cast(dict[str, Any], packet.payload["session"])
    current = cast(dict[str, Any], cast(dict[str, Any], packet.payload["current_trace_judge_prompt"])["contract"])
    diagnostics = cast(dict[str, int], packet.payload["aggregate_diagnostics"])
    return canonical_json_bytes(
        {
            "candidate_prompt": (
                f"{current['text']} Candidate {packet.payload['next_candidate_index']} correction: "
                f"calibrate absolute bias {diagnostics['absolute_bias_micros']} and pairwise agreement "
                f"{diagnostics['pairwise_agreement_micros']} using only aggregate diagnostics."
            ),
            "input_packet_content_hash": packet.content_hash,
            "input_payload_canonical_hash": sha256_hex(canonical_json_bytes(packet.payload)),
            "optimizer_session_id": session["optimizer_session_id"],
            "packet_id": packet.payload["packet_id"],
            "schema_version": "certification-optimizer-output/1.0.0",
            "seed": packet.payload["seed"],
        }
    )


def _verdict(packet: Artifact) -> str:
    diagnostics = compute_aggregate_diagnostics(packet)
    policy = cast(dict[str, int], packet.payload["alignment_policy"])
    passed = (
        diagnostics["mean_absolute_error_micros"] <= policy["max_mean_absolute_error_micros"]
        and diagnostics["p95_absolute_error_micros"] <= policy["max_p95_absolute_error_micros"]
        and diagnostics["absolute_bias_micros"] <= policy["max_absolute_bias_micros"]
        and diagnostics["pairwise_agreement_micros"] >= policy["min_pairwise_agreement_micros"]
        and diagnostics["student_variance_micros"] >= policy["min_student_variance_micros"]
        and diagnostics["failure_rate_micros"] <= policy["max_failure_rate_micros"]
    )
    return "pass" if passed else "fail"
