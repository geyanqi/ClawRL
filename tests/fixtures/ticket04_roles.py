"""Schema examples for testing Ticket 04 role-boundary rejection and normalization."""

from __future__ import annotations

from typing import Any, cast

from clawrl.artifacts import Artifact, canonical_json_bytes, sha256_hex


def contract_example_teacher_raw(packet: Artifact) -> bytes:
    """Build a content-derived schema example; this is not an isolated role response."""

    session = cast(dict[str, Any], packet.payload["scoring_session"])
    packet_turns = cast(list[dict[str, Any]], session["turns"])
    threads = cast(list[str], session["resident_thread_ids"])
    thread_sessions = []
    for thread in threads:
        thread_sessions.append(
            {
                "scoring_session_id": session["scoring_session_id"],
                "thread_id": thread,
                "turn_indices": [turn["turn_index"] for turn in packet_turns if turn["thread_id"] == thread],
            }
        )
    turns = []
    labels = []
    dimensions = ("correctness", "reasoning_quality", "task_completion", "tool_discipline")
    for turn in packet_turns:
        items = cast(list[dict[str, Any]], turn["items"])
        turns.append(
            {
                "items": [
                    {
                        "content_hash": item["content_hash"],
                        "prompt_hash": item["prompt_hash"],
                        "response_hash": item["response_hash"],
                        "trajectory_id": item["trajectory_id"],
                    }
                    for item in items
                ],
                "thread_id": turn["thread_id"],
                "turn_index": turn["turn_index"],
                "wave_index": turn["wave_index"],
            }
        )
        for item_index, item in enumerate(items, start=1):
            response_hash = cast(str, item["response_hash"])
            scores = {
                name: 40 + int(response_hash[index * 2 : index * 2 + 2], 16) % 61
                for index, name in enumerate(dimensions)
            }
            labels.append(
                {
                    "dimension_scores": scores,
                    "evidence": {
                        name: f"contract-{name}-{response_hash[index * 4 : index * 4 + 8]}"
                        for index, name in enumerate(dimensions)
                    },
                    "failure_tags": [
                        {"dimension": name, "severity": "minor", "type": "contract_low_score"}
                        for name, score in scores.items()
                        if score < 50
                    ],
                    "input_content_hash": item["content_hash"],
                    "item_index": item_index,
                    "prompt_hash": item["prompt_hash"],
                    "response_hash": item["response_hash"],
                    "scalar_micros": sum(scores.values()) * 250_000,
                    "thread_id": turn["thread_id"],
                    "trajectory_id": item["trajectory_id"],
                    "turn_index": turn["turn_index"],
                    "wave_index": turn["wave_index"],
                }
            )
    return canonical_json_bytes(
        {
            "input_packet_content_hash": packet.content_hash,
            "input_payload_canonical_hash": sha256_hex(canonical_json_bytes(packet.payload)),
            "labels": labels,
            "packet_id": packet.payload["packet_id"],
            "schema_version": "teacher-scorer-output/1.0.0",
            "scoring_session_id": session["scoring_session_id"],
            "seed": packet.payload["seed"],
            "thread_sessions": thread_sessions,
            "turns": turns,
        }
    )
