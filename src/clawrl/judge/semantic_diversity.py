"""Generic visible-text semantic diversity checks for Ticket 05 holdouts."""

from __future__ import annotations

import re
from collections import Counter
from typing import cast

from clawrl.artifacts import JsonValue, canonical_json_bytes, sha256_hex

_NUMBER = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?![A-Za-z0-9_])")
_SPACE = re.compile(r"\s+")
_ANSWER = re.compile(r"(?:final answer|result|answer)\s*(?:is|=|:)\s*(-?\d+)", re.IGNORECASE)
_EXPRESSION = re.compile(r"(-?\d+)\s*([+\-*])\s*(-?\d+)")
_REASONING_TERMS = ("because", "therefore", "since", "so", "verify", "check", "which gives")
_CONSTRAINT_TERMS = ("constraint", "must", "without", "include", "state", "show", "verify")


class SemanticDiversityError(ValueError):
    """Visible semantic structure cannot be safely measured."""


def structural_semantic_profile(prompt: str, response: str) -> dict[str, JsonValue]:
    """Describe visible semantic structure without item IDs, hashes, seeds, or opaque tokens."""

    if (
        type(prompt) is not str
        or type(response) is not str
        or not 8 <= len(prompt) <= 16_384
        or not 8 <= len(response) <= 16_384
    ):
        raise SemanticDiversityError("semantic profile requires bounded visible prompt and response text")
    prompt_lower = prompt.casefold()
    response_lower = response.casefold()
    expression = _EXPRESSION.search(prompt)
    expected = _expected_arithmetic_result(prompt)
    answer_match = _ANSWER.search(response)
    answer = int(answer_match.group(1)) if answer_match is not None else None
    if answer is None or expected is None:
        answer_relation = "missing_or_uncheckable"
    else:
        answer_relation = "matches_visible_task" if answer == expected else "conflicts_with_visible_task"
    unsupported_tool_claim = (
        any(term in response_lower for term in ("calculator", "python", "spreadsheet", "tool"))
        and "shown calculation" not in response_lower
        and "tool evidence" not in response_lower
    )
    prompt_skeleton = _semantic_skeleton(prompt_lower)
    response_skeleton = _semantic_skeleton(response_lower)
    return {
        "answer_relation": answer_relation,
        "arithmetic_signature": (
            {
                "left_operand_width_bucket": min(4, len(expression.group(1).lstrip("-"))),
                "left_sign": _integer_sign(int(expression.group(1))),
                "operator": expression.group(2),
                "right_operand_width_bucket": min(4, len(expression.group(3).lstrip("-"))),
                "right_sign": _integer_sign(int(expression.group(3))),
            }
            if expression is not None
            else None
        ),
        "constraint_term_count_bucket": min(4, sum(prompt_lower.count(term) for term in _CONSTRAINT_TERMS)),
        "numeric_claim_count_bucket": min(6, len(_NUMBER.findall(response))),
        "operation_class": _operation_class(prompt),
        "prompt_skeleton": prompt_skeleton,
        "reasoning_connector_count_bucket": min(5, sum(response_lower.count(term) for term in _REASONING_TERMS)),
        "response_sentence_count_bucket": min(6, sum(response.count(mark) for mark in ".!?") or 1),
        "response_skeleton": response_skeleton,
        "tool_claim_status": "unsupported" if unsupported_tool_claim else "absent_or_evidenced",
    }


def semantic_profile_fingerprint(prompt: str, response: str) -> str:
    return sha256_hex(canonical_json_bytes(structural_semantic_profile(prompt, response)))


def visible_semantic_content_fingerprint(prompt: str, response: str) -> str:
    """Identify exact visible content for no-reuse checks, separately from diversity structure."""

    structural_semantic_profile(prompt, response)
    return sha256_hex(
        canonical_json_bytes(
            {
                "domain": "visible-semantic-content/1.0.0",
                "prompt": prompt,
                "response": response,
            }
        )
    )


def semantic_diversity_summary(items: list[dict[str, object]]) -> dict[str, JsonValue]:
    """Measure distributional diversity; no score or judge contract depends on this summary."""

    if len(items) != 32:
        raise SemanticDiversityError("semantic diversity requires exactly 32 visible items")
    profiles: list[dict[str, JsonValue]] = []
    for item in items:
        if type(item.get("prompt")) is not str or type(item.get("response")) is not str:
            raise SemanticDiversityError("semantic diversity item text is invalid")
        profiles.append(structural_semantic_profile(cast(str, item["prompt"]), cast(str, item["response"])))
    encoded = [canonical_json_bytes(profile).decode("utf-8") for profile in profiles]
    counts = Counter(encoded)
    answer_relations = Counter(cast(str, profile["answer_relation"]) for profile in profiles)
    operation_classes = Counter(cast(str, profile["operation_class"]) for profile in profiles)
    maximum_profile_count = max(counts.values())
    # These are broad anti-collapse bounds, intentionally well below the simulator's normal diversity.
    passed = (
        len(counts) >= 12 and maximum_profile_count <= 8 and len(answer_relations) >= 3 and len(operation_classes) >= 3
    )
    return {
        "answer_relation_counts": dict(sorted(answer_relations.items())),
        "maximum_profile_count": maximum_profile_count,
        "operation_class_counts": dict(sorted(operation_classes.items())),
        "profile_count": len(counts),
        "profile_distribution_root_hash": sha256_hex(canonical_json_bytes(sorted(encoded))),
        "status": "passed" if passed else "failed",
        "thresholds": {
            "maximum_profile_count": 8,
            "minimum_answer_relation_classes": 3,
            "minimum_operation_classes": 3,
            "minimum_unique_profiles": 12,
        },
    }


def _expected_arithmetic_result(prompt: str) -> int | None:
    match = _EXPRESSION.search(prompt)
    if match is None:
        return None
    left, operator, right = int(match.group(1)), match.group(2), int(match.group(3))
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    return left * right


def _operation_class(prompt: str) -> str:
    match = _EXPRESSION.search(prompt)
    return {"+": "addition", "-": "subtraction", "*": "multiplication"}.get(
        match.group(2) if match is not None else "", "other"
    )


def _semantic_skeleton(text: str) -> str:
    without_numbers = _NUMBER.sub("<number>", text)
    return _SPACE.sub(" ", without_numbers).strip()


def _integer_sign(value: int) -> str:
    if value < 0:
        return "negative"
    if value > 0:
        return "positive"
    return "zero"
