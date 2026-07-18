"""Persistent fenced CFS reward roundtrip acceptance tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal, cast

import pytest

from clawrl.artifacts import ArtifactStore, CanonicalizationError, JsonValue
from clawrl.judge.certification_models import RewardSchema, Scalarizer
from clawrl.training.reward_roundtrip import (
    FenceAuthority,
    FixtureCfsBackend,
    FixtureCfsConfig,
    RewardIntegrityError,
    RewardRoundtripError,
    RewardRoundtripWorkflow,
    RewardSlotKey,
)


def _setup(root: Path, *, fault: str = "valid") -> tuple[RewardSlotKey, str, str, str]:
    store = ArtifactStore(root)
    reward = store.put("RewardSchema", "1.0.0", RewardSchema().artifact_payload())
    scalarizer = store.put("Scalarizer", "1.0.0", Scalarizer().artifact_payload())
    probe = FixtureCfsBackend(
        root,
        FixtureCfsConfig(
            "ticket10-backend",
            fault=cast(Literal["valid", "overwrite", "partial_visibility", "cleanup_failure"], fault),
        ),
    ).probe()
    return (
        RewardSlotKey("ticket10-run", 11, "uid-001", 0, "judge-pack-001"),
        probe.content_hash,
        reward.content_hash,
        scalarizer.content_hash,
    )


def _result(reward_hash: str, scalarizer_hash: str, *, scalar: int = 73_000_000) -> dict[str, JsonValue]:
    return {
        "confidence_basis_points": 8_750,
        "failure_tags": [],
        "reward_micros": scalar,
        "reward_schema_hash": reward_hash,
        "scalarizer_hash": scalarizer_hash,
        "turn_local_tie_groups": [["candidate-0"], ["candidate-1", "candidate-2"]],
    }


def _start(root: Path, key: RewardSlotKey, probe: str, reward: str, scalarizer: str, text: str = "answer-v1"):
    return RewardRoundtripWorkflow.start(
        root,
        key=key,
        probe_evidence_hash=probe,
        trajectory_payload={"prompt": "fixture prompt", "response": text, "sanitized": True},
        reward_schema_hash=reward,
        scalarizer_hash=scalarizer,
    )


def _fresh_resume(root: Path, key: RewardSlotKey) -> dict[str, object]:
    code = """
import json,sys
from clawrl.training.reward_roundtrip import RewardRoundtripWorkflow,RewardSlotKey
k=RewardSlotKey(sys.argv[2],int(sys.argv[3]),sys.argv[4],int(sys.argv[5]),sys.argv[6])
s=RewardRoundtripWorkflow.resume(sys.argv[1],key=k)
payload = {
    'attempts': list(s.attempt_ordinals),
    'request': s.reward_request.content_hash,
    'resolved': None if s.resolved_reward is None else s.resolved_reward.content_hash,
    'trajectory': s.trajectory_manifest.content_hash,
}
print(json.dumps(payload, sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(root),
            key.run_id,
            str(key.global_step),
            key.uid,
            str(key.rollout_index),
            key.judge_pack_id,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


def test_qualified_backend_out_of_order_attempts_fencing_and_restart(tmp_path: Path) -> None:
    key, probe, reward_hash, scalarizer_hash = _setup(tmp_path)
    initial = _start(tmp_path, key, probe, reward_hash, scalarizer_hash)
    replay = _start(tmp_path, key, probe, reward_hash, scalarizer_hash)
    assert replay.reward_request.content_hash == initial.reward_request.content_hash
    store = ArtifactStore(tmp_path)
    policy = store.read(
        cast(str, initial.reward_request.payload["attempt_policy_hash"]),
        expected_schema_name="RewardAttemptPolicy",
    )
    decision = store.read(
        cast(str, initial.reward_request.payload["decision_record_hash"]), expected_schema_name="DecisionRecord"
    )
    assert policy.payload["lease_ticks"] == 4 and policy.payload["max_attempts"] == 3
    assert decision.payload["attempt_policy_hash"] == policy.content_hash
    with pytest.raises(RewardIntegrityError, match="TRAJECTORY_STABLE_KEY_CONFLICT"):
        _start(tmp_path, key, probe, reward_hash, scalarizer_hash, "different-answer")
    alternate_reward = ArtifactStore(tmp_path).put(
        "RewardSchema", "1.0.0", RewardSchema(schema_id="alternate-calibrated-reward-v1").artifact_payload()
    )
    with pytest.raises(RewardIntegrityError, match="REWARD_REQUEST_STABLE_KEY_CONFLICT"):
        _start(tmp_path, key, probe, alternate_reward.content_hash, scalarizer_hash)

    first = RewardRoundtripWorkflow.claim_attempt(tmp_path, key=key, resolver_epoch=1, clock_tick=10)
    with pytest.raises(RewardRoundtripError, match="PREVIOUS_ATTEMPT"):
        RewardRoundtripWorkflow.claim_attempt(tmp_path, key=key, resolver_epoch=1, clock_tick=12)
    second = RewardRoundtripWorkflow.claim_attempt(tmp_path, key=key, resolver_epoch=1, clock_tick=15)
    assert first.payload["attempt_ordinal"] == 1 and second.payload["attempt_ordinal"] == 2
    result2 = RewardRoundtripWorkflow.commit_result(
        tmp_path, key=key, attempt_ordinal=2, result_payload=_result(reward_hash, scalarizer_hash, scalar=61_000_000)
    )
    result1 = RewardRoundtripWorkflow.commit_result(
        tmp_path, key=key, attempt_ordinal=1, result_payload=_result(reward_hash, scalarizer_hash)
    )
    assert (
        RewardRoundtripWorkflow.commit_result(
            tmp_path, key=key, attempt_ordinal=1, result_payload=_result(reward_hash, scalarizer_hash)
        ).content_hash
        == result1.content_hash
    )
    with pytest.raises(RewardIntegrityError, match="ATTEMPT_RESULT_HASH_CONFLICT"):
        RewardRoundtripWorkflow.commit_result(
            tmp_path,
            key=key,
            attempt_ordinal=1,
            result_payload=_result(reward_hash, scalarizer_hash, scalar=72_000_000),
        )

    authority = FenceAuthority(tmp_path)
    authority.advance(1)
    authority.advance(2)
    with pytest.raises(RewardRoundtripError, match="STALE_RESOLVER"):
        RewardRoundtripWorkflow.resolve(tmp_path, key=key, resolver_epoch=1)
    resolved = RewardRoundtripWorkflow.resolve(tmp_path, key=key, resolver_epoch=2)
    assert resolved.payload["attempt_ordinal"] == 1
    assert resolved.payload["result_hash"] == result1.content_hash
    assert resolved.payload["result_hash"] != result2.content_hash
    fresh = _fresh_resume(tmp_path, key)
    assert fresh["resolved"] == resolved.content_hash and fresh["attempts"] == [1, 2]
    with pytest.raises(RewardRoundtripError, match="LATE_RESULT_QUARANTINED"):
        RewardRoundtripWorkflow.commit_result(
            tmp_path, key=key, attempt_ordinal=2, result_payload=_result(reward_hash, scalarizer_hash)
        )


def test_invalid_results_never_resolve_or_fill_neutral(tmp_path: Path) -> None:
    variants: list[dict[str, JsonValue]] = []
    for index in range(4):
        root = tmp_path / str(index)
        key, probe, reward_hash, scalarizer_hash = _setup(root)
        _start(root, key, probe, reward_hash, scalarizer_hash)
        RewardRoundtripWorkflow.claim_attempt(root, key=key, resolver_epoch=1, clock_tick=1)
        payload = _result(reward_hash, scalarizer_hash)
        if index == 0:
            payload["confidence_basis_points"] = 10_001
        elif index == 1:
            payload["turn_local_tie_groups"] = [["duplicate", "duplicate"]]
        elif index == 2:
            payload["reward_schema_hash"] = "f" * 64
        else:
            del payload["failure_tags"]
        variants.append(payload)
        RewardRoundtripWorkflow.commit_result(root, key=key, attempt_ordinal=1, result_payload=payload)
        FenceAuthority(root).advance(1)
        with pytest.raises(RewardRoundtripError):
            RewardRoundtripWorkflow.resolve(root, key=key, resolver_epoch=1)
        assert RewardRoundtripWorkflow.resume(root, key=key).resolved_reward is None
        assert _fresh_resume(root, key)["resolved"] is None

    nonfinite_root = tmp_path / "nonfinite"
    key, probe, reward_hash, scalarizer_hash = _setup(nonfinite_root)
    _start(nonfinite_root, key, probe, reward_hash, scalarizer_hash)
    RewardRoundtripWorkflow.claim_attempt(nonfinite_root, key=key, resolver_epoch=1, clock_tick=1)
    nonfinite = _result(reward_hash, scalarizer_hash)
    nonfinite["reward_micros"] = cast(JsonValue, float("nan"))
    with pytest.raises(CanonicalizationError):
        RewardRoundtripWorkflow.commit_result(nonfinite_root, key=key, attempt_ordinal=1, result_payload=nonfinite)
    assert _fresh_resume(nonfinite_root, key)["resolved"] is None


def test_resolver_selects_minimum_valid_ordinal_after_invalid_terminal_result(tmp_path: Path) -> None:
    key, probe, reward_hash, scalarizer_hash = _setup(tmp_path)
    _start(tmp_path, key, probe, reward_hash, scalarizer_hash)
    RewardRoundtripWorkflow.claim_attempt(tmp_path, key=key, resolver_epoch=1, clock_tick=1)
    invalid = _result(reward_hash, scalarizer_hash)
    invalid["confidence_basis_points"] = 10_001
    RewardRoundtripWorkflow.commit_result(tmp_path, key=key, attempt_ordinal=1, result_payload=invalid)
    RewardRoundtripWorkflow.claim_attempt(tmp_path, key=key, resolver_epoch=1, clock_tick=2)
    valid = RewardRoundtripWorkflow.commit_result(
        tmp_path, key=key, attempt_ordinal=2, result_payload=_result(reward_hash, scalarizer_hash)
    )
    FenceAuthority(tmp_path).advance(1)
    resolved = RewardRoundtripWorkflow.resolve(tmp_path, key=key, resolver_epoch=1)
    assert resolved.payload["attempt_ordinal"] == 2
    assert resolved.payload["result_hash"] == valid.content_hash


def test_unqualified_backend_blocks_before_reward_request(tmp_path: Path) -> None:
    key, probe, reward_hash, scalarizer_hash = _setup(tmp_path, fault="overwrite")
    with pytest.raises(RewardRoundtripError, match="CFS_BACKEND_UNQUALIFIED"):
        _start(tmp_path, key, probe, reward_hash, scalarizer_hash)
    schemas = [json.loads(path.read_text())["schema_name"] for path in (tmp_path / "artifacts").glob("*.json")]
    assert "RewardRoundtripBlockedEvidence" in schemas
    assert "RewardRequest" not in schemas


def test_status_only_probe_cannot_bypass_capability_validation(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    reward = store.put("RewardSchema", "1.0.0", RewardSchema().artifact_payload())
    scalarizer = store.put("Scalarizer", "1.0.0", Scalarizer().artifact_payload())
    forged = store.put("CfsCapabilityProbeEvidence", "1.0.0", {"status": "qualified"})
    key = RewardSlotKey("ticket10-forged", 1, "uid-001", 0, "judge-pack-001")
    with pytest.raises(RewardIntegrityError, match="probe evidence fields"):
        _start(tmp_path, key, forged.content_hash, reward.content_hash, scalarizer.content_hash)
    schemas = [json.loads(path.read_text())["schema_name"] for path in (tmp_path / "artifacts").glob("*.json")]
    assert "RewardRequest" not in schemas
