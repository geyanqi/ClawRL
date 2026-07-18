"""Fresh-process immutable FinalEvaluationProtocol preregistration acceptance."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from clawrl.evaluation.protocol_models import EvaluationEnvironment, FinalEvaluationProtocolConfig
from clawrl.evaluation.protocol_workflow import (
    FinalEvaluationProtocolError,
    FinalEvaluationProtocolRegistry,
    ProtocolBindingConflictError,
)
from clawrl.judge.fit_models import InitialEvalRubric


def _config() -> FinalEvaluationProtocolConfig:
    return FinalEvaluationProtocolConfig(
        protocol_id="fixture-future-100-v1",
        rubric=InitialEvalRubric.fixture_default(),
        environment=EvaluationEnvironment(),
        sample_seed=90_001,
        balanced_ab_seed=90_002,
    )


def _files(root: Path) -> dict[str, tuple[int, int, str]]:
    result: dict[str, tuple[int, int, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            result[str(path.relative_to(root))] = (path.stat().st_mtime_ns, len(raw), hashlib.sha256(raw).hexdigest())
    return result


def _fresh_load(root: Path, campaign_id: str) -> dict[str, str]:
    code = """
import json,sys
from clawrl.evaluation.protocol_workflow import FinalEvaluationProtocolRegistry
s=FinalEvaluationProtocolRegistry.load(sys.argv[1],campaign_id=sys.argv[2])
print(json.dumps({'protocol':s.protocol.content_hash,'receipt':s.receipt.content_hash,'environment':s.environment.content_hash},sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root), campaign_id],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


def test_preregistration_is_process_stable_idempotent_and_required_for_freeze(tmp_path: Path) -> None:
    snapshot = FinalEvaluationProtocolRegistry.preregister(tmp_path, campaign_id="campaign-ticket09", config=_config())
    before = _files(tmp_path)
    loaded = _fresh_load(tmp_path, "campaign-ticket09")
    assert loaded == {
        "environment": snapshot.environment.content_hash,
        "protocol": snapshot.protocol.content_hash,
        "receipt": snapshot.receipt.content_hash,
    }
    replay = FinalEvaluationProtocolRegistry.preregister(tmp_path, campaign_id="campaign-ticket09", config=_config())
    assert replay.protocol.content_hash == snapshot.protocol.content_hash
    assert _files(tmp_path) == before

    with pytest.raises(FinalEvaluationProtocolError, match="RECEIPT_REQUIRED"):
        FinalEvaluationProtocolRegistry.authorize_candidate_freeze(
            tmp_path,
            campaign_id="campaign-ticket09",
            protocol_hash=snapshot.protocol.content_hash,
            preregistration_receipt_hash=None,
        )
    with pytest.raises(FinalEvaluationProtocolError, match="RECEIPT_MISMATCH"):
        FinalEvaluationProtocolRegistry.authorize_candidate_freeze(
            tmp_path,
            campaign_id="campaign-ticket09",
            protocol_hash="f" * 64,
            preregistration_receipt_hash=snapshot.receipt.content_hash,
        )
    assert _files(tmp_path) == before
    authorization = FinalEvaluationProtocolRegistry.authorize_candidate_freeze(
        tmp_path,
        campaign_id="campaign-ticket09",
        protocol_hash=snapshot.protocol.content_hash,
        preregistration_receipt_hash=snapshot.receipt.content_hash,
    )
    assert authorization.payload["status"] == "authorized"


def test_incompatible_protocol_invalidates_attempt_without_replacing_binding(tmp_path: Path) -> None:
    original = FinalEvaluationProtocolRegistry.preregister(tmp_path, campaign_id="campaign-bound", config=_config())
    ref = tmp_path / "protocol-bindings" / "campaign-bound" / "active.ref"
    original_ref = ref.read_bytes()
    with pytest.raises(ProtocolBindingConflictError, match="INCOMPATIBLE_PROTOCOL"):
        FinalEvaluationProtocolRegistry.preregister(
            tmp_path,
            campaign_id="campaign-bound",
            config=replace(_config(), sample_seed=90_003),
        )
    assert ref.read_bytes() == original_ref
    loaded = _fresh_load(tmp_path, "campaign-bound")
    assert loaded["protocol"] == original.protocol.content_hash
    schemas = [json.loads(path.read_text())["schema_name"] for path in (tmp_path / "artifacts").glob("*.json")]
    assert "FinalEvaluationProtocolConflict" in schemas
    assert "FinalEvaluationCampaignInvalidation" in schemas
    with pytest.raises(FinalEvaluationProtocolError, match="CAMPAIGN_INVALID"):
        FinalEvaluationProtocolRegistry.authorize_candidate_freeze(
            tmp_path,
            campaign_id="campaign-bound",
            protocol_hash=original.protocol.content_hash,
            preregistration_receipt_hash=original.receipt.content_hash,
        )


def test_protocol_hash_is_root_independent_and_semantics_change_hash(tmp_path: Path) -> None:
    left = FinalEvaluationProtocolRegistry.preregister(tmp_path / "left", campaign_id="same", config=_config())
    right = FinalEvaluationProtocolRegistry.preregister(tmp_path / "right", campaign_id="same", config=_config())
    changed = FinalEvaluationProtocolRegistry.preregister(
        tmp_path / "changed",
        campaign_id="same",
        config=replace(_config(), environment=EvaluationEnvironment(environment_id="fixture-final-eval-env-v2")),
    )
    assert left.protocol.content_hash == right.protocol.content_hash
    assert left.receipt.content_hash == right.receipt.content_hash
    assert changed.protocol.content_hash != left.protocol.content_hash
    assert (tmp_path / "left" / "artifacts" / f"{left.protocol.content_hash}.json").read_bytes() == (
        tmp_path / "right" / "artifacts" / f"{right.protocol.content_hash}.json"
    ).read_bytes()
