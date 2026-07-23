"""Focused Ticket 28 CandidateFreeze boundary tests."""

from dataclasses import replace
from pathlib import Path

import pytest

from clawrl.artifacts import ArtifactStore
from clawrl.evaluation import (
    CandidateFreezeConfig,
    CandidateFreezeReadinessError,
    CandidateFreezeWorkflow,
    EvaluationEnvironment,
    FinalEvaluationProtocolConfig,
    FinalEvaluationProtocolRegistry,
    FixtureControllerClock,
)
from clawrl.judge.fit_models import InitialEvalRubric


def _inputs(root: Path, *, env_hash: str | None = None):
    protocol = FinalEvaluationProtocolRegistry.preregister(
        root,
        campaign_id="ticket28",
        config=FinalEvaluationProtocolConfig(
            "future-100-v1", InitialEvalRubric.fixture_default(), EvaluationEnvironment(), 1, 2
        ),
    )
    store = ArtifactStore(root)
    dataset = store.put("DatasetVersion", "1.0.0", {"purpose": "training_allowed"})
    report_hashes = [
        store.put(
            "CertificationReport",
            "1.0.0",
            {"status": "certified", "trace_id": f"trace-{i:03d}", "variant": "success"},
        ).content_hash
        for i in range(100)
    ]
    coverage = store.put(
        "RecertificationCoverageManifest",
        "1.0.0",
        {
            "dataset_version_hash": dataset.content_hash,
            "report_hashes": report_hashes,
            "status": "complete",
            "trace_count": 100,
        },
    )
    bundle = store.put(
        "JudgeBundle",
        "1.0.0",
        {
            "dataset_version_hash": dataset.content_hash,
            "coverage_manifest_hash": coverage.content_hash,
            "fresh_rollout_count_per_trace": 32,
            "report_hashes": report_hashes,
            "status": "terminal",
            "certification_phase": "TRAIN_122B",
            "trace_count": 100,
            "version": "122b-recertified/1.0.0",
        },
    )
    approval = store.put(
        "ExperimentSpecApproval",
        "1.0.0",
        {
            "approval_hash": "a" * 64,
            "approved_by": "platform",
            "authoring_mode": "independent",
            "experiment_id": "ticket28-spec",
            "model_size": "122B",
            "status": "approved",
        },
    )
    spec = store.put(
        "ExperimentSpec",
        "1.0.0",
        {
            "dataset_version_hash": dataset.content_hash,
            "judge_bundle_hash": bundle.content_hash,
            "model_size": "122B",
            "authoring_mode": "independent",
            "approval_evidence_hash": approval.content_hash,
            "approval_hash": "a" * 64,
            "experiment_id": "ticket28-spec",
            "learning_rate_micros": 100,
            "monitoring": {"heartbeat_ticks": 1},
            "optimizer": "adamw",
            "parallelism": {"tensor": 1},
            "resource": {"gpu": "fixture"},
            "retry": {"max_attempts": 1},
            "status": "approved",
        },
    )
    environment = env_hash or protocol.environment.content_hash
    environment_payload = protocol.environment.payload
    semantic = {
        key: environment_payload[key]
        for key in (
            "semantic_decoding",
            "tool_harness_policy",
            "prompt_wrapper",
            "evaluator_visible_trajectory_schema",
        )
    }
    base = store.put(
        "Checkpoint",
        "1.0.0",
        {"evaluation_environment_hash": environment, "model_identity": "base", **semantic},
    )
    trained = store.put(
        "Checkpoint",
        "1.0.0",
        {"evaluation_environment_hash": environment, "model_identity": "trained", **semantic},
    )
    step_applied = store.put(
        "StepApplied",
        "1.0.0",
        {"checkpoint_hash": trained.content_hash, "status": "applied"},
    )
    step_applied = store.put(
        "StepApplied",
        "1.0.0",
        {"checkpoint_hash": trained.content_hash, "status": "applied"},
    )
    run = store.put(
        "RunRecord",
        "1.0.0",
        {
            "checkpoint_hash": trained.content_hash,
            "dataset_version_hash": dataset.content_hash,
            "experiment_spec_hash": spec.content_hash,
            "judge_bundle_hash": bundle.content_hash,
            "optimizer_update_count": 1,
            "step_applied_hash": step_applied.content_hash,
            "phase": "TRAIN_122B",
            "status": "succeeded",
        },
    )
    return protocol, CandidateFreezeConfig(
        "ticket28",
        protocol.protocol.content_hash,
        protocol.receipt.content_hash,
        run.content_hash,
        base.content_hash,
        dataset.content_hash,
        bundle.content_hash,
        spec.content_hash,
        trained_checkpoint_hash=trained.content_hash,
    )


def test_incomplete_122b_bundle_is_rejected_before_freeze(tmp_path: Path) -> None:
    _, config = _inputs(tmp_path)
    with pytest.raises(CandidateFreezeReadinessError, match="DEEP_RECERTIFICATION"):
        CandidateFreezeWorkflow.run(tmp_path, config=config, clock=FixtureControllerClock("2026-03-01T00:00:00Z"))


def test_environment_mismatch_and_incomplete_run_fail_closed(tmp_path: Path) -> None:
    _, config = _inputs(tmp_path, env_hash="f" * 64)
    with pytest.raises(CandidateFreezeReadinessError):
        CandidateFreezeWorkflow.run(tmp_path, config=config)


def test_production_is_blocked_before_freeze(tmp_path: Path) -> None:
    _, config = _inputs(tmp_path)
    blocked = CandidateFreezeWorkflow.run(tmp_path, config=replace(config, execution_profile="production"))
    assert blocked.freeze is None
    assert blocked.readiness.payload["status"] == "blocked"
