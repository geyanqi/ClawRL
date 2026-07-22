"""Focused contract checks for the gated 122B short-run boundary."""

from pathlib import Path

import pytest

from clawrl.artifacts import ArtifactStore
from clawrl.training.gated_122b_run import (
    Gated122BReadinessError,
    Gated122BRunConfig,
    Gated122BRunWorkflow,
    Independent122BExperimentSpecConfig,
)


def test_production_122b_readiness_is_blocked_before_submit(tmp_path: Path) -> None:
    report = Gated122BRunWorkflow.production_readiness(tmp_path)
    assert report.payload["phase"] == "TRAIN_122B"
    assert report.payload["status"] == "blocked"
    assert report.payload["side_effects_permitted"] is False
    assert report.payload["submit_attempted"] is False


def test_independent_spec_is_explicit_and_content_addressed(tmp_path: Path) -> None:
    config = Independent122BExperimentSpecConfig(
        experiment_id="ticket27-spec",
        dataset_version_hash="a" * 64,
        judge_bundle_hash="b" * 64,
        parallelism={"tensor": 8, "pipeline": 4},
        resource={"gpu": "122b-fixture", "count": 8},
        retry={"max_attempts": 2, "backoff_ticks": 1},
        monitoring={"heartbeat_ticks": 4, "max_kl_millis": 250},
        approval_hash="c" * 64,
        approved_by="platform-approver",
    )
    spec = Gated122BRunWorkflow.build_experiment_spec(tmp_path, config)
    assert spec.payload["authoring_mode"] == "independent"
    assert spec.payload["model_size"] == "122B"
    assert set(spec.payload) >= {
        "optimizer",
        "learning_rate_micros",
        "parallelism",
        "resource",
        "retry",
        "monitoring",
    }


def test_old_or_partial_inputs_fail_before_optimizer_artifacts(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    old_bundle = store.put(
        "JudgeBundle",
        "1.0.0",
        {"certification_phase": "TRAIN_35B", "status": "total", "trace_count": 1},
    )
    spec = store.put(
        "ExperimentSpec",
        "1.0.0",
        {
            "approval_hash": "c" * 64,
            "approved_by": "platform-approver",
            "authoring_mode": "independent",
            "dataset_version_hash": "a" * 64,
            "experiment_id": "ticket27-old",
            "judge_bundle_hash": old_bundle.content_hash,
            "learning_rate_micros": 100,
            "model_size": "122B",
            "monitoring": {"heartbeat_ticks": 4},
            "optimizer": "adamw",
            "parallelism": {"tensor": 8},
            "resource": {"gpu": "fixture"},
            "retry": {"max_attempts": 2},
            "status": "approved",
        },
    )
    config = Gated122BRunConfig("ticket27-reject", "a" * 64, old_bundle.content_hash, spec.content_hash)
    with pytest.raises(Gated122BReadinessError):
        Gated122BRunWorkflow.run(tmp_path, config=config)
    assert not (tmp_path / "durable-step-applied" / config.run_id).exists()
