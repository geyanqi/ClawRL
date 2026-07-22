"""Small contract checks for the Ticket 26 boundary.

The full 100-trace fixture is exercised by the integration campaign; these
checks intentionally run without a model or scorer and protect the fail-closed
production seam and immutable configuration aliases.
"""

from pathlib import Path
from typing import cast

from clawrl.judge.recertification_122b import (
    Fixture122BRecertificationConfig,
    Recertification122BWorkflow,
)


def test_production_122b_readiness_is_blocked(tmp_path: Path) -> None:
    report = Recertification122BWorkflow.production_readiness(tmp_path)
    assert report.payload["phase"] == "TRAIN_122B"
    assert report.payload["status"] == "blocked"
    assert report.payload["side_effects_permitted"] is False
    checks = cast(list[dict[str, object]], report.payload["checks"])
    assert {item["code"] for item in checks} >= {
        "INFERENCE_MODEL_UNAVAILABLE",
        "REAL_SCORER_UNAVAILABLE",
    }


def test_compatibility_candidate_and_dataset_aliases_are_bound() -> None:
    config = Fixture122BRecertificationConfig(
        run_id="ticket26-alias",
        candidate_hash="a" * 64,
        dataset_hash="b" * 64,
        reward_schema_hash="c" * 64,
        scalarizer_hash="d" * 64,
        algorithm_contract_hash="e" * 64,
    )
    assert config.transfer_candidate_hash == "a" * 64
    assert config.dataset_version_hash == "b" * 64
    assert config.rollout_count == 32
