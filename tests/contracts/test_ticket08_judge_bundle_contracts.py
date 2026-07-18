"""Closed-world Ticket 08 JudgeBundle compiler contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import ArtifactStore
from clawrl.judge.judge_bundle_models import (
    FixtureJudgeBundleConfig,
    JudgeBundleContractError,
    ProductionJudgeBundleConfig,
)
from clawrl.judge.judge_bundle_workflow import JudgeBundleWorkflow, UnsupportedRewardStrategyError

HASH = "a" * 64


def test_fixture_config_is_closed_world_and_content_addressed() -> None:
    config = FixtureJudgeBundleConfig(
        run_id="ticket08-contract",
        dataset_version_hash=HASH,
        reward_schema_hash="b" * 64,
        scalarizer_hash="c" * 64,
        algorithm_contract_hash="d" * 64,
    )
    assert FixtureJudgeBundleConfig.from_mapping(config.immutable_input_payload) == config
    widened = {**config.immutable_input_payload, "implicit_verifier": "spark"}
    with pytest.raises(JudgeBundleContractError):
        FixtureJudgeBundleConfig.from_mapping(widened)


@pytest.mark.parametrize("aggregation", ["hierarchical_rank", "local_microgroup_rank"])
def test_unsupported_strategy_fails_before_store_or_boundary_call(tmp_path: Path, aggregation: str) -> None:
    root = tmp_path / aggregation
    config = FixtureJudgeBundleConfig(
        run_id=f"unsupported-{aggregation}",
        dataset_version_hash=HASH,
        reward_schema_hash="b" * 64,
        scalarizer_hash="c" * 64,
        algorithm_contract_hash="d" * 64,
        aggregation=aggregation,
    )
    with pytest.raises(UnsupportedRewardStrategyError, match="UNSUPPORTED_REWARD_STRATEGY"):
        JudgeBundleWorkflow.bootstrap(root, config, epoch=8001)
    assert not root.exists()


def test_production_boundary_is_fail_closed_and_never_permits_side_effects(tmp_path: Path) -> None:
    report = JudgeBundleWorkflow.production_readiness(tmp_path, ProductionJudgeBundleConfig())
    persisted = ArtifactStore(tmp_path).read(report.content_hash, expected_schema_name="ReadinessReport")
    assert persisted.payload["phase"] == "JUDGE_CERTIFY"
    assert persisted.payload["status"] == "blocked"
    assert persisted.payload["side_effects_permitted"] is False
    assert len(cast(list[object], persisted.payload["checks"])) == 6
