from pathlib import Path
from typing import Literal, cast

import pytest

from clawrl.artifacts import ArtifactStore
from clawrl.governor.bounded_iteration import BoundedGovernorConfig, BoundedGovernorError, BoundedGovernorWorkflow
from clawrl.governor.six_arm_cohort import SixArmCohortConfig, SixArmCohortWorkflow
from clawrl.router.trace_router import RouterCapacityConfig


def _root(tmp_path: Path) -> Path:
    SixArmCohortWorkflow.run(tmp_path, config=SixArmCohortConfig("ticket25-cohort"))
    return tmp_path


def test_decision_proposal_and_child_contracts_are_bounded(tmp_path: Path) -> None:
    snapshot = BoundedGovernorWorkflow.run(
        _root(tmp_path),
        config=BoundedGovernorConfig(
            "ticket25-governor", "ticket25-cohort", max_children=2, router_capacity=RouterCapacityConfig(2)
        ),
    )
    assert snapshot.plan.payload["bounded"] is True
    assert snapshot.plan.payload["recursive_spawn"] is False
    assert {"hypothesis", "evidence", "config_diff", "worst_case_budget", "expected_result"} <= set(
        snapshot.proposal.payload
    )
    assert len(snapshot.children) == 2
    assert len({item.payload["idempotency_key"] for item in snapshot.children}) == 2
    assert all(
        item.payload["harness_steps"] == ["submit", "status", "logs", "artifact", "checkpoint"]
        for item in snapshot.children
    )


@pytest.mark.parametrize("branch", ["new_dataset", "new_experiment", "stop_and_transfer"])
def test_fixture_branches_publish_immutable_result(tmp_path: Path, branch: str) -> None:
    root = _root(tmp_path)
    snapshot = BoundedGovernorWorkflow.run(
        root,
        config=BoundedGovernorConfig(
            f"ticket25-{branch}",
            "ticket25-cohort",
            branch=cast(Literal["new_dataset", "new_experiment", "stop_and_transfer"], branch),
        ),
    )
    assert snapshot.summary is not None
    if branch == "stop_and_transfer":
        assert snapshot.transfer_candidate is not None
        assert snapshot.transfer_candidate.payload["terminal_action"] == "stop_and_transfer"
    elif branch == "new_dataset":
        dataset = ArtifactStore(root).read(
            cast(str, snapshot.summary.payload["dataset_version_hash"]), expected_schema_name="DatasetVersion"
        )
        assert dataset.payload["purpose"] == "training_allowed"
    else:
        assert isinstance(snapshot.summary.payload["recertification_hash"], str)


def test_capacity_reserved_aggregation_future_and_production_fail_closed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    deferred = BoundedGovernorWorkflow.run(
        root,
        config=BoundedGovernorConfig(
            "ticket25-capacity", "ticket25-cohort", max_children=2, router_capacity=RouterCapacityConfig(1)
        ),
    )
    assert deferred.state.payload["status"] == "deferred"
    with pytest.raises(BoundedGovernorError, match="RESERVED_AGGREGATION"):
        BoundedGovernorConfig("ticket25-aggregation", "ticket25-cohort", reserved_aggregation=True)
    with pytest.raises(BoundedGovernorError, match="Future EvaluationDataset"):
        BoundedGovernorConfig("ticket25-future", "ticket25-cohort", future_evaluation_dataset_hash="a" * 64)
    readiness = BoundedGovernorWorkflow.production_readiness(root)
    assert readiness.payload["status"] == "blocked"
    assert readiness.payload["side_effects_permitted"] is False
