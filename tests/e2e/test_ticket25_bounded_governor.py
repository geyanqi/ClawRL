from pathlib import Path

import pytest

from clawrl.governor.bounded_iteration import (
    BoundedGovernorConfig,
    BoundedGovernorWorkflow,
    InjectedBoundedGovernorCrash,
)
from clawrl.governor.six_arm_cohort import SixArmCohortConfig, SixArmCohortWorkflow
from clawrl.router.trace_router import RouterCapacityConfig


def test_crash_restart_does_not_repeat_committed_child(tmp_path: Path) -> None:
    SixArmCohortWorkflow.run(tmp_path, config=SixArmCohortConfig("ticket25-restart-cohort"))
    config = BoundedGovernorConfig(
        "ticket25-restart", "ticket25-restart-cohort", max_children=2, router_capacity=RouterCapacityConfig(2)
    )
    with pytest.raises(InjectedBoundedGovernorCrash):
        BoundedGovernorWorkflow.run(tmp_path, config=config, crash_after_child=0)
    resumed = BoundedGovernorWorkflow.resume(tmp_path, config.governor_id)
    again = BoundedGovernorWorkflow.resume(tmp_path, config.governor_id)
    assert resumed.state.content_hash == again.state.content_hash
    assert len(resumed.children) == 2
    assert len({item.payload["action_id"] for item in resumed.children}) == 2
