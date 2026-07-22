from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import ArtifactStore
from clawrl.governor.six_arm_cohort import SixArmCohortConfig, SixArmCohortError, SixArmCohortWorkflow


def test_each_spec_is_frozen_and_replication_only_changes_role_seed_and_run(tmp_path: Path) -> None:
    snapshot = SixArmCohortWorkflow.run(tmp_path, config=SixArmCohortConfig("ticket24-contract"))
    store = ArtifactStore(tmp_path)
    specs = [
        store.read(cast(str, item.payload["experiment_spec_hash"]), expected_schema_name="ExperimentSpec")
        for item in snapshot.arms
    ]
    required = {
        "aggregation",
        "algorithm",
        "dataset_version_hash",
        "generation",
        "judge_bundle_hash",
        "prompt",
        "resources",
        "reward_schema",
        "scalarizer",
        "status",
    }
    assert all(required <= set(item.payload) and item.payload["status"] == "frozen" for item in specs)
    cohort_spec = store.read(
        cast(str, snapshot.protocol.payload["cohort_spec_hash"]), expected_schema_name="CohortSpec"
    )
    assert cohort_spec.payload["replication_of"] == "control"
    assert cohort_spec.payload["exploration_arm_ids"] == [
        "exploration-1",
        "exploration-2",
        "exploration-3",
        "exploration-4",
    ]
    control, replication = specs[0].payload, specs[5].payload
    assert {key for key in control if control[key] != replication[key]} >= {"arm_role", "run_id", "seed"}
    assert control["dataset_version_hash"] == replication["dataset_version_hash"]
    assert control["judge_bundle_hash"] == replication["judge_bundle_hash"]
    assert control["algorithm"] == replication["algorithm"]
    assert control["generation"] == replication["generation"]
    assert {key for key in control if control[key] != replication[key]} == {"arm_role", "run_id", "seed"}


def test_protocol_tamper_and_changed_input_fail_closed(tmp_path: Path) -> None:
    config = SixArmCohortConfig("ticket24-integrity")
    first = SixArmCohortWorkflow.run(tmp_path, config=config)
    state_ref = tmp_path / "six-arm-cohorts" / config.cohort_id / "state.ref"
    state = ArtifactStore(tmp_path).read(
        state_ref.read_text(encoding="ascii").strip(), expected_schema_name="SixArmCohortState"
    )
    forged = ArtifactStore(tmp_path).put("SixArmCohortState", "1.0.0", {**state.payload, "protocol_hash": "0" * 64})
    state_ref.write_text(forged.content_hash + "\n", encoding="ascii")
    with pytest.raises(SixArmCohortError, match="protocol"):
        SixArmCohortWorkflow.resume(tmp_path, config.cohort_id)
    assert first.protocol.payload["roles"] == [
        "control",
        "exploration-1",
        "exploration-2",
        "exploration-3",
        "exploration-4",
        "control-replication",
    ]


def test_execution_profile_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SixArmCohortError, match="production six-arm cohort"):
        SixArmCohortWorkflow.run(
            tmp_path,
            config=SixArmCohortConfig("ticket24-production"),
            execution_profile="production",
        )
    with pytest.raises(SixArmCohortError, match="execution_profile"):
        SixArmCohortWorkflow.run(
            tmp_path,
            config=SixArmCohortConfig("ticket24-unknown-profile"),
            execution_profile="PRODUCTION",
        )
