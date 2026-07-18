"""Ticket 21 typed ClusterAdapter contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from clawrl.artifacts import ArtifactStore, JsonValue
from clawrl.training.cluster_lifecycle import (
    CLUSTER_OPERATIONS,
    ClusterLifecycleWorkflow,
    FixtureClusterLifecycleConfig,
    ProductionClusterLifecycleConfig,
)


def _spec(root: Path) -> str:
    return (
        ArtifactStore(root)
        .put("ExperimentSpec", "1.0.0", {"experiment_id": "ticket21-spec", "status": "frozen"})
        .content_hash
    )


def test_all_cluster_operations_have_distinct_stable_action_keys(tmp_path: Path) -> None:
    spec_hash = _spec(tmp_path)
    config = FixtureClusterLifecycleConfig("cluster-run-21", "TRAIN_35B")
    left = ClusterLifecycleWorkflow.action_contracts(config, spec_hash)
    right = ClusterLifecycleWorkflow.action_contracts(config, spec_hash)
    assert tuple(item.operation for item in left) == CLUSTER_OPERATIONS[:-1]
    assert [item.idempotency_key for item in left] == [item.idempotency_key for item in right]
    assert len({item.idempotency_key for item in left}) == len(left)
    assert all(item.phase == "TRAIN_35B" and item.spec_hash == spec_hash for item in left)


def test_force_cancel_plan_is_typed_and_does_not_alias_graceful_stop(tmp_path: Path) -> None:
    spec_hash = _spec(tmp_path)
    graceful = ClusterLifecycleWorkflow.action_contracts(
        FixtureClusterLifecycleConfig("graceful-run-21", "TRAIN_122B"), spec_hash
    )
    canceled = ClusterLifecycleWorkflow.action_contracts(
        FixtureClusterLifecycleConfig("cancel-run-21", "TRAIN_122B", close_mode="force_cancel"), spec_hash
    )
    assert graceful[-1].operation == "graceful_stop"
    assert canceled[-1].operation == "force_cancel"
    assert graceful[-1].idempotency_key != canceled[-1].idempotency_key


def test_train_readiness_blocks_every_missing_external_cluster_input(tmp_path: Path) -> None:
    for phase in ("TRAIN_35B", "TRAIN_122B"):
        report = ClusterLifecycleWorkflow.production_readiness(
            tmp_path / phase,
            ProductionClusterLifecycleConfig(phase=phase),
        )
        checks = cast(list[dict[str, JsonValue]], report.payload["checks"])
        assert report.payload["status"] == "blocked"
        assert report.payload["side_effects_permitted"] is False
        assert report.payload["submit_attempted"] is False
        assert {item["code"] for item in checks} >= {
            "MISSING_JOBBUILDER",
            "MISSING_IMAGE",
            "MISSING_CFS_MOUNT",
            "MISSING_NAS_MOUNT",
            "MISSING_QUEUE",
            "MISSING_RESOURCE_SPEC",
            "MISSING_CLUSTER_CREDENTIAL",
        }
