"""Persistent Ticket 13 ExpectedTrajectorySet acceptance tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import JsonValue
from clawrl.training.classic_identity import ClassicSourceRow
from clawrl.training.expected_trajectory_set import (
    ExpectedTrajectorySetConfig,
    ExpectedTrajectorySetError,
    ExpectedTrajectorySetWorkflow,
)


def _sources() -> tuple[ClassicSourceRow, ...]:
    return (
        ClassicSourceRow("trace-001", "uid-001", "judge-pack-001", "prompt one"),
        ClassicSourceRow("trace-002", "uid-002", "judge-pack-002", "prompt two"),
    )


def _fresh(root: Path, run_id: str, step: int) -> dict[str, object]:
    code = """
import json,sys
from clawrl.training.expected_trajectory_set import ExpectedTrajectorySetWorkflow
s=ExpectedTrajectorySetWorkflow.resume(sys.argv[1],sys.argv[2],int(sys.argv[3]))
payload={
    'authorization':s.authorization.content_hash,
    'expected_set':s.expected_set.content_hash,
    'manifest_hashes':[x.content_hash for x in s.manifests],
}
print(json.dumps(payload,sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root), run_id, str(step)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize("transport", ["classic", "v1"])
def test_out_of_order_complete_set_uses_shared_validator_before_scorer_and_fresh_resume(
    tmp_path: Path, transport: str
) -> None:
    config = ExpectedTrajectorySetConfig(f"expected-{transport}-13", 12, rollout_count=3)
    snapshot = ExpectedTrajectorySetWorkflow.run(
        tmp_path,
        config=config,
        sources=_sources(),
        arrival_ordinals=(5, 1, 4, 0, 3, 2),
        transport=transport,
    )
    assert snapshot.authorization.payload["status"] == "scoring_authorized"
    assert snapshot.authorization.payload["validator_id"] == "shared-expected-trajectory-set/1.0.0"
    assert snapshot.authorization.payload["transport"] == transport
    slot_keys = cast(list[dict[str, JsonValue]], [item.payload["slot_key"] for item in snapshot.manifests])
    assert [item["rollout_index"] for item in slot_keys] == [0, 1, 2, 0, 1, 2]
    fresh = _fresh(tmp_path, config.run_id, config.global_step)
    assert fresh["authorization"] == snapshot.authorization.content_hash
    assert fresh["expected_set"] == snapshot.expected_set.content_hash
    assert fresh["manifest_hashes"] == [item.content_hash for item in snapshot.manifests]


def test_missing_or_duplicate_batch_blocks_before_any_scorer_authorization(tmp_path: Path) -> None:
    config = ExpectedTrajectorySetConfig("expected-missing-13", 13, rollout_count=2)
    frozen = ExpectedTrajectorySetWorkflow.freeze(tmp_path, config=config, sources=_sources())
    rows = ExpectedTrajectorySetWorkflow.generated_rows(config, _sources())
    for row in rows[:-1]:
        ExpectedTrajectorySetWorkflow.publish_slot(tmp_path, expected_set=frozen, row=row)
    with pytest.raises(ExpectedTrajectorySetError, match="EXPECTED_TRAJECTORY_SET_INCOMPLETE"):
        ExpectedTrajectorySetWorkflow.authorize_scoring(tmp_path, expected_set=frozen, transport="classic")
    schemas = [json.loads(path.read_text())["schema_name"] for path in (tmp_path / "artifacts").glob("*.json")]
    assert "ScorerRequestAuthorization" not in schemas
    with pytest.raises(ExpectedTrajectorySetError, match="duplicate"):
        ExpectedTrajectorySetWorkflow.publish_batch(
            tmp_path,
            expected_set=frozen,
            rows=(rows[0], rows[0]),
        )


def test_same_step_changed_mapping_is_immutable_conflict(tmp_path: Path) -> None:
    config = ExpectedTrajectorySetConfig("expected-mapping-13", 14, rollout_count=2)
    ExpectedTrajectorySetWorkflow.freeze(tmp_path, config=config, sources=_sources())
    changed = (
        ClassicSourceRow("trace-changed", "uid-001", "judge-pack-001", "prompt one"),
        _sources()[1],
    )
    with pytest.raises(ExpectedTrajectorySetError, match="EXPECTED_SET_CONFLICT"):
        ExpectedTrajectorySetWorkflow.freeze(tmp_path, config=config, sources=changed)
