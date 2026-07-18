"""Persistent Ticket 11 classic identity pipeline acceptance tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import ArtifactStore, JsonValue
from clawrl.training.classic_identity import (
    ClassicIdentityConfig,
    ClassicIdentityError,
    ClassicIdentityWorkflow,
    ClassicSourceRow,
)


def _sources(response_suffix: str = "") -> tuple[ClassicSourceRow, ...]:
    return (
        ClassicSourceRow("trace-001", "uid-001", "judge-pack-001", f"prompt one{response_suffix}"),
        ClassicSourceRow("trace-002", "uid-002", "judge-pack-002", f"prompt two{response_suffix}"),
    )


def _fresh_resume(root: Path, run_id: str) -> dict[str, object]:
    code = """
import json,sys
from clawrl.training.classic_identity import ClassicIdentityWorkflow
s=ClassicIdentityWorkflow.resume(sys.argv[1],sys.argv[2])
print(json.dumps({
    'dump': s.dump.content_hash,
    'report': s.report.content_hash,
    'row_count': len(s.dump.payload['rows']),
    'stage_hashes': s.report.payload['stage_hashes'],
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root), run_id],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    decoded = json.loads(completed.stdout)
    assert isinstance(decoded, dict)
    return decoded


def test_identity_survives_all_declared_classic_paths_and_fresh_resume(tmp_path: Path) -> None:
    config = ClassicIdentityConfig("classic-run-11", 9, rollout_count=3, chunk_size=4)
    snapshot = ClassicIdentityWorkflow.run(tmp_path, config=config, sources=_sources())
    assert snapshot.report.payload["status"] == "passed"
    assert snapshot.report.payload["declared_paths"] == [
        "wide_prefactor",
        "generation",
        "repeat",
        "union",
        "balance",
        "chunk_padding",
        "reward_loop_worker",
        "dump",
        "classic_resume",
    ]
    assert snapshot.report.payload["cfs_integration_claimed"] is False
    rows = snapshot.dump.payload["rows"]
    assert isinstance(rows, list) and len(rows) == 6
    row_payloads = cast(list[dict[str, JsonValue]], rows)
    identities = cast(list[dict[str, JsonValue]], [row["identity"] for row in row_payloads])
    assert len({json.dumps(item, sort_keys=True) for item in identities}) == 6
    for uid in ("uid-001", "uid-002"):
        indexes = [item["rollout_index"] for item in identities if item["uid"] == uid]
        assert all(type(index) is int for index in indexes)
        assert sorted(cast(list[int], indexes)) == [0, 1, 2]
    assert snapshot.dump.payload["global_steps"] == [9] * 6
    chunks = snapshot.dump.payload["transport_chunks"]
    assert isinstance(chunks, list)
    chunk_payloads = cast(list[list[JsonValue]], chunks)
    assert any(None in chunk for chunk in chunk_payloads)
    replay = ClassicIdentityWorkflow.run(tmp_path, config=config, sources=_sources())
    assert replay.report.content_hash == snapshot.report.content_hash
    fresh = _fresh_resume(tmp_path, config.run_id)
    assert fresh["report"] == snapshot.report.content_hash
    assert fresh["dump"] == snapshot.dump.content_hash
    assert fresh["row_count"] == 6


@pytest.mark.parametrize("fault_kind", ["missing_identity", "duplicate_index", "global_step_mismatch"])
def test_corruption_at_reward_worker_fails_closed_without_dump(tmp_path: Path, fault_kind: str) -> None:
    config = ClassicIdentityConfig(
        f"classic-run-{fault_kind}",
        9,
        rollout_count=2,
        chunk_size=3,
        fault_stage="reward_loop_worker",
        fault_kind=fault_kind,
    )
    with pytest.raises(ClassicIdentityError, match="CLASSIC_IDENTITY_CORRUPTION"):
        ClassicIdentityWorkflow.run(tmp_path, config=config, sources=_sources())
    schemas = [json.loads(path.read_text())["schema_name"] for path in (tmp_path / "artifacts").glob("*.json")]
    assert "ClassicIdentityFailure" in schemas
    assert "ClassicTrajectoryDump" not in schemas
    assert "ClassicTrajectoryContractReport" not in schemas


def test_same_run_with_changed_source_is_integrity_conflict(tmp_path: Path) -> None:
    config = ClassicIdentityConfig("classic-conflict-11", 2, rollout_count=2, chunk_size=4)
    ClassicIdentityWorkflow.run(tmp_path, config=config, sources=_sources())
    with pytest.raises(ClassicIdentityError, match="RUN_INPUT_CONFLICT"):
        ClassicIdentityWorkflow.run(tmp_path, config=config, sources=_sources(" changed"))
    recovered = ClassicIdentityWorkflow.resume(tmp_path, config.run_id)
    ArtifactStore(tmp_path).read(recovered.report.content_hash, expected_schema_name="ClassicTrajectoryContractReport")
