"""Ticket 04 production-shaped Phase-A fit generation workflow."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from clawrl.artifacts import ArtifactStore, canonical_json_bytes
from clawrl.data.workflow import GovernedDataIngestWorkflow
from clawrl.judge.fit_models import (
    FixtureFitConfig,
    InitialEvalRubric,
    ProductionJudgeCertifyConfig,
    SolInferenceConfig,
)
from clawrl.judge.fit_workflow import FitTrajectoryWorkflow, FitWorkflowError
from clawrl.training.run_journal import StaleFencingEpoch
from tests.e2e.test_ticket03_successful_dataset_workflow import config as data_config
from tests.e2e.test_ticket03_successful_dataset_workflow import start as start_data

_DATASET_TEMPLATE: Path | None = None


def _dataset(root: Path) -> tuple[str, str]:
    global _DATASET_TEMPLATE
    if _DATASET_TEMPLATE is not None:
        shutil.copytree(_DATASET_TEMPLATE, root, dirs_exist_ok=True)
        # The root-independent DatasetVersion hash is carried by the template's
        # completed ingest event; recertify it rather than caching an object.
        data_snapshot = GovernedDataIngestWorkflow.resume(root, "ticket03-for-ticket04", epoch=3001)
        assert data_snapshot.dataset_version is not None
        trace_refs = cast(list[dict[str, Any]], data_snapshot.dataset_version.payload["trace_refs"])
        assert isinstance(trace_refs, list)
        return data_snapshot.dataset_version.content_hash, str(trace_refs[0]["trace_id"])
    snapshot = start_data(root, data_config("ticket03-for-ticket04"))
    while not snapshot.terminal:
        snapshot = GovernedDataIngestWorkflow.resume(root, "ticket03-for-ticket04", epoch=3001)
    assert snapshot.dataset_version is not None
    dataset_hash = snapshot.dataset_version.content_hash
    trace_refs = cast(list[dict[str, Any]], snapshot.dataset_version.payload["trace_refs"])
    assert isinstance(trace_refs, list)
    template = root.parent / "_ticket04_dataset_template"
    shutil.copytree(root, template, dirs_exist_ok=True)
    _DATASET_TEMPLATE = template
    return dataset_hash, str(trace_refs[0]["trace_id"])


def _config(dataset_hash: str, trace_id: str, **changes: object) -> FixtureFitConfig:
    values: dict[str, object] = {
        "run_id": "ticket04-phase-a",
        "dataset_version_hash": dataset_hash,
        "trace_id": trace_id,
        "generator_seed": 4004,
        "role_seed": 4304,
        "rubric": InitialEvalRubric.fixture_default(),
        "sol_inference": SolInferenceConfig.fixture_default(),
        "fault_schedule": ("success",),
    }
    values.update(changes)
    return FixtureFitConfig(**values)  # type: ignore[arg-type]


def _resume_subprocess(root: Path, run_id: str, epoch: int) -> dict[str, Any]:
    code = """
import json,sys
from clawrl.judge.fit_workflow import FitTrajectoryWorkflow
s=FitTrajectoryWorkflow.resume(sys.argv[1],sys.argv[2],epoch=int(sys.argv[3]))
print(json.dumps({
  'checkpoint':s.phase_a_checkpoint is not None,
  'events':[event.payload['event_type'] for event in s.events],
  'fit_set':None if s.fit_trajectory_set is None else s.fit_trajectory_set.content_hash,
  'packet':None if s.teacher_packet is None else s.teacher_packet.content_hash,
},sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root), run_id, str(epoch)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def _public_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    snapshot: dict[str, tuple[int, str]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            snapshot[str(path.relative_to(root))] = (len(raw), hashlib.sha256(raw).hexdigest())
    return snapshot


def test_fresh_process_phase_a_commits_exact_32_and_is_read_only_on_replay(tmp_path: Path) -> None:
    dataset_hash, trace_id = _dataset(tmp_path)
    config = _config(dataset_hash, trace_id)
    FitTrajectoryWorkflow.bootstrap(tmp_path, config, epoch=4004)
    state: dict[str, Any] = {"checkpoint": False}
    while not state["checkpoint"]:
        state = _resume_subprocess(tmp_path, config.run_id, 4004)

    store = ArtifactStore(tmp_path)
    fit_set = store.read(state["fit_set"], expected_schema_name="FitTrajectorySet")
    refs = cast(list[dict[str, Any]], fit_set.payload["trajectory_refs"])
    assert isinstance(refs, list) and len(refs) == 32
    assert len({ref["trajectory_id"] for ref in refs}) == 32
    manifests = [store.read(str(ref["manifest_hash"]), expected_schema_name="TrajectoryManifest") for ref in refs]
    assert all(manifest.payload["split"] == "fit" for manifest in manifests)
    assert len({manifest.payload["response_hash"] for manifest in manifests}) == 32
    assert all(manifest.payload["trace_id"] == trace_id for manifest in manifests)

    packet = store.read(state["packet"], expected_schema_name="TeacherScorerInputPacket")
    encoded = canonical_json_bytes(packet.payload).decode()
    assert "generator" not in encoded.lower()
    session = cast(dict[str, Any], packet.payload["scoring_session"])
    assert isinstance(session, dict)
    turns = cast(list[dict[str, Any]], session["turns"])
    assert isinstance(turns, list) and len(turns) == 8
    assert all(len(turn["items"]) == 4 for turn in turns)
    assert len({turn["thread_id"] for turn in turns}) == 5
    assert [turn["wave_index"] for turn in turns] == [1, 1, 1, 1, 1, 2, 2, 2]
    assert sum(len(turn["items"]) for turn in turns) == 32

    before = _public_snapshot(tmp_path)
    replay = FitTrajectoryWorkflow.run_once(tmp_path, config, epoch=9004)
    after = _public_snapshot(tmp_path)
    assert replay.phase_a_checkpoint is not None
    assert before == after


@pytest.mark.parametrize(
    "fault",
    ["short_response", "duplicate_response", "copied_online_response", "wrong_count", "out_of_order"],
)
def test_invalid_generator_batches_fail_closed_without_fit_set(tmp_path: Path, fault: str) -> None:
    dataset_hash, trace_id = _dataset(tmp_path)
    config = _config(
        dataset_hash,
        trace_id,
        run_id=f"ticket04-{fault}",
        include_online_response=(fault == "copied_online_response"),
        output_fault=fault,
    )
    snapshot = FitTrajectoryWorkflow.run_once(tmp_path, config, epoch=4004)
    assert snapshot.terminal
    assert snapshot.fit_trajectory_set is None
    assert snapshot.phase_a_checkpoint is None
    assert snapshot.events[-1].payload["event_type"] == "RUN_CLOSED"


def test_timeout_retry_and_fresh_restart_preserve_attempt_lineage(tmp_path: Path) -> None:
    dataset_hash, trace_id = _dataset(tmp_path)
    config = _config(
        dataset_hash,
        trace_id,
        run_id="ticket04-timeout-retry",
        fault_schedule=("timeout", "success"),
    )
    snapshot = FitTrajectoryWorkflow.bootstrap(tmp_path, config, epoch=4004)
    while snapshot.phase_a_checkpoint is None and not snapshot.terminal:
        snapshot = FitTrajectoryWorkflow.resume(tmp_path, config.run_id, epoch=4004)
    event_types = [event.payload["event_type"] for event in snapshot.events]
    assert event_types.count("GENERATION_RETRY_SCHEDULED") == 1
    assert event_types[-1] == "PHASE_A_CHECKPOINT"
    attempts = sorted((tmp_path / "boundaries" / "generator" / "attempts").rglob("*.ref"))
    assert [path.name for path in attempts] == ["00000000000000000001.ref", "00000000000000000002.ref"]


def test_run_identity_conflict_stale_fence_and_invalid_trace_fail_closed(tmp_path: Path) -> None:
    dataset_hash, trace_id = _dataset(tmp_path)
    config = _config(dataset_hash, trace_id, run_id="ticket04-identity")
    FitTrajectoryWorkflow.bootstrap(tmp_path, config, epoch=4004)
    with pytest.raises(FitWorkflowError):
        FitTrajectoryWorkflow.bootstrap(
            tmp_path,
            _config(dataset_hash, trace_id, run_id=config.run_id, generator_seed=9999),
            epoch=4005,
        )
    FitTrajectoryWorkflow.resume(tmp_path, config.run_id, epoch=5000)
    with pytest.raises(StaleFencingEpoch):
        FitTrajectoryWorkflow.resume(tmp_path, config.run_id, epoch=4004)
    with pytest.raises(FitWorkflowError):
        FitTrajectoryWorkflow.bootstrap(
            tmp_path / "missing-trace",
            _config(dataset_hash, "tt-" + "f" * 40, run_id="missing-trace"),
            epoch=4004,
        )


def test_online_response_is_counted_only_when_plan_explicitly_lists_it(tmp_path: Path) -> None:
    dataset_hash, trace_id = _dataset(tmp_path)
    no_online = FitTrajectoryWorkflow.run_once(
        tmp_path,
        _config(dataset_hash, trace_id, run_id="without-online"),
        epoch=4004,
    )
    with_online = FitTrajectoryWorkflow.run_once(
        tmp_path,
        _config(dataset_hash, trace_id, run_id="with-online", include_online_response=True),
        epoch=4004,
    )
    assert no_online.fit_trajectory_set is not None and with_online.fit_trajectory_set is not None
    store = ArtifactStore(tmp_path)
    first_plan = store.read(
        str(no_online.fit_trajectory_set.payload["generator_plan_hash"]), expected_schema_name="GeneratorPlan"
    )
    second_plan = store.read(
        str(with_online.fit_trajectory_set.payload["generator_plan_hash"]), expected_schema_name="GeneratorPlan"
    )
    first_slots = cast(list[dict[str, Any]], first_plan.payload["slots"])
    second_slots = cast(list[dict[str, Any]], second_plan.payload["slots"])
    assert all(slot["origin"] == "fresh_rollout" for slot in first_slots)
    assert sum(slot["origin"] == "online_response" for slot in second_slots) == 1


def test_production_judge_certify_blocks_before_boundary_creation(tmp_path: Path) -> None:
    report = FitTrajectoryWorkflow.production_readiness(tmp_path, ProductionJudgeCertifyConfig())
    assert report.payload["phase"] == "JUDGE_CERTIFY"
    assert report.payload["status"] == "blocked"
    checks = cast(list[dict[str, Any]], report.payload["checks"])
    codes = {check["code"] for check in checks}
    assert codes == {
        "GENERATOR_CONFIGURATION_UNAVAILABLE",
        "INITIAL_EVAL_RUBRIC_UNAVAILABLE",
        "PERMANENT_TRACE_GOVERNANCE_APPROVAL_UNAVAILABLE",
        "SOL_MODEL_CONFIGURATION_UNAVAILABLE",
    }
    assert not (tmp_path / "boundaries").exists()
