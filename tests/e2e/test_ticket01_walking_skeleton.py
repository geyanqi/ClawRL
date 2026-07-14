from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from typing import Any

import pytest

import clawrl.artifacts as artifacts_module
from clawrl.adapters.cluster.fixture import PersistentFixtureCluster
from clawrl.adapters.scorers.fixture import PersistentFixtureScorer
from clawrl.artifacts import (
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    canonical_json_bytes,
)
from clawrl.profiles import (
    FixtureProfileConfig,
    ProductionProfileConfig,
    ProfileConfigurationError,
)
from clawrl.training.run_journal import RunJournal, RunNotStarted
from clawrl.training.walking_skeleton import (
    InjectedProcessCrash,
    ProfiledTraceWorkflow,
    TraceWorkflowInput,
    WorkflowIntegrityError,
)

STUDENT_PACKET: dict[str, Any] = {
    "schema_version": "student-judge-input/1.0.0",
    "role": "student_judge",
    "seed": 1701,
    "packet_id": "sjp-ticket01-0001",
    "run_id": "fixture-run-ticket01",
    "global_step": 0,
    "trace_id": "training-trace-0001",
    "trajectory": {
        "trajectory_id": "trajectory-0001",
        "prompt": "Give a two-step recovery procedure after a durable worker restart.",
        "response": (
            "First reload and verify the committed checkpoint and its content hash. "
            "Then resume only unfinished work using the persisted idempotency key; if the "
            "checkpoint or hash is inconsistent, stop fail-closed."
        ),
    },
    "judge_pack": {
        "judge_pack_id": "judge-pack-precertified-0001",
        "items_per_turn": 8,
        "reward_schema_version": "reward-schema/1.0.0",
        "scalarizer_version": "scalarizer/1.0.0",
        "dimensions": {
            "instruction_following": {"min": 0, "max": 4},
            "recovery_safety": {"min": 0, "max": 4},
            "evidence_specificity": {"min": 0, "max": 2},
        },
        "scalarizer": {
            "formula": "instruction_following + recovery_safety + evidence_specificity",
            "min": 0,
            "max": 10,
        },
    },
    "rubric": {
        "instruction_following": "0-4: both requested ordered steps are explicit and actionable",
        "recovery_safety": (
            "0-4: checkpoint verification, unfinished-only resume, and fail-closed inconsistency handling"
        ),
        "evidence_specificity": ("0-2: references durable evidence and idempotency identity"),
    },
    "output_schema": {
        "schema_version": "student-judge-output/1.0.0",
        "required": [
            "packet_id",
            "seed",
            "dimensions",
            "overall_scalar",
            "confidence",
            "failure_tags",
            "evidence",
            "turn_local_tie_groups",
        ],
    },
    "allowlisted_fields": [
        "schema_version",
        "role",
        "seed",
        "packet_id",
        "run_id",
        "global_step",
        "trace_id",
        "trajectory",
        "judge_pack",
        "rubric",
        "output_schema",
        "allowlisted_fields",
    ],
}

RAW_STUDENT_OUTPUT = (
    '{"packet_id":"sjp-ticket01-0001","seed":1701,"dimensions":'
    '{"instruction_following":4,"recovery_safety":4,"evidence_specificity":2},'
    '"overall_scalar":10,"confidence":1.0,"failure_tags":[],"evidence":['
    '"Two ordered, actionable steps are explicit: verify the committed checkpoint, then '
    'resume unfinished work.","The response requires checkpoint and content-hash verification, '
    'unfinished-only resumption, and fail-closed handling of inconsistencies.","It cites durable '
    "evidence and identity through the committed checkpoint, content hash, and persisted "
    'idempotency key."],"turn_local_tie_groups":[]}'
)


def trace_input(*, run_id: str = "fixture-run-ticket01") -> TraceWorkflowInput:
    packet = json.loads(json.dumps(STUDENT_PACKET))
    packet["run_id"] = run_id
    return TraceWorkflowInput(
        run_id=run_id,
        dataset_version_id="dataset-precertified-ticket01",
        experiment_spec_id="experiment-precertified-ticket01",
        trace_id=packet["trace_id"],
        trajectory_id=packet["trajectory"]["trajectory_id"],
        prompt=packet["trajectory"]["prompt"],
        response=packet["trajectory"]["response"],
        judge_pack=packet["judge_pack"],
        student_input_packet=packet,
        raw_student_output=RAW_STUDENT_OUTPUT.encode(),
        role_session_lineage="/root/ticket01_student_judge",
    )


def event_types(root: Path, run_id: str) -> list[str]:
    journal = RunJournal(root, ArtifactStore(root), run_id)
    return [str(event.payload["event_type"]) for event in journal.events()]


def resume_in_fresh_process(root: Path, run_id: str, *, epoch: int = 1701) -> dict[str, object]:
    script = """
import json
import sys
from clawrl.training.walking_skeleton import ProfiledTraceWorkflow

snapshot = ProfiledTraceWorkflow.resume(sys.argv[1], sys.argv[2], epoch=int(sys.argv[3]))
print(json.dumps({
    "closed": snapshot.closed is not None,
    "event_type": snapshot.events[-1].payload["event_type"],
    "reason_code": snapshot.closed.payload["reason_code"] if snapshot.closed is not None else None,
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-c", script, str(root), run_id, str(epoch)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    decoded = json.loads(result.stdout)
    assert isinstance(decoded, dict)
    return decoded


def resume_attempt_in_fresh_process(
    root: Path,
    run_id: str,
    *,
    epoch: int,
) -> dict[str, object]:
    script = """
import json
import sys
from clawrl.training.walking_skeleton import ProfiledTraceWorkflow

try:
    snapshot = ProfiledTraceWorkflow.resume(sys.argv[1], sys.argv[2], epoch=int(sys.argv[3]))
except Exception as error:
    print(json.dumps({"exception": type(error).__name__}))
else:
    print(json.dumps({
        "event_type": snapshot.events[-1].payload["event_type"],
        "exception": None,
    }))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-c", script, str(root), run_id, str(epoch)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    decoded = json.loads(result.stdout)
    assert isinstance(decoded, dict)
    return decoded


def inject_adapter_exception_in_fresh_process(
    root: Path,
    run_id: str,
    *,
    boundary: str,
    stage: str,
    epoch: int = 1701,
) -> dict[str, object]:
    secret = "runtime-adapter-secret-DO-NOT-PERSIST"
    script = """
import json
import sys
from clawrl.adapters.cluster.fixture import PersistentFixtureCluster
from clawrl.adapters.scorers.fixture import PersistentFixtureScorer
from clawrl.profiles import FixtureProfileConfig
from clawrl.training.walking_skeleton import ProfiledTraceWorkflow

root, run_id, boundary, stage, epoch_text, secret = sys.argv[1:]

class ExplodingScorer(PersistentFixtureScorer):
    def query(self, *args, **kwargs):
        if boundary == "scorer" and stage == "query":
            raise RuntimeError(secret)
        return super().query(*args, **kwargs)

    def execute(self, *args, **kwargs):
        if boundary == "scorer" and stage == "execute":
            raise RuntimeError(secret)
        return super().execute(*args, **kwargs)

class ExplodingCluster(PersistentFixtureCluster):
    def query(self, *args, **kwargs):
        if boundary == "cluster" and stage == "query":
            raise RuntimeError(secret)
        return super().query(*args, **kwargs)

    def execute(self, *args, **kwargs):
        if boundary == "cluster" and stage == "execute":
            raise RuntimeError(secret)
        return super().execute(*args, **kwargs)

class ExplodingFactory:
    def build_fixture(self, root, store, config):
        if stage == "factory":
            raise RuntimeError(secret)
        return (
            ExplodingScorer(
                root,
                store,
                config.scorer_failure_code,
                config.effective_scorer_fault_schedule(),
            ),
            ExplodingCluster(
                root,
                store,
                config.cluster_failure_code,
                config.effective_cluster_fault_schedule(),
            ),
        )

loader = ProfiledTraceWorkflow(root, FixtureProfileConfig())
item, config = loader._load_persisted_input(run_id)
snapshot = ProfiledTraceWorkflow(
    root,
    config,
    adapter_factory=ExplodingFactory(),
).advance(item, epoch=int(epoch_text))
last = snapshot.events[-1]
details = last.payload["details"]
print(json.dumps({
    "event_type": last.payload["event_type"],
    "failure_code": details["failure_code"],
    "observation_hash": details["observation_hash"],
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(root),
            run_id,
            boundary,
            stage,
            str(epoch),
            secret,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert secret not in result.stdout
    assert secret not in result.stderr
    decoded = json.loads(result.stdout)
    assert isinstance(decoded, dict)
    return decoded


def run_with_fresh_session_after_every_event(
    root: Path,
    item: TraceWorkflowInput,
    config: FixtureProfileConfig,
) -> None:
    snapshot = ProfiledTraceWorkflow(root, config).advance(item, epoch=1701)
    previous_count = 1
    assert len(snapshot.events) == previous_count
    for _ in range(15):
        snapshot = ProfiledTraceWorkflow.resume(root, item.run_id, epoch=1701)
        fresh_journal = RunJournal(root, ArtifactStore(root), item.run_id)
        fresh_journal.verify()
        current_count = len(fresh_journal.events())
        assert current_count == previous_count + 1
        previous_count = current_count
        if snapshot.closed is not None:
            return
    pytest.fail("workflow did not reach an immutable terminal event")


def test_subprocess_resumes_from_only_disk_root_run_id_and_epoch_after_every_event(
    tmp_path: Path,
) -> None:
    item = trace_input(run_id="fixture-run-subprocess-recovery")
    ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(item, epoch=1701)
    script = """
import json
import sys
from clawrl.training.walking_skeleton import ProfiledTraceWorkflow

snapshot = ProfiledTraceWorkflow.resume(sys.argv[1], sys.argv[2], epoch=int(sys.argv[3]))
print(json.dumps({"event_count": len(snapshot.events), "closed": snapshot.closed is not None}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    previous_count = 1
    for _ in range(15):
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), item.run_id, "1701"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        state = json.loads(result.stdout)
        assert state["event_count"] == previous_count + 1
        previous_count = state["event_count"]
        RunJournal(tmp_path, ArtifactStore(tmp_path), item.run_id).verify()
        if state["closed"]:
            break
    else:
        pytest.fail("disk-only subprocess recovery did not close the run")
    assert PersistentFixtureScorer(tmp_path, ArtifactStore(tmp_path), None).execution_count() == 1
    assert PersistentFixtureCluster(tmp_path, ArtifactStore(tmp_path), None).execution_count() == 1


def test_fixture_trace_restarts_after_every_event_and_persists_role_lineage(tmp_path: Path) -> None:
    item = trace_input()
    config = FixtureProfileConfig()

    run_with_fresh_session_after_every_event(tmp_path, item, config)

    assert event_types(tmp_path, item.run_id) == [
        "RUN_STARTED",
        "SCORING_REQUESTED",
        "REWARD_COMMITTED",
        "CLUSTER_SUBMIT_REQUESTED",
        "CLUSTER_ACCEPTED",
        "RUN_RECORD_COMMITTED",
        "DECISION_OUTCOME_COMMITTED",
        "RUN_CLOSED",
    ]
    result = ProfiledTraceWorkflow(tmp_path, config).load_result(item.run_id)
    assert result.closed is not None
    assert result.closed.payload["status"] == "succeeded"
    assert result.reward is not None
    assert result.reward.payload["reward_basis_points"] == 10_000
    assert result.run_record is not None
    assert result.run_record.payload["reward_hash"] == result.reward.content_hash
    assert result.decision_outcome is not None
    assert result.decision_outcome.payload["run_record_hash"] == result.run_record.content_hash

    store = ArtifactStore(tmp_path)
    role_invocation = store.read(
        str(result.reward.payload["role_invocation_hash"]),
        expected_schema_name="RoleInvocation",
    )
    assert role_invocation.payload["seed"] == 1701
    assert role_invocation.payload["packet_id"] == "sjp-ticket01-0001"
    assert role_invocation.payload["session_lineage"] == "/root/ticket01_student_judge"
    assert role_invocation.payload["input_packet_hash"]
    assert role_invocation.payload["raw_output_hash"] == hashlib.sha256(RAW_STUDENT_OUTPUT.encode()).hexdigest()
    assert role_invocation.payload["normalized_output_hash"]
    raw_manifest = store.read(
        str(role_invocation.payload["raw_output_manifest_hash"]),
        expected_schema_name="RawRoleOutput",
    )
    byte_size = raw_manifest.payload["byte_size"]
    assert isinstance(byte_size, int) and not isinstance(byte_size, bool)
    assert (
        store.read_blob(str(raw_manifest.payload["blob_hash"]), expected_size=byte_size) == RAW_STUDENT_OUTPUT.encode()
    )
    # All role artifacts remain independently verifiable in a brand-new store session.
    ArtifactStore(tmp_path).read(
        str(role_invocation.payload["normalized_output_hash"]),
        expected_schema_name="NormalizedRoleOutput",
    )

    scorer = PersistentFixtureScorer(tmp_path, ArtifactStore(tmp_path), config.scorer_failure_code)
    cluster = PersistentFixtureCluster(tmp_path, ArtifactStore(tmp_path), config.cluster_failure_code)
    assert scorer.execution_count() == 1
    assert cluster.execution_count() == 1


class ConstructionTrap:
    def __init__(self) -> None:
        self.calls = 0

    def build_fixture(
        self,
        _root: Path,
        _store: ArtifactStore,
        _config: FixtureProfileConfig,
    ) -> tuple[PersistentFixtureScorer, PersistentFixtureCluster]:
        self.calls += 1
        raise AssertionError("an adapter was constructed before production readiness")


class MissingOutcomeScorer(PersistentFixtureScorer):
    def execute(
        self,
        idempotency_key: str,
        request: Artifact,
        input_packet: Mapping[str, object],
        raw_output: bytes,
        session_lineage: str,
        attempt_sequence: int = 1,
    ) -> Artifact:
        del input_packet, raw_output, session_lineage, attempt_sequence
        observation = self.store.put(
            "Observation",
            "1.0.0",
            {
                "failure_code": "TRANSIENT_ONLY",
                "idempotency_key": idempotency_key,
                "producer": "fixture_student_judge",
                "request_hash": request.content_hash,
                "status": "failed",
            },
        )
        observation.path.unlink()
        return observation


class MissingOutcomeCluster(PersistentFixtureCluster):
    def execute(
        self,
        idempotency_key: str,
        request: Artifact,
        attempt_sequence: int = 1,
    ) -> Artifact:
        del attempt_sequence
        observation = self.store.put(
            "Observation",
            "1.0.0",
            {
                "failure_code": "TRANSIENT_ONLY",
                "idempotency_key": idempotency_key,
                "producer": "fixture_cluster",
                "request_hash": request.content_hash,
                "status": "failed",
            },
        )
        observation.path.unlink()
        return observation


class MissingOutcomeFactory:
    def __init__(self, boundary: str) -> None:
        self.boundary = boundary

    def build_fixture(
        self,
        root: Path,
        store: ArtifactStore,
        config: FixtureProfileConfig,
    ) -> tuple[PersistentFixtureScorer, PersistentFixtureCluster]:
        scorer: PersistentFixtureScorer = PersistentFixtureScorer(root, store, config.scorer_failure_code)
        cluster: PersistentFixtureCluster = PersistentFixtureCluster(root, store, config.cluster_failure_code)
        if self.boundary == "scorer":
            scorer = MissingOutcomeScorer(root, store, config.scorer_failure_code)
        else:
            cluster = MissingOutcomeCluster(root, store, config.cluster_failure_code)
        return scorer, cluster


def test_production_placeholder_returns_phase_report_before_adapter_construction(
    tmp_path: Path,
) -> None:
    fixture_hash = hashlib.sha256(b"fixture-only-proof").hexdigest()
    config = ProductionProfileConfig(
        phase="TRAIN_35B",
        model_id="fixture-student-judge",
        fixture_evidence_hashes=(fixture_hash,),
    )
    factory = ConstructionTrap()

    snapshot = ProfiledTraceWorkflow(tmp_path, config, adapter_factory=factory).advance(trace_input(), epoch=1701)

    assert factory.calls == 0
    assert snapshot.closed is None
    assert snapshot.readiness_report is not None
    assert snapshot.readiness_report.schema_name == "ReadinessReport"
    assert snapshot.readiness_report.payload["execution_profile"] == "production"
    assert snapshot.readiness_report.payload["phase"] == "TRAIN_35B"
    assert snapshot.readiness_report.payload["status"] == "blocked"
    assert snapshot.readiness_report.payload["side_effects_permitted"] is False
    checks = snapshot.readiness_report.payload["checks"]
    assert isinstance(checks, list)
    assert all(isinstance(check, dict) for check in checks)
    codes = {check["code"] for check in checks if isinstance(check, dict)}
    assert "FIXTURE_EVIDENCE_NOT_PRODUCTION_PROOF" in codes
    assert "PLACEHOLDER_MODEL_ID" in codes
    assert "PRODUCTION_ADAPTER_CONTRACT_UNVERIFIED" in codes
    assert not (tmp_path / "boundaries").exists()
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("phase", 7),
        ("model_id", 122),
        ("dataset_version_hash", 1),
        ("judge_bundle_hash", False),
        ("experiment_spec_hash", []),
        ("cfs_root", 9),
        ("cluster_credential_ref", {}),
        ("fixture_evidence_hashes", "abc"),
        ("fixture_evidence_hashes", ("not-a-content-hash",)),
    ],
)
def test_production_profile_rejects_invalid_runtime_types_and_shapes(
    field: str,
    invalid: object,
) -> None:
    values: dict[str, object] = {"phase": "TRAIN_35B", field: invalid}

    with pytest.raises(ProfileConfigurationError, match="production"):
        ProductionProfileConfig(**values)  # type: ignore[arg-type]


def test_production_readiness_defensively_blocks_runtime_type_corruption_before_adapters(
    tmp_path: Path,
) -> None:
    config = ProductionProfileConfig(phase="TRAIN_35B")
    object.__setattr__(config, "model_id", 122)
    object.__setattr__(config, "dataset_version_hash", 7)
    factory = ConstructionTrap()

    snapshot = ProfiledTraceWorkflow(
        tmp_path,
        config,
        adapter_factory=factory,
    ).advance(trace_input(), epoch=1701)

    assert factory.calls == 0
    assert snapshot.readiness_report is not None
    assert snapshot.readiness_report.payload["status"] == "blocked"
    checks = snapshot.readiness_report.payload["checks"]
    assert isinstance(checks, list)
    codes = {check["code"] for check in checks if isinstance(check, dict)}
    assert "INVALID_PRODUCTION_CONFIGURATION" in codes
    assert not (tmp_path / "boundaries").exists()
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("profile_kind", ["fixture_as_production", "production_as_fixture"])
def test_runtime_profile_discriminator_mismatch_blocks_before_fixture_dispatch(
    tmp_path: Path,
    profile_kind: str,
) -> None:
    if profile_kind == "fixture_as_production":
        config: FixtureProfileConfig | ProductionProfileConfig = FixtureProfileConfig()
        object.__setattr__(config, "execution_profile", "production")
    else:
        config = ProductionProfileConfig(phase="TRAIN_35B")
        object.__setattr__(config, "execution_profile", "fixture")
    factory = ConstructionTrap()

    snapshot = ProfiledTraceWorkflow(
        tmp_path,
        config,
        adapter_factory=factory,
    ).advance(trace_input(), epoch=1701)

    assert factory.calls == 0
    assert snapshot.readiness_report is not None
    assert snapshot.readiness_report.payload["status"] == "blocked"
    checks = snapshot.readiness_report.payload["checks"]
    assert isinstance(checks, list)
    assert any(isinstance(check, dict) and check.get("code") == "INVALID_PRODUCTION_CONFIGURATION" for check in checks)
    assert not (tmp_path / "boundaries").exists()
    assert not (tmp_path / "runs").exists()


def test_production_surrogate_value_yields_sanitized_blocked_report(
    tmp_path: Path,
) -> None:
    sentinel = "SURROGATE_SECRET_DO_NOT_PERSIST"
    config = ProductionProfileConfig(phase="TRAIN_35B")
    object.__setattr__(config, "phase", f"\ud800{sentinel}")
    factory = ConstructionTrap()

    snapshot = ProfiledTraceWorkflow(
        tmp_path,
        config,
        adapter_factory=factory,
    ).advance(trace_input(), epoch=1701)

    assert factory.calls == 0
    assert snapshot.readiness_report is not None
    assert snapshot.readiness_report.payload["status"] == "blocked"
    assert snapshot.readiness_report.payload["phase"] == "INVALID"
    persisted = b"\n".join(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())
    assert sentinel.encode() not in persisted
    assert not (tmp_path / "boundaries").exists()
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("forbidden", ["api_key", "credential_ref", "adapter", "cfs_root"])
def test_fixture_profile_structurally_rejects_production_configuration(forbidden: str) -> None:
    with pytest.raises(ProfileConfigurationError, match=forbidden):
        FixtureProfileConfig.from_mapping({"execution_profile": "fixture", forbidden: "secret"})


@pytest.mark.parametrize(
    "schedule",
    [
        ("timeout",),
        ("success", "timeout"),
        ("bogus", "success"),
        ("timeout", "success", "permanent_failure"),
        ("timeout",) * 16 + ("success",),
    ],
)
def test_fixture_fault_schedule_rejects_illegal_or_unbounded_programs(
    schedule: tuple[str, ...],
) -> None:
    with pytest.raises(ProfileConfigurationError, match="must end once"):
        FixtureProfileConfig(scorer_fault_schedule=schedule)


@pytest.mark.parametrize("boundary", ["scorer", "cluster"])
def test_committed_side_effect_is_queried_and_not_repeated_after_crash(
    tmp_path: Path,
    boundary: str,
) -> None:
    item = trace_input(run_id=f"fixture-run-crash-{boundary}")
    config = FixtureProfileConfig()
    target = "SCORING_REQUESTED" if boundary == "scorer" else "CLUSTER_SUBMIT_REQUESTED"
    while not event_types(tmp_path, item.run_id) or event_types(tmp_path, item.run_id)[-1] != target:
        ProfiledTraceWorkflow(tmp_path, config).advance(item, epoch=1701)

    with pytest.raises(InjectedProcessCrash, match=boundary):
        ProfiledTraceWorkflow(tmp_path, config).advance(
            item,
            epoch=1701,
            crash_after_side_effect=boundary,
        )
    events_before_recovery = event_types(tmp_path, item.run_id)
    ProfiledTraceWorkflow(tmp_path, config).advance(item, epoch=1701)
    assert len(event_types(tmp_path, item.run_id)) == len(events_before_recovery) + 1

    scorer = PersistentFixtureScorer(tmp_path, ArtifactStore(tmp_path), None)
    cluster = PersistentFixtureCluster(tmp_path, ArtifactStore(tmp_path), None)
    assert scorer.execution_count() == 1
    if boundary == "cluster":
        assert cluster.execution_count() == 1


@pytest.mark.parametrize("boundary", ["scorer", "cluster"])
def test_transient_in_memory_boundary_outcome_missing_from_store_is_integrity_failure(
    tmp_path: Path,
    boundary: str,
) -> None:
    item = trace_input(run_id=f"fixture-run-missing-{boundary}-outcome")
    workflow = ProfiledTraceWorkflow(
        tmp_path,
        FixtureProfileConfig(),
        adapter_factory=MissingOutcomeFactory(boundary),
    )
    target = "SCORING_REQUESTED" if boundary == "scorer" else "CLUSTER_SUBMIT_REQUESTED"
    snapshot = workflow.advance(item, epoch=1701)
    while snapshot.events[-1].payload["event_type"] != target:
        snapshot = workflow.advance(item, epoch=1701)

    snapshot = workflow.advance(item, epoch=1701)

    expected_event = "SCORING_FAILED" if boundary == "scorer" else "CLUSTER_FAILED"
    expected_reason = "SCORER_OUTCOME_CORRUPTION" if boundary == "scorer" else "CLUSTER_OUTCOME_CORRUPTION"
    assert snapshot.events[-1].payload["event_type"] == expected_event
    while snapshot.closed is None:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    assert snapshot.closed.payload["reason_code"] == expected_reason
    assert "REWARD_COMMITTED" not in event_types(tmp_path, item.run_id) or boundary == "cluster"
    assert "CLUSTER_ACCEPTED" not in event_types(tmp_path, item.run_id)


@pytest.mark.parametrize("boundary", ["scorer", "cluster"])
def test_boundary_adapter_construction_failure_is_audited_in_fresh_process(
    tmp_path: Path,
    boundary: str,
) -> None:
    item = trace_input(run_id=f"fixture-run-{boundary}-adapter-lock-unavailable")
    config = FixtureProfileConfig()
    target = "SCORING_REQUESTED" if boundary == "scorer" else "CLUSTER_SUBMIT_REQUESTED"
    snapshot = ProfiledTraceWorkflow(tmp_path, config).advance(item, epoch=1701)
    while snapshot.events[-1].payload["event_type"] != target:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    requested = snapshot.events[-1]
    lock_path = tmp_path / "boundaries" / boundary / "adapter.lock"
    if lock_path.exists():
        lock_path.unlink()
    lock_path.mkdir(parents=True)

    resume_in_fresh_process(tmp_path, item.run_id)
    snapshot = ProfiledTraceWorkflow(tmp_path, config).load_result(item.run_id)

    expected_event = "SCORING_FAILED" if boundary == "scorer" else "CLUSTER_FAILED"
    expected_reason = "SCORER_OUTCOME_CORRUPTION" if boundary == "scorer" else "CLUSTER_OUTCOME_CORRUPTION"
    assert snapshot.events[-1].payload["event_type"] == expected_event
    failed_details = snapshot.events[-1].payload["details"]
    assert isinstance(failed_details, dict)
    observation = ArtifactStore(tmp_path).read(
        str(failed_details["observation_hash"]),
        expected_schema_name="Observation",
    )
    assert observation.payload["source_event_hash"] == requested.content_hash
    while snapshot.closed is None:
        resume_in_fresh_process(tmp_path, item.run_id)
        snapshot = ProfiledTraceWorkflow(tmp_path, config).load_result(item.run_id)
    assert snapshot.closed.payload["reason_code"] == expected_reason


@pytest.mark.parametrize("boundary", ["scorer", "cluster"])
@pytest.mark.parametrize("stage", ["factory", "query", "execute"])
def test_non_artifact_boundary_exception_is_sanitized_and_closed_from_fresh_process(
    tmp_path: Path,
    boundary: str,
    stage: str,
) -> None:
    item = trace_input(run_id=f"fixture-run-{boundary}-{stage}-runtime-error")
    target = "SCORING_REQUESTED" if boundary == "scorer" else "CLUSTER_SUBMIT_REQUESTED"
    snapshot = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(
        item,
        epoch=1701,
    )
    while snapshot.events[-1].payload["event_type"] != target:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    source_event = snapshot.events[-1]

    result = inject_adapter_exception_in_fresh_process(
        tmp_path,
        item.run_id,
        boundary=boundary,
        stage=stage,
    )

    expected_event = "SCORING_FAILED" if boundary == "scorer" else "CLUSTER_FAILED"
    expected_code = "SCORER_ADAPTER_FAILURE" if boundary == "scorer" else "CLUSTER_ADAPTER_FAILURE"
    expected_producer = "scorer_adapter_gate" if boundary == "scorer" else "cluster_adapter_gate"
    expected_detail = (
        "scorer adapter interaction failed safely"
        if boundary == "scorer"
        else "cluster adapter interaction failed safely"
    )
    assert result["event_type"] == expected_event
    assert result["failure_code"] == expected_code
    observation = ArtifactStore(tmp_path).read(
        str(result["observation_hash"]),
        expected_schema_name="Observation",
    )
    assert observation.payload["candidate_observation_hash"] is None
    assert observation.payload["detail"] == expected_detail
    assert observation.payload["failure_code"] == expected_code
    assert observation.payload["producer"] == expected_producer
    assert observation.payload["source_event_hash"] == source_event.content_hash
    if boundary == "scorer":
        assert observation.payload["role_invocation_hash"] is None
    else:
        assert "role_invocation_hash" not in observation.payload

    secret = b"runtime-adapter-secret-DO-NOT-PERSIST"
    assert all(secret not in path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())

    snapshot = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    while snapshot.closed is None:
        resume_in_fresh_process(tmp_path, item.run_id, epoch=1701)
        snapshot = ProfiledTraceWorkflow(
            tmp_path,
            FixtureProfileConfig(),
        ).load_result(item.run_id)
    assert snapshot.closed.payload["reason_code"] == expected_code
    RunJournal(tmp_path, ArtifactStore(tmp_path), item.run_id).verify()


@pytest.mark.parametrize("boundary", ["scorer", "cluster"])
def test_missing_boundary_request_artifact_is_audited_in_fresh_process(
    tmp_path: Path,
    boundary: str,
) -> None:
    item = trace_input(run_id=f"fixture-run-{boundary}-missing-request-artifact")
    config = FixtureProfileConfig()
    target = "SCORING_REQUESTED" if boundary == "scorer" else "CLUSTER_SUBMIT_REQUESTED"
    snapshot = ProfiledTraceWorkflow(tmp_path, config).advance(item, epoch=1701)
    while snapshot.events[-1].payload["event_type"] != target:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    requested = snapshot.events[-1]
    details = requested.payload["details"]
    assert isinstance(details, dict)
    request_hash = str(details["request_hash"])
    (tmp_path / "artifacts" / f"{request_hash}.json").unlink()

    resume_in_fresh_process(tmp_path, item.run_id)
    snapshot = ProfiledTraceWorkflow(tmp_path, config).load_result(item.run_id)

    expected_event = "SCORING_FAILED" if boundary == "scorer" else "CLUSTER_FAILED"
    expected_reason = "SCORER_OUTCOME_CORRUPTION" if boundary == "scorer" else "CLUSTER_OUTCOME_CORRUPTION"
    assert snapshot.events[-1].payload["event_type"] == expected_event
    failed_details = snapshot.events[-1].payload["details"]
    assert isinstance(failed_details, dict)
    observation = ArtifactStore(tmp_path).read(
        str(failed_details["observation_hash"]),
        expected_schema_name="Observation",
    )
    assert observation.payload["request_hash"] == request_hash
    assert observation.payload["source_event_hash"] == requested.content_hash
    while snapshot.closed is None:
        resume_in_fresh_process(tmp_path, item.run_id)
        snapshot = ProfiledTraceWorkflow(tmp_path, config).load_result(item.run_id)
    assert snapshot.closed.payload["reason_code"] == expected_reason


@pytest.mark.parametrize(
    ("config", "failure_event", "reason_code"),
    [
        (
            FixtureProfileConfig(scorer_failure_code="SCORER_TIMEOUT"),
            "SCORING_FAILED",
            "SCORER_TIMEOUT",
        ),
        (
            FixtureProfileConfig(cluster_failure_code="CLUSTER_REJECTED"),
            "CLUSTER_FAILED",
            "CLUSTER_REJECTED",
        ),
    ],
)
def test_fixture_boundary_failure_closes_an_auditable_failed_run(
    tmp_path: Path,
    config: FixtureProfileConfig,
    failure_event: str,
    reason_code: str,
) -> None:
    item = trace_input(run_id=f"fixture-run-failure-{reason_code.lower()}")

    run_with_fresh_session_after_every_event(tmp_path, item, config)

    result = ProfiledTraceWorkflow(tmp_path, config).load_result(item.run_id)
    assert result.closed is not None
    assert result.closed.payload == {
        "controller_epoch": 1701,
        "previous_event_hash": result.events[-2].content_hash,
        "reason_code": reason_code,
        "run_id": item.run_id,
        "status": "failed",
        "terminal_sequence": len(result.events),
    }
    assert failure_event in [event.payload["event_type"] for event in result.events]
    assert result.run_record is not None
    assert result.run_record.payload["status"] == "failed"
    assert result.decision_outcome is not None
    assert result.decision_outcome.payload["status"] == "failed"
    executions_before = (
        PersistentFixtureScorer(tmp_path, ArtifactStore(tmp_path), config.scorer_failure_code).execution_count(),
        PersistentFixtureCluster(tmp_path, ArtifactStore(tmp_path), config.cluster_failure_code).execution_count(),
    )
    # Terminal replay is read-only and cannot repeat either failed or successful actions.
    ProfiledTraceWorkflow(tmp_path, config).advance(item, epoch=1701)
    assert (
        PersistentFixtureScorer(tmp_path, ArtifactStore(tmp_path), config.scorer_failure_code).execution_count(),
        PersistentFixtureCluster(tmp_path, ArtifactStore(tmp_path), config.cluster_failure_code).execution_count(),
    ) == executions_before


@pytest.mark.parametrize("boundary", ["scorer", "cluster"])
def test_durable_timeout_attempt_retries_then_succeeds_across_fresh_processes(
    tmp_path: Path,
    boundary: str,
) -> None:
    schedule = ("timeout", "success")
    config = (
        FixtureProfileConfig(scorer_fault_schedule=schedule)
        if boundary == "scorer"
        else FixtureProfileConfig(cluster_fault_schedule=schedule)
    )
    item = trace_input(run_id=f"fixture-run-{boundary}-timeout-retry-success")
    snapshot = ProfiledTraceWorkflow(tmp_path, config).advance(item, epoch=1701)
    while snapshot.closed is None:
        resume_in_fresh_process(tmp_path, item.run_id)
        snapshot = ProfiledTraceWorkflow(tmp_path, config).load_result(item.run_id)

    retry_event = "SCORING_RETRY_SCHEDULED" if boundary == "scorer" else "CLUSTER_RETRY_SCHEDULED"
    assert retry_event in event_types(tmp_path, item.run_id)
    assert snapshot.closed.payload["status"] == "succeeded"
    attempt_refs = sorted((tmp_path / "boundaries" / boundary / "attempts").rglob("*.ref"))
    assert len(attempt_refs) == 2
    attempts = [
        ArtifactStore(tmp_path).read(
            path.read_text().strip(),
            expected_schema_name="FixtureBoundaryAttempt",
        )
        for path in attempt_refs
    ]
    assert [attempt.payload["attempt_sequence"] for attempt in attempts] == [1, 2]
    assert [attempt.payload["directive"] for attempt in attempts] == [
        "timeout",
        "success",
    ]


@pytest.mark.parametrize("boundary", ["scorer", "cluster"])
def test_delayed_out_of_order_completion_is_durably_ignored_by_newer_attempt(
    tmp_path: Path,
    boundary: str,
) -> None:
    schedule = ("delayed", "timeout", "success")
    config = (
        FixtureProfileConfig(scorer_fault_schedule=schedule)
        if boundary == "scorer"
        else FixtureProfileConfig(cluster_fault_schedule=schedule)
    )
    item = trace_input(run_id=f"fixture-run-{boundary}-delayed-out-of-order")
    snapshot = ProfiledTraceWorkflow(tmp_path, config).advance(item, epoch=1701)
    while snapshot.closed is None:
        resume_in_fresh_process(tmp_path, item.run_id)
        snapshot = ProfiledTraceWorkflow(tmp_path, config).load_result(item.run_id)

    attempt_refs = sorted((tmp_path / "boundaries" / boundary / "attempts").rglob("*.ref"))
    assert len(attempt_refs) == 3
    attempts = [
        ArtifactStore(tmp_path).read(
            path.read_text().strip(),
            expected_schema_name="FixtureBoundaryAttempt",
        )
        for path in attempt_refs
    ]
    assert [attempt.payload["directive"] for attempt in attempts] == [
        "delayed",
        "timeout",
        "success",
    ]
    late_hashes = attempts[1].payload["late_completion_hashes"]
    assert isinstance(late_hashes, list) and len(late_hashes) == 1
    late = ArtifactStore(tmp_path).read(
        str(late_hashes[0]),
        expected_schema_name="FixtureLateBoundaryCompletion",
    )
    assert late.payload["origin_attempt_sequence"] == 1
    assert late.payload["observed_at_attempt"] == 2
    assert late.payload["status"] == "available_late_ignored"
    assert snapshot.closed.payload["status"] == "succeeded"


@pytest.mark.parametrize("boundary", ["scorer", "cluster"])
def test_retry_attempt_is_idempotent_across_crash_before_journal_commit(
    tmp_path: Path,
    boundary: str,
) -> None:
    schedule = ("timeout", "success")
    config = (
        FixtureProfileConfig(scorer_fault_schedule=schedule)
        if boundary == "scorer"
        else FixtureProfileConfig(cluster_fault_schedule=schedule)
    )
    item = trace_input(run_id=f"fixture-run-{boundary}-attempt-crash-idempotency")
    target = "SCORING_REQUESTED" if boundary == "scorer" else "CLUSTER_SUBMIT_REQUESTED"
    snapshot = ProfiledTraceWorkflow(tmp_path, config).advance(item, epoch=1701)
    while snapshot.events[-1].payload["event_type"] != target:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)

    with pytest.raises(InjectedProcessCrash, match=boundary):
        ProfiledTraceWorkflow.resume(
            tmp_path,
            item.run_id,
            epoch=1701,
            crash_after_side_effect=boundary,
        )
    attempt_dir = tmp_path / "boundaries" / boundary / "attempts"
    assert len(list(attempt_dir.rglob("*.ref"))) == 1

    ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)

    assert len(list(attempt_dir.rglob("*.ref"))) == 1


@pytest.mark.parametrize("boundary", ["scorer", "cluster"])
def test_concurrent_same_attempt_commits_one_durable_attempt_and_outcome(
    tmp_path: Path,
    boundary: str,
) -> None:
    schedule = ("timeout", "success")
    config = (
        FixtureProfileConfig(scorer_fault_schedule=schedule)
        if boundary == "scorer"
        else FixtureProfileConfig(cluster_fault_schedule=schedule)
    )
    item = trace_input(run_id=f"fixture-run-{boundary}-concurrent-attempt")
    target = "SCORING_REQUESTED" if boundary == "scorer" else "CLUSTER_SUBMIT_REQUESTED"
    snapshot = ProfiledTraceWorkflow(tmp_path, config).advance(item, epoch=1701)
    while snapshot.events[-1].payload["event_type"] != target:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)

    details = snapshot.events[-1].payload["details"]
    assert isinstance(details, dict)
    key = str(details["idempotency_key"])
    request_hash = str(details["request_hash"])
    store = ArtifactStore(tmp_path)
    request = store.read(
        request_hash,
        expected_schema_name=("ScoringRequest" if boundary == "scorer" else "ClusterSubmitRequest"),
    )
    scorer = PersistentFixtureScorer(tmp_path, store, None, schedule)
    cluster = PersistentFixtureCluster(tmp_path, store, None, schedule)

    def run_attempt(attempt_sequence: int, barrier: Barrier) -> str:
        barrier.wait()
        if boundary == "scorer":
            observation = scorer.execute(
                key,
                request,
                item.student_input_packet,
                item.raw_student_output,
                item.role_session_lineage,
                attempt_sequence,
            )
        else:
            observation = cluster.execute(key, request, attempt_sequence)
        return observation.content_hash

    for attempt_sequence in (1, 2):
        barrier = Barrier(16)
        with ThreadPoolExecutor(max_workers=16) as pool:
            hashes = list(
                pool.map(
                    lambda _, attempt_sequence=attempt_sequence, barrier=barrier: run_attempt(
                        attempt_sequence,
                        barrier,
                    ),
                    range(16),
                )
            )
        assert len(set(hashes)) == 1
        attempt_refs = sorted((tmp_path / "boundaries" / boundary / "attempts").rglob("*.ref"))
        assert len(attempt_refs) == attempt_sequence

    outcome_refs = list((tmp_path / "boundaries" / boundary / "outcomes").glob("*.ref"))
    assert len(outcome_refs) == 1
    with pytest.raises(ArtifactCorruption, match="exhausted or invalid"):
        if boundary == "scorer":
            scorer.execute(
                key,
                request,
                item.student_input_packet,
                item.raw_student_output,
                item.role_session_lineage,
                3,
            )
        else:
            cluster.execute(key, request, 3)


@pytest.mark.parametrize("boundary", ["scorer", "cluster"])
@pytest.mark.parametrize(
    "tampering",
    [
        "missing_ref",
        "missing_artifact",
        "extra_field",
        "wrong_request_hash",
        "wrong_schedule_hash",
        "wrong_previous_hash",
        "wrong_sequence",
    ],
)
def test_attempt_ledger_corruption_fails_closed_before_next_boundary_attempt(
    tmp_path: Path,
    boundary: str,
    tampering: str,
) -> None:
    schedule = ("timeout", "success")
    config = (
        FixtureProfileConfig(scorer_fault_schedule=schedule)
        if boundary == "scorer"
        else FixtureProfileConfig(cluster_fault_schedule=schedule)
    )
    item = trace_input(run_id=f"fixture-run-{boundary}-attempt-corrupt-{tampering}")
    retry_event = "SCORING_RETRY_SCHEDULED" if boundary == "scorer" else "CLUSTER_RETRY_SCHEDULED"
    snapshot = ProfiledTraceWorkflow(tmp_path, config).advance(item, epoch=1701)
    while snapshot.events[-1].payload["event_type"] != retry_event:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)

    attempt_refs = list((tmp_path / "boundaries" / boundary / "attempts").rglob("*.ref"))
    assert len(attempt_refs) == 1
    attempt_ref = attempt_refs[0]
    if tampering == "missing_ref":
        attempt_ref.unlink()
    elif tampering == "missing_artifact":
        attempt_ref.write_bytes(b"0" * 64 + b"\n")
        assert not (tmp_path / "artifacts" / f"{'0' * 64}.json").exists()
    else:
        store = ArtifactStore(tmp_path)
        genuine = store.read(
            attempt_ref.read_text().strip(),
            expected_schema_name="FixtureBoundaryAttempt",
        )
        payload: dict[str, object] = dict(genuine.payload)
        if tampering == "extra_field":
            payload["untrusted_attempt_state"] = "accepted"
        elif tampering == "wrong_request_hash":
            payload["request_hash"] = "f" * 64
        elif tampering == "wrong_schedule_hash":
            payload["schedule_hash"] = "f" * 64
        elif tampering == "wrong_previous_hash":
            payload["previous_attempt_hash"] = "f" * 64
        elif tampering == "wrong_sequence":
            payload["attempt_sequence"] = 2
        forged = store.put("FixtureBoundaryAttempt", "1.0.0", payload)
        attempt_ref.write_bytes(f"{forged.content_hash}\n".encode("ascii"))

    resume_in_fresh_process(tmp_path, item.run_id)
    snapshot = ProfiledTraceWorkflow(tmp_path, config).load_result(item.run_id)
    failed_event = "SCORING_FAILED" if boundary == "scorer" else "CLUSTER_FAILED"
    reason_code = "SCORER_OUTCOME_CORRUPTION" if boundary == "scorer" else "CLUSTER_OUTCOME_CORRUPTION"
    assert snapshot.events[-1].payload["event_type"] == failed_event
    while snapshot.closed is None:
        resume_in_fresh_process(tmp_path, item.run_id)
        snapshot = ProfiledTraceWorkflow(tmp_path, config).load_result(item.run_id)
    assert snapshot.closed.payload["reason_code"] == reason_code
    assert len(list((attempt_ref.parent).glob("*.ref"))) <= 1
    assert not list((tmp_path / "boundaries" / boundary / "outcomes").glob("*.ref"))


@pytest.mark.parametrize("boundary", ["scorer", "cluster"])
def test_permanent_failure_directive_cannot_commit_a_rogue_success(
    tmp_path: Path,
    boundary: str,
) -> None:
    config = (
        FixtureProfileConfig(scorer_failure_code="SCORER_TIMEOUT")
        if boundary == "scorer"
        else FixtureProfileConfig(cluster_failure_code="CLUSTER_REJECTED")
    )
    item = trace_input(run_id=f"fixture-run-{boundary}-permanent-rogue-success")
    target = "SCORING_REQUESTED" if boundary == "scorer" else "CLUSTER_SUBMIT_REQUESTED"
    snapshot = ProfiledTraceWorkflow(tmp_path, config).advance(item, epoch=1701)
    while snapshot.events[-1].payload["event_type"] != target:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)

    details = snapshot.events[-1].payload["details"]
    assert isinstance(details, dict)
    key = str(details["idempotency_key"])
    request_hash = str(details["request_hash"])
    store = ArtifactStore(tmp_path)
    request = store.read(
        request_hash,
        expected_schema_name=("ScoringRequest" if boundary == "scorer" else "ClusterSubmitRequest"),
    )
    if boundary == "scorer":
        scratch = PersistentFixtureScorer(
            tmp_path / "scratch",
            store,
            None,
            ("success",),
        )
        genuine = scratch.execute(
            key,
            request,
            item.student_input_packet,
            item.raw_student_output,
            item.role_session_lineage,
            1,
        )
        configured: PersistentFixtureScorer | PersistentFixtureCluster = PersistentFixtureScorer(
            tmp_path,
            store,
            config.scorer_failure_code,
            config.effective_scorer_fault_schedule(),
        )
    else:
        scratch_cluster = PersistentFixtureCluster(
            tmp_path / "scratch",
            store,
            None,
            ("success",),
        )
        genuine = scratch_cluster.execute(key, request, 1)
        configured = PersistentFixtureCluster(
            tmp_path,
            store,
            config.cluster_failure_code,
            config.effective_cluster_fault_schedule(),
        )
    rogue_payload: dict[str, object] = dict(genuine.payload)
    rogue_payload["directive"] = "permanent_failure"
    rogue = store.put("Observation", "1.0.0", rogue_payload)
    attempt = store.put(
        "FixtureBoundaryAttempt",
        "1.0.0",
        {
            "attempt_sequence": 1,
            "available_after_attempt": None,
            "boundary": boundary,
            "directive": "permanent_failure",
            "idempotency_key": key,
            "late_completion_hashes": [],
            "observation_hash": rogue.content_hash,
            "previous_attempt_hash": None,
            "request_hash": request_hash,
            "schedule_hash": configured.attempt_ledger.schedule_hash,
            "schedule_version": "fixture-fault-schedule/1.0.0",
            "status": "final",
        },
    )
    attempt_ref = tmp_path / "boundaries" / boundary / "attempts" / key / "00000000000000000001.ref"
    ArtifactStore._publish(
        attempt_ref,
        f"{attempt.content_hash}\n".encode("ascii"),
    )
    ArtifactStore._publish(
        tmp_path / "boundaries" / boundary / "outcomes" / f"{key}.ref",
        f"{rogue.content_hash}\n".encode("ascii"),
    )

    snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)

    failed_event = "SCORING_FAILED" if boundary == "scorer" else "CLUSTER_FAILED"
    reason = "SCORER_OUTCOME_CORRUPTION" if boundary == "scorer" else "CLUSTER_OUTCOME_CORRUPTION"
    assert snapshot.events[-1].payload["event_type"] == failed_event
    while snapshot.closed is None:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    assert snapshot.closed.payload["reason_code"] == reason


def test_student_packet_rejects_non_allowlisted_teacher_data_fail_closed(tmp_path: Path) -> None:
    item = trace_input(run_id="fixture-run-student-isolation")
    packet = dict(item.student_input_packet)
    packet["teacher_label"] = "teacher-only"
    isolated = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id=item.dataset_version_id,
        experiment_spec_id=item.experiment_spec_id,
        trace_id=item.trace_id,
        trajectory_id=item.trajectory_id,
        prompt=item.prompt,
        response=item.response,
        judge_pack=item.judge_pack,
        student_input_packet=packet,
        raw_student_output=item.raw_student_output,
        role_session_lineage=item.role_session_lineage,
    )

    run_with_fresh_session_after_every_event(tmp_path, isolated, FixtureProfileConfig())

    result = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    assert result.closed is not None
    assert result.closed.payload["status"] == "failed"
    assert result.closed.payload["reason_code"] == "ROLE_PACKET_NOT_ALLOWLISTED"


@pytest.mark.parametrize(
    "marker",
    [
        "TEACHER-ONLY",
        "Holdout Only",
        "teacher_label",
        "Teacher.Secret",
    ],
)
def test_prohibited_role_marker_in_allowlisted_string_is_rejected_before_persist(
    tmp_path: Path,
    marker: str,
) -> None:
    sentinel = f"{marker}_SENSITIVE_SENTINEL"
    item = trace_input(run_id=f"fixture-run-string-marker-{hashlib.sha256(marker.encode()).hexdigest()[:10]}")
    packet = json.loads(json.dumps(item.student_input_packet))
    packet["rubric"]["instruction_following"] = sentinel
    rejected = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id=item.dataset_version_id,
        experiment_spec_id=item.experiment_spec_id,
        trace_id=item.trace_id,
        trajectory_id=item.trajectory_id,
        prompt=item.prompt,
        response=item.response,
        judge_pack=item.judge_pack,
        student_input_packet=packet,
        raw_student_output=item.raw_student_output,
        role_session_lineage=item.role_session_lineage,
    )

    run_with_fresh_session_after_every_event(
        tmp_path,
        rejected,
        FixtureProfileConfig(),
    )

    result = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    assert result.closed is not None
    assert result.closed.payload["reason_code"] == "ROLE_PACKET_NOT_ALLOWLISTED"
    persisted = b"\n".join(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())
    assert sentinel.encode() not in persisted
    schemas = {json.loads(path.read_text())["schema_name"] for path in (tmp_path / "artifacts").glob("*.json")}
    assert "StudentJudgeInputPacket" not in schemas


def test_natural_language_teacher_word_without_role_marker_remains_allowlisted(
    tmp_path: Path,
) -> None:
    item = trace_input(run_id="fixture-run-natural-teacher-word")
    packet = json.loads(json.dumps(item.student_input_packet))
    packet["rubric"]["instruction_following"] = "A teacher can explain why both ordered recovery steps are actionable."
    allowed = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id=item.dataset_version_id,
        experiment_spec_id=item.experiment_spec_id,
        trace_id=item.trace_id,
        trajectory_id=item.trajectory_id,
        prompt=item.prompt,
        response=item.response,
        judge_pack=item.judge_pack,
        student_input_packet=packet,
        raw_student_output=item.raw_student_output,
        role_session_lineage=item.role_session_lineage,
    )

    run_with_fresh_session_after_every_event(
        tmp_path,
        allowed,
        FixtureProfileConfig(),
    )
    result = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    assert result.closed is not None
    assert result.closed.payload["status"] == "succeeded"


def test_rejected_student_packet_never_persists_prohibited_keys_or_values(tmp_path: Path) -> None:
    sentinel = "TEACHER_ONLY_SENTINEL_DO_NOT_PERSIST"
    item = trace_input(run_id="fixture-run-student-sentinel-isolation")
    packet = json.loads(json.dumps(item.student_input_packet))
    packet["teacher_label"] = sentinel
    rejected = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id=item.dataset_version_id,
        experiment_spec_id=item.experiment_spec_id,
        trace_id=item.trace_id,
        trajectory_id=item.trajectory_id,
        prompt=item.prompt,
        response=item.response,
        judge_pack=item.judge_pack,
        student_input_packet=packet,
        raw_student_output=item.raw_student_output,
        role_session_lineage=item.role_session_lineage,
    )

    run_with_fresh_session_after_every_event(tmp_path, rejected, FixtureProfileConfig())

    result = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    assert result.closed is not None
    assert result.closed.payload["reason_code"] == "ROLE_PACKET_NOT_ALLOWLISTED"
    persisted_bytes = b"\n".join(path.read_bytes() for path in sorted(tmp_path.rglob("*")) if path.is_file())
    assert sentinel.encode() not in persisted_bytes
    assert b"teacher_label" not in persisted_bytes
    schemas = {json.loads(path.read_text())["schema_name"] for path in (tmp_path / "artifacts").glob("*.json")}
    assert "StudentJudgeInputPacket" not in schemas
    assert "StudentJudgeFixtureExchange" not in schemas
    assert "RawRoleOutput" not in schemas


def test_student_role_identity_value_is_rejected_before_any_packet_materialization(
    tmp_path: Path,
) -> None:
    sentinel = "teacher_scorer_SECRET_SENTINEL"
    item = trace_input(run_id="fixture-run-student-role-value-isolation")
    packet = json.loads(json.dumps(item.student_input_packet))
    packet["role"] = sentinel
    rejected = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id=item.dataset_version_id,
        experiment_spec_id=item.experiment_spec_id,
        trace_id=item.trace_id,
        trajectory_id=item.trajectory_id,
        prompt=item.prompt,
        response=item.response,
        judge_pack=item.judge_pack,
        student_input_packet=packet,
        raw_student_output=item.raw_student_output,
        role_session_lineage=item.role_session_lineage,
    )

    run_with_fresh_session_after_every_event(tmp_path, rejected, FixtureProfileConfig())

    result = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    assert result.closed is not None
    assert result.closed.payload["reason_code"] == "ROLE_PACKET_NOT_ALLOWLISTED"
    persisted_bytes = b"\n".join(path.read_bytes() for path in sorted(tmp_path.rglob("*")) if path.is_file())
    assert sentinel.encode() not in persisted_bytes
    schemas = {json.loads(path.read_text())["schema_name"] for path in (tmp_path / "artifacts").glob("*.json")}
    assert "StudentJudgeInputPacket" not in schemas


def test_prohibited_judge_pack_is_rejected_before_judge_pack_or_experiment_store(
    tmp_path: Path,
) -> None:
    sentinel = "TEACHER_LABEL_SENTINEL_MUST_NEVER_PERSIST"
    item = trace_input(run_id="fixture-run-judge-pack-preflight-isolation")
    packet = json.loads(json.dumps(item.student_input_packet))
    packet["judge_pack"]["teacher_label"] = sentinel
    rejected = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id=item.dataset_version_id,
        experiment_spec_id=item.experiment_spec_id,
        trace_id=item.trace_id,
        trajectory_id=item.trajectory_id,
        prompt=item.prompt,
        response=item.response,
        judge_pack=packet["judge_pack"],
        student_input_packet=packet,
        raw_student_output=item.raw_student_output,
        role_session_lineage=item.role_session_lineage,
    )

    run_with_fresh_session_after_every_event(tmp_path, rejected, FixtureProfileConfig())

    result = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    assert result.closed is not None
    assert result.closed.payload["reason_code"] == "ROLE_PACKET_NOT_ALLOWLISTED"
    persisted_bytes = b"\n".join(path.read_bytes() for path in sorted(tmp_path.rglob("*")) if path.is_file())
    assert sentinel.encode() not in persisted_bytes
    assert b"teacher_label" not in persisted_bytes
    schemas = {json.loads(path.read_text())["schema_name"] for path in (tmp_path / "artifacts").glob("*.json")}
    assert "StudentJudgeInputPacket" not in schemas
    assert "JudgePack" not in schemas
    assert "ExperimentSpec" not in schemas


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("items_per_turn", False),
        ("reward_schema_version", None),
        ("scalarizer_version", True),
        ("reward_schema_version", "reward-schema/2.0.0"),
    ],
)
def test_judge_pack_types_and_versions_fail_closed_before_reward(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    case_id = hashlib.sha256(f"{field}:{invalid!r}".encode()).hexdigest()[:12]
    item = trace_input(run_id=f"fixture-run-invalid-judge-pack-{case_id}")
    packet = json.loads(json.dumps(item.student_input_packet))
    packet["judge_pack"][field] = invalid
    rejected = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id=item.dataset_version_id,
        experiment_spec_id=item.experiment_spec_id,
        trace_id=item.trace_id,
        trajectory_id=item.trajectory_id,
        prompt=item.prompt,
        response=item.response,
        judge_pack=packet["judge_pack"],
        student_input_packet=packet,
        raw_student_output=item.raw_student_output,
        role_session_lineage=item.role_session_lineage,
    )

    run_with_fresh_session_after_every_event(tmp_path, rejected, FixtureProfileConfig())

    result = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    assert result.closed is not None
    assert result.closed.payload["reason_code"] == "ROLE_PACKET_NOT_ALLOWLISTED"
    assert result.reward is None


@pytest.mark.parametrize("container", ["trajectory", "judge_pack", "output_schema"])
def test_student_packet_rejects_teacher_data_nested_inside_an_allowed_container(tmp_path: Path, container: str) -> None:
    item = trace_input(run_id=f"fixture-run-nested-isolation-{container}")
    packet = json.loads(json.dumps(item.student_input_packet))
    packet[container]["teacher_label"] = "teacher-only"
    isolated = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id=item.dataset_version_id,
        experiment_spec_id=item.experiment_spec_id,
        trace_id=item.trace_id,
        trajectory_id=item.trajectory_id,
        prompt=item.prompt,
        response=item.response,
        judge_pack=item.judge_pack,
        student_input_packet=packet,
        raw_student_output=item.raw_student_output,
        role_session_lineage=item.role_session_lineage,
    )

    run_with_fresh_session_after_every_event(tmp_path, isolated, FixtureProfileConfig())

    result = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    assert result.closed is not None
    assert result.closed.payload["reason_code"] == "ROLE_PACKET_NOT_ALLOWLISTED"


def test_student_seed_is_validated_and_persisted_as_data_not_a_success_constant(
    tmp_path: Path,
) -> None:
    item = trace_input(run_id="fixture-run-alternate-safe-seed")
    packet = json.loads(json.dumps(item.student_input_packet))
    packet["seed"] = 1702
    raw = json.loads(item.raw_student_output)
    raw["seed"] = 1702
    changed = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id=item.dataset_version_id,
        experiment_spec_id=item.experiment_spec_id,
        trace_id=item.trace_id,
        trajectory_id=item.trajectory_id,
        prompt=item.prompt,
        response=item.response,
        judge_pack=item.judge_pack,
        student_input_packet=packet,
        raw_student_output=json.dumps(raw, separators=(",", ":")).encode(),
        role_session_lineage=item.role_session_lineage,
    )

    run_with_fresh_session_after_every_event(tmp_path, changed, FixtureProfileConfig())

    result = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    assert result.reward is not None
    role = ArtifactStore(tmp_path).read(
        str(result.reward.payload["role_invocation_hash"]), expected_schema_name="RoleInvocation"
    )
    assert role.payload["seed"] == 1702


def test_scorer_normalizes_dimensions_instead_of_returning_a_constant_reward(tmp_path: Path) -> None:
    item = trace_input(run_id="fixture-run-content-derived")
    raw = json.loads(item.raw_student_output)
    raw["dimensions"] = {
        "instruction_following": 2,
        "recovery_safety": 1,
        "evidence_specificity": 0,
    }
    raw["overall_scalar"] = 3
    raw["confidence"] = "0.7500"
    lower = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id=item.dataset_version_id,
        experiment_spec_id=item.experiment_spec_id,
        trace_id=item.trace_id,
        trajectory_id=item.trajectory_id,
        prompt=item.prompt,
        response=item.response,
        judge_pack=item.judge_pack,
        student_input_packet=item.student_input_packet,
        raw_student_output=canonical_json_bytes(raw),
        role_session_lineage=item.role_session_lineage,
    )

    run_with_fresh_session_after_every_event(tmp_path, lower, FixtureProfileConfig())

    result = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    assert result.reward is not None
    assert result.reward.payload["reward_basis_points"] == 3_000
    assert result.reward.payload["confidence_basis_points"] == 7_500


@pytest.mark.parametrize(
    "malformation",
    [
        "nan_confidence",
        "huge_positive_confidence",
        "huge_negative_confidence",
        "decimal_tie_group",
        "missing_evidence",
        "deep_nesting",
        "oversized_output",
        "escaped_string_limit",
        "escaped_top_level_key_limit",
        "escaped_nested_key_limit",
        "aggregate_string_limit",
        "duplicate_top_level",
        "duplicate_nested",
    ],
)
def test_invalid_student_output_closes_failed_and_persists_complete_error_lineage(
    tmp_path: Path,
    malformation: str,
) -> None:
    item = trace_input(run_id=f"fixture-run-invalid-output-{malformation}")
    if malformation == "deep_nesting":
        nested = b"[" * 1_500 + b'"nested"' + b"]" * 1_500
        raw_bytes = item.raw_student_output.replace(
            b'"turn_local_tie_groups":[]',
            b'"turn_local_tie_groups":' + nested,
        )
    elif malformation == "oversized_output":
        raw_bytes = item.raw_student_output + b" " * 300_000
    elif malformation in {
        "escaped_top_level_key_limit",
        "escaped_nested_key_limit",
    }:
        escaped_key = json.dumps('"' * 40_000).encode()
        if malformation == "escaped_top_level_key_limit":
            raw_bytes = b"{" + escaped_key + b":0," + item.raw_student_output[1:]
        else:
            raw_bytes = item.raw_student_output.replace(
                b'"dimensions":{',
                b'"dimensions":{' + escaped_key + b":0,",
                1,
            )
    else:
        raw = json.loads(item.raw_student_output)
        if malformation == "nan_confidence":
            raw["confidence"] = "NaN"
        elif malformation == "huge_positive_confidence":
            raw["confidence"] = "1e9999999"
        elif malformation == "huge_negative_confidence":
            raw["confidence"] = "-1e9999999"
        elif malformation == "decimal_tie_group":
            raw["turn_local_tie_groups"] = [[1.25]]
        elif malformation == "escaped_string_limit":
            raw["evidence"] = ["\\" * 40_000]
        elif malformation == "aggregate_string_limit":
            raw["evidence"] = [str(index) + "x" * 29_999 for index in range(5)]
        elif malformation == "duplicate_top_level":
            raw_bytes = item.raw_student_output.replace(
                b"{",
                b'{"packet_id":"duplicate-must-not-win",',
                1,
            )
        elif malformation == "duplicate_nested":
            raw_bytes = item.raw_student_output.replace(
                b'"dimensions":{',
                b'"dimensions":{"instruction_following":0,',
                1,
            )
        else:
            raw.pop("evidence")
        if malformation not in {"duplicate_top_level", "duplicate_nested"}:
            raw_bytes = json.dumps(raw, separators=(",", ":")).encode()
    invalid = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id=item.dataset_version_id,
        experiment_spec_id=item.experiment_spec_id,
        trace_id=item.trace_id,
        trajectory_id=item.trajectory_id,
        prompt=item.prompt,
        response=item.response,
        judge_pack=item.judge_pack,
        student_input_packet=item.student_input_packet,
        raw_student_output=raw_bytes,
        role_session_lineage=item.role_session_lineage,
    )

    run_with_fresh_session_after_every_event(tmp_path, invalid, FixtureProfileConfig())

    result = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    assert result.closed is not None
    assert result.closed.payload["status"] == "failed"
    assert result.closed.payload["reason_code"] == "STUDENT_OUTPUT_INVALID"
    failed_event = next(event for event in result.events if event.payload["event_type"] == "SCORING_FAILED")
    failed_details = failed_event.payload["details"]
    assert isinstance(failed_details, dict)
    store = ArtifactStore(tmp_path)
    observation = store.read(str(failed_details["observation_hash"]), expected_schema_name="Observation")
    role = store.read(str(observation.payload["role_invocation_hash"]), expected_schema_name="RoleInvocation")
    assert role.payload["status"] == "failed"
    assert role.payload["seed"] == 1701
    assert role.payload["session_lineage"] == "/root/ticket01_student_judge"
    assert role.payload["input_hash"]
    assert role.payload["output_hash"] == hashlib.sha256(raw_bytes).hexdigest()
    if malformation in {
        "escaped_top_level_key_limit",
        "escaped_nested_key_limit",
    }:
        normalized = store.read(
            str(role.payload["normalized_output_hash"]),
            expected_schema_name="NormalizedRoleOutput",
        )
        assert normalized.payload["failure_detail"] == ("decoded role output string exceeds the frozen limit")
    raw_manifest = store.read(str(role.payload["raw_output_manifest_hash"]), expected_schema_name="RawRoleOutput")
    size = raw_manifest.payload["byte_size"]
    assert isinstance(size, int) and not isinstance(size, bool)
    assert store.read_blob(str(raw_manifest.payload["blob_hash"]), expected_size=size) == raw_bytes
    store.read(str(role.payload["input_packet_hash"]), expected_schema_name="StudentJudgeInputPacket")
    normalized = store.read(
        str(role.payload["normalized_output_hash"]),
        expected_schema_name="NormalizedRoleOutput",
    )
    assert normalized.payload["status"] == "invalid"
    assert normalized.payload["failure_code"] == "STUDENT_OUTPUT_INVALID"
    assert normalized.payload["validation_limits"] == {
        "max_bytes": 262_144,
        "max_depth": 64,
        "max_nodes": 10_000,
        "max_string_characters": 32_768,
        "max_total_string_characters": 131_072,
    }
    assert result.run_record is not None
    assert result.run_record.payload["failure_observation_hash"] == observation.content_hash
    assert result.run_record.payload["role_invocation_hash"] == role.content_hash


def test_scalarizer_domain_mismatch_fails_instead_of_emitting_unbounded_reward(
    tmp_path: Path,
) -> None:
    item = trace_input(run_id="fixture-run-invalid-scalarizer-range")
    packet = json.loads(json.dumps(item.student_input_packet))
    packet["judge_pack"]["scalarizer"]["max"] = 5
    invalid = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id=item.dataset_version_id,
        experiment_spec_id=item.experiment_spec_id,
        trace_id=item.trace_id,
        trajectory_id=item.trajectory_id,
        prompt=item.prompt,
        response=item.response,
        judge_pack=packet["judge_pack"],
        student_input_packet=packet,
        raw_student_output=item.raw_student_output,
        role_session_lineage=item.role_session_lineage,
    )

    run_with_fresh_session_after_every_event(tmp_path, invalid, FixtureProfileConfig())

    result = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    assert result.reward is None
    assert result.closed is not None
    assert result.closed.payload["status"] == "failed"
    assert result.closed.payload["reason_code"] == "REWARD_RANGE_INVALID"


def test_concurrent_fresh_sessions_commit_each_logical_transition_once(tmp_path: Path) -> None:
    item = trace_input(run_id="fixture-run-concurrent-cas")
    config = FixtureProfileConfig()

    def initial(_: int) -> None:
        ProfiledTraceWorkflow(tmp_path, config).advance(item, epoch=1701)

    def resume(_: int) -> None:
        ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(initial, range(16)))
    assert event_types(tmp_path, item.run_id) == ["RUN_STARTED"]

    expected = [
        "RUN_STARTED",
        "SCORING_REQUESTED",
        "REWARD_COMMITTED",
        "CLUSTER_SUBMIT_REQUESTED",
        "CLUSTER_ACCEPTED",
        "RUN_RECORD_COMMITTED",
        "DECISION_OUTCOME_COMMITTED",
        "RUN_CLOSED",
    ]
    previous_count = 1
    for _ in range(len(expected)):
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(resume, range(16)))
        committed = event_types(tmp_path, item.run_id)
        assert committed == expected[: len(committed)]
        assert len(committed) > previous_count
        previous_count = len(committed)
        if committed[-1] == "RUN_CLOSED":
            break
    assert event_types(tmp_path, item.run_id) == expected
    RunJournal(tmp_path, ArtifactStore(tmp_path), item.run_id).verify()
    assert PersistentFixtureScorer(tmp_path, ArtifactStore(tmp_path), None).execution_count() == 1
    assert PersistentFixtureCluster(tmp_path, ArtifactStore(tmp_path), None).execution_count() == 1


def test_auxiliary_observations_interleave_every_lifecycle_phase_and_fresh_resume(
    tmp_path: Path,
) -> None:
    item = trace_input(run_id="fixture-run-auxiliary-lifecycle-interleave")
    snapshot = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(item, epoch=1701)
    observed_phases: list[str] = []
    while snapshot.closed is None:
        lifecycle_event = next(
            event for event in reversed(snapshot.events) if event.payload["event_type"] != "OBSERVATION_RECORDED"
        )
        phase = str(lifecycle_event.payload["event_type"])
        observed_phases.append(phase)
        receipt = RunJournal(
            tmp_path,
            ArtifactStore(tmp_path),
            item.run_id,
        ).record_observation(
            1701,
            "HEARTBEAT",
            {"lifecycle_phase": phase},
        )
        assert receipt.quarantined is False
        resumed = resume_in_fresh_process(tmp_path, item.run_id)
        snapshot = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
        assert resumed["event_type"] == snapshot.events[-1].payload["event_type"]

    assert observed_phases == [
        "RUN_STARTED",
        "SCORING_REQUESTED",
        "REWARD_COMMITTED",
        "CLUSTER_SUBMIT_REQUESTED",
        "CLUSTER_ACCEPTED",
        "RUN_RECORD_COMMITTED",
        "DECISION_OUTCOME_COMMITTED",
    ]
    RunJournal(tmp_path, ArtifactStore(tmp_path), item.run_id).verify()
    assert snapshot.closed.payload["reason_code"] == "TRACE_COMPLETE"


def test_auxiliary_observation_racing_lifecycle_cas_cannot_strand_run(
    tmp_path: Path,
) -> None:
    item = trace_input(run_id="fixture-run-auxiliary-cas-race")
    ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(item, epoch=1701)
    ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)

    with ThreadPoolExecutor(max_workers=2) as pool:
        lifecycle_future = pool.submit(
            ProfiledTraceWorkflow.resume,
            tmp_path,
            item.run_id,
            epoch=1701,
        )
        observation_future = pool.submit(
            RunJournal(tmp_path, ArtifactStore(tmp_path), item.run_id).record_observation,
            1701,
            "CONCURRENT_HEARTBEAT",
            {"status": "alive"},
        )
        lifecycle_future.result()
        observation_future.result()

    snapshot = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    while snapshot.closed is None:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    RunJournal(tmp_path, ArtifactStore(tmp_path), item.run_id).verify()
    assert snapshot.closed.payload["reason_code"] == "TRACE_COMPLETE"
    assert event_types(tmp_path, item.run_id).count("REWARD_COMMITTED") == 1
    assert PersistentFixtureScorer(tmp_path, ArtifactStore(tmp_path), None).execution_count() == 1


def test_concurrent_initial_submissions_reserve_one_identity_without_loser_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_root = tmp_path / "expected"
    race_root = tmp_path / "race"
    first = trace_input(run_id="fixture-run-concurrent-input-conflict")
    second = TraceWorkflowInput(
        run_id=first.run_id,
        dataset_version_id="different-dataset",
        experiment_spec_id="different-spec",
        trace_id="different-trace",
        trajectory_id="different-trajectory",
        prompt="Different prompt",
        response="Different response",
        judge_pack=first.judge_pack,
        student_input_packet=first.student_input_packet,
        raw_student_output=first.raw_student_output,
        role_session_lineage=first.role_session_lineage,
    )

    ProfiledTraceWorkflow(expected_root, FixtureProfileConfig()).advance(first, epoch=1)
    expected_tree = {
        path.relative_to(expected_root).as_posix(): path.read_bytes()
        for path in sorted(expected_root.rglob("*"))
        if path.is_file()
    }
    expected_started = RunJournal(
        expected_root,
        ArtifactStore(expected_root),
        first.run_id,
    ).events()[0]
    expected_details = expected_started.payload["details"]
    assert isinstance(expected_details, dict)
    winner_input_hash = str(expected_details["input_hash"])

    original_reserve = RunJournal.reserve_identity
    ready = Barrier(2)
    winner_reserved = Event()

    def ordered_reserve(journal: RunJournal, input_hash: str) -> Artifact:
        ready.wait()
        if input_hash == winner_input_hash:
            reservation = original_reserve(journal, input_hash)
            winner_reserved.set()
            return reservation
        assert winner_reserved.wait(timeout=5)
        return original_reserve(journal, input_hash)

    monkeypatch.setattr(RunJournal, "reserve_identity", ordered_reserve)

    def submit(item: TraceWorkflowInput, epoch: int) -> str:
        try:
            ProfiledTraceWorkflow(race_root, FixtureProfileConfig()).advance(
                item,
                epoch=epoch,
            )
        except WorkflowIntegrityError:
            return "conflict"
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(submit, first, 1),
            pool.submit(submit, second, 999),
        ]
        outcomes = [future.result() for future in futures]
    monkeypatch.setattr(RunJournal, "reserve_identity", original_reserve)

    assert outcomes == ["committed", "conflict"]
    assert event_types(race_root, first.run_id) == ["RUN_STARTED"]
    race_tree = {
        path.relative_to(race_root).as_posix(): path.read_bytes()
        for path in sorted(race_root.rglob("*"))
        if path.is_file()
    }
    assert race_tree == expected_tree
    fence_refs = list((race_root / "runs" / first.run_id / "fences").glob("*.ref"))
    assert [path.name for path in fence_refs] == ["00000000000000000001.ref"]

    snapshot = ProfiledTraceWorkflow(race_root, FixtureProfileConfig()).advance(
        first,
        epoch=1,
    )
    assert snapshot.events[-1].payload["event_type"] == "SCORING_REQUESTED"


def test_identity_reservation_crash_rejects_other_input_and_fresh_process_recovers(
    tmp_path: Path,
) -> None:
    item = trace_input(run_id="fixture-run-reservation-crash-recovery")
    with pytest.raises(InjectedProcessCrash, match="identity reservation"):
        ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(
            item,
            epoch=1,
            crash_after_side_effect="reservation",
        )

    run_dir = tmp_path / "runs" / item.run_id
    assert (run_dir / "identity.ref").is_file()
    assert not list((run_dir / "events").glob("*.ref"))
    assert not list((run_dir / "fences").glob("*.ref"))
    schemas = {json.loads(path.read_text())["schema_name"] for path in (tmp_path / "artifacts").glob("*.json")}
    assert {"RunIdentityReservation", "TraceRunInput"} <= schemas

    conflicting = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id="different-dataset",
        experiment_spec_id="different-spec",
        trace_id="different-trace",
        trajectory_id="different-trajectory",
        prompt="Different prompt",
        response="Different response",
        judge_pack=item.judge_pack,
        student_input_packet=item.student_input_packet,
        raw_student_output=item.raw_student_output,
        role_session_lineage=item.role_session_lineage,
    )
    tree_before_conflict = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    with pytest.raises(WorkflowIntegrityError, match="reserved run identity"):
        ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(
            conflicting,
            epoch=999,
        )
    assert {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    } == tree_before_conflict

    recovered = resume_in_fresh_process(tmp_path, item.run_id, epoch=1)
    assert recovered["event_type"] == "RUN_STARTED"
    assert [path.name for path in sorted((run_dir / "fences").glob("*.ref"))] == ["00000000000000000001.ref"]


@pytest.mark.parametrize(
    ("boundary", "fence_committed"),
    [("start_event", False), ("start_fence", True)],
)
def test_authoritative_start_crash_boundaries_recover_without_lower_epoch_takeover(
    tmp_path: Path,
    boundary: str,
    fence_committed: bool,
) -> None:
    item = trace_input(run_id=f"fixture-run-{boundary}-crash-recovery")
    with pytest.raises(InjectedProcessCrash):
        ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(
            item,
            epoch=999,
            crash_after_side_effect=boundary,
        )

    run_dir = tmp_path / "runs" / item.run_id
    events = RunJournal(tmp_path, ArtifactStore(tmp_path), item.run_id).events()
    assert [event.payload["event_type"] for event in events] == ["RUN_STARTED"]
    assert events[0].payload["controller_epoch"] == 999
    assert bool(list((run_dir / "fences").glob("*.ref"))) is fence_committed

    assert resume_attempt_in_fresh_process(
        tmp_path,
        item.run_id,
        epoch=1,
    ) == {"exception": "StaleFencingEpoch"}
    recovered = resume_in_fresh_process(tmp_path, item.run_id, epoch=999)
    assert recovered["event_type"] == "SCORING_REQUESTED"


def test_legacy_orphan_fence_is_audited_but_cannot_poison_disk_only_start(
    tmp_path: Path,
) -> None:
    item = trace_input(run_id="fixture-run-legacy-orphan-fence")
    with pytest.raises(InjectedProcessCrash):
        ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(
            item,
            epoch=999,
            crash_after_side_effect="reservation",
        )

    store = ArtifactStore(tmp_path)
    journal = RunJournal(tmp_path, store, item.run_id)
    orphan = store.put(
        "FenceClaim",
        "1.0.0",
        {"epoch": 999, "run_id": item.run_id},
    )
    orphan_ref = journal.fences_dir / "00000000000000000999.ref"
    RunJournal._publish_ref(orphan_ref, orphan.content_hash)
    preserved_bytes = orphan_ref.read_bytes()
    preserved_hash = hashlib.sha256(preserved_bytes).hexdigest()

    recovered = resume_in_fresh_process(tmp_path, item.run_id, epoch=1)
    assert recovered["event_type"] == "RUN_STARTED"
    assert orphan_ref.read_bytes() == preserved_bytes
    quarantine_ref = journal.startup_quarantine_dir / orphan_ref.name
    quarantine = store.read(
        RunJournal._read_ref(quarantine_ref),
        expected_schema_name="StartupFenceQuarantine",
    )
    assert quarantine.payload == {
        "claim_hash": orphan.content_hash,
        "epoch": 999,
        "fence_ref_hash": preserved_hash,
        "fence_ref_name": orphan_ref.name,
        "reason_code": "ORPHAN_FENCE_WITHOUT_RUN_STARTED",
        "run_id": item.run_id,
    }
    assert sorted(path.name for path in journal.fences_dir.glob("*.ref")) == [
        "00000000000000000001.ref",
        "00000000000000000999.ref",
    ]
    journal.verify()
    assert resume_in_fresh_process(tmp_path, item.run_id, epoch=1)["event_type"] == ("SCORING_REQUESTED")


def test_atomic_startup_prevents_early_observation_from_taking_event_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = trace_input(run_id="fixture-run-atomic-start-observation-race")
    entered_start_append = Event()
    release_start_append = Event()
    observation_entered = Event()
    original_append = RunJournal._append_unlocked

    def paused_append(
        journal: RunJournal,
        epoch: int,
        event_type: str,
        details: Mapping[str, object],
        events: list[Artifact],
    ) -> Artifact:
        if event_type == "RUN_STARTED":
            entered_start_append.set()
            assert release_start_append.wait(timeout=5)
        return original_append(journal, epoch, event_type, details, events)

    monkeypatch.setattr(RunJournal, "_append_unlocked", paused_append)

    def start() -> None:
        ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(
            item,
            epoch=1,
        )

    def observe() -> None:
        observation_entered.set()
        RunJournal(tmp_path, ArtifactStore(tmp_path), item.run_id).record_observation(
            1,
            "EARLY_HEARTBEAT",
            {"status": "alive"},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        start_future = pool.submit(start)
        assert entered_start_append.wait(timeout=5)
        observation_future = pool.submit(observe)
        assert observation_entered.wait(timeout=5)
        assert not observation_future.done()
        release_start_append.set()
        start_future.result()
        observation_future.result()

    assert event_types(tmp_path, item.run_id) == [
        "RUN_STARTED",
        "OBSERVATION_RECORDED",
    ]
    resumed = resume_in_fresh_process(tmp_path, item.run_id, epoch=1)
    assert resumed["event_type"] == "SCORING_REQUESTED"


def test_observation_before_any_atomic_start_is_rejected_without_artifact(
    tmp_path: Path,
) -> None:
    journal = RunJournal(
        tmp_path,
        ArtifactStore(tmp_path),
        "fixture-run-observation-before-start",
    )
    with pytest.raises(RunNotStarted, match="before RUN_STARTED"):
        journal.record_observation(1, "EARLY_HEARTBEAT", {"status": "alive"})
    assert journal.events() == []
    assert not list((tmp_path / "artifacts").glob("*.json"))


@pytest.mark.parametrize(
    "config",
    [FixtureProfileConfig(), FixtureProfileConfig(scorer_failure_code="SCORER_TIMEOUT")],
)
def test_terminal_submit_with_different_input_is_rejected_as_identity_conflict(
    tmp_path: Path,
    config: FixtureProfileConfig,
) -> None:
    item = trace_input(run_id=f"fixture-run-terminal-conflict-{config.scorer_failure_code or 'success'}")
    run_with_fresh_session_after_every_event(tmp_path, item, config)
    different = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id="different-dataset",
        experiment_spec_id="different-spec",
        trace_id="different-trace",
        trajectory_id="different-trajectory",
        prompt="Different prompt",
        response="Different response",
        judge_pack=item.judge_pack,
        student_input_packet=item.student_input_packet,
        raw_student_output=item.raw_student_output,
        role_session_lineage=item.role_session_lineage,
    )

    with pytest.raises(WorkflowIntegrityError, match="does not match"):
        ProfiledTraceWorkflow(tmp_path, config).advance(different, epoch=1701)


def test_wrong_input_cannot_poison_fence_before_identity_validation(tmp_path: Path) -> None:
    item = trace_input(run_id="fixture-run-fence-poisoning")
    workflow = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig())
    workflow.advance(item, epoch=1)
    wrong = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id="wrong-dataset",
        experiment_spec_id=item.experiment_spec_id,
        trace_id=item.trace_id,
        trajectory_id=item.trajectory_id,
        prompt=item.prompt,
        response=item.response,
        judge_pack=item.judge_pack,
        student_input_packet=item.student_input_packet,
        raw_student_output=item.raw_student_output,
        role_session_lineage=item.role_session_lineage,
    )

    with pytest.raises(WorkflowIntegrityError, match="does not match"):
        ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(wrong, epoch=999)

    assert len(list((tmp_path / "runs" / item.run_id / "fences").glob("*.ref"))) == 1
    snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1)
    assert snapshot.events[-1].payload["event_type"] == "SCORING_REQUESTED"


def test_terminal_replay_and_snapshot_reads_have_zero_publish_or_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = trace_input(run_id="fixture-run-terminal-read-only")
    run_with_fresh_session_after_every_event(tmp_path, item, FixtureProfileConfig())
    fence_dir = tmp_path / "runs" / item.run_id / "fences"
    fence_hashes_before = {path.name: path.read_bytes() for path in fence_dir.glob("*.ref")}
    tree_before = {
        path.relative_to(tmp_path).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }

    def reject_publish(_path: Path, _data: bytes) -> None:
        raise AssertionError("terminal replay attempted publication")

    def reject_fsync(_path: Path) -> None:
        raise AssertionError("terminal replay attempted directory fsync")

    monkeypatch.setattr(ArtifactStore, "_publish", staticmethod(reject_publish))
    monkeypatch.setattr(artifacts_module, "_fsync_directory", reject_fsync)

    snapshot = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(item, epoch=9_999)
    loaded = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)

    assert snapshot.closed is not None
    assert loaded.closed is not None
    assert {path.name: path.read_bytes() for path in fence_dir.glob("*.ref")} == fence_hashes_before
    assert {
        path.relative_to(tmp_path).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    } == tree_before


def test_terminal_identity_conflict_with_sentinel_has_zero_filesystem_mutation(
    tmp_path: Path,
) -> None:
    item = trace_input(run_id="fixture-run-terminal-zero-write-conflict")
    run_with_fresh_session_after_every_event(tmp_path, item, FixtureProfileConfig())
    files_before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    packet = json.loads(json.dumps(item.student_input_packet))
    sentinel = "teacher_scorer_TERMINAL_CONFLICT_SENTINEL"
    packet["role"] = sentinel
    conflicting = TraceWorkflowInput(
        run_id=item.run_id,
        dataset_version_id="different-dataset",
        experiment_spec_id=item.experiment_spec_id,
        trace_id=item.trace_id,
        trajectory_id=item.trajectory_id,
        prompt=item.prompt,
        response=item.response,
        judge_pack=item.judge_pack,
        student_input_packet=packet,
        raw_student_output=item.raw_student_output,
        role_session_lineage=item.role_session_lineage,
    )

    with pytest.raises(WorkflowIntegrityError, match="does not match"):
        ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(
            conflicting,
            epoch=99_999,
        )

    files_after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    assert files_after == files_before
    assert sentinel.encode() not in b"\n".join(files_after.values())


@pytest.mark.parametrize("forged_status", ["succeeded", "failed"])
def test_scorer_outcome_ref_substitution_closes_integrity_failure(
    tmp_path: Path,
    forged_status: str,
) -> None:
    item = trace_input(run_id=f"fixture-run-forged-scorer-{forged_status}")
    ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(item, epoch=1701)
    ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    journal = RunJournal(tmp_path, ArtifactStore(tmp_path), item.run_id)
    requested = journal.events()[-1]
    details = requested.payload["details"]
    assert isinstance(details, dict)
    key = str(details["idempotency_key"])
    wrong_request = "f" * 64
    store = ArtifactStore(tmp_path)
    payload: dict[str, object] = {
        "idempotency_key": key,
        "producer": "fixture_student_judge",
        "request_hash": wrong_request,
        "status": forged_status,
    }
    if forged_status == "succeeded":
        normalized = store.put("NormalizedRoleOutput", "1.0.0", {"status": "succeeded"})
        role = store.put(
            "RoleInvocation",
            "1.0.0",
            {
                "normalized_output_hash": normalized.content_hash,
                "request_hash": wrong_request,
                "status": "succeeded",
            },
        )
        evidence = store.put(
            "ScoringEvidence",
            "1.0.0",
            {
                "normalized_output_hash": normalized.content_hash,
                "request_hash": wrong_request,
                "role_invocation_hash": role.content_hash,
            },
        )
        reward = store.put(
            "EvidenceLinkedReward",
            "1.0.0",
            {
                "evidence_hash": evidence.content_hash,
                "normalized_output_hash": normalized.content_hash,
                "request_hash": wrong_request,
                "reward_basis_points": 10_000,
                "role_invocation_hash": role.content_hash,
            },
        )
        payload.update(
            evidence_hash=evidence.content_hash,
            reward_hash=reward.content_hash,
            role_invocation_hash=role.content_hash,
        )
    else:
        payload["failure_code"] = "FORGED_FAILURE"
    forged = store.put("Observation", "1.0.0", payload)
    outcome_ref = tmp_path / "boundaries" / "scorer" / "outcomes" / f"{key}.ref"
    ArtifactStore._publish(outcome_ref, f"{forged.content_hash}\n".encode())

    snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)

    assert snapshot.events[-1].payload["event_type"] == "SCORING_FAILED"
    while snapshot.closed is None:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    assert snapshot.closed.payload["status"] == "failed"
    assert snapshot.closed.payload["reason_code"] == "SCORER_OUTCOME_CORRUPTION"
    assert "REWARD_COMMITTED" not in event_types(tmp_path, item.run_id)


def test_scorer_outcome_with_incomplete_role_lineage_closes_integrity_failure(
    tmp_path: Path,
) -> None:
    item = trace_input(run_id="fixture-run-forged-incomplete-role-lineage")
    ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(item, epoch=1701)
    ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    store = ArtifactStore(tmp_path)
    requested = RunJournal(tmp_path, store, item.run_id).events()[-1]
    details = requested.payload["details"]
    assert isinstance(details, dict)
    key = str(details["idempotency_key"])
    request_hash = str(details["request_hash"])

    normalized = store.put(
        "NormalizedRoleOutput",
        "1.0.0",
        {
            "confidence_basis_points": 10_000,
            "dimensions": {"forged": 1},
            "evidence": ["forged evidence"],
        },
    )
    role = store.put(
        "RoleInvocation",
        "1.0.0",
        {
            "normalized_output_hash": normalized.content_hash,
            "request_hash": request_hash,
            "status": "succeeded",
        },
    )
    evidence = store.put(
        "ScoringEvidence",
        "1.0.0",
        {
            "dimension_scores": normalized.payload["dimensions"],
            "evidence": normalized.payload["evidence"],
            "normalized_output_hash": normalized.content_hash,
            "request_hash": request_hash,
            "role_invocation_hash": role.content_hash,
        },
    )
    reward = store.put(
        "EvidenceLinkedReward",
        "1.0.0",
        {
            "confidence_basis_points": 10_000,
            "evidence_hash": evidence.content_hash,
            "normalized_output_hash": normalized.content_hash,
            "request_hash": request_hash,
            "reward_basis_points": 10_000,
            "role_invocation_hash": role.content_hash,
        },
    )
    forged = store.put(
        "Observation",
        "1.0.0",
        {
            "evidence_hash": evidence.content_hash,
            "idempotency_key": key,
            "producer": "fixture_student_judge",
            "request_hash": request_hash,
            "reward_hash": reward.content_hash,
            "role_invocation_hash": role.content_hash,
            "status": "succeeded",
        },
    )
    outcome_ref = tmp_path / "boundaries" / "scorer" / "outcomes" / f"{key}.ref"
    ArtifactStore._publish(outcome_ref, f"{forged.content_hash}\n".encode())

    snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)

    assert snapshot.events[-1].payload["event_type"] == "SCORING_FAILED"
    while snapshot.closed is None:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    assert snapshot.closed.payload["status"] == "failed"
    assert snapshot.closed.payload["reason_code"] == "SCORER_OUTCOME_CORRUPTION"
    assert "REWARD_COMMITTED" not in event_types(tmp_path, item.run_id)


@pytest.mark.parametrize("forged_status", ["succeeded", "failed"])
def test_cluster_outcome_ref_substitution_closes_integrity_failure(
    tmp_path: Path,
    forged_status: str,
) -> None:
    item = trace_input(run_id=f"fixture-run-forged-cluster-{forged_status}")
    ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(item, epoch=1701)
    for _ in range(3):
        ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    store = ArtifactStore(tmp_path)
    journal = RunJournal(tmp_path, store, item.run_id)
    requested = journal.events()[-1]
    assert requested.payload["event_type"] == "CLUSTER_SUBMIT_REQUESTED"
    details = requested.payload["details"]
    assert isinstance(details, dict)
    key = str(details["idempotency_key"])
    request = store.read(str(details["request_hash"]), expected_schema_name="ClusterSubmitRequest")
    reward_hash = str(request.payload["reward_hash"])
    wrong_request = "e" * 64
    payload: dict[str, object] = {
        "idempotency_key": key,
        "producer": "fixture_cluster",
        "request_hash": wrong_request,
        "status": forged_status,
    }
    if forged_status == "succeeded":
        job = store.put(
            "FixtureClusterJob",
            "1.0.0",
            {
                "idempotency_key": key,
                "job_id": "forged-job",
                "request_hash": wrong_request,
                "reward_hash": reward_hash,
                "status": "accepted",
            },
        )
        payload["job_hash"] = job.content_hash
    else:
        payload["failure_code"] = "FORGED_FAILURE"
    forged = store.put("Observation", "1.0.0", payload)
    outcome_ref = tmp_path / "boundaries" / "cluster" / "outcomes" / f"{key}.ref"
    ArtifactStore._publish(outcome_ref, f"{forged.content_hash}\n".encode())

    snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)

    assert snapshot.events[-1].payload["event_type"] == "CLUSTER_FAILED"
    while snapshot.closed is None:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    assert snapshot.closed.payload["status"] == "failed"
    assert snapshot.closed.payload["reason_code"] == "CLUSTER_OUTCOME_CORRUPTION"


@pytest.mark.parametrize(
    "tampering",
    [
        "missing_job_id",
        "wrong_job_id",
        "extra_job_field",
        "minor_job_schema",
        "wrong_job_status",
        "wrong_job_reward",
        "extra_observation_field",
    ],
)
def test_cluster_outcome_closed_world_field_mismatch_matrix_fails_integrity(
    tmp_path: Path,
    tampering: str,
) -> None:
    item = trace_input(run_id=f"fixture-run-forged-cluster-fields-{tampering}")
    ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(item, epoch=1701)
    for _ in range(3):
        ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    store = ArtifactStore(tmp_path)
    requested = RunJournal(tmp_path, store, item.run_id).events()[-1]
    details = requested.payload["details"]
    assert isinstance(details, dict)
    key = str(details["idempotency_key"])
    request_hash = str(details["request_hash"])
    request = store.read(request_hash, expected_schema_name="ClusterSubmitRequest")
    reward_hash = str(request.payload["reward_hash"])
    job_schema_version = "1.0.0"
    job_payload: dict[str, object] = {
        "idempotency_key": key,
        "job_id": f"fixture-job-{request_hash[:16]}",
        "request_hash": request_hash,
        "reward_hash": reward_hash,
        "status": "accepted",
    }
    if tampering == "missing_job_id":
        job_payload.pop("job_id")
    elif tampering == "wrong_job_id":
        job_payload["job_id"] = "fixture-job-forged"
    elif tampering == "extra_job_field":
        job_payload["scheduler_secret"] = "must-not-be-accepted"
    elif tampering == "minor_job_schema":
        job_schema_version = "1.1.0"
    elif tampering == "wrong_job_status":
        job_payload["status"] = "succeeded"
    elif tampering == "wrong_job_reward":
        job_payload["reward_hash"] = "f" * 64
    job = store.put("FixtureClusterJob", job_schema_version, job_payload)
    observation_payload: dict[str, object] = {
        "idempotency_key": key,
        "job_hash": job.content_hash,
        "producer": "fixture_cluster",
        "request_hash": request_hash,
        "status": "succeeded",
    }
    if tampering == "extra_observation_field":
        observation_payload["untrusted_scheduler_state"] = "accepted"
    forged = store.put("Observation", "1.0.0", observation_payload)
    outcome_ref = tmp_path / "boundaries" / "cluster" / "outcomes" / f"{key}.ref"
    ArtifactStore._publish(outcome_ref, f"{forged.content_hash}\n".encode())

    snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)

    assert snapshot.events[-1].payload["event_type"] == "CLUSTER_FAILED"
    while snapshot.closed is None:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    assert snapshot.closed.payload["status"] == "failed"
    assert snapshot.closed.payload["reason_code"] == "CLUSTER_OUTCOME_CORRUPTION"


@pytest.mark.parametrize("boundary", ["scorer", "cluster"])
@pytest.mark.parametrize(
    "malformation",
    ["non_ascii", "wrong_length", "wrong_newline", "non_hex", "missing_artifact"],
)
def test_malformed_boundary_outcome_ref_is_audited_and_fresh_process_recoverable(
    tmp_path: Path,
    boundary: str,
    malformation: str,
) -> None:
    item = trace_input(run_id=f"fixture-run-{boundary}-bad-ref-{malformation}")
    ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).advance(item, epoch=1701)
    resume_count = 1 if boundary == "scorer" else 3
    for _ in range(resume_count):
        ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    requested = RunJournal(tmp_path, ArtifactStore(tmp_path), item.run_id).events()[-1]
    details = requested.payload["details"]
    assert isinstance(details, dict)
    key = str(details["idempotency_key"])
    if malformation == "non_ascii":
        ref_bytes = b"\xff" + b"a" * 63 + b"\n"
    elif malformation == "wrong_length":
        ref_bytes = b"a" * 63 + b"\n"
    elif malformation == "wrong_newline":
        ref_bytes = b"a" * 64 + b"!"
    elif malformation == "non_hex":
        ref_bytes = b"g" * 64 + b"\n"
    else:
        ref_bytes = b"a" * 64 + b"\n"
        assert not (tmp_path / "artifacts" / f"{'a' * 64}.json").exists()
    outcome_ref = tmp_path / "boundaries" / boundary / "outcomes" / f"{key}.ref"
    ArtifactStore._publish(outcome_ref, ref_bytes)

    first_retry = resume_in_fresh_process(tmp_path, item.run_id)

    expected_event = "SCORING_FAILED" if boundary == "scorer" else "CLUSTER_FAILED"
    expected_reason = "SCORER_OUTCOME_CORRUPTION" if boundary == "scorer" else "CLUSTER_OUTCOME_CORRUPTION"
    assert first_retry["event_type"] == expected_event
    second_retry = resume_in_fresh_process(tmp_path, item.run_id)
    assert second_retry["event_type"] == "RUN_RECORD_COMMITTED"
    snapshot = ProfiledTraceWorkflow(tmp_path, FixtureProfileConfig()).load_result(item.run_id)
    while snapshot.closed is None:
        snapshot = ProfiledTraceWorkflow.resume(tmp_path, item.run_id, epoch=1701)
    assert snapshot.closed.payload["status"] == "failed"
    assert snapshot.closed.payload["reason_code"] == expected_reason
