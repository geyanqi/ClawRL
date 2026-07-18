"""Production-shaped synthetic total calibrated JudgeBundle acceptance tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from clawrl.adapters.scorers.judge_bundle_fixture import FixtureJudgeTerminalSource, FixtureTerminalVariant
from clawrl.artifacts import Artifact, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.data.validation import load_training_dataset
from clawrl.judge.certification_models import RewardSchema, Scalarizer
from clawrl.judge.judge_bundle_models import FixtureExperimentSpecConfig, FixtureJudgeBundleConfig
from clawrl.judge.judge_bundle_workflow import JudgeBundleWorkflow, JudgeBundleWorkflowError
from clawrl.training.run_journal import RunJournal
from tests.e2e.test_ticket03_successful_dataset_workflow import (
    complete_with_fresh_process_per_stage,
    public_file_snapshot,
)
from tests.e2e.test_ticket03_successful_dataset_workflow import (
    config as dataset_config,
)


def _setup(root: Path, run_id: str) -> tuple[FixtureJudgeBundleConfig, FixtureJudgeTerminalSource]:
    dataset_hash = complete_with_fresh_process_per_stage(root, dataset_config(f"{run_id}-dataset"))
    store = ArtifactStore(root)
    reward = store.put("RewardSchema", "1.0.0", RewardSchema().artifact_payload())
    scalarizer = store.put("Scalarizer", "1.0.0", Scalarizer().artifact_payload())
    algorithm = store.put(
        "RLAlgorithmContract",
        "1.0.0",
        {
            "advantage_estimator": "grpo",
            "algorithm_id": "fixture-ticket08-grpo-v1",
            "reward_aggregation": "calibrated_scalar",
            "schema_version": "rl-algorithm-contract/1.0.0",
        },
    )
    config = FixtureJudgeBundleConfig(
        run_id=run_id,
        dataset_version_hash=dataset_hash,
        reward_schema_hash=reward.content_hash,
        scalarizer_hash=scalarizer.content_hash,
        algorithm_contract_hash=algorithm.content_hash,
    )
    source = FixtureJudgeTerminalSource(
        root,
        dataset_version_hash=dataset_hash,
        reward_schema_hash=reward.content_hash,
        scalarizer_hash=scalarizer.content_hash,
        algorithm_contract_hash=algorithm.content_hash,
    )
    return config, source


def _trace_ids(source: FixtureJudgeTerminalSource) -> list[str]:
    return [cast(str, trace.payload["trace_id"]) for trace in source.dataset.training_traces]


def _fresh_resume(root: Path, run_id: str, epoch: int) -> dict[str, object]:
    code = """
import json,sys
from clawrl.judge.judge_bundle_workflow import JudgeBundleWorkflow
s=JudgeBundleWorkflow.resume(sys.argv[1],sys.argv[2],epoch=int(sys.argv[3]))
print(json.dumps({
  'bundle_hash':None if s.judge_bundle is None else s.judge_bundle.content_hash,
  'event_count':len(s.events),
  'missing':list(s.missing_trace_ids),
  'terminal':s.terminal,
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
    return cast(dict[str, object], json.loads(completed.stdout))


def _submit_all(
    root: Path,
    run_id: str,
    source: FixtureJudgeTerminalSource,
    *,
    epoch: int,
    uncertifiable_trace_id: str | None = None,
) -> None:
    for trace_id in reversed(_trace_ids(source)):
        variant: FixtureTerminalVariant = "group_relative_all_low" if trace_id == uncertifiable_trace_id else "valid"
        outcome = source.materialize(trace_id, variant=variant)
        JudgeBundleWorkflow.submit_outcome(root, run_id, outcome_hash=outcome.content_hash, epoch=epoch)


def test_exact_100_by_32_bundle_restart_total_hash_and_experiment_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, source = _setup(tmp_path, "ticket08-total")
    trace_ids = _trace_ids(source)
    initial = JudgeBundleWorkflow.bootstrap(tmp_path, config, epoch=8101)
    assert len(initial.missing_trace_ids) == 100

    committed: dict[str, Artifact] = {}
    for trace_id in trace_ids[:37]:
        outcome = source.materialize(trace_id)
        snapshot = JudgeBundleWorkflow.submit_outcome(
            tmp_path, config.run_id, outcome_hash=outcome.content_hash, epoch=8101
        )
        committed[trace_id] = snapshot.committed_pack_by_trace[trace_id]
    before_duplicate = public_file_snapshot(tmp_path)
    duplicate = source.materialize(trace_ids[0])
    replay = JudgeBundleWorkflow.submit_outcome(
        tmp_path, config.run_id, outcome_hash=duplicate.content_hash, epoch=8101
    )
    assert public_file_snapshot(tmp_path) == before_duplicate
    assert replay.committed_pack_by_trace[trace_ids[0]].content_hash == committed[trace_ids[0]].content_hash

    before_restart = public_file_snapshot(tmp_path)
    restarted = _fresh_resume(tmp_path, config.run_id, 8102)
    assert len(cast(list[str], restarted["missing"])) == 63
    assert restarted["terminal"] is False
    assert public_file_snapshot(tmp_path) == before_restart
    for trace_id in trace_ids[37:]:
        outcome = source.materialize(trace_id)
        JudgeBundleWorkflow.submit_outcome(tmp_path, config.run_id, outcome_hash=outcome.content_hash, epoch=8102)

    original_append = RunJournal.append

    def fail_bundle_append(
        self: RunJournal,
        epoch: int,
        event_type: str,
        details: Mapping[str, object],
        **kwargs: Any,
    ) -> Artifact:
        if event_type == "JUDGE_BUNDLE_COMMITTED":
            raise RuntimeError("fixture crash after durable coverage event")
        return original_append(self, epoch, event_type, details, **kwargs)

    monkeypatch.setattr(RunJournal, "append", fail_bundle_append)
    with pytest.raises(RuntimeError, match="after durable coverage"):
        JudgeBundleWorkflow.resume(tmp_path, config.run_id, epoch=8102)
    monkeypatch.setattr(RunJournal, "append", original_append)

    original_close = RunJournal.close

    def fail_close(self: RunJournal, epoch: int, **kwargs: Any) -> Artifact:
        raise RuntimeError("fixture crash after durable bundle event")

    monkeypatch.setattr(RunJournal, "close", fail_close)
    with pytest.raises(RuntimeError, match="after durable bundle"):
        JudgeBundleWorkflow.resume(tmp_path, config.run_id, epoch=8102)
    monkeypatch.setattr(RunJournal, "close", original_close)

    terminal = JudgeBundleWorkflow.resume(tmp_path, config.run_id, epoch=8102)
    assert terminal.terminal and terminal.judge_bundle is not None and terminal.coverage_manifest is not None
    assert not terminal.missing_trace_ids and len(terminal.committed_pack_by_trace) == 100
    store = ArtifactStore(tmp_path)
    bundle = terminal.judge_bundle
    manifest = terminal.coverage_manifest
    refs = cast(list[dict[str, object]], bundle.payload["judge_pack_refs"])
    assert len(refs) == 100
    assert {cast(str, ref["trace_id"]) for ref in refs} == set(trace_ids)
    assert bundle.payload["coverage_manifest_hash"] == manifest.content_hash
    assert manifest.payload["judge_pack_refs"] == refs
    assert manifest.payload["coverage_count"] == 100
    assert manifest.payload["golden_terminal_count"] == 10
    assert manifest.payload["reward_schema_hash"] == config.reward_schema_hash
    assert manifest.payload["scalarizer_hash"] == config.scalarizer_hash
    assert manifest.payload["algorithm_contract_hash"] == config.algorithm_contract_hash

    tiers: dict[str, int] = {"luna": 0, "sol": 0}
    all_fit_manifest_hashes: set[str] = set()
    for ref in refs:
        pack = store.read(cast(str, ref["judge_pack_hash"]), expected_schema_name="JudgePack")
        tier = cast(str, pack.payload["scorer_tier"])
        tiers[tier] += 1
        fit = store.read(cast(str, pack.payload["fit_trajectory_set_hash"]), expected_schema_name="FitTrajectorySet")
        teacher = store.read(cast(str, pack.payload["teacher_label_set_hash"]), expected_schema_name="TeacherLabelSet")
        fit_refs = cast(list[dict[str, object]], fit.payload["trajectory_refs"])
        labels = cast(list[dict[str, object]], teacher.payload["labels"])
        assert len(fit_refs) == len(labels) == 32
        hashes = [cast(str, item["manifest_hash"]) for item in fit_refs]
        assert len(set(hashes)) == 32
        assert [item["trajectory_manifest_hash"] for item in labels] == hashes
        all_fit_manifest_hashes.update(hashes)
    assert tiers == {"luna": 90, "sol": 10}
    assert len(all_fit_manifest_hashes) == 3_200

    expected_total = sha256_hex(
        canonical_json_bytes(
            {
                "aggregation": "calibrated_scalar",
                "algorithm_contract_hash": config.algorithm_contract_hash,
                "dataset_version_hash": config.dataset_version_hash,
                "domain": "judge-bundle-total/1.0.0",
                "golden_hard_entry_hashes": manifest.payload["golden_hard_entry_hashes"],
                "pack_refs": refs,
                "reward_schema_hash": config.reward_schema_hash,
                "scalarizer_hash": config.scalarizer_hash,
            }
        )
    )
    assert bundle.payload["total_hash"] == manifest.payload["total_hash"] == expected_total

    dataset = load_training_dataset(store, config.dataset_version_hash).dataset_version
    prohibited = {"judge_bundle", "judge_pack", "teacher_label", "holdout"}
    assert prohibited.isdisjoint(dataset.payload)
    experiment = JudgeBundleWorkflow.build_experiment_spec(
        tmp_path,
        FixtureExperimentSpecConfig(
            experiment_id="ticket08-experiment",
            dataset_version_id=cast(str, dataset.payload["dataset_version_id"]),
            dataset_version_hash=config.dataset_version_hash,
            judge_bundle_hash=bundle.content_hash,
            trace_set_hash=cast(str, dataset.payload["trace_set_hash"]),
        ),
    )
    assert experiment.payload["judge_bundle_hash"] == bundle.content_hash
    incomplete_payload = dict(bundle.payload)
    incomplete_payload["judge_pack_refs"] = cast(list[JsonValue], refs[:-1])
    duplicate_key_payload = dict(bundle.payload)
    duplicate_key_payload["judge_pack_refs"] = cast(list[JsonValue], [*refs[:-1], refs[0]])
    non_total_payload = dict(bundle.payload)
    non_total_payload["status"] = "partial"
    wrong_count_payload = dict(bundle.payload)
    wrong_count_payload["trace_count"] = 99
    forged_manifest_payload = dict(manifest.payload)
    forged_manifest_payload["judge_pack_refs"] = cast(list[JsonValue], refs[:-1])
    forged_manifest = store.put("JudgeBundleCoverageManifest", "1.0.0", forged_manifest_payload)
    manifest_mismatch_payload = dict(bundle.payload)
    manifest_mismatch_payload["coverage_manifest_hash"] = forged_manifest.content_hash
    forged_bundles = [
        store.put("JudgeBundle", "1.0.0", payload)
        for payload in (
            incomplete_payload,
            duplicate_key_payload,
            non_total_payload,
            wrong_count_payload,
            manifest_mismatch_payload,
        )
    ]
    forged_count = len(tuple(store.artifact_dir.glob("*.json")))
    for index, forged_bundle in enumerate(forged_bundles):
        with pytest.raises(JudgeBundleWorkflowError):
            JudgeBundleWorkflow.build_experiment_spec(
                tmp_path,
                FixtureExperimentSpecConfig(
                    experiment_id=f"ticket08-crafted-{index}",
                    dataset_version_id=cast(str, dataset.payload["dataset_version_id"]),
                    dataset_version_hash=config.dataset_version_hash,
                    judge_bundle_hash=forged_bundle.content_hash,
                    trace_set_hash=cast(str, dataset.payload["trace_set_hash"]),
                ),
            )
    assert len(tuple(store.artifact_dir.glob("*.json"))) == forged_count
    artifact_count = len(tuple(store.artifact_dir.glob("*.json")))
    mismatches = (
        FixtureExperimentSpecConfig(
            experiment_id="ticket08-id-mismatch",
            dataset_version_id="wrong-dataset-id",
            dataset_version_hash=config.dataset_version_hash,
            judge_bundle_hash=bundle.content_hash,
            trace_set_hash=cast(str, dataset.payload["trace_set_hash"]),
        ),
        FixtureExperimentSpecConfig(
            experiment_id="ticket08-hash-mismatch",
            dataset_version_id=cast(str, dataset.payload["dataset_version_id"]),
            dataset_version_hash="f" * 64,
            judge_bundle_hash=bundle.content_hash,
            trace_set_hash=cast(str, dataset.payload["trace_set_hash"]),
        ),
        FixtureExperimentSpecConfig(
            experiment_id="ticket08-keyset-mismatch",
            dataset_version_id=cast(str, dataset.payload["dataset_version_id"]),
            dataset_version_hash=config.dataset_version_hash,
            judge_bundle_hash=bundle.content_hash,
            trace_set_hash="e" * 64,
        ),
    )
    for mismatch in mismatches:
        with pytest.raises(JudgeBundleWorkflowError):
            JudgeBundleWorkflow.build_experiment_spec(tmp_path, mismatch)
    assert len(tuple(store.artifact_dir.glob("*.json"))) == artifact_count
    before_terminal_restart = public_file_snapshot(tmp_path)
    fresh_terminal = _fresh_resume(tmp_path, config.run_id, 8103)
    assert fresh_terminal["bundle_hash"] == bundle.content_hash and fresh_terminal["terminal"] is True
    assert public_file_snapshot(tmp_path) == before_terminal_restart


def test_uncertifiable_all_low_exact_keyset_blocks_publication(tmp_path: Path) -> None:
    config, source = _setup(tmp_path, "ticket08-uncertifiable")
    trace_ids = _trace_ids(source)
    JudgeBundleWorkflow.bootstrap(tmp_path, config, epoch=8201)
    _submit_all(
        tmp_path,
        config.run_id,
        source,
        epoch=8201,
        uncertifiable_trace_id=trace_ids[0],
    )
    terminal = JudgeBundleWorkflow.resume(tmp_path, config.run_id, epoch=8201)
    assert terminal.terminal
    assert terminal.judge_bundle is None and terminal.coverage_manifest is None
    assert terminal.uncertifiable_trace_ids == (trace_ids[0],)
    assert not terminal.missing_trace_ids
    assert [event.payload["event_type"] for event in terminal.events][-2:] == [
        "BUNDLE_TOTALITY_BLOCKED",
        "RUN_CLOSED",
    ]
    before = public_file_snapshot(tmp_path)
    replay = _fresh_resume(tmp_path, config.run_id, 8202)
    assert replay["terminal"] is True and replay["bundle_hash"] is None
    assert public_file_snapshot(tmp_path) == before


def test_mismatch_nonfinite_and_conflicting_pack_fail_closed(tmp_path: Path) -> None:
    config, source = _setup(tmp_path, "ticket08-invalid")
    trace_ids = _trace_ids(source)
    JudgeBundleWorkflow.bootstrap(tmp_path, config, epoch=8301)
    event_count = 1
    invalid_variants: tuple[FixtureTerminalVariant, ...] = (
        "reward_mismatch",
        "scalarizer_mismatch",
        "scalar_nonfinite",
        "scalar_incomparable",
    )
    for variant in invalid_variants:
        outcome = source.materialize(trace_ids[1], variant=variant)
        with pytest.raises(JudgeBundleWorkflowError):
            JudgeBundleWorkflow.submit_outcome(tmp_path, config.run_id, outcome_hash=outcome.content_hash, epoch=8301)
        assert len(JudgeBundleWorkflow.resume(tmp_path, config.run_id, epoch=8301).events) == event_count

    original = source.materialize(trace_ids[1])
    committed = JudgeBundleWorkflow.submit_outcome(
        tmp_path, config.run_id, outcome_hash=original.content_hash, epoch=8301
    )
    original_hash = committed.committed_pack_by_trace[trace_ids[1]].content_hash
    alternate = source.materialize(trace_ids[1], variant="valid_alternate")
    conflicted = JudgeBundleWorkflow.submit_outcome(
        tmp_path, config.run_id, outcome_hash=alternate.content_hash, epoch=8301
    )
    assert conflicted.conflict_trace_ids == (trace_ids[1],)
    assert conflicted.committed_pack_by_trace[trace_ids[1]].content_hash == original_hash
    terminal = JudgeBundleWorkflow.resume(tmp_path, config.run_id, epoch=8301)
    assert terminal.terminal and terminal.judge_bundle is None
