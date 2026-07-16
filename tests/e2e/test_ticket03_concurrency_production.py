"""Concurrent controller and DATA_INGEST production-readiness contracts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from clawrl.adapters.sources.fixture import FixtureDataSource
from clawrl.artifacts import ArtifactStore
from clawrl.data.models import (
    FixtureDataIngestConfig,
    ProductionDataIngestConfig,
    QueryWindow,
)
from clawrl.data.sql import SqlPolicyError
from clawrl.data.workflow import (
    DataIngestSnapshot,
    DataIngestWorkflowError,
    GovernedDataIngestWorkflow,
)
from clawrl.training.run_journal import RunJournal, StaleFencingEpoch
from tests.contracts.test_ticket03_sql_and_dataset_count import APPROVED_SQL
from tests.e2e.test_ticket03_successful_dataset_workflow import complete_with_fresh_process_per_stage
from tests.fixtures.ticket03_data import stage_data_provider


def config(
    run_id: str,
    *,
    sql: str = APPROVED_SQL,
    window_start: str = "2026-01-01T00:00:00Z",
) -> FixtureDataIngestConfig:
    return FixtureDataIngestConfig(
        run_id=run_id,
        query_sql=sql,
        window=QueryWindow(window_start, "2026-01-02T00:00:00Z"),
    )


def bootstrap(root: Path, item: FixtureDataIngestConfig, *, epoch: int = 3001) -> DataIngestSnapshot:
    return GovernedDataIngestWorkflow.bootstrap(
        root,
        item,
        role_ingress=stage_data_provider(root),
        epoch=epoch,
    )


def test_sixteen_way_sessions_commit_one_transition_and_one_query_attempt_per_stage(tmp_path: Path) -> None:
    item = config("concurrent-sixteen")
    snapshot = bootstrap(tmp_path, item)
    while not snapshot.terminal:
        before = len(snapshot.events)

        def advance(_: int) -> DataIngestSnapshot:
            return GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3001)

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(advance, range(16)))
        snapshot = max(results, key=lambda result: len(result.events))
        assert len(snapshot.events) > before
    final = GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=9999)
    assert final.dataset_version is not None and final.decision_record is not None
    assert all(
        result.dataset_version is None or result.dataset_version.content_hash == final.dataset_version.content_hash
        for result in results
    )
    assert len(list((tmp_path / "private-boundary" / "data-source").rglob("invocations/*.json"))) == 1
    event_refs = list((tmp_path / "runs" / item.run_id / "events").glob("*.ref"))
    assert len(event_refs) == 10
    artifact_hashes = {path.stem for path in (tmp_path / "artifacts").glob("*.json")}
    for ref in event_refs:
        assert ref.read_text().strip() in artifact_hashes


@pytest.mark.parametrize(
    "loser",
    [
        config("conflicting-init", sql="  " + APPROVED_SQL),
        config("conflicting-init", window_start="2026-01-01T00:00:01Z"),
    ],
)
def test_concurrent_conflicting_initial_input_loser_has_no_boundary_write_or_fence_poison(
    tmp_path: Path,
    loser: FixtureDataIngestConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    winner = config("conflicting-init")
    winner_entered_adapter = Event()
    release_winner = Event()
    original = FixtureDataSource.bootstrap.__func__  # type: ignore[attr-defined]

    def paused_bootstrap(
        cls: type[FixtureDataSource],
        root: object,
        exchange: object,
        item: object,
    ) -> FixtureDataSource:
        if item == winner:
            winner_entered_adapter.set()
            assert release_winner.wait(timeout=5)
        return original(cls, root, exchange, item)

    monkeypatch.setattr(FixtureDataSource, "bootstrap", classmethod(paused_bootstrap))
    with ThreadPoolExecutor(max_workers=2) as pool:
        winning = pool.submit(bootstrap, tmp_path, winner, epoch=1)
        assert winner_entered_adapter.wait(timeout=5)
        losing = pool.submit(bootstrap, tmp_path, loser, epoch=9999)
        with pytest.raises(DataIngestWorkflowError, match="different immutable ingest input"):
            losing.result(timeout=5)
        release_winner.set()
        winning.result(timeout=5)
    private = tmp_path / "private-boundary" / "data-source" / "fixture-online-traces-v1"
    persisted = json.loads((private / "fixture-config.canonical.json").read_bytes())
    assert persisted == winner.artifact_payload()
    assert len(list(private.glob("role-output.raw.json"))) == 1
    assert not (tmp_path / "runs" / winner.run_id / "fences" / "00000000000000009999.ref").exists()
    assert (
        GovernedDataIngestWorkflow.resume(tmp_path, winner.run_id, epoch=1).events[-1].payload["event_type"]
        == "QUERY_PLANNED"
    )


def test_auxiliary_and_post_terminal_late_observations_do_not_strand_lifecycle(tmp_path: Path) -> None:
    item = config("aux-observations")
    bootstrap(tmp_path, item, epoch=3)
    store = ArtifactStore(tmp_path)
    journal = RunJournal(
        tmp_path,
        store,
        item.run_id,
        input_schema_name="DataIngestWorkflowInput",
        input_schema_version="1.0.0",
    )
    receipt = journal.record_observation(3, "HEARTBEAT", {"sequence": 1})
    assert not receipt.quarantined
    snapshot = GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3)
    assert snapshot.events[-1].payload["event_type"] == "QUERY_PLANNED"
    while not snapshot.terminal:
        snapshot = GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=3)
    late = journal.record_observation(3, "LATE_SOURCE_RESULT", {"result_hash": "0" * 64})
    assert late.quarantined
    replay = GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=999)
    assert replay.terminal and replay.dataset_version is not None


def test_stale_controller_epoch_is_rejected_without_advancing(tmp_path: Path) -> None:
    item = config("stale-epoch")
    bootstrap(tmp_path, item, epoch=1)
    first = GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=2)
    with pytest.raises(StaleFencingEpoch):
        GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=1)
    store = ArtifactStore(tmp_path)
    journal = RunJournal(
        tmp_path,
        store,
        item.run_id,
        input_schema_name="DataIngestWorkflowInput",
        input_schema_version="1.0.0",
    )
    assert len(journal.events()) == len(first.events)


def test_terminal_replay_is_read_only_before_adapter_or_publish_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = config("terminal-read-only")
    complete_with_fresh_process_per_stage(tmp_path, item)
    ingress = stage_data_provider(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("terminal replay reached a mutation or private query path")

    monkeypatch.setattr(ArtifactStore, "_publish", forbidden)
    monkeypatch.setattr(ArtifactStore, "_publish", staticmethod(forbidden))
    monkeypatch.setattr(FixtureDataSource, "open", classmethod(forbidden))
    monkeypatch.setattr(FixtureDataSource, "query", forbidden)
    replay = GovernedDataIngestWorkflow.resume(tmp_path, item.run_id, epoch=9999)
    assert replay.terminal and replay.dataset_version is not None
    run_once = GovernedDataIngestWorkflow.run_once(
        tmp_path,
        item,
        role_ingress=ingress,
        epoch=9999,
    )
    assert run_once.dataset_version is not None
    assert run_once.dataset_version.content_hash == replay.dataset_version.content_hash


@pytest.mark.parametrize(
    "mutation",
    [
        {},
        {"schema_artifact_hash": "bad"},
        {"credential_ref": "credential with spaces"},
        {"governance_approval_hash": "f" * 64, "approved_table": object()},
        {"approved_columns_hash": True},
        {"sanitizer_approval_hash": "a" * 64, "execution_profile": "fixture"},
        {"credential_ref": "bad\ud800ref"},
    ],
)
def test_production_readiness_is_sanitized_blocked_before_any_adapter_or_run_side_effect(
    tmp_path: Path,
    mutation: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values: dict[str, object] = {
        "approved_columns_hash": None,
        "approved_table": None,
        "credential_ref": None,
        "execution_profile": "production",
        "governance_approval_hash": None,
        "run_id": "production-readiness",
        "sanitizer_approval_hash": None,
        "schema_artifact_hash": None,
    }
    values.update(mutation)
    production = ProductionDataIngestConfig(**values)  # type: ignore[arg-type]

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("production readiness constructed a fixture adapter")

    monkeypatch.setattr(FixtureDataSource, "bootstrap", classmethod(forbidden))
    report = GovernedDataIngestWorkflow.production_readiness(tmp_path, production)
    assert report.schema_name == "ReadinessReport"
    assert report.payload["phase"] == "DATA_INGEST"
    assert report.payload["status"] == "blocked"
    assert report.payload["side_effects_permitted"] is False
    assert "\ud800" not in report.raw_bytes.decode("utf-8")
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "private-boundary").exists()
    assert "DatasetVersion" not in [
        json.loads(path.read_bytes())["schema_name"] for path in (tmp_path / "artifacts").glob("*.json")
    ]


def test_sql_rejection_precedes_adapter_construction_query_and_all_boundary_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"bootstrap": 0, "query": 0}

    def constructed(*args: object, **kwargs: object) -> object:
        calls["bootstrap"] += 1
        raise AssertionError

    def queried(*args: object, **kwargs: object) -> object:
        calls["query"] += 1
        raise AssertionError

    monkeypatch.setattr(FixtureDataSource, "bootstrap", classmethod(constructed))
    monkeypatch.setattr(FixtureDataSource, "query", queried)
    item = config("sql-preflight", sql="SELECT * FROM fixture.online_trace")
    ingress = stage_data_provider(tmp_path)
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(SqlPolicyError):
        GovernedDataIngestWorkflow.bootstrap(
            tmp_path,
            item,
            role_ingress=ingress,
            epoch=3001,
        )
    assert calls == {"bootstrap": 0, "query": 0}
    assert not (tmp_path / "runs").exists()
    after = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_fixture_mapping_rejects_every_credential_like_field() -> None:
    for field in ("credential", "credential_ref", "password", "database_uri", "production_schema"):
        with pytest.raises(Exception, match="production field"):
            FixtureDataIngestConfig.from_mapping(
                {
                    field: "secret",
                    "query_sql": APPROVED_SQL,
                    "run_id": "fixture-no-credentials",
                    "window": {
                        "end_utc": "2026-01-02T00:00:00Z",
                        "start_utc": "2026-01-01T00:00:00Z",
                    },
                }
            )
