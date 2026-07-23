"""Focused Ticket 29 Future-100 collection contracts."""

from pathlib import Path

from clawrl.artifacts import ArtifactStore
from clawrl.evaluation import (
    EvaluationEnvironment,
    FinalEvaluationProtocolConfig,
    FinalEvaluationProtocolRegistry,
    Future100DatasetConfig,
    FutureDatasetConflictError,
    FutureEvaluationDatasetWorkflow,
    load_evaluation_dataset,
)
from clawrl.judge.fit_models import InitialEvalRubric


def _fixture(root: Path, rows: tuple[dict[str, object], ...], extension: tuple[dict[str, object], ...] = ()):
    protocol = FinalEvaluationProtocolRegistry.preregister(
        root,
        campaign_id="ticket29",
        config=FinalEvaluationProtocolConfig(
            "future-100-v1", InitialEvalRubric.fixture_default(), EvaluationEnvironment(), 11, 12
        ),
    )
    freeze = ArtifactStore(root).put(
        "CandidateFreeze",
        "1.0.0",
        {
            "campaign_id": "ticket29",
            "protocol_hash": protocol.protocol.content_hash,
            "status": "immutable",
            "t0_utc": "2026-02-01T00:00:00Z",
        },
    )
    ArtifactStore._publish(
        root / "candidate-freeze" / "ticket29" / "active.ref",
        f"{freeze.content_hash}\n".encode("ascii"),
    )
    config = Future100DatasetConfig(
        "ticket29", freeze.content_hash, protocol.protocol.content_hash, initial_rows=rows, extension_rows=extension
    )
    return FutureEvaluationDatasetWorkflow.run(root, config=config)


def _row(i: int, *, second: int = 1) -> dict[str, object]:
    return {
        "provider_row_id": f"row-{i}",
        "prompt": f"\r\n Future prompt {i} \r\n",
        "difficulty": "hard",
        "purpose": "eval_only",
        "event_time_utc": f"2026-02-01T00:{second:02d}:00Z",
        "ingestion_time_utc": f"2026-02-01T00:{second:02d}:01Z",
    }


def test_future_dataset_is_exactly_100_and_restart_readable(tmp_path: Path) -> None:
    rows = tuple(_row(i, second=(i % 59) + 1) for i in range(100))
    snapshot = _fixture(tmp_path, rows)
    assert snapshot.status == "committed"
    assert snapshot.dataset is not None
    assert snapshot.dataset.payload["purpose"] == "eval_only"
    assert len(snapshot.dataset.payload["prompt_rows"]) == 100
    assert (
        load_evaluation_dataset(ArtifactStore(tmp_path), snapshot.dataset.content_hash).content_hash
        == snapshot.dataset.content_hash
    )
    assert FutureEvaluationDatasetWorkflow.resume(tmp_path, "ticket29").dataset == snapshot.dataset


def test_only_one_same_length_extension_then_invalid(tmp_path: Path) -> None:
    initial = tuple(_row(i, second=1) for i in range(99))
    extension = (
        {**_row(99, second=1), "event_time_utc": "2026-02-02T00:01:00Z", "ingestion_time_utc": "2026-02-02T00:01:01Z"},
    )
    snapshot = _fixture(tmp_path, initial, extension)
    assert snapshot.status == "committed"
    invalid_root = tmp_path / "invalid"
    invalid = _fixture(invalid_root, initial)
    assert invalid.status == "invalid"
    assert invalid.reason_code == "FUTURE_EVAL_UNDER_100_AFTER_SINGLE_EXTENSION"


def test_predicate_and_t0_are_strict(tmp_path: Path) -> None:
    rows = tuple(_row(i, second=(i % 59) + 1) for i in range(99)) + (
        {**_row(99), "event_time_utc": "2026-02-01T00:00:00Z"},
    )
    snapshot = _fixture(tmp_path, rows)
    assert snapshot.status == "invalid"


def test_production_readiness_blocks_before_collection(tmp_path: Path) -> None:
    report = FutureEvaluationDatasetWorkflow.production_readiness(tmp_path)
    assert report.payload["phase"] == "FINAL_EVAL"
    assert report.payload["status"] == "blocked"
    assert report.payload["side_effects_permitted"] is False


def test_committed_ref_rejects_alternate_source_rows(tmp_path: Path) -> None:
    rows = tuple(_row(i, second=(i % 59) + 1) for i in range(100))
    _fixture(tmp_path, rows)
    protocol = FinalEvaluationProtocolRegistry.load(tmp_path, campaign_id="ticket29")
    freeze_hash = (tmp_path / "candidate-freeze" / "ticket29" / "active.ref").read_text().strip()
    config = Future100DatasetConfig("ticket29", freeze_hash, protocol.protocol.content_hash)
    alternate = tuple(_row(i + 1000, second=(i % 59) + 1) for i in range(100))
    try:
        FutureEvaluationDatasetWorkflow.run(tmp_path, config=config, source_rows=alternate)
    except FutureDatasetConflictError:
        return
    raise AssertionError("committed EvaluationDataset accepted alternate source rows")
