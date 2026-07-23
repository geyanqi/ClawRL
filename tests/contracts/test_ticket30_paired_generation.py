"""Focused Ticket 30 paired-generation and blinding contracts."""

from pathlib import Path
from typing import cast

import pytest

from clawrl.artifacts import ArtifactStore
from clawrl.evaluation import (
    EvaluationEnvironment,
    FinalEvaluationProtocolConfig,
    FinalEvaluationProtocolRegistry,
    FixturePairedGenerationProvider,
    PairedGenerationConfig,
    PairedGenerationReadinessError,
    PairedGenerationWorkflow,
)
from clawrl.evaluation.future_dataset import prompt_identity_hash
from clawrl.judge.fit_models import InitialEvalRubric


def _fixture(root: Path, *, provider=None):
    store = ArtifactStore(root)
    protocol = FinalEvaluationProtocolRegistry.preregister(
        root,
        campaign_id="ticket30",
        config=FinalEvaluationProtocolConfig(
            "future-100-v1", InitialEvalRubric.fixture_default(), EvaluationEnvironment(), 11, 12
        ),
    )
    freeze = store.put(
        "CandidateFreeze",
        "1.0.0",
        {
            "campaign_id": "ticket30",
            "protocol_hash": protocol.protocol.content_hash,
            "evaluation_environment_hash": protocol.environment.content_hash,
            "base_checkpoint_hash": "b" * 64,
            "trained_checkpoint_hash": "c" * 64,
            "status": "immutable",
            "t0_utc": "2026-02-01T00:00:00Z",
        },
    )
    ArtifactStore._publish(
        root / "candidate-freeze" / "ticket30" / "active.ref", f"{freeze.content_hash}\n".encode("ascii")
    )
    rows = [
        {
            "prompt": f"future prompt {index}",
            "prompt_identity_hash": prompt_identity_hash(f"future prompt {index}"),
            "provider_row_id": f"row-{index}",
            "event_time_utc": "2026-02-01T00:01:00Z",
            "ingestion_time_utc": "2026-02-01T00:01:01Z",
        }
        for index in range(100)
    ]
    dataset = store.put(
        "EvaluationDataset",
        "1.0.0",
        {
            "campaign_id": "ticket30",
            "candidate_freeze_hash": freeze.content_hash,
            "protocol_hash": protocol.protocol.content_hash,
            "identity_normalizer_hash": cast(dict[str, object], protocol.protocol.payload["identity_exclusion"])[
                "normalizer_hash"
            ],
            "source_contract_hash": "a" * 64,
            "governance_hash": "a" * 64,
            "window_hash": "a" * 64,
            "t0_utc": "2026-02-01T00:00:00Z",
            "window": {"initial_end_utc": "2026-02-02T00:00:00Z", "extension_used": False},
            "prompt_rows": rows,
            "purpose": "eval_only",
            "status": "immutable",
        },
    )
    config = PairedGenerationConfig("ticket30", freeze.content_hash, dataset.content_hash)
    return config, rows, provider or FixturePairedGenerationProvider()


def test_exactly_200_generations_and_balanced_sealed_mapping(tmp_path: Path) -> None:
    config, rows, provider = _fixture(tmp_path)
    snapshot = PairedGenerationWorkflow.run(tmp_path, config=config, provider=provider)
    assert snapshot.status == "committed"
    assert len(snapshot.generations or ()) == 200
    mapping = snapshot.sealed_mapping
    assert mapping is not None
    assert mapping.payload["mapping_count"] == 100
    assert mapping.payload["trained_as_a_count"] == 50
    assert mapping.payload["trained_as_b_count"] == 50
    assert not hasattr(mapping, "unseal")
    payload = PairedGenerationWorkflow.scorer_payload(tmp_path, snapshot, rows[0]["prompt_identity_hash"])
    assert set(payload) == {"rubric", "a_trajectory", "b_trajectory"}
    assert "checkpoint_identity" not in cast(dict[str, object], payload["a_trajectory"])


def test_production_blocks_before_provider_call(tmp_path: Path) -> None:
    config, _, provider = _fixture(tmp_path)
    blocked = PairedGenerationWorkflow.run(
        tmp_path,
        config=PairedGenerationConfig(
            config.campaign_id,
            config.candidate_freeze_hash,
            config.evaluation_dataset_hash,
            execution_profile="production",
        ),
        provider=provider,
    )
    assert blocked.status == "blocked"
    assert provider.calls == []


def test_non_idempotent_provider_is_readiness_blocked(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)

    class Provider:
        idempotency_supported = False
        result_lookup_supported = False

        def generate(self, **kwargs):
            raise AssertionError("provider must not be called")

        def lookup(self, idempotency_key: str):
            return None

    with pytest.raises(PairedGenerationReadinessError):
        PairedGenerationWorkflow.run(tmp_path, config=config, provider=Provider())
