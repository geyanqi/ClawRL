"""Focused Ticket 31 Sol verdict and immutable terminal contracts."""

from pathlib import Path
from typing import cast

from clawrl.evaluation import (
    FinalVerdictConfig,
    FixtureSolVerdictProvider,
    PairedGenerationWorkflow,
    SolFinalVerdictWorkflow,
)
from clawrl.evaluation.paired_generation import FixturePairedGenerationProvider
from tests.contracts.test_ticket30_paired_generation import _fixture


def _setup(root: Path):
    config, rows, provider = _fixture(root, provider=FixturePairedGenerationProvider())
    paired = PairedGenerationWorkflow.run(root, config=config, provider=provider)
    manifest_hash = (root / "paired-generations" / config.campaign_id / "active.ref").read_text().strip()
    verdict_config = FinalVerdictConfig(
        config.campaign_id, config.candidate_freeze_hash, config.evaluation_dataset_hash, manifest_hash
    )
    evaluator = paired._evaluator_mapping
    assert evaluator is not None
    mapping = evaluator.unseal(evaluator._credential)  # trusted test adapter
    identities = [cast(str, row["prompt_identity_hash"]) for row in rows]
    return verdict_config, identities, mapping


def _verdicts(identities, mapping, wins: int):
    return {
        identity: ("B" if mapping[identity] == "A" else "A") if i < wins else mapping[identity]
        for i, identity in enumerate(identities)
    }


def test_sixty_wins_passes_and_fifty_nine_fails(tmp_path: Path) -> None:
    config, identities, mapping = _setup(tmp_path / "pass")
    passed = SolFinalVerdictWorkflow.run(tmp_path / "pass", config=config, verdicts=_verdicts(identities, mapping, 60))
    assert passed.status == "PASSED"
    assert passed.trained_wins == 60

    config, identities, mapping = _setup(tmp_path / "fail")
    failed = SolFinalVerdictWorkflow.run(tmp_path / "fail", config=config, verdicts=_verdicts(identities, mapping, 59))
    assert failed.status == "FAILED"
    assert failed.trained_wins == 59


def test_ties_do_not_count_and_unseal_requires_all_items(tmp_path: Path) -> None:
    config, identities, mapping = _setup(tmp_path)
    incomplete = {identity: "tie" for identity in identities[:-1]}
    collecting = SolFinalVerdictWorkflow.run(tmp_path, config=config, verdicts=incomplete)
    assert collecting.status == "INVALID"
    assert collecting.reason_code == "MISSING_VERDICT"


def test_unknown_or_malformed_verdict_is_terminal_invalid(tmp_path: Path) -> None:
    config, identities, _ = _setup(tmp_path)
    invalid = SolFinalVerdictWorkflow.commit_verdict(
        tmp_path, config=config, prompt_identity_hash=identities[0], verdict="unknown"
    )
    assert invalid.status == "INVALID"
    replay = SolFinalVerdictWorkflow.run(tmp_path, config=config)
    assert replay.status == "INVALID"


def test_stateful_sol_provider_persists_payload_lineage(tmp_path: Path) -> None:
    config, identities, mapping = _setup(tmp_path)
    provider = FixtureSolVerdictProvider(outcomes=_verdicts(identities, mapping, 60))
    result = SolFinalVerdictWorkflow.run(tmp_path, config=config, provider=provider)
    assert result.status == "PASSED"
    assert len(provider.calls) == 100
    assert result.verdicts
    first = result.verdicts[0].payload
    assert first["raw_output_hash"]
    assert first["normalized_output_hash"]
    assert first["input_artifact_hash"]
    assert first["scorer_payload_hash"]
