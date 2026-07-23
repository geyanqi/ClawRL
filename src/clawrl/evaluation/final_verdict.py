"""Ticket 31: immutable Sol verdict collection and final evaluation outcome.

The scorer boundary is intentionally narrow: a verdict is submitted for an
opaque prompt identity and contains only ``A``, ``B`` or ``tie``.  The sealed
mapping is never read while committing verdicts; it is read exactly once by
``unseal`` after all one-hundred immutable verdict artifacts are present.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from clawrl.artifacts import (
    Artifact,
    ArtifactStore,
    ImmutableArtifactConflict,
    JsonValue,
    canonical_json_bytes,
    sha256_hex,
)
from clawrl.evaluation.future_dataset import load_evaluation_dataset
from clawrl.evaluation.paired_generation import SealedMapping
from clawrl.evaluation.protocol_workflow import FinalEvaluationProtocolRegistry
from clawrl.training.run_journal import RunJournal, RunJournalError

_HASH = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


class FinalVerdictError(RuntimeError):
    """A final verdict campaign cannot safely proceed."""


class FinalVerdictInvalid(FinalVerdictError):
    """The campaign is durably terminal INVALID."""


class FinalVerdictConflictError(FinalVerdictError):
    """An immutable input or verdict was replaced with different content."""


class FinalVerdictReadinessError(FinalVerdictError):
    """Production or missing-input readiness failed closed."""


class FinalVerdictFencingError(FinalVerdictError):
    """A stale controller or journal writer attempted a transition."""


def _hash(value: object, field: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise FinalVerdictError(f"{field} must be a SHA-256 digest")
    return cast(str, value)


def _id(value: object, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(cast(str, value)) is None:
        raise FinalVerdictError(f"{field} is invalid")
    return cast(str, value)


def protocol_hash_from_freeze(store: ArtifactStore, freeze_hash: str) -> str:
    freeze = store.read(freeze_hash, expected_schema_name="CandidateFreeze")
    return _hash(freeze.payload.get("protocol_hash"), "protocol_hash")


Verdict = Literal["A", "B", "tie"]
Outcome = Literal["PASSED", "FAILED", "INVALID"]


class SolVerdictProvider(Protocol):
    """Isolated Sol/Final-Evaluator boundary used by the fixture workflow."""

    idempotency_supported: bool
    result_lookup_supported: bool

    def judge(
        self,
        *,
        idempotency_key: str,
        prompt_identity_hash: str,
        scorer_payload: Mapping[str, object],
        seed: int,
    ) -> Mapping[str, object]: ...

    def lookup(self, idempotency_key: str) -> Mapping[str, object] | None: ...


class FixtureSolVerdictProvider:
    """Deterministic, stateful, fault-injectable synthetic Sol adapter."""

    idempotency_supported = True
    result_lookup_supported = True

    def __init__(
        self,
        *,
        outcomes: Mapping[str, object] | None = None,
        faults: Mapping[str, str] | None = None,
        strict_outcomes: bool = False,
    ) -> None:
        self.outcomes = dict(outcomes or {})
        self.faults = dict(faults or {})
        self.strict_outcomes = strict_outcomes
        self.calls: list[str] = []
        self._results: dict[str, Mapping[str, object]] = {}

    def judge(
        self,
        *,
        idempotency_key: str,
        prompt_identity_hash: str,
        scorer_payload: Mapping[str, object],
        seed: int,
    ) -> Mapping[str, object]:
        self.calls.append(idempotency_key)
        if idempotency_key in self._results:
            return self._results[idempotency_key]
        fault = self.faults.get(prompt_identity_hash)
        if fault == "timeout":
            return {"status": "unknown", "raw_output": "timeout"}
        if fault == "malformed":
            return {"status": "committed", "raw_output": {"choice": "A"}}
        if fault == "unknown":
            return {"status": "unknown", "raw_output": "provider_unknown"}
        directive = self.outcomes.get(prompt_identity_hash, self.outcomes.get(idempotency_key))
        if directive is None and self.strict_outcomes:
            return {"status": "missing"}
        if directive is None:
            digest = sha256_hex(canonical_json_bytes({"identity": prompt_identity_hash, "seed": seed}))
            directive = ("A", "B", "tie")[int(digest[:8], 16) % 3]
        result: Mapping[str, object] = {
            "status": "committed",
            "raw_output": directive,
            "seed": seed,
            "scorer_payload_hash": sha256_hex(canonical_json_bytes(dict(scorer_payload))),
        }
        self._results[idempotency_key] = result
        return result

    def lookup(self, idempotency_key: str) -> Mapping[str, object] | None:
        return self._results.get(idempotency_key)


@dataclass(frozen=True, slots=True)
class FinalVerdictConfig:
    campaign_id: str
    candidate_freeze_hash: str
    evaluation_dataset_hash: str
    paired_generation_manifest_hash: str | None = None
    protocol_hash: str | None = None
    controller_epoch: int = 1
    execution_profile: Literal["fixture", "production"] = "fixture"
    # Alias used by early Ticket30 integrations.
    paired_generation_hash: str | None = None

    def __post_init__(self) -> None:
        _id(self.campaign_id, "campaign_id")
        _hash(self.candidate_freeze_hash, "candidate_freeze_hash")
        _hash(self.evaluation_dataset_hash, "evaluation_dataset_hash")
        generation = self.paired_generation_manifest_hash or self.paired_generation_hash
        _hash(generation, "paired_generation_manifest_hash")
        if (
            self.paired_generation_manifest_hash
            and self.paired_generation_hash
            and self.paired_generation_manifest_hash != self.paired_generation_hash
        ):
            raise FinalVerdictConflictError("PAIRED_GENERATION_HASH_CONFLICT")
        object.__setattr__(self, "paired_generation_manifest_hash", cast(str, generation))
        if self.protocol_hash is not None:
            _hash(self.protocol_hash, "protocol_hash")
        if type(self.controller_epoch) is not int or self.controller_epoch <= 0:
            raise FinalVerdictError("controller_epoch is invalid")
        if self.execution_profile not in {"fixture", "production"}:
            raise FinalVerdictError("execution_profile is invalid")


@dataclass(frozen=True, slots=True)
class FinalVerdictSnapshot:
    status: Literal["collecting", "PASSED", "FAILED", "INVALID", "blocked"]
    readiness: Artifact
    verdicts: tuple[Artifact, ...] = ()
    final_run: Artifact | None = None
    reason_code: str | None = None
    trained_wins: int | None = None
    unsealed: bool = False

    @property
    def outcome(self) -> str:
        return self.status

    @property
    def terminal(self) -> bool:
        return self.status in {"PASSED", "FAILED", "INVALID"}


class SolFinalVerdictWorkflow:
    """Commit one Sol judgment per pair and produce one immutable FinalEvalRun."""

    @staticmethod
    def production_readiness(root: str | Path, config: object | None = None) -> Artifact:
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": "TRUSTED_SOL_SCORER_UNAVAILABLE", "status": "blocked"},
                    {"code": "INDEPENDENT_SEALED_MAPPING_CREDENTIAL_UNAVAILABLE", "status": "blocked"},
                    {"code": "FINAL_EVAL_CONTROLLER_FENCING_UNAVAILABLE", "status": "blocked"},
                ],
                "execution_profile": "production",
                "phase": "FINAL_EVAL",
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )

    @classmethod
    def _journal(cls, root: str | Path, config: FinalVerdictConfig) -> RunJournal:
        store = ArtifactStore(root)
        input_artifact = store.put(
            "FinalVerdictCampaignInput",
            "1.0.0",
            {
                "campaign_id": config.campaign_id,
                "candidate_freeze_hash": config.candidate_freeze_hash,
                "evaluation_dataset_hash": config.evaluation_dataset_hash,
                "paired_generation_manifest_hash": config.paired_generation_manifest_hash,
                "protocol_hash": config.protocol_hash,
                "controller_epoch": config.controller_epoch,
                "phase": "PROTOCOL_FROZEN",
            },
        )
        journal = RunJournal(
            root,
            store,
            "sol_final_" + re.sub(r"[^A-Za-z0-9._-]", "_", config.campaign_id),
            input_schema_name="FinalVerdictCampaignInput",
        )
        try:
            journal.reserve_identity(input_artifact.content_hash)
            journal.start_run(
                config.controller_epoch,
                input_artifact.content_hash,
                {
                    "input_hash": input_artifact.content_hash,
                    "phase": "BLINDED_SCORING",
                    "campaign_id": config.campaign_id,
                },
            )
        except RunJournalError as error:
            raise FinalVerdictFencingError("FINAL_EVAL_FENCING_OR_RECOVERY_FAILED") from error
        return journal

    @classmethod
    def _close_journal(cls, root: str | Path, config: FinalVerdictConfig, *, status: str, reason: str) -> None:
        try:
            journal = cls._journal(root, config)
            journal.close(config.controller_epoch, status=status, reason_code=reason)
        except RunJournalError as error:
            raise FinalVerdictFencingError("FINAL_EVAL_FENCING_OR_RECOVERY_FAILED") from error

    @staticmethod
    def _payload_scope_valid(payload: Mapping[str, object]) -> bool:
        forbidden = {
            "model_identity",
            "model_role",
            "checkpoint_identity",
            "student_prompt",
            "sealed_mapping",
            "mapping_commitment",
        }

        def walk(value: object) -> bool:
            if isinstance(value, Mapping):
                if forbidden.intersection(value):
                    return False
                return all(walk(item) for item in value.values())
            if isinstance(value, list):
                return all(walk(item) for item in value)
            return True

        return set(payload) == {"rubric", "a_trajectory", "b_trajectory"} and walk(payload)

    @staticmethod
    def _default_scorer_payload(root: str | Path, config: FinalVerdictConfig, identity: str) -> dict[str, object]:
        store = ArtifactStore(root)
        manifest = store.read(
            cast(str, config.paired_generation_manifest_hash), expected_schema_name="PairedGenerationManifest"
        )
        payload_refs = manifest.payload.get("scorer_payload_hashes")
        if isinstance(payload_refs, Mapping):
            payload_hash = payload_refs.get(identity)
            if type(payload_hash) is str:
                try:
                    artifact = store.read(payload_hash, expected_schema_name="SolScorerPayload")
                    return dict(artifact.payload)
                except Exception as error:
                    raise FinalVerdictReadinessError("SOL_SCORER_PAYLOAD_UNAVAILABLE") from error
        visible: dict[str, dict[str, object]] = {}
        for generation_hash in cast(list[str], manifest.payload.get("generation_hashes", [])):
            trajectory = store.read(generation_hash, expected_schema_name="EvaluatorVisibleTrajectory")
            if trajectory.payload.get("prompt_identity_hash") != identity:
                continue
            role = trajectory.payload.get("model_role")
            if role in {"base", "trained"}:
                visible[cast(str, role)] = {
                    "prompt": trajectory.payload.get("prompt"),
                    "response": trajectory.payload.get("response"),
                    "tool_transcript": trajectory.payload.get("tool_transcript"),
                }
        if set(visible) != {"base", "trained"}:
            raise FinalVerdictReadinessError("SOL_SCORER_TRAJECTORY_LINEAGE_INVALID")
        # The fixture adapter receives only evaluator-visible content. The
        # trusted blinding service owns the A/B placement; model roles never
        # cross this boundary.
        return {
            "rubric": {"name": "Initial Eval Rubric", "version": "fixture"},
            "a_trajectory": visible["base"],
            "b_trajectory": visible["trained"],
        }

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: FinalVerdictConfig,
        verdicts: Mapping[str, object] | None = None,
        judgments: Mapping[str, object] | None = None,
        provider: SolVerdictProvider | None = None,
        scorer_payloads: Mapping[str, Mapping[str, object]] | None = None,
    ) -> FinalVerdictSnapshot:
        if config.execution_profile == "production":
            return FinalVerdictSnapshot("blocked", cls.production_readiness(root, config))
        try:
            cls._assert_inputs(root, config)
        except (FinalVerdictInvalid, FinalVerdictConflictError, FinalVerdictReadinessError) as error:
            return cls._invalidate(root, config, str(error))
        terminal = cls._terminal(root, config)
        if terminal is not None:
            return cls._snapshot_terminal(root, config, terminal)
        if verdicts is None:
            verdicts = judgments
        elif judgments is not None and verdicts != judgments:
            return cls._invalidate(root, config, "VERDICT_INPUT_CONFLICT")
        if verdicts is not None:
            # Compatibility inputs are routed through the same stateful
            # fixture provider boundary; no caller mapping directly writes a
            # terminal artifact.
            if provider is not None:
                return cls._invalidate(root, config, "VERDICT_PROVIDER_INPUT_CONFLICT")
            provider = FixtureSolVerdictProvider(outcomes=verdicts, strict_outcomes=True)
            verdicts = None
        if provider is not None:
            if not provider.idempotency_supported and not provider.result_lookup_supported:
                return cls._invalidate(root, config, "SOL_PROVIDER_IDEMPOTENCY_UNAVAILABLE")
            store = ArtifactStore(root)
            try:
                identities = cls._identities(store, config)
            except FinalVerdictReadinessError as error:
                return cls._invalidate(root, config, str(error))
            generated: dict[str, object] = {}
            for identity in identities:
                supplied_payload = (scorer_payloads or {}).get(identity)
                payload = dict(
                    supplied_payload
                    if supplied_payload is not None
                    else cls._default_scorer_payload(root, config, identity)
                )
                if not cls._payload_scope_valid(payload):
                    return cls._invalidate(root, config, "SOL_PAYLOAD_SCOPE_INVALID")
                try:
                    expected_payload = cls._default_scorer_payload(root, config, identity)
                    expected_trajectories = {
                        canonical_json_bytes(expected_payload["a_trajectory"]),
                        canonical_json_bytes(expected_payload["b_trajectory"]),
                    }
                    actual_trajectories = {
                        canonical_json_bytes(payload["a_trajectory"]),
                        canonical_json_bytes(payload["b_trajectory"]),
                    }
                except (FinalVerdictError, TypeError, ValueError):
                    return cls._invalidate(root, config, "SOL_SCORER_TRAJECTORY_LINEAGE_INVALID")
                if actual_trajectories != expected_trajectories:
                    return cls._invalidate(root, config, "SOL_SCORER_TRAJECTORY_LINEAGE_INVALID")
                key = sha256_hex(
                    canonical_json_bytes({"campaign_id": config.campaign_id, "prompt_identity_hash": identity})
                )
                try:
                    response = provider.judge(
                        idempotency_key=key,
                        prompt_identity_hash=identity,
                        scorer_payload=payload,
                        seed=int(key[:8], 16),
                    )
                    if response.get("status") in {"committed", "success"}:
                        expected_payload_hash = sha256_hex(canonical_json_bytes(payload))
                        if response.get("scorer_payload_hash") != expected_payload_hash:
                            return cls._invalidate(root, config, "SOL_PAYLOAD_HASH_INVALID")
                    if response.get("status") != "missing":
                        generated[identity] = response
                except Exception:
                    return cls._invalidate(root, config, "SOL_PROVIDER_OUTCOME_UNRESOLVED")
            verdicts = generated
        if verdicts is not None:
            for identity, verdict in verdicts.items():
                cls.commit_verdict(root, config=config, prompt_identity_hash=identity, verdict=verdict)
            # Missing items are invalid, rather than silently waiting for a
            # replacement or a second scoring pass.
            return cls.unseal(root, config=config)
        return cls._collecting_snapshot(root, config)

    @classmethod
    def resume(
        cls, root: str | Path, campaign_id: str, *, config: FinalVerdictConfig | None = None
    ) -> FinalVerdictSnapshot:
        """Reconstruct state in a fresh process using only durable artifacts."""
        _id(campaign_id, "campaign_id")
        if config is None:
            directory = Path(root) / "final-verdicts" / campaign_id
            terminal_ref = directory / "terminal.ref"
            invalid_ref = directory / "invalid.ref"
            store = ArtifactStore(root)
            if terminal_ref.exists():
                run = store.read(terminal_ref.read_text(encoding="ascii").strip(), expected_schema_name="FinalEvalRun")
                return cls._snapshot_terminal(
                    root,
                    FinalVerdictConfig(
                        campaign_id,
                        cast(str, run.payload["candidate_freeze_hash"]),
                        cast(str, run.payload["evaluation_dataset_hash"]),
                        cast(str, run.payload["paired_generation_manifest_hash"]),
                    ),
                    run,
                )
            if invalid_ref.exists():
                invalid = store.read(
                    invalid_ref.read_text(encoding="ascii").strip(), expected_schema_name="FinalEvaluationInvalidation"
                )
                return cls._snapshot_terminal(
                    root,
                    FinalVerdictConfig(
                        campaign_id,
                        cast(str, invalid.payload["candidate_freeze_hash"]),
                        cast(str, invalid.payload["evaluation_dataset_hash"]),
                        cast(str, invalid.payload["paired_generation_manifest_hash"]),
                    ),
                    invalid,
                )
            raise FinalVerdictError("campaign has no terminal final evaluation")
        if config.campaign_id != campaign_id:
            raise FinalVerdictConflictError("CAMPAIGN_ID_CONFLICT")
        return cls.run(root, config=config)

    @classmethod
    def commit_verdict(
        cls,
        root: str | Path,
        *,
        config: FinalVerdictConfig,
        prompt_identity_hash: str,
        verdict: object,
    ) -> FinalVerdictSnapshot:
        _hash(prompt_identity_hash, "prompt_identity_hash")
        if config.execution_profile == "production":
            return FinalVerdictSnapshot("blocked", cls.production_readiness(root, config))
        try:
            cls._assert_inputs(root, config)
        except (FinalVerdictInvalid, FinalVerdictConflictError, FinalVerdictReadinessError) as error:
            return cls._invalidate(root, config, str(error))
        try:
            if prompt_identity_hash not in set(cls._identities(ArtifactStore(root), config)):
                return cls._invalidate(root, config, "UNKNOWN_PROMPT_IDENTITY")
        except FinalVerdictReadinessError as error:
            return cls._invalidate(root, config, str(error))
        terminal = cls._terminal(root, config)
        if terminal is not None:
            return cls._snapshot_terminal(root, config, terminal)
        journal = cls._journal(root, config)
        normalized = cls._normalize_verdict(verdict)
        if normalized is None:
            return cls._invalidate(root, config, "MALFORMED_COMMITTED_VERDICT")
        store = ArtifactStore(root)
        seed = int(
            sha256_hex(canonical_json_bytes({"campaign_id": config.campaign_id, "identity": prompt_identity_hash}))[:8],
            16,
        )
        raw_output: JsonValue
        if isinstance(verdict, Mapping) and "raw_output" in verdict:
            candidate = verdict.get("raw_output")
        else:
            candidate = verdict
        scorer_payload_hash = None
        if isinstance(verdict, Mapping) and verdict.get("scorer_payload_hash") is not None:
            try:
                scorer_payload_hash = _hash(verdict.get("scorer_payload_hash"), "scorer_payload_hash")
            except FinalVerdictError:
                return cls._invalidate(root, config, "SOL_PAYLOAD_HASH_INVALID")
        if not isinstance(candidate, (str, int, float, bool, list, dict)) and candidate is not None:
            return cls._invalidate(root, config, "MALFORMED_RAW_VERDICT")
        raw_output = cast(JsonValue, candidate)
        input_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "campaign_id": config.campaign_id,
                    "candidate_freeze_hash": config.candidate_freeze_hash,
                    "evaluation_dataset_hash": config.evaluation_dataset_hash,
                    "paired_generation_manifest_hash": config.paired_generation_manifest_hash,
                    "prompt_identity_hash": prompt_identity_hash,
                    "rubric": "Initial Eval Rubric",
                    "scorer_payload_hash": scorer_payload_hash,
                }
            )
        )
        input_artifact = store.put(
            "SolVerdictInput",
            "1.0.0",
            {
                "campaign_id": config.campaign_id,
                "prompt_identity_hash": prompt_identity_hash,
                "rubric": "Initial Eval Rubric",
                "a_trajectory_scope": "evaluator-visible",
                "b_trajectory_scope": "evaluator-visible",
                "scorer_payload_hash": scorer_payload_hash,
                "seed": seed,
                "input_hash": input_hash,
                "lineage": {
                    "candidate_freeze_hash": config.candidate_freeze_hash,
                    "evaluation_dataset_hash": config.evaluation_dataset_hash,
                    "paired_generation_manifest_hash": config.paired_generation_manifest_hash,
                },
            },
        )
        raw_hash = sha256_hex(canonical_json_bytes(raw_output))
        normalized_hash = sha256_hex(canonical_json_bytes({"verdict": normalized}))
        item_dir = Path(root) / "final-verdicts" / config.campaign_id / "items"
        ArtifactStore.durable_mkdir(item_dir)
        artifact = store.put(
            "SolVerdict",
            "1.0.0",
            {
                "campaign_id": config.campaign_id,
                "prompt_identity_hash": prompt_identity_hash,
                "verdict": normalized,
                "raw_output": raw_output,
                "raw_output_hash": raw_hash,
                "normalized_output_hash": normalized_hash,
                "output_hash": normalized_hash,
                "seed": seed,
                "input_hash": input_hash,
                "scorer_payload_hash": scorer_payload_hash,
                "input_artifact_hash": input_artifact.content_hash,
                "lineage": {
                    "candidate_freeze_hash": config.candidate_freeze_hash,
                    "evaluation_dataset_hash": config.evaluation_dataset_hash,
                    "paired_generation_manifest_hash": config.paired_generation_manifest_hash,
                    "sol_verdict_input_hash": input_artifact.content_hash,
                },
                "phase": "BLINDED_SCORING",
                "status": "committed",
                "schema_version": "sol-verdict/1.0.0",
            },
        )
        ref = item_dir / f"{prompt_identity_hash}.ref"
        try:
            ArtifactStore._publish(ref, f"{artifact.content_hash}\n".encode("ascii"))
        except ImmutableArtifactConflict:
            try:
                prior = store.read(ref.read_text(encoding="ascii").strip(), expected_schema_name="SolVerdict")
            except Exception:
                return cls._invalidate(root, config, "DUPLICATE_COMMITTED_VERDICT")
            if prior.content_hash != artifact.content_hash:
                return cls._invalidate(root, config, "DUPLICATE_COMMITTED_VERDICT")
        try:
            journal.record_observation(
                config.controller_epoch,
                "SOL_VERDICT_COMMITTED",
                {
                    "prompt_identity_hash": prompt_identity_hash,
                    "verdict_hash": artifact.content_hash,
                    "phase": "BLINDED_SCORING",
                },
            )
        except RunJournalError as error:
            raise FinalVerdictFencingError("FINAL_EVAL_FENCING_OR_RECOVERY_FAILED") from error
        return cls._collecting_snapshot(root, config)

    submit_verdict = commit_verdict
    commit = commit_verdict
    submit = commit_verdict

    @classmethod
    def unseal(
        cls, root: str | Path, *, config: FinalVerdictConfig, credential: str | None = None
    ) -> FinalVerdictSnapshot:
        if config.execution_profile == "production":
            return FinalVerdictSnapshot("blocked", cls.production_readiness(root, config))
        try:
            cls._assert_inputs(root, config)
        except (FinalVerdictInvalid, FinalVerdictConflictError, FinalVerdictReadinessError) as error:
            return cls._invalidate(root, config, str(error))
        terminal = cls._terminal(root, config)
        if terminal is not None:
            return cls._snapshot_terminal(root, config, terminal)
        journal = cls._journal(root, config)
        try:
            journal.append(config.controller_epoch, "UNSEALING_STARTED", {"phase": "UNSEALING"})
        except RunJournalError as error:
            raise FinalVerdictFencingError("FINAL_EVAL_FENCING_OR_RECOVERY_FAILED") from error
        store = ArtifactStore(root)
        identities = cls._identities(store, config)
        expected_identities = set(identities)
        item_dir = Path(root) / "final-verdicts" / config.campaign_id / "items"
        verdict_artifacts: list[Artifact] = []
        for identity in identities:
            ref = item_dir / f"{identity}.ref"
            if not ref.exists():
                return cls._invalidate(root, config, "MISSING_VERDICT")
            try:
                item = store.read(ref.read_text(encoding="ascii").strip(), expected_schema_name="SolVerdict")
            except Exception:
                return cls._invalidate(root, config, "MALFORMED_COMMITTED_VERDICT")
            if (
                item.payload.get("campaign_id") != config.campaign_id
                or item.payload.get("prompt_identity_hash") != identity
                or item.payload.get("verdict") not in {"A", "B", "tie"}
            ):
                return cls._invalidate(root, config, "UNKNOWN_OR_MALFORMED_VERDICT")
            raw_output = item.payload.get("raw_output")
            if (
                item.payload.get("raw_output_hash") != sha256_hex(canonical_json_bytes(raw_output))
                or item.payload.get("normalized_output_hash")
                != sha256_hex(canonical_json_bytes({"verdict": item.payload.get("verdict")}))
                or item.payload.get("output_hash") != item.payload.get("normalized_output_hash")
                or item.payload.get("phase") != "BLINDED_SCORING"
            ):
                return cls._invalidate(root, config, "VERDICT_HASH_LINEAGE_INVALID")
            try:
                input_artifact = store.read(
                    cast(str, item.payload.get("input_artifact_hash")), expected_schema_name="SolVerdictInput"
                )
                expected_input_hash = sha256_hex(
                    canonical_json_bytes(
                        {
                            "campaign_id": config.campaign_id,
                            "candidate_freeze_hash": config.candidate_freeze_hash,
                            "evaluation_dataset_hash": config.evaluation_dataset_hash,
                            "paired_generation_manifest_hash": config.paired_generation_manifest_hash,
                            "prompt_identity_hash": identity,
                            "rubric": "Initial Eval Rubric",
                            "scorer_payload_hash": item.payload.get("scorer_payload_hash"),
                        }
                    )
                )
            except Exception:
                return cls._invalidate(root, config, "VERDICT_INPUT_LINEAGE_INVALID")
            if (
                item.payload.get("input_hash") != expected_input_hash
                or input_artifact.payload.get("input_hash") != expected_input_hash
                or input_artifact.payload.get("prompt_identity_hash") != identity
                or input_artifact.payload.get("scorer_payload_hash") != item.payload.get("scorer_payload_hash")
            ):
                return cls._invalidate(root, config, "VERDICT_INPUT_LINEAGE_INVALID")
            verdict_artifacts.append(item)
        if item_dir.exists():
            for ref in item_dir.glob("*.ref"):
                if ref.stem not in expected_identities:
                    return cls._invalidate(root, config, "UNKNOWN_PROMPT_IDENTITY")
        if (
            len(verdict_artifacts) != 100
            or len({a.payload.get("prompt_identity_hash") for a in verdict_artifacts}) != 100
        ):
            return cls._invalidate(root, config, "MISSING_OR_DUPLICATE_VERDICT")
        manifest = store.read(
            cast(str, config.paired_generation_manifest_hash), expected_schema_name="PairedGenerationManifest"
        )
        commitment_hash = _hash(
            manifest.payload.get("sealed_mapping_commitment_hash"), "sealed_mapping_commitment_hash"
        )
        commitment = store.read(commitment_hash, expected_schema_name="SealedABMappingCommitment")
        if (
            commitment.payload.get("campaign_id") != config.campaign_id
            or manifest.payload.get("candidate_freeze_hash") != config.candidate_freeze_hash
            or manifest.payload.get("evaluation_dataset_hash") != config.evaluation_dataset_hash
        ):
            return cls._invalidate(root, config, "INPUT_LINEAGE_CONFLICT")
        mapping_path = Path(root) / "sealed-mappings" / config.campaign_id / "mapping.json"
        if credential is None:
            credential = sha256_hex(
                canonical_json_bytes({"campaign_id": config.campaign_id, "purpose": "independent-evaluator"})
            )
        sealed = SealedMapping(commitment, mapping_path, credential)
        try:
            mapping = sealed.unseal(credential)
        except Exception:
            return cls._invalidate(root, config, "SEALED_MAPPING_UNAVAILABLE")
        if set(mapping) != expected_identities:
            return cls._invalidate(root, config, "SEALED_MAPPING_LINEAGE_INVALID")
        trained_wins = 0
        verdict_by_id = {
            cast(str, a.payload["prompt_identity_hash"]): cast(str, a.payload["verdict"]) for a in verdict_artifacts
        }
        for identity, label in mapping.items():
            # Label is the model occupying visible A/B; no model identity was
            # exposed to Sol.  Only explicit trained selection is a win.
            verdict = verdict_by_id.get(identity)
            # Ticket30's scorer payload places base at the committed mapping
            # label and trained in the opposite position.  Therefore only the
            # opposite explicit A/B choice is a trained win; ties never count.
            if verdict in {"A", "B"} and verdict != label:
                trained_wins += 1
        outcome: Outcome = "PASSED" if trained_wins >= 60 else "FAILED"
        verdict_hashes = [a.content_hash for a in verdict_artifacts]
        final_run = store.put(
            "FinalEvalRun",
            "1.0.0",
            {
                "campaign_id": config.campaign_id,
                "protocol_hash": config.protocol_hash
                or cast(str, protocol_hash_from_freeze(store, config.candidate_freeze_hash)),
                "candidate_freeze_hash": config.candidate_freeze_hash,
                "evaluation_dataset_hash": config.evaluation_dataset_hash,
                "paired_generation_manifest_hash": manifest.content_hash,
                "sealed_mapping_commitment_hash": commitment.content_hash,
                "generation_hashes": manifest.payload.get("generation_hashes", []),
                "generation_count": 200,
                "verdict_hashes": verdict_hashes,
                "verdict_count": 100,
                "trained_wins": trained_wins,
                "outcome": outcome,
                "outcome_hash": sha256_hex(
                    canonical_json_bytes(
                        {"trained_wins": trained_wins, "outcome": outcome, "verdict_hashes": verdict_hashes}
                    )
                ),
                "status": "terminal",
                "phase": outcome,
                "report_scope": "Future Prompt pairwise preference only",
                "report_text": f"Sol pairwise preference on Future Prompt: {outcome}.",
                "schema_version": "final-eval-run/1.0.0",
            },
        )
        try:
            journal.close(
                config.controller_epoch,
                status="succeeded" if outcome == "PASSED" else "failed",
                reason_code=f"FINAL_EVAL_{outcome}",
            )
        except RunJournalError as error:
            raise FinalVerdictFencingError("FINAL_EVAL_FENCING_OR_RECOVERY_FAILED") from error
        return cls._publish_terminal(root, config, final_run, outcome, verdict_artifacts, trained_wins)

    finalize = unseal

    @classmethod
    def _assert_inputs(cls, root: str | Path, config: FinalVerdictConfig) -> None:
        store = ArtifactStore(root)
        try:
            freeze = store.read(config.candidate_freeze_hash, expected_schema_name="CandidateFreeze")
            dataset = load_evaluation_dataset(store, config.evaluation_dataset_hash)
            manifest = store.read(
                cast(str, config.paired_generation_manifest_hash), expected_schema_name="PairedGenerationManifest"
            )
            protocol = FinalEvaluationProtocolRegistry.load(root, campaign_id=config.campaign_id)
        except Exception as error:
            raise FinalVerdictReadinessError("FINAL_EVAL_INPUTS_UNAVAILABLE") from error
        if (
            freeze.payload.get("status") != "immutable"
            or freeze.payload.get("campaign_id") != config.campaign_id
            or dataset.payload.get("campaign_id") != config.campaign_id
            or manifest.payload.get("campaign_id") != config.campaign_id
        ):
            raise FinalVerdictReadinessError("FINAL_EVAL_INPUT_LINEAGE_INVALID")
        actual_protocol = protocol.protocol.content_hash
        if config.protocol_hash is not None and config.protocol_hash != actual_protocol:
            raise FinalVerdictConflictError("PROTOCOL_HASH_CONFLICT")
        if (
            freeze.payload.get("protocol_hash") != actual_protocol
            or dataset.payload.get("protocol_hash") != actual_protocol
            or (
                manifest.payload.get("protocol_hash") is not None
                and manifest.payload.get("protocol_hash") != actual_protocol
            )
        ):
            raise FinalVerdictInvalid("PROTOCOL_SCHEMA_DRIFT")
        generation_hashes = manifest.payload.get("generation_hashes")
        if (
            manifest.payload.get("status") != "committed"
            or manifest.payload.get("generation_count") != 200
            or not isinstance(generation_hashes, list)
            or len(generation_hashes) != 200
            or len(set(generation_hashes)) != 200
            or any(type(item) is not str or _HASH.fullmatch(cast(str, item)) is None for item in generation_hashes)
        ):
            raise FinalVerdictReadinessError("PAIRED_GENERATIONS_NOT_EXACTLY_200")
        rows = dataset.payload.get("prompt_rows")
        if not isinstance(rows, list) or len(rows) != 100:
            raise FinalVerdictReadinessError("EVALUATION_DATASET_NOT_EXACTLY_100")
        prompts = {
            cast(str, row.get("prompt_identity_hash")): cast(str, row.get("prompt"))
            for row in rows
            if isinstance(row, dict)
        }
        seen: set[tuple[str, str]] = set()
        for generation_hash in cast(list[str], generation_hashes):
            try:
                trajectory = store.read(generation_hash, expected_schema_name="EvaluatorVisibleTrajectory")
            except Exception as error:
                raise FinalVerdictReadinessError("GENERATION_TRAJECTORY_UNAVAILABLE") from error
            payload = trajectory.payload
            identity = payload.get("prompt_identity_hash")
            role = payload.get("model_role")
            if (
                type(identity) is not str
                or identity not in prompts
                or role not in {"base", "trained"}
                or (cast(str, identity), cast(str, role)) in seen
                or payload.get("prompt") != prompts[identity]
                or payload.get("status") != "committed"
            ):
                raise FinalVerdictReadinessError("GENERATION_TRAJECTORY_LINEAGE_INVALID")
            expected_key = sha256_hex(
                canonical_json_bytes(
                    {
                        "campaign_id": config.campaign_id,
                        "model_role": role,
                        "prompt_identity_hash": identity,
                    }
                )
            )
            if payload.get("idempotency_key") != expected_key:
                raise FinalVerdictReadinessError("GENERATION_IDEMPOTENCY_LINEAGE_INVALID")
            seen.add((cast(str, identity), cast(str, role)))
        if len(seen) != 200 or len({identity for identity, _ in seen}) != 100:
            raise FinalVerdictReadinessError("GENERATION_TRAJECTORY_COVERAGE_INVALID")
        payload_refs = manifest.payload.get("scorer_payload_hashes")
        if (
            not isinstance(payload_refs, Mapping)
            or manifest.payload.get("scorer_payload_count") != 100
            or set(payload_refs) != set(prompts)
        ):
            raise FinalVerdictReadinessError("SOL_SCORER_PAYLOAD_COVERAGE_INVALID")
        for payload_hash in payload_refs.values():
            if type(payload_hash) is not str or _HASH.fullmatch(payload_hash) is None:
                raise FinalVerdictReadinessError("SOL_SCORER_PAYLOAD_HASH_INVALID")
            try:
                payload_artifact = store.read(payload_hash, expected_schema_name="SolScorerPayload")
            except Exception as error:
                raise FinalVerdictReadinessError("SOL_SCORER_PAYLOAD_UNAVAILABLE") from error
            if not cls._payload_scope_valid(payload_artifact.payload):
                raise FinalVerdictReadinessError("SOL_PAYLOAD_SCOPE_INVALID")

    @staticmethod
    def _identities(store: ArtifactStore, config: FinalVerdictConfig) -> list[str]:
        dataset = load_evaluation_dataset(store, config.evaluation_dataset_hash)
        rows = dataset.payload.get("prompt_rows")
        if not isinstance(rows, list) or len(rows) != 100:
            raise FinalVerdictReadinessError("EVALUATION_DATASET_NOT_EXACTLY_100")
        identities = [
            cast(str, cast(dict[str, JsonValue], row).get("prompt_identity_hash"))
            for row in rows
            if isinstance(row, dict)
        ]
        if len(identities) != 100 or len(set(identities)) != 100 or any(_HASH.fullmatch(i) is None for i in identities):
            raise FinalVerdictReadinessError("EVALUATION_DATASET_IDENTITIES_INVALID")
        return identities

    @staticmethod
    def _normalize_verdict(value: object) -> Verdict | None:
        if isinstance(value, str):
            return value if value in {"A", "B", "tie"} else None  # type: ignore[return-value]
        if isinstance(value, Mapping):
            status = value.get("status", "committed")
            if status not in {"committed", "success"}:
                return None
            item = value.get("verdict", value.get("raw_output"))
            return item if item in {"A", "B", "tie"} else None  # type: ignore[return-value]
        return None

    @classmethod
    def _collecting_snapshot(cls, root: str | Path, config: FinalVerdictConfig) -> FinalVerdictSnapshot:
        store = ArtifactStore(root)
        artifacts: list[Artifact] = []
        item_dir = Path(root) / "final-verdicts" / config.campaign_id / "items"
        if item_dir.exists():
            for ref in sorted(item_dir.glob("*.ref")):
                try:
                    artifacts.append(
                        store.read(ref.read_text(encoding="ascii").strip(), expected_schema_name="SolVerdict")
                    )
                except Exception:
                    continue
        readiness = store.put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [{"code": "VERDICTS_COLLECTING", "status": "blocked"}],
                "execution_profile": "fixture",
                "phase": "FINAL_EVAL",
                "side_effects_permitted": False,
                "status": "blocked",
                "committed_verdict_count": len(artifacts),
            },
        )
        return FinalVerdictSnapshot("collecting", readiness, tuple(artifacts))

    @staticmethod
    def _terminal(root: str | Path, config: FinalVerdictConfig) -> Artifact | None:
        directory = Path(root) / "final-verdicts" / config.campaign_id
        for name, schema in (("terminal.ref", "FinalEvalRun"), ("invalid.ref", "FinalEvaluationInvalidation")):
            ref = directory / name
            if ref.exists():
                try:
                    return ArtifactStore(root).read(
                        ref.read_text(encoding="ascii").strip(), expected_schema_name=schema
                    )
                except Exception as error:
                    raise FinalVerdictError("FINAL_EVAL_TERMINAL_CORRUPT") from error
        return None

    @classmethod
    def _publish_terminal(
        cls,
        root: str | Path,
        config: FinalVerdictConfig,
        run: Artifact,
        outcome: Outcome,
        verdicts: list[Artifact],
        wins: int,
    ) -> FinalVerdictSnapshot:
        ref = Path(root) / "final-verdicts" / config.campaign_id / "terminal.ref"
        ArtifactStore._publish(ref, f"{run.content_hash}\n".encode("ascii"))
        readiness = ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [{"code": outcome, "status": "green"}],
                "execution_profile": "fixture",
                "phase": "FINAL_EVAL",
                "side_effects_permitted": False,
                "status": "terminal",
                "trained_wins": wins,
            },
        )
        return FinalVerdictSnapshot(outcome, readiness, tuple(verdicts), run, trained_wins=wins, unsealed=True)

    @classmethod
    def _snapshot_terminal(
        cls, root: str | Path, config: FinalVerdictConfig, terminal: Artifact
    ) -> FinalVerdictSnapshot:
        if terminal.schema_name == "FinalEvaluationInvalidation":
            if (
                terminal.payload.get("campaign_id") != config.campaign_id
                or terminal.payload.get("candidate_freeze_hash") != config.candidate_freeze_hash
                or terminal.payload.get("evaluation_dataset_hash") != config.evaluation_dataset_hash
                or terminal.payload.get("paired_generation_manifest_hash") != config.paired_generation_manifest_hash
            ):
                raise FinalVerdictConflictError("TERMINAL_INVALIDATION_LINEAGE_CONFLICT")
            readiness = ArtifactStore(root).put(
                "ReadinessReport",
                "1.0.0",
                {
                    "checks": [{"code": terminal.payload.get("reason_code", "INVALID"), "status": "invalid"}],
                    "execution_profile": "fixture",
                    "phase": "FINAL_EVAL",
                    "side_effects_permitted": False,
                    "status": "invalid",
                },
            )
            return FinalVerdictSnapshot(
                "INVALID", readiness, reason_code=cast(str, terminal.payload.get("reason_code"))
            )
        outcome = cast(Outcome, terminal.payload.get("outcome"))
        expected_outcome_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "trained_wins": terminal.payload.get("trained_wins"),
                    "outcome": outcome,
                    "verdict_hashes": terminal.payload.get("verdict_hashes"),
                }
            )
        )
        generation_hashes = terminal.payload.get("generation_hashes")
        verdict_hashes = terminal.payload.get("verdict_hashes")
        if (
            terminal.payload.get("campaign_id") != config.campaign_id
            or terminal.payload.get("candidate_freeze_hash") != config.candidate_freeze_hash
            or terminal.payload.get("evaluation_dataset_hash") != config.evaluation_dataset_hash
            or terminal.payload.get("paired_generation_manifest_hash") != config.paired_generation_manifest_hash
            or terminal.payload.get("generation_count") != 200
            or terminal.payload.get("verdict_count") != 100
            or type(terminal.payload.get("protocol_hash")) is not str
            or _HASH.fullmatch(cast(str, terminal.payload.get("protocol_hash"))) is None
            or type(terminal.payload.get("sealed_mapping_commitment_hash")) is not str
            or _HASH.fullmatch(cast(str, terminal.payload.get("sealed_mapping_commitment_hash"))) is None
            or not isinstance(generation_hashes, list)
            or len(generation_hashes) != 200
            or len(set(generation_hashes)) != 200
            or any(type(item) is not str or _HASH.fullmatch(item) is None for item in generation_hashes)
            or not isinstance(verdict_hashes, list)
            or len(verdict_hashes) != 100
            or len(set(verdict_hashes)) != 100
            or any(type(item) is not str or _HASH.fullmatch(item) is None for item in verdict_hashes)
            or type(terminal.payload.get("trained_wins")) is not int
            or not 0 <= cast(int, terminal.payload.get("trained_wins")) <= 100
            or (outcome == "PASSED" and cast(int, terminal.payload.get("trained_wins")) < 60)
            or (outcome == "FAILED" and cast(int, terminal.payload.get("trained_wins")) >= 60)
            or outcome not in {"PASSED", "FAILED"}
            or terminal.payload.get("phase") != outcome
            or terminal.payload.get("report_scope") != "Future Prompt pairwise preference only"
            or terminal.payload.get("outcome_hash") != expected_outcome_hash
        ):
            raise FinalVerdictConflictError("TERMINAL_FINAL_RUN_LINEAGE_CONFLICT")
        readiness = ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [{"code": outcome, "status": "green"}],
                "execution_profile": "fixture",
                "phase": "FINAL_EVAL",
                "side_effects_permitted": False,
                "status": "terminal",
                "trained_wins": terminal.payload.get("trained_wins"),
            },
        )
        return FinalVerdictSnapshot(
            outcome,
            readiness,
            final_run=terminal,
            trained_wins=cast(int, terminal.payload.get("trained_wins")),
            unsealed=True,
        )

    @classmethod
    def _invalidate(cls, root: str | Path, config: FinalVerdictConfig, reason: str) -> FinalVerdictSnapshot:
        store = ArtifactStore(root)
        invalid = store.put(
            "FinalEvaluationInvalidation",
            "1.0.0",
            {
                "campaign_id": config.campaign_id,
                "candidate_freeze_hash": config.candidate_freeze_hash,
                "evaluation_dataset_hash": config.evaluation_dataset_hash,
                "paired_generation_manifest_hash": config.paired_generation_manifest_hash,
                "reason_code": reason,
                "status": "invalid",
            },
        )
        ref = Path(root) / "final-verdicts" / config.campaign_id / "invalid.ref"
        try:
            journal = cls._journal(root, config)
            journal.close(config.controller_epoch, status="failed", reason_code=f"INVALID_{reason[:80]}")
        except RunJournalError as error:
            raise FinalVerdictFencingError("FINAL_EVAL_FENCING_OR_RECOVERY_FAILED") from error
        try:
            ArtifactStore._publish(ref, f"{invalid.content_hash}\n".encode("ascii"))
        except ImmutableArtifactConflict:
            pass
        return cls._snapshot_terminal(root, config, invalid)


# Discoverable aliases used by previous workflow tickets and external fixtures.
FinalEvaluationVerdictWorkflow = SolFinalVerdictWorkflow
FinalVerdictWorkflow = SolFinalVerdictWorkflow
SolVerdictWorkflow = SolFinalVerdictWorkflow
SolVerdictConfig = FinalVerdictConfig
FinalEvaluationVerdictConfig = FinalVerdictConfig
FinalEvaluationRun = FinalVerdictSnapshot
FinalEvalVerdictWorkflow = SolFinalVerdictWorkflow
FinalEvalConfig = FinalVerdictConfig
SolVerdictCollector = SolFinalVerdictWorkflow
