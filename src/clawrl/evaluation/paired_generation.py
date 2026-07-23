"""Ticket 30: production-shaped paired generation and sealed blinding.

The implementation intentionally keeps the provider boundary small.  Fixture
providers are deterministic and never contact a model service; production is
blocked before that boundary can be called.  The campaign is represented by
immutable artifacts and a single active reference, so a fresh process can
reconstruct every decision without session memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from clawrl.artifacts import (
    Artifact,
    ArtifactStore,
    ImmutableArtifactConflict,
    JsonValue,
    canonical_json_bytes,
    sha256_hex,
)
from clawrl.evaluation.future_dataset import load_evaluation_dataset
from clawrl.evaluation.protocol_workflow import FinalEvaluationProtocolRegistry
from clawrl.training.run_journal import RunJournal, RunJournalError

_HASH = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


class PairedGenerationError(RuntimeError):
    """A paired-generation campaign failed closed."""


class PairedGenerationReadinessError(PairedGenerationError):
    """Required input or provider capability is unavailable."""


class PairedGenerationInvalid(PairedGenerationError):
    """The immutable campaign is terminal INVALID."""


class PairedGenerationConflictError(PairedGenerationError):
    """An immutable campaign was called with different input or output."""


class PairedGenerationFencingError(PairedGenerationError):
    """A stale writer attempted a durable transition."""


def _id(value: object, name: str) -> str:
    if type(value) is not str or _ID.fullmatch(cast(str, value)) is None:
        raise PairedGenerationError(f"{name} is invalid")
    return cast(str, value)


def _hash(value: object, name: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise PairedGenerationError(f"{name} must be a SHA-256 digest")
    return cast(str, value)


class GenerationProvider(Protocol):
    """Provider contract used by the fixture workflow.

    A provider must either support querying an idempotency key or explicitly
    guarantee that repeating the exact key is idempotent.  This is checked
    before generation and unknown outcomes are never guessed.
    """

    idempotency_supported: bool
    result_lookup_supported: bool

    def generate(
        self,
        *,
        idempotency_key: str,
        prompt: str,
        model_role: str,
        checkpoint_identity: str,
        evaluation_environment_hash: str,
    ) -> Mapping[str, object]: ...

    def lookup(self, idempotency_key: str) -> Mapping[str, object] | None: ...


class FixturePairedGenerationProvider:
    """Deterministic synthetic provider; no network or real model resources."""

    idempotency_supported = True
    result_lookup_supported = True

    def __init__(self, *, outcomes: Mapping[str, object] | None = None) -> None:
        self.outcomes = dict(outcomes or {})
        self.calls: list[str] = []
        self._results: dict[str, Mapping[str, object]] = {}

    def generate(
        self,
        *,
        idempotency_key: str,
        prompt: str,
        model_role: str,
        checkpoint_identity: str,
        evaluation_environment_hash: str,
    ) -> Mapping[str, object]:
        self.calls.append(idempotency_key)
        directive = self.outcomes.get(idempotency_key)
        if directive is not None:
            if isinstance(directive, Mapping):
                result = dict(directive)
            else:
                return {"status": str(directive)}
        else:
            result = {
                "status": "committed",
                "prompt": prompt,
                "response": (
                    f"fixture-{model_role}-{hashlib.sha256((idempotency_key + prompt).encode()).hexdigest()[:16]}"
                ),
                "tool_transcript": [],
                "checkpoint_identity": checkpoint_identity,
                "evaluation_environment_hash": evaluation_environment_hash,
            }
        if result.get("status") in {"committed", "success"}:
            self._results[idempotency_key] = result
        return result

    def lookup(self, idempotency_key: str) -> Mapping[str, object] | None:
        return self._results.get(idempotency_key)


@dataclass(frozen=True, slots=True)
class PairedGenerationConfig:
    campaign_id: str
    candidate_freeze_hash: str
    evaluation_dataset_hash: str
    protocol_hash: str | None = None
    balanced_ab_seed: int | None = None
    controller_epoch: int = 1
    execution_profile: Literal["fixture", "production"] = "fixture"

    def __post_init__(self) -> None:
        _id(self.campaign_id, "campaign_id")
        _hash(self.candidate_freeze_hash, "candidate_freeze_hash")
        _hash(self.evaluation_dataset_hash, "evaluation_dataset_hash")
        if self.protocol_hash is not None:
            _hash(self.protocol_hash, "protocol_hash")
        if self.balanced_ab_seed is not None and (type(self.balanced_ab_seed) is not int or self.balanced_ab_seed < 0):
            raise PairedGenerationError("balanced_ab_seed is invalid")
        if type(self.controller_epoch) is not int or self.controller_epoch <= 0:
            raise PairedGenerationError("controller_epoch is invalid")
        if self.execution_profile not in {"fixture", "production"}:
            raise PairedGenerationError("execution_profile is invalid")


@dataclass(frozen=True, slots=True)
class SealedMapping:
    """A commitment handle; the mapping bytes are outside ordinary artifacts."""

    commitment: Artifact
    _path: Path
    _credential: str

    def unseal(self, credential: str) -> dict[str, str]:
        if credential != self._credential:
            raise PairedGenerationError("SEALED_MAPPING_ACCESS_DENIED")
        try:
            raw = self._path.read_bytes()
            decoded = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as error:
            raise PairedGenerationError("SEALED_MAPPING_CORRUPT") from error
        if not isinstance(decoded, dict) or decoded.get("commitment") != self.commitment.content_hash:
            raise PairedGenerationError("SEALED_MAPPING_COMMITMENT_MISMATCH")
        mapping = decoded.get("mapping")
        if not isinstance(mapping, dict) or set(mapping) != {str(k) for k in mapping}:
            raise PairedGenerationError("SEALED_MAPPING_MALFORMED")
        normalized = {cast(str, key): cast(str, value) for key, value in mapping.items()}
        if (
            len(normalized) != 100
            or set(normalized.values()) != {"A", "B"}
            or list(normalized.values()).count("A") != 50
            or list(normalized.values()).count("B") != 50
            or sha256_hex(canonical_json_bytes(normalized)) != self.commitment.payload.get("mapping_hash")
        ):
            raise PairedGenerationError("SEALED_MAPPING_COMMITMENT_MISMATCH")
        return normalized

    @property
    def payload(self) -> dict[str, JsonValue]:
        # Deliberately expose only the commitment to scorer-facing callers.
        return self.commitment.payload


@dataclass(frozen=True, slots=True)
class PairedGenerationSnapshot:
    generations: tuple[Artifact, ...] | None
    sealed_mapping: Artifact | None
    readiness: Artifact
    status: Literal["committed", "invalid", "blocked"]
    reason_code: str | None = None
    input_artifact: Artifact | None = None
    _evaluator_mapping: SealedMapping | None = field(default=None, repr=False, compare=False)

    @property
    def mapping(self) -> Artifact | None:
        return self.sealed_mapping

    @property
    def mapping_commitment_hash(self) -> str | None:
        return self.sealed_mapping.content_hash if self.sealed_mapping is not None else None

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        return self.generations or ()


class PairedGenerationWorkflow:
    """Generate exactly one base/trained trajectory per Future-100 prompt."""

    @staticmethod
    def production_readiness(root: str | Path, config: object | None = None) -> Artifact:
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": "TRUSTED_PROVIDER_RESULT_LOOKUP_UNAVAILABLE", "status": "blocked"},
                    {"code": "PROVIDER_IDEMPOTENCY_CONTRACT_UNAVAILABLE", "status": "blocked"},
                    {"code": "SEALED_MAPPING_ACL_KMS_UNAVAILABLE", "status": "blocked"},
                    {"code": "INDEPENDENT_EVALUATOR_CREDENTIAL_UNAVAILABLE", "status": "blocked"},
                ],
                "execution_profile": "production",
                "phase": "FINAL_EVAL",
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )

    @classmethod
    def run(
        cls, root: str | Path, *, config: PairedGenerationConfig, provider: GenerationProvider | None = None
    ) -> PairedGenerationSnapshot:
        if config.execution_profile == "production":
            return PairedGenerationSnapshot(None, None, cls.production_readiness(root, config), "blocked")
        store = ArtifactStore(root)
        ref = Path(root) / "paired-generations" / config.campaign_id / "active.ref"
        invalid_ref = ref.parent / "invalid.ref"
        if ref.exists():
            existing = cls._read_ref(store, ref)
            cls._validate_existing(existing, config, store)
            cls._repair_journal(root, store, existing, config.controller_epoch)
            return cls._snapshot_from_manifest(root, store, existing)
        if invalid_ref.exists():
            invalid = store.read(
                invalid_ref.read_text(encoding="ascii").strip(), expected_schema_name="PairedGenerationInvalidation"
            )
            if (
                invalid.payload.get("campaign_id") != config.campaign_id
                or invalid.payload.get("candidate_freeze_hash") != config.candidate_freeze_hash
                or invalid.payload.get("evaluation_dataset_hash") != config.evaluation_dataset_hash
            ):
                raise PairedGenerationConflictError("PAIRED_GENERATION_INVALIDATION_CONFLICT")
            return PairedGenerationSnapshot(
                None,
                None,
                store.put(
                    "ReadinessReport",
                    "1.0.0",
                    {
                        "checks": [
                            {"code": invalid.payload.get("reason_code", "CAMPAIGN_INVALID"), "status": "invalid"}
                        ],
                        "execution_profile": "fixture",
                        "phase": "FINAL_EVAL",
                        "side_effects_permitted": False,
                        "status": "invalid",
                    },
                ),
                "invalid",
                cast(str, invalid.payload.get("reason_code")),
            )
        if provider is None:
            provider = FixturePairedGenerationProvider()
        if not bool(cls._provider_capability(provider, "idempotency") or cls._provider_capability(provider, "lookup")):
            raise PairedGenerationReadinessError("TRUSTED_PROVIDER_RESULT_LOOKUP_OR_IDEMPOTENCY_REQUIRED")
        freeze, dataset, protocol = cls._load_inputs(root, store, config)
        rows = dataset.payload.get("prompt_rows")
        if not isinstance(rows, list) or len(rows) != 100:
            return cls._invalidate(root, store, config, "EVALUATION_DATASET_NOT_EXACTLY_100")
        env_hash = freeze.payload.get("evaluation_environment_hash")
        _hash(env_hash, "evaluation_environment_hash")
        seed = config.balanced_ab_seed
        if seed is None:
            assignment = protocol.protocol.payload.get("balanced_ab_assignment", {})
            seed = assignment.get("seed") if isinstance(assignment, dict) else None
        if type(seed) is not int or seed < 0:
            raise PairedGenerationReadinessError("BALANCED_AB_SEED_NOT_PRECOMMITTED")
        prereg_seed = cast(dict[str, object], protocol.protocol.payload.get("balanced_ab_assignment", {})).get("seed")
        if config.balanced_ab_seed is not None and config.balanced_ab_seed != prereg_seed:
            raise PairedGenerationConflictError("BALANCED_AB_SEED_CONFLICT")
        try:
            assignment = cls._assignment(rows, seed)
        except (PairedGenerationInvalid, PairedGenerationConflictError) as error:
            return cls._invalidate(root, store, config, str(error))
        input_artifact = store.put(
            "PairedGenerationInput",
            "1.0.0",
            {
                "campaign_id": config.campaign_id,
                "candidate_freeze_hash": freeze.content_hash,
                "evaluation_dataset_hash": dataset.content_hash,
                "protocol_hash": protocol.protocol.content_hash,
                "evaluation_environment_hash": cast(str, env_hash),
                "balanced_ab_seed": seed,
                "controller_epoch": config.controller_epoch,
                "schema_version": "paired-generation-input/1.0.0",
            },
        )
        cls._preflight_sealed_storage(root, config.campaign_id)
        journal = RunJournal(
            root,
            store,
            "paired_eval_" + re.sub(r"[^A-Za-z0-9._-]", "_", config.campaign_id),
            input_schema_name="PairedGenerationInput",
        )
        try:
            journal.reserve_identity(input_artifact.content_hash)
            events = journal.events()
            if not events:
                journal.start_run(
                    config.controller_epoch,
                    input_artifact.content_hash,
                    {"input_hash": input_artifact.content_hash, "phase": "FINAL_EVAL"},
                )
                journal.append(config.controller_epoch, "READINESS_GREEN", {"evaluation_environment_hash": env_hash})
            else:
                journal.start_run(
                    config.controller_epoch,
                    input_artifact.content_hash,
                    {"input_hash": input_artifact.content_hash, "phase": "FINAL_EVAL"},
                )
        except RunJournalError as error:
            raise PairedGenerationFencingError("PAIRED_GENERATION_FENCING_UNAVAILABLE") from error
        committed: list[Artifact] = []
        try:
            for row in rows:
                if not isinstance(row, dict):
                    raise PairedGenerationInvalid("MALFORMED_EVALUATION_DATASET_ROW")
                identity = row.get("prompt_identity_hash")
                prompt = row.get("prompt")
                _hash(identity, "prompt_identity_hash")
                if type(prompt) is not str or not prompt:
                    raise PairedGenerationInvalid("MALFORMED_EVALUATION_DATASET_ROW")
                for role in ("base", "trained"):
                    checkpoint = (
                        freeze.payload.get(f"{role}_checkpoint_hash")
                        if role == "base"
                        else freeze.payload.get("trained_checkpoint_hash")
                    )
                    # Base checkpoint is mandatory for pairing; candidate freeze stores it under this name.
                    if role == "base":
                        checkpoint = freeze.payload.get("base_checkpoint_hash")
                    _hash(checkpoint, f"{role}_checkpoint_hash")
                    key = sha256_hex(
                        canonical_json_bytes(
                            {"campaign_id": config.campaign_id, "model_role": role, "prompt_identity_hash": identity}
                        )
                    )
                    try:
                        result = cls._provider_result(
                            provider, key, cast(str, prompt), role, cast(str, checkpoint), cast(str, env_hash)
                        )
                    except Exception as error:
                        raise PairedGenerationInvalid("PROVIDER_OUTCOME_UNRESOLVED") from error
                    payload = cls._trajectory_payload(
                        result,
                        key,
                        config,
                        cast(str, identity),
                        cast(str, prompt),
                        role,
                        cast(str, checkpoint),
                        cast(str, env_hash),
                    )
                    try:
                        artifact = store.put("EvaluatorVisibleTrajectory", "1.0.0", payload)
                    except Exception as error:
                        raise PairedGenerationInvalid("MALFORMED_COMMITTED_GENERATION") from error
                    committed.append(artifact)
                    item_ref = ref.parent / "items" / f"{identity}.{role}.ref"
                    try:
                        ArtifactStore._publish(item_ref, f"{artifact.content_hash}\n".encode("ascii"))
                    except ImmutableArtifactConflict:
                        winner = store.read(
                            item_ref.read_text(encoding="ascii").strip(),
                            expected_schema_name="EvaluatorVisibleTrajectory",
                        )
                        if winner.content_hash != artifact.content_hash:
                            raise PairedGenerationConflictError("DUPLICATE_COMMITTED_GENERATION") from None
            if len(committed) != 200:
                raise PairedGenerationInvalid("MISSING_COMMITTED_GENERATION")
            mapping = cls._seal_mapping(
                root, store, config.campaign_id, assignment, config.controller_epoch, input_artifact.content_hash
            )
            manifest = store.put(
                "PairedGenerationManifest",
                "1.0.0",
                {
                    "campaign_id": config.campaign_id,
                    "candidate_freeze_hash": freeze.content_hash,
                    "evaluation_dataset_hash": dataset.content_hash,
                    "input_hash": input_artifact.content_hash,
                    "evaluation_environment_hash": env_hash,
                    "generation_hashes": [a.content_hash for a in committed],
                    "sealed_mapping_commitment_hash": mapping.commitment.content_hash,
                    "generation_count": 200,
                    "schema_version": "paired-generation-manifest/1.0.0",
                    "status": "committed",
                },
            )
            try:
                ArtifactStore._publish(ref, f"{manifest.content_hash}\n".encode("ascii"))
            except ImmutableArtifactConflict:
                winner = cls._read_ref(store, ref)
                cls._validate_existing(winner, config, store)
                manifest = winner
            events = journal.events()
            if not any(event.payload.get("event_type") == "PAIRED_GENERATIONS_COMMITTED" for event in events):
                journal.append(
                    config.controller_epoch,
                    "PAIRED_GENERATIONS_COMMITTED",
                    {"manifest_hash": manifest.content_hash},
                )
                events = journal.events()
            if events and events[-1].payload.get("event_type") != "RUN_CLOSED":
                journal.close(
                    config.controller_epoch,
                    status="succeeded",
                    reason_code="PAIRED_GENERATIONS_COMMITTED",
                    expected_sequence=len(events) + 1,
                    expected_previous_hash=events[-1].content_hash,
                )
            return cls._snapshot_from_manifest(root, store, manifest)
        except (PairedGenerationInvalid, PairedGenerationConflictError) as error:
            return cls._invalidate(root, store, config, str(error), journal=journal)

    generate = run
    build = run
    create = run

    @classmethod
    def resume(
        cls, root: str | Path, campaign_id: str, *, config: PairedGenerationConfig | None = None
    ) -> PairedGenerationSnapshot:
        _id(campaign_id, "campaign_id")
        store = ArtifactStore(root)
        ref = Path(root) / "paired-generations" / campaign_id / "active.ref"
        if not ref.exists():
            raise PairedGenerationError("campaign has no committed paired generations")
        manifest = cls._read_ref(store, ref)
        if config is not None:
            cls._validate_existing(manifest, config, store)
            cls._repair_journal(root, store, manifest, config.controller_epoch)
        return cls._snapshot_from_manifest(root, store, manifest)

    @staticmethod
    def _repair_journal(root: str | Path, store: ArtifactStore, manifest: Artifact, epoch: int) -> None:
        input_hash = manifest.payload.get("input_hash")
        if not isinstance(input_hash, str):
            raise PairedGenerationFencingError("PAIRED_GENERATION_INPUT_MISSING")
        journal = RunJournal(
            root,
            store,
            "paired_eval_" + re.sub(r"[^A-Za-z0-9._-]", "_", cast(str, manifest.payload.get("campaign_id"))),
            input_schema_name="PairedGenerationInput",
        )
        try:
            journal.reserve_identity(input_hash)
            events = journal.events()
            if not events:
                raise PairedGenerationFencingError("PAIRED_GENERATION_JOURNAL_MISSING")
            if events[-1].payload.get("event_type") != "RUN_CLOSED":
                journal.start_run(epoch, input_hash, {"input_hash": input_hash, "phase": "FINAL_EVAL"})
                if not any(e.payload.get("event_type") == "PAIRED_GENERATIONS_COMMITTED" for e in events):
                    journal.append(epoch, "PAIRED_GENERATIONS_COMMITTED", {"manifest_hash": manifest.content_hash})
                    events = journal.events()
                journal.close(
                    epoch,
                    status="succeeded",
                    reason_code="PAIRED_GENERATIONS_COMMITTED",
                    expected_sequence=len(events) + 1,
                    expected_previous_hash=events[-1].content_hash,
                )
        except RunJournalError as error:
            raise PairedGenerationFencingError("PAIRED_GENERATION_JOURNAL_RECOVERY_FAILED") from error

    @staticmethod
    def scorer_payload(
        root: str | Path,
        snapshot: PairedGenerationSnapshot,
        prompt_identity_hash: str,
        *,
        rubric: Mapping[str, object] | None = None,
    ) -> dict[str, JsonValue]:
        if snapshot.status != "committed" or snapshot.generations is None:
            raise PairedGenerationError("PAIRED_GENERATIONS_NOT_COMMITTED")
        _hash(prompt_identity_hash, "prompt_identity_hash")
        selected = [a for a in snapshot.generations if a.payload.get("prompt_identity_hash") == prompt_identity_hash]
        if len(selected) != 2:
            raise PairedGenerationError("MISSING_OR_DUPLICATE_COMMITTED_GENERATION")
        trajectories = {cast(str, a.payload["model_role"]): a.payload for a in selected}
        if set(trajectories) != {"base", "trained"}:
            raise PairedGenerationError("MALFORMED_COMMITTED_GENERATION")
        # Resolve the assignment inside this trusted adapter and discard it
        # before returning the scorer payload.  The caller never receives the
        # sealed manifest or a model-role field.
        if snapshot._evaluator_mapping is None:
            raise PairedGenerationError("SEALED_MAPPING_UNAVAILABLE")
        assignment = snapshot._evaluator_mapping.unseal(snapshot._evaluator_mapping._credential)
        label = assignment.get(prompt_identity_hash)
        if label not in {"A", "B"}:
            raise PairedGenerationError("SEALED_MAPPING_MALFORMED")

        def visible(payload: Mapping[str, object]) -> dict[str, JsonValue]:
            return {
                "prompt": cast(str, payload["prompt"]),
                "response": cast(str, payload["response"]),
                "tool_transcript": cast(list[JsonValue], payload["tool_transcript"]),
            }

        a_role, b_role = ("base", "trained") if label == "A" else ("trained", "base")
        # No checkpoint, model role, mapping, or student prompt metadata crosses this boundary.
        from clawrl.judge.fit_models import InitialEvalRubric

        return {
            "rubric": cast(
                JsonValue,
                dict(rubric) if rubric is not None else InitialEvalRubric.fixture_default().artifact_payload(),
            ),
            "a_trajectory": visible(trajectories[a_role]),
            "b_trajectory": visible(trajectories[b_role]),
        }

    build_scorer_payload = scorer_payload
    build_sol_payload = scorer_payload

    @staticmethod
    def _load_inputs(
        root: str | Path, store: ArtifactStore, config: PairedGenerationConfig
    ) -> tuple[Artifact, Artifact, Any]:
        try:
            freeze = store.read(config.candidate_freeze_hash, expected_schema_name="CandidateFreeze")
            dataset = load_evaluation_dataset(store, config.evaluation_dataset_hash)
            campaign = cast(str, freeze.payload.get("campaign_id"))
            protocol = FinalEvaluationProtocolRegistry.load(root, campaign_id=campaign)
        except Exception as error:
            raise PairedGenerationReadinessError("CANDIDATE_FREEZE_OR_EVALUATION_DATASET_UNAVAILABLE") from error
        if freeze.payload.get("status") != "immutable" or campaign != config.campaign_id:
            raise PairedGenerationReadinessError("CANDIDATE_FREEZE_INVALID")
        freeze_ref = Path(root) / "candidate-freeze" / config.campaign_id / "active.ref"
        if not freeze_ref.exists() or freeze_ref.read_text(encoding="ascii").strip() != freeze.content_hash:
            raise PairedGenerationReadinessError("CANDIDATE_FREEZE_NOT_DURABLY_ACTIVE")
        if dataset.payload.get("candidate_freeze_hash") != freeze.content_hash:
            raise PairedGenerationReadinessError("EVALUATION_DATASET_FREEZE_LINEAGE_MISMATCH")
        if dataset.payload.get("campaign_id") != config.campaign_id:
            raise PairedGenerationReadinessError("EVALUATION_DATASET_CAMPAIGN_LINEAGE_MISMATCH")
        if config.protocol_hash is not None and config.protocol_hash != protocol.protocol.content_hash:
            raise PairedGenerationConflictError("PROTOCOL_HASH_CONFLICT")
        if freeze.payload.get("protocol_hash") != protocol.protocol.content_hash:
            raise PairedGenerationReadinessError("CANDIDATE_FREEZE_PROTOCOL_LINEAGE_MISMATCH")
        if freeze.payload.get("evaluation_environment_hash") is None:
            raise PairedGenerationReadinessError("EVALUATION_ENVIRONMENT_MISSING")
        if dataset.payload.get("protocol_hash") != protocol.protocol.content_hash:
            raise PairedGenerationReadinessError("EVALUATION_DATASET_PROTOCOL_LINEAGE_MISMATCH")
        return freeze, dataset, protocol

    @staticmethod
    def _assignment(rows: Sequence[object], seed: int) -> dict[str, str]:
        identities = [cast(str, cast(dict[str, object], row).get("prompt_identity_hash")) for row in rows]
        if len(identities) != 100 or len(set(identities)) != 100 or any(_HASH.fullmatch(i) is None for i in identities):
            raise PairedGenerationInvalid("EVALUATION_DATASET_IDENTITIES_INVALID")
        ordered = sorted(identities, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).digest())
        mapping = {identity: ("A" if index < 50 else "B") for index, identity in enumerate(ordered)}
        if list(mapping.values()).count("A") != 50 or list(mapping.values()).count("B") != 50:
            raise PairedGenerationInvalid("BALANCED_MAPPING_NOT_EXACTLY_50_50")
        return mapping

    @staticmethod
    def _provider_result(
        provider: GenerationProvider, key: str, prompt: str, role: str, checkpoint: str, env: str
    ) -> Mapping[str, object]:
        result = provider.generate(
            idempotency_key=key,
            prompt=prompt,
            model_role=role,
            checkpoint_identity=checkpoint,
            evaluation_environment_hash=env,
        )
        if not isinstance(result, Mapping):
            raise PairedGenerationInvalid("MALFORMED_COMMITTED_GENERATION")
        status = result.get("status", "committed")
        if status in {"unknown", "unknown_outcome", "indeterminate"}:
            found = provider.lookup(key) if PairedGenerationWorkflow._provider_capability(provider, "lookup") else None
            if found is None and not PairedGenerationWorkflow._provider_capability(provider, "idempotency"):
                raise PairedGenerationInvalid("NON_IDEMPOTENT_UNKNOWN_OUTCOME")
            if found is None:
                found = provider.generate(
                    idempotency_key=key,
                    prompt=prompt,
                    model_role=role,
                    checkpoint_identity=checkpoint,
                    evaluation_environment_hash=env,
                )
            result = found
        if result.get("status", "committed") not in {"committed", "success"}:
            raise PairedGenerationInvalid("PROVIDER_RESULT_NOT_COMMITTED")
        return result

    @staticmethod
    def _provider_capability(provider: GenerationProvider, capability: Literal["idempotency", "lookup"]) -> bool:
        names = (
            ("idempotency_supported", "supports_idempotency", "supports_same_idempotency_key")
            if capability == "idempotency"
            else ("result_lookup_supported", "supports_result_lookup", "supports_lookup")
        )
        explicit = [getattr(provider, name, None) for name in names if hasattr(provider, name)]
        if explicit:
            return any(value is True for value in explicit)
        return capability == "lookup" and callable(getattr(provider, "lookup", None))

    @staticmethod
    def _trajectory_payload(
        result: Mapping[str, object],
        key: str,
        config: PairedGenerationConfig,
        identity: str,
        prompt: str,
        role: str,
        checkpoint: str,
        env: str,
    ) -> dict[str, JsonValue]:
        nested = result.get("trajectory")
        if isinstance(nested, Mapping):
            merged = dict(result)
            merged.update(nested)
            result = merged
        required = ("response", "tool_transcript")
        if (
            any(field not in result for field in required)
            or type(result.get("response")) is not str
            or not isinstance(result.get("tool_transcript"), list)
        ):
            raise PairedGenerationInvalid("MALFORMED_COMMITTED_GENERATION")
        if "idempotency_key" in result and result.get("idempotency_key") != key:
            raise PairedGenerationInvalid("MALFORMED_COMMITTED_GENERATION")
        if (
            result.get("prompt", prompt) != prompt
            or result.get("checkpoint_identity", checkpoint) != checkpoint
            or result.get("evaluation_environment_hash", env) != env
        ):
            raise PairedGenerationInvalid("MALFORMED_COMMITTED_GENERATION")
        transcript = cast(list[JsonValue], result["tool_transcript"])
        return {
            "campaign_id": config.campaign_id,
            "evaluation_environment_hash": env,
            "idempotency_key": key,
            "model_role": role,
            "prompt": prompt,
            "prompt_identity_hash": identity,
            "response": cast(str, result["response"]),
            "tool_transcript": transcript,
            "checkpoint_identity": checkpoint,
            "status": "committed",
            "schema_version": "evaluator-visible-trajectory/1.0.0",
        }

    @staticmethod
    def _seal_mapping(
        root: str | Path, store: ArtifactStore, campaign: str, mapping: dict[str, str], epoch: int, input_hash: str
    ) -> SealedMapping:
        commitment = store.put(
            "SealedABMappingCommitment",
            "1.0.0",
            {
                "campaign_id": campaign,
                "input_hash": input_hash,
                "mapping_count": 100,
                "trained_as_a_count": 50,
                "trained_as_b_count": 50,
                "mapping_hash": sha256_hex(canonical_json_bytes(mapping)),
                "controller_epoch": epoch,
                "storage": "independent-sealed-store",
                "scorer_read": False,
                "schema_version": "sealed-ab-mapping-commitment/1.0.0",
            },
        )
        path = Path(root) / "sealed-mappings" / campaign / "mapping.json"
        ArtifactStore.durable_mkdir(path.parent)
        body = canonical_json_bytes({"commitment": commitment.content_hash, "mapping": mapping})
        if path.exists() and path.read_bytes() != body:
            raise PairedGenerationConflictError("SEALED_MAPPING_CONFLICT")
        if not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(body)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    pass
        credential = sha256_hex(canonical_json_bytes({"campaign_id": campaign, "purpose": "independent-evaluator"}))
        return SealedMapping(commitment, path, credential)

    @staticmethod
    def _preflight_sealed_storage(root: str | Path, campaign: str) -> None:
        """Verify the fixture sealed boundary before any provider invocation."""

        directory = Path(root) / "sealed-mappings" / campaign
        ArtifactStore.durable_mkdir(directory)
        try:
            os.chmod(directory, 0o700)
        except OSError as error:
            raise PairedGenerationReadinessError("SEALED_MAPPING_ACL_KMS_UNAVAILABLE") from error
        if not os.access(directory, os.W_OK | os.X_OK):
            raise PairedGenerationReadinessError("SEALED_MAPPING_ACL_KMS_UNAVAILABLE")
        path = directory / "mapping.json"
        if path.exists() and (path.stat().st_mode & 0o077):
            raise PairedGenerationReadinessError("SEALED_MAPPING_ACL_KMS_UNAVAILABLE")

    @staticmethod
    def _read_ref(store: ArtifactStore, ref: Path) -> Artifact:
        try:
            return store.read(ref.read_text(encoding="ascii").strip())
        except Exception as error:
            raise PairedGenerationError("PAIRED_GENERATION_MANIFEST_CORRUPT") from error

    @staticmethod
    def _validate_existing(manifest: Artifact, config: PairedGenerationConfig, store: ArtifactStore) -> None:
        p = manifest.payload
        if (
            manifest.schema_name != "PairedGenerationManifest"
            or p.get("status") != "committed"
            or p.get("campaign_id") != config.campaign_id
        ):
            raise PairedGenerationConflictError("PAIRED_GENERATION_MANIFEST_INVALID")
        if (
            p.get("candidate_freeze_hash") != config.candidate_freeze_hash
            or p.get("evaluation_dataset_hash") != config.evaluation_dataset_hash
        ):
            raise PairedGenerationConflictError("PAIRED_GENERATION_INPUT_CONFLICT")
        hashes = p.get("generation_hashes")
        if not isinstance(hashes, list) or len(hashes) != 200 or len(set(hashes)) != 200:
            raise PairedGenerationConflictError("GENERATION_MANIFEST_CARDINALITY_INVALID")
        expected_environment = p.get("evaluation_environment_hash")
        _hash(expected_environment, "evaluation_environment_hash")
        seen: set[tuple[str, str]] = set()
        for item in hashes:
            try:
                trajectory = store.read(cast(str, item), expected_schema_name="EvaluatorVisibleTrajectory")
                payload = trajectory.payload
            except Exception as error:
                raise PairedGenerationConflictError("GENERATION_MANIFEST_ARTIFACT_MISSING") from error
            identity = payload.get("prompt_identity_hash")
            role = payload.get("model_role")
            if (
                payload.get("status") != "committed"
                or payload.get("campaign_id") != config.campaign_id
                or payload.get("evaluation_environment_hash") != expected_environment
                or type(identity) is not str
                or _HASH.fullmatch(identity) is None
                or role not in {"base", "trained"}
                or (identity, cast(str, role)) in seen
            ):
                raise PairedGenerationConflictError("GENERATION_MANIFEST_LINEAGE_INVALID")
            seen.add((identity, cast(str, role)))
        identities = {identity for identity, _ in seen}
        if (
            len(identities) != 100
            or len([item for item in seen if item[1] == "base"]) != 100
            or len([item for item in seen if item[1] == "trained"]) != 100
        ):
            raise PairedGenerationConflictError("GENERATION_MANIFEST_COVERAGE_INVALID")
        try:
            freeze = store.read(config.candidate_freeze_hash, expected_schema_name="CandidateFreeze")
            dataset = load_evaluation_dataset(store, config.evaluation_dataset_hash)
            input_hash = p.get("input_hash")
            input_artifact = store.read(cast(str, input_hash), expected_schema_name="PairedGenerationInput")
            commitment = store.read(
                cast(str, p.get("sealed_mapping_commitment_hash")), expected_schema_name="SealedABMappingCommitment"
            )
        except Exception as error:
            raise PairedGenerationConflictError("PAIRED_GENERATION_LINEAGE_MISSING") from error
        protocol = FinalEvaluationProtocolRegistry.load(store.root, campaign_id=config.campaign_id)
        if (
            freeze.payload.get("evaluation_environment_hash") != protocol.environment.content_hash
            or expected_environment != protocol.environment.content_hash
            or input_artifact.payload.get("protocol_hash") != protocol.protocol.content_hash
            or input_artifact.payload.get("balanced_ab_seed")
            != cast(dict[str, object], protocol.protocol.payload.get("balanced_ab_assignment", {})).get("seed")
        ):
            raise PairedGenerationConflictError("PAIRED_GENERATION_ENVIRONMENT_OR_INPUT_LINEAGE_INVALID")
        row_map = {
            cast(str, row["prompt_identity_hash"]): cast(str, row["prompt"])
            for row in cast(list[dict[str, JsonValue]], dataset.payload["prompt_rows"])
        }
        for item in hashes:
            trajectory = store.read(cast(str, item), expected_schema_name="EvaluatorVisibleTrajectory")
            tp = trajectory.payload
            identity = cast(str, tp["prompt_identity_hash"])
            role = cast(str, tp["model_role"])
            checkpoint = freeze.payload.get("base_checkpoint_hash" if role == "base" else "trained_checkpoint_hash")
            expected_key = sha256_hex(
                canonical_json_bytes(
                    {"campaign_id": config.campaign_id, "model_role": role, "prompt_identity_hash": identity}
                )
            )
            if (
                tp.get("prompt") != row_map.get(identity)
                or tp.get("idempotency_key") != expected_key
                or tp.get("checkpoint_identity") != checkpoint
                or tp.get("evaluation_environment_hash") != expected_environment
            ):
                raise PairedGenerationConflictError("GENERATION_TRAJECTORY_LINEAGE_INVALID")
        if (
            commitment.payload.get("mapping_count") != 100
            or commitment.payload.get("trained_as_a_count") != 50
            or commitment.payload.get("trained_as_b_count") != 50
            or input_artifact.payload.get("candidate_freeze_hash") != freeze.content_hash
            or input_artifact.payload.get("evaluation_dataset_hash") != dataset.content_hash
        ):
            raise PairedGenerationConflictError("PAIRED_GENERATION_COMMITMENT_LINEAGE_INVALID")

    @classmethod
    def _snapshot_from_manifest(
        cls, root: str | Path, store: ArtifactStore, manifest: Artifact
    ) -> PairedGenerationSnapshot:
        hashes = cast(list[object], manifest.payload.get("generation_hashes", []))
        generations = tuple(store.read(cast(str, h), expected_schema_name="EvaluatorVisibleTrajectory") for h in hashes)
        commitment_hash = cast(str, manifest.payload.get("sealed_mapping_commitment_hash"))
        commitment = store.read(commitment_hash, expected_schema_name="SealedABMappingCommitment")
        campaign = cast(str, manifest.payload.get("campaign_id"))
        credential = sha256_hex(canonical_json_bytes({"campaign_id": campaign, "purpose": "independent-evaluator"}))
        mapping_path = Path(root) / "sealed-mappings" / campaign / "mapping.json"
        if (
            not mapping_path.is_file()
            or mapping_path.stat().st_mode & 0o077
            or mapping_path.parent.stat().st_mode & 0o077
        ):
            raise PairedGenerationError("SEALED_MAPPING_UNAVAILABLE")
        sealed = SealedMapping(commitment, mapping_path, credential)
        sealed.unseal(credential)
        readiness = store.put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [{"code": "PAIRED_GENERATIONS_COMMITTED", "status": "green"}],
                "execution_profile": "fixture",
                "phase": "FINAL_EVAL",
                "side_effects_permitted": True,
                "status": "ready",
            },
        )
        input_hash = manifest.payload.get("input_hash")
        input_artifact = (
            store.read(cast(str, input_hash), expected_schema_name="PairedGenerationInput")
            if isinstance(input_hash, str)
            else None
        )
        return PairedGenerationSnapshot(
            generations,
            commitment,
            readiness,
            "committed",
            input_artifact=input_artifact,
            _evaluator_mapping=sealed,
        )

    @classmethod
    def _invalidate(
        cls,
        root: str | Path,
        store: ArtifactStore,
        config: PairedGenerationConfig,
        reason: str,
        *,
        journal: RunJournal | None = None,
    ) -> PairedGenerationSnapshot:
        invalid = store.put(
            "PairedGenerationInvalidation",
            "1.0.0",
            {
                "campaign_id": config.campaign_id,
                "candidate_freeze_hash": config.candidate_freeze_hash,
                "evaluation_dataset_hash": config.evaluation_dataset_hash,
                "reason_code": reason,
                "status": "invalid",
            },
        )
        ref = Path(root) / "paired-generations" / config.campaign_id / "invalid.ref"
        try:
            ArtifactStore._publish(ref, f"{invalid.content_hash}\n".encode("ascii"))
        except ImmutableArtifactConflict:
            pass
        readiness = store.put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [{"code": reason, "status": "invalid"}],
                "execution_profile": "fixture",
                "phase": "FINAL_EVAL",
                "side_effects_permitted": False,
                "status": "invalid",
            },
        )
        if journal is not None:
            try:
                events = journal.events()
                if events and events[-1].payload.get("event_type") != "RUN_CLOSED":
                    journal.append(
                        config.controller_epoch, "CAMPAIGN_INVALID", {"invalidation_hash": invalid.content_hash}
                    )
                    events = journal.events()
                    journal.close(
                        config.controller_epoch,
                        status="failed",
                        reason_code=reason,
                        expected_sequence=len(events) + 1,
                        expected_previous_hash=events[-1].content_hash,
                    )
            except RunJournalError as error:
                raise PairedGenerationFencingError("PAIRED_GENERATION_INVALIDATION_NOT_TERMINAL") from error
        return PairedGenerationSnapshot(None, None, readiness, "invalid", reason)


# Naming aliases keep the public seam discoverable to Ticket 31 and fixture users.
PairedGeneration = PairedGenerationWorkflow
PairedEvaluationWorkflow = PairedGenerationWorkflow
PairedGenerationProvider = FixturePairedGenerationProvider
SealedABMapping = SealedMapping
PairedGenerationInput = PairedGenerationConfig
FixtureGenerationProvider = FixturePairedGenerationProvider
FixtureEvaluationGenerationProvider = FixturePairedGenerationProvider
FinalEvaluationGenerationWorkflow = PairedGenerationWorkflow
