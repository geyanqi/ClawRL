"""Ticket 29: durable, unseen ``eval_only`` Future-100 collection.

The collector is deliberately a small application boundary: rows are supplied by a
fixture source (or by a production adapter after its readiness gate), while all
selection semantics are reconstructed from the preregistered protocol and the
immutable CandidateFreeze.  The published EvaluationDataset is content addressed
and never has a DatasetVersion-shaped payload, so the normal RL loader cannot
consume it accidentally.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from clawrl.artifacts import (
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    ImmutableArtifactConflict,
    JsonValue,
    canonical_json_bytes,
    sha256_hex,
)
from clawrl.evaluation.protocol_workflow import FinalEvaluationProtocolRegistry
from clawrl.training.run_journal import RunJournal, RunJournalError

_HASH = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")


class FutureDatasetError(RuntimeError):
    """Collection cannot be safely certified."""


class FutureDatasetReadinessError(FutureDatasetError):
    """A source, time window, or governance boundary is unavailable."""


class FutureDatasetInvalid(FutureDatasetError):
    """The frozen campaign was deterministically invalidated."""


class FutureDatasetConflictError(FutureDatasetError):
    """An immutable campaign was called with different input."""


class FutureDatasetFencingError(FutureDatasetError):
    """A stale controller attempted a durable transition."""


def _safe_id(value: object, name: str) -> str:
    if type(value) is not str or _ID.fullmatch(cast(str, value)) is None:
        raise FutureDatasetError(f"{name} is invalid")
    return cast(str, value)


def _hash(value: object, name: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise FutureDatasetError(f"{name} must be a SHA-256 digest")
    return cast(str, value)


def _utc(value: object, name: str) -> tuple[_dt.datetime, str]:
    if type(value) is not str:
        raise FutureDatasetError(f"{name} must be an RFC3339 UTC instant")
    text = cast(str, value)
    # Accept an explicit UTC offset at the adapter boundary, then persist only
    # the canonical Z representation.  Naive local times are never guessed.
    try:
        parsed = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise FutureDatasetError(f"{name} is not a valid timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != _dt.timedelta(0):
        raise FutureDatasetError(f"{name} must be UTC")
    parsed = parsed.astimezone(_dt.UTC)
    rendered = parsed.isoformat(timespec="microseconds").replace("+00:00", "Z").rstrip("0").rstrip(".")
    if rendered.endswith("T00:00:00Z") or _UTC.fullmatch(rendered):
        return parsed, rendered
    raise FutureDatasetError(f"{name} has unsupported precision")


def normalize_prompt(prompt: object) -> str:
    if type(prompt) is not str or not prompt:
        raise FutureDatasetError("prompt is invalid")
    return unicodedata.normalize("NFKC", cast(str, prompt).replace("\r\n", "\n")).strip()


def prompt_identity_hash(prompt: object) -> str:
    normalized = normalize_prompt(prompt)
    return sha256_hex(canonical_json_bytes({"domain": "prompt-identity/1.0.0", "prompt": normalized}))


def load_evaluation_dataset(store: ArtifactStore, dataset_hash: str) -> Artifact:
    """Resolve an immutable Future-100 artifact without ever widening it to RL data."""

    _hash(dataset_hash, "evaluation dataset hash")
    try:
        dataset = store.read(dataset_hash, expected_schema_name="EvaluationDataset")
    except Exception as error:
        raise FutureDatasetError("EvaluationDataset cannot be resolved") from error
    payload = dataset.payload
    rows = payload.get("prompt_rows")
    if (
        dataset.schema_version != "1.0.0"
        or payload.get("status") != "immutable"
        or payload.get("purpose") != "eval_only"
    ):
        raise FutureDatasetError("EvaluationDataset is not immutable eval_only data")
    if not isinstance(rows, list) or len(rows) != 100:
        raise FutureDatasetError("EvaluationDataset must contain exactly 100 rows")
    identities: set[str] = set()
    provider_ids: set[str] = set()
    t0_dt, _ = _utc(payload.get("t0_utc"), "t0_utc")
    window = payload.get("window")
    if not isinstance(window, dict):
        raise FutureDatasetError("EvaluationDataset window is invalid")
    end_key = "extension_end_utc" if window.get("extension_used") is True else "initial_end_utc"
    end_dt, _ = _utc(window.get(end_key), end_key)
    for item in rows:
        if not isinstance(item, dict) or set(item) != {
            "event_time_utc",
            "ingestion_time_utc",
            "prompt",
            "prompt_identity_hash",
            "provider_row_id",
        }:
            raise FutureDatasetError("EvaluationDataset row schema is invalid")
        observed = item.get("prompt_identity_hash")
        expected = prompt_identity_hash(item.get("prompt"))
        if observed != expected or expected in identities:
            raise FutureDatasetError("EvaluationDataset identities are invalid or duplicated")
        provider_id = item.get("provider_row_id")
        if not isinstance(provider_id, str) or provider_id in provider_ids:
            raise FutureDatasetError("EvaluationDataset provider row identities are invalid or duplicated")
        event_dt, _ = _utc(item.get("event_time_utc"), "event_time_utc")
        ingestion_dt, _ = _utc(item.get("ingestion_time_utc"), "ingestion_time_utc")
        if not (event_dt > t0_dt and ingestion_dt > t0_dt and event_dt <= end_dt and ingestion_dt <= end_dt):
            raise FutureDatasetError("EvaluationDataset row is outside the frozen time window")
        identities.add(expected)
        provider_ids.add(provider_id)
    _hash(payload.get("candidate_freeze_hash"), "candidate_freeze_hash")
    _hash(payload.get("protocol_hash"), "protocol_hash")
    _hash(payload.get("identity_normalizer_hash"), "identity_normalizer_hash")
    _hash(payload.get("source_contract_hash"), "source_contract_hash")
    _hash(payload.get("governance_hash"), "governance_hash")
    _hash(payload.get("window_hash"), "window_hash")
    return dataset


@dataclass(frozen=True, slots=True)
class FuturePrompt:
    """Public, already-sanitized source row accepted by the fixture adapter."""

    provider_row_id: str
    prompt: str
    difficulty: str
    purpose: Literal["eval_only"]
    event_time_utc: str
    ingestion_time_utc: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FuturePrompt:
        required = {"provider_row_id", "prompt", "difficulty", "purpose", "event_time_utc", "ingestion_time_utc"}
        if set(value) < required:
            raise FutureDatasetError("future source row is missing required fields")
        _safe_id(value.get("provider_row_id"), "provider_row_id")
        _, event = _utc(value.get("event_time_utc"), "event_time_utc")
        _, ingestion = _utc(value.get("ingestion_time_utc"), "ingestion_time_utc")
        prompt = normalize_prompt(value.get("prompt"))
        if value.get("difficulty") != "hard" or value.get("purpose") != "eval_only":
            raise FutureDatasetError("future row does not satisfy the frozen predicate")
        observed_identity = value.get("prompt_identity_hash")
        identity = prompt_identity_hash(prompt)
        if observed_identity is not None and observed_identity != identity:
            raise FutureDatasetError("prompt identity hash does not match frozen normalizer")
        return cls(cast(str, value["provider_row_id"]), prompt, "hard", "eval_only", event, ingestion)

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "event_time_utc": self.event_time_utc,
            "ingestion_time_utc": self.ingestion_time_utc,
            "prompt": self.prompt,
            "prompt_identity_hash": prompt_identity_hash(self.prompt),
            "provider_row_id": self.provider_row_id,
        }


@dataclass(frozen=True, slots=True)
class Future100DatasetConfig:
    campaign_id: str
    candidate_freeze_hash: str
    protocol_hash: str | None = None
    controller_epoch: int = 1
    execution_profile: Literal["fixture", "production"] = "fixture"
    source_contract_hash: str | None = None
    governance_approval_hash: str | None = None
    initial_rows: tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    extension_rows: tuple[Mapping[str, object], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _safe_id(self.campaign_id, "campaign_id")
        _hash(self.candidate_freeze_hash, "candidate_freeze_hash")
        if self.protocol_hash is not None:
            _hash(self.protocol_hash, "protocol_hash")
        else:
            raise FutureDatasetError("protocol_hash is required")
        if self.source_contract_hash is not None:
            _hash(self.source_contract_hash, "source_contract_hash")
        if self.governance_approval_hash is not None:
            _hash(self.governance_approval_hash, "governance_approval_hash")
        if type(self.controller_epoch) is not int or self.controller_epoch <= 0:
            raise FutureDatasetError("controller_epoch is invalid")
        if self.execution_profile not in {"fixture", "production"}:
            raise FutureDatasetError("execution_profile is invalid")


@dataclass(frozen=True, slots=True)
class FutureDatasetSnapshot:
    dataset: Artifact | None
    readiness: Artifact
    status: Literal["collecting", "committed", "invalid", "blocked"]
    reason_code: str | None = None

    @property
    def evaluation_dataset(self) -> Artifact | None:
        return self.dataset


def _row_sort_key(row: FuturePrompt) -> bytes:
    return canonical_json_bytes(row.artifact_payload())


class FutureEvaluationDatasetWorkflow:
    """Collect and atomically publish one immutable Future-100 dataset."""

    @staticmethod
    def production_readiness(root: str | Path, config: object | None = None) -> Artifact:
        checks = [
            {"code": "FINAL_EVAL_DATA_SOURCE_UNAVAILABLE", "status": "blocked"},
            {"code": "FINAL_EVAL_WINDOW_UNVERIFIED", "status": "blocked"},
            {"code": "FINAL_EVAL_GOVERNANCE_APPROVAL_UNAVAILABLE", "status": "blocked"},
        ]
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": checks,
                "execution_profile": "production",
                "phase": "FINAL_EVAL",
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: Future100DatasetConfig,
        initial_rows: Sequence[Mapping[str, object]] | None = None,
        extension_rows: Sequence[Mapping[str, object]] | None = None,
        source_rows: Sequence[Mapping[str, object]] | None = None,
        extension_source_rows: Sequence[Mapping[str, object]] | None = None,
    ) -> FutureDatasetSnapshot:
        if config.execution_profile == "production":
            return FutureDatasetSnapshot(None, cls.production_readiness(root, config), "blocked")
        store = ArtifactStore(root)
        ref = Path(root) / "evaluation-datasets" / config.campaign_id / "active.ref"
        invalid_ref = ref.parent / "invalid.ref"
        if ref.exists():
            existing = cls._read_ref(store, ref)
            replay_rows = initial_rows if initial_rows is not None else source_rows
            replay_extension = extension_rows if extension_rows is not None else extension_source_rows
            cls._validate_existing(existing, config, initial_rows=replay_rows, extension_rows=replay_extension)
            cls._repair_journal(root, store, existing, config.controller_epoch)
            return FutureDatasetSnapshot(existing, cls._green_readiness(store, existing), "committed")
        if invalid_ref.exists():
            try:
                invalid = store.read(
                    invalid_ref.read_text(encoding="ascii").strip(),
                    expected_schema_name="EvaluationCampaignInvalidation",
                )
            except Exception as error:
                raise FutureDatasetError("evaluation campaign invalidation is corrupt") from error
            if (
                invalid.payload.get("campaign_id") != config.campaign_id
                or invalid.payload.get("status") != "invalid"
                or invalid.payload.get("candidate_freeze_hash") != config.candidate_freeze_hash
            ):
                raise FutureDatasetConflictError("evaluation campaign invalidation identity changed")
            readiness = store.put(
                "ReadinessReport",
                "1.0.0",
                {
                    "checks": [{"code": invalid.payload.get("reason_code", "CAMPAIGN_INVALID"), "status": "invalid"}],
                    "execution_profile": "fixture",
                    "phase": "FINAL_EVAL",
                    "side_effects_permitted": False,
                    "status": "invalid",
                },
            )
            return FutureDatasetSnapshot(None, readiness, "invalid", cast(str, invalid.payload.get("reason_code")))

        freeze = cls._load_freeze(store, config)
        protocol = FinalEvaluationProtocolRegistry.load(root, campaign_id=cast(str, freeze.payload["campaign_id"]))
        if freeze.payload.get("protocol_hash") != protocol.protocol.content_hash:
            raise FutureDatasetReadinessError("CANDIDATE_FREEZE_PROTOCOL_LINEAGE_MISMATCH")
        if config.protocol_hash is not None and config.protocol_hash != protocol.protocol.content_hash:
            raise FutureDatasetConflictError("protocol hash differs from CandidateFreeze")
        rows = tuple(
            initial_rows
            if initial_rows is not None
            else source_rows
            if source_rows is not None
            else config.initial_rows
        )
        extension = tuple(
            extension_rows
            if extension_rows is not None
            else extension_source_rows
            if extension_source_rows is not None
            else config.extension_rows
        )
        source_rows_hash = sha256_hex(
            canonical_json_bytes({"extension_rows": list(extension), "initial_rows": list(rows)})
        )
        # Source/governance hashes are mandatory in production-shaped records;
        # fixture uses deterministic synthetic contracts published below.
        source_contract = cls._contract_artifact(store, config.source_contract_hash, "FutureEvalDataSourceContract")
        governance = cls._contract_artifact(store, config.governance_approval_hash, "FinalEvalGovernanceApproval")
        t0, window, normalizer_hash, sample_seed = cls._frozen_semantics(store, freeze, protocol)
        input_artifact = store.put(
            "Future100DatasetInput",
            "1.0.0",
            {
                "campaign_id": config.campaign_id,
                "candidate_freeze_hash": freeze.content_hash,
                "governance_hash": governance.content_hash,
                "protocol_hash": protocol.protocol.content_hash,
                "source_contract_hash": source_contract.content_hash,
                "controller_epoch": config.controller_epoch,
                "schema_version": "future-100-input/1.0.0",
                "source_rows_hash": source_rows_hash,
                "window": window,
            },
        )
        # CandidateFreeze owns the campaign-id journal; this phase has its own
        # durable writer identity so a completed freeze cannot fence collection.
        journal_id = "future_eval_" + re.sub(r"[^A-Za-z0-9._-]", "_", config.campaign_id)
        journal = RunJournal(root, store, journal_id, input_schema_name="Future100DatasetInput")
        try:
            journal.reserve_identity(input_artifact.content_hash)
            events = journal.events()
            if not events:
                journal.start_run(
                    config.controller_epoch,
                    input_artifact.content_hash,
                    {"input_hash": input_artifact.content_hash, "phase": "FINAL_EVAL"},
                )
                journal.append(
                    config.controller_epoch, "READINESS_GREEN", {"protocol_hash": protocol.protocol.content_hash}
                )
        except (RunJournalError, OSError) as error:
            raise FutureDatasetFencingError("FUTURE_DATASET_FENCING_UNAVAILABLE") from error

        try:
            first = cls._eligible(rows, t0, cast(str, window["initial_end_utc"]), t0, normalizer_hash)
            union = list(first)
            extended = False
            if len({prompt_identity_hash(item.prompt) for item in union}) < 100:
                extended = True
                second = cls._eligible(
                    extension,
                    t0,
                    cast(str, window["extension_end_utc"]),
                    cast(str, window["initial_end_utc"]),
                    normalizer_hash,
                )
                union.extend(second)
            unique = cls._dedupe(union)
            if len(unique) < 100:
                return cls._invalidate(
                    root, store, journal, config, input_artifact, "FUTURE_EVAL_UNDER_100_AFTER_SINGLE_EXTENSION"
                )
            unique = cls._exclude_development(store, unique, freeze)
            if len(unique) < 100:
                return cls._invalidate(
                    root, store, journal, config, input_artifact, "FUTURE_EVAL_IDENTITY_EXCLUSION_UNDER_100"
                )
            selected = sorted(
                unique,
                key=lambda row: hashlib.sha256(
                    canonical_json_bytes({"seed": sample_seed, "identity": prompt_identity_hash(row.prompt)})
                ).digest(),
            )[:100]
            payload: dict[str, JsonValue] = {
                "campaign_id": config.campaign_id,
                "candidate_freeze_hash": freeze.content_hash,
                "identity_normalizer_hash": normalizer_hash,
                "input_hash": input_artifact.content_hash,
                "predicate": {"difficulty": "hard", "purpose": "eval_only", "version": "future-hard-predicate/1.0.0"},
                "prompt_rows": [cast(JsonValue, row.artifact_payload()) for row in selected],
                "purpose": "eval_only",
                "sample_seed": sample_seed,
                "schema_version": "evaluation-dataset/1.0.0",
                "source_contract_hash": source_contract.content_hash,
                "governance_hash": governance.content_hash,
                "status": "immutable",
                "t0_utc": t0,
                "window": cast(JsonValue, {**window, "extension_used": extended}),
                "window_hash": sha256_hex(canonical_json_bytes(window)),
                "protocol_hash": protocol.protocol.content_hash,
            }
            dataset = store.put("EvaluationDataset", "1.0.0", payload)
            try:
                ArtifactStore._publish(ref, f"{dataset.content_hash}\n".encode("ascii"))
            except ImmutableArtifactConflict:
                winner = cls._read_ref(store, ref)
                cls._validate_existing(winner, config)
                dataset = winner
            events = journal.events()
            if not any(event.payload.get("event_type") == "EVALUATION_DATASET_COMMITTED" for event in events):
                journal.append(
                    config.controller_epoch, "EVALUATION_DATASET_COMMITTED", {"dataset_hash": dataset.content_hash}
                )
            events = journal.events()
            if events[-1].payload.get("event_type") != "RUN_CLOSED":
                journal.close(
                    config.controller_epoch,
                    status="succeeded",
                    reason_code="EVALUATION_DATASET_COMMITTED",
                    expected_sequence=len(events) + 1,
                    expected_previous_hash=events[-1].content_hash,
                )
            return FutureDatasetSnapshot(dataset, cls._green_readiness(store, dataset), "committed")
        except FutureDatasetInvalid:
            raise
        except (FutureDatasetError, ArtifactCorruption, KeyError, TypeError, ValueError) as error:
            raise FutureDatasetError("FUTURE_EVAL_COLLECTION_FAILED_CLOSED") from error

    collect = run
    build = run
    create = run

    @classmethod
    def resume(
        cls, root: str | Path, campaign_id: str, *, config: Future100DatasetConfig | None = None
    ) -> FutureDatasetSnapshot:
        _safe_id(campaign_id, "campaign_id")
        store = ArtifactStore(root)
        ref = Path(root) / "evaluation-datasets" / campaign_id / "active.ref"
        if not ref.exists():
            raise FutureDatasetError("campaign has no durable EvaluationDataset")
        dataset = cls._read_ref(store, ref)
        if config is not None:
            cls._validate_existing(dataset, config)
            cls._repair_journal(root, store, dataset, config.controller_epoch)
        return FutureDatasetSnapshot(dataset, cls._green_readiness(store, dataset), "committed")

    @staticmethod
    def _repair_journal(root: str | Path, store: ArtifactStore, dataset: Artifact, epoch: int) -> None:
        """Finish a journal left open after the immutable dataset ref was published."""

        input_hash = dataset.payload.get("input_hash")
        if not isinstance(input_hash, str) or _HASH.fullmatch(input_hash) is None:
            raise FutureDatasetFencingError("FUTURE_DATASET_INPUT_LINEAGE_MISSING")
        try:
            input_artifact = store.read(input_hash, expected_schema_name="Future100DatasetInput")
            if input_artifact.payload.get("candidate_freeze_hash") != dataset.payload.get("candidate_freeze_hash"):
                raise FutureDatasetFencingError("FUTURE_DATASET_INPUT_LINEAGE_MISMATCH")
            journal_id = "future_eval_" + re.sub(r"[^A-Za-z0-9._-]", "_", cast(str, dataset.payload.get("campaign_id")))
            journal = RunJournal(root, store, journal_id, input_schema_name="Future100DatasetInput")
            journal.reserve_identity(input_hash)
            events = journal.events()
            if not events:
                journal.start_run(epoch, input_hash, {"input_hash": input_hash, "phase": "FINAL_EVAL"})
                journal.append(epoch, "READINESS_GREEN", {"protocol_hash": dataset.payload.get("protocol_hash")})
                events = journal.events()
            if not any(event.payload.get("event_type") == "EVALUATION_DATASET_COMMITTED" for event in events):
                journal.append(epoch, "EVALUATION_DATASET_COMMITTED", {"dataset_hash": dataset.content_hash})
                events = journal.events()
            if events and events[-1].payload.get("event_type") != "RUN_CLOSED":
                journal.close(
                    epoch,
                    status="succeeded",
                    reason_code="EVALUATION_DATASET_COMMITTED",
                    expected_sequence=len(events) + 1,
                    expected_previous_hash=events[-1].content_hash,
                )
        except (RunJournalError, ArtifactCorruption, OSError) as error:
            raise FutureDatasetFencingError("FUTURE_DATASET_JOURNAL_RECOVERY_FAILED") from error

    @staticmethod
    def _contract_artifact(store: ArtifactStore, value: str | None, schema: str) -> Artifact:
        if value is not None:
            _hash(value, schema)
            try:
                artifact = store.read(value, expected_schema_name=schema)
                if artifact.payload.get("status") not in {"approved", "fixture-approved"}:
                    raise FutureDatasetReadinessError(f"{schema}_UNAVAILABLE")
                return artifact
            except ArtifactCorruption as error:
                raise FutureDatasetReadinessError(f"{schema}_UNAVAILABLE") from error
        return store.put(schema, "1.0.0", {"status": "fixture-approved", "schema_version": schema + "/1.0.0"})

    @staticmethod
    def _load_freeze(store: ArtifactStore, config: Future100DatasetConfig) -> Artifact:
        try:
            freeze = store.read(config.candidate_freeze_hash, expected_schema_name="CandidateFreeze")
        except Exception as error:
            raise FutureDatasetReadinessError("CANDIDATE_FREEZE_UNAVAILABLE") from error
        if freeze.payload.get("status") != "immutable" or freeze.payload.get("campaign_id") != config.campaign_id:
            raise FutureDatasetReadinessError("CANDIDATE_FREEZE_INVALID")
        active_ref = store.root / "candidate-freeze" / config.campaign_id / "active.ref"
        if not active_ref.exists() or active_ref.read_text(encoding="ascii").strip() != freeze.content_hash:
            raise FutureDatasetReadinessError("CANDIDATE_FREEZE_NOT_DURABLY_ACTIVE")
        return freeze

    @staticmethod
    def _frozen_semantics(store: ArtifactStore, freeze: Artifact, protocol) -> tuple[str, dict[str, object], str, int]:
        t0 = cast(str, freeze.payload.get("t0_utc"))
        t0_dt, t0 = _utc(t0, "CandidateFreeze.t0_utc")
        window_payload = cast(dict[str, object], protocol.protocol.payload.get("window", {}))
        duration = window_payload.get("initial_duration_seconds")
        if type(duration) is not int or duration <= 0 or window_payload.get("extension_count") != 1:
            raise FutureDatasetReadinessError("FROZEN_WINDOW_UNAVAILABLE")
        end = t0_dt + _dt.timedelta(seconds=duration)
        extension = end + _dt.timedelta(seconds=duration)
        identity = cast(dict[str, object], protocol.protocol.payload.get("identity_exclusion", {}))
        normalizer_hash = identity.get("normalizer_hash")
        _hash(normalizer_hash, "identity_normalizer_hash")
        try:
            normalizer = store.read(cast(str, normalizer_hash), expected_schema_name="PromptIdentityNormalizer")
        except ArtifactCorruption as error:
            raise FutureDatasetReadinessError("FROZEN_IDENTITY_NORMALIZER_UNAVAILABLE") from error
        if normalizer.payload.get("algorithm_version") != "prompt-identity-nfkc-lf-trim-v1":
            raise FutureDatasetReadinessError("FROZEN_IDENTITY_NORMALIZER_UNSUPPORTED")
        sample = cast(dict[str, object], protocol.protocol.payload.get("sample", {}))
        seed = sample.get("seed")
        if type(seed) is not int:
            raise FutureDatasetReadinessError("FROZEN_SAMPLE_SEED_UNAVAILABLE")
        return (
            t0,
            {
                "initial_end_utc": _utc(end.isoformat().replace("+00:00", "Z"), "window end")[1],
                "extension_end_utc": _utc(extension.isoformat().replace("+00:00", "Z"), "extension end")[1],
                "extension_count": 1,
            },
            cast(str, normalizer_hash),
            seed,
        )

    @staticmethod
    def _eligible(
        rows: Sequence[Mapping[str, object]], t0: str, end: str, lower: str, normalizer_hash: str
    ) -> list[FuturePrompt]:
        t0_dt, _ = _utc(t0, "T0")
        end_dt, _ = _utc(end, "window end")
        lower_dt, _ = _utc(lower, "window start")
        result: list[FuturePrompt] = []
        for raw in rows:
            # Predicate filtering is performed before row validation: an
            # approved source window may contain ordinary/evaluation rows that
            # are simply ineligible, not malformed.
            if isinstance(raw, Mapping):
                required = {
                    "provider_row_id",
                    "prompt",
                    "difficulty",
                    "purpose",
                    "event_time_utc",
                    "ingestion_time_utc",
                }
                if not required <= set(raw):
                    raise FutureDatasetError("future source row is missing required fields")
                if raw.get("difficulty") != "hard" or raw.get("purpose") != "eval_only":
                    continue
            row = raw if isinstance(raw, FuturePrompt) else FuturePrompt.from_mapping(raw)
            if row.difficulty != "hard" or row.purpose != "eval_only":
                continue
            event_dt, event = _utc(row.event_time_utc, "event_time_utc")
            ingestion_dt, ingestion = _utc(row.ingestion_time_utc, "ingestion_time_utc")
            if not (
                event_dt > t0_dt
                and ingestion_dt > t0_dt
                and event_dt > lower_dt
                and ingestion_dt > lower_dt
                and event_dt <= end_dt
                and ingestion_dt <= end_dt
            ):
                continue
            result.append(FuturePrompt(row.provider_row_id, row.prompt, row.difficulty, row.purpose, event, ingestion))
        return result

    @staticmethod
    def _dedupe(rows: Sequence[FuturePrompt]) -> list[FuturePrompt]:
        by_identity: dict[str, FuturePrompt] = {}
        for row in rows:
            identity = prompt_identity_hash(row.prompt)
            current = by_identity.get(identity)
            if current is None or _row_sort_key(row) < _row_sort_key(current):
                by_identity[identity] = row
        return list(by_identity.values())

    @staticmethod
    def _exclude_development(
        store: ArtifactStore, selected: Sequence[FuturePrompt], freeze: Artifact
    ) -> list[FuturePrompt]:
        excluded: set[str] = set()
        for path in sorted(store.artifact_dir.glob("*.json")):
            try:
                artifact = store.read(path.stem)
            except Exception as error:
                raise FutureDatasetError("ARTIFACT_CORRUPTION_DURING_IDENTITY_EXCLUSION") from error
            if artifact.schema_name != "DatasetVersion":
                continue
            if artifact.payload.get("purpose") not in {"training_allowed", "judge_only", "train", "development"}:
                continue
            refs = artifact.payload.get("trace_refs")
            if not isinstance(refs, list):
                raise FutureDatasetError("DEVELOPMENT_DATASET_LINEAGE_INVALID")
            for ref in refs:
                if not isinstance(ref, dict) or not isinstance(ref.get("artifact_hash"), str):
                    raise FutureDatasetError("DEVELOPMENT_DATASET_TRACE_REF_INVALID")
                try:
                    trace = store.read(cast(str, ref["artifact_hash"]), expected_schema_name="TrainingTrace")
                except Exception as error:
                    raise FutureDatasetError("DEVELOPMENT_DATASET_TRACE_MISSING") from error
                prompt = trace.payload.get("prompt")
                if isinstance(prompt, str):
                    excluded.add(prompt_identity_hash(prompt))
            for key in ("prompt_identity_hashes", "identity_hashes"):
                values = artifact.payload.get(key)
                if isinstance(values, list):
                    excluded.update(value for value in values if isinstance(value, str) and _HASH.fullmatch(value))
        return [row for row in selected if prompt_identity_hash(row.prompt) not in excluded]

    @staticmethod
    def _invalidate(
        root: str | Path,
        store: ArtifactStore,
        journal: RunJournal,
        config: Future100DatasetConfig,
        input_artifact: Artifact,
        reason: str,
    ) -> FutureDatasetSnapshot:
        invalid = store.put(
            "EvaluationCampaignInvalidation",
            "1.0.0",
            {
                "campaign_id": config.campaign_id,
                "candidate_freeze_hash": config.candidate_freeze_hash,
                "input_hash": input_artifact.content_hash,
                "reason_code": reason,
                "status": "invalid",
            },
        )
        invalid_ref = Path(root) / "evaluation-datasets" / config.campaign_id / "invalid.ref"
        try:
            ArtifactStore._publish(invalid_ref, f"{invalid.content_hash}\n".encode("ascii"))
        except ImmutableArtifactConflict:
            existing = store.read(
                invalid_ref.read_text(encoding="ascii").strip(), expected_schema_name="EvaluationCampaignInvalidation"
            )
            if existing.content_hash != invalid.content_hash:
                raise FutureDatasetConflictError("evaluation campaign invalidation conflict") from None
        try:
            events = journal.events()
            journal.append(config.controller_epoch, "CAMPAIGN_INVALID", {"invalidation_hash": invalid.content_hash})
            events = journal.events()
            journal.close(
                config.controller_epoch,
                status="failed",
                reason_code=reason,
                expected_sequence=len(events) + 1,
                expected_previous_hash=events[-1].content_hash,
            )
        except RunJournalError as error:
            raise FutureDatasetFencingError("FUTURE_EVAL_INVALIDATION_COMMIT_FAILED") from error
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
        return FutureDatasetSnapshot(None, readiness, "invalid", reason)

    @staticmethod
    def _validate_existing(
        existing: Artifact,
        config: Future100DatasetConfig,
        *,
        initial_rows: Sequence[Mapping[str, object]] | None = None,
        extension_rows: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        p = existing.payload
        if (
            existing.schema_name != "EvaluationDataset"
            or p.get("status") != "immutable"
            or p.get("purpose") != "eval_only"
            or p.get("campaign_id") != config.campaign_id
        ):
            raise FutureDatasetConflictError("EvaluationDataset is not a valid immutable campaign artifact")
        if p.get("candidate_freeze_hash") != config.candidate_freeze_hash:
            raise FutureDatasetConflictError("EVALUATION_DATASET_ALREADY_BOUND_TO_DIFFERENT_FREEZE")
        if config.protocol_hash is not None and p.get("protocol_hash") != config.protocol_hash:
            raise FutureDatasetConflictError("EVALUATION_DATASET_PROTOCOL_CONFLICT")
        effective_initial = config.initial_rows if initial_rows is None else tuple(initial_rows)
        effective_extension = config.extension_rows if extension_rows is None else tuple(extension_rows)
        if effective_initial or effective_extension:
            input_hash = p.get("input_hash")
            if not isinstance(input_hash, str) or _HASH.fullmatch(input_hash) is None:
                raise FutureDatasetConflictError("EVALUATION_DATASET_INPUT_LINEAGE_MISSING")
            # Input lineage is checked against the durable input artifact.  A
            # caller cannot replace source rows while replaying a campaign.
            try:
                input_artifact = ArtifactStore(existing.path.parent.parent).read(
                    input_hash,
                    expected_schema_name="Future100DatasetInput",
                )
            except Exception as error:
                raise FutureDatasetConflictError("EVALUATION_DATASET_INPUT_LINEAGE_MISSING") from error
            source_rows_hash = sha256_hex(
                canonical_json_bytes(
                    {"extension_rows": list(effective_extension), "initial_rows": list(effective_initial)}
                )
            )
            if input_artifact.payload.get("source_rows_hash") != source_rows_hash:
                raise FutureDatasetConflictError("EVALUATION_DATASET_INPUT_CONFLICT")
        rows = p.get("prompt_rows")
        if (
            not isinstance(rows, list)
            or len(rows) != 100
            or len(
                {prompt_identity_hash(cast(dict[str, object], r).get("prompt")) for r in rows if isinstance(r, dict)}
            )
            != 100
        ):
            raise FutureDatasetConflictError("EVALUATION_DATASET_CARDINALITY_INVALID")

    @staticmethod
    def _read_ref(store: ArtifactStore, ref: Path) -> Artifact:
        try:
            content_hash = ref.read_text(encoding="ascii").strip()
            _hash(content_hash, "dataset ref")
            return load_evaluation_dataset(store, content_hash)
        except Exception as error:
            raise FutureDatasetError("evaluation dataset ref is corrupt") from error

    @staticmethod
    def _green_readiness(store: ArtifactStore, dataset: Artifact) -> Artifact:
        return store.put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [{"code": "FUTURE_EVAL_DATASET_COMMITTED", "status": "green"}],
                "dataset_hash": dataset.content_hash,
                "execution_profile": "fixture",
                "phase": "FINAL_EVAL",
                "side_effects_permitted": True,
                "status": "green",
            },
        )


# Ticket-oriented aliases used by integrations and contract tests.
FutureDatasetConfig = Future100DatasetConfig
EvaluationDatasetConfig = Future100DatasetConfig
FutureEvaluationDatasetConfig = Future100DatasetConfig
FutureDatasetWorkflow = FutureEvaluationDatasetWorkflow
EvaluationDatasetWorkflow = FutureEvaluationDatasetWorkflow
Future100DatasetWorkflow = FutureEvaluationDatasetWorkflow
FutureEvaluationDataset = FutureEvaluationDatasetWorkflow
