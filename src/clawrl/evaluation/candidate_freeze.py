"""Ticket 28: immutable, fail-closed CandidateFreeze and trusted T0.

The implementation deliberately keeps the boundary small.  Everything needed to
reconstruct a freeze is an immutable artifact; the only mutable object is the
content-addressed ``active.ref`` published with an atomic no-replace operation.
Fixture callers may use the deterministic controller clock below.  Production is
always reported blocked and never creates a candidate or submits a run.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from clawrl.artifacts import (
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    ImmutableArtifactConflict,
    JsonValue,
    UnknownSchemaMajor,
)
from clawrl.evaluation.protocol_workflow import FinalEvaluationProtocolError, FinalEvaluationProtocolRegistry
from clawrl.training.run_journal import RunJournal, RunJournalError

_HASH = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")


class CandidateFreezeError(RuntimeError):
    """Candidate freeze cannot be safely certified."""


class CandidateFreezeReadinessError(CandidateFreezeError):
    """One of the 122B inputs or its completion evidence is not ready."""


@dataclass(frozen=True, slots=True)
class FixtureControllerClock:
    """A deterministic trusted clock boundary used by fixture tests."""

    t0_utc: str = "2026-02-01T00:00:00Z"

    def now_utc(self) -> str:
        _parse_utc(self.t0_utc)
        return self.t0_utc


@dataclass(frozen=True, slots=True)
class CandidateFreezeConfig:
    campaign_id: str
    protocol_hash: str
    preregistration_receipt_hash: str
    trained_run_record_hash: str
    base_checkpoint_hash: str
    dataset_version_hash: str
    judge_bundle_hash: str
    experiment_spec_hash: str
    trained_checkpoint_hash: str | None = None
    candidate_id: str = "candidate-122b"
    controller_epoch: int = 1
    t0_utc: str | None = None
    execution_profile: Literal["fixture", "production"] = "fixture"

    def __post_init__(self) -> None:
        for name in ("campaign_id", "candidate_id"):
            if type(getattr(self, name)) is not str or _ID.fullmatch(cast(str, getattr(self, name))) is None:
                raise CandidateFreezeError(f"{name} is invalid")
        for name in (
            "protocol_hash",
            "preregistration_receipt_hash",
            "trained_run_record_hash",
            "base_checkpoint_hash",
            "dataset_version_hash",
            "judge_bundle_hash",
            "experiment_spec_hash",
        ):
            if _HASH.fullmatch(cast(str, getattr(self, name))) is None:
                raise CandidateFreezeError(f"{name} is invalid")
        if self.trained_checkpoint_hash is not None and _HASH.fullmatch(self.trained_checkpoint_hash) is None:
            raise CandidateFreezeError("trained_checkpoint_hash is invalid")
        if type(self.controller_epoch) is not int or self.controller_epoch <= 0:
            raise CandidateFreezeError("controller_epoch is invalid")
        if self.t0_utc is not None:
            _parse_utc(self.t0_utc)
        if self.execution_profile not in {"fixture", "production"}:
            raise CandidateFreezeError("execution_profile is invalid")


@dataclass(frozen=True, slots=True)
class CandidateFreezeSnapshot:
    freeze: Artifact | None
    readiness: Artifact
    t0_utc: str | None = None

    @property
    def candidate_freeze(self) -> Artifact | None:
        return self.freeze


def _parse_utc(value: str) -> _dt.datetime:
    if type(value) is not str or _UTC.fullmatch(value) is None:
        raise CandidateFreezeError("trusted controller T0 must be an RFC3339 UTC instant")
    try:
        parsed = _dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise CandidateFreezeError("trusted controller T0 is not a valid UTC instant") from error
    if parsed.tzinfo is None or parsed.utcoffset() != _dt.timedelta(0):
        raise CandidateFreezeError("trusted controller T0 is not UTC")
    return parsed


def _hash(value: object, name: str) -> str:
    if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
        raise CandidateFreezeReadinessError(f"{name} is not a SHA-256 artifact hash")
    return cast(str, value)


def _nested(payload: dict[str, JsonValue], *keys: str) -> object | None:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


class CandidateFreezeWorkflow:
    """Validate and atomically publish exactly one CandidateFreeze."""

    @staticmethod
    def production_readiness(root: str | Path, config: object | None = None) -> Artifact:
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": code, "status": "blocked"}
                    for code in (
                        "TRUSTED_CONTROLLER_CLOCK_UNAVAILABLE",
                        "CANDIDATE_RUN_REGISTRY_UNAVAILABLE",
                        "CHECKPOINT_STORE_UNAVAILABLE",
                        "EVALUATION_ENVIRONMENT_UNVERIFIED",
                    )
                ],
                "execution_profile": "production",
                "phase": "FINAL_EVAL",
                "side_effects_permitted": False,
                "status": "blocked",
                "freeze_attempted": False,
            },
        )

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: CandidateFreezeConfig,
        clock: FixtureControllerClock | None = None,
    ) -> CandidateFreezeSnapshot:
        if config.execution_profile == "production":
            return CandidateFreezeSnapshot(None, cls.production_readiness(root, config), None)
        store = ArtifactStore(root)
        ref = Path(root) / "candidate-freeze" / config.campaign_id / "active.ref"
        # A terminal freeze is immutable and replay is strictly idempotent.
        if ref.exists():
            existing = cls._read_ref(store, ref)
            cls._validate_existing(existing, config)
            cls._repair_journal(root, store, existing, config)
            return CandidateFreezeSnapshot(
                existing,
                cls._green_readiness(store, existing),
                cast(str, existing.payload["t0_utc"]),
            )

        protocol = cls._validate_protocol(root, config)
        dataset, bundle, spec, run, trained, base = cls._validate_inputs(store, config, protocol)
        source_clock = clock or FixtureControllerClock()
        getter = getattr(source_clock, "now_utc", None) or getattr(source_clock, "now", None)
        if not callable(getter):
            raise CandidateFreezeError("TRUSTED_CONTROLLER_CLOCK_UNAVAILABLE")
        t0 = getter()
        if config.t0_utc is not None and t0 != config.t0_utc:
            raise CandidateFreezeError("TRUSTED_CONTROLLER_T0_MISMATCH")
        _parse_utc(t0)

        # The clock record is itself immutable evidence of the boundary used.
        clock_artifact = store.put(
            "TrustedControllerClock",
            "1.0.0",
            {
                "controller_epoch": config.controller_epoch,
                "source": "fixture-deterministic-controller-clock",
                "status": "trusted",
                "t0_utc": t0,
            },
        )
        input_artifact = store.put(
            "CandidateFreezeInput",
            "1.0.0",
            {
                "base_checkpoint_hash": base.content_hash,
                "campaign_id": config.campaign_id,
                "dataset_version_hash": dataset.content_hash,
                "experiment_spec_hash": spec.content_hash,
                "judge_bundle_hash": bundle.content_hash,
                "protocol_hash": protocol.protocol.content_hash,
                "schema_version": "candidate-freeze-input/1.0.0",
                "t0_clock_hash": clock_artifact.content_hash,
                "trained_checkpoint_hash": trained.content_hash,
                "trained_run_record_hash": run.content_hash,
            },
        )
        # Protocol campaign IDs may contain ':'; the RunJournal filesystem
        # identity is intentionally narrower, so derive a stable safe name.
        journal_id = re.sub(r"[^A-Za-z0-9._-]", "_", config.campaign_id)
        journal = RunJournal(root, store, journal_id, input_schema_name="CandidateFreezeInput")
        try:
            journal.reserve_identity(input_artifact.content_hash)
            events = journal.events()
            if not events:
                journal.start_run(
                    config.controller_epoch,
                    input_artifact.content_hash,
                    {"input_hash": input_artifact.content_hash, "phase": "FINAL_EVAL", "t0_utc": t0},
                )
                journal.append(
                    config.controller_epoch,
                    "READINESS_GREEN",
                    {"protocol_hash": protocol.protocol.content_hash},
                )
        except RunJournalError as error:
            raise CandidateFreezeError("CANDIDATE_FREEZE_FENCING_UNAVAILABLE") from error
        freeze_payload: dict[str, JsonValue] = {
            "base_checkpoint_hash": base.content_hash,
            "candidate_id": config.candidate_id,
            "campaign_id": config.campaign_id,
            "controller_epoch": config.controller_epoch,
            "dataset_version_hash": dataset.content_hash,
            "evaluation_environment_hash": protocol.environment.content_hash,
            "experiment_spec_hash": spec.content_hash,
            "judge_bundle_hash": bundle.content_hash,
            "protocol_hash": protocol.protocol.content_hash,
            "schema_version": "candidate-freeze/1.0.0",
            "status": "immutable",
            "t0_clock_hash": clock_artifact.content_hash,
            "t0_utc": t0,
            "trained_checkpoint_hash": trained.content_hash,
            "trained_run_record_hash": run.content_hash,
        }
        freeze = store.put("CandidateFreeze", "1.0.0", freeze_payload)
        try:
            ArtifactStore._publish(ref, f"{freeze.content_hash}\n".encode("ascii"))
        except ImmutableArtifactConflict:
            # Another fenced controller won.  Re-read and only accept an exact
            # replay; a different candidate is a deterministic conflict.
            winner = cls._read_ref(store, ref)
            cls._validate_existing(winner, config)
            freeze = winner
        try:
            events = journal.events()
            if not any(event.payload.get("event_type") == "CANDIDATE_FREEZE_COMMITTED" for event in events):
                journal.append(
                    config.controller_epoch,
                    "CANDIDATE_FREEZE_COMMITTED",
                    {"freeze_hash": freeze.content_hash},
                )
            events = journal.events()
            if events and events[-1].payload.get("event_type") != "RUN_CLOSED":
                journal.close(
                    config.controller_epoch,
                    status="succeeded",
                    reason_code="CANDIDATE_FREEZE_COMMITTED",
                    expected_sequence=len(events) + 1,
                    expected_previous_hash=events[-1].content_hash,
                )
        except RunJournalError as error:
            raise CandidateFreezeError("CANDIDATE_FREEZE_JOURNAL_COMMIT_FAILED") from error
        readiness = cls._green_readiness(store, freeze)
        return CandidateFreezeSnapshot(freeze, readiness, t0)

    @classmethod
    def freeze(
        cls,
        root: str | Path,
        *,
        config: CandidateFreezeConfig,
        clock: FixtureControllerClock | None = None,
    ) -> Artifact:
        result = cls.run(root, config=config, clock=clock)
        if result.freeze is None:
            raise CandidateFreezeError("production CandidateFreeze is blocked")
        return result.freeze

    # ``build``/``create`` are intentionally aliases: neither can mutate a
    # published CandidateFreeze or select a second candidate.
    build = freeze
    create = freeze

    @classmethod
    def resume(
        cls,
        root: str | Path,
        campaign_id: str,
        *,
        config: CandidateFreezeConfig | None = None,
    ) -> CandidateFreezeSnapshot:
        store = ArtifactStore(root)
        ref = Path(root) / "candidate-freeze" / campaign_id / "active.ref"
        if not ref.exists():
            raise CandidateFreezeError("candidate campaign has no durable freeze")
        freeze = cls._read_ref(store, ref)
        if config is not None:
            cls._validate_existing(freeze, config)
            cls._repair_journal(root, store, freeze, config)
        return CandidateFreezeSnapshot(
            freeze,
            cls._green_readiness(store, freeze),
            cast(str, freeze.payload["t0_utc"]),
        )

    @staticmethod
    def _validate_protocol(root: str | Path, config: CandidateFreezeConfig):
        try:
            authorization = FinalEvaluationProtocolRegistry.authorize_candidate_freeze(
                root,
                campaign_id=config.campaign_id,
                protocol_hash=config.protocol_hash,
                preregistration_receipt_hash=config.preregistration_receipt_hash,
            )
            snapshot = FinalEvaluationProtocolRegistry.load(root, campaign_id=config.campaign_id)
            if authorization.payload.get("protocol_hash") != snapshot.protocol.content_hash:
                raise CandidateFreezeReadinessError("protocol authorization changed")
            return snapshot
        except (FinalEvaluationProtocolError, ArtifactCorruption, CandidateFreezeError) as error:
            if isinstance(error, CandidateFreezeError):
                raise
            raise CandidateFreezeReadinessError("FINAL_EVAL_PROTOCOL_NOT_PREREGISTERED") from error

    @classmethod
    def _validate_inputs(cls, store: ArtifactStore, config: CandidateFreezeConfig, protocol):
        try:
            dataset = store.read(config.dataset_version_hash, expected_schema_name="DatasetVersion")
            bundle = store.read(config.judge_bundle_hash, expected_schema_name="JudgeBundle")
            spec = store.read(config.experiment_spec_hash, expected_schema_name="ExperimentSpec")
            run = store.read(config.trained_run_record_hash, expected_schema_name="RunRecord")
            base = cls._read_checkpoint(store, config.base_checkpoint_hash)
            trained_hash = config.trained_checkpoint_hash or cast(str, run.payload.get("checkpoint_hash"))
            trained = cls._read_checkpoint(store, trained_hash)
        except (ArtifactCorruption, KeyError, TypeError, ValueError) as error:
            raise CandidateFreezeReadinessError("CANDIDATE_FREEZE_ARTIFACT_MISSING") from error
        if run.payload.get("status") != "succeeded" or run.payload.get("phase") != "TRAIN_122B":
            raise CandidateFreezeReadinessError("CANDIDATE_RUN_NOT_COMPLETED")
        # Do not infer a winner from an arbitrary latest artifact.  A campaign
        # may freeze only when one and only one completed 122B run exists for
        # this Dataset/Judge/spec lineage.
        matching_runs: list[str] = []
        for path in sorted(store.artifact_dir.glob("*.json")):
            try:
                candidate = store.read(path.stem, expected_schema_name="RunRecord")
            except (ArtifactCorruption, OSError, UnknownSchemaMajor):
                continue
            payload = candidate.payload
            if (
                payload.get("status") == "succeeded"
                and payload.get("phase") == "TRAIN_122B"
                and payload.get("dataset_version_hash") == dataset.content_hash
                and payload.get("judge_bundle_hash") == bundle.content_hash
                and payload.get("experiment_spec_hash") == spec.content_hash
            ):
                matching_runs.append(candidate.content_hash)
        if len(matching_runs) != 1 or matching_runs[0] != run.content_hash:
            raise CandidateFreezeReadinessError("MULTIPLE_OR_UNREGISTERED_122B_CANDIDATES")
        if run.payload.get("checkpoint_hash") != trained.content_hash:
            raise CandidateFreezeReadinessError("TRAINED_CHECKPOINT_RUN_LINEAGE_MISMATCH")
        if base.content_hash == trained.content_hash:
            raise CandidateFreezeReadinessError("BASE_TRAINED_CHECKPOINT_IDENTITY_MUST_DIFFER")
        if (
            run.payload.get("dataset_version_hash") != dataset.content_hash
            or run.payload.get("judge_bundle_hash") != bundle.content_hash
        ):
            raise CandidateFreezeReadinessError("CANDIDATE_RUN_ARTIFACT_LINEAGE_MISMATCH")
        update_count = run.payload.get("optimizer_update_count")
        if (
            run.payload.get("experiment_spec_hash") != spec.content_hash
            or not isinstance(update_count, int)
            or update_count <= 0
        ):
            raise CandidateFreezeReadinessError("CANDIDATE_RUN_HAS_NO_COMPLETED_UPDATE")
        step_applied_hash = run.payload.get("step_applied_hash")
        if not isinstance(step_applied_hash, str) or _HASH.fullmatch(step_applied_hash) is None:
            raise CandidateFreezeReadinessError("CANDIDATE_STEP_APPLIED_EVIDENCE_MISSING")
        try:
            step_applied = store.read(step_applied_hash, expected_schema_name="StepApplied")
        except (ArtifactCorruption, UnknownSchemaMajor) as error:
            raise CandidateFreezeReadinessError("CANDIDATE_STEP_APPLIED_EVIDENCE_MISSING") from error
        if step_applied.payload.get("checkpoint_hash") != trained.content_hash or step_applied.payload.get(
            "status"
        ) not in {"applied", "committed"}:
            raise CandidateFreezeReadinessError("CANDIDATE_STEP_APPLIED_LINEAGE_MISMATCH")
        if dataset.payload.get("purpose") not in {"training_allowed", "train"}:
            raise CandidateFreezeReadinessError("DATASET_PURPOSE_NOT_TRAINING_ALLOWED")
        if (
            bundle.payload.get("dataset_version_hash") != dataset.content_hash
            or bundle.payload.get("status") != "terminal"
            or bundle.payload.get("certification_phase") != "TRAIN_122B"
            or bundle.payload.get("version") != "122b-recertified/1.0.0"
            or bundle.payload.get("trace_count") != 100
            or bundle.payload.get("fresh_rollout_count_per_trace") != 32
        ):
            raise CandidateFreezeReadinessError("JUDGE_BUNDLE_NOT_COMPLETE")
        coverage_hash = bundle.payload.get("coverage_manifest_hash")
        if not isinstance(coverage_hash, str):
            raise CandidateFreezeReadinessError("JUDGE_BUNDLE_COVERAGE_MISSING")
        try:
            coverage = store.read(coverage_hash, expected_schema_name="RecertificationCoverageManifest")
        except (ArtifactCorruption, UnknownSchemaMajor) as error:
            raise CandidateFreezeReadinessError("JUDGE_BUNDLE_COVERAGE_MISSING") from error
        report_hashes = coverage.payload.get("report_hashes")
        if (
            coverage.payload.get("status") != "complete"
            or coverage.payload.get("dataset_version_hash") != dataset.content_hash
            or coverage.payload.get("trace_count") != 100
            or not isinstance(report_hashes, list)
            or len(report_hashes) != 100
            or any(type(item) is not str or _HASH.fullmatch(cast(str, item)) is None for item in report_hashes)
            or len(set(report_hashes)) != 100
        ):
            raise CandidateFreezeReadinessError("JUDGE_BUNDLE_COVERAGE_INCOMPLETE")
        bundle_reports = bundle.payload.get("report_hashes")
        if bundle_reports != report_hashes:
            raise CandidateFreezeReadinessError("JUDGE_BUNDLE_REPORT_COVERAGE_MISMATCH")
        report_trace_ids: set[str] = set()
        for report_hash in cast(list[object], report_hashes):
            try:
                report = store.read(cast(str, report_hash), expected_schema_name="CertificationReport")
            except (ArtifactCorruption, UnknownSchemaMajor) as error:
                raise CandidateFreezeReadinessError("JUDGE_BUNDLE_REPORT_MISSING") from error
            trace_id = report.payload.get("trace_id")
            if (
                report.payload.get("status") != "certified"
                or report.payload.get("variant") != "success"
                or not isinstance(trace_id, str)
                or trace_id in report_trace_ids
            ):
                raise CandidateFreezeReadinessError("JUDGE_BUNDLE_REPORT_UNCERTIFIED")
            report_trace_ids.add(trace_id)
        if len(report_trace_ids) != 100:
            raise CandidateFreezeReadinessError("JUDGE_BUNDLE_REPORT_TRACE_COVERAGE_INCOMPLETE")
        # Reuse the production-shaped Ticket 27 gate for the complete report,
        # reward-contract, transfer-candidate, and trace-set lineage.  A
        # metadata-only bundle is never sufficient to freeze a candidate.
        try:
            from clawrl.data.validation import load_training_dataset
            from clawrl.training.gated_122b_run import Gated122BRunWorkflow

            loaded_dataset = load_training_dataset(store, dataset.content_hash)
            Gated122BRunWorkflow._validate_bundle(store, loaded_dataset, bundle)
        except Exception as error:
            raise CandidateFreezeReadinessError("JUDGE_BUNDLE_DEEP_RECERTIFICATION_FAILED") from error
        if (
            spec.payload.get("dataset_version_hash") != dataset.content_hash
            or spec.payload.get("judge_bundle_hash") != bundle.content_hash
            or spec.payload.get("status") not in {"approved", "frozen"}
            or spec.payload.get("model_size") != "122B"
            or spec.payload.get("authoring_mode") != "independent"
        ):
            raise CandidateFreezeReadinessError("EXPERIMENT_SPEC_NOT_122B_APPROVED")
        if (
            not isinstance(spec.payload.get("optimizer"), str)
            or not cast(str, spec.payload.get("optimizer"))
            or type(spec.payload.get("learning_rate_micros")) is not int
            or cast(int, spec.payload.get("learning_rate_micros")) <= 0
            or any(
                not isinstance(spec.payload.get(field), dict) or not cast(dict[str, object], spec.payload[field])
                for field in ("parallelism", "resource", "retry", "monitoring")
            )
            or _HASH.fullmatch(cast(str, spec.payload.get("approval_hash", ""))) is None
            or _ID.fullmatch(cast(str, spec.payload.get("approved_by", ""))) is None
            or _ID.fullmatch(cast(str, spec.payload.get("experiment_id", ""))) is None
        ):
            raise CandidateFreezeReadinessError("EXPERIMENT_SPEC_CONFIGURATION_INCOMPLETE")
        approval_hash = spec.payload.get("approval_evidence_hash")
        if not isinstance(approval_hash, str):
            raise CandidateFreezeReadinessError("EXPERIMENT_SPEC_APPROVAL_EVIDENCE_MISSING")
        try:
            approval = store.read(approval_hash, expected_schema_name="ExperimentSpecApproval")
        except (ArtifactCorruption, UnknownSchemaMajor) as error:
            raise CandidateFreezeReadinessError("EXPERIMENT_SPEC_APPROVAL_EVIDENCE_MISSING") from error
        if (
            approval.payload.get("status") != "approved"
            or approval.payload.get("model_size") != "122B"
            or approval.payload.get("authoring_mode") != "independent"
            or approval.payload.get("experiment_id") != spec.payload.get("experiment_id")
            or approval.payload.get("approval_hash") != spec.payload.get("approval_hash")
            or approval.payload.get("approved_by") != spec.payload.get("approved_by")
        ):
            raise CandidateFreezeReadinessError("EXPERIMENT_SPEC_APPROVAL_EVIDENCE_MISMATCH")
        cls._validate_environments(protocol, base, trained)
        return dataset, bundle, spec, run, trained, base

    @staticmethod
    def _read_checkpoint(store: ArtifactStore, value: str) -> Artifact:
        _hash(value, "checkpoint")
        for schema in ("DurableCheckpointManifest", "CheckpointManifest", "Checkpoint", "ModelCheckpoint"):
            try:
                return store.read(value, expected_schema_name=schema)
            except (ArtifactCorruption, UnknownSchemaMajor):
                continue
        raise CandidateFreezeReadinessError("CHECKPOINT_ARTIFACT_MISSING")

    @staticmethod
    def _validate_environments(protocol, base: Artifact, trained: Artifact) -> None:
        expected = protocol.environment.content_hash
        hashes = []
        for item in (base, trained):
            observed = _nested(
                item.payload,
                "evaluation_environment_hash",
                "environment_hash",
                "evaluationEnvironmentHash",
            )
            if observed is not None:
                hashes.append(observed)
                if observed != expected:
                    raise CandidateFreezeReadinessError("EVALUATION_ENVIRONMENT_MISMATCH")
            for key in (
                "semantic_decoding",
                "tool_harness_policy",
                "prompt_wrapper",
                "evaluator_visible_trajectory_schema",
            ):
                if key not in item.payload:
                    raise CandidateFreezeReadinessError("EVALUATION_ENVIRONMENT_SEMANTICS_MISSING")
                if item.payload[key] != protocol.environment.payload.get(key):
                    raise CandidateFreezeReadinessError("EVALUATION_ENVIRONMENT_SEMANTICS_MISMATCH")
        if hashes and len(hashes) != 2:
            raise CandidateFreezeReadinessError("EVALUATION_ENVIRONMENT_MISSING")
        if not hashes:
            raise CandidateFreezeReadinessError("EVALUATION_ENVIRONMENT_MISSING")
        # If checkpoints carry an identity bundle, every semantic field must be
        # equal; only model/checkpoint identity is allowed to differ.
        semantic_keys = (
            "semantic_generation",
            "generation_config",
            "tool_harness_policy",
            "prompt_wrapper",
            "evaluator_visible_trajectory_schema",
        )
        for key in semantic_keys:
            left, right = base.payload.get(key), trained.payload.get(key)
            if left is not None or right is not None:
                if left != right:
                    raise CandidateFreezeReadinessError("BASE_TRAINED_SEMANTIC_ENVIRONMENT_MISMATCH")

    @staticmethod
    def _read_ref(store: ArtifactStore, ref: Path) -> Artifact:
        try:
            value = ref.read_text(encoding="ascii").strip()
            return store.read(value, expected_schema_name="CandidateFreeze")
        except (OSError, ArtifactCorruption) as error:
            raise CandidateFreezeError("CandidateFreeze reference is corrupt") from error

    @classmethod
    def _repair_journal(
        cls,
        root: str | Path,
        store: ArtifactStore,
        freeze: Artifact,
        config: CandidateFreezeConfig,
    ) -> None:
        """Complete a journal left open after the immutable ref won a race."""
        journal_id = re.sub(r"[^A-Za-z0-9._-]", "_", config.campaign_id)
        journal = RunJournal(root, store, journal_id, input_schema_name="CandidateFreezeInput")
        try:
            events = journal.events()
            if events and events[-1].payload.get("event_type") == "RUN_CLOSED":
                return
            input_hash: str | None = None
            input_matches: list[str] = []
            for path in sorted(store.artifact_dir.glob("*.json")):
                try:
                    item = store.read(path.stem, expected_schema_name="CandidateFreezeInput")
                except (ArtifactCorruption, UnknownSchemaMajor, OSError):
                    continue
                if (
                    item.payload.get("campaign_id") == config.campaign_id
                    and item.payload.get("trained_run_record_hash") == freeze.payload.get("trained_run_record_hash")
                    and item.payload.get("t0_clock_hash") == freeze.payload.get("t0_clock_hash")
                    and item.payload.get("trained_checkpoint_hash") == freeze.payload.get("trained_checkpoint_hash")
                ):
                    input_matches.append(item.content_hash)
            if len(input_matches) != 1:
                raise CandidateFreezeError("CANDIDATE_FREEZE_INPUT_MISSING")
            input_hash = input_matches[0]
            journal.reserve_identity(input_hash)
            events = journal.events()
            if not events:
                journal.start_run(
                    config.controller_epoch,
                    input_hash,
                    {"input_hash": input_hash, "phase": "FINAL_EVAL", "t0_utc": freeze.payload["t0_utc"]},
                )
                journal.append(config.controller_epoch, "READINESS_GREEN", {"protocol_hash": config.protocol_hash})
            else:
                # start_run on an existing open journal claims the new fence.
                journal.start_run(config.controller_epoch, input_hash, {"input_hash": input_hash})
            events = journal.events()
            if not any(event.payload.get("event_type") == "CANDIDATE_FREEZE_COMMITTED" for event in events):
                journal.append(
                    config.controller_epoch,
                    "CANDIDATE_FREEZE_COMMITTED",
                    {"freeze_hash": freeze.content_hash},
                )
            events = journal.events()
            journal.close(
                config.controller_epoch,
                status="succeeded",
                reason_code="CANDIDATE_FREEZE_COMMITTED",
                expected_sequence=len(events) + 1,
                expected_previous_hash=events[-1].content_hash,
            )
        except RunJournalError as error:
            raise CandidateFreezeError("CANDIDATE_FREEZE_JOURNAL_RECOVERY_FAILED") from error

    @staticmethod
    def _validate_existing(existing: Artifact, config: CandidateFreezeConfig) -> None:
        p = existing.payload
        if (
            p.get("status") != "immutable"
            or p.get("campaign_id") != config.campaign_id
            or p.get("candidate_id") != config.candidate_id
        ):
            raise CandidateFreezeError("CandidateFreeze is not a valid terminal artifact")
        expected = {
            "protocol_hash": config.protocol_hash,
            "dataset_version_hash": config.dataset_version_hash,
            "judge_bundle_hash": config.judge_bundle_hash,
            "experiment_spec_hash": config.experiment_spec_hash,
            "base_checkpoint_hash": config.base_checkpoint_hash,
            "trained_run_record_hash": config.trained_run_record_hash,
        }
        if any(p.get(k) != v for k, v in expected.items()):
            raise CandidateFreezeError("CANDIDATE_FREEZE_ALREADY_BOUND_TO_DIFFERENT_CANDIDATE")
        if type(p.get("controller_epoch")) is not int or cast(int, p["controller_epoch"]) > config.controller_epoch:
            raise CandidateFreezeError("CANDIDATE_FREEZE_STALE_FENCING_EPOCH")
        if (
            config.trained_checkpoint_hash is not None
            and p.get("trained_checkpoint_hash") != config.trained_checkpoint_hash
        ):
            raise CandidateFreezeError("CANDIDATE_FREEZE_CHECKPOINT_REPLACEMENT_FORBIDDEN")
        if config.t0_utc is not None and p.get("t0_utc") != config.t0_utc:
            raise CandidateFreezeError("CANDIDATE_FREEZE_T0_REPLACEMENT_FORBIDDEN")

    @staticmethod
    def _green_readiness(store: ArtifactStore, freeze: Artifact) -> Artifact:
        return store.put(
            "ReadinessReport",
            "1.0.0",
            {
                "candidate_freeze_hash": freeze.content_hash,
                "checks": [{"code": "CANDIDATE_FREEZE_GREEN", "status": "green"}],
                "execution_profile": "fixture",
                "phase": "FINAL_EVAL",
                "side_effects_permitted": True,
                "status": "green",
            },
        )


# Ticket-oriented aliases retained for integrations and fixture tests.
FixtureCandidateFreezeConfig = CandidateFreezeConfig
CandidateFreezeWorkflowConfig = CandidateFreezeConfig
FixtureCandidateFreezeWorkflow = CandidateFreezeWorkflow
CandidateFreeze = CandidateFreezeWorkflow
TrustedControllerClock = FixtureControllerClock
