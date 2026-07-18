"""Fenced total calibrated-scalar JudgeBundle compiler."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.data.models import DatasetValidationError
from clawrl.data.validation import LoadedTrainingDataset, load_training_dataset, validate_dataset_for_experiment
from clawrl.judge.judge_bundle_models import (
    FixtureExperimentSpecConfig,
    FixtureJudgeBundleConfig,
    JudgeBundleContractError,
    ProductionJudgeBundleConfig,
)
from clawrl.training.run_journal import RunJournal, RunJournalError

_INPUT_SCHEMA = "JudgeBundleCompilerInput"
_HASH = re.compile(r"^[0-9a-f]{64}$")


class JudgeBundleWorkflowError(RuntimeError):
    """A total comparable JudgeBundle cannot safely advance or recertify."""


class UnsupportedRewardStrategyError(JudgeBundleWorkflowError):
    """The requested reward strategy has no v1 semantics."""


@dataclass(frozen=True, slots=True)
class JudgeBundleSnapshot:
    events: tuple[Artifact, ...]
    committed_pack_by_trace: dict[str, Artifact]
    uncertifiable_trace_ids: tuple[str, ...]
    conflict_trace_ids: tuple[str, ...]
    missing_trace_ids: tuple[str, ...]
    coverage_manifest: Artifact | None
    judge_bundle: Artifact | None
    terminal: bool


@dataclass(frozen=True, slots=True)
class _State:
    root: Path
    store: ArtifactStore
    journal: RunJournal
    config: FixtureJudgeBundleConfig
    workflow_input: Artifact
    dataset: LoadedTrainingDataset
    events: tuple[Artifact, ...]
    pack_by_trace: dict[str, Artifact]
    uncertifiable: dict[str, Artifact]
    conflicts: dict[str, Artifact]
    coverage_manifest: Artifact | None
    judge_bundle: Artifact | None


class JudgeBundleWorkflow:
    """Compile exact per-Trace terminal certification into one total bundle."""

    @staticmethod
    def production_readiness(root: str | Path, config: ProductionJudgeBundleConfig) -> Artifact:
        values = (
            ("DATASET_APPROVAL_UNAVAILABLE", config.dataset_approval_hash),
            ("CERTIFICATION_CONTROLLER_UNAVAILABLE", config.certification_controller_approval_hash),
            ("REWARD_SCHEMA_UNAVAILABLE", config.reward_schema_approval_hash),
            ("SCALARIZER_UNAVAILABLE", config.scalarizer_approval_hash),
            ("RL_ALGORITHM_CONTRACT_UNAVAILABLE", config.algorithm_contract_approval_hash),
            ("PERMANENT_TRACE_GOVERNANCE_APPROVAL_UNAVAILABLE", config.permanent_trace_governance_approval_hash),
        )
        checks = [
            {"code": code, "status": "blocked"}
            for code, value in values
            if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None
        ]
        if not checks:
            checks.append({"code": "PRODUCTION_JUDGE_BUNDLE_BOUNDARIES_NOT_CONFIGURED", "status": "blocked"})
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": checks,
                "execution_profile": "production",
                "phase": "JUDGE_CERTIFY",
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )

    @classmethod
    def bootstrap(cls, root: str | Path, config: FixtureJudgeBundleConfig, *, epoch: int) -> JudgeBundleSnapshot:
        if config.aggregation != "calibrated_scalar":
            raise UnsupportedRewardStrategyError("UNSUPPORTED_REWARD_STRATEGY")
        root_path = Path(root)
        store = ArtifactStore(root_path)
        try:
            dataset = load_training_dataset(store, config.dataset_version_hash)
            cls._validate_common_contracts(store, config)
            payload: dict[str, object] = {
                **config.immutable_input_payload,
                "dataset_version_id": dataset.dataset_version.payload["dataset_version_id"],
                "trace_set_hash": dataset.dataset_version.payload["trace_set_hash"],
            }
            input_hash = cls._artifact_hash(_INPUT_SCHEMA, payload)
            journal = RunJournal(
                root_path, store, config.run_id, input_schema_name=_INPUT_SCHEMA, input_schema_version="1.0.0"
            )
            journal.reserve_identity(input_hash)
        except (ArtifactCorruption, DatasetValidationError, JudgeBundleContractError, RunJournalError) as error:
            raise JudgeBundleWorkflowError("JudgeBundle immutable input cannot be verified") from error
        path = store.artifact_dir / f"{input_hash}.json"
        if path.exists():
            persisted = store.read(input_hash, expected_schema_name=_INPUT_SCHEMA)
            if persisted.payload != payload:
                raise JudgeBundleWorkflowError("persisted bundle input conflicts with reserved identity")
            if journal.events():
                return cls._snapshot(cls._load_state(root_path, config.run_id))
        workflow_input = store.put(_INPUT_SCHEMA, "1.0.0", payload)
        journal.start_run(
            epoch,
            workflow_input.content_hash,
            {
                "dataset_version_hash": config.dataset_version_hash,
                "input_hash": workflow_input.content_hash,
                "phase": "judge_bundle_compile",
                "trace_count": 100,
            },
        )
        return cls._snapshot(cls._load_state(root_path, config.run_id))

    @classmethod
    def submit_outcome(cls, root: str | Path, run_id: str, *, outcome_hash: str, epoch: int) -> JudgeBundleSnapshot:
        state = cls._load_state(Path(root), run_id)
        if cls._terminal(state.events) or state.coverage_manifest is not None or state.judge_bundle is not None:
            raise JudgeBundleWorkflowError("terminal bundle compiler cannot accept outcomes")
        try:
            outcome = state.store.read(outcome_hash, expected_schema_name="TraceCertificationOutcome")
            trace_id = cast(str, outcome.payload["trace_id"])
            kind = outcome.payload["kind"]
            artifact_hash = cast(str, outcome.payload["artifact_hash"])
        except (ArtifactCorruption, KeyError, TypeError) as error:
            raise JudgeBundleWorkflowError("terminal certification outcome is invalid") from error
        expected_ids = cls._trace_ids(state.dataset)
        if set(outcome.payload) != {"artifact_hash", "kind", "trace_id"} or trace_id not in expected_ids:
            raise JudgeBundleWorkflowError("outcome trace key is outside the DatasetVersion")
        if kind == "judge_pack":
            artifact = state.store.read(artifact_hash, expected_schema_name="JudgePack")
            cls._validate_pack(state.store, state.dataset, state.config, artifact, trace_id)
        elif kind == "uncertifiable":
            artifact = state.store.read(artifact_hash, expected_schema_name="UncertifiableJudgeReport")
            cls._validate_uncertifiable(state.store, state.dataset, state.config, artifact, trace_id)
        else:
            raise JudgeBundleWorkflowError("terminal outcome kind is unsupported")
        existing = state.pack_by_trace.get(trace_id) or state.uncertifiable.get(trace_id)
        if existing is not None:
            existing_outcome = cls._outcome_event(state.events, trace_id)
            if (
                existing_outcome is not None
                and cast(dict[str, object], existing_outcome.payload["details"])["outcome_hash"] == outcome.content_hash
            ):
                return cls._snapshot(state)
            conflict = state.store.put(
                "JudgePackConflictQuarantine",
                "1.0.0",
                {
                    "committed_artifact_hash": existing.content_hash,
                    "conflicting_artifact_hash": artifact.content_hash,
                    "conflicting_outcome_hash": outcome.content_hash,
                    "reason_code": "TRACE_PACK_HASH_CONFLICT",
                    "status": "quarantined",
                    "trace_id": trace_id,
                },
            )
            state.journal.claim_epoch(epoch)
            cls._append(
                state,
                epoch,
                "PACK_CONFLICT_QUARANTINED",
                {"conflict_hash": conflict.content_hash, "trace_id": trace_id},
            )
            return cls._snapshot(cls._load_state(state.root, run_id))
        state.journal.claim_epoch(epoch)
        cls._append(
            state,
            epoch,
            "OUTCOME_COMMITTED",
            {
                "artifact_hash": artifact.content_hash,
                "kind": kind,
                "outcome_hash": outcome.content_hash,
                "trace_id": trace_id,
            },
        )
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def resume(cls, root: str | Path, run_id: str, *, epoch: int) -> JudgeBundleSnapshot:
        state = cls._load_state(Path(root), run_id)
        if cls._terminal(state.events):
            return cls._snapshot(state)
        last_type = state.events[-1].payload.get("event_type")
        if last_type == "JUDGE_BUNDLE_COMMITTED":
            state.journal.claim_epoch(epoch)
            state.journal.close(
                epoch,
                status="succeeded",
                reason_code="TOTAL_JUDGE_BUNDLE_PUBLISHED",
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
            return cls._snapshot(cls._load_state(state.root, run_id))
        if last_type == "BUNDLE_TOTALITY_BLOCKED":
            details = cast(dict[str, object], state.events[-1].payload["details"])
            reason = cast(str, details["reason_code"])
            state.journal.claim_epoch(epoch)
            state.journal.close(
                epoch,
                status="failed",
                reason_code=reason,
                expected_sequence=len(state.events) + 1,
                expected_previous_hash=state.events[-1].content_hash,
            )
            return cls._snapshot(cls._load_state(state.root, run_id))
        if last_type == "COVERAGE_MANIFEST_COMMITTED":
            manifest, bundle = cls._build_bundle(state)
            if state.coverage_manifest is None or state.coverage_manifest.content_hash != manifest.content_hash:
                raise JudgeBundleWorkflowError("incomplete coverage commit changed during recovery")
            state.journal.claim_epoch(epoch)
            bundle_event = cls._append(
                state,
                epoch,
                "JUDGE_BUNDLE_COMMITTED",
                {"judge_bundle_hash": bundle.content_hash},
            )
            state.journal.close(
                epoch,
                status="succeeded",
                reason_code="TOTAL_JUDGE_BUNDLE_PUBLISHED",
                expected_sequence=len(state.events) + 2,
                expected_previous_hash=bundle_event.content_hash,
            )
            return cls._snapshot(cls._load_state(state.root, run_id))
        expected = cls._trace_ids(state.dataset)
        observed = set(state.pack_by_trace) | set(state.uncertifiable)
        if state.conflicts:
            cls._close_blocked(state, epoch, "TRACE_PACK_HASH_CONFLICT")
        elif observed != expected:
            return cls._snapshot(state)
        elif state.uncertifiable:
            cls._close_blocked(state, epoch, "UNCERTIFIABLE_TRACE_BLOCKS_TOTALITY")
        else:
            manifest, bundle = cls._build_bundle(state)
            state.journal.claim_epoch(epoch)
            committed = cls._append(
                state,
                epoch,
                "COVERAGE_MANIFEST_COMMITTED",
                {"coverage_manifest_hash": manifest.content_hash},
            )
            bundle_event = state.journal.append(
                epoch,
                "JUDGE_BUNDLE_COMMITTED",
                {"judge_bundle_hash": bundle.content_hash},
                expected_sequence=len(state.events) + 2,
                expected_previous_hash=committed.content_hash,
            )
            state.journal.close(
                epoch,
                status="succeeded",
                reason_code="TOTAL_JUDGE_BUNDLE_PUBLISHED",
                expected_sequence=len(state.events) + 3,
                expected_previous_hash=bundle_event.content_hash,
            )
        return cls._snapshot(cls._load_state(state.root, run_id))

    @classmethod
    def build_experiment_spec(cls, root: str | Path, config: FixtureExperimentSpecConfig) -> Artifact:
        if config.aggregation != "calibrated_scalar":
            raise UnsupportedRewardStrategyError("UNSUPPORTED_REWARD_STRATEGY")
        store = ArtifactStore(root)
        try:
            dataset = validate_dataset_for_experiment(store, config.dataset_version_hash)
            bundle = store.read(config.judge_bundle_hash, expected_schema_name="JudgeBundle")
            cls._validate_bundle_for_experiment(store, dataset, bundle)
        except (
            ArtifactCorruption,
            DatasetValidationError,
            JudgeBundleContractError,
            JudgeBundleWorkflowError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise JudgeBundleWorkflowError("ExperimentSpec inputs cannot be verified") from error
        if (
            dataset.dataset_version.payload.get("dataset_version_id") != config.dataset_version_id
            or dataset.dataset_version.payload.get("trace_set_hash") != config.trace_set_hash
            or bundle.payload.get("dataset_version_hash") != config.dataset_version_hash
            or bundle.payload.get("dataset_version_id") != config.dataset_version_id
            or bundle.payload.get("trace_set_hash") != config.trace_set_hash
            or bundle.payload.get("aggregation") != "calibrated_scalar"
        ):
            raise JudgeBundleWorkflowError("ExperimentSpec DatasetVersion/JudgeBundle identity mismatch")
        return store.put(
            "ExperimentSpec",
            "1.0.0",
            {
                "aggregation": config.aggregation,
                "dataset_version_hash": config.dataset_version_hash,
                "dataset_version_id": config.dataset_version_id,
                "experiment_id": config.experiment_id,
                "judge_bundle_hash": config.judge_bundle_hash,
                "trace_set_hash": config.trace_set_hash,
            },
        )

    @classmethod
    def _validate_bundle_for_experiment(
        cls, store: ArtifactStore, dataset: LoadedTrainingDataset, bundle: Artifact
    ) -> None:
        bundle_fields = {
            "aggregation",
            "coverage_manifest_hash",
            "dataset_version_hash",
            "dataset_version_id",
            "judge_pack_refs",
            "reward_schema_hash",
            "scalarizer_hash",
            "status",
            "total_hash",
            "trace_count",
            "trace_set_hash",
        }
        if set(bundle.payload) != bundle_fields:
            raise JudgeBundleWorkflowError("ExperimentSpec JudgeBundle fields are incomplete")
        manifest = store.read(
            cast(str, bundle.payload["coverage_manifest_hash"]), expected_schema_name="JudgeBundleCoverageManifest"
        )
        refs = bundle.payload.get("judge_pack_refs")
        expected_trace_ids = sorted(cls._trace_ids(dataset))
        if not isinstance(refs, list) or len(refs) != 100:
            raise JudgeBundleWorkflowError("ExperimentSpec JudgeBundle coverage is not total")
        typed_refs = cast(list[dict[str, object]], refs)
        if any(set(ref) != {"judge_pack_hash", "trace_id"} for ref in typed_refs):
            raise JudgeBundleWorkflowError("ExperimentSpec JudgeBundle refs are malformed")
        observed_trace_ids = [cast(str, ref["trace_id"]) for ref in typed_refs]
        if observed_trace_ids != expected_trace_ids:
            raise JudgeBundleWorkflowError("ExperimentSpec JudgeBundle trace keyset is not exact")
        config = FixtureJudgeBundleConfig(
            run_id="experiment-bundle-recertification",
            dataset_version_hash=cast(str, bundle.payload["dataset_version_hash"]),
            reward_schema_hash=cast(str, bundle.payload["reward_schema_hash"]),
            scalarizer_hash=cast(str, bundle.payload["scalarizer_hash"]),
            algorithm_contract_hash=cast(str, manifest.payload["algorithm_contract_hash"]),
        )
        cls._validate_common_contracts(store, config)
        packs: list[Artifact] = []
        for trace_id, ref in zip(observed_trace_ids, typed_refs, strict=True):
            pack = store.read(cast(str, ref["judge_pack_hash"]), expected_schema_name="JudgePack")
            cls._validate_pack(store, dataset, config, pack, trace_id)
            packs.append(pack)
        golden = sorted(
            cast(str, pack.payload["golden_hard_entry_hash"])
            for pack in packs
            if pack.payload.get("golden_hard_entry_hash") is not None
        )
        total_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "aggregation": "calibrated_scalar",
                    "algorithm_contract_hash": config.algorithm_contract_hash,
                    "dataset_version_hash": config.dataset_version_hash,
                    "domain": "judge-bundle-total/1.0.0",
                    "golden_hard_entry_hashes": golden,
                    "pack_refs": refs,
                    "reward_schema_hash": config.reward_schema_hash,
                    "scalarizer_hash": config.scalarizer_hash,
                }
            )
        )
        expected_manifest: dict[str, object] = {
            "aggregation": "calibrated_scalar",
            "algorithm_contract_hash": config.algorithm_contract_hash,
            "coverage_count": 100,
            "dataset_version_hash": config.dataset_version_hash,
            "golden_hard_entry_hashes": golden,
            "golden_terminal_count": len(golden),
            "judge_pack_refs": refs,
            "reward_schema_hash": config.reward_schema_hash,
            "scalarizer_hash": config.scalarizer_hash,
            "total_hash": total_hash,
            "trace_set_hash": dataset.dataset_version.payload["trace_set_hash"],
        }
        expected_bundle: dict[str, object] = {
            "aggregation": "calibrated_scalar",
            "coverage_manifest_hash": manifest.content_hash,
            "dataset_version_hash": dataset.dataset_version.content_hash,
            "dataset_version_id": dataset.dataset_version.payload["dataset_version_id"],
            "judge_pack_refs": refs,
            "reward_schema_hash": config.reward_schema_hash,
            "scalarizer_hash": config.scalarizer_hash,
            "status": "total",
            "total_hash": total_hash,
            "trace_count": 100,
            "trace_set_hash": dataset.dataset_version.payload["trace_set_hash"],
        }
        if manifest.payload != expected_manifest or bundle.payload != expected_bundle:
            raise JudgeBundleWorkflowError("ExperimentSpec JudgeBundle graph cannot be freshly recertified")

    @classmethod
    def _build_bundle(cls, state: _State) -> tuple[Artifact, Artifact]:
        manifest_payload, bundle_payload = cls._bundle_payloads(state)
        manifest = state.store.put("JudgeBundleCoverageManifest", "1.0.0", manifest_payload)
        if bundle_payload["coverage_manifest_hash"] != manifest.content_hash:
            raise JudgeBundleWorkflowError("coverage manifest hash derivation changed")
        bundle = state.store.put("JudgeBundle", "1.0.0", bundle_payload)
        return manifest, bundle

    @classmethod
    def _bundle_payloads(cls, state: _State) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
        refs = [
            {"judge_pack_hash": state.pack_by_trace[trace_id].content_hash, "trace_id": trace_id}
            for trace_id in sorted(state.pack_by_trace)
        ]
        golden = sorted(
            cast(str, pack.payload["golden_hard_entry_hash"])
            for pack in state.pack_by_trace.values()
            if pack.payload.get("golden_hard_entry_hash") is not None
        )
        total_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "aggregation": "calibrated_scalar",
                    "algorithm_contract_hash": state.config.algorithm_contract_hash,
                    "dataset_version_hash": state.config.dataset_version_hash,
                    "domain": "judge-bundle-total/1.0.0",
                    "golden_hard_entry_hashes": golden,
                    "pack_refs": refs,
                    "reward_schema_hash": state.config.reward_schema_hash,
                    "scalarizer_hash": state.config.scalarizer_hash,
                }
            )
        )
        manifest_payload = cast(
            dict[str, JsonValue],
            {
                "aggregation": "calibrated_scalar",
                "algorithm_contract_hash": state.config.algorithm_contract_hash,
                "coverage_count": 100,
                "dataset_version_hash": state.config.dataset_version_hash,
                "golden_hard_entry_hashes": golden,
                "golden_terminal_count": len(golden),
                "judge_pack_refs": refs,
                "reward_schema_hash": state.config.reward_schema_hash,
                "scalarizer_hash": state.config.scalarizer_hash,
                "total_hash": total_hash,
                "trace_set_hash": state.workflow_input.payload["trace_set_hash"],
            },
        )
        manifest_hash = cls._artifact_hash("JudgeBundleCoverageManifest", cast(dict[str, object], manifest_payload))
        bundle_payload = cast(
            dict[str, JsonValue],
            {
                "aggregation": "calibrated_scalar",
                "coverage_manifest_hash": manifest_hash,
                "dataset_version_hash": state.config.dataset_version_hash,
                "dataset_version_id": state.workflow_input.payload["dataset_version_id"],
                "judge_pack_refs": refs,
                "reward_schema_hash": state.config.reward_schema_hash,
                "scalarizer_hash": state.config.scalarizer_hash,
                "status": "total",
                "total_hash": total_hash,
                "trace_count": 100,
                "trace_set_hash": state.workflow_input.payload["trace_set_hash"],
            },
        )
        return manifest_payload, bundle_payload

    @classmethod
    def _close_blocked(cls, state: _State, epoch: int, reason: str) -> None:
        report = state.store.put("JudgeBundleTotalityFailureReport", "1.0.0", cls._failure_payload(state, reason))
        state.journal.claim_epoch(epoch)
        event = cls._append(
            state,
            epoch,
            "BUNDLE_TOTALITY_BLOCKED",
            {"failure_report_hash": report.content_hash, "reason_code": reason},
        )
        state.journal.close(
            epoch,
            status="failed",
            reason_code=reason,
            expected_sequence=len(state.events) + 2,
            expected_previous_hash=event.content_hash,
        )

    @classmethod
    def _failure_payload(cls, state: _State, reason: str) -> dict[str, JsonValue]:
        return cast(
            dict[str, JsonValue],
            {
                "conflict_trace_ids": sorted(state.conflicts),
                "missing_trace_ids": sorted(
                    cls._trace_ids(state.dataset) - set(state.pack_by_trace) - set(state.uncertifiable)
                ),
                "reason_code": reason,
                "side_effects_permitted": False,
                "status": "blocked",
                "uncertifiable_trace_ids": sorted(state.uncertifiable),
            },
        )

    @classmethod
    def _validate_pack(
        cls,
        store: ArtifactStore,
        dataset: LoadedTrainingDataset,
        config: FixtureJudgeBundleConfig,
        pack: Artifact,
        trace_id: str,
    ) -> None:
        required = {
            "aggregation",
            "algorithm_contract_hash",
            "calibrated_training_authorized",
            "certification_level",
            "certification_mode",
            "dataset_version_hash",
            "dataset_version_id",
            "fit_trajectory_set_hash",
            "golden_hard_entry_hash",
            "items_per_turn",
            "local_tie_groups_diagnostic_only",
            "reward_schema_hash",
            "scalar_comparability",
            "scalarizer_hash",
            "scorer_tier",
            "status",
            "teacher_label_set_hash",
            "terminal_certification_hash",
            "trace_id",
            "training_trace_hash",
        }
        if set(pack.payload) != required:
            raise JudgeBundleWorkflowError("JudgePack fields are not the total compiler contract")
        trace_by_id = {cast(str, item.payload["trace_id"]): item for item in dataset.training_traces}
        comparability = pack.payload.get("scalar_comparability")
        if (
            pack.payload.get("trace_id") != trace_id
            or pack.payload.get("training_trace_hash") != trace_by_id[trace_id].content_hash
            or pack.payload.get("dataset_version_hash") != config.dataset_version_hash
            or pack.payload.get("dataset_version_id") != dataset.dataset_version.payload["dataset_version_id"]
            or pack.payload.get("reward_schema_hash") != config.reward_schema_hash
            or pack.payload.get("scalarizer_hash") != config.scalarizer_hash
            or pack.payload.get("algorithm_contract_hash") != config.algorithm_contract_hash
            or pack.payload.get("aggregation") != "calibrated_scalar"
            or pack.payload.get("calibrated_training_authorized") is not True
            or pack.payload.get("local_tie_groups_diagnostic_only") is not True
            or pack.payload.get("status") != "terminal"
            or not isinstance(comparability, dict)
            or comparability
            != {
                "finite": True,
                "maximum_micros": 100_000_000,
                "minimum_micros": 0,
                "scale_id": "sol-calibrated-scalar-v1",
            }
        ):
            raise JudgeBundleWorkflowError("JudgePack scalar identity is not comparable")
        fit = store.read(cast(str, pack.payload["fit_trajectory_set_hash"]), expected_schema_name="FitTrajectorySet")
        teacher = store.read(cast(str, pack.payload["teacher_label_set_hash"]), expected_schema_name="TeacherLabelSet")
        cls._validate_fit_and_labels(store, fit, teacher, trace_id, config, trace_by_id[trace_id])
        terminal = store.read(
            cast(str, pack.payload["terminal_certification_hash"]), expected_schema_name="TraceCertificationTerminal"
        )
        if terminal.payload != {
            "fit_trajectory_set_hash": fit.content_hash,
            "scorer_tier": pack.payload["scorer_tier"],
            "status": "succeeded",
            "teacher_label_set_hash": teacher.content_hash,
            "trace_id": trace_id,
        }:
            raise JudgeBundleWorkflowError("JudgePack terminal lineage is invalid")
        tier = pack.payload.get("scorer_tier")
        golden_hash = pack.payload.get("golden_hard_entry_hash")
        if tier == "luna":
            if (
                pack.payload.get("certification_mode") != "student_holdout"
                or golden_hash is not None
                or pack.payload.get("certification_level") not in {4, 8, 16, 32}
                or pack.payload.get("items_per_turn") != pack.payload.get("certification_level")
            ):
                raise JudgeBundleWorkflowError("Luna JudgePack mode is invalid")
        elif tier == "sol":
            if (
                pack.payload.get("certification_mode") != "teacher_fallback"
                or type(golden_hash) is not str
                or pack.payload.get("certification_level") != 0
                or pack.payload.get("items_per_turn") != 4
            ):
                raise JudgeBundleWorkflowError("Sol fallback lacks Golden Hard lineage")
            golden = store.read(cast(str, golden_hash), expected_schema_name="GoldenHardEntry")
            if golden.payload != {
                "reason_tags": ["LUNA8_AND_LUNA4_EXHAUSTED", "CALIBRATED_SOL_FALLBACK"],
                "status": "active",
                "trace_id": trace_id,
            }:
                raise JudgeBundleWorkflowError("Golden Hard entry belongs to another trace")
        else:
            raise JudgeBundleWorkflowError("JudgePack scorer tier is unsupported")

    @classmethod
    def _validate_uncertifiable(
        cls,
        store: ArtifactStore,
        dataset: LoadedTrainingDataset,
        config: FixtureJudgeBundleConfig,
        report: Artifact,
        trace_id: str,
    ) -> None:
        if (
            set(report.payload)
            != {
                "blocks_bundle_publication",
                "diagnostic_only",
                "fit_trajectory_set_hash",
                "reason_code",
                "sol_fallback_evidence_hash",
                "status",
                "teacher_label_set_hash",
                "trace_id",
                "training_authorized",
            }
            or report.payload.get("trace_id") != trace_id
            or report.payload.get("status") != "uncertifiable"
            or report.payload.get("blocks_bundle_publication") is not True
            or report.payload.get("training_authorized") is not False
            or report.payload.get("reason_code")
            not in {"SOL_CALIBRATED_SCALAR_UNCERTIFIABLE", "SOL_GROUP_RELATIVE_ONLY_UNSUPPORTED_V1"}
        ):
            raise JudgeBundleWorkflowError("uncertifiable outcome is invalid")
        fit = store.read(cast(str, report.payload["fit_trajectory_set_hash"]), expected_schema_name="FitTrajectorySet")
        teacher = store.read(
            cast(str, report.payload["teacher_label_set_hash"]), expected_schema_name="TeacherLabelSet"
        )
        trace_by_id = {cast(str, item.payload["trace_id"]): item for item in dataset.training_traces}
        cls._validate_fit_and_labels(store, fit, teacher, trace_id, config, trace_by_id[trace_id])
        evidence = store.read(
            cast(str, report.payload["sol_fallback_evidence_hash"]), expected_schema_name="SolFallbackEvidence"
        )
        relative = report.payload["reason_code"] == "SOL_GROUP_RELATIVE_ONLY_UNSUPPORTED_V1"
        expected_scalars: list[JsonValue] = [] if relative else [50_000_000] * 32
        if (
            evidence.payload
            != {
                "all_low_count": 32 if relative else 0,
                "calibrated_scalar_available": not relative,
                "fit_trajectory_set_hash": fit.content_hash,
                "item_count": 32,
                "relative_order_available": relative,
                "scalar_micros": expected_scalars,
                "teacher_label_set_hash": teacher.content_hash,
                "trace_id": trace_id,
            }
            or report.payload.get("diagnostic_only") is not relative
        ):
            raise JudgeBundleWorkflowError("uncertifiable Sol fallback evidence is invalid")

    @staticmethod
    def _validate_fit_and_labels(
        store: ArtifactStore,
        fit: Artifact,
        teacher: Artifact,
        trace_id: str,
        config: FixtureJudgeBundleConfig,
        training_trace: Artifact,
    ) -> None:
        refs = fit.payload.get("trajectory_refs")
        labels = teacher.payload.get("labels")
        if (
            set(fit.payload) != {"dataset_version_hash", "item_count", "trace_id", "trajectory_refs"}
            or set(teacher.payload)
            != {
                "fit_trajectory_set_hash",
                "label_count",
                "labels",
                "reward_schema_hash",
                "scalarizer_hash",
                "trace_id",
            }
            or fit.payload.get("trace_id") != trace_id
            or fit.payload.get("dataset_version_hash") != config.dataset_version_hash
            or fit.payload.get("item_count") != 32
            or not isinstance(refs, list)
            or len(refs) != 32
            or teacher.payload.get("trace_id") != trace_id
            or teacher.payload.get("fit_trajectory_set_hash") != fit.content_hash
            or teacher.payload.get("label_count") != 32
            or teacher.payload.get("reward_schema_hash") != config.reward_schema_hash
            or teacher.payload.get("scalarizer_hash") != config.scalarizer_hash
            or not isinstance(labels, list)
            or len(labels) != 32
        ):
            raise JudgeBundleWorkflowError("fit/teacher cardinality or lineage is invalid")
        manifests: list[str] = []
        contents: list[str] = []
        responses: set[str] = set()
        for index, ref in enumerate(cast(list[dict[str, object]], refs)):
            if set(ref) != {"manifest_hash", "trajectory_index"} or ref.get("trajectory_index") != index:
                raise JudgeBundleWorkflowError("fit trajectory refs are malformed")
            manifest = store.read(cast(str, ref["manifest_hash"]), expected_schema_name="TrajectoryManifest")
            content = store.read(cast(str, manifest.payload["content_hash"]), expected_schema_name="TrajectoryContent")
            response = content.payload.get("response")
            response_hash = content.payload.get("response_hash")
            if (
                set(manifest.payload)
                != {"content_hash", "dataset_version_hash", "split", "trace_id", "trajectory_index"}
                or manifest.payload.get("dataset_version_hash") != config.dataset_version_hash
                or manifest.payload.get("split") != "fit"
                or manifest.payload.get("trace_id") != trace_id
                or manifest.payload.get("trajectory_index") != index
                or set(content.payload) != {"prompt_hash", "response", "response_hash", "trace_id", "trajectory_index"}
                or content.payload.get("trace_id") != trace_id
                or content.payload.get("trajectory_index") != index
                or content.payload.get("prompt_hash")
                != sha256_hex(cast(str, training_trace.payload["prompt"]).encode())
                or type(response) is not str
                or response_hash != sha256_hex(cast(str, response).encode())
                or cast(str, response_hash) in responses
            ):
                raise JudgeBundleWorkflowError("fit trajectories are duplicated or cross-trace")
            manifests.append(manifest.content_hash)
            contents.append(content.content_hash)
            responses.add(cast(str, response_hash))
        scalars: set[int] = set()
        for expected_manifest, expected_content, label in zip(
            manifests, contents, cast(list[dict[str, object]], labels), strict=True
        ):
            value = label.get("scalar_micros")
            if (
                set(label) != {"evidence_hash", "scalar_micros", "trajectory_manifest_hash"}
                or label.get("trajectory_manifest_hash") != expected_manifest
                or type(value) is not int
                or not 0 <= cast(int, value) <= 100_000_000
                or label.get("evidence_hash")
                != sha256_hex(canonical_json_bytes({"content": expected_content, "scalar": value}))
            ):
                raise JudgeBundleWorkflowError("teacher calibrated label is invalid")
            scalars.add(cast(int, value))
        if len(scalars) < 2:
            raise JudgeBundleWorkflowError("constant teacher scalar evidence is not comparable")

    @staticmethod
    def _validate_common_contracts(store: ArtifactStore, config: FixtureJudgeBundleConfig) -> None:
        reward = store.read(config.reward_schema_hash, expected_schema_name="RewardSchema")
        scalarizer = store.read(config.scalarizer_hash, expected_schema_name="Scalarizer")
        algorithm = store.read(config.algorithm_contract_hash, expected_schema_name="RLAlgorithmContract")
        weights = scalarizer.payload.get("dimension_weights_micros")
        if (
            reward.payload.get("aggregation") != "calibrated_scalar"
            or reward.payload.get("minimum_micros") != 0
            or reward.payload.get("maximum_micros") != 100_000_000
            or scalarizer.payload.get("schema_version") != "scalarizer/1.0.0"
            or not isinstance(weights, dict)
            or set(weights) != {"correctness", "reasoning_quality", "task_completion", "tool_discipline"}
            or any(type(value) is not int or value <= 0 for value in weights.values())
            or sum(cast(int, value) for value in weights.values()) != 1_000_000
            or algorithm.payload.get("reward_aggregation") != "calibrated_scalar"
        ):
            raise JudgeBundleWorkflowError("common scalar contracts are incompatible")

    @classmethod
    def _load_state(cls, root: Path, run_id: str) -> _State:
        store = ArtifactStore(root)
        journal = RunJournal(root, store, run_id, input_schema_name=_INPUT_SCHEMA, input_schema_version="1.0.0")
        try:
            journal.verify()
            workflow_input = store.read(journal.reserved_input_hash(), expected_schema_name=_INPUT_SCHEMA)
            fields = set(FixtureJudgeBundleConfig.__dataclass_fields__)  # type: ignore[attr-defined]
            config = FixtureJudgeBundleConfig.from_mapping(
                cast(dict[str, object], {key: value for key, value in workflow_input.payload.items() if key in fields})
            )
            if config.aggregation != "calibrated_scalar":
                raise UnsupportedRewardStrategyError("UNSUPPORTED_REWARD_STRATEGY")
            dataset = load_training_dataset(store, config.dataset_version_hash)
            cls._validate_common_contracts(store, config)
            if (
                workflow_input.payload.get("dataset_version_id")
                != dataset.dataset_version.payload["dataset_version_id"]
                or workflow_input.payload.get("trace_set_hash") != dataset.dataset_version.payload["trace_set_hash"]
            ):
                raise JudgeBundleWorkflowError("compiler DatasetVersion identity changed")
            events = tuple(journal.events())
            cls._validate_event_order(events)
            pack_by_trace: dict[str, Artifact] = {}
            uncertifiable: dict[str, Artifact] = {}
            conflicts: dict[str, Artifact] = {}
            for event in events:
                details = cast(dict[str, object], event.payload["details"])
                if event.payload.get("event_type") == "OUTCOME_COMMITTED":
                    trace_id = cast(str, details["trace_id"])
                    outcome = store.read(
                        cast(str, details["outcome_hash"]), expected_schema_name="TraceCertificationOutcome"
                    )
                    artifact_hash = cast(str, details["artifact_hash"])
                    if set(details) != {"artifact_hash", "kind", "outcome_hash", "trace_id"} or outcome.payload != {
                        "artifact_hash": artifact_hash,
                        "kind": details["kind"],
                        "trace_id": trace_id,
                    }:
                        raise JudgeBundleWorkflowError("committed outcome identity changed")
                    if details["kind"] == "judge_pack":
                        pack = store.read(artifact_hash, expected_schema_name="JudgePack")
                        cls._validate_pack(store, dataset, config, pack, trace_id)
                        pack_by_trace[trace_id] = pack
                    else:
                        report = store.read(artifact_hash, expected_schema_name="UncertifiableJudgeReport")
                        cls._validate_uncertifiable(store, dataset, config, report, trace_id)
                        uncertifiable[trace_id] = report
                elif event.payload.get("event_type") == "PACK_CONFLICT_QUARANTINED":
                    conflict = store.read(
                        cast(str, details["conflict_hash"]), expected_schema_name="JudgePackConflictQuarantine"
                    )
                    trace_id = cast(str, details["trace_id"])
                    committed = pack_by_trace.get(trace_id) or uncertifiable.get(trace_id)
                    if committed is None:
                        raise JudgeBundleWorkflowError("pack conflict precedes its committed outcome")
                    conflicting_outcome = store.read(
                        cast(str, conflict.payload.get("conflicting_outcome_hash")),
                        expected_schema_name="TraceCertificationOutcome",
                    )
                    if (
                        set(details) != {"conflict_hash", "trace_id"}
                        or set(conflict.payload)
                        != {
                            "committed_artifact_hash",
                            "conflicting_artifact_hash",
                            "conflicting_outcome_hash",
                            "reason_code",
                            "status",
                            "trace_id",
                        }
                        or conflict.payload.get("trace_id") != trace_id
                        or conflict.payload.get("committed_artifact_hash") != committed.content_hash
                        or conflict.payload.get("conflicting_artifact_hash")
                        != conflicting_outcome.payload.get("artifact_hash")
                        or conflicting_outcome.payload.get("trace_id") != trace_id
                        or conflict.payload.get("reason_code") != "TRACE_PACK_HASH_CONFLICT"
                        or conflict.payload.get("status") != "quarantined"
                    ):
                        raise JudgeBundleWorkflowError("pack conflict identity changed")
                    conflicts[trace_id] = conflict
            coverage = cls._last_artifact(
                store, events, "COVERAGE_MANIFEST_COMMITTED", "coverage_manifest_hash", "JudgeBundleCoverageManifest"
            )
            bundle = cls._last_artifact(store, events, "JUDGE_BUNDLE_COMMITTED", "judge_bundle_hash", "JudgeBundle")
            state = _State(
                root,
                store,
                journal,
                config,
                workflow_input,
                dataset,
                events,
                pack_by_trace,
                uncertifiable,
                conflicts,
                coverage,
                bundle,
            )
            if bundle is not None and coverage is None:
                raise JudgeBundleWorkflowError("JudgeBundle exists without its coverage manifest")
            if coverage is not None:
                expected_manifest, expected_bundle = cls._bundle_payloads(state)
                if coverage.payload != expected_manifest or (bundle is not None and bundle.payload != expected_bundle):
                    raise JudgeBundleWorkflowError("published JudgeBundle cannot be freshly recertified")
            blocked_events = [event for event in events if event.payload.get("event_type") == "BUNDLE_TOTALITY_BLOCKED"]
            if blocked_events:
                details = cast(dict[str, object], blocked_events[0].payload["details"])
                reason = cast(str, details.get("reason_code"))
                report = store.read(
                    cast(str, details.get("failure_report_hash")),
                    expected_schema_name="JudgeBundleTotalityFailureReport",
                )
                if (
                    set(details) != {"failure_report_hash", "reason_code"}
                    or report.payload != cls._failure_payload(state, reason)
                    or coverage is not None
                    or bundle is not None
                ):
                    raise JudgeBundleWorkflowError("blocked JudgeBundle decision cannot be freshly recertified")
        except (
            ArtifactCorruption,
            DatasetValidationError,
            JudgeBundleContractError,
            JudgeBundleWorkflowError,
            KeyError,
            RunJournalError,
            TypeError,
            ValueError,
        ) as error:
            if isinstance(error, JudgeBundleWorkflowError):
                raise
            raise JudgeBundleWorkflowError("persisted JudgeBundle graph cannot be recertified") from error
        return state

    @staticmethod
    def _validate_event_order(events: tuple[Artifact, ...]) -> None:
        if not events or events[0].payload.get("event_type") != "RUN_STARTED":
            raise JudgeBundleWorkflowError("bundle journal does not start correctly")
        allowed = {
            "RUN_STARTED",
            "OUTCOME_COMMITTED",
            "PACK_CONFLICT_QUARANTINED",
            "BUNDLE_TOTALITY_BLOCKED",
            "COVERAGE_MANIFEST_COMMITTED",
            "JUDGE_BUNDLE_COMMITTED",
            "RUN_CLOSED",
        }
        terminal_marker = False
        phase = "outcomes"
        committed_trace_ids: set[object] = set()
        for index, event in enumerate(events):
            event_type = event.payload.get("event_type")
            if event_type not in allowed:
                raise JudgeBundleWorkflowError("bundle journal contains an unknown event")
            if terminal_marker and event_type != "RUN_CLOSED":
                raise JudgeBundleWorkflowError("event follows bundle terminal marker")
            if event_type == "RUN_STARTED":
                if index != 0:
                    raise JudgeBundleWorkflowError("RUN_STARTED is duplicated")
            elif event_type == "OUTCOME_COMMITTED":
                details = cast(dict[str, object], event.payload["details"])
                trace_id = details.get("trace_id")
                if phase != "outcomes" or trace_id in committed_trace_ids:
                    raise JudgeBundleWorkflowError("trace outcome is duplicated or follows publication")
                committed_trace_ids.add(trace_id)
            elif event_type == "PACK_CONFLICT_QUARANTINED":
                if phase != "outcomes":
                    raise JudgeBundleWorkflowError("pack conflict follows publication")
            elif event_type == "COVERAGE_MANIFEST_COMMITTED":
                if phase != "outcomes":
                    raise JudgeBundleWorkflowError("coverage manifest is duplicated or out of order")
                phase = "coverage"
            elif event_type == "JUDGE_BUNDLE_COMMITTED":
                if phase != "coverage":
                    raise JudgeBundleWorkflowError("JudgeBundle precedes coverage manifest")
                phase = "bundle"
                terminal_marker = True
            elif event_type == "BUNDLE_TOTALITY_BLOCKED":
                if phase != "outcomes":
                    raise JudgeBundleWorkflowError("blocked decision follows publication")
                phase = "blocked"
                terminal_marker = True
            elif event_type == "RUN_CLOSED":
                if not terminal_marker or index != len(events) - 1:
                    raise JudgeBundleWorkflowError("RUN_CLOSED is out of order")

    @staticmethod
    def _trace_ids(dataset: LoadedTrainingDataset) -> set[str]:
        return {cast(str, trace.payload["trace_id"]) for trace in dataset.training_traces}

    @classmethod
    def _snapshot(cls, state: _State) -> JudgeBundleSnapshot:
        missing = cls._trace_ids(state.dataset) - set(state.pack_by_trace) - set(state.uncertifiable)
        return JudgeBundleSnapshot(
            events=state.events,
            committed_pack_by_trace=dict(state.pack_by_trace),
            uncertifiable_trace_ids=tuple(sorted(state.uncertifiable)),
            conflict_trace_ids=tuple(sorted(state.conflicts)),
            missing_trace_ids=tuple(sorted(missing)),
            coverage_manifest=state.coverage_manifest,
            judge_bundle=state.judge_bundle,
            terminal=cls._terminal(state.events),
        )

    @staticmethod
    def _terminal(events: tuple[Artifact, ...]) -> bool:
        return bool(events and events[-1].payload.get("event_type") == "RUN_CLOSED")

    @staticmethod
    def _append(state: _State, epoch: int, event_type: str, details: dict[str, object]) -> Artifact:
        return state.journal.append(
            epoch,
            event_type,
            cast(dict[str, JsonValue], details),
            expected_sequence=len(state.events) + 1,
            expected_previous_hash=state.events[-1].content_hash,
        )

    @staticmethod
    def _outcome_event(events: tuple[Artifact, ...], trace_id: str) -> Artifact | None:
        matches = [
            event
            for event in events
            if event.payload.get("event_type") == "OUTCOME_COMMITTED"
            and cast(dict[str, object], event.payload["details"]).get("trace_id") == trace_id
        ]
        if len(matches) > 1:
            raise JudgeBundleWorkflowError("trace has duplicate committed outcomes")
        return matches[0] if matches else None

    @staticmethod
    def _last_artifact(
        store: ArtifactStore, events: tuple[Artifact, ...], event_type: str, key: str, schema: str
    ) -> Artifact | None:
        matches = [event for event in events if event.payload.get("event_type") == event_type]
        if len(matches) > 1:
            raise JudgeBundleWorkflowError(f"multiple {event_type} events exist")
        if not matches:
            return None
        return store.read(
            cast(str, cast(dict[str, object], matches[0].payload["details"])[key]), expected_schema_name=schema
        )

    @staticmethod
    def _artifact_hash(schema_name: str, payload: dict[str, object]) -> str:
        return sha256_hex(
            canonical_json_bytes({"payload": payload, "schema_name": schema_name, "schema_version": "1.0.0"})
        )
