"""Fixture closed-loop composition for the complete acceptance seam.

This module is deliberately a controller/composition layer.  It does not
implement data ingest, judging, reward routing, cluster execution, or final
evaluation again; callers hand it the immutable hashes produced by those
workflows.  The fixture controller records the cross-phase lineage and uses
small deterministic synthetic manifests for the boundaries that are not
available in a test process.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.data.models import FixtureDataIngestConfig, QueryWindow
from clawrl.data.validation import load_training_dataset
from clawrl.data.workflow import GovernedDataIngestWorkflow
from clawrl.evaluation import (
    EvaluationEnvironment,
    FinalEvaluationProtocolConfig,
    FinalEvaluationProtocolRegistry,
    FinalVerdictConfig,
    FixturePairedGenerationProvider,
    FixtureSolVerdictProvider,
    PairedGenerationConfig,
    PairedGenerationWorkflow,
    SolFinalVerdictWorkflow,
)
from clawrl.evaluation.candidate_freeze import CandidateFreezeConfig, CandidateFreezeWorkflow, FixtureControllerClock
from clawrl.evaluation.future_dataset import (
    Future100DatasetConfig,
    FutureEvaluationDatasetWorkflow,
    prompt_identity_hash,
)
from clawrl.governor.bounded_iteration import BoundedGovernorConfig, BoundedGovernorWorkflow
from clawrl.governor.six_arm_cohort import SixArmCohortConfig, SixArmCohortWorkflow
from clawrl.judge.fit_models import InitialEvalRubric
from clawrl.judge.recertification_122b import (
    Fixture122BRecertificationConfig,
    Fixture122BRecertificationSource,
    Recertification122BWorkflow,
)
from clawrl.training.classic_identity import ClassicSourceRow
from clawrl.training.expected_trajectory_set import ExpectedTrajectorySetConfig, ExpectedTrajectorySetWorkflow
from clawrl.training.gated_122b_run import Gated122BRunConfig, Gated122BRunWorkflow, Independent122BExperimentSpecConfig
from clawrl.training.reward_roundtrip import (
    FenceAuthority,
    FixtureCfsBackend,
    FixtureCfsConfig,
    RewardRoundtripError,
    RewardRoundtripWorkflow,
    RewardSlotKey,
)

_HASH = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
PHASES = ("DATA_INGEST", "JUDGE_CERTIFY", "TRAIN_35B", "TRAIN_122B", "FINAL_EVAL")
NEGATIVE_CONSTRAINTS: dict[str, JsonValue] = {
    "future_data_training": False,
    "gt_training": False,
    "reserved_aggregation": False,
    "router_auto_scale": False,
    "run_goal_mutation": False,
    "shadow_sol": False,
}


class ClosedLoopError(RuntimeError):
    """The fixture campaign cannot advance without violating a gate."""


class ClosedLoopReadinessError(ClosedLoopError):
    """An immutable input is absent, malformed, or conflicts with its phase."""


@dataclass(frozen=True, slots=True)
class FixtureClosedLoopConfig:
    """Top-level seam: only immutable input artifact hashes cross this boundary."""

    campaign_id: str
    dataset_version_hash: str
    judge_bundle_hash: str
    experiment_spec_hash: str
    run_id: str | None = None
    execution_profile: Literal["fixture", "production"] = "fixture"
    controller_epoch: int = 1
    fail_at: str | None = None
    conflict_artifact: str | None = None

    def __post_init__(self) -> None:
        for name in ("campaign_id", "run_id"):
            value = getattr(self, name)
            if value is not None and _ID.fullmatch(value) is None:
                raise ClosedLoopError(f"{name} is invalid")
        for name in ("dataset_version_hash", "judge_bundle_hash", "experiment_spec_hash"):
            if _HASH.fullmatch(getattr(self, name)) is None:
                raise ClosedLoopError(f"{name} must be a SHA-256 digest")
        if self.execution_profile not in {"fixture", "production"}:
            raise ClosedLoopError("execution_profile is invalid")
        if type(self.controller_epoch) is not int or self.controller_epoch <= 0:
            raise ClosedLoopError("controller_epoch is invalid")
        if self.fail_at is not None and self.fail_at not in PHASES:
            raise ClosedLoopError("fail_at is not a known phase")
        if self.conflict_artifact is not None and self.conflict_artifact not in {
            "dataset_version",
            "dataset_version_hash",
            "judge_bundle",
            "judge_bundle_hash",
            "experiment_spec",
            "experiment_spec_hash",
        }:
            raise ClosedLoopError("conflict_artifact is invalid")

    @property
    def effective_run_id(self) -> str:
        return self.run_id or f"{self.campaign_id}-run"


@dataclass(frozen=True, slots=True)
class ClosedLoopSnapshot:
    status: Literal["succeeded", "failed", "blocked"]
    phase: str
    phase_matrix: dict[str, str]
    readiness: Artifact
    events: tuple[Artifact, ...] = ()
    run_record: Artifact | None = None
    decision_record: Artifact | None = None
    rewards: tuple[Artifact, ...] = ()
    reward_manifest: Artifact | None = None
    transfer_candidate: Artifact | None = None
    final_eval: Artifact | None = None
    reason_code: str | None = None
    side_effects: tuple[str, ...] = ()

    @property
    def terminal(self) -> bool:
        return self.status in {"succeeded", "failed"}

    @property
    def outcome(self) -> str:
        return self.status

    @property
    def phase_gates(self) -> dict[str, str]:
        return self.phase_matrix

    @property
    def run(self) -> Artifact | None:
        return self.run_record

    @property
    def decision(self) -> Artifact | None:
        return self.decision_record

    @property
    def reward(self) -> Artifact | None:
        return self.reward_manifest

    @property
    def transfer(self) -> Artifact | None:
        return self.transfer_candidate

    @property
    def final_evaluation(self) -> Artifact | None:
        return self.final_eval


def _artifact_hash(payload: object) -> str:
    return sha256_hex(canonical_json_bytes(payload))


def _fixture_ingest_dataset(root: Path, campaign_id: str) -> Artifact:
    """Run the existing governed fixture ingest instead of fabricating a DatasetVersion."""
    try:
        from tests.fixtures.ticket03_data import stage_data_provider
    except ImportError as error:  # pragma: no cover - fixture package is part of this repository
        raise ClosedLoopReadinessError("FIXTURE_DATA_PROVIDER_UNAVAILABLE") from error
    ingest_config = FixtureDataIngestConfig(
        run_id=f"{campaign_id}-ingest",
        query_sql=(
            "SELECT trace_pk, report_id, event_time_utc, ingestion_time_utc, purpose, prompt, response, "
            "tool_name, model_id, private_sentinel FROM fixture.online_trace "
            "WHERE event_time_utc >= :start_utc AND event_time_utc < :end_utc AND purpose = :purpose "
            "ORDER BY trace_pk ASC, ingestion_time_utc ASC"
        ),
        window=QueryWindow("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
    )
    snapshot = GovernedDataIngestWorkflow.bootstrap(
        root, ingest_config, role_ingress=stage_data_provider(root), epoch=1
    )
    while not snapshot.terminal:
        snapshot = GovernedDataIngestWorkflow.resume(root, ingest_config.run_id, epoch=1)
    if snapshot.dataset_version is None:
        raise ClosedLoopReadinessError("DATASET_INGEST_NOT_PUBLISHED")
    return snapshot.dataset_version


def _fixture_final_evaluation(
    root: Path,
    config: FixtureClosedLoopConfig,
    run_record: Artifact,
    *,
    dataset_hash: str,
    judge_bundle_hash: str,
    experiment_spec_hash: str,
    trained_checkpoint_hash: str,
) -> Artifact:
    """Drive Tickets 28-31 through their durable workflow seams."""

    store = ArtifactStore(root)
    protocol = FinalEvaluationProtocolRegistry.preregister(
        root,
        campaign_id=config.campaign_id,
        config=FinalEvaluationProtocolConfig(
            "future-100-v1", InitialEvalRubric.fixture_default(), EvaluationEnvironment(), 11, 12
        ),
    )
    base_checkpoint = store.put(
        "Checkpoint",
        "1.0.0",
        {
            "evaluation_environment_hash": protocol.environment.content_hash,
            "model_identity": "base-122B",
            "run_id": run_record.payload["run_id"],
            **{
                name: protocol.environment.payload[name]
                for name in (
                    "semantic_decoding",
                    "tool_harness_policy",
                    "prompt_wrapper",
                    "evaluator_visible_trajectory_schema",
                )
            },
        },
    )
    freeze_result = CandidateFreezeWorkflow.run(
        root,
        config=CandidateFreezeConfig(
            campaign_id=config.campaign_id,
            protocol_hash=protocol.protocol.content_hash,
            preregistration_receipt_hash=protocol.receipt.content_hash,
            trained_run_record_hash=run_record.content_hash,
            base_checkpoint_hash=base_checkpoint.content_hash,
            dataset_version_hash=dataset_hash,
            judge_bundle_hash=judge_bundle_hash,
            experiment_spec_hash=experiment_spec_hash,
            trained_checkpoint_hash=trained_checkpoint_hash,
            controller_epoch=config.controller_epoch,
            t0_utc="2026-02-01T00:00:00Z",
        ),
        clock=FixtureControllerClock("2026-02-01T00:00:00Z"),
    )
    if freeze_result.freeze is None:
        raise ClosedLoopReadinessError("CANDIDATE_FREEZE_NOT_COMMITTED")
    freeze = freeze_result.freeze
    rows = [
        {
            "difficulty": "hard",
            "purpose": "eval_only",
            "prompt": f"future prompt {index}",
            "prompt_identity_hash": prompt_identity_hash(f"future prompt {index}"),
            "provider_row_id": f"row-{index}",
            "event_time_utc": "2026-02-01T00:01:00Z",
            "ingestion_time_utc": "2026-02-01T00:01:01Z",
        }
        for index in range(100)
    ]
    dataset_result = FutureEvaluationDatasetWorkflow.run(
        root,
        config=Future100DatasetConfig(
            campaign_id=config.campaign_id,
            candidate_freeze_hash=freeze.content_hash,
            protocol_hash=protocol.protocol.content_hash,
            initial_rows=tuple(rows),
        ),
        initial_rows=rows,
    )
    if dataset_result.dataset is None or dataset_result.status != "committed":
        raise ClosedLoopReadinessError("EVALUATION_DATASET_NOT_COMMITTED")
    dataset = dataset_result.dataset
    paired = PairedGenerationWorkflow.run(
        root,
        config=PairedGenerationConfig(config.campaign_id, freeze.content_hash, dataset.content_hash),
        provider=FixturePairedGenerationProvider(),
    )
    if paired.status != "committed":
        raise ClosedLoopReadinessError("PAIRED_GENERATION_NOT_COMMITTED")
    # Recompute the precommitted fixture assignment without opening the sealed
    # evaluator credential.  The Sol provider receives only persisted A/B
    # payload artifacts; the sealed mapping remains unread until Ticket31's
    # terminal unseal after all 100 verdicts.
    mapping = PairedGenerationWorkflow._assignment(rows, seed=12)
    outcomes = {
        identity: ("B" if mapping[identity] == "A" else "A") if index < 60 else mapping[identity]
        for index, identity in enumerate(mapping)
    }
    final = SolFinalVerdictWorkflow.run(
        root,
        config=FinalVerdictConfig(
            config.campaign_id,
            freeze.content_hash,
            dataset.content_hash,
            paired_generation_manifest_hash=Path(root, "paired-generations", config.campaign_id, "active.ref")
            .read_text()
            .strip(),
            run_record_hash=run_record.content_hash,
        ),
        provider=FixtureSolVerdictProvider(outcomes=outcomes),
    )
    if final.status != "PASSED" or final.final_run is None:
        raise ClosedLoopReadinessError("FINAL_EVAL_NOT_PASSED")
    return final.final_run


class FixtureClosedLoopWorkflow:
    """Compose the existing slices behind one deterministic fixture controller."""

    @staticmethod
    def production_readiness(root: str | Path, *, phase: str = "DATA_INGEST") -> Artifact:
        if phase not in PHASES:
            raise ClosedLoopError("phase is invalid")
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [{"code": "PRODUCTION_FIXTURE_ONLY", "status": "blocked"}],
                "execution_profile": "production",
                "phase": phase,
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )

    @classmethod
    def phase_matrix(cls, root: str | Path, *, config: FixtureClosedLoopConfig) -> dict[str, str]:
        """Evaluate every gate independently; a green earlier gate is not inheritance."""
        store = ArtifactStore(root)
        matrix = {phase: "blocked" for phase in PHASES}
        if config.execution_profile == "production":
            return matrix
        refs = {
            "DATA_INGEST": (config.dataset_version_hash, "DatasetVersion"),
            "JUDGE_CERTIFY": (config.judge_bundle_hash, "JudgeBundle"),
            "TRAIN_35B": (config.experiment_spec_hash, "ExperimentSpec"),
            "TRAIN_122B": (config.experiment_spec_hash, "ExperimentSpec"),
            "FINAL_EVAL": (config.experiment_spec_hash, "ExperimentSpec"),
        }
        for phase, (digest, schema) in refs.items():
            try:
                artifact = store.read(digest, expected_schema_name=schema)
                if (
                    phase == "JUDGE_CERTIFY"
                    and artifact.payload.get("dataset_version_hash") != config.dataset_version_hash
                ):
                    continue
                if (
                    phase in {"TRAIN_35B", "TRAIN_122B", "FINAL_EVAL"}
                    and artifact.payload.get("dataset_version_hash") != config.dataset_version_hash
                ):
                    continue
                matrix[phase] = "green"
            except (ArtifactCorruption, KeyError, OSError, ValueError):
                continue
        # Transfer and final evaluation have independent evidence.  A green
        # 35B spec therefore never implicitly releases either later phase.
        transfer_run_hashes: set[str] = set()
        final_seen = False
        try:
            final_artifacts: list[Artifact] = []
            for path in (Path(root) / "artifacts").glob("*.json"):
                try:
                    artifact = store.read(path.stem)
                except Exception:
                    continue
                if artifact.schema_name == "FinalEvalRun":
                    final_artifacts.append(artifact)
                    continue
                if artifact.schema_name == "TransferCandidate":
                    run_hash = artifact.payload.get("run_record_hash")
                    if type(run_hash) is not str:
                        continue
                    try:
                        run = store.read(run_hash, expected_schema_name="RunRecord")
                    except Exception:
                        continue
                    if (
                        run.payload.get("status") == "succeeded"
                        and run.payload.get("phase") == "TRAIN_122B"
                        and run.payload.get("dataset_version_hash") == config.dataset_version_hash
                        and artifact.payload.get("status") == "immutable"
                        and artifact.payload.get("terminal_action") == "stop_and_transfer"
                        and artifact.payload.get("cohort_id") == f"{config.campaign_id}-35b"
                    ):
                        try:
                            run_bundle = store.read(
                                cast(str, run.payload["judge_bundle_hash"]), expected_schema_name="JudgeBundle"
                            )
                            run_spec = store.read(
                                cast(str, run.payload["experiment_spec_hash"]), expected_schema_name="ExperimentSpec"
                            )
                        except Exception:
                            continue
                        if (
                            run_bundle.payload.get("dataset_version_hash") == config.dataset_version_hash
                            and run_bundle.payload.get("status") == "terminal"
                            and run_bundle.payload.get("certification_phase") == "TRAIN_122B"
                            and run_spec.payload.get("dataset_version_hash") == config.dataset_version_hash
                            and run_spec.payload.get("model_size") == "122B"
                            and run_spec.payload.get("status") in {"approved", "frozen"}
                        ):
                            transfer_run_hashes.add(run_hash)
            for artifact in final_artifacts:
                final_run_hash = artifact.payload.get("run_record_hash")
                freeze_hash = artifact.payload.get("candidate_freeze_hash")
                if type(final_run_hash) is not str or final_run_hash not in transfer_run_hashes:
                    continue
                try:
                    freeze = store.read(cast(str, freeze_hash), expected_schema_name="CandidateFreeze")
                except Exception:
                    continue
                final_seen = final_seen or (
                    artifact.payload.get("campaign_id") == config.campaign_id
                    and artifact.payload.get("status") == "terminal"
                    and artifact.payload.get("verdict_count") == 100
                    and freeze.payload.get("dataset_version_hash") == config.dataset_version_hash
                    and freeze.payload.get("judge_bundle_hash") is not None
                    and freeze.payload.get("experiment_spec_hash") is not None
                )
        except OSError:
            pass
        if not transfer_run_hashes:
            matrix["TRAIN_122B"] = "blocked"
            matrix["FINAL_EVAL"] = "blocked"
        elif not final_seen:
            matrix["FINAL_EVAL"] = "blocked"
        return matrix

    @classmethod
    def run(cls, root: str | Path, *, config: FixtureClosedLoopConfig) -> ClosedLoopSnapshot:
        root_path = Path(root)
        store = ArtifactStore(root_path)
        if config.execution_profile == "production":
            readiness = cls.production_readiness(root_path, phase="DATA_INGEST")
            return ClosedLoopSnapshot("blocked", "DATA_INGEST", {phase: "blocked" for phase in PHASES}, readiness)

        matrix = cls.phase_matrix(root_path, config=config)
        events: list[Artifact] = []
        effects: list[str] = []
        run_id = config.effective_run_id
        # Validate in phase order, before creating any scorer/optimizer/cluster artifact.
        required = (
            ("DATA_INGEST", config.dataset_version_hash, "DatasetVersion", "dataset_version"),
            ("JUDGE_CERTIFY", config.judge_bundle_hash, "JudgeBundle", "judge_bundle"),
            ("TRAIN_35B", config.experiment_spec_hash, "ExperimentSpec", "experiment_spec"),
        )
        failed_phase: str | None = None
        reason = None
        for phase, digest, schema, label in required:
            try:
                artifact = store.read(digest, expected_schema_name=schema)
                if config.conflict_artifact in {label, f"{label}_hash"}:
                    raise ClosedLoopReadinessError(f"{label.upper()}_HASH_CONFLICT")
                if phase == "DATA_INGEST":
                    refs = artifact.payload.get("trace_refs")
                    if (
                        artifact.payload.get("trace_count") != 100
                        or not isinstance(refs, list)
                        or len(refs) != 100
                        or len({item.get("trace_id") if isinstance(item, dict) else item for item in refs}) != 100
                        or any(
                            not isinstance(item, dict)
                            or type(item.get("artifact_hash")) is not str
                            or type(item.get("trace_id")) is not str
                            for item in refs
                        )
                    ):
                        raise ClosedLoopReadinessError("TRAINING_TRACE_COVERAGE_INVALID")
                    for trace_ref in refs:
                        trace_hash = cast(str, cast(dict[str, object], trace_ref)["artifact_hash"])
                        trace = store.read(trace_hash, expected_schema_name="TrainingTrace")
                        if trace.payload.get("purpose") != "training_allowed" or not trace.payload.get("trace_id"):
                            raise ClosedLoopReadinessError("TRAINING_TRACE_LINEAGE_INVALID")
                if phase == "JUDGE_CERTIFY":
                    fit_refs = artifact.payload.get("fit_trajectory_refs")
                    if (
                        artifact.payload.get("fit_trajectory_count") != 3200
                        or not isinstance(fit_refs, list)
                        or len(fit_refs) != 3200
                        or len(set(fit_refs)) != 3200
                        or any(type(item) is not str for item in fit_refs)
                    ):
                        raise ClosedLoopReadinessError("FIT_TRAJECTORY_COVERAGE_INVALID")
                    counts: dict[str, int] = {}
                    trajectory_ids: set[str] = set()
                    for fit_hash in cast(list[str], fit_refs):
                        fit = store.read(fit_hash, expected_schema_name="FitTrajectory")
                        trace_id = fit.payload.get("trace_id")
                        trajectory_id = fit.payload.get("trajectory_id")
                        if (
                            type(trace_id) is not str
                            or type(trajectory_id) is not str
                            or trajectory_id in trajectory_ids
                        ):
                            raise ClosedLoopReadinessError("FIT_TRAJECTORY_LINEAGE_INVALID")
                        trajectory_ids.add(trajectory_id)
                        counts[trace_id] = counts.get(trace_id, 0) + 1
                    trace_refs = store.read(
                        config.dataset_version_hash, expected_schema_name="DatasetVersion"
                    ).payload.get("trace_refs")
                    trace_ids = {
                        cast(
                            str,
                            store.read(
                                cast(str, cast(dict[str, object], item)["artifact_hash"]),
                                expected_schema_name="TrainingTrace",
                            ).payload["trace_id"],
                        )
                        for item in cast(list[dict[str, object]], trace_refs)
                    }
                    if counts != {trace_id: 32 for trace_id in trace_ids}:
                        raise ClosedLoopReadinessError("FIT_TRAJECTORY_PER_TRACE_COVERAGE_INVALID")
                if phase == "TRAIN_35B":
                    if (
                        artifact.payload.get("dataset_version_hash") != config.dataset_version_hash
                        or artifact.payload.get("judge_bundle_hash") != config.judge_bundle_hash
                        or artifact.payload.get("status") not in {"approved", "frozen"}
                    ):
                        raise ClosedLoopReadinessError("EXPERIMENT_SPEC_LINEAGE_INVALID")
                if (
                    phase == "JUDGE_CERTIFY"
                    and artifact.payload.get("dataset_version_hash") != config.dataset_version_hash
                ):
                    raise ClosedLoopReadinessError("JUDGE_BUNDLE_DATASET_LINEAGE_CONFLICT")
                if phase == "TRAIN_35B" and artifact.payload.get("dataset_version_hash") != config.dataset_version_hash:
                    raise ClosedLoopReadinessError("EXPERIMENT_DATASET_LINEAGE_CONFLICT")
            except (ArtifactCorruption, ClosedLoopReadinessError, OSError, ValueError, KeyError) as error:
                failed_phase, reason = phase, str(error) or f"{label.upper()}_MISSING"
                break
            if config.fail_at == phase:
                failed_phase, reason = phase, f"{phase}_INJECTED_FAILURE"
                break
            event = store.put(
                "ClosedLoopEvent",
                "1.0.0",
                {"campaign_id": config.campaign_id, "event": f"{phase}_GREEN", "phase": phase},
            )
            events.append(event)

        readiness_payload: dict[str, JsonValue] = {
            "campaign_id": config.campaign_id,
            "execution_profile": "fixture",
            "phase": failed_phase or "FINAL_EVAL",
            "phase_matrix": cast(JsonValue, matrix),
            "side_effects_permitted": failed_phase is None,
            "status": "blocked" if failed_phase else "green",
        }
        if failed_phase:
            failed_index = PHASES.index(failed_phase)
            for downstream in PHASES[failed_index:]:
                matrix[downstream] = "blocked"
            readiness_payload["checks"] = [{"code": reason, "status": "blocked"}]
            readiness = store.put("ReadinessReport", "1.0.0", readiness_payload)
            decision = store.put(
                "DecisionRecord",
                "1.0.0",
                {
                    "campaign_id": config.campaign_id,
                    "phase": failed_phase,
                    "reason_code": reason,
                    "status": "failed_closed",
                    "run_id": run_id,
                },
            )
            effects.extend(("readiness", "decision"))
            return ClosedLoopSnapshot(
                "failed",
                failed_phase,
                matrix,
                readiness,
                tuple(events),
                decision_record=decision,
                reason_code=reason,
                side_effects=tuple(effects),
            )

        readiness_payload["checks"] = [{"code": f"{phase}_GREEN", "status": "green"} for phase in PHASES]
        readiness_payload["negative_constraints"] = dict(NEGATIVE_CONSTRAINTS)
        readiness = store.put("ReadinessReport", "1.0.0", readiness_payload)

        bundle = store.read(config.judge_bundle_hash, expected_schema_name="JudgeBundle")
        reward_schema_hash = bundle.payload.get("reward_schema_hash")
        scalarizer_hash = bundle.payload.get("scalarizer_hash")
        if type(reward_schema_hash) is not str or type(scalarizer_hash) is not str:
            raise ClosedLoopReadinessError("REWARD_CONTRACT_LINEAGE_INVALID")
        RewardRoundtripWorkflow._contracts(store, reward_schema_hash, scalarizer_hash)
        judge_pack_id = f"{config.campaign_id}-judge-pack"
        pack_refs = bundle.payload.get("judge_pack_refs")
        if isinstance(pack_refs, list) and pack_refs and isinstance(pack_refs[0], dict):
            configured_pack = pack_refs[0].get("judge_pack_id")
            if type(configured_pack) is str:
                judge_pack_id = configured_pack

        # Reuse the durable six-arm fixture controller for the development
        # cohort.  It is synthetic and never constructs a production adapter.
        cohort = SixArmCohortWorkflow.run(
            root_path,
            config=SixArmCohortConfig(
                cohort_id=f"{config.campaign_id}-35b",
                dataset_version_hash=config.dataset_version_hash,
                judge_bundle_hash=config.judge_bundle_hash,
                controller_epoch=config.controller_epoch,
            ),
        )
        if not cohort.terminal or cohort.promotion is None:
            raise ClosedLoopReadinessError("SIX_ARM_COHORT_NOT_TERMINAL")
        evaluation_environment = store.put("EvaluationEnvironment", "1.0.0", EvaluationEnvironment().artifact_payload())
        environment_payload = evaluation_environment.payload
        # Establish fresh 122B recertification before producing reward slots;
        # the reward manifest must bind the new JudgeBundle.
        governor = BoundedGovernorWorkflow.run(
            root_path,
            config=BoundedGovernorConfig(
                governor_id=f"{config.campaign_id}-governor",
                cohort_id=f"{config.campaign_id}-35b",
                branch="stop_and_transfer",
                candidate_id=f"{config.campaign_id}-candidate",
                controller_epoch=config.controller_epoch,
            ),
        )
        transfer = governor.transfer_candidate
        if transfer is None or not governor.terminal:
            raise ClosedLoopReadinessError("GOVERNOR_TRANSFER_NOT_TERMINAL")
        algorithm_hash = bundle.payload.get("algorithm_contract_hash")
        if type(algorithm_hash) is not str:
            algorithm = store.put(
                "RLAlgorithmContract",
                "1.0.0",
                {
                    "advantage_estimator": "grpo",
                    "algorithm_id": f"{config.campaign_id}-grpo-v1",
                    "reward_aggregation": "calibrated_scalar",
                    "schema_version": "rl-algorithm-contract/1.0.0",
                },
            )
            algorithm_hash = algorithm.content_hash
        recert_config = Fixture122BRecertificationConfig(
            run_id=f"{run_id}-recert",
            transfer_candidate_hash=transfer.content_hash,
            dataset_version_hash=config.dataset_version_hash,
            reward_schema_hash=cast(str, reward_schema_hash),
            scalarizer_hash=cast(str, scalarizer_hash),
            algorithm_contract_hash=algorithm_hash,
            old_judge_bundle_hash=config.judge_bundle_hash,
        )
        recert_source = Fixture122BRecertificationSource(
            root_path,
            dataset_version_hash=config.dataset_version_hash,
            reward_schema_hash=cast(str, reward_schema_hash),
            scalarizer_hash=cast(str, scalarizer_hash),
            algorithm_contract_hash=algorithm_hash,
        )
        recert = Recertification122BWorkflow.run(
            root_path, recert_config, source=recert_source, epoch=config.controller_epoch
        )
        if recert.judge_bundle is None or not recert.terminal:
            raise ClosedLoopReadinessError("122B_RECERTIFICATION_NOT_TERMINAL")
        reward_bundle_hash = recert.judge_bundle.content_hash
        reward_pack_id = f"{config.campaign_id}-recertified-judge-pack"
        report_hashes = recert.judge_bundle.payload.get("report_hashes")
        if isinstance(report_hashes, list) and report_hashes:
            first_report = store.read(cast(str, report_hashes[0]), expected_schema_name="CertificationReport")
            pack_hash = first_report.payload.get("judge_pack_hash")
            if type(pack_hash) is str:
                pack = store.read(pack_hash, expected_schema_name="JudgePack")
                if type(pack.payload.get("judge_pack_id")) is str:
                    reward_pack_id = cast(str, pack.payload["judge_pack_id"])
        judge_pack_id = reward_pack_id
        # Reward seam: drive the same stateful CFS/attempt/fencing workflow
        # used by production-shaped reward paths for all 16 x 128 slots.
        probe = FixtureCfsBackend(root_path, FixtureCfsConfig(f"{config.campaign_id}-cfs")).probe()
        authority = FenceAuthority(root_path)
        try:
            current_epoch = authority.current().payload.get("epoch")
        except RewardRoundtripError:
            authority.advance(config.controller_epoch)
            current_epoch = config.controller_epoch
        if type(current_epoch) is not int or current_epoch != config.controller_epoch:
            raise ClosedLoopReadinessError("REWARD_FENCING_EPOCH_INVALID")
        rewards: list[Artifact] = []
        first_trace = cast(
            str,
            load_training_dataset(store, config.dataset_version_hash).training_traces[0].payload["trace_id"],
        )
        for step in range(16):
            expected_source = ClassicSourceRow(
                trace_id=first_trace,
                uid=f"trace-{step:02d}",
                judge_pack_id=judge_pack_id,
                prompt=f"{config.campaign_id} expected trajectory source {step}",
            )
            ExpectedTrajectorySetWorkflow.run(
                root_path,
                config=ExpectedTrajectorySetConfig(run_id=f"{run_id}-expected", global_step=step, rollout_count=128),
                sources=(expected_source,),
                arrival_ordinals=tuple(range(128)),
                transport="classic",
            )
            for rollout in range(128):
                uid = f"trace-{step:02d}"
                key = RewardSlotKey(run_id, step, uid, rollout, judge_pack_id)
                reward_state = RewardRoundtripWorkflow.start(
                    root_path,
                    key=key,
                    probe_evidence_hash=probe.content_hash,
                    trajectory_payload={
                        "campaign_id": config.campaign_id,
                        "dataset_version_hash": config.dataset_version_hash,
                        "global_step": step,
                        "rollout_index": rollout,
                        "uid": uid,
                    },
                    reward_schema_hash=reward_schema_hash,
                    scalarizer_hash=scalarizer_hash,
                )
                if reward_state.resolved_reward is None:
                    attempt = RewardRoundtripWorkflow.claim_attempt(
                        root_path, key=key, resolver_epoch=config.controller_epoch, clock_tick=0
                    )
                    reward_digest = int(key.content_hash[:8], 16)
                    RewardRoundtripWorkflow.commit_result(
                        root_path,
                        key=key,
                        attempt_ordinal=cast(int, attempt.payload["attempt_ordinal"]),
                        result_payload={
                            "confidence_basis_points": 7_500 + reward_digest % 2_501,
                            "failure_tags": [],
                            "reward_micros": reward_digest % 100_000_001,
                            "reward_schema_hash": reward_schema_hash,
                            "scalarizer_hash": scalarizer_hash,
                            "turn_local_tie_groups": [[uid]],
                        },
                    )
                    reward = RewardRoundtripWorkflow.resolve(root_path, key=key, resolver_epoch=config.controller_epoch)
                else:
                    reward = reward_state.resolved_reward
                rewards.append(reward)
        reward_manifest = store.put(
            "RewardStepManifest",
            "1.0.0",
            {
                "aggregation": "calibrated_scalar",
                "dataset_version_hash": config.dataset_version_hash,
                "judge_bundle_hash": reward_bundle_hash,
                "reward_hashes": [item.content_hash for item in rewards],
                "run_id": run_id,
                "slot_count": len(rewards),
                "steps": 16,
                "rollouts_per_step": 128,
            },
        )
        effects.extend(("scorer", "optimizer"))

        independent_spec = Gated122BRunWorkflow.build_experiment_spec(
            root_path,
            Independent122BExperimentSpecConfig(
                experiment_id=f"{config.campaign_id}-122b-spec",
                dataset_version_hash=config.dataset_version_hash,
                judge_bundle_hash=recert.judge_bundle.content_hash,
                parallelism={"tensor": 8, "pipeline": 4},
                resource={"gpu": "fixture-122b", "count": 8},
                retry={"max_attempts": 2, "backoff_ticks": 1},
                monitoring={"heartbeat_ticks": 4, "max_kl_millis": 250},
                approval_hash=_artifact_hash({"campaign_id": config.campaign_id, "approver": "fixture"}),
                approved_by="fixture-approver",
                evaluation_environment_hash=evaluation_environment.content_hash,
                semantic_decoding=cast(dict[str, JsonValue], environment_payload["semantic_decoding"]),
                tool_harness_policy=cast(dict[str, JsonValue], environment_payload["tool_harness_policy"]),
                prompt_wrapper=cast(dict[str, JsonValue], environment_payload["prompt_wrapper"]),
                evaluator_visible_trajectory_schema=cast(
                    dict[str, JsonValue], environment_payload["evaluator_visible_trajectory_schema"]
                ),
            ),
        )
        gated = Gated122BRunWorkflow.run(
            root_path,
            config=Gated122BRunConfig(
                run_id=run_id,
                dataset_version_hash=config.dataset_version_hash,
                judge_bundle_hash=recert.judge_bundle.content_hash,
                experiment_spec_hash=independent_spec.content_hash,
                global_step=0,
                reward_manifest_hash=reward_manifest.content_hash,
            ),
            epoch=config.controller_epoch,
        )
        run_record = gated.run_record
        gated_decision = gated.decision
        checkpoint = gated.checkpoint
        if run_record is None or gated_decision is None or checkpoint is None:
            raise ClosedLoopReadinessError("122B_DURABLE_RUN_NOT_TERMINAL")
        transfer = BoundedGovernorWorkflow.bind_run_record(
            root_path, f"{config.campaign_id}-governor", run_record_hash=run_record.content_hash
        )
        final_eval = _fixture_final_evaluation(
            root_path,
            config,
            run_record,
            dataset_hash=config.dataset_version_hash,
            judge_bundle_hash=recert.judge_bundle.content_hash,
            experiment_spec_hash=independent_spec.content_hash,
            trained_checkpoint_hash=cast(str, run_record.payload["checkpoint_hash"]),
        )
        effects.extend(("cluster", "final-eval"))
        matrix = {phase: "green" for phase in PHASES}
        return ClosedLoopSnapshot(
            "succeeded",
            "FINAL_EVAL",
            matrix,
            readiness,
            tuple(events),
            run_record,
            gated_decision,
            tuple(rewards),
            reward_manifest,
            transfer,
            final_eval,
            side_effects=tuple(effects),
        )

    # Explicit names make the controller seam convenient for integrations and
    # keep replay code self-documenting without introducing a second workflow.
    execute = run
    run_campaign = run

    @classmethod
    def run_fixture(cls, root: str | Path, *, campaign_id: str = "fixture-closed-loop") -> ClosedLoopSnapshot:
        """Create exact-scale synthetic input artifacts and run the public seam."""
        store = ArtifactStore(root)
        dataset = _fixture_ingest_dataset(Path(root), campaign_id)
        trace_ids = [
            item.payload["trace_id"] for item in load_training_dataset(store, dataset.content_hash).training_traces
        ]
        fit_trajectories = [
            store.put(
                "FitTrajectory",
                "1.0.0",
                {
                    "fit_index": fit_index,
                    "status": "committed",
                    "trace_id": trace_ids[fit_index % 100],
                    "trajectory_id": f"fit-{fit_index:04d}",
                },
            )
            for fit_index in range(3200)
        ]
        reward_schema = store.put(
            "RewardSchema",
            "1.0.0",
            {
                "aggregation": "calibrated_scalar",
                "maximum_micros": 100_000_000,
                "minimum_micros": 0,
                "schema_id": f"{campaign_id}-reward-schema",
                "schema_version": "reward-schema/1.0.0",
            },
        )
        scalarizer = store.put(
            "Scalarizer",
            "1.0.0",
            {
                "dimension_weights_micros": {
                    "correctness": 250_000,
                    "reasoning_quality": 250_000,
                    "task_completion": 250_000,
                    "tool_discipline": 250_000,
                },
                "scalarizer_id": f"{campaign_id}-scalarizer",
                "schema_version": "scalarizer/1.0.0",
            },
        )
        algorithm = store.put(
            "RLAlgorithmContract",
            "1.0.0",
            {
                "advantage_estimator": "grpo",
                "algorithm_id": f"{campaign_id}-grpo-v1",
                "reward_aggregation": "calibrated_scalar",
                "schema_version": "rl-algorithm-contract/1.0.0",
            },
        )
        judge_pack = store.put(
            "JudgePack",
            "1.0.0",
            {"judge_pack_id": f"{campaign_id}-judge-pack", "status": "committed", "version": "fixture-v1"},
        )
        bundle = store.put(
            "JudgeBundle",
            "1.0.0",
            {
                "dataset_hash": dataset.content_hash,
                "dataset_version_hash": dataset.content_hash,
                "fit_trajectory_count": 3200,
                "fit_trajectory_refs": [trajectory.content_hash for trajectory in fit_trajectories],
                "judge_pack_refs": [
                    {"artifact_hash": judge_pack.content_hash, "judge_pack_id": f"{campaign_id}-judge-pack"}
                ],
                "reward_schema_hash": reward_schema.content_hash,
                "scalarizer_hash": scalarizer.content_hash,
                "algorithm_contract_hash": algorithm.content_hash,
                "status": "total",
                "trace_count": 100,
            },
        )
        spec = store.put(
            "ExperimentSpec",
            "1.0.0",
            {
                "dataset_version_hash": dataset.content_hash,
                "judge_bundle_hash": bundle.content_hash,
                "status": "approved",
                "model_size": "122B",
            },
        )
        return cls.run(
            root,
            config=FixtureClosedLoopConfig(campaign_id, dataset.content_hash, bundle.content_hash, spec.content_hash),
        )


# Ticket-oriented aliases used by integrations.
FixtureClosedLoop = FixtureClosedLoopWorkflow
ClosedLoopConfig = FixtureClosedLoopConfig
ClosedLoopSnapshot = ClosedLoopSnapshot
ClosedLoopCampaign = FixtureClosedLoopWorkflow
ClosedLoopCampaignWorkflow = FixtureClosedLoopWorkflow
ClosedLoopFixtureConfig = FixtureClosedLoopConfig
