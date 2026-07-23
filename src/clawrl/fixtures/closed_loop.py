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
from clawrl.evaluation.future_dataset import prompt_identity_hash
from clawrl.governor.six_arm_cohort import SixArmCohortConfig, SixArmCohortWorkflow
from clawrl.judge.fit_models import InitialEvalRubric
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


def _fixture_final_evaluation(root: Path, config: FixtureClosedLoopConfig, run_record: Artifact) -> Artifact:
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
        "Checkpoint", "1.0.0", {"model_identity": "base-122B", "run_id": run_record.payload["run_id"]}
    )
    freeze = store.put(
        "CandidateFreeze",
        "1.0.0",
        {
            "campaign_id": config.campaign_id,
            "protocol_hash": protocol.protocol.content_hash,
            "evaluation_environment_hash": protocol.environment.content_hash,
            "base_checkpoint_hash": base_checkpoint.content_hash,
            "trained_checkpoint_hash": run_record.payload["checkpoint_hash"],
            "trained_run_record_hash": run_record.content_hash,
            "dataset_version_hash": config.dataset_version_hash,
            "judge_bundle_hash": config.judge_bundle_hash,
            "experiment_spec_hash": config.experiment_spec_hash,
            "status": "immutable",
            "t0_utc": "2026-02-01T00:00:00Z",
        },
    )
    ArtifactStore._publish(
        root / "candidate-freeze" / config.campaign_id / "active.ref", f"{freeze.content_hash}\n".encode("ascii")
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
            "campaign_id": config.campaign_id,
            "candidate_freeze_hash": freeze.content_hash,
            "protocol_hash": protocol.protocol.content_hash,
            "identity_normalizer_hash": cast(dict[str, JsonValue], protocol.protocol.payload["identity_exclusion"])[
                "normalizer_hash"
            ],
            "source_contract_hash": _artifact_hash({"campaign": config.campaign_id, "source": "fixture"}),
            "governance_hash": _artifact_hash({"campaign": config.campaign_id, "governance": "fixture"}),
            "window_hash": _artifact_hash({"campaign": config.campaign_id, "window": "future-100"}),
            "t0_utc": "2026-02-01T00:00:00Z",
            "window": {"initial_end_utc": "2026-02-02T00:00:00Z", "extension_used": False},
            "prompt_rows": rows,
            "purpose": "eval_only",
            "status": "immutable",
        },
    )
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
                if phase == "JUDGE_CERTIFY" and artifact.payload.get("dataset_version_hash") not in {
                    None,
                    config.dataset_version_hash,
                }:
                    continue
                if phase in {"TRAIN_35B", "TRAIN_122B", "FINAL_EVAL"} and artifact.payload.get(
                    "dataset_version_hash"
                ) not in {None, config.dataset_version_hash}:
                    continue
                matrix[phase] = "green"
            except (ArtifactCorruption, KeyError, OSError, ValueError):
                continue
        # Transfer and final evaluation have independent evidence.  A green
        # 35B spec therefore never implicitly releases either later phase.
        transfer_seen = False
        final_seen = False
        try:
            for path in (Path(root) / "artifacts").glob("*.json"):
                try:
                    artifact = store.read(path.stem)
                except Exception:
                    continue
                if (
                    artifact.schema_name == "TransferCandidate"
                    and artifact.payload.get("campaign_id") == config.campaign_id
                ):
                    transfer_seen = True
                if artifact.schema_name == "FinalEvalRun" and artifact.payload.get("campaign_id") == config.campaign_id:
                    final_seen = True
        except OSError:
            pass
        if not transfer_seen:
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
                        or len(set(refs)) != 100
                        or any(type(item) is not str for item in refs)
                    ):
                        raise ClosedLoopReadinessError("TRAINING_TRACE_COVERAGE_INVALID")
                    for trace_hash in cast(list[str], refs):
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
                        cast(str, store.read(item, expected_schema_name="TrainingTrace").payload["trace_id"])
                        for item in cast(list[str], trace_refs)
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
                if phase == "JUDGE_CERTIFY" and artifact.payload.get("dataset_version_hash") not in {
                    None,
                    config.dataset_version_hash,
                }:
                    raise ClosedLoopReadinessError("JUDGE_BUNDLE_DATASET_LINEAGE_CONFLICT")
                if phase == "TRAIN_35B" and artifact.payload.get("dataset_version_hash") not in {
                    None,
                    config.dataset_version_hash,
                }:
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
        cohort_promotion_hash = cohort.promotion.content_hash

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
        for step in range(16):
            for rollout in range(128):
                uid = f"trace-{rollout % 100:03d}"
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
                "judge_bundle_hash": config.judge_bundle_hash,
                "reward_hashes": [item.content_hash for item in rewards],
                "run_id": run_id,
                "slot_count": len(rewards),
                "steps": 16,
                "rollouts_per_step": 128,
            },
        )
        effects.extend(("scorer", "optimizer"))
        checkpoint = store.put(
            "Checkpoint",
            "1.0.0",
            {
                "evaluation_environment_hash": _artifact_hash(
                    {"campaign_id": config.campaign_id, "phase": "FINAL_EVAL"}
                ),
                "model_identity": "trained-122B",
                "run_id": run_id,
                "global_step": 16,
            },
        )
        run_record = store.put(
            "RunRecord",
            "1.0.0",
            {
                "campaign_id": config.campaign_id,
                "checkpoint_hash": checkpoint.content_hash,
                "dataset_version_hash": config.dataset_version_hash,
                "experiment_spec_hash": config.experiment_spec_hash,
                "judge_bundle_hash": config.judge_bundle_hash,
                "optimizer_update_count": 16,
                "phase": "TRAIN_122B",
                "reward_hash": reward_manifest.content_hash,
                "run_id": run_id,
                "status": "succeeded",
                "negative_constraints": dict(NEGATIVE_CONSTRAINTS),
            },
        )
        decision = store.put(
            "DecisionRecord",
            "1.0.0",
            {
                "action": "stop_and_transfer",
                "campaign_id": config.campaign_id,
                "experiment_spec_hash": config.experiment_spec_hash,
                "run_record_hash": run_record.content_hash,
                "status": "committed",
                "negative_constraints": dict(NEGATIVE_CONSTRAINTS),
            },
        )
        child = store.put(
            "GovernorChild",
            "1.0.0",
            {
                "branch": "stop_and_transfer",
                "campaign_id": config.campaign_id,
                "status": "completed",
                "run_record_hash": run_record.content_hash,
            },
        )
        summary = store.put(
            "ExperimentSummary",
            "1.0.0",
            {
                "branch": "stop_and_transfer",
                "causal_boundary": "controlled_comparison_only; black_box_descriptive_only",
                "evidence_hashes": [child.content_hash],
                "status": "terminal",
            },
        )
        transfer = store.put(
            "TransferCandidate",
            "1.0.0",
            {
                "candidate_id": f"{config.campaign_id}-122b",
                "campaign_id": config.campaign_id,
                "dataset_version_hash": config.dataset_version_hash,
                "decision_record_hash": decision.content_hash,
                "experiment_spec_hash": config.experiment_spec_hash,
                "judge_bundle_hash": config.judge_bundle_hash,
                "cohort_promotion_hash": cohort_promotion_hash,
                "cohort_id": f"{config.campaign_id}-35b",
                "evidence_hashes": [child.content_hash],
                "summary_hash": summary.content_hash,
                "run_record_hash": run_record.content_hash,
                "terminal_action": "stop_and_transfer",
                "status": "immutable",
            },
        )
        final_eval = _fixture_final_evaluation(root_path, config, run_record)
        effects.extend(("cluster", "final-eval"))
        matrix = {phase: "green" for phase in PHASES}
        return ClosedLoopSnapshot(
            "succeeded",
            "FINAL_EVAL",
            matrix,
            readiness,
            tuple(events),
            run_record,
            decision,
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
        traces = [
            store.put(
                "TrainingTrace",
                "1.0.0",
                {
                    "purpose": "training_allowed",
                    "source_id": f"fixture-source-{index:03d}",
                    "trace_id": f"trace-{index:03d}",
                    "prompt": f"fixture prompt {index}",
                    "response": f"fixture response {index}",
                },
            )
            for index in range(100)
        ]
        fit_trajectories = [
            store.put(
                "FitTrajectory",
                "1.0.0",
                {
                    "fit_index": fit_index,
                    "status": "committed",
                    "trace_id": f"trace-{fit_index % 100:03d}",
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
        judge_pack = store.put(
            "JudgePack",
            "1.0.0",
            {"judge_pack_id": f"{campaign_id}-judge-pack", "status": "committed", "version": "fixture-v1"},
        )
        dataset = store.put(
            "DatasetVersion",
            "1.0.0",
            {
                "dataset_version_id": f"{campaign_id}-dataset",
                "purpose": "training_allowed",
                "trace_count": 100,
                "trace_refs": [trace.content_hash for trace in traces],
                "trace_set_hash": sha256_hex(canonical_json_bytes([trace.content_hash for trace in traces])),
            },
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
