"""Ticket 27: the fail-closed, production-shaped 122B short run.

This module is intentionally a controller seam, rather than a second trainer.
It consumes the durable output of Ticket 26, validates an independently authored
and approved ExperimentSpec, then delegates reward and checkpoint semantics to
the existing CFS and checkpoint-first workflows.  The fixture adapter is the
only implementation that can become ready in this repository; production
readiness is an explicit blocked report and never submits a job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.data.models import DatasetValidationError
from clawrl.data.validation import LoadedTrainingDataset, load_training_dataset
from clawrl.governor.six_arm_cohort import SixArmCohortError, SixArmCohortWorkflow
from clawrl.judge.recertification_122b import (
    Fixture122BRecertificationConfig,
    Recertification122BError,
    Recertification122BWorkflow,
)
from clawrl.training.cluster_lifecycle import (
    ClusterLifecycleWorkflow,
    FixtureClusterLifecycleConfig,
)
from clawrl.training.durable_step_applied import DurableStepAppliedConfig, DurableStepAppliedWorkflow
from clawrl.training.reward_roundtrip import (
    FenceAuthority,
    FixtureCfsBackend,
    FixtureCfsConfig,
    RewardRoundtripError,
    RewardRoundtripWorkflow,
    RewardSlotKey,
)
from clawrl.training.run_journal import RunJournal, RunJournalError

_HASH = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


class Gated122BRunError(RuntimeError):
    """A 122B gate or its durable short run failed closed."""


class Gated122BReadinessError(Gated122BRunError):
    """The bundle, independent spec, or adapter is not ready."""


@dataclass(frozen=True, slots=True)
class Independent122BExperimentSpecConfig:
    """Explicit, independently authored 122B trainer configuration."""

    experiment_id: str
    dataset_version_hash: str
    judge_bundle_hash: str
    optimizer: str = "adamw"
    learning_rate_micros: int = 100
    parallelism: dict[str, JsonValue] | None = None
    resource: dict[str, JsonValue] | None = None
    retry: dict[str, JsonValue] | None = None
    monitoring: dict[str, JsonValue] | None = None
    approval_hash: str = ""
    approved_by: str = ""
    authoring_mode: str = "independent"
    model_size: str = "122B"

    def __post_init__(self) -> None:
        if _ID.fullmatch(self.experiment_id) is None:
            raise Gated122BReadinessError("ExperimentSpec experiment_id is invalid")
        for name in ("dataset_version_hash", "judge_bundle_hash"):
            value = getattr(self, name)
            if type(value) is not str or _HASH.fullmatch(value) is None:
                raise Gated122BReadinessError(f"ExperimentSpec {name} is invalid")
        if not self.optimizer or type(self.optimizer) is not str:
            raise Gated122BReadinessError("ExperimentSpec optimizer is missing")
        if type(self.learning_rate_micros) is not int or self.learning_rate_micros <= 0:
            raise Gated122BReadinessError("ExperimentSpec learning rate is invalid")
        if self.authoring_mode != "independent" or self.model_size != "122B":
            raise Gated122BReadinessError("122B spec must be independently authored for 122B")
        if _HASH.fullmatch(self.approval_hash) is None or _ID.fullmatch(self.approved_by) is None:
            raise Gated122BReadinessError("122B ExperimentSpec approval is missing")
        for name, value in (
            ("parallelism", self.parallelism),
            ("resource", self.resource),
            ("retry", self.retry),
            ("monitoring", self.monitoring),
        ):
            if not isinstance(value, dict) or not value:
                raise Gated122BReadinessError(f"122B ExperimentSpec {name} is missing")

    def payload(self) -> dict[str, JsonValue]:
        return {
            "approval_hash": self.approval_hash,
            "approved_by": self.approved_by,
            "authoring_mode": self.authoring_mode,
            "dataset_version_hash": self.dataset_version_hash,
            "experiment_id": self.experiment_id,
            "judge_bundle_hash": self.judge_bundle_hash,
            "learning_rate_micros": self.learning_rate_micros,
            "model_size": self.model_size,
            "monitoring": cast(dict[str, JsonValue], self.monitoring),
            "optimizer": self.optimizer,
            "parallelism": cast(dict[str, JsonValue], self.parallelism),
            "resource": cast(dict[str, JsonValue], self.resource),
            "retry": cast(dict[str, JsonValue], self.retry),
            "status": "approved",
            "schema_version": "independent-122b-experiment-spec/1.0.0",
        }


@dataclass(frozen=True, slots=True)
class Gated122BRunConfig:
    run_id: str
    dataset_version_hash: str
    judge_bundle_hash: str
    experiment_spec_hash: str
    execution_profile: Literal["fixture", "production"] = "fixture"
    global_step: int = 0
    trainer_path: Literal["classic", "v1"] = "classic"

    def __post_init__(self) -> None:
        if _ID.fullmatch(self.run_id) is None:
            raise Gated122BRunError("run_id is invalid")
        for name in ("dataset_version_hash", "judge_bundle_hash", "experiment_spec_hash"):
            if _HASH.fullmatch(cast(str, getattr(self, name))) is None:
                raise Gated122BRunError(f"{name} is invalid")
        if self.execution_profile not in {"fixture", "production"} or self.global_step != 0:
            raise Gated122BRunError("122B fixture short run is fixed to step zero")


@dataclass(frozen=True, slots=True)
class Gated122BRunSnapshot:
    readiness: Artifact
    reward: Artifact | None
    checkpoint: Artifact | None
    step_applied: Artifact | None
    monitor: Artifact | None
    run_record: Artifact | None
    decision: Artifact | None
    events: tuple[Artifact, ...]
    terminal: bool

    @property
    def readiness_report(self) -> Artifact:
        return self.readiness


# Friendly aliases used by integrations and ticket-oriented callers.
Fixture122BRunConfig = Gated122BRunConfig
Gated122BConfig = Gated122BRunConfig
IndependentExperimentSpecConfig = Independent122BExperimentSpecConfig
Fixture122BExperimentSpecConfig = Independent122BExperimentSpecConfig
Independent122BSpecConfig = Independent122BExperimentSpecConfig


class Gated122BRunWorkflow:
    """Gate a single 122B optimizer update behind all durable prerequisites."""

    @staticmethod
    def production_readiness(root: str | Path, config: object | None = None) -> Artifact:
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": code, "status": "blocked"}
                    for code in (
                        "INFERENCE_MODEL_UNAVAILABLE",
                        "REAL_SCORER_UNAVAILABLE",
                        "CLUSTER_ADAPTER_UNAVAILABLE",
                        "TRAIN_122B_RESOURCE_CONFIG_UNAVAILABLE",
                        "TRAIN_122B_CREDENTIALS_UNAVAILABLE",
                    )
                ],
                "execution_profile": "production",
                "phase": "TRAIN_122B",
                "side_effects_permitted": False,
                "status": "blocked",
                "submit_attempted": False,
            },
        )

    @staticmethod
    def build_experiment_spec(root: str | Path, config: Independent122BExperimentSpecConfig) -> Artifact:
        """Persist a new spec; it never accepts or copies a 35B spec."""
        store = ArtifactStore(root)
        approval = store.put(
            "ExperimentSpecApproval",
            "1.0.0",
            {
                "approval_hash": config.approval_hash,
                "approved_by": config.approved_by,
                "authoring_mode": config.authoring_mode,
                "experiment_id": config.experiment_id,
                "model_size": config.model_size,
                "status": "approved",
            },
        )
        return store.put(
            "ExperimentSpec",
            "1.0.0",
            {**config.payload(), "approval_evidence_hash": approval.content_hash},
        )

    @classmethod
    def run(cls, root: str | Path, *, config: Gated122BRunConfig, epoch: int = 1) -> Gated122BRunSnapshot:
        if config.execution_profile == "production":
            return Gated122BRunSnapshot(
                readiness=cls.production_readiness(root),
                reward=None,
                checkpoint=None,
                step_applied=None,
                monitor=None,
                run_record=None,
                decision=None,
                events=(),
                terminal=False,
            )
        root_path = Path(root)
        store = ArtifactStore(root_path)
        # No artifact is written before every immutable input is validated.
        dataset, bundle, spec = cls._validate_gate(store, config)
        readiness = store.put(
            "ReadinessReport",
            "1.0.0",
            {
                "bundle_hash": bundle.content_hash,
                "checks": [
                    {"code": "JUDGE_BUNDLE_GREEN", "status": "green"},
                    {"code": "INDEPENDENT_122B_SPEC_GREEN", "status": "green"},
                    {"code": "CLUSTER_ADAPTER_GREEN", "status": "green"},
                    {"code": "TRAIN_122B_READY", "status": "green"},
                ],
                "dataset_version_hash": dataset.dataset_version.content_hash,
                "execution_profile": "fixture",
                "experiment_spec_hash": spec.content_hash,
                "phase": "TRAIN_122B",
                "side_effects_permitted": True,
                "status": "green",
                "submit_attempted": False,
            },
        )
        input_artifact = store.put(
            "Gated122BRunInput",
            "1.0.0",
            {
                "dataset_version_hash": config.dataset_version_hash,
                "experiment_spec_hash": config.experiment_spec_hash,
                "judge_bundle_hash": config.judge_bundle_hash,
                "phase": "TRAIN_122B",
                "run_id": config.run_id,
                "schema_version": "gated-122b-run-input/1.0.0",
            },
        )
        journal = RunJournal(root_path, store, config.run_id, input_schema_name="Gated122BRunInput")
        try:
            journal.reserve_identity(input_artifact.content_hash)
            events = journal.events()
            if events and events[-1].payload.get("event_type") == "RUN_CLOSED":
                started_details = cast(dict[str, object], events[0].payload.get("details", {}))
                persisted_readiness = started_details.get("readiness_hash")
                if isinstance(persisted_readiness, str):
                    readiness = store.read(persisted_readiness, expected_schema_name="ReadinessReport")
                return cls._snapshot(root_path, config, readiness)
            if not events:
                journal.start_run(
                    epoch,
                    input_artifact.content_hash,
                    {
                        "input_hash": input_artifact.content_hash,
                        "phase": "TRAIN_122B",
                        "readiness_hash": readiness.content_hash,
                    },
                )
            events = journal.events()
            if not any(event.payload.get("event_type") == "READINESS_GREEN" for event in events):
                journal.append(epoch, "READINESS_GREEN", {"readiness_hash": readiness.content_hash})

            # Exercise the same typed ClusterAdapter seam used by production;
            # the fixture provider performs no real submit and is restart-safe.
            cluster = ClusterLifecycleWorkflow.run(
                root_path,
                config=FixtureClusterLifecycleConfig(config.run_id, "TRAIN_122B"),
                spec_hash=spec.content_hash,
            )
            reward = cls._reward(root_path, config, bundle, resolver_epoch=epoch)
            events = journal.events()
            if not any(event.payload.get("event_type") == "REWARD_COMMITTED" for event in events):
                journal.append(epoch, "REWARD_COMMITTED", {"reward_hash": reward.content_hash})
            step_ready = cls._step_ready(store, config, spec, reward)
            durable = DurableStepAppliedWorkflow.run(
                root_path,
                config=DurableStepAppliedConfig(
                    config.run_id,
                    config.global_step,
                    spec.content_hash,
                    trainer_path=config.trainer_path,
                    phase="TRAIN_122B",
                ),
                step_ready=step_ready,
                experiment_spec=spec,
            )
            events = journal.events()
            if not any(event.payload.get("event_type") == "CHECKPOINT_COMMITTED" for event in events):
                journal.append(
                    epoch,
                    "CHECKPOINT_COMMITTED",
                    {
                        "checkpoint_hash": durable.checkpoint.content_hash,
                        "step_applied_hash": durable.step_applied.content_hash,
                    },
                )
            monitor = store.put(
                "Observation",
                "1.0.0",
                {
                    "checkpoint_hash": durable.checkpoint.content_hash,
                    "cluster_run_record_hash": cluster.run_record.content_hash,
                    "experiment_spec_hash": spec.content_hash,
                    "global_step": config.global_step,
                    "monitoring_policy": spec.payload["monitoring"],
                    "phase": "TRAIN_122B",
                    "reward_hash": reward.content_hash,
                    "run_id": config.run_id,
                    "status": "healthy",
                },
            )
            run_record = store.put(
                "RunRecord",
                "1.0.0",
                {
                    "checkpoint_hash": durable.checkpoint.content_hash,
                    "experiment_spec_hash": spec.content_hash,
                    "dataset_version_hash": config.dataset_version_hash,
                    "global_step": config.global_step,
                    "judge_bundle_hash": bundle.content_hash,
                    "monitor_observation_hash": monitor.content_hash,
                    "optimizer_update_count": 1,
                    "phase": "TRAIN_122B",
                    "reward_hash": reward.content_hash,
                    "reward_root_hash": cast(str, step_ready.payload["reward_root_hash"]),
                    "run_id": config.run_id,
                    "step_applied_hash": durable.step_applied.content_hash,
                    "status": "succeeded",
                },
            )
            decision = store.put(
                "DecisionRecord",
                "1.0.0",
                {
                    "action": "allow_first_122b_optimizer_update",
                    "checkpoint_hash": durable.checkpoint.content_hash,
                    "dataset_version_hash": config.dataset_version_hash,
                    "experiment_spec_hash": spec.content_hash,
                    "judge_bundle_hash": bundle.content_hash,
                    "readiness_hash": readiness.content_hash,
                    "reward_hash": reward.content_hash,
                    "run_id": config.run_id,
                    "step_applied_hash": durable.step_applied.content_hash,
                    "status": "committed",
                },
            )
            events = journal.events()
            if not any(event.payload.get("event_type") == "MONITOR_OBSERVED" for event in events):
                journal.append(epoch, "MONITOR_OBSERVED", {"observation_hash": monitor.content_hash})
            if not any(event.payload.get("event_type") == "RUN_RECORD_COMMITTED" for event in events):
                journal.append(
                    epoch,
                    "RUN_RECORD_COMMITTED",
                    {
                        "decision_hash": decision.content_hash,
                        "run_record_hash": run_record.content_hash,
                    },
                )
            events = journal.events()
            if events[-1].payload.get("event_type") != "RUN_CLOSED":
                journal.close(
                    epoch,
                    status="succeeded",
                    reason_code="122B_FIXTURE_SHORT_RUN_SUCCEEDED",
                    expected_sequence=len(events) + 1,
                    expected_previous_hash=events[-1].content_hash,
                )
            return cls._snapshot(root_path, config, readiness)
        except (RunJournalError, ArtifactCorruption, Gated122BRunError):
            raise

    @classmethod
    def resume(cls, root: str | Path, run_id: str, *, epoch: int = 1) -> Gated122BRunSnapshot:
        store = ArtifactStore(root)
        journal = RunJournal(root, store, run_id, input_schema_name="Gated122BRunInput")
        events = journal.events()
        if not events:
            raise Gated122BRunError("122B run has no durable journal")
        details = cast(dict[str, object], events[0].payload["details"])
        inp = store.read(
            cast(str, details["input_hash"]),
            expected_schema_name="Gated122BRunInput",
        )
        config = Gated122BRunConfig(
            run_id,
            cast(str, inp.payload["dataset_version_hash"]),
            cast(str, inp.payload["judge_bundle_hash"]),
            cast(str, inp.payload["experiment_spec_hash"]),
        )
        if events[-1].payload.get("event_type") != "RUN_CLOSED":
            return cls.run(root, config=config, epoch=epoch)
        first_details = cast(dict[str, object], events[0].payload.get("details", {}))
        readiness_hash = first_details.get("readiness_hash")
        if not isinstance(readiness_hash, str):
            raise Gated122BRunError("122B RUN_STARTED event has no readiness identity")
        readiness = store.read(readiness_hash, expected_schema_name="ReadinessReport")
        return cls._snapshot(Path(root), config, readiness)

    @classmethod
    def _validate_gate(cls, store: ArtifactStore, config: Gated122BRunConfig):
        try:
            dataset = load_training_dataset(store, config.dataset_version_hash)
            bundle = store.read(config.judge_bundle_hash, expected_schema_name="JudgeBundle")
            spec = store.read(config.experiment_spec_hash, expected_schema_name="ExperimentSpec")
            cls._validate_bundle(store, dataset, bundle)
            cls._validate_spec(store, spec, config, bundle)
            return dataset, bundle, spec
        except (
            ArtifactCorruption,
            DatasetValidationError,
            KeyError,
            Recertification122BError,
            RewardRoundtripError,
            SixArmCohortError,
            TypeError,
            ValueError,
            Gated122BReadinessError,
        ) as error:
            raise Gated122BReadinessError("TRAIN_122B gate is not green") from error

    @staticmethod
    def _validate_bundle(store: ArtifactStore, dataset: LoadedTrainingDataset, bundle: Artifact) -> None:
        dataset_hash = dataset.dataset_version.content_hash
        expected_trace_ids = {
            cast(str, trace.payload["trace_id"])
            for trace in dataset.training_traces
            if isinstance(trace.payload.get("trace_id"), str)
        }
        p = bundle.payload
        if (
            p.get("certification_phase") != "TRAIN_122B"
            or p.get("status") != "terminal"
            or p.get("version") != "122b-recertified/1.0.0"
            or p.get("aggregation") != "calibrated_scalar"
        ):
            raise Gated122BReadinessError("old or non-terminal JudgeBundle is not accepted")
        if (
            p.get("dataset_version_hash") != dataset_hash
            or p.get("trace_count") != 100
            or p.get("fresh_rollout_count_per_trace") != 32
            or not isinstance(p.get("inference_model_id"), str)
            or not isinstance(p.get("reward_schema_hash"), str)
            or not isinstance(p.get("scalarizer_hash"), str)
            or not isinstance(p.get("algorithm_contract_hash"), str)
            or not isinstance(p.get("transfer_candidate_hash"), str)
        ):
            raise Gated122BReadinessError("JudgeBundle coverage is partial or bound to another dataset")
        for name, schema in (
            (p["reward_schema_hash"], "RewardSchema"),
            (p["scalarizer_hash"], "Scalarizer"),
            (p["algorithm_contract_hash"], "RLAlgorithmContract"),
        ):
            store.read(cast(str, name), expected_schema_name=schema)
        RewardRoundtripWorkflow._contracts(store, cast(str, p["reward_schema_hash"]), cast(str, p["scalarizer_hash"]))
        algorithm = store.read(cast(str, p["algorithm_contract_hash"]), expected_schema_name="RLAlgorithmContract")
        if (
            algorithm.payload.get("schema_version") != "rl-algorithm-contract/1.0.0"
            or algorithm.payload.get("reward_aggregation") != "calibrated_scalar"
            or not isinstance(algorithm.payload.get("algorithm_id"), str)
            or not isinstance(algorithm.payload.get("advantage_estimator"), str)
        ):
            raise Gated122BReadinessError("RLAlgorithmContract is not calibrated-scalar compatible")
        candidate = store.read(cast(str, p["transfer_candidate_hash"]), expected_schema_name="TransferCandidate")
        Gated122BRunWorkflow._validate_transfer_candidate(store, candidate)
        coverage = store.read(
            cast(str, p["coverage_manifest_hash"]),
            expected_schema_name="RecertificationCoverageManifest",
        )
        coverage_reports = coverage.payload.get("report_hashes")
        if (
            coverage.payload.get("status") != "complete"
            or coverage.payload.get("trace_count") != 100
            or coverage.payload.get("dataset_version_hash") != dataset_hash
            or coverage.payload.get("inference_model_id") != p.get("inference_model_id")
            or coverage.payload.get("reward_schema_hash") != p.get("reward_schema_hash")
            or coverage.payload.get("scalarizer_hash") != p.get("scalarizer_hash")
            or not isinstance(coverage_reports, list)
            or len(coverage_reports) != 100
        ):
            raise Gated122BReadinessError("122B JudgeBundle coverage manifest is incomplete")
        reports = p.get("report_hashes")
        if (
            not isinstance(reports, list)
            or reports != coverage_reports
            or len(reports) != 100
            or len(set(reports)) != 100
        ):
            raise Gated122BReadinessError("122B JudgeBundle report coverage is incomplete")
        observed_trace_ids: set[str] = set()
        recert_config = Fixture122BRecertificationConfig(
            run_id="ticket27-bundle-validator",
            transfer_candidate_hash=cast(str, p["transfer_candidate_hash"]),
            dataset_version_hash=dataset_hash,
            reward_schema_hash=cast(str, p["reward_schema_hash"]),
            scalarizer_hash=cast(str, p["scalarizer_hash"]),
            algorithm_contract_hash=cast(str, p["algorithm_contract_hash"]),
            inference_model_id=cast(str, p["inference_model_id"]),
        )
        recert_state = SimpleNamespace(config=recert_config, store=store, dataset=dataset)
        for report_hash in reports:
            report = store.read(cast(str, report_hash), expected_schema_name="CertificationReport")
            trace_id = report.payload.get("trace_id")
            if not isinstance(trace_id, str) or trace_id in observed_trace_ids:
                raise Gated122BReadinessError("JudgeBundle report trace identity is duplicated")
            try:
                Recertification122BWorkflow._validate_report(cast(Any, recert_state), report, trace_id)
            except Recertification122BError as error:
                raise Gated122BReadinessError("JudgeBundle contains uncertified or stale report") from error
            if report.payload.get("status") != "certified" or report.payload.get("variant") != "success":
                raise Gated122BReadinessError("JudgeBundle contains an uncertifiable report")
            observed_trace_ids.add(trace_id)
        if len(expected_trace_ids) != 100 or observed_trace_ids != expected_trace_ids:
            raise Gated122BReadinessError("JudgeBundle report coverage does not match DatasetVersion traces")

    @staticmethod
    def _validate_transfer_candidate(store: ArtifactStore, candidate: Artifact) -> None:
        payload = candidate.payload
        if payload.get("status") != "immutable" or payload.get("terminal_action") != "stop_and_transfer":
            raise Gated122BReadinessError("JudgeBundle transfer candidate is not terminal")
        summary_hash = payload.get("summary_hash")
        evidence = payload.get("evidence_hashes")
        cohort_id = payload.get("cohort_id")
        if (
            not isinstance(summary_hash, str)
            or not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(item, str) for item in evidence)
            or len(set(evidence)) != len(evidence)
            or not isinstance(cohort_id, str)
        ):
            raise Gated122BReadinessError("TransferCandidate lineage is incomplete")
        summary = store.read(summary_hash, expected_schema_name="ExperimentSummary")
        if (
            summary.payload.get("branch") != "stop_and_transfer"
            or summary.payload.get("status") != "terminal"
            or summary.payload.get("evidence_hashes") != evidence
            or summary.payload.get("causal_boundary") != "controlled_comparison_only; black_box_descriptive_only"
        ):
            raise Gated122BReadinessError("TransferCandidate summary is not terminal stop evidence")
        for child_hash in evidence:
            child = store.read(cast(str, child_hash), expected_schema_name="GovernorChild")
            if child.payload.get("status") != "completed" or child.payload.get("branch") != "stop_and_transfer":
                raise Gated122BReadinessError("TransferCandidate child evidence is not terminal")
        cohort = SixArmCohortWorkflow.resume(store.root, cohort_id)
        if not cohort.terminal or cohort.promotion is None:
            raise Gated122BReadinessError("TransferCandidate cohort is not terminal")
        try:
            Recertification122BWorkflow._validate_governor_lineage(
                store.root, store, candidate, summary_hash, cast(list[str], evidence)
            )
        except Recertification122BError as error:
            raise Gated122BReadinessError("TransferCandidate Governor lineage is invalid") from error

    @staticmethod
    def _validate_spec(store: ArtifactStore, spec: Artifact, config: Gated122BRunConfig, bundle: Artifact) -> None:
        p = spec.payload
        if (
            p.get("dataset_version_hash") != config.dataset_version_hash
            or p.get("judge_bundle_hash") != bundle.content_hash
        ):
            raise Gated122BReadinessError("ExperimentSpec does not bind the new 122B bundle")
        if p.get("model_size") != "122B" or p.get("authoring_mode") != "independent" or p.get("status") != "approved":
            raise Gated122BReadinessError("ExperimentSpec is not independently approved for 122B")
        if (
            any(
                not isinstance(p.get(field), dict) or not cast(dict[str, object], p[field])
                for field in ("parallelism", "resource", "retry", "monitoring")
            )
            or not isinstance(p.get("optimizer"), str)
            or not p.get("optimizer")
            or not any(p.get(field) for field in ("learning_rate_micros", "learning_rate", "lr"))
        ):
            raise Gated122BReadinessError(
                "ExperimentSpec lacks explicit optimizer/LR/parallelism/resource/retry/monitoring"
            )
        learning_rate = next(
            (p.get(field) for field in ("learning_rate_micros", "learning_rate", "lr") if p.get(field)),
            None,
        )
        if type(learning_rate) is not int or cast(int, learning_rate) <= 0:
            raise Gated122BReadinessError("ExperimentSpec learning rate is invalid")
        if any(
            key in p
            for key in (
                "copied_from_35b",
                "derived_from_35b_spec_hash",
                "source_35b_spec_hash",
                "auto_copied",
                "derived_from_35b",
                "copied_from_spec_hash",
                "source_experiment_spec_hash",
                "origin_spec_hash",
                "auto_generated",
                "generated_from",
            )
        ):
            raise Gated122BReadinessError("automatically copied 35B ExperimentSpec is forbidden")
        if (
            _HASH.fullmatch(cast(str, p.get("approval_hash", ""))) is None
            or _ID.fullmatch(cast(str, p.get("approved_by", ""))) is None
        ):
            raise Gated122BReadinessError("ExperimentSpec approval identity is missing")
        approval_hash = p.get("approval_evidence_hash")
        if not isinstance(approval_hash, str):
            raise Gated122BReadinessError("ExperimentSpec approval evidence is missing")
        approval = store.read(approval_hash, expected_schema_name="ExperimentSpecApproval")
        if approval.payload != {
            "approval_hash": p.get("approval_hash"),
            "approved_by": p.get("approved_by"),
            "authoring_mode": "independent",
            "experiment_id": p.get("experiment_id"),
            "model_size": "122B",
            "status": "approved",
        }:
            raise Gated122BReadinessError("ExperimentSpec approval evidence is not bound")

    @staticmethod
    def _reward(root: Path, config: Gated122BRunConfig, bundle: Artifact, *, resolver_epoch: int) -> Artifact:
        store = ArtifactStore(root)
        dataset = load_training_dataset(store, config.dataset_version_hash)
        if not dataset.training_traces:
            raise Gated122BRunError("122B reward cannot bind an empty DatasetVersion")
        trace_id = dataset.training_traces[0].payload.get("trace_id")
        if not isinstance(trace_id, str):
            raise Gated122BRunError("122B reward trace identity is invalid")
        probe = FixtureCfsBackend(root, FixtureCfsConfig("ticket27-122b-cfs")).probe()
        authority = FenceAuthority(root)
        try:
            current = authority.current()
        except RewardRoundtripError:
            authority.advance(resolver_epoch)
            current = authority.current()
        except Exception as error:
            raise Gated122BRunError("122B reward fencing authority is unavailable") from error
        current_epoch = current.payload.get("epoch")
        if type(current_epoch) is not int or cast(int, current_epoch) <= 0:
            raise Gated122BRunError("122B reward resolver fencing epoch is invalid")
        if cast(int, current_epoch) > resolver_epoch:
            raise Gated122BRunError("122B reward resolver epoch is stale")
        if cast(int, current_epoch) < resolver_epoch:
            authority.advance(resolver_epoch)
            current_epoch = resolver_epoch
        key = RewardSlotKey(config.run_id, config.global_step, trace_id, 0, bundle.content_hash)
        snapshot = RewardRoundtripWorkflow.start(
            root,
            key=key,
            probe_evidence_hash=probe.content_hash,
            trajectory_payload={
                "phase": "TRAIN_122B",
                "judge_bundle_hash": bundle.content_hash,
                "trace_id": trace_id,
            },
            reward_schema_hash=cast(str, bundle.payload["reward_schema_hash"]),
            scalarizer_hash=cast(str, bundle.payload["scalarizer_hash"]),
        )
        if snapshot.resolved_reward is not None:
            return snapshot.resolved_reward
        attempt = RewardRoundtripWorkflow.claim_attempt(
            root, key=key, resolver_epoch=cast(int, current_epoch), clock_tick=0
        )
        RewardRoundtripWorkflow.commit_result(
            root,
            key=key,
            attempt_ordinal=cast(int, attempt.payload["attempt_ordinal"]),
            result_payload={
                "confidence_basis_points": 9_500,
                "failure_tags": [],
                "reward_micros": 75_000_000,
                "reward_schema_hash": cast(str, bundle.payload["reward_schema_hash"]),
                "scalarizer_hash": cast(str, bundle.payload["scalarizer_hash"]),
                "turn_local_tie_groups": [[trace_id]],
            },
        )
        return RewardRoundtripWorkflow.resolve(root, key=key, resolver_epoch=cast(int, current_epoch))

    @staticmethod
    def _step_ready(store: ArtifactStore, config: Gated122BRunConfig, spec: Artifact, reward: Artifact) -> Artifact:
        root_hash = sha256_hex(canonical_json_bytes([reward.content_hash]))
        return store.put(
            "StepReady",
            "1.0.0",
            {
                "experiment_spec_hash": spec.content_hash,
                "expected_slot_count": 1,
                "global_step": config.global_step,
                "optimizer_update": False,
                "reward_hashes": [reward.content_hash],
                "reward_root_hash": root_hash,
                "run_id": config.run_id,
                "status": "step_ready",
                "trainer_path": config.trainer_path,
            },
        )

    @staticmethod
    def _snapshot(root: Path, config: Gated122BRunConfig, readiness: Artifact) -> Gated122BRunSnapshot:
        store = ArtifactStore(root)
        journal = RunJournal(root, store, config.run_id, input_schema_name="Gated122BRunInput")
        events = tuple(journal.events())
        reward = checkpoint = marker = monitor = run_record = decision = None
        for event in events:
            details = cast(dict[str, object], event.payload.get("details", {}))
            for key, expected, target in (
                ("reward_hash", "ResolvedReward", "reward"),
                ("checkpoint_hash", "DurableCheckpointManifest", "checkpoint"),
                ("step_applied_hash", "StepApplied", "marker"),
                ("observation_hash", "Observation", "monitor"),
                ("run_record_hash", "RunRecord", "run_record"),
                ("decision_hash", "DecisionRecord", "decision"),
            ):
                value = details.get(key)
                if isinstance(value, str):
                    try:
                        artifact = store.read(value, expected_schema_name=expected)
                    except ArtifactCorruption:
                        continue
                    if target == "reward":
                        reward = artifact
                    elif target == "checkpoint":
                        checkpoint = artifact
                    elif target == "marker":
                        marker = artifact
                    elif target == "monitor":
                        monitor = artifact
                    elif target == "run_record":
                        run_record = artifact
                    else:
                        decision = artifact
        # Recover artifacts when the event was committed before a process crash.
        if reward is None:
            dataset = load_training_dataset(store, config.dataset_version_hash)
            trace_id = dataset.training_traces[0].payload.get("trace_id")
            if not isinstance(trace_id, str):
                raise Gated122BRunError("durable reward trace identity cannot be recovered")
            slot_key = RewardSlotKey(config.run_id, config.global_step, trace_id, 0, config.judge_bundle_hash)
            ref = root / "reward-slots" / slot_key.content_hash / "resolved.ref"
            if ref.exists():
                reward = store.read(ref.read_text().strip(), expected_schema_name="ResolvedReward")
        if checkpoint is None:
            ref = root / "durable-step-applied" / config.run_id / "0" / config.trainer_path / "checkpoint.ref"
            if ref.exists():
                checkpoint = store.read(
                    ref.read_text().strip(),
                    expected_schema_name="DurableCheckpointManifest",
                )
        if marker is None:
            ref = root / "durable-step-applied" / config.run_id / "0" / config.trainer_path / "step-applied.ref"
            if ref.exists():
                marker = store.read(ref.read_text().strip(), expected_schema_name="StepApplied")
        terminal = bool(events and events[-1].payload.get("event_type") == "RUN_CLOSED")
        return Gated122BRunSnapshot(
            readiness,
            reward,
            checkpoint,
            marker,
            monitor,
            run_record,
            decision,
            events,
            terminal,
        )


Gated122BWorkflow = Gated122BRunWorkflow
Fixture122BRunWorkflow = Gated122BRunWorkflow
Run122BWorkflow = Gated122BRunWorkflow
FixtureGated122BRunWorkflow = Gated122BRunWorkflow
