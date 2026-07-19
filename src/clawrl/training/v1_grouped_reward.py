"""Production-shaped verl v1/TransferQueue grouped reward fixture.

The real verl v1 runtime is intentionally kept behind the production
readiness gate.  This module provides the fixture seam used by contract tests:
the complete UID group is published before routing, transport can return items
in any order, and the trainer only receives a tensor after all 128 slots have
resolved.  Slot identity is persisted by Ticket 12's TransferQueue and never
derived from queue keys or response content.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from clawrl.artifacts import (
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    ImmutableArtifactConflict,
    JsonValue,
    UnknownSchemaMajor,
)
from clawrl.router.trace_router import RouterCapacityConfig, TraceRouterError
from clawrl.training.classic_grouped_reward import (
    ClassicGroupedRewardConfig,
    ClassicGroupedRewardError,
    ClassicGroupedRewardWorkflow,
    InjectedClassicGroupedRewardCrash,
)
from clawrl.training.classic_identity import ClassicTrajectoryIdentity
from clawrl.training.expected_trajectory_set import ExpectedTrajectorySetWorkflow
from clawrl.training.v1_transfer_identity import (
    PersistentFixtureTransferQueue,
    TransferQueueSlot,
    V1RewardEnvelope,
    V1RewardIdentityError,
    V1RewardManager,
)

_HASH = set("0123456789abcdef")
_EXPERIMENT_FIELDS = {
    "aggregation",
    "dataset_version_hash",
    "dataset_version_id",
    "experiment_id",
    "judge_bundle_hash",
    "trace_set_hash",
}


class V1GroupedRewardError(RuntimeError):
    """The v1 grouped reward seam failed closed before an optimizer update."""


class InjectedV1GroupedRewardCrash(V1GroupedRewardError):
    """Synthetic process crash after a durable TransferQueue checkpoint."""


def _hash(value: object, field: str) -> str:
    if type(value) is not str or len(value) != 64 or set(cast(str, value)) - _HASH:
        raise V1GroupedRewardError(f"{field} is not a content hash")
    return cast(str, value)


@dataclass(frozen=True, slots=True)
class V1GroupedRewardConfig:
    run_id: str
    global_step: int
    uid: str
    experiment_spec_hash: str
    resolver_epoch: int = 1
    chunk_size: int = 16
    capacity: RouterCapacityConfig = RouterCapacityConfig(5)
    aggregation: str = "calibrated_scalar"
    prompt_hash: str | None = None
    scalarizer_hash: str | None = None
    scalarizer_version: str | None = None
    algorithm_contract: str | None = None
    initial_return_ordinals: tuple[int, ...] = (0,)
    fault_kind: str | None = None
    fault_index: int | None = None

    def __post_init__(self) -> None:
        if not all(type(item) is str and item for item in (self.run_id, self.uid)):
            raise V1GroupedRewardError("v1 grouped identity is invalid")
        if type(self.global_step) is not int or self.global_step < 0:
            raise V1GroupedRewardError("global_step is invalid")
        _hash(self.experiment_spec_hash, "experiment_spec_hash")
        if type(self.resolver_epoch) is not int or self.resolver_epoch < 1:
            raise V1GroupedRewardError("resolver_epoch is invalid")
        if type(self.chunk_size) is not int or not 1 <= self.chunk_size <= 128:
            raise V1GroupedRewardError("v1 worker chunk_size is invalid")
        if self.aggregation not in {"calibrated_scalar", "hierarchical_rank", "local_microgroup_rank"}:
            raise V1GroupedRewardError("aggregation is invalid")
        for field, value in (("prompt_hash", self.prompt_hash), ("scalarizer_hash", self.scalarizer_hash)):
            if value is not None:
                _hash(value, field)
        if (self.fault_kind is None) != (self.fault_index is None):
            raise V1GroupedRewardError("fault kind/index must be configured together")
        if self.fault_kind not in {None, "missing", "nan", "version"}:
            raise V1GroupedRewardError("unknown fixture fault")
        if self.fault_index is not None and (type(self.fault_index) is not int or self.fault_index < 0):
            raise V1GroupedRewardError("fault index is invalid")
        if not self.initial_return_ordinals or len(set(self.initial_return_ordinals)) != len(
            self.initial_return_ordinals
        ):
            raise V1GroupedRewardError("initial return ordinals are invalid")
        if any(type(item) is not int or item < 0 or item >= 128 for item in self.initial_return_ordinals):
            raise V1GroupedRewardError("initial return ordinal is invalid")


# Naming used by callers that refer to the transport rather than the trainer.
V1TransferQueueRewardConfig = V1GroupedRewardConfig


@dataclass(frozen=True, slots=True)
class V1GroupedRewardSnapshot:
    report: Artifact
    barrier: Artifact
    rewards: tuple[Artifact, ...]
    trainer_output: Artifact
    dump: Artifact


class V1GroupedRewardWorkflow:
    """Durable v1 grouped calibrated-scalar trainer seam."""

    @staticmethod
    def _namespace(root: str | Path, config: V1GroupedRewardConfig) -> Path:
        return Path(root) / "v1-grouped-reward" / config.run_id / str(config.global_step) / config.uid

    @classmethod
    def production_readiness(cls, root: str | Path, *, phase: str = "TRAIN_35B") -> Artifact:
        if phase not in {"TRAIN_35B", "TRAIN_122B"}:
            raise V1GroupedRewardError("production phase is invalid")
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": "REAL_VERL_V1_TRANSFER_QUEUE_UNVERIFIED", "status": "blocked"},
                    {"code": "PRODUCTION_CFS_PUBLISH_AND_BARRIER_UNVERIFIED", "status": "blocked"},
                    {"code": "PRODUCTION_TRACE_ACL_AND_ENCRYPTION_UNVERIFIED", "status": "blocked"},
                ],
                "execution_profile": "production",
                "phase": phase,
                "scorer_request_attempted": False,
                "side_effects_permitted": False,
                "status": "blocked",
                "variant": "v1/colocated-async-transfer-queue",
            },
        )

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: V1GroupedRewardConfig,
        expected_set: Artifact,
        judge_pack: Artifact,
        experiment_spec: Artifact,
        crash_after_dispatches: int | None = None,
        crash_after_checkpoint: bool = False,
        crash_after_requests: int | None = None,
    ) -> V1GroupedRewardSnapshot:
        """Run the whole group.  All 128 dispatches are durable before routing."""

        store = ArtifactStore(root)
        cls._validate_inputs(config, expected_set, judge_pack, experiment_spec)
        namespace = cls._namespace(root, config)
        report_ref = namespace / "report.ref"
        if report_ref.exists():
            snapshot = cls.resume(root, config)
            if (
                snapshot.report.payload.get("experiment_spec_hash") != experiment_spec.content_hash
                or snapshot.report.payload.get("expected_set_hash") != expected_set.content_hash
                or snapshot.report.payload.get("judge_pack_hash") != judge_pack.content_hash
            ):
                raise V1GroupedRewardError("experiment identity changed on resume")
            return snapshot

        manifests = cls._manifests(root, expected_set, config)
        if len(manifests) != 128:
            raise V1GroupedRewardError("v1 UID group requires exactly 128 manifests")
        queue = PersistentFixtureTransferQueue(root, config.run_id)
        dispatches: list[Artifact] = []
        if crash_after_dispatches is not None and (crash_after_dispatches < 0 or crash_after_dispatches > 128):
            raise V1GroupedRewardError("crash dispatch index is invalid")
        if crash_after_dispatches == 0:
            raise InjectedV1GroupedRewardCrash("crash before TransferQueue dispatch")
        for ordinal, manifest in enumerate(manifests):
            trajectory = manifest.payload.get("trajectory")
            if not isinstance(trajectory, dict):
                raise V1GroupedRewardError("trajectory manifest is invalid")
            try:
                identity = ClassicTrajectoryIdentity.from_mapping(trajectory.get("identity"))
            except Exception as error:
                raise V1GroupedRewardError("trajectory identity is invalid") from error
            envelope = V1RewardEnvelope(ordinal, identity, cast(str, trajectory.get("prompt")), config.global_step)
            dispatches.append(queue.dispatch(TransferQueueSlot.from_envelope(envelope)))
            if (
                crash_after_dispatches is not None
                and crash_after_dispatches < 128
                and ordinal + 1 == crash_after_dispatches
            ):
                raise InjectedV1GroupedRewardCrash("crash after full-group dispatch prefix")
        if crash_after_requests is not None:
            # Compatibility spelling used by the classic seam.  Requests are
            # intentionally not singleton awaited in v1; dispatch is the
            # durable all-group publication boundary.
            if crash_after_requests < 0 or crash_after_requests > 128:
                raise V1GroupedRewardError("crash request index is invalid")
            if crash_after_requests < 128:
                raise InjectedV1GroupedRewardCrash("crash after nonblocking group publication")

        # Router is called only after the whole UID group is published.
        try:
            classic = ClassicGroupedRewardWorkflow.run(
                root,
                config=ClassicGroupedRewardConfig(
                    config.run_id,
                    config.global_step,
                    config.uid,
                    config.experiment_spec_hash,
                    resolver_epoch=config.resolver_epoch,
                    chunk_size=config.chunk_size,
                    capacity=config.capacity,
                    aggregation=config.aggregation,
                    prompt_hash=config.prompt_hash,
                    scalarizer_hash=config.scalarizer_hash,
                    scalarizer_version=config.scalarizer_version,
                    algorithm_contract=config.algorithm_contract,
                    fault_kind=config.fault_kind,
                    fault_index=config.fault_index,
                ),
                expected_set=expected_set,
                judge_pack=judge_pack,
                experiment_spec=experiment_spec,
            )
        except (ClassicGroupedRewardError, InjectedClassicGroupedRewardCrash) as error:
            raise V1GroupedRewardError(str(error)) from error

        by_ordinal = {ordinal: dispatch for ordinal, dispatch in enumerate(dispatches)}
        initial = set(config.initial_return_ordinals)
        returns: dict[int, Artifact] = {}
        for ordinal in sorted(initial):
            returns[ordinal] = cls._commit_return(queue, by_ordinal[ordinal], manifests[ordinal], config)
        checkpoint = queue.checkpoint(tuple(dispatches), tuple(returns[item] for item in sorted(returns)))
        ArtifactStore.durable_mkdir(namespace)
        cls._publish_ref(namespace / "checkpoint.ref", checkpoint)
        if crash_after_checkpoint:
            raise InjectedV1GroupedRewardCrash("crash after durable TransferQueue checkpoint")
        reissues: dict[int, Artifact] = {}
        for ordinal in reversed(sorted(set(by_ordinal) - set(returns))):
            reissues[ordinal] = queue.reissue(by_ordinal[ordinal], checkpoint)
            returns[ordinal] = cls._commit_return(queue, by_ordinal[ordinal], manifests[ordinal], config)
        # Explicit second pass validates all return refs after reissue and
        # models a fresh process observing the queue state.
        for _ordinal, returned in returns.items():
            queue.validate_return(returned, by_ordinal)

        reward_by_ordinal = {cast(int, item.payload["rollout_index"]): item for item in classic.rewards}
        classic_extra = {
            cast(int, item["rollout_index"]): item
            for item in cast(list[dict[str, JsonValue]], classic.trainer_output.payload["extra_info"])
        }
        if sorted(reward_by_ordinal) != list(range(128)):
            raise V1GroupedRewardError("classic reward output does not cover v1 slots")
        reward_refs: list[Artifact] = []
        for ordinal in range(128):
            reward = reward_by_ordinal[ordinal]
            reward_refs.append(
                store.put(
                    "V1TransferQueueReward",
                    "1.0.0",
                    {
                        "classic_reward_hash": reward.content_hash,
                        "dispatch_hash": by_ordinal[ordinal].content_hash,
                        "extra_info": classic_extra[ordinal],
                        "global_step": config.global_step,
                        "judge_result_hash": reward.payload.get("judge_result_hash"),
                        "reward_micros": reward.payload.get("reward_micros"),
                        "reissue_hash": reissues[ordinal].content_hash if ordinal in reissues else None,
                        "rollout_index": ordinal,
                        "run_id": config.run_id,
                        "trace_id": reward.payload.get("trace_id"),
                        "trace_ref": reward.payload.get("trajectory_manifest_hash"),
                        "trajectory_manifest_hash": reward.payload.get("trajectory_manifest_hash"),
                        "transfer_return_hash": returns[ordinal].content_hash,
                        "uid": config.uid,
                    },
                )
            )
        chunks: list[Artifact] = []
        ordered_transport = list(reversed(reward_refs))
        for index in range(0, 128, config.chunk_size):
            chunk = ordered_transport[index : index + config.chunk_size]
            chunks.append(
                store.put(
                    "V1TransferQueueChunk",
                    "1.0.0",
                    {
                        "chunk_index": len(chunks),
                        "padding_count": max(0, config.chunk_size - len(chunk))
                        if index + config.chunk_size >= 128
                        else 0,
                        "reward_hashes": [item.content_hash for item in chunk],
                        "rollout_indices": [item.payload["rollout_index"] for item in chunk],
                        "reordered": True,
                    },
                )
            )
        dump = store.put(
            "V1TransferQueueRewardDump",
            "1.0.0",
            {
                "chunk_hashes": [item.content_hash for item in chunks],
                "dispatch_hashes": [item.content_hash for item in dispatches],
                "extra_info": [item.payload["extra_info"] for item in reward_refs],
                "reissue_hashes": [reissues[item].content_hash for item in sorted(reissues)],
                "return_hashes": [returns[item].content_hash for item in sorted(returns)],
                "reward_hashes": [item.content_hash for item in reward_refs],
                "reward_micros": [item.payload["reward_micros"] for item in reward_refs],
                "run_id": config.run_id,
                "trace_refs": [item.payload["trace_ref"] for item in reward_refs],
                "global_step": config.global_step,
                "uid": config.uid,
                "status": "dumped",
            },
        )
        barrier = store.put(
            "V1GroupedRewardBarrier",
            "1.0.0",
            {
                "dispatch_hashes": [item.content_hash for item in dispatches],
                "expected_count": 128,
                "expected_set_hash": expected_set.content_hash,
                "judge_pack_hash": judge_pack.content_hash,
                "reward_hashes": [item.content_hash for item in reward_refs],
                "run_id": config.run_id,
                "global_step": config.global_step,
                "status": "trace_ready",
                "uid": config.uid,
            },
        )
        # Keep the trainer-visible schema byte-for-byte compatible with classic;
        # TQ references remain in the durable reward/dump lineage above.
        trainer = store.put(
            "V1TrainerRewardOutput",
            "1.0.0",
            dict(classic.trainer_output.payload) | {"barrier_hash": barrier.content_hash},
        )
        report = store.put(
            "V1GroupedRewardReport",
            "1.0.0",
            {
                "aggregation": config.aggregation,
                "barrier_hash": barrier.content_hash,
                "classic_report_hash": classic.report.content_hash,
                "dump_hash": dump.content_hash,
                "experiment_spec_hash": experiment_spec.content_hash,
                "expected_set_hash": expected_set.content_hash,
                "judge_pack_hash": judge_pack.content_hash,
                "prompt_hash": config.prompt_hash,
                "production_readiness": "blocked",
                "run_id": config.run_id,
                "scalarizer_hash": config.scalarizer_hash,
                "scalarizer_version": config.scalarizer_version,
                "trainer_output_hash": trainer.content_hash,
                "uid": config.uid,
                "algorithm_contract": config.algorithm_contract,
                "resolver_epoch": config.resolver_epoch,
                "chunk_size": config.chunk_size,
                "capacity": config.capacity.max_global_subthreads,
                "initial_return_ordinals": list(config.initial_return_ordinals),
                "fault_kind": config.fault_kind,
                "fault_index": config.fault_index,
                "status": "resolved",
            },
        )
        cls._publish_ref(report_ref, report)
        return cls.resume(root, config)

    @classmethod
    def resume(cls, root: str | Path, config: V1GroupedRewardConfig) -> V1GroupedRewardSnapshot:
        store = ArtifactStore(root)
        namespace = cls._namespace(root, config)
        try:
            report = store.read(
                (namespace / "report.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="V1GroupedRewardReport",
            )
            expected_report_fields = {
                "aggregation",
                "algorithm_contract",
                "barrier_hash",
                "capacity",
                "classic_report_hash",
                "chunk_size",
                "dump_hash",
                "experiment_spec_hash",
                "expected_set_hash",
                "fault_index",
                "fault_kind",
                "initial_return_ordinals",
                "judge_pack_hash",
                "production_readiness",
                "prompt_hash",
                "resolver_epoch",
                "run_id",
                "scalarizer_hash",
                "scalarizer_version",
                "status",
                "trainer_output_hash",
                "uid",
            }
            if report.schema_version != "1.0.0" or set(report.payload) != expected_report_fields:
                raise V1GroupedRewardError("v1 grouped report schema is invalid")
            expected_set = store.read(
                cast(str, report.payload["expected_set_hash"]), expected_schema_name="ExpectedTrajectorySet"
            )
            try:
                expected_snapshot = ExpectedTrajectorySetWorkflow.resume(root, config.run_id, config.global_step)
            except Exception as error:
                raise V1GroupedRewardError("v1 ExpectedTrajectorySet ref cannot be recovered") from error
            if expected_snapshot.expected_set.content_hash != expected_set.content_hash:
                raise V1GroupedRewardError("v1 ExpectedTrajectorySet durable identity changed")
            judge_pack = store.read(cast(str, report.payload["judge_pack_hash"]), expected_schema_name="JudgePack")
            experiment_spec = store.read(
                cast(str, report.payload["experiment_spec_hash"]), expected_schema_name="ExperimentSpec"
            )
            cls._validate_inputs(config, expected_set, judge_pack, experiment_spec)
            try:
                classic_config = ClassicGroupedRewardConfig(
                    config.run_id,
                    config.global_step,
                    config.uid,
                    config.experiment_spec_hash,
                    resolver_epoch=config.resolver_epoch,
                    chunk_size=config.chunk_size,
                    capacity=config.capacity,
                    aggregation=config.aggregation,
                    prompt_hash=config.prompt_hash,
                    scalarizer_hash=config.scalarizer_hash,
                    scalarizer_version=config.scalarizer_version,
                    algorithm_contract=config.algorithm_contract,
                    fault_kind=config.fault_kind,
                    fault_index=config.fault_index,
                )
                classic_snapshot = ClassicGroupedRewardWorkflow.resume(root, classic_config)
            except ClassicGroupedRewardError as error:
                raise V1GroupedRewardError("v1 classic lineage cannot be recovered") from error
            if (
                classic_snapshot.report.content_hash != report.payload.get("classic_report_hash")
                or classic_snapshot.report.payload.get("expected_set_hash") != report.payload.get("expected_set_hash")
                or classic_snapshot.report.payload.get("judge_pack_hash") != report.payload.get("judge_pack_hash")
            ):
                raise V1GroupedRewardError("v1 classic report identity changed")
            barrier = store.read(
                cast(str, report.payload["barrier_hash"]), expected_schema_name="V1GroupedRewardBarrier"
            )
            trainer = store.read(
                cast(str, report.payload["trainer_output_hash"]), expected_schema_name="V1TrainerRewardOutput"
            )
            dump = store.read(cast(str, report.payload["dump_hash"]), expected_schema_name="V1TransferQueueRewardDump")
            expected_barrier_fields = {
                "dispatch_hashes",
                "expected_count",
                "expected_set_hash",
                "global_step",
                "judge_pack_hash",
                "reward_hashes",
                "run_id",
                "status",
                "uid",
            }
            expected_trainer_fields = {
                "barrier_hash",
                "dimensions",
                "extra_info",
                "expected_set_hash",
                "global_step",
                "judge_pack_hash",
                "reward_count",
                "reward_micros",
                "run_id",
                "uid",
            }
            expected_dump_fields = {
                "chunk_hashes",
                "dispatch_hashes",
                "extra_info",
                "global_step",
                "reissue_hashes",
                "return_hashes",
                "reward_hashes",
                "reward_micros",
                "run_id",
                "status",
                "trace_refs",
                "uid",
            }
            if (
                barrier.schema_version != "1.0.0"
                or set(barrier.payload) != expected_barrier_fields
                or trainer.schema_version != "1.0.0"
                or set(trainer.payload) != expected_trainer_fields
                or dump.schema_version != "1.0.0"
                or set(dump.payload) != expected_dump_fields
            ):
                raise V1GroupedRewardError("v1 grouped durable schema is invalid")
        except (ArtifactCorruption, UnknownSchemaMajor, OSError, KeyError, TypeError, ValueError) as error:
            raise V1GroupedRewardError("v1 grouped route cannot be recovered") from error
        if any(
            report.payload.get(field) != expected
            for field, expected in {
                "aggregation": config.aggregation,
                "algorithm_contract": config.algorithm_contract,
                "capacity": config.capacity.max_global_subthreads,
                "chunk_size": config.chunk_size,
                "experiment_spec_hash": config.experiment_spec_hash,
                "fault_index": config.fault_index,
                "fault_kind": config.fault_kind,
                "initial_return_ordinals": list(config.initial_return_ordinals),
                "production_readiness": "blocked",
                "prompt_hash": config.prompt_hash,
                "resolver_epoch": config.resolver_epoch,
                "run_id": config.run_id,
                "scalarizer_hash": config.scalarizer_hash,
                "scalarizer_version": config.scalarizer_version,
                "status": "resolved",
                "uid": config.uid,
            }.items()
        ):
            raise V1GroupedRewardError("v1 grouped config mutation changed durable run")
        if (
            report.payload.get("run_id") != config.run_id
            or report.payload.get("uid") != config.uid
            or report.payload.get("experiment_spec_hash") != config.experiment_spec_hash
            or any(
                report.payload.get(field) != getattr(config, field)
                for field in ("prompt_hash", "scalarizer_hash", "scalarizer_version", "algorithm_contract")
                if getattr(config, field) is not None
            )
            or barrier.payload.get("expected_count") != 128
            or barrier.payload.get("run_id") != config.run_id
            or barrier.payload.get("global_step") != config.global_step
            or barrier.payload.get("uid") != config.uid
            or barrier.payload.get("status") != "trace_ready"
            or trainer.payload.get("barrier_hash") != barrier.content_hash
            or trainer.payload.get("expected_set_hash") != report.payload.get("expected_set_hash")
            or trainer.payload.get("judge_pack_hash") != report.payload.get("judge_pack_hash")
            or trainer.payload.get("global_step") != config.global_step
            or trainer.payload.get("run_id") != config.run_id
            or trainer.payload.get("uid") != config.uid
            or trainer.payload.get("reward_count") != 128
            or len(cast(list[object], trainer.payload.get("reward_micros"))) != 128
            or len(cast(list[object], trainer.payload.get("extra_info"))) != 128
            or barrier.payload.get("dispatch_hashes") != dump.payload.get("dispatch_hashes")
            or barrier.payload.get("reward_hashes") != dump.payload.get("reward_hashes")
            or barrier.payload.get("expected_set_hash") != report.payload.get("expected_set_hash")
            or barrier.payload.get("judge_pack_hash") != report.payload.get("judge_pack_hash")
            or dump.payload.get("status") != "dumped"
            or dump.payload.get("run_id") != config.run_id
            or dump.payload.get("global_step") != config.global_step
            or dump.payload.get("uid") != config.uid
        ):
            raise V1GroupedRewardError("v1 grouped trainer output is incomplete")
        reward_hashes = dump.payload.get("reward_hashes")
        if not isinstance(reward_hashes, list) or len(reward_hashes) != 128:
            raise V1GroupedRewardError("v1 reward dump is incomplete")
        if len(set(reward_hashes)) != 128 or any(type(item) is not str or len(item) != 64 for item in reward_hashes):
            raise V1GroupedRewardError("v1 reward hash set is invalid")
        try:
            rewards = tuple(
                store.read(cast(str, item), expected_schema_name="V1TransferQueueReward") for item in reward_hashes
            )
        except (ArtifactCorruption, UnknownSchemaMajor, OSError, KeyError, TypeError, ValueError) as error:
            raise V1GroupedRewardError("v1 reward artifact cannot be recovered") from error
        expected_reward_fields = {
            "classic_reward_hash",
            "dispatch_hash",
            "extra_info",
            "global_step",
            "judge_result_hash",
            "reissue_hash",
            "reward_micros",
            "rollout_index",
            "run_id",
            "trace_id",
            "trace_ref",
            "trajectory_manifest_hash",
            "transfer_return_hash",
            "uid",
        }
        if any(item.schema_version != "1.0.0" or set(item.payload) != expected_reward_fields for item in rewards):
            raise V1GroupedRewardError("v1 reward schema is invalid")
        if [item.payload.get("rollout_index") for item in rewards] != list(range(128)):
            raise V1GroupedRewardError("v1 reward rollout identity is incomplete")
        classic_by_ordinal = {cast(int, item.payload["rollout_index"]): item for item in classic_snapshot.rewards}
        classic_extra_by_ordinal = {
            cast(int, item["rollout_index"]): item
            for item in cast(list[dict[str, JsonValue]], classic_snapshot.trainer_output.payload["extra_info"])
        }
        for reward in rewards:
            ordinal = cast(int, reward.payload["rollout_index"])
            classic_reward = classic_by_ordinal.get(ordinal)
            if (
                classic_reward is None
                or reward.payload.get("classic_reward_hash") != classic_reward.content_hash
                or reward.payload.get("reward_micros") != classic_reward.payload.get("reward_micros")
                or reward.payload.get("judge_result_hash") != classic_reward.payload.get("judge_result_hash")
                or reward.payload.get("trace_ref") != classic_reward.payload.get("trajectory_manifest_hash")
                or reward.payload.get("extra_info") != classic_extra_by_ordinal.get(ordinal)
            ):
                raise V1GroupedRewardError("v1 reward diverges from classic calibrated result")
        dispatch_hashes_from_dump = dump.payload.get("dispatch_hashes")
        if not isinstance(dispatch_hashes_from_dump, list) or len(dispatch_hashes_from_dump) != 128:
            raise V1GroupedRewardError("v1 dispatch dump is incomplete")
        if any(
            item.payload.get("run_id") != config.run_id
            or item.payload.get("global_step") != config.global_step
            or item.payload.get("uid") != config.uid
            or item.payload.get("dispatch_hash") != dispatch_hashes_from_dump[index]
            or item.payload.get("trace_ref") != item.payload.get("trajectory_manifest_hash")
            or type(item.payload.get("reward_micros")) is not int
            for index, item in enumerate(rewards)
        ):
            raise V1GroupedRewardError("v1 reward lineage is incomplete")
        if (
            dump.payload.get("reward_micros") != [item.payload.get("reward_micros") for item in rewards]
            or dump.payload.get("trace_refs") != [item.payload.get("trace_ref") for item in rewards]
            or dump.payload.get("extra_info") != [item.payload.get("extra_info") for item in rewards]
        ):
            raise V1GroupedRewardError("v1 scalar/extra-info/trace references changed in dump")
        if trainer.payload.get("reward_micros") != [
            item.payload.get("reward_micros") for item in rewards
        ] or trainer.payload.get("extra_info") != [item.payload.get("extra_info") for item in rewards]:
            raise V1GroupedRewardError("v1 trainer tensor does not match TransferQueue slots")
        # A trainer tensor is not recoverable from a dump alone: every TQ slot
        # and its durable checkpoint must still be present and identity-valid.
        try:
            chunk_hashes = dump.payload.get("chunk_hashes")
            if not isinstance(chunk_hashes, list) or not chunk_hashes:
                raise V1GroupedRewardError("v1 chunk dump is incomplete")
            chunks = tuple(
                store.read(cast(str, item), expected_schema_name="V1TransferQueueChunk") for item in chunk_hashes
            )
            transported: list[str] = []
            for chunk in chunks:
                hashes = chunk.payload.get("reward_hashes")
                indices = chunk.payload.get("rollout_indices")
                if (
                    chunk.schema_version != "1.0.0"
                    or set(chunk.payload)
                    != {"chunk_index", "padding_count", "reward_hashes", "rollout_indices", "reordered"}
                    or chunk.payload.get("reordered") is not True
                    or not isinstance(hashes, list)
                    or not isinstance(indices, list)
                    or len(hashes) != len(indices)
                ):
                    raise V1GroupedRewardError("v1 chunk identity is incomplete")
                if indices != [
                    store.read(cast(str, item), expected_schema_name="V1TransferQueueReward").payload.get(
                        "rollout_index"
                    )
                    for item in hashes
                ]:
                    raise V1GroupedRewardError("v1 chunk rollout identity changed")
                transported.extend(cast(str, item) for item in hashes)
            if transported != list(reversed(cast(list[str], reward_hashes))):
                raise V1GroupedRewardError("v1 chunk/reorder lineage changed")
            queue = PersistentFixtureTransferQueue(root, config.run_id)
            checkpoint_hash = (namespace / "checkpoint.ref").read_text(encoding="ascii").strip()
            checkpoint = store.read(checkpoint_hash, expected_schema_name="V1TransferQueueCheckpoint")
            queue.validate_checkpoint(checkpoint)
            dispatch_hashes = dump.payload.get("dispatch_hashes")
            return_hashes = dump.payload.get("return_hashes")
            reissue_hashes = dump.payload.get("reissue_hashes")
            if (
                not isinstance(dispatch_hashes, list)
                or len(dispatch_hashes) != 128
                or not isinstance(return_hashes, list)
                or len(return_hashes) != 128
                or not isinstance(reissue_hashes, list)
                or len(reissue_hashes) != 128 - len(config.initial_return_ordinals)
            ):
                raise V1GroupedRewardError("v1 TransferQueue slot coverage is incomplete")
            dispatches = tuple(
                store.read(cast(str, item), expected_schema_name="V1TransferDispatch") for item in dispatch_hashes
            )
            by_ordinal = {cast(int, item.payload["original_slot_ordinal"]): item for item in dispatches}
            if sorted(by_ordinal) != list(range(128)):
                raise V1GroupedRewardError("v1 TransferQueue dispatch ordinal set is incomplete")
            returns = tuple(
                store.read(cast(str, item), expected_schema_name="V1TransferReturn") for item in return_hashes
            )
            if (
                len(set(return_hashes)) != 128
                or len({cast(int, item.payload["slot_ordinal"]) for item in returns}) != 128
            ):
                raise V1GroupedRewardError("v1 return slot set is incomplete")
            for returned in returns:
                queue.validate_return(returned, by_ordinal)
                ordinal = cast(int, returned.payload["slot_ordinal"])
                persisted_return = queue._read_slot_ref(ordinal, "return.ref", "V1TransferReturn")
                if persisted_return.content_hash != returned.content_hash:
                    raise V1GroupedRewardError("v1 persisted return ref changed")
            dispatch_by_ordinal = {cast(int, item.payload["original_slot_ordinal"]): item for item in dispatches}
            return_by_ordinal = {cast(int, item.payload["slot_ordinal"]): item for item in returns}
            if any(
                reward.payload.get("dispatch_hash") != dispatch_by_ordinal[ordinal].content_hash
                or reward.payload.get("transfer_return_hash") != return_by_ordinal[ordinal].content_hash
                for ordinal, reward in enumerate(rewards)
            ):
                raise V1GroupedRewardError("v1 reward TransferQueue refs changed")
            expected_reissue_ordinals = sorted(set(range(128)) - set(config.initial_return_ordinals))
            reissues = tuple(
                store.read(cast(str, item), expected_schema_name="V1TransferReissue") for item in reissue_hashes
            )
            if len(set(reissue_hashes)) != len(reissues):
                raise V1GroupedRewardError("v1 reissue hash set is invalid")
            for ordinal, reissue in zip(expected_reissue_ordinals, reissues, strict=True):
                persisted_reissue = queue._read_slot_ref(ordinal, "reissue.ref", "V1TransferReissue")
                if (
                    persisted_reissue.content_hash != reissue.content_hash
                    or reissue.payload.get("original_slot_ordinal") != ordinal
                    or reissue.payload.get("slot_ordinal") != ordinal
                    or reissue.payload.get("run_id") != config.run_id
                    or reissue.payload.get("status") != "reissued"
                    or reissue.payload.get("checkpoint_hash") != checkpoint.content_hash
                    or reissue.payload.get("dispatch_hash") != by_ordinal[ordinal].content_hash
                ):
                    raise V1GroupedRewardError("v1 persisted reissue ref changed")
            reissue_by_ordinal = {
                ordinal: reissue for ordinal, reissue in zip(expected_reissue_ordinals, reissues, strict=True)
            }
            if any(
                reward.payload.get("reissue_hash")
                != (reissue_by_ordinal[ordinal].content_hash if ordinal in reissue_by_ordinal else None)
                for ordinal, reward in enumerate(rewards)
            ):
                raise V1GroupedRewardError("v1 reward reissue refs changed")
        except (ArtifactCorruption, UnknownSchemaMajor, OSError, KeyError, TypeError, V1RewardIdentityError) as error:
            raise V1GroupedRewardError("v1 TransferQueue slot is missing or corrupt") from error
        return V1GroupedRewardSnapshot(report, barrier, rewards, trainer, dump)

    @staticmethod
    def _manifests(root: str | Path, expected: Artifact, config: V1GroupedRewardConfig) -> tuple[Artifact, ...]:
        try:
            # Ticket 12 owns the v1 authorization/identity boundary.  Calling
            # it here makes the grouped seam fail closed if a caller supplied
            # a classic authorization or changed transport after freezing.
            V1RewardManager.authorize_expected_trajectory_set(root, expected)
            snapshot = ExpectedTrajectorySetWorkflow.resume(root, config.run_id, config.global_step)
        except Exception as error:
            raise V1GroupedRewardError("v1 ExpectedTrajectorySet cannot be recovered") from error
        manifests = tuple(
            sorted(
                snapshot.manifests,
                key=lambda item: cast(int, cast(dict[str, JsonValue], item.payload["slot_key"])["rollout_index"]),
            )
        )
        if snapshot.expected_set.content_hash != expected.content_hash:
            raise V1GroupedRewardError("ExpectedTrajectorySet identity changed")
        slots = [cast(dict[str, JsonValue], item.payload["slot_key"]) for item in manifests]
        if any(slot.get("uid") != config.uid for slot in slots) or [
            cast(int, slot.get("rollout_index")) for slot in slots
        ] != list(range(128)):
            raise V1GroupedRewardError("ExpectedTrajectorySet UID/rollout mapping is invalid")
        return manifests

    @staticmethod
    def _commit_return(
        queue: PersistentFixtureTransferQueue, dispatch: Artifact, manifest: Artifact, config: V1GroupedRewardConfig
    ) -> Artifact:
        trajectory = cast(dict[str, JsonValue], manifest.payload["trajectory"])
        response = trajectory.get("response")
        if type(response) is not str or not response:
            raise V1GroupedRewardError("missing trajectory response blocks reward tensor")
        try:
            # ``missing``/``nan``/``version`` are scorer faults owned by the
            # classic grouped resolver.  TransferQueue faults deliberately
            # use its independent identity-fault vocabulary and must never be
            # inferred from a reward fault name.
            return queue.commit_return(dispatch, response)
        except V1RewardIdentityError as error:
            raise V1GroupedRewardError("TransferQueue return identity failed") from error

    @staticmethod
    def _validate_inputs(config: V1GroupedRewardConfig, expected: Artifact, pack: Artifact, spec: Artifact) -> None:
        if (
            spec.content_hash != config.experiment_spec_hash
            or spec.schema_name != "ExperimentSpec"
            or spec.schema_version != "1.0.0"
        ):
            raise V1GroupedRewardError("ExperimentSpec identity is invalid")
        if set(spec.payload) != _EXPERIMENT_FIELDS or spec.payload.get("aggregation") != "calibrated_scalar":
            raise V1GroupedRewardError("reserved or malformed ExperimentSpec is rejected")
        if config.aggregation != "calibrated_scalar":
            raise V1GroupedRewardError("UNSUPPORTED_REWARD_STRATEGY")
        if pack.schema_name != "JudgePack" or pack.payload.get("aggregation") != "calibrated_scalar":
            raise V1GroupedRewardError("JudgePack aggregation is invalid")
        if pack.schema_version != "1.0.0":
            raise V1GroupedRewardError("JudgePack schema version is invalid")
        try:
            from clawrl.router.trace_router import FixtureTraceRouter

            FixtureTraceRouter._pack(pack, config.uid, expected)
        except TraceRouterError as error:
            raise V1GroupedRewardError("JudgePack routing contract is invalid") from error
        for field, requested, observed in (
            ("prompt_hash", config.prompt_hash, pack.payload.get("prompt_hash")),
            ("scalarizer_hash", config.scalarizer_hash, pack.payload.get("scalarizer_hash")),
            ("scalarizer_version", config.scalarizer_version, pack.payload.get("scalarizer_version")),
            ("algorithm_contract", config.algorithm_contract, pack.payload.get("algorithm_contract")),
        ):
            if requested is not None and requested != observed:
                raise V1GroupedRewardError(f"frozen ExperimentSpec mutation: {field}")
        if (
            expected.schema_name != "ExpectedTrajectorySet"
            or expected.payload.get("run_id") != config.run_id
            or expected.payload.get("global_step") != config.global_step
        ):
            raise V1GroupedRewardError("ExpectedTrajectorySet identity is invalid")
        if pack.payload.get("uid") != config.uid:
            raise V1GroupedRewardError("JudgePack UID mapping is invalid")

    @staticmethod
    def _publish_ref(ref: Path, artifact: Artifact) -> None:
        ArtifactStore.durable_mkdir(ref.parent)
        try:
            ArtifactStore._publish(ref, f"{artifact.content_hash}\n".encode("ascii"))
        except ImmutableArtifactConflict:
            try:
                existing = ref.read_text(encoding="ascii").strip()
            except OSError as error:
                raise V1GroupedRewardError("v1 durable ref unavailable") from error
            if existing != artifact.content_hash:
                raise V1GroupedRewardError("v1 durable ref conflict") from None


V1TransferQueueGroupedRewardWorkflow = V1GroupedRewardWorkflow
V1RewardLoopWorkflow = V1GroupedRewardWorkflow
V1GroupedRewardManagerWorkflow = V1GroupedRewardWorkflow
V1RewardLoopConfig = V1GroupedRewardConfig
InjectedV1ControllerCrash = InjectedV1GroupedRewardCrash
