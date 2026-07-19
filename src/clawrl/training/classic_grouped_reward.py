"""Classic importlib RewardManager grouped calibrated-reward fixture.

This module is deliberately a narrow, persistent seam around the Ticket 14
trace router.  It models the classic RewardLoop contract (publish all sample
requests, wait for a per-UID barrier, then return the original rollout order)
without importing a real verl runtime.  Every intermediate object is an
immutable ArtifactStore record so recovery never relies on Python memory.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from clawrl.artifacts import (
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    JsonValue,
    UnknownSchemaMajor,
    canonical_json_bytes,
    sha256_hex,
)
from clawrl.router.trace_router import (
    FixtureTraceRouter,
    RouterCapacityConfig,
    TraceRouterConfig,
    TraceRouterError,
)
from clawrl.training.classic_identity import ClassicSourceRow

_HASH = set("0123456789abcdef")
_DIMS = ("correctness", "reasoning_quality", "task_completion", "tool_discipline")
_EXPERIMENT_FIELDS = {
    "aggregation",
    "dataset_version_hash",
    "dataset_version_id",
    "experiment_id",
    "judge_bundle_hash",
    "trace_set_hash",
}


class ClassicGroupedRewardError(RuntimeError):
    """Classic grouped reward failed closed before an optimizer update."""


class InjectedClassicGroupedRewardCrash(ClassicGroupedRewardError):
    """Synthetic crash after durable request publication."""


def _hash(value: object, field: str) -> str:
    if type(value) is not str or len(value) != 64 or set(cast(str, value)) - _HASH:
        raise ClassicGroupedRewardError(f"{field} is not a content hash")
    return cast(str, value)


def _int(value: object, field: str, minimum: int = 0) -> int:
    if type(value) is not int or cast(int, value) < minimum:
        raise ClassicGroupedRewardError(f"{field} is invalid")
    return cast(int, value)


@dataclass(frozen=True, slots=True)
class ClassicGroupedRewardConfig:
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
    fault_kind: str | None = None
    fault_index: int | None = None

    def __post_init__(self) -> None:
        if not all(type(item) is str and item for item in (self.run_id, self.uid)):
            raise ClassicGroupedRewardError("classic grouped identity is invalid")
        _int(self.global_step, "global_step")
        _hash(self.experiment_spec_hash, "experiment_spec_hash")
        _int(self.resolver_epoch, "resolver_epoch", minimum=1)
        if type(self.chunk_size) is not int or not 1 <= self.chunk_size <= 128:
            raise ClassicGroupedRewardError("classic worker chunk_size is invalid")
        if self.aggregation not in {"calibrated_scalar", "hierarchical_rank", "local_microgroup_rank"}:
            raise ClassicGroupedRewardError("aggregation is invalid")
        if self.prompt_hash is not None:
            _hash(self.prompt_hash, "prompt_hash")
        if self.scalarizer_hash is not None:
            _hash(self.scalarizer_hash, "scalarizer_hash")
        if self.fault_kind not in {None, "missing", "nan", "version"}:
            raise ClassicGroupedRewardError("unknown fixture fault")
        if self.fault_kind is not None and (type(self.fault_index) is not int or self.fault_index < 0):
            raise ClassicGroupedRewardError("fault index is invalid")


# Alias matching the language used by the ticket and likely callers.
ClassicRewardLoopConfig = ClassicGroupedRewardConfig


@dataclass(frozen=True, slots=True)
class ClassicGroupedRewardSnapshot:
    report: Artifact
    barrier: Artifact
    rewards: tuple[Artifact, ...]
    trainer_output: Artifact


class ClassicGroupedRewardWorkflow:
    """Persistent classic RewardManager/trainer contract."""

    @classmethod
    def production_readiness(cls, root: str | Path, *, phase: str = "TRAIN_35B") -> Artifact:
        if phase not in {"TRAIN_35B", "TRAIN_122B"}:
            raise ClassicGroupedRewardError("production phase is invalid")
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "checks": [
                    {"code": "REAL_VERL_IMPORTLIB_REWARD_MANAGER_UNVERIFIED", "status": "blocked"},
                    {"code": "PRODUCTION_CFS_PUBLISH_AND_BARRIER_UNVERIFIED", "status": "blocked"},
                    {"code": "PRODUCTION_TRACE_ACL_AND_ENCRYPTION_UNVERIFIED", "status": "blocked"},
                ],
                "execution_profile": "production",
                "phase": phase,
                "scorer_request_attempted": False,
                "side_effects_permitted": False,
                "status": "blocked",
            },
        )

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: ClassicGroupedRewardConfig,
        expected_set: Artifact,
        judge_pack: Artifact,
        experiment_spec: Artifact,
        sources: tuple[ClassicSourceRow, ...] | None = None,
        crash_after_requests: int | None = None,
    ) -> ClassicGroupedRewardSnapshot:
        """Publish requests asynchronously, resolve the UID barrier, and dump ordered rewards."""

        store = ArtifactStore(root)
        cls._validate_inputs(config, expected_set, judge_pack, experiment_spec)
        namespace = cls._namespace(root, config)
        report_ref = namespace / "report.ref"
        if report_ref.exists():
            recovered = cls.resume(root, config)
            if (
                recovered.report.payload.get("experiment_spec_hash") != experiment_spec.content_hash
                or recovered.report.payload.get("expected_set_hash") != expected_set.content_hash
                or recovered.report.payload.get("judge_pack_hash") != judge_pack.content_hash
            ):
                raise ClassicGroupedRewardError("experiment identity changed on resume")
            # A caller may omit frozen fields while reconstructing a config in
            # a fresh process, but an explicitly supplied value is a mutation
            # attempt and must match the durable JudgePack before returning the
            # already-resolved tensor.
            barrier_pack = store.read(
                cast(str, recovered.barrier.payload["judge_pack_hash"]), expected_schema_name="JudgePack"
            )
            cls._validate_frozen_config(config, barrier_pack)
            return recovered

        # The trace router is the shared application workflow; it is the only
        # component allowed to turn ExpectedTrajectorySet slots into judge
        # results.  It performs all expected-set and pack lineage checks.
        try:
            routed = FixtureTraceRouter.run(
                root,
                config=TraceRouterConfig(
                    config.run_id,
                    config.global_step,
                    config.uid,
                    config.resolver_epoch,
                    config.capacity,
                ),
                expected_set=expected_set,
                judge_pack=judge_pack,
            )
        except TraceRouterError as error:
            raise ClassicGroupedRewardError("classic router did not reach trace-ready") from error
        if len(routed.results) != 128:
            raise ClassicGroupedRewardError("classic grouped barrier requires exactly 128 results")

        request_hashes = asyncio.run(
            cls._publish_requests(
                store,
                config,
                expected_set,
                judge_pack,
                routed.results,
                crash_after_requests=crash_after_requests,
            )
        )
        # Poll the deterministic request/result manifests through an async
        # cooperative loop.  The fixture has already committed the results,
        # so this normally completes on the first pass; importantly no
        # blocking ``run_single`` or actor-loop lock is used.
        asyncio.run(cls._wait_for_trace_ready(store, request_hashes))
        # Worker transport intentionally reverses the published order and
        # pads fixed-size chunks.  Identity, never transport position, drives
        # trainer order.
        records = [store.read(item, expected_schema_name="ClassicRewardRequest") for item in request_hashes]
        worker_rows = cls._worker_reorder_and_validate(store, records, config, config.chunk_size)
        rewards = cls._resolve_rewards(store, worker_rows, config, judge_pack)
        barrier = store.put(
            "ClassicGroupedRewardBarrier",
            "1.0.0",
            {
                "expected_count": 128,
                "expected_set_hash": expected_set.content_hash,
                "judge_pack_hash": judge_pack.content_hash,
                "request_hashes": request_hashes,
                "resolved_reward_hashes": [item.content_hash for item in rewards],
                "run_id": config.run_id,
                "global_step": config.global_step,
                "uid": config.uid,
                "status": "trace_ready",
            },
        )
        trainer = cls._trainer_output(store, config, expected_set, judge_pack, barrier, rewards)
        report = store.put(
            "ClassicGroupedRewardReport",
            "1.0.0",
            {
                "aggregation": config.aggregation,
                "barrier_hash": barrier.content_hash,
                "experiment_spec_hash": experiment_spec.content_hash,
                "expected_set_hash": expected_set.content_hash,
                "global_step": config.global_step,
                "judge_pack_hash": judge_pack.content_hash,
                "production_readiness": "blocked",
                "request_hashes": request_hashes,
                "reward_root_hash": sha256_hex(canonical_json_bytes([item.content_hash for item in rewards])),
                "trainer_output_hash": trainer.content_hash,
                "run_id": config.run_id,
                "uid": config.uid,
                "status": "resolved",
            },
        )
        ArtifactStore.durable_mkdir(namespace)
        ArtifactStore._publish(report_ref, f"{report.content_hash}\n".encode("ascii"))
        return cls.resume(root, config)

    @classmethod
    def resume(cls, root: str | Path, config: ClassicGroupedRewardConfig) -> ClassicGroupedRewardSnapshot:
        store = ArtifactStore(root)
        namespace = cls._namespace(root, config)
        try:
            report_hash = namespace.joinpath("report.ref").read_text().strip()
            report = store.read(report_hash, expected_schema_name="ClassicGroupedRewardReport")
            expected = {
                "aggregation",
                "barrier_hash",
                "experiment_spec_hash",
                "expected_set_hash",
                "global_step",
                "judge_pack_hash",
                "production_readiness",
                "request_hashes",
                "reward_root_hash",
                "trainer_output_hash",
                "run_id",
                "uid",
                "status",
            }
            if report.schema_version != "1.0.0" or set(report.payload) != expected:
                raise ClassicGroupedRewardError("classic grouped report schema is invalid")
            if any(
                report.payload.get(field) != value
                for field, value in {
                    "aggregation": config.aggregation,
                    "experiment_spec_hash": config.experiment_spec_hash,
                    "global_step": config.global_step,
                    "run_id": config.run_id,
                    "uid": config.uid,
                    "status": "resolved",
                    "production_readiness": "blocked",
                }.items()
            ):
                raise ClassicGroupedRewardError("classic grouped report identity changed")
            frozen_pack = store.read(cast(str, report.payload["judge_pack_hash"]), expected_schema_name="JudgePack")
            if frozen_pack.payload.get("uid") != config.uid:
                raise ClassicGroupedRewardError("classic grouped JudgePack UID changed")
            cls._validate_frozen_config(config, frozen_pack)
            expected_set = store.read(
                cast(str, report.payload["expected_set_hash"]), expected_schema_name="ExpectedTrajectorySet"
            )
            if (
                expected_set.payload.get("run_id") != config.run_id
                or expected_set.payload.get("global_step") != config.global_step
            ):
                raise ClassicGroupedRewardError("classic grouped ExpectedTrajectorySet identity changed")
            experiment_spec = store.read(
                cast(str, report.payload["experiment_spec_hash"]), expected_schema_name="ExperimentSpec"
            )
            if (
                set(experiment_spec.payload) != _EXPERIMENT_FIELDS
                or experiment_spec.payload.get("aggregation") != "calibrated_scalar"
            ):
                raise ClassicGroupedRewardError("classic grouped ExperimentSpec is invalid")
            barrier = store.read(
                cast(str, report.payload["barrier_hash"]), expected_schema_name="ClassicGroupedRewardBarrier"
            )
            trainer = store.read(
                cast(str, report.payload["trainer_output_hash"]), expected_schema_name="ClassicTrainerRewardOutput"
            )
            requests = [
                store.read(value, expected_schema_name="ClassicRewardRequest")
                for value in _hash_list(report.payload, "request_hashes")
            ]
            rewards = [
                store.read(value, expected_schema_name="ClassicGroupedReward")
                for value in _hash_list(barrier.payload, "resolved_reward_hashes")
            ]
            results = [
                store.read(cast(str, item.payload["judge_result_hash"]), expected_schema_name="FencedJudgeResult")
                for item in requests
            ]
            cls._validate_durable_graph(report, barrier, trainer, requests, rewards, results, frozen_pack, config)
            return ClassicGroupedRewardSnapshot(report, barrier, tuple(rewards), trainer)
        except (ArtifactCorruption, UnknownSchemaMajor, OSError, KeyError, TypeError, ValueError) as error:
            raise ClassicGroupedRewardError("classic grouped route cannot be recovered") from error

    @staticmethod
    def _namespace(root: str | Path, config: ClassicGroupedRewardConfig) -> Path:
        return Path(root) / "classic-grouped-reward" / config.run_id / str(config.global_step) / config.uid

    @classmethod
    def _validate_inputs(
        cls, config: ClassicGroupedRewardConfig, expected: Artifact, pack: Artifact, spec: Artifact
    ) -> None:
        if (
            spec.content_hash != config.experiment_spec_hash
            or spec.schema_name != "ExperimentSpec"
            or spec.schema_version != "1.0.0"
        ):
            raise ClassicGroupedRewardError("ExperimentSpec identity is invalid")
        if set(spec.payload) != _EXPERIMENT_FIELDS or spec.payload.get("aggregation") != "calibrated_scalar":
            raise ClassicGroupedRewardError("reserved or malformed ExperimentSpec is rejected")
        if config.aggregation != "calibrated_scalar":
            raise ClassicGroupedRewardError("UNSUPPORTED_REWARD_STRATEGY")
        if pack.schema_name != "JudgePack" or pack.payload.get("aggregation") != "calibrated_scalar":
            raise ClassicGroupedRewardError("JudgePack aggregation is invalid")
        cls._validate_frozen_config(config, pack)
        if (
            expected.schema_name != "ExpectedTrajectorySet"
            or expected.payload.get("run_id") != config.run_id
            or expected.payload.get("global_step") != config.global_step
        ):
            raise ClassicGroupedRewardError("ExpectedTrajectorySet identity is invalid")
        if pack.payload.get("uid") != config.uid:
            raise ClassicGroupedRewardError("JudgePack UID mapping is invalid")

    @staticmethod
    def _validate_frozen_config(config: ClassicGroupedRewardConfig, pack: Artifact) -> None:
        """Reject explicit run-time mutations while allowing resume wildcards."""

        for field, requested, observed in (
            ("prompt_hash", config.prompt_hash, pack.payload.get("prompt_hash")),
            ("scalarizer_hash", config.scalarizer_hash, pack.payload.get("scalarizer_hash")),
            ("scalarizer_version", config.scalarizer_version, pack.payload.get("scalarizer_version")),
            ("algorithm_contract", config.algorithm_contract, pack.payload.get("algorithm_contract")),
        ):
            if requested is not None and requested != observed:
                raise ClassicGroupedRewardError(f"frozen ExperimentSpec mutation: {field}")

    @staticmethod
    async def _wait_for_trace_ready(store: ArtifactStore, request_hashes: list[str], *, max_polls: int = 8) -> None:
        """Cooperatively poll all request result manifests without blocking an actor loop."""

        if not request_hashes or len(request_hashes) != 128:
            raise ClassicGroupedRewardError("trace-ready barrier requires exactly 128 requests")
        for _ in range(max_polls):
            ready = True
            for request_hash in request_hashes:
                # Yield between manifest reads so a long 128-item barrier
                # cannot monopolize the RewardLoop actor event loop.
                await asyncio.sleep(0)
                request = await asyncio.to_thread(store.read, request_hash, expected_schema_name="ClassicRewardRequest")
                result_hash = request.payload.get("judge_result_hash")
                if type(result_hash) is not str:
                    ready = False
                    continue
                try:
                    await asyncio.to_thread(store.read, result_hash, expected_schema_name="FencedJudgeResult")
                except (ArtifactCorruption, OSError):
                    ready = False
            if ready:
                return
            await asyncio.sleep(0)
        raise ClassicGroupedRewardError("trace-ready barrier polling exhausted")

    @classmethod
    async def _publish_requests(
        cls,
        store: ArtifactStore,
        config: ClassicGroupedRewardConfig,
        expected: Artifact,
        pack: Artifact,
        results: tuple[Artifact, ...],
        *,
        crash_after_requests: int | None,
    ) -> list[str]:
        async def publish(index: int, result: Artifact) -> str:
            await asyncio.sleep(0)
            trajectory_hash = result.payload.get("trajectory_manifest_hash")
            if type(trajectory_hash) is not str:
                raise ClassicGroupedRewardError("router result trajectory identity is missing")
            manifest = await asyncio.to_thread(store.read, trajectory_hash, expected_schema_name="TrajectoryManifest")
            slot = manifest.payload.get("slot_key")
            if not isinstance(slot, dict) or type(slot.get("rollout_index")) is not int:
                raise ClassicGroupedRewardError("trajectory rollout identity is missing")
            artifact = await asyncio.to_thread(
                store.put,
                "ClassicRewardRequest",
                "1.0.0",
                {
                    "expected_set_hash": expected.content_hash,
                    "global_step": config.global_step,
                    "judge_pack_hash": pack.content_hash,
                    "judge_result_hash": result.content_hash,
                    "rollout_index": slot["rollout_index"],
                    "request_index": index,
                    "run_id": config.run_id,
                    "trajectory_manifest_hash": trajectory_hash,
                    "uid": config.uid,
                },
            )
            return artifact.content_hash

        if crash_after_requests is not None and (crash_after_requests < 0 or crash_after_requests > len(results)):
            raise ClassicGroupedRewardError("crash index is invalid")
        # Publish only the requested prefix for the injected crash.  Each
        # request is content addressed, so the retrying process can safely
        # replay the complete deterministic set and converge idempotently.
        limit = crash_after_requests if crash_after_requests is not None else len(results)
        hashes = await asyncio.gather(*(publish(index, result) for index, result in enumerate(results[:limit])))
        if crash_after_requests is not None and crash_after_requests < len(results):
            raise InjectedClassicGroupedRewardCrash("crash after nonblocking request publication")
        return list(hashes)

    @staticmethod
    def _worker_reorder_and_validate(
        store: ArtifactStore,
        requests: list[Artifact],
        config: ClassicGroupedRewardConfig,
        chunk_size: int,
    ) -> list[Artifact]:
        expected_fields = {
            "expected_set_hash",
            "global_step",
            "judge_pack_hash",
            "judge_result_hash",
            "request_index",
            "rollout_index",
            "run_id",
            "trajectory_manifest_hash",
            "uid",
        }
        if len(requests) != 128:
            raise ClassicGroupedRewardError("classic worker request identity is incomplete")
        for request in requests:
            payload = request.payload
            if request.schema_version != "1.0.0" or set(payload) != expected_fields:
                raise ClassicGroupedRewardError("classic worker request schema is invalid")
            if (
                payload.get("run_id") != config.run_id
                or payload.get("global_step") != config.global_step
                or payload.get("uid") != config.uid
                or type(payload.get("request_index")) is not int
                or not 0 <= cast(int, payload["request_index"]) < 128
                or type(payload.get("rollout_index")) is not int
                or not 0 <= cast(int, payload["rollout_index"]) < 128
            ):
                raise ClassicGroupedRewardError("classic worker request lineage is invalid")
            for field in ("expected_set_hash", "judge_pack_hash", "judge_result_hash", "trajectory_manifest_hash"):
                _hash(payload.get(field), field)
        ordered = sorted(requests, key=lambda item: cast(str, item.payload["trajectory_manifest_hash"]), reverse=True)
        chunks = [ordered[index : index + chunk_size] for index in range(0, len(ordered), chunk_size)]
        for chunk_index, chunk in enumerate(chunks):
            store.put(
                "ClassicWorkerChunk",
                "1.0.0",
                {
                    "chunk_index": chunk_index,
                    "padding_count": (chunk_size - len(chunk)) if chunk_index == len(chunks) - 1 else 0,
                    "request_hashes": [item.content_hash for item in chunk],
                    "reordered": True,
                },
            )
        if len({cast(str, item.payload.get("trajectory_manifest_hash")) for item in requests}) != 128:
            raise ClassicGroupedRewardError("classic worker request identity is incomplete")
        if len({cast(int, item.payload.get("request_index")) for item in requests}) != 128:
            raise ClassicGroupedRewardError("classic worker request index is duplicated")
        indices = sorted(cast(int, item.payload.get("rollout_index")) for item in requests)
        if indices != list(range(128)):
            raise ClassicGroupedRewardError("classic worker rollout order is incomplete")
        return ordered

    @classmethod
    def _resolve_rewards(
        cls, store: ArtifactStore, requests: list[Artifact], config: ClassicGroupedRewardConfig, pack: Artifact
    ) -> tuple[Artifact, ...]:
        rewards: list[Artifact] = []
        for request in requests:
            result = store.read(
                cast(str, request.payload["judge_result_hash"]), expected_schema_name="FencedJudgeResult"
            )
            payload = result.payload
            scalar = payload.get("scalar_micros")
            rollout_index = request.payload.get("rollout_index")
            if type(rollout_index) is not int or not 0 <= rollout_index < 128:
                raise ClassicGroupedRewardError("rollout identity is invalid")
            if config.fault_kind == "missing" and rollout_index == config.fault_index:
                raise ClassicGroupedRewardError("missing rollout blocks reward tensor")
            if config.fault_kind == "nan" and rollout_index == config.fault_index:
                scalar = "nan"
            if config.fault_kind == "version" and rollout_index == config.fault_index:
                payload = dict(payload)
                payload["scalarizer_version"] = "forged-version"
            if type(scalar) is not int or scalar < 0 or not math.isfinite(float(scalar)):
                raise ClassicGroupedRewardError("non-finite reward blocks reward tensor")
            if (
                payload.get("judge_pack_hash") != pack.content_hash
                or payload.get("judge_pack_version") != pack.schema_version
                or payload.get("scalarizer_hash") != pack.payload.get("scalarizer_hash")
                or payload.get("scalarizer_version") != pack.payload.get("scalarizer_version")
            ):
                raise ClassicGroupedRewardError("reward version lineage mismatch blocks reward tensor")
            dimensions = payload.get("dimensions")
            confidence = payload.get("confidence_basis_points")
            evidence = payload.get("evidence")
            failures = payload.get("failure_tags")
            ties = payload.get("turn_local_tie_groups")
            session_hash = payload.get("session_hash")
            thread_ref = payload.get("thread_ref")
            turn_ref = payload.get("turn_ref")
            if (
                not isinstance(dimensions, dict)
                or set(dimensions) != set(_DIMS)
                or any(type(value) is not int or not 0 <= cast(int, value) <= 100 for value in dimensions.values())
                or not isinstance(confidence, int)
                or not 0 <= confidence <= 10_000
                or not isinstance(evidence, list)
                or not evidence
                or any(type(value) is not str or len(value) != 64 for value in evidence)
                or not isinstance(failures, list)
                or any(type(value) is not str for value in failures)
                or not isinstance(ties, list)
                or not ties
                or any(
                    not isinstance(group, list)
                    or not group
                    or any(type(value) is not str or len(value) != 64 for value in group)
                    for group in ties
                )
                or any(value in payload for value in ("global_rank", "global_rank_groups", "rank", "rank_groups"))
                or type(session_hash) is not str
                or len(session_hash) != 64
                or type(thread_ref) is not str
                or not thread_ref
                or type(turn_ref) is not str
                or not turn_ref
            ):
                raise ClassicGroupedRewardError("incomplete Judge result blocks reward tensor")
            expected_scalar = sum(cast(int, value) for value in dimensions.values()) * 250_000
            if scalar != expected_scalar:
                raise ClassicGroupedRewardError("scalar does not match calibrated dimensions")
            # Tie groups are strictly turn-local diagnostics.  A single group
            # covering all 128 rollouts (or any rank-like field) is a global
            # ordering masquerade and is rejected before a tensor is emitted.
            if len(ties) == 1 and len(cast(list[JsonValue], ties[0])) == 128:
                raise ClassicGroupedRewardError("global rank/tie group is not a calibrated scalar")
            manifest = store.read(
                cast(str, request.payload["trajectory_manifest_hash"]), expected_schema_name="TrajectoryManifest"
            )
            slot = manifest.payload.get("slot_key")
            if not isinstance(slot, dict) or slot.get("rollout_index") != rollout_index:
                raise ClassicGroupedRewardError("trajectory rollout identity changed")
            reward = store.put(
                "ClassicGroupedReward",
                "1.0.0",
                {
                    "confidence_basis_points": confidence,
                    "dimensions": cast(JsonValue, dimensions),
                    "evidence": cast(JsonValue, evidence),
                    "failure_tags": cast(JsonValue, failures),
                    "judge_pack_hash": pack.content_hash,
                    "judge_pack_version": pack.schema_version,
                    "judge_result_hash": result.content_hash,
                    "reward_micros": scalar,
                    "rollout_index": rollout_index,
                    "trace_id": slot.get("trace_id"),
                    "run_id": config.run_id,
                    "global_step": config.global_step,
                    "uid": config.uid,
                    "scalarizer_hash": pack.payload.get("scalarizer_hash"),
                    "scalarizer_version": pack.payload.get("scalarizer_version"),
                    "session_hash": session_hash,
                    "thread_ref": thread_ref,
                    "trajectory_manifest_hash": request.payload.get("trajectory_manifest_hash"),
                    "turn_local_tie_groups": cast(JsonValue, ties),
                    "turn_ref": turn_ref,
                },
            )
            rewards.append(reward)
        # Ordered only after worker validation: rollout identity determines order.
        rewards.sort(key=lambda item: cast(int, item.payload["rollout_index"]))
        return tuple(rewards)

    @staticmethod
    def _trainer_output(
        store: ArtifactStore,
        config: ClassicGroupedRewardConfig,
        expected: Artifact,
        pack: Artifact,
        barrier: Artifact,
        rewards: tuple[Artifact, ...],
    ) -> Artifact:
        scalars = [cast(int, item.payload["reward_micros"]) for item in rewards]
        extra = [
            {
                "confidence_basis_points": item.payload["confidence_basis_points"],
                "dimensions": item.payload["dimensions"],
                "evidence": item.payload["evidence"],
                "failure_tags": item.payload["failure_tags"],
                "global_step": config.global_step,
                "judge_pack_hash": pack.content_hash,
                "judge_pack_version": pack.schema_version,
                "judge_result_hash": item.payload["judge_result_hash"],
                "rollout_index": item.payload["rollout_index"],
                "run_id": config.run_id,
                "scalarizer_hash": pack.payload.get("scalarizer_hash"),
                "scalarizer_version": pack.payload.get("scalarizer_version"),
                "session_hash": item.payload["session_hash"],
                "thread_ref": item.payload["thread_ref"],
                "trace_id": item.payload["trace_id"],
                "trajectory_manifest_hash": item.payload["trajectory_manifest_hash"],
                "turn_local_tie_groups": item.payload["turn_local_tie_groups"],
                "turn_ref": item.payload["turn_ref"],
                "uid": config.uid,
            }
            for item in rewards
        ]
        return store.put(
            "ClassicTrainerRewardOutput",
            "1.0.0",
            {
                "barrier_hash": barrier.content_hash,
                "dimensions": [cast(dict[str, JsonValue], item["dimensions"]) for item in extra],
                "extra_info": cast(JsonValue, extra),
                "expected_set_hash": expected.content_hash,
                "global_step": config.global_step,
                "judge_pack_hash": pack.content_hash,
                "reward_count": len(scalars),
                "reward_micros": scalars,
                "run_id": config.run_id,
                "uid": config.uid,
            },
        )

    @staticmethod
    def _validate_durable_graph(
        report: Artifact,
        barrier: Artifact,
        trainer: Artifact,
        requests: list[Artifact],
        rewards: list[Artifact],
        results: list[Artifact],
        pack: Artifact,
        config: ClassicGroupedRewardConfig,
    ) -> None:
        expected_barrier_fields = {
            "expected_count",
            "expected_set_hash",
            "judge_pack_hash",
            "request_hashes",
            "resolved_reward_hashes",
            "run_id",
            "global_step",
            "uid",
            "status",
        }
        expected_request_fields = {
            "expected_set_hash",
            "global_step",
            "judge_pack_hash",
            "judge_result_hash",
            "request_index",
            "rollout_index",
            "run_id",
            "trajectory_manifest_hash",
            "uid",
        }
        expected_reward_fields = {
            "confidence_basis_points",
            "dimensions",
            "evidence",
            "failure_tags",
            "global_step",
            "judge_pack_hash",
            "judge_pack_version",
            "judge_result_hash",
            "reward_micros",
            "rollout_index",
            "run_id",
            "scalarizer_hash",
            "scalarizer_version",
            "session_hash",
            "thread_ref",
            "trace_id",
            "trajectory_manifest_hash",
            "turn_local_tie_groups",
            "turn_ref",
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
        if (
            barrier.schema_version != "1.0.0"
            or set(barrier.payload) != expected_barrier_fields
            or trainer.schema_version != "1.0.0"
            or set(trainer.payload) != expected_trainer_fields
            or any(
                request.schema_version != "1.0.0" or set(request.payload) != expected_request_fields
                for request in requests
            )
            or any(
                reward.schema_version != "1.0.0" or set(reward.payload) != expected_reward_fields for reward in rewards
            )
        ):
            raise ClassicGroupedRewardError("classic grouped durable schema is invalid")
        if (
            barrier.payload.get("status") != "trace_ready"
            or barrier.payload.get("expected_count") != 128
            or barrier.payload.get("expected_set_hash") != report.payload.get("expected_set_hash")
            or barrier.payload.get("judge_pack_hash") != report.payload.get("judge_pack_hash")
            or barrier.payload.get("request_hashes") != report.payload.get("request_hashes")
            or barrier.payload.get("resolved_reward_hashes") != [item.content_hash for item in rewards]
            or barrier.payload.get("run_id") != config.run_id
            or barrier.payload.get("global_step") != config.global_step
            or barrier.payload.get("uid") != config.uid
            or len(requests) != 128
            or len(rewards) != 128
            or len(results) != 128
        ):
            raise ClassicGroupedRewardError("classic grouped barrier is incomplete")
        if sorted(cast(int, item.payload.get("rollout_index")) for item in requests) != list(range(128)):
            raise ClassicGroupedRewardError("classic request rollout identity is incomplete")
        expected_set_hash = report.payload.get("expected_set_hash")
        judge_pack_hash = report.payload.get("judge_pack_hash")
        if any(
            item.payload.get("expected_set_hash") != expected_set_hash
            or item.payload.get("judge_pack_hash") != judge_pack_hash
            or item.payload.get("run_id") != config.run_id
            or item.payload.get("global_step") != config.global_step
            or item.payload.get("uid") != config.uid
            for item in requests
        ):
            raise ClassicGroupedRewardError("classic request lineage changed")
        if sorted(cast(int, item.payload.get("rollout_index")) for item in rewards) != list(range(128)):
            raise ClassicGroupedRewardError("classic reward rollout identity is incomplete")
        if [cast(int, item.payload.get("rollout_index")) for item in rewards] != list(range(128)):
            raise ClassicGroupedRewardError("classic reward order is not canonical")
        if any(
            item.payload.get("judge_pack_hash") != judge_pack_hash
            or item.payload.get("judge_pack_version") != pack.schema_version
            or item.payload.get("scalarizer_hash") != pack.payload.get("scalarizer_hash")
            or item.payload.get("scalarizer_version") != pack.payload.get("scalarizer_version")
            or item.payload.get("run_id") != config.run_id
            or item.payload.get("global_step") != config.global_step
            or item.payload.get("uid") != config.uid
            or type(item.payload.get("reward_micros")) is not int
            or not math.isfinite(float(cast(int, item.payload.get("reward_micros"))))
            for item in rewards
        ):
            raise ClassicGroupedRewardError("classic reward lineage changed")
        request_by_rollout = {cast(int, item.payload["rollout_index"]): item for item in requests}
        if len(request_by_rollout) != 128 or any(
            reward.payload.get("judge_result_hash")
            != request_by_rollout[cast(int, reward.payload["rollout_index"])].payload.get("judge_result_hash")
            or reward.payload.get("trajectory_manifest_hash")
            != request_by_rollout[cast(int, reward.payload["rollout_index"])].payload.get("trajectory_manifest_hash")
            for reward in rewards
        ):
            raise ClassicGroupedRewardError("classic reward request lineage changed")
        if (
            any(
                type(item.payload.get("request_index")) is not int
                or not 0 <= cast(int, item.payload["request_index"]) < 128
                for item in requests
            )
            or len({cast(int, item.payload.get("request_index")) for item in requests}) != 128
        ):
            raise ClassicGroupedRewardError("classic request index lineage changed")
        result_by_hash = {item.content_hash: item for item in results}
        if len(result_by_hash) != 128 or any(
            request.payload.get("judge_result_hash") not in result_by_hash
            or result_by_hash[cast(str, request.payload["judge_result_hash"])].payload.get("trajectory_manifest_hash")
            != request.payload.get("trajectory_manifest_hash")
            or result_by_hash[cast(str, request.payload["judge_result_hash"])].payload.get("judge_pack_hash")
            != report.payload.get("judge_pack_hash")
            or type(result_by_hash[cast(str, request.payload["judge_result_hash"])].payload.get("scalar_micros"))
            is not int
            for request in requests
        ):
            raise ClassicGroupedRewardError("classic result request lineage changed")
        for result in results:
            payload = result.payload
            scalar = payload.get("scalar_micros")
            dimensions = payload.get("dimensions")
            confidence = payload.get("confidence_basis_points")
            evidence = payload.get("evidence")
            failures = payload.get("failure_tags")
            ties = payload.get("turn_local_tie_groups")
            if (
                result.schema_version != "1.0.0"
                or set(payload)
                != {
                    "confidence_basis_points",
                    "dimensions",
                    "evidence",
                    "failure_tags",
                    "judge_pack_hash",
                    "judge_pack_version",
                    "resolver_epoch",
                    "scalar_micros",
                    "scalarizer_hash",
                    "scalarizer_version",
                    "session_hash",
                    "thread_ref",
                    "trajectory_manifest_hash",
                    "turn_local_tie_groups",
                    "turn_ref",
                }
                or payload.get("resolver_epoch") != config.resolver_epoch
                or type(scalar) is not int
                or cast(int, scalar) < 0
                or not math.isfinite(float(cast(int, scalar)))
                or not isinstance(dimensions, dict)
                or set(dimensions) != set(_DIMS)
                or any(type(value) is not int or not 0 <= cast(int, value) <= 100 for value in dimensions.values())
                or cast(int, scalar) != sum(cast(int, value) for value in dimensions.values()) * 250_000
                or type(confidence) is not int
                or not 0 <= cast(int, confidence) <= 10_000
                or not isinstance(evidence, list)
                or not evidence
                or any(type(value) is not str or len(value) != 64 or set(value) - _HASH for value in evidence)
                or not isinstance(failures, list)
                or any(type(value) is not str for value in failures)
                or not isinstance(ties, list)
                or not ties
                or any(
                    not isinstance(group, list)
                    or not group
                    or any(type(value) is not str or len(value) != 64 or set(value) - _HASH for value in group)
                    for group in ties
                )
                or any(value in payload for value in ("global_rank", "global_rank_groups", "rank", "rank_groups"))
                or payload.get("judge_pack_hash") != pack.content_hash
                or payload.get("judge_pack_version") != pack.schema_version
                or payload.get("scalarizer_hash") != pack.payload.get("scalarizer_hash")
                or payload.get("scalarizer_version") != pack.payload.get("scalarizer_version")
            ):
                raise ClassicGroupedRewardError("classic result payload integrity changed")
            _hash(payload.get("session_hash"), "session_hash")
            if type(payload.get("thread_ref")) is not str or not payload.get("thread_ref"):
                raise ClassicGroupedRewardError("classic result thread lineage changed")
            if type(payload.get("turn_ref")) is not str or not payload.get("turn_ref"):
                raise ClassicGroupedRewardError("classic result turn lineage changed")
        reward_by_rollout = {cast(int, item.payload["rollout_index"]): item for item in rewards}
        if any(
            reward.payload.get("reward_micros")
            != result_by_hash[cast(str, request.payload["judge_result_hash"])].payload.get("scalar_micros")
            or reward.payload.get("session_hash")
            != result_by_hash[cast(str, request.payload["judge_result_hash"])].payload.get("session_hash")
            or reward.payload.get("thread_ref")
            != result_by_hash[cast(str, request.payload["judge_result_hash"])].payload.get("thread_ref")
            or reward.payload.get("turn_ref")
            != result_by_hash[cast(str, request.payload["judge_result_hash"])].payload.get("turn_ref")
            or any(
                reward.payload.get(field)
                != result_by_hash[cast(str, request.payload["judge_result_hash"])].payload.get(field)
                for field in (
                    "confidence_basis_points",
                    "dimensions",
                    "evidence",
                    "failure_tags",
                    "judge_pack_version",
                    "scalarizer_hash",
                    "scalarizer_version",
                    "turn_local_tie_groups",
                )
            )
            for request in requests
            for reward in [reward_by_rollout[cast(int, request.payload["rollout_index"])]]
        ):
            raise ClassicGroupedRewardError("classic reward result scalar lineage changed")
        if (
            trainer.payload.get("barrier_hash") != barrier.content_hash
            or trainer.payload.get("expected_set_hash") != report.payload.get("expected_set_hash")
            or trainer.payload.get("judge_pack_hash") != report.payload.get("judge_pack_hash")
            or trainer.payload.get("global_step") != config.global_step
            or trainer.payload.get("run_id") != config.run_id
            or trainer.payload.get("uid") != config.uid
            or trainer.payload.get("reward_count") != 128
            or len(cast(list[object], trainer.payload.get("reward_micros"))) != 128
            or any(type(item) is not int for item in cast(list[object], trainer.payload.get("reward_micros")))
            or len(cast(list[object], trainer.payload.get("extra_info"))) != 128
            or any(
                not isinstance(item, dict)
                or item.get("run_id") != config.run_id
                or item.get("global_step") != config.global_step
                or item.get("uid") != config.uid
                or type(item.get("rollout_index")) is not int
                for item in cast(list[object], trainer.payload.get("extra_info"))
            )
        ):
            raise ClassicGroupedRewardError("trainer reward tensor is incomplete")
        trainer_scalars = cast(list[object], trainer.payload.get("reward_micros"))
        reward_scalars = [item.payload.get("reward_micros") for item in rewards]
        if trainer_scalars != reward_scalars:
            raise ClassicGroupedRewardError("trainer reward tensor does not match rewards")
        extra_info = cast(list[object], trainer.payload.get("extra_info"))
        expected_extra_fields = {
            "confidence_basis_points",
            "dimensions",
            "evidence",
            "failure_tags",
            "global_step",
            "judge_pack_hash",
            "judge_pack_version",
            "judge_result_hash",
            "rollout_index",
            "run_id",
            "scalarizer_hash",
            "scalarizer_version",
            "session_hash",
            "thread_ref",
            "trace_id",
            "trajectory_manifest_hash",
            "turn_local_tie_groups",
            "turn_ref",
            "uid",
        }
        for extra, reward in zip(extra_info, rewards, strict=True):
            if not isinstance(extra, dict) or set(extra) != expected_extra_fields:
                raise ClassicGroupedRewardError("trainer extra_info schema is incomplete")
            if any(extra.get(field) != reward.payload.get(field) for field in expected_extra_fields):
                raise ClassicGroupedRewardError("trainer extra_info lineage changed")
        if report.payload.get("reward_root_hash") != sha256_hex(
            canonical_json_bytes([item.content_hash for item in rewards])
        ):
            raise ClassicGroupedRewardError("classic reward root hash mismatch")


ClassicRewardLoopWorkflow = ClassicGroupedRewardWorkflow
ClassicRewardManagerWorkflow = ClassicGroupedRewardWorkflow


def _hash_list(payload: dict[str, JsonValue], field: str) -> list[str]:
    value = payload.get(field)
    if (
        not isinstance(value, list)
        or any(type(item) is not str or len(item) != 64 for item in value)
        or len(set(value)) != len(value)
    ):
        raise ClassicGroupedRewardError(f"{field} hash list is invalid")
    return cast(list[str], value)
