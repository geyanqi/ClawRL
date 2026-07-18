"""Wide-prefactor trajectory identity for the declared classic verl fixture path."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from clawrl.artifacts import (
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    ImmutableArtifactConflict,
    JsonValue,
    canonical_json_bytes,
    sha256_hex,
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_VARIANT = "classic/v1-wide-prefactor"
_DECLARED_PATHS = (
    "wide_prefactor",
    "generation",
    "repeat",
    "union",
    "balance",
    "chunk_padding",
    "reward_loop_worker",
    "dump",
    "classic_resume",
)
_STAGE_NAMES = _DECLARED_PATHS[:7]
_FAULT_STAGES = frozenset(_STAGE_NAMES)
_FAULT_KINDS = {"missing_identity", "duplicate_index", "global_step_mismatch"}


class ClassicIdentityError(RuntimeError):
    """Classic identity was missing, ambiguous, or changed in transit."""


def _safe_id(value: object, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(cast(str, value)) is None:
        raise ClassicIdentityError(f"{field} is invalid")
    return cast(str, value)


def _integer(value: object, field: str, *, minimum: int = 0, maximum: int = 1_000_000) -> int:
    if type(value) is not int or not minimum <= cast(int, value) <= maximum:
        raise ClassicIdentityError(f"{field} is invalid")
    return cast(int, value)


@dataclass(frozen=True, slots=True)
class ClassicTrajectoryIdentity:
    run_id: str
    global_step: int
    trace_id: str
    uid: str
    rollout_index: int
    expected_rollout_count: int
    judge_pack_id: str

    def __post_init__(self) -> None:
        for field, value in (
            ("run_id", self.run_id),
            ("trace_id", self.trace_id),
            ("uid", self.uid),
            ("judge_pack_id", self.judge_pack_id),
        ):
            _safe_id(value, field)
        _integer(self.global_step, "global_step", maximum=9_007_199_254_740_991)
        _integer(self.rollout_index, "rollout_index")
        _integer(self.expected_rollout_count, "expected_rollout_count", minimum=1)
        if self.rollout_index >= self.expected_rollout_count:
            raise ClassicIdentityError("rollout_index is outside expected_rollout_count")

    @classmethod
    def from_mapping(cls, value: object) -> ClassicTrajectoryIdentity:
        expected = {
            "expected_rollout_count",
            "global_step",
            "judge_pack_id",
            "rollout_index",
            "run_id",
            "trace_id",
            "uid",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ClassicIdentityError("classic trajectory identity fields are invalid")
        return cls(
            run_id=_safe_id(value.get("run_id"), "run_id"),
            global_step=_integer(value.get("global_step"), "global_step", maximum=9_007_199_254_740_991),
            trace_id=_safe_id(value.get("trace_id"), "trace_id"),
            uid=_safe_id(value.get("uid"), "uid"),
            rollout_index=_integer(value.get("rollout_index"), "rollout_index"),
            expected_rollout_count=_integer(value.get("expected_rollout_count"), "expected_rollout_count", minimum=1),
            judge_pack_id=_safe_id(value.get("judge_pack_id"), "judge_pack_id"),
        )

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "expected_rollout_count": self.expected_rollout_count,
            "global_step": self.global_step,
            "judge_pack_id": self.judge_pack_id,
            "rollout_index": self.rollout_index,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "uid": self.uid,
        }

    @property
    def content_hash(self) -> str:
        return sha256_hex(
            canonical_json_bytes({"domain": "classic-trajectory-identity/1.0.0", "identity": self.artifact_payload()})
        )


@dataclass(frozen=True, slots=True)
class ClassicSourceRow:
    trace_id: str
    uid: str
    judge_pack_id: str
    prompt: str

    def __post_init__(self) -> None:
        _safe_id(self.trace_id, "trace_id")
        _safe_id(self.uid, "uid")
        _safe_id(self.judge_pack_id, "judge_pack_id")
        if type(self.prompt) is not str or not 1 <= len(self.prompt) <= 65_536:
            raise ClassicIdentityError("classic source prompt is invalid")

    @classmethod
    def from_mapping(cls, value: object) -> ClassicSourceRow:
        if not isinstance(value, Mapping) or set(value) != {"judge_pack_id", "prompt", "trace_id", "uid"}:
            raise ClassicIdentityError("classic source row fields are invalid")
        prompt = value.get("prompt")
        if type(prompt) is not str:
            raise ClassicIdentityError("classic source prompt is invalid")
        return cls(
            trace_id=_safe_id(value.get("trace_id"), "trace_id"),
            uid=_safe_id(value.get("uid"), "uid"),
            judge_pack_id=_safe_id(value.get("judge_pack_id"), "judge_pack_id"),
            prompt=prompt,
        )

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "judge_pack_id": self.judge_pack_id,
            "prompt": self.prompt,
            "trace_id": self.trace_id,
            "uid": self.uid,
        }


@dataclass(frozen=True, slots=True)
class ClassicTrajectoryRow:
    identity: ClassicTrajectoryIdentity
    prompt: str
    response: str | None = None

    def __post_init__(self) -> None:
        if type(self.prompt) is not str or not self.prompt:
            raise ClassicIdentityError("classic trajectory prompt is invalid")
        if self.response is not None and (type(self.response) is not str or not self.response):
            raise ClassicIdentityError("classic trajectory response is invalid")

    @classmethod
    def from_mapping(cls, value: object) -> ClassicTrajectoryRow:
        if not isinstance(value, Mapping) or set(value) != {"identity", "prompt", "response"}:
            raise ClassicIdentityError("classic trajectory row fields are invalid")
        prompt = value.get("prompt")
        response = value.get("response")
        if type(prompt) is not str or response is not None and type(response) is not str:
            raise ClassicIdentityError("classic trajectory text fields are invalid")
        return cls(ClassicTrajectoryIdentity.from_mapping(value.get("identity")), prompt, cast(str | None, response))

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {"identity": self.identity.artifact_payload(), "prompt": self.prompt, "response": self.response}

    @property
    def content_hash(self) -> str:
        return sha256_hex(
            canonical_json_bytes({"domain": "classic-trajectory-row/1.0.0", "row": self.artifact_payload()})
        )


@dataclass(frozen=True, slots=True)
class ClassicBatch:
    rows: tuple[ClassicTrajectoryRow, ...]
    global_steps: tuple[int, ...]
    transport_chunks: tuple[tuple[str | None, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.rows or len(self.global_steps) != len(self.rows):
            raise ClassicIdentityError("classic batch row/global_steps cardinality is invalid")
        identities = [row.identity for row in self.rows]
        identity_hashes = [item.content_hash for item in identities]
        if len(identity_hashes) != len(set(identity_hashes)):
            raise ClassicIdentityError("duplicate classic trajectory identity")
        for step, identity in zip(self.global_steps, identities, strict=True):
            if type(step) is not int or step != identity.global_step:
                raise ClassicIdentityError("batch global_steps does not match row global_step")
        by_uid: dict[str, list[ClassicTrajectoryIdentity]] = {}
        for identity in identities:
            by_uid.setdefault(identity.uid, []).append(identity)
        run_steps = {(item.run_id, item.global_step) for item in identities}
        if len(run_steps) != 1:
            raise ClassicIdentityError("classic batch mixes run or global_step")
        for uid, items in by_uid.items():
            counts = {item.expected_rollout_count for item in items}
            trace_packs = {(item.trace_id, item.judge_pack_id) for item in items}
            if len(counts) != 1 or len(trace_packs) != 1:
                raise ClassicIdentityError(f"identity mapping changed for UID {uid}")
            expected_count = next(iter(counts))
            if sorted(item.rollout_index for item in items) != list(range(expected_count)):
                raise ClassicIdentityError(f"rollout index set is incomplete for UID {uid}")
        if self.transport_chunks:
            transported = [item for chunk in self.transport_chunks for item in chunk if item is not None]
            row_hashes = [row.content_hash for row in self.rows]
            if len(transported) != len(row_hashes) or sorted(transported) != sorted(row_hashes):
                raise ClassicIdentityError("chunk/padding transport changed the trajectory set")
            chunk_lengths = {len(chunk) for chunk in self.transport_chunks}
            if len(chunk_lengths) != 1 or any(item is None for chunk in self.transport_chunks[:-1] for item in chunk):
                raise ClassicIdentityError("padding is only valid in the final fixed-size chunk")
            for chunk in self.transport_chunks:
                seen_padding = False
                invalid_padding = False
                for item in chunk:
                    if item is None:
                        seen_padding = True
                    elif seen_padding:
                        invalid_padding = True
                if not chunk or invalid_padding:
                    raise ClassicIdentityError("padding sentinel appeared before a chunk tail")

    @classmethod
    def from_mapping(cls, value: object) -> ClassicBatch:
        if not isinstance(value, Mapping) or set(value) != {"global_steps", "rows", "transport_chunks"}:
            raise ClassicIdentityError("classic batch fields are invalid")
        rows_value = value.get("rows")
        steps_value = value.get("global_steps")
        chunks_value = value.get("transport_chunks")
        if not isinstance(rows_value, list) or not isinstance(steps_value, list) or not isinstance(chunks_value, list):
            raise ClassicIdentityError("classic batch arrays are invalid")
        rows = tuple(ClassicTrajectoryRow.from_mapping(item) for item in rows_value)
        steps = tuple(_integer(item, "batch global_steps", maximum=9_007_199_254_740_991) for item in steps_value)
        chunks: list[tuple[str | None, ...]] = []
        for chunk in chunks_value:
            if not isinstance(chunk, list):
                raise ClassicIdentityError("transport chunk is invalid")
            normalized: list[str | None] = []
            for item in chunk:
                if item is not None and (type(item) is not str or _HASH.fullmatch(cast(str, item)) is None):
                    raise ClassicIdentityError("transport chunk row hash is invalid")
                normalized.append(cast(str | None, item))
            chunks.append(tuple(normalized))
        return cls(rows, steps, tuple(chunks))

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "global_steps": list(self.global_steps),
            "rows": [row.artifact_payload() for row in self.rows],
            "transport_chunks": [list(chunk) for chunk in self.transport_chunks],
        }

    @property
    def identity_set_hash(self) -> str:
        return sha256_hex(canonical_json_bytes(sorted(row.identity.content_hash for row in self.rows)))


@dataclass(frozen=True, slots=True)
class ClassicIdentityConfig:
    run_id: str
    global_step: int
    rollout_count: int = 2
    chunk_size: int = 4
    variant: str = _SUPPORTED_VARIANT
    fault_stage: str | None = None
    fault_kind: str | None = None

    def __post_init__(self) -> None:
        _safe_id(self.run_id, "run_id")
        _integer(self.global_step, "global_step", maximum=9_007_199_254_740_991)
        _integer(self.rollout_count, "rollout_count", minimum=1, maximum=128)
        _integer(self.chunk_size, "chunk_size", minimum=1, maximum=2048)
        if self.variant != _SUPPORTED_VARIANT:
            raise ClassicIdentityError("fixture only supports classic/v1-wide-prefactor")
        if (self.fault_stage is None) != (self.fault_kind is None):
            raise ClassicIdentityError("fault stage and kind must be configured together")
        if self.fault_stage is not None and self.fault_stage not in _FAULT_STAGES:
            raise ClassicIdentityError("fixture fault stage is invalid")
        if self.fault_kind is not None and self.fault_kind not in _FAULT_KINDS:
            raise ClassicIdentityError("fixture fault kind is invalid")

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "chunk_size": self.chunk_size,
            "fault_kind": self.fault_kind,
            "fault_stage": self.fault_stage,
            "global_step": self.global_step,
            "rollout_count": self.rollout_count,
            "run_id": self.run_id,
            "variant": self.variant,
        }


@dataclass(frozen=True, slots=True)
class ProductionClassicIdentityConfig:
    variant: str

    def __post_init__(self) -> None:
        if type(self.variant) is not str or not 1 <= len(self.variant) <= 256:
            raise ClassicIdentityError("production classic variant is invalid")


@dataclass(frozen=True, slots=True)
class ClassicIdentitySnapshot:
    report: Artifact
    dump: Artifact
    stages: tuple[Artifact, ...]


class ClassicRewardManager:
    """CPU-only contract seam; it transports identity and does not resolve rewards."""

    @staticmethod
    def prefactor(config: ClassicIdentityConfig, sources: tuple[ClassicSourceRow, ...]) -> ClassicBatch:
        if not sources:
            raise ClassicIdentityError("classic source set is empty")
        uids = [item.uid for item in sources]
        if len(uids) != len(set(uids)):
            raise ClassicIdentityError("classic source UID is duplicated")
        rows = tuple(
            ClassicTrajectoryRow(
                ClassicTrajectoryIdentity(
                    config.run_id,
                    config.global_step,
                    source.trace_id,
                    source.uid,
                    rollout_index,
                    config.rollout_count,
                    source.judge_pack_id,
                ),
                source.prompt,
            )
            for source in sources
            for rollout_index in range(config.rollout_count)
        )
        return ClassicBatch(rows, tuple(config.global_step for _ in rows))

    @staticmethod
    def worker_accept(batch_payload: object) -> ClassicBatch:
        batch = ClassicBatch.from_mapping(batch_payload)
        if any(row.response is None for row in batch.rows):
            raise ClassicIdentityError("RewardLoop worker received an ungenerated trajectory")
        return batch


class ClassicIdentityWorkflow:
    @staticmethod
    def production_readiness(root: str | Path, config: ProductionClassicIdentityConfig) -> Artifact:
        checks: list[dict[str, str]] = []
        if config.variant != _SUPPORTED_VARIANT:
            checks.append({"code": "UNVERIFIED_CLASSIC_VARIANT", "status": "blocked"})
        else:
            checks.append({"code": "REAL_VERL_RUNTIME_UNAVAILABLE", "status": "blocked"})
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "cfs_integration_claimed": False,
                "checks": checks,
                "execution_profile": "production",
                "phase": "TRAIN_35B",
                "side_effects_permitted": False,
                "status": "blocked",
                "variant": config.variant,
            },
        )

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: ClassicIdentityConfig,
        sources: tuple[ClassicSourceRow, ...],
    ) -> ClassicIdentitySnapshot:
        store = ArtifactStore(root)
        run_root = Path(root) / "classic-identity-runs" / config.run_id
        ArtifactStore.durable_mkdir(run_root)
        input_artifact = store.put(
            "ClassicIdentityRunInput",
            "1.0.0",
            {"config": config.artifact_payload(), "sources": [item.artifact_payload() for item in sources]},
        )
        cls._publish_ref(store, run_root / "input.ref", input_artifact, "RUN_INPUT_CONFLICT")
        if (run_root / "report.ref").exists():
            return cls.resume(root, config.run_id)

        decision = store.put(
            "DecisionRecord",
            "1.0.0",
            {
                "decision": "ADOPT_CLASSIC_V1_WIDE_PREFACTOR_FIXTURE",
                "declared_paths": list(_DECLARED_PATHS),
                "rationale": "Ticket11 verifies only the explicit classic v1 fixture path; real verl remains blocked.",
                "scope": "fixture_only",
            },
        )
        contract = store.put(
            "ClassicWidePrefactorContract",
            "1.0.0",
            {
                "batch_step_field": "global_steps",
                "decision_record_hash": decision.content_hash,
                "identity_fields": [
                    "run_id",
                    "global_step",
                    "trace_id",
                    "uid",
                    "rollout_index",
                    "expected_rollout_count",
                    "judge_pack_id",
                ],
                "prefactor_position": "before_generation_and_async_dispatch",
                "variant": config.variant,
            },
        )
        manager = ClassicRewardManager()
        batch = manager.prefactor(config, sources)
        stages: list[Artifact] = []
        parent_hash: str = input_artifact.content_hash
        baseline_identity_set_hash = batch.identity_set_hash
        try:
            for stage_index, stage_name in enumerate(_STAGE_NAMES, start=1):
                if stage_name != "wide_prefactor":
                    batch = cls._transform(stage_name, batch, config)
                if config.fault_stage == stage_name:
                    batch = cls._inject_and_validate(batch, cast(str, config.fault_kind))
                cls._validate_stage_batch(stage_name, batch, baseline_identity_set_hash)
                stage = store.put(
                    "ClassicTrajectoryStage",
                    "1.0.0",
                    {
                        "batch": batch.artifact_payload(),
                        "contract_hash": contract.content_hash,
                        "identity_set_hash": batch.identity_set_hash,
                        "input_hash": input_artifact.content_hash,
                        "parent_hash": parent_hash,
                        "stage_index": stage_index,
                        "stage_name": stage_name,
                    },
                )
                stages.append(stage)
                parent_hash = stage.content_hash
        except ClassicIdentityError as error:
            failure = store.put(
                "ClassicIdentityFailure",
                "1.0.0",
                {
                    "fault_kind": config.fault_kind,
                    "input_hash": input_artifact.content_hash,
                    "reason_code": "CLASSIC_IDENTITY_CORRUPTION",
                    "stage": config.fault_stage,
                    "status": "failed",
                },
            )
            raise ClassicIdentityError(f"CLASSIC_IDENTITY_CORRUPTION:{failure.content_hash}") from error

        dump = store.put(
            "ClassicTrajectoryDump",
            "1.0.0",
            {
                **batch.artifact_payload(),
                "cfs_integration_claimed": False,
                "identity_set_hash": batch.identity_set_hash,
                "input_hash": input_artifact.content_hash,
                "source_stage_hash": stages[-1].content_hash,
                "variant": config.variant,
            },
        )
        report = store.put(
            "ClassicTrajectoryContractReport",
            "1.0.0",
            {
                "cfs_integration_claimed": False,
                "contract_hash": contract.content_hash,
                "decision_record_hash": decision.content_hash,
                "declared_paths": list(_DECLARED_PATHS),
                "dump_hash": dump.content_hash,
                "identity_set_hash": batch.identity_set_hash,
                "input_hash": input_artifact.content_hash,
                "row_count": len(batch.rows),
                "run_id": config.run_id,
                "stage_hashes": [item.content_hash for item in stages],
                "status": "passed",
                "train_readiness_claimed": False,
                "variant": config.variant,
            },
        )
        cls._publish_ref(store, run_root / "report.ref", report, "REPORT_CONFLICT")
        return cls.resume(root, config.run_id)

    @classmethod
    def resume(cls, root: str | Path, run_id: str) -> ClassicIdentitySnapshot:
        _safe_id(run_id, "run_id")
        store = ArtifactStore(root)
        run_root = Path(root) / "classic-identity-runs" / run_id
        try:
            input_artifact = store.read(
                (run_root / "input.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="ClassicIdentityRunInput",
            )
            report = store.read(
                (run_root / "report.ref").read_text(encoding="ascii").strip(),
                expected_schema_name="ClassicTrajectoryContractReport",
            )
        except (ArtifactCorruption, OSError) as error:
            raise ClassicIdentityError("classic identity run cannot be recovered") from error
        config, sources = cls._input_contract(input_artifact)
        if config.run_id != run_id:
            raise ClassicIdentityError("classic resume run identity changed")
        expected_report_fields = {
            "cfs_integration_claimed",
            "contract_hash",
            "decision_record_hash",
            "declared_paths",
            "dump_hash",
            "identity_set_hash",
            "input_hash",
            "row_count",
            "run_id",
            "stage_hashes",
            "status",
            "train_readiness_claimed",
            "variant",
        }
        stage_hashes = report.payload.get("stage_hashes")
        if (
            set(report.payload) != expected_report_fields
            or report.payload.get("input_hash") != input_artifact.content_hash
            or report.payload.get("run_id") != run_id
            or report.payload.get("variant") != _SUPPORTED_VARIANT
            or report.payload.get("declared_paths") != list(_DECLARED_PATHS)
            or report.payload.get("status") != "passed"
            or report.payload.get("cfs_integration_claimed") is not False
            or report.payload.get("train_readiness_claimed") is not False
            or not isinstance(stage_hashes, list)
            or len(stage_hashes) != len(_STAGE_NAMES)
        ):
            raise ClassicIdentityError("classic contract report is invalid")
        contract = store.read(
            cast(str, report.payload["contract_hash"]), expected_schema_name="ClassicWidePrefactorContract"
        )
        decision = store.read(cast(str, report.payload["decision_record_hash"]), expected_schema_name="DecisionRecord")
        cls._validate_contract(contract, decision)
        baseline = ClassicRewardManager.prefactor(config, sources).identity_set_hash
        stages: list[Artifact] = []
        parent_hash = input_artifact.content_hash
        for index, (stage_name, stage_hash) in enumerate(zip(_STAGE_NAMES, stage_hashes, strict=True), start=1):
            if type(stage_hash) is not str:
                raise ClassicIdentityError("classic stage hash is invalid")
            stage = store.read(stage_hash, expected_schema_name="ClassicTrajectoryStage")
            batch = cls._stage_contract(
                stage,
                expected_name=stage_name,
                expected_index=index,
                expected_parent=parent_hash,
                input_hash=input_artifact.content_hash,
                contract_hash=contract.content_hash,
            )
            cls._validate_stage_batch(stage_name, batch, baseline)
            stages.append(stage)
            parent_hash = stage.content_hash
        dump = store.read(cast(str, report.payload["dump_hash"]), expected_schema_name="ClassicTrajectoryDump")
        dump_batch = cls._dump_contract(dump, input_artifact.content_hash, stages[-1].content_hash, baseline)
        if (
            report.payload.get("identity_set_hash") != baseline
            or report.payload.get("row_count") != len(dump_batch.rows)
            or dump_batch.artifact_payload()
            != ClassicBatch.from_mapping(stages[-1].payload["batch"]).artifact_payload()
        ):
            raise ClassicIdentityError("classic report/dump lineage is invalid")
        return ClassicIdentitySnapshot(report, dump, tuple(stages))

    @staticmethod
    def _transform(stage: str, batch: ClassicBatch, config: ClassicIdentityConfig) -> ClassicBatch:
        if stage == "generation":
            rows = tuple(
                ClassicTrajectoryRow(
                    row.identity,
                    row.prompt,
                    "fixture-generation/" + sha256_hex(canonical_json_bytes(row.artifact_payload()))[:24],
                )
                for row in batch.rows
            )
            return ClassicBatch(rows, tuple(row.identity.global_step for row in rows))
        if stage == "repeat":
            return ClassicBatch.from_mapping(json.loads(canonical_json_bytes(batch.artifact_payload())))
        if stage == "union":
            rows = batch.rows[::2] + batch.rows[1::2]
            return ClassicBatch(rows, tuple(row.identity.global_step for row in rows))
        if stage == "balance":
            by_uid: dict[str, list[ClassicTrajectoryRow]] = {}
            for row in batch.rows:
                by_uid.setdefault(row.identity.uid, []).append(row)
            rows = tuple(
                next(row for row in by_uid[uid] if row.identity.rollout_index == rollout_index)
                for rollout_index in range(config.rollout_count)
                for uid in sorted(by_uid)
            )
            return ClassicBatch(rows, tuple(row.identity.global_step for row in rows))
        if stage == "chunk_padding":
            chunks: list[tuple[str | None, ...]] = []
            hashes = [row.content_hash for row in batch.rows]
            for offset in range(0, len(hashes), config.chunk_size):
                chunk: list[str | None] = list(hashes[offset : offset + config.chunk_size])
                chunk.extend([None] * (config.chunk_size - len(chunk)))
                chunks.append(tuple(chunk))
            return ClassicBatch(batch.rows, batch.global_steps, tuple(chunks))
        if stage == "reward_loop_worker":
            accepted = ClassicRewardManager.worker_accept(batch.artifact_payload())
            rows_by_hash = {row.content_hash: row for row in accepted.rows}
            transported = [item for chunk in accepted.transport_chunks for item in chunk if item is not None]
            rows = tuple(rows_by_hash[cast(str, item)] for item in transported)
            return ClassicBatch(rows, tuple(row.identity.global_step for row in rows), accepted.transport_chunks)
        raise ClassicIdentityError(f"unsupported classic stage {stage}")

    @staticmethod
    def _inject_and_validate(batch: ClassicBatch, fault_kind: str) -> ClassicBatch:
        payload = batch.artifact_payload()
        rows = cast(list[JsonValue], payload["rows"])
        if fault_kind == "missing_identity":
            first = cast(dict[str, JsonValue], rows[0])
            del first["identity"]
        elif fault_kind == "duplicate_index":
            first = cast(dict[str, JsonValue], rows[0])
            second = cast(dict[str, JsonValue], rows[1])
            second["identity"] = first["identity"]
        elif fault_kind == "global_step_mismatch":
            steps = cast(list[JsonValue], payload["global_steps"])
            steps[0] = cast(int, steps[0]) + 1
        else:
            raise ClassicIdentityError("unknown classic identity fault")
        return ClassicBatch.from_mapping(payload)

    @staticmethod
    def _validate_stage_batch(stage: str, batch: ClassicBatch, baseline: str) -> None:
        if batch.identity_set_hash != baseline:
            raise ClassicIdentityError(f"identity set changed at {stage}")
        if stage in {"wide_prefactor"} and any(row.response is not None for row in batch.rows):
            raise ClassicIdentityError("generation occurred before the wide prefactor boundary")
        if stage != "wide_prefactor" and any(row.response is None for row in batch.rows):
            raise ClassicIdentityError(f"generated response was lost at {stage}")
        if stage in {"chunk_padding", "reward_loop_worker"} and not batch.transport_chunks:
            raise ClassicIdentityError(f"chunk/padding evidence is missing at {stage}")

    @staticmethod
    def _input_contract(input_artifact: Artifact) -> tuple[ClassicIdentityConfig, tuple[ClassicSourceRow, ...]]:
        if set(input_artifact.payload) != {"config", "sources"}:
            raise ClassicIdentityError("classic run input fields are invalid")
        config_value = input_artifact.payload.get("config")
        sources_value = input_artifact.payload.get("sources")
        if not isinstance(config_value, dict) or set(config_value) != {
            "chunk_size",
            "fault_kind",
            "fault_stage",
            "global_step",
            "rollout_count",
            "run_id",
            "variant",
        }:
            raise ClassicIdentityError("classic fixture config artifact is invalid")
        if not isinstance(sources_value, list):
            raise ClassicIdentityError("classic source artifact is invalid")
        config = ClassicIdentityConfig(
            run_id=_safe_id(config_value.get("run_id"), "run_id"),
            global_step=_integer(config_value.get("global_step"), "global_step", maximum=9_007_199_254_740_991),
            rollout_count=_integer(config_value.get("rollout_count"), "rollout_count", minimum=1, maximum=128),
            chunk_size=_integer(config_value.get("chunk_size"), "chunk_size", minimum=1, maximum=2048),
            variant=cast(str, config_value.get("variant")),
            fault_stage=cast(str | None, config_value.get("fault_stage")),
            fault_kind=cast(str | None, config_value.get("fault_kind")),
        )
        sources = tuple(ClassicSourceRow.from_mapping(item) for item in sources_value)
        return config, sources

    @staticmethod
    def _validate_contract(contract: Artifact, decision: Artifact) -> None:
        expected_identity_fields = [
            "run_id",
            "global_step",
            "trace_id",
            "uid",
            "rollout_index",
            "expected_rollout_count",
            "judge_pack_id",
        ]
        if (
            set(contract.payload)
            != {"batch_step_field", "decision_record_hash", "identity_fields", "prefactor_position", "variant"}
            or contract.payload.get("batch_step_field") != "global_steps"
            or contract.payload.get("identity_fields") != expected_identity_fields
            or contract.payload.get("prefactor_position") != "before_generation_and_async_dispatch"
            or contract.payload.get("variant") != _SUPPORTED_VARIANT
            or contract.payload.get("decision_record_hash") != decision.content_hash
            or set(decision.payload) != {"decision", "declared_paths", "rationale", "scope"}
            or decision.payload.get("decision") != "ADOPT_CLASSIC_V1_WIDE_PREFACTOR_FIXTURE"
            or decision.payload.get("declared_paths") != list(_DECLARED_PATHS)
            or decision.payload.get("scope") != "fixture_only"
        ):
            raise ClassicIdentityError("wide-prefactor contract lineage is invalid")

    @staticmethod
    def _stage_contract(
        stage: Artifact,
        *,
        expected_name: str,
        expected_index: int,
        expected_parent: str,
        input_hash: str,
        contract_hash: str,
    ) -> ClassicBatch:
        if (
            set(stage.payload)
            != {"batch", "contract_hash", "identity_set_hash", "input_hash", "parent_hash", "stage_index", "stage_name"}
            or stage.payload.get("stage_name") != expected_name
            or stage.payload.get("stage_index") != expected_index
            or stage.payload.get("parent_hash") != expected_parent
            or stage.payload.get("input_hash") != input_hash
            or stage.payload.get("contract_hash") != contract_hash
        ):
            raise ClassicIdentityError("classic stage lineage is invalid")
        batch = ClassicBatch.from_mapping(stage.payload.get("batch"))
        if stage.payload.get("identity_set_hash") != batch.identity_set_hash:
            raise ClassicIdentityError("classic stage identity root changed")
        return batch

    @staticmethod
    def _dump_contract(dump: Artifact, input_hash: str, source_stage_hash: str, baseline: str) -> ClassicBatch:
        expected = {
            "cfs_integration_claimed",
            "global_steps",
            "identity_set_hash",
            "input_hash",
            "rows",
            "source_stage_hash",
            "transport_chunks",
            "variant",
        }
        if (
            set(dump.payload) != expected
            or dump.payload.get("input_hash") != input_hash
            or dump.payload.get("source_stage_hash") != source_stage_hash
            or dump.payload.get("identity_set_hash") != baseline
            or dump.payload.get("variant") != _SUPPORTED_VARIANT
            or dump.payload.get("cfs_integration_claimed") is not False
        ):
            raise ClassicIdentityError("classic trajectory dump lineage is invalid")
        batch = ClassicBatch.from_mapping(
            {
                "global_steps": dump.payload["global_steps"],
                "rows": dump.payload["rows"],
                "transport_chunks": dump.payload["transport_chunks"],
            }
        )
        if batch.identity_set_hash != baseline:
            raise ClassicIdentityError("classic dump identity root changed")
        return batch

    @staticmethod
    def _publish_ref(store: ArtifactStore, ref: Path, artifact: Artifact, code: str) -> None:
        try:
            ArtifactStore._publish(ref, f"{artifact.content_hash}\n".encode("ascii"))
            return
        except ImmutableArtifactConflict:
            pass
        try:
            existing = ref.read_text(encoding="ascii").strip()
        except OSError as error:
            raise ClassicIdentityError(f"{code}:ref unavailable") from error
        if existing == artifact.content_hash:
            return
        conflict = store.put(
            "ClassicIdentityConflict",
            "1.0.0",
            {
                "conflicting_hash": artifact.content_hash,
                "existing_hash": existing,
                "reason_code": code,
                "status": "corruption",
            },
        )
        raise ClassicIdentityError(f"{code}:{conflict.content_hash}")
