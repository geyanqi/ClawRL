"""Immutable registry and preregistration gate for FinalEvaluationProtocol."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, JsonValue
from clawrl.data.models import ApprovedDataMix, DataContractError, DataSourceSkill, QueryWindow
from clawrl.evaluation.protocol_models import (
    EvaluationEnvironment,
    FinalEvaluationContractError,
    FinalEvaluationProtocolConfig,
    ProductionFinalEvaluationConfig,
    PromptIdentityNormalizer,
)
from clawrl.judge.fit_models import FitContractError, InitialEvalRubric

_HASH = re.compile(r"^[0-9a-f]{64}$")


class FinalEvaluationProtocolError(RuntimeError):
    """Protocol preregistration or receipt authorization failed closed."""


class ProtocolBindingConflictError(FinalEvaluationProtocolError):
    """A campaign attempted to replace its immutable protocol."""


@dataclass(frozen=True, slots=True)
class ProtocolPreregistrationSnapshot:
    protocol: Artifact
    receipt: Artifact
    decision_record: Artifact
    environment: Artifact


class FinalEvaluationProtocolRegistry:
    @staticmethod
    def production_readiness(root: str | Path, config: ProductionFinalEvaluationConfig) -> Artifact:
        checks = []
        for code, value in (
            ("TRUSTED_CONTROLLER_CLOCK_UNAVAILABLE", config.trusted_clock_approval_hash),
            ("FINAL_EVAL_DATA_SOURCE_UNAVAILABLE", config.data_source_approval_hash),
            ("SEALED_MAPPING_STORAGE_UNAVAILABLE", config.sealed_storage_approval_hash),
            ("PROVIDER_IDEMPOTENCY_CONTRACT_UNAVAILABLE", config.provider_contract_approval_hash),
        ):
            if type(value) is not str or _HASH.fullmatch(cast(str, value)) is None:
                checks.append({"code": code, "status": "blocked"})
        if not checks:
            checks.append({"code": "PRODUCTION_FINAL_EVAL_BOUNDARIES_NOT_CONFIGURED", "status": "blocked"})
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
    def preregister(
        cls, root: str | Path, *, campaign_id: str, config: FinalEvaluationProtocolConfig
    ) -> ProtocolPreregistrationSnapshot:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", campaign_id) is None:
            raise FinalEvaluationProtocolError("campaign_id is invalid")
        store = ArtifactStore(root)
        binding_dir = Path(root) / "protocol-bindings" / campaign_id
        cls._assert_campaign_valid(store, binding_dir, campaign_id)
        wrapper = store.put(
            "FinalEvaluationPromptWrapper",
            "1.0.0",
            {"template": "{{prompt}}", "version": "identity-wrapper-v1"},
        )
        environment = store.put("EvaluationEnvironment", "1.0.0", config.environment.artifact_payload())
        generation_configs = {
            role: store.put(
                "SemanticGenerationConfig",
                "1.0.0",
                {
                    "checkpoint_binding": "deferred_to_candidate_freeze",
                    "evaluation_environment_hash": environment.content_hash,
                    "model_role": role,
                    "prompt_wrapper_hash": wrapper.content_hash,
                    "semantic_decoding": config.environment.artifact_payload()["semantic_decoding"],
                    "semantic_equivalence_required": True,
                },
            )
            for role in ("base", "trained")
        }
        skill = cls._fixture_data_source_skill(config)
        data_source = store.put("DataSourceSkill", "1.0.0", skill.artifact_payload())
        normalizer_contract = PromptIdentityNormalizer(config.identity_normalizer_version)
        normalizer = store.put(
            "PromptIdentityNormalizer",
            "1.0.0",
            normalizer_contract.artifact_payload(),
        )
        idempotency = store.put(
            "ProviderIdempotencyKeySchema",
            "1.0.0",
            {
                "canonicalization": "rfc8785-json",
                "hash_algorithm": "sha256",
                "key_fields": ["campaign_id", "prompt_identity_hash", "model_role"],
                "retry_behavior": "reuse_exact_same_key",
                "schema_version": "provider-idempotency-key/1.0.0",
            },
        )
        decision = store.put(
            "DecisionRecord",
            "1.0.0",
            {
                "decision": "ADOPT_FIXTURE_FINAL_EVAL_V1_DEFAULTS",
                "rationale": "Ticket09 fixture freezes explicit semantics without guessing production configuration.",
                "scope": "fixture_only",
            },
        )
        protocol = store.put(
            "FinalEvaluationProtocol",
            "1.0.0",
            config.canonical_payload(
                environment_hash=environment.content_hash,
                decision_record_hash=decision.content_hash,
                data_source_skill_hash=data_source.content_hash,
                identity_normalizer_hash=normalizer.content_hash,
                idempotency_key_schema_hash=idempotency.content_hash,
                prompt_wrapper_hash=wrapper.content_hash,
                base_generation_config_hash=generation_configs["base"].content_hash,
                trained_generation_config_hash=generation_configs["trained"].content_hash,
            ),
        )
        ref = binding_dir / "active.ref"
        if ref.exists():
            current = cls._load_binding(store, ref, campaign_id)
            if current.protocol.content_hash == protocol.content_hash:
                return current
            conflict = store.put(
                "FinalEvaluationProtocolConflict",
                "1.0.0",
                {
                    "bound_protocol_hash": current.protocol.content_hash,
                    "campaign_id": campaign_id,
                    "conflicting_protocol_hash": protocol.content_hash,
                    "reason_code": "INCOMPATIBLE_PROTOCOL_CANNOT_REPLACE_BOUND_CAMPAIGN",
                    "status": "invalid",
                },
            )
            invalidation = store.put(
                "FinalEvaluationCampaignInvalidation",
                "1.0.0",
                {
                    "campaign_id": campaign_id,
                    "conflict_hash": conflict.content_hash,
                    "side_effects_permitted": False,
                    "status": "invalid",
                },
            )
            ArtifactStore.durable_mkdir(binding_dir)
            ArtifactStore._publish(binding_dir / "invalid.ref", f"{invalidation.content_hash}\n".encode("ascii"))
            raise ProtocolBindingConflictError("INCOMPATIBLE_PROTOCOL_CANNOT_REPLACE_BOUND_CAMPAIGN")
        receipt = store.put(
            "FinalEvaluationPreregistrationReceipt",
            "1.0.0",
            {
                "campaign_id": campaign_id,
                "decision_record_hash": decision.content_hash,
                "evaluation_environment_hash": environment.content_hash,
                "execution_profile": "fixture",
                "protocol_hash": protocol.content_hash,
                "sequence": 1,
                "status": "preregistered_before_candidate_results",
            },
        )
        binding = store.put(
            "FinalEvaluationProtocolBinding",
            "1.0.0",
            {
                "campaign_id": campaign_id,
                "protocol_hash": protocol.content_hash,
                "receipt_hash": receipt.content_hash,
            },
        )
        ArtifactStore.durable_mkdir(binding_dir)
        ArtifactStore._publish(ref, f"{binding.content_hash}\n".encode("ascii"))
        return cls._load_binding(store, ref, campaign_id)

    @classmethod
    def authorize_candidate_freeze(
        cls,
        root: str | Path,
        *,
        campaign_id: str,
        protocol_hash: str,
        preregistration_receipt_hash: str | None,
    ) -> Artifact:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", campaign_id) is None:
            raise FinalEvaluationProtocolError("campaign_id is invalid")
        store = ArtifactStore(root)
        binding_dir = Path(root) / "protocol-bindings" / campaign_id
        cls._assert_campaign_valid(store, binding_dir, campaign_id)
        ref = binding_dir / "active.ref"
        if preregistration_receipt_hash is None or not ref.exists():
            raise FinalEvaluationProtocolError("PREREGISTRATION_RECEIPT_REQUIRED")
        snapshot = cls._load_binding(store, ref, campaign_id)
        if (
            snapshot.protocol.content_hash != protocol_hash
            or snapshot.receipt.content_hash != preregistration_receipt_hash
        ):
            raise FinalEvaluationProtocolError("PREREGISTRATION_RECEIPT_MISMATCH")
        return store.put(
            "CandidateFreezeProtocolAuthorization",
            "1.0.0",
            {
                "campaign_id": campaign_id,
                "protocol_hash": protocol_hash,
                "receipt_hash": preregistration_receipt_hash,
                "status": "authorized",
            },
        )

    @staticmethod
    def _assert_campaign_valid(store: ArtifactStore, binding_dir: Path, campaign_id: str) -> None:
        invalid_ref = binding_dir / "invalid.ref"
        if not invalid_ref.exists():
            return
        try:
            invalidation = store.read(
                invalid_ref.read_text(encoding="ascii").strip(),
                expected_schema_name="FinalEvaluationCampaignInvalidation",
            )
        except (ArtifactCorruption, OSError) as error:
            raise FinalEvaluationProtocolError("campaign invalidation cannot be recertified") from error
        if invalidation.payload.get("campaign_id") != campaign_id or invalidation.payload.get("status") != "invalid":
            raise FinalEvaluationProtocolError("campaign invalidation identity changed")
        raise FinalEvaluationProtocolError("CAMPAIGN_INVALID_AFTER_PROTOCOL_CONFLICT")

    @classmethod
    def load(cls, root: str | Path, *, campaign_id: str) -> ProtocolPreregistrationSnapshot:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", campaign_id) is None:
            raise FinalEvaluationProtocolError("campaign_id is invalid")
        store = ArtifactStore(root)
        ref = Path(root) / "protocol-bindings" / campaign_id / "active.ref"
        if not ref.exists():
            raise FinalEvaluationProtocolError("campaign has no preregistered protocol")
        return cls._load_binding(store, ref, campaign_id)

    @staticmethod
    def _load_binding(store: ArtifactStore, ref: Path, expected_campaign_id: str) -> ProtocolPreregistrationSnapshot:
        try:
            binding_hash = ref.read_text(encoding="ascii").strip()
            binding = store.read(binding_hash, expected_schema_name="FinalEvaluationProtocolBinding")
            protocol = store.read(
                cast(str, binding.payload["protocol_hash"]), expected_schema_name="FinalEvaluationProtocol"
            )
            receipt = store.read(
                cast(str, binding.payload["receipt_hash"]), expected_schema_name="FinalEvaluationPreregistrationReceipt"
            )
            decision = store.read(
                cast(str, receipt.payload["decision_record_hash"]), expected_schema_name="DecisionRecord"
            )
            environment = store.read(
                cast(str, receipt.payload["evaluation_environment_hash"]), expected_schema_name="EvaluationEnvironment"
            )
            data_source = store.read(
                cast(str, protocol.payload["data_source_skill_hash"]), expected_schema_name="DataSourceSkill"
            )
            loaded_skill = DataSourceSkill.from_mapping(cast(dict[str, object], data_source.payload))
            identity = cast(dict[str, object], protocol.payload["identity_exclusion"])
            normalizer = store.read(
                cast(str, identity["normalizer_hash"]), expected_schema_name="PromptIdentityNormalizer"
            )
            loaded_normalizer = PromptIdentityNormalizer.from_mapping(cast(dict[str, object], normalizer.payload))
            retry = cast(dict[str, object], protocol.payload["retry_idempotency"])
            idempotency = store.read(
                cast(str, retry["idempotency_key_schema_hash"]),
                expected_schema_name="ProviderIdempotencyKeySchema",
            )
            generation = cast(dict[str, object], protocol.payload["generation_configs"])
            wrapper = store.read(
                cast(str, generation["prompt_wrapper_hash"]),
                expected_schema_name="FinalEvaluationPromptWrapper",
            )
            base_generation = store.read(
                cast(str, generation["base_generation_config_hash"]),
                expected_schema_name="SemanticGenerationConfig",
            )
            trained_generation = store.read(
                cast(str, generation["trained_generation_config_hash"]),
                expected_schema_name="SemanticGenerationConfig",
            )
        except (
            ArtifactCorruption,
            DataContractError,
            FinalEvaluationContractError,
            FitContractError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            raise FinalEvaluationProtocolError("persisted protocol binding cannot be recertified") from error
        expected_binding: dict[str, JsonValue] = {
            "campaign_id": expected_campaign_id,
            "protocol_hash": protocol.content_hash,
            "receipt_hash": receipt.content_hash,
        }
        try:
            rubric = InitialEvalRubric.from_mapping(
                cast(dict[str, object], protocol.payload["initial_eval_rubric_raw"])
            )
            sample = cast(dict[str, object], protocol.payload["sample"])
            assignment = cast(dict[str, object], protocol.payload["balanced_ab_assignment"])
            query = cast(dict[str, object], protocol.payload["query_schema"])
            identity = cast(dict[str, object], protocol.payload["identity_exclusion"])
            window = cast(dict[str, object], protocol.payload["window"])
            reconstructed = FinalEvaluationProtocolConfig(
                protocol_id=cast(str, protocol.payload["protocol_id"]),
                rubric=rubric,
                environment=EvaluationEnvironment(environment_id=cast(str, environment.payload["environment_id"])),
                sample_seed=cast(int, sample["seed"]),
                balanced_ab_seed=cast(int, assignment["seed"]),
                source_contract_id=cast(str, query["source_contract_id"]),
                identity_normalizer_version=cast(str, identity["algorithm_version"]),
                window_duration_seconds=cast(int, window["initial_duration_seconds"]),
            )
            expected_protocol = reconstructed.canonical_payload(
                environment_hash=environment.content_hash,
                decision_record_hash=decision.content_hash,
                data_source_skill_hash=data_source.content_hash,
                identity_normalizer_hash=normalizer.content_hash,
                idempotency_key_schema_hash=idempotency.content_hash,
                prompt_wrapper_hash=wrapper.content_hash,
                base_generation_config_hash=base_generation.content_hash,
                trained_generation_config_hash=trained_generation.content_hash,
            )
        except (FinalEvaluationContractError, FitContractError, KeyError, TypeError, ValueError) as error:
            raise FinalEvaluationProtocolError("persisted protocol semantics cannot be reconstructed") from error
        expected_receipt: dict[str, object] = {
            "campaign_id": expected_campaign_id,
            "decision_record_hash": decision.content_hash,
            "evaluation_environment_hash": environment.content_hash,
            "execution_profile": "fixture",
            "protocol_hash": protocol.content_hash,
            "sequence": 1,
            "status": "preregistered_before_candidate_results",
        }
        expected_idempotency: dict[str, object] = {
            "canonicalization": "rfc8785-json",
            "hash_algorithm": "sha256",
            "key_fields": ["campaign_id", "prompt_identity_hash", "model_role"],
            "retry_behavior": "reuse_exact_same_key",
            "schema_version": "provider-idempotency-key/1.0.0",
        }
        expected_wrapper: dict[str, object] = {
            "template": "{{prompt}}",
            "version": "identity-wrapper-v1",
        }
        generation_common: dict[str, object] = {
            "checkpoint_binding": "deferred_to_candidate_freeze",
            "evaluation_environment_hash": environment.content_hash,
            "prompt_wrapper_hash": wrapper.content_hash,
            "semantic_decoding": environment.payload["semantic_decoding"],
            "semantic_equivalence_required": True,
        }
        expected_base_generation = {**generation_common, "model_role": "base"}
        expected_trained_generation = {**generation_common, "model_role": "trained"}
        if (
            binding.payload != expected_binding
            or protocol.payload != expected_protocol
            or receipt.payload != expected_receipt
            or loaded_skill.artifact_payload() != data_source.payload
            or loaded_normalizer.artifact_payload() != normalizer.payload
            or idempotency.payload != expected_idempotency
            or wrapper.payload != expected_wrapper
            or base_generation.payload != expected_base_generation
            or trained_generation.payload != expected_trained_generation
        ):
            raise FinalEvaluationProtocolError("persisted protocol lineage changed")
        return ProtocolPreregistrationSnapshot(protocol, receipt, decision, environment)

    @staticmethod
    def _fixture_data_source_skill(config: FinalEvaluationProtocolConfig) -> DataSourceSkill:
        columns = (
            "trace_pk",
            "prompt",
            "online_response",
            "event_time_utc",
            "ingestion_time_utc",
            "model_id",
            "tool_name",
            "private_sentinel",
            "difficulty",
            "purpose",
        )
        return DataSourceSkill(
            source_id=config.source_contract_id,
            table="fixture.online_trace",
            approved_window=QueryWindow("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
            approved_data_mix=ApprovedDataMix(),
            approved_columns=columns,
            purpose="eval_only",
            field_mapping=tuple(
                sorted(
                    {
                        "dedupe_key": "trace_pk",
                        "event_time": "event_time_utc",
                        "ingestion_time": "ingestion_time_utc",
                        "model": "model_id",
                        "primary_key": "trace_pk",
                        "prompt": "prompt",
                        "response": "online_response",
                        "tool": "tool_name",
                    }.items()
                )
            ),
            event_time_column="event_time_utc",
            ingestion_time_column="ingestion_time_utc",
            primary_key="trace_pk",
            dedupe_key="trace_pk",
            duplicate_semantics="identical normalized prompt rows collapse by trace_pk",
            allowed_filters=(
                "event_time_utc > :candidate_freeze_t0",
                "ingestion_time_utc > :candidate_freeze_t0",
                "difficulty = :difficulty",
                "purpose = :purpose",
            ),
            allowed_joins=(),
            sanitizer_version="sentinel-drop-v1",
            sensitive_columns=("private_sentinel",),
            max_rows=10_000,
            max_bytes=8_388_608,
            max_elapsed_ticks=32,
            expected_raw_rows=100,
            expected_unique_rows=100,
            selection_count=100,
            selection_policy_version="badcase-sha256-v1",
            post_query_invariants=(
                "field-completeness",
                "strict-event-and-ingestion-time-after-t0",
                "purpose-eval-only",
                "difficulty-hard",
                "identity-exclusion",
                "unique-prompt-identity",
            ),
        )
