"""Fixture and production execution profiles with phase-scoped readiness gates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from clawrl.artifacts import (
    CanonicalizationError,
    JsonValue,
    canonical_json_bytes,
)

_FAILURE_CODE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_CONTENT_HASH = re.compile(r"^[0-9a-f]{64}$")
_PRODUCTION_PHASES = {"DATA_INGEST", "JUDGE_CERTIFY", "TRAIN_35B", "TRAIN_122B", "FINAL_EVAL"}
_FAULT_DIRECTIVES = {"timeout", "delayed", "success", "permanent_failure"}
_FAULT_TERMINALS = {"success", "permanent_failure"}


class ProfileConfigurationError(ValueError):
    """A profile attempts to carry fields outside its structural boundary."""


@dataclass(frozen=True, slots=True)
class FixtureProfileConfig:
    """Fixture-only knobs; no field can contain a production adapter or credential."""

    execution_profile: Literal["fixture"] = "fixture"
    scorer_failure_code: str | None = None
    cluster_failure_code: str | None = None
    fault_schedule_version: Literal["fixture-fault-schedule/1.0.0"] = "fixture-fault-schedule/1.0.0"
    scorer_fault_schedule: tuple[str, ...] = ()
    cluster_fault_schedule: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.execution_profile != "fixture":
            raise ProfileConfigurationError("fixture config requires execution_profile='fixture'")
        if self.fault_schedule_version != "fixture-fault-schedule/1.0.0":
            raise ProfileConfigurationError("fixture fault schedule requires version 1.0.0")
        for field_name, code in (
            ("scorer_failure_code", self.scorer_failure_code),
            ("cluster_failure_code", self.cluster_failure_code),
        ):
            if code is not None and _FAILURE_CODE.fullmatch(code) is None:
                raise ProfileConfigurationError(f"{field_name} must be a machine-readable code")
        self._validate_fault_schedule(
            "scorer_fault_schedule",
            self.scorer_fault_schedule,
            self.scorer_failure_code,
        )
        self._validate_fault_schedule(
            "cluster_fault_schedule",
            self.cluster_fault_schedule,
            self.cluster_failure_code,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FixtureProfileConfig:
        allowed = {
            "execution_profile",
            "scorer_failure_code",
            "cluster_failure_code",
            "fault_schedule_version",
            "scorer_fault_schedule",
            "cluster_fault_schedule",
        }
        unexpected = sorted(set(value) - allowed)
        if unexpected:
            raise ProfileConfigurationError(
                f"fixture profile cannot contain production configuration field {unexpected[0]!r}"
            )
        profile = value.get("execution_profile", "fixture")
        scorer_failure = value.get("scorer_failure_code")
        cluster_failure = value.get("cluster_failure_code")
        schedule_version = value.get("fault_schedule_version", "fixture-fault-schedule/1.0.0")
        scorer_schedule = value.get("scorer_fault_schedule", [])
        cluster_schedule = value.get("cluster_fault_schedule", [])
        if not isinstance(profile, str):
            raise ProfileConfigurationError("execution_profile must be a string")
        if scorer_failure is not None and not isinstance(scorer_failure, str):
            raise ProfileConfigurationError("scorer_failure_code must be a string or null")
        if cluster_failure is not None and not isinstance(cluster_failure, str):
            raise ProfileConfigurationError("cluster_failure_code must be a string or null")
        if not isinstance(schedule_version, str):
            raise ProfileConfigurationError("fault_schedule_version must be a string")
        if not isinstance(scorer_schedule, list) or not all(isinstance(item, str) for item in scorer_schedule):
            raise ProfileConfigurationError("scorer_fault_schedule must be a string list")
        if not isinstance(cluster_schedule, list) or not all(isinstance(item, str) for item in cluster_schedule):
            raise ProfileConfigurationError("cluster_fault_schedule must be a string list")
        return cls(
            execution_profile=cast(Literal["fixture"], profile),
            scorer_failure_code=scorer_failure,
            cluster_failure_code=cluster_failure,
            fault_schedule_version=cast(Literal["fixture-fault-schedule/1.0.0"], schedule_version),
            scorer_fault_schedule=tuple(scorer_schedule),
            cluster_fault_schedule=tuple(cluster_schedule),
        )

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "cluster_failure_code": self.cluster_failure_code,
            "cluster_fault_schedule": list(self.cluster_fault_schedule),
            "execution_profile": self.execution_profile,
            "fault_schedule_version": self.fault_schedule_version,
            "scorer_failure_code": self.scorer_failure_code,
            "scorer_fault_schedule": list(self.scorer_fault_schedule),
        }

    def effective_scorer_fault_schedule(self) -> tuple[str, ...]:
        if self.scorer_fault_schedule:
            return self.scorer_fault_schedule
        return ("permanent_failure",) if self.scorer_failure_code else ("success",)

    def effective_cluster_fault_schedule(self) -> tuple[str, ...]:
        if self.cluster_fault_schedule:
            return self.cluster_fault_schedule
        return ("permanent_failure",) if self.cluster_failure_code else ("success",)

    @staticmethod
    def _validate_fault_schedule(
        field_name: str,
        schedule: tuple[str, ...],
        failure_code: str | None,
    ) -> None:
        if not isinstance(schedule, tuple) or not all(isinstance(item, str) for item in schedule):
            raise ProfileConfigurationError(f"{field_name} must be a tuple of directives")
        if not schedule:
            return
        if (
            len(schedule) > 16
            or any(item not in _FAULT_DIRECTIVES for item in schedule)
            or schedule[-1] not in _FAULT_TERMINALS
            or any(item in _FAULT_TERMINALS for item in schedule[:-1])
        ):
            raise ProfileConfigurationError(f"{field_name} must end once with success or permanent_failure")
        if schedule[-1] == "permanent_failure" and failure_code is None:
            raise ProfileConfigurationError(f"{field_name} permanent_failure requires a failure code")


@dataclass(frozen=True, slots=True)
class ProductionProfileConfig:
    """External references for a real phase; Ticket 01 has no real adapters."""

    phase: str
    model_id: str | None = None
    dataset_version_hash: str | None = None
    judge_bundle_hash: str | None = None
    experiment_spec_hash: str | None = None
    cfs_root: str | None = None
    cluster_credential_ref: str | None = None
    fixture_evidence_hashes: tuple[str, ...] = ()
    execution_profile: Literal["production"] = "production"

    def __post_init__(self) -> None:
        if not isinstance(self.execution_profile, str):
            raise ProfileConfigurationError("production execution_profile must be a string")
        if self.execution_profile != "production":
            raise ProfileConfigurationError("production config requires execution_profile='production'")
        if not isinstance(self.phase, str):
            raise ProfileConfigurationError("production phase must be a string")
        if self.phase not in _PRODUCTION_PHASES:
            raise ProfileConfigurationError(f"unknown production readiness phase {self.phase!r}")
        for field_name in (
            "model_id",
            "dataset_version_hash",
            "judge_bundle_hash",
            "experiment_spec_hash",
            "cfs_root",
            "cluster_credential_ref",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ProfileConfigurationError(f"production {field_name} must be a string or null")
        for field_name in (
            "dataset_version_hash",
            "judge_bundle_hash",
            "experiment_spec_hash",
        ):
            value = getattr(self, field_name)
            if isinstance(value, str) and value and _CONTENT_HASH.fullmatch(value) is None:
                raise ProfileConfigurationError(f"production {field_name} must be a SHA-256 content hash")
        if not isinstance(self.fixture_evidence_hashes, tuple) or not all(
            isinstance(item, str) and _CONTENT_HASH.fullmatch(item) is not None for item in self.fixture_evidence_hashes
        ):
            raise ProfileConfigurationError("production fixture_evidence_hashes must be a tuple of SHA-256 hashes")


ProfileConfig = FixtureProfileConfig | ProductionProfileConfig


def runtime_profile_is_valid(config: object) -> bool:
    """Defensively revalidate frozen profiles at the dispatch boundary."""

    try:
        if type(config) is FixtureProfileConfig:
            fixture = cast(FixtureProfileConfig, config)
            validated_fixture = FixtureProfileConfig(
                execution_profile=fixture.execution_profile,
                scorer_failure_code=fixture.scorer_failure_code,
                cluster_failure_code=fixture.cluster_failure_code,
                fault_schedule_version=fixture.fault_schedule_version,
                scorer_fault_schedule=fixture.scorer_fault_schedule,
                cluster_fault_schedule=fixture.cluster_fault_schedule,
            )
            canonical_json_bytes(validated_fixture.artifact_payload())
            return True
        if type(config) is ProductionProfileConfig:
            production = cast(ProductionProfileConfig, config)
            validated_production = ProductionProfileConfig(
                phase=production.phase,
                model_id=production.model_id,
                dataset_version_hash=production.dataset_version_hash,
                judge_bundle_hash=production.judge_bundle_hash,
                experiment_spec_hash=production.experiment_spec_hash,
                cfs_root=production.cfs_root,
                cluster_credential_ref=production.cluster_credential_ref,
                fixture_evidence_hashes=production.fixture_evidence_hashes,
                execution_profile=production.execution_profile,
            )
            canonical_json_bytes(
                {
                    "cfs_root": validated_production.cfs_root,
                    "cluster_credential_ref": validated_production.cluster_credential_ref,
                    "dataset_version_hash": validated_production.dataset_version_hash,
                    "execution_profile": validated_production.execution_profile,
                    "experiment_spec_hash": validated_production.experiment_spec_hash,
                    "fixture_evidence_hashes": list(validated_production.fixture_evidence_hashes),
                    "judge_bundle_hash": validated_production.judge_bundle_hash,
                    "model_id": validated_production.model_id,
                    "phase": validated_production.phase,
                }
            )
            return True
    except (CanonicalizationError, ProfileConfigurationError, TypeError):
        return False
    return False


def invalid_profile_readiness_payload() -> dict[str, JsonValue]:
    """Return a fixed report that cannot echo malformed runtime configuration."""

    return {
        "checks": [
            {
                "code": "INVALID_PRODUCTION_CONFIGURATION",
                "detail": "execution profile failed defensive runtime validation",
                "status": "blocked",
            },
            {
                "code": "PRODUCTION_ADAPTER_CONTRACT_UNVERIFIED",
                "detail": "real scorer, CFS, and cluster contracts are unavailable in this environment",
                "status": "blocked",
            },
        ],
        "execution_profile": "production",
        "fixture_evidence_accepted": False,
        "phase": "INVALID",
        "side_effects_permitted": False,
        "status": "blocked",
    }


def production_readiness_payload(config: ProductionProfileConfig) -> dict[str, JsonValue]:
    """Return deterministic blockers before any production adapter construction."""

    checks: list[JsonValue] = []

    def block(code: str, detail: str) -> None:
        checks.append({"code": code, "detail": detail, "status": "blocked"})

    scalar_fields = (
        config.phase,
        config.model_id,
        config.dataset_version_hash,
        config.judge_bundle_hash,
        config.experiment_spec_hash,
        config.cfs_root,
        config.cluster_credential_ref,
        config.execution_profile,
    )
    evidence_shape_valid = isinstance(config.fixture_evidence_hashes, tuple) and all(
        isinstance(item, str) and _CONTENT_HASH.fullmatch(item) is not None for item in config.fixture_evidence_hashes
    )
    if (
        not isinstance(config.phase, str)
        or not all(value is None or isinstance(value, str) for value in scalar_fields)
        or not evidence_shape_valid
    ):
        block(
            "INVALID_PRODUCTION_CONFIGURATION",
            "production readiness fields have invalid runtime types or shapes",
        )

    required = (
        ("MISSING_DATASET_VERSION", config.dataset_version_hash),
        ("MISSING_JUDGE_BUNDLE", config.judge_bundle_hash),
        ("MISSING_EXPERIMENT_SPEC", config.experiment_spec_hash),
        ("MISSING_CFS_ROOT", config.cfs_root),
        ("MISSING_CLUSTER_CREDENTIAL_REF", config.cluster_credential_ref),
    )
    for code, value in required:
        if not isinstance(value, str) or not value.strip():
            block(code, "required external production reference is absent")
    model_id = config.model_id if isinstance(config.model_id, str) else ""
    if not model_id or model_id.lower().startswith(("fixture", "placeholder", "synthetic")):
        block("PLACEHOLDER_MODEL_ID", "fixture, synthetic, or missing model identity is invalid")
    if isinstance(config.fixture_evidence_hashes, tuple) and config.fixture_evidence_hashes:
        block(
            "FIXTURE_EVIDENCE_NOT_PRODUCTION_PROOF",
            "fixture artifact hashes cannot satisfy a production readiness check",
        )
    block(
        "PRODUCTION_ADAPTER_CONTRACT_UNVERIFIED",
        "real scorer, CFS, and cluster contracts are unavailable in this environment",
    )
    return {
        "checks": checks,
        "execution_profile": "production",
        "fixture_evidence_accepted": False,
        "phase": config.phase if isinstance(config.phase, str) else "INVALID",
        "side_effects_permitted": False,
        "status": "blocked",
    }
