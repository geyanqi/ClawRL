"""Closed-world domain inputs and profile configuration for Harness actions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, cast

from clawrl.artifacts import MAX_SAFE_INTEGER, JsonValue, canonical_json_bytes

INCREMENT_ACTION_TYPE: Final[Literal["fixture.resource.increment.v1"]] = "fixture.resource.increment.v1"
HARNESS_POLICY_VERSION: Final[Literal["fixture-harness-policy/1.0.0"]] = "fixture-harness-policy/1.0.0"
HARNESS_FAULT_SCHEDULE_VERSION: Final[Literal["fixture-harness-fault-schedule/1.0.0"]] = (
    "fixture-harness-fault-schedule/1.0.0"
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PHASE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_FAULT_DIRECTIVES = {
    "applied_then_timeout",
    "delayed",
    "permanent_failure",
    "success",
    "timeout",
}
_TRANSIENT_DIRECTIVES = {"applied_then_timeout", "delayed", "timeout"}
_TERMINAL_DIRECTIVES = {"permanent_failure", "success"}


class HarnessConfigurationError(ValueError):
    """A Harness profile or domain input violates the closed-world contract."""


HarnessDecisionValue = Literal["allow", "deny"]


def fixture_policy_payload(config: FixtureHarnessConfig) -> dict[str, JsonValue]:
    """Return the single frozen policy representation used by every boundary."""

    return {
        "allowed_action_types": [INCREMENT_ACTION_TYPE],
        "allowed_callers": cast(list[JsonValue], sorted(config.allowed_callers)),
        "allowed_phases": cast(list[JsonValue], sorted(config.allowed_phases)),
        "allowed_resources": cast(list[JsonValue], sorted(config.allowed_resources)),
        "policy_id": config.policy_id,
        "policy_version": config.policy_version,
    }


def fixture_policy_capability_payload(
    config: FixtureHarnessConfig,
    policy_hash: str,
) -> dict[str, JsonValue]:
    """Return the provider's immutable, least-authority policy capability."""

    return {
        "allowed_action_types": [INCREMENT_ACTION_TYPE],
        "allowed_resources": cast(list[JsonValue], sorted(config.allowed_resources)),
        "policy_hash": policy_hash,
        "policy_version": config.policy_version,
    }


def valid_fixture_policy_capability(
    capability: Mapping[str, object],
    policy: Mapping[str, object],
    *,
    policy_content_hash: str,
) -> bool:
    """Validate the exact immutable capability/policy pair without side effects."""

    capability_fields = {
        "allowed_action_types",
        "allowed_resources",
        "policy_hash",
        "policy_version",
    }
    policy_fields = {
        "allowed_action_types",
        "allowed_callers",
        "allowed_phases",
        "allowed_resources",
        "policy_id",
        "policy_version",
    }
    if set(capability) != capability_fields or set(policy) != policy_fields:
        return False
    capability_resources = capability.get("allowed_resources")
    policy_resources = policy.get("allowed_resources")
    if (
        capability.get("allowed_action_types") != [INCREMENT_ACTION_TYPE]
        or policy.get("allowed_action_types") != [INCREMENT_ACTION_TYPE]
        or not _valid_sorted_unique_ids(capability_resources, _SAFE_ID)
        or capability_resources != policy_resources
        or not _valid_sorted_unique_ids(policy.get("allowed_callers"), _SAFE_ID)
        or not _valid_sorted_unique_ids(policy.get("allowed_phases"), _PHASE)
        or not isinstance(policy.get("policy_id"), str)
        or _SAFE_ID.fullmatch(cast(str, policy.get("policy_id"))) is None
        or capability.get("policy_hash") != policy_content_hash
        or _HASH.fullmatch(policy_content_hash) is None
        or capability.get("policy_version") != HARNESS_POLICY_VERSION
        or policy.get("policy_version") != HARNESS_POLICY_VERSION
    ):
        return False
    return True


def fixture_capability_allows(
    capability: Mapping[str, object],
    *,
    action_type: object,
    resource_id: object,
) -> bool:
    """Return whether an already validated capability grants this exact action."""

    allowed_action_types = capability.get("allowed_action_types")
    allowed_resources = capability.get("allowed_resources")
    return bool(
        isinstance(action_type, str)
        and isinstance(resource_id, str)
        and isinstance(allowed_action_types, list)
        and action_type in allowed_action_types
        and isinstance(allowed_resources, list)
        and resource_id in allowed_resources
    )


def _valid_sorted_unique_ids(value: object, pattern: re.Pattern[str]) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and pattern.fullmatch(item) is not None for item in value)
        and value == sorted(set(value))
    )


def expected_harness_decision(
    *,
    caller: object,
    phase: object,
    action_type: object,
    resource_id: object,
    policy: Mapping[str, object],
) -> tuple[HarnessDecisionValue, str]:
    """Evaluate the exact closed-world decision tuple from one frozen policy.

    Retries are intentionally re-authorized to the same tuple. Recovery is an
    outcome property, not an additional value that can widen the allow branch.
    """

    if not isinstance(action_type, str) or action_type not in _policy_string_set(policy, "allowed_action_types"):
        return "deny", "ACTION_TYPE_NOT_ALLOWLISTED"
    if not isinstance(caller, str) or caller not in _policy_string_set(policy, "allowed_callers"):
        return "deny", "CALLER_NOT_ALLOWLISTED"
    if not isinstance(phase, str) or phase not in _policy_string_set(policy, "allowed_phases"):
        return "deny", "PHASE_NOT_ALLOWLISTED"
    if not isinstance(resource_id, str) or resource_id not in _policy_string_set(policy, "allowed_resources"):
        return "deny", "RESOURCE_NOT_ALLOWLISTED"
    return "allow", "POLICY_ALLOWED"


def _policy_string_set(policy: Mapping[str, object], field: str) -> frozenset[str]:
    value = policy.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return frozenset()
    return frozenset(value)


def _require_safe_id(field: str, value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise HarnessConfigurationError(f"{field} must be a filesystem-safe stable identifier")
    return value


def _require_phase(value: object) -> str:
    if not isinstance(value, str) or _PHASE.fullmatch(value) is None:
        raise HarnessConfigurationError("phase must be an uppercase machine-readable identifier")
    return value


@dataclass(frozen=True, slots=True)
class IncrementFixtureResource:
    """The sole side-effect-capable action in the Ticket 02 closed world."""

    resource_id: str
    expected_version: int
    amount: int = 1

    def __post_init__(self) -> None:
        _require_safe_id("resource_id", self.resource_id)
        if (
            not isinstance(self.expected_version, int)
            or isinstance(self.expected_version, bool)
            or not 0 <= self.expected_version < MAX_SAFE_INTEGER
        ):
            raise HarnessConfigurationError("expected_version must be a nonnegative safe integer")
        if type(self.amount) is not int or self.amount != 1:
            raise HarnessConfigurationError("Ticket 02 increment amount is fixed to exactly one")

    @property
    def action_type(self) -> Literal["fixture.resource.increment.v1"]:
        return INCREMENT_ACTION_TYPE

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "action_type": self.action_type,
            "amount": self.amount,
            "expected_version": self.expected_version,
            "resource_id": self.resource_id,
        }


@dataclass(frozen=True, slots=True)
class DecisionProposal:
    """A typed proposal. Raw model text cannot inhabit this domain type."""

    proposal_id: str
    caller: str
    phase: str
    action: IncrementFixtureResource

    def __post_init__(self) -> None:
        _require_safe_id("proposal_id", self.proposal_id)
        _require_safe_id("caller", self.caller)
        _require_phase(self.phase)
        if type(self.action) is not IncrementFixtureResource:
            raise HarnessConfigurationError("DecisionProposal requires the exact closed-world action type")


@dataclass(frozen=True, slots=True)
class UntrustedHarnessInput:
    """Opaque model/user material that is hashed and disposed without action parsing."""

    input_id: str
    caller: str
    phase: str
    value: object

    def __post_init__(self) -> None:
        _require_safe_id("input_id", self.input_id)
        _require_safe_id("caller", self.caller)
        _require_phase(self.phase)


@dataclass(frozen=True, slots=True)
class FixtureHarnessConfig:
    """Synthetic policy and deterministic provider schedule; no production handles."""

    execution_profile: Literal["fixture"] = "fixture"
    policy_id: str = "fixture-harness-policy"
    policy_version: Literal["fixture-harness-policy/1.0.0"] = HARNESS_POLICY_VERSION
    allowed_callers: tuple[str, ...] = ("fixture-governor",)
    allowed_phases: tuple[str, ...] = ("TRAIN_35B",)
    allowed_resources: tuple[str, ...] = ("fixture-counter",)
    provider_fault_schedule: tuple[str, ...] = ("success",)
    fault_schedule_version: Literal["fixture-harness-fault-schedule/1.0.0"] = HARNESS_FAULT_SCHEDULE_VERSION

    def __post_init__(self) -> None:
        if self.execution_profile != "fixture":
            raise HarnessConfigurationError("fixture Harness config requires execution_profile='fixture'")
        _require_safe_id("policy_id", self.policy_id)
        if self.policy_version != HARNESS_POLICY_VERSION:
            raise HarnessConfigurationError("fixture Harness policy requires version 1.0.0")
        if self.fault_schedule_version != HARNESS_FAULT_SCHEDULE_VERSION:
            raise HarnessConfigurationError("fixture Harness fault schedule requires version 1.0.0")
        self._validate_ids("allowed_callers", self.allowed_callers)
        self._validate_ids("allowed_resources", self.allowed_resources)
        if (
            not isinstance(self.allowed_phases, tuple)
            or not self.allowed_phases
            or any(not isinstance(item, str) or _PHASE.fullmatch(item) is None for item in self.allowed_phases)
            or len(set(self.allowed_phases)) != len(self.allowed_phases)
        ):
            raise HarnessConfigurationError("allowed_phases must be a nonempty unique phase tuple")
        schedule = self.provider_fault_schedule
        if (
            not isinstance(schedule, tuple)
            or not 1 <= len(schedule) <= 16
            or any(not isinstance(item, str) or item not in _FAULT_DIRECTIVES for item in schedule)
            or schedule[-1] not in _TERMINAL_DIRECTIVES
            or any(item not in _TRANSIENT_DIRECTIVES for item in schedule[:-1])
        ):
            raise HarnessConfigurationError(
                "provider_fault_schedule must contain transient directives followed by one terminal directive"
            )
        canonical_json_bytes(self.artifact_payload())

    @staticmethod
    def _validate_ids(field: str, values: tuple[str, ...]) -> None:
        if (
            not isinstance(values, tuple)
            or not values
            or any(not isinstance(item, str) or _SAFE_ID.fullmatch(item) is None for item in values)
            or len(set(values)) != len(values)
        ):
            raise HarnessConfigurationError(f"{field} must be a nonempty unique safe-identifier tuple")

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "allowed_action_types": [INCREMENT_ACTION_TYPE],
            "allowed_callers": cast(list[JsonValue], sorted(self.allowed_callers)),
            "allowed_phases": cast(list[JsonValue], sorted(self.allowed_phases)),
            "allowed_resources": cast(list[JsonValue], sorted(self.allowed_resources)),
            "execution_profile": self.execution_profile,
            "fault_schedule_version": self.fault_schedule_version,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "provider_fault_schedule": list(self.provider_fault_schedule),
        }

    @classmethod
    def from_mapping(cls, value: object) -> FixtureHarnessConfig:
        if not isinstance(value, dict):
            raise HarnessConfigurationError("fixture Harness config mapping must be an object")
        allowed = {
            "allowed_callers",
            "allowed_phases",
            "allowed_resources",
            "execution_profile",
            "fault_schedule_version",
            "policy_id",
            "policy_version",
            "provider_fault_schedule",
        }
        unexpected = sorted(set(value) - allowed)
        if unexpected:
            raise HarnessConfigurationError(
                f"fixture Harness config cannot contain unexpected production field {unexpected[0]!r}"
            )

        def string_tuple(field: str, default: tuple[str, ...]) -> tuple[str, ...]:
            raw = value.get(field, list(default))
            if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
                raise HarnessConfigurationError(f"fixture Harness {field} must be a string list")
            return tuple(raw)

        execution_profile = value.get("execution_profile", "fixture")
        policy_id = value.get("policy_id", "fixture-harness-policy")
        policy_version = value.get("policy_version", HARNESS_POLICY_VERSION)
        fault_version = value.get("fault_schedule_version", HARNESS_FAULT_SCHEDULE_VERSION)
        if not all(isinstance(item, str) for item in (execution_profile, policy_id, policy_version, fault_version)):
            raise HarnessConfigurationError("fixture Harness scalar config fields must be strings")
        return cls(
            execution_profile=cast(Literal["fixture"], execution_profile),
            policy_id=cast(str, policy_id),
            policy_version=cast(Literal["fixture-harness-policy/1.0.0"], policy_version),
            allowed_callers=string_tuple("allowed_callers", ("fixture-governor",)),
            allowed_phases=string_tuple("allowed_phases", ("TRAIN_35B",)),
            allowed_resources=string_tuple("allowed_resources", ("fixture-counter",)),
            provider_fault_schedule=string_tuple("provider_fault_schedule", ("success",)),
            fault_schedule_version=cast(
                Literal["fixture-harness-fault-schedule/1.0.0"],
                fault_version,
            ),
        )

    @classmethod
    def from_artifact_payload(cls, payload: object) -> FixtureHarnessConfig:
        if not isinstance(payload, dict):
            raise HarnessConfigurationError("fixture Harness config payload must be an object")
        expected_fields = {
            "allowed_action_types",
            "allowed_callers",
            "allowed_phases",
            "allowed_resources",
            "execution_profile",
            "fault_schedule_version",
            "policy_id",
            "policy_version",
            "provider_fault_schedule",
        }
        if set(payload) != expected_fields or payload.get("allowed_action_types") != [INCREMENT_ACTION_TYPE]:
            raise HarnessConfigurationError("fixture Harness config artifact fields are invalid")

        def string_list(field: str) -> tuple[str, ...]:
            value = payload.get(field)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise HarnessConfigurationError(f"fixture Harness {field} must be a string list")
            return tuple(value)

        execution_profile = payload.get("execution_profile")
        policy_id = payload.get("policy_id")
        policy_version = payload.get("policy_version")
        fault_version = payload.get("fault_schedule_version")
        if not all(isinstance(item, str) for item in (execution_profile, policy_id, policy_version, fault_version)):
            raise HarnessConfigurationError("fixture Harness scalar config fields are invalid")
        return cls(
            execution_profile=cast(Literal["fixture"], execution_profile),
            policy_id=cast(str, policy_id),
            policy_version=cast(Literal["fixture-harness-policy/1.0.0"], policy_version),
            allowed_callers=string_list("allowed_callers"),
            allowed_phases=string_list("allowed_phases"),
            allowed_resources=string_list("allowed_resources"),
            provider_fault_schedule=string_list("provider_fault_schedule"),
            fault_schedule_version=cast(
                Literal["fixture-harness-fault-schedule/1.0.0"],
                fault_version,
            ),
        )


@dataclass(frozen=True, slots=True)
class ProductionHarnessConfig:
    """Production references. Ticket 02 deliberately has no production boundary."""

    phase: str
    whitelist_artifact_hash: str | None = None
    audit_config_hash: str | None = None
    policy_artifact_hash: str | None = None
    caller_identity: str | None = None
    execution_profile: Literal["production"] = "production"

    def __post_init__(self) -> None:
        _require_phase(self.phase)
        if self.execution_profile != "production":
            raise HarnessConfigurationError("production Harness config requires execution_profile='production'")
        for field in ("whitelist_artifact_hash", "audit_config_hash", "policy_artifact_hash"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or _HASH.fullmatch(value) is None):
                raise HarnessConfigurationError(f"{field} must be a SHA-256 artifact hash or null")
        if self.caller_identity is not None:
            _require_safe_id("caller_identity", self.caller_identity)


HarnessProfileConfig = FixtureHarnessConfig | ProductionHarnessConfig
