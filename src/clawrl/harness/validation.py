"""Closed-world validation of every committed Harness artifact and hash edge."""

from __future__ import annotations

import re
from pathlib import Path
from typing import NoReturn, cast

from clawrl.artifacts import Artifact, ArtifactError, ArtifactStore, canonical_json_bytes, sha256_hex
from clawrl.harness.journal import HarnessJournal, HarnessJournalCorruption
from clawrl.harness.models import (
    HARNESS_FAULT_SCHEDULE_VERSION,
    INCREMENT_ACTION_TYPE,
    DecisionProposal,
    FixtureHarnessConfig,
    expected_harness_decision,
    fixture_capability_allows,
    fixture_policy_capability_payload,
    fixture_policy_payload,
    valid_fixture_policy_capability,
)

_HASH = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TRANSITIONS = {
    "ActionObservation",
    "ActionPlan",
    "DecisionOutcome",
    "DecisionProposal",
    "HarnessDecision",
}
_COMMON = {
    "controller_epoch",
    "previous_transition_hash",
    "proposal_id",
    "workflow_sequence",
}
_PROPOSAL = _COMMON | {
    "action_type",
    "caller",
    "input_hash",
    "phase",
    "policy_capability_hash",
    "policy_hash",
}
_PLAN = _COMMON | {
    "action_input_hash",
    "child_index",
    "idempotency_key",
    "policy_capability_hash",
    "policy_hash",
    "proposal_hash",
}
_DECISION = _COMMON | {
    "attempt_sequence",
    "caller",
    "decision",
    "idempotency_key",
    "input_hash",
    "plan_hash",
    "policy_capability_hash",
    "policy_hash",
    "policy_version",
    "previous_outcome_hash",
    "proposal_hash",
    "reason_code",
}
_OBSERVATION = _COMMON | {
    "attempt_sequence",
    "decision_hash",
    "failure_code",
    "idempotency_key",
    "integrity_evidence_hash",
    "input_hash",
    "late_outcome_hashes",
    "output_hash",
    "policy_capability_hash",
    "provider_result_hash",
    "recovered",
    "receipt_hash",
    "retryable",
    "status",
}
_OUTCOME = _COMMON | {
    "attempt_sequence",
    "audit_event_hash",
    "decision_hash",
    "idempotency_key",
    "input_hash",
    "integrity_evidence_hash",
    "observation_hash",
    "output_hash",
    "policy_capability_hash",
    "policy_hash",
    "previous_outcome_hash",
    "proposal_hash",
    "reason_code",
    "receipt_hash",
    "status",
    "terminal",
}
_ACTION = {
    "action_type",
    "amount",
    "child_index",
    "expected_version",
    "policy_capability_hash",
    "proposal_hash",
    "resource_id",
}
_AUDIT = {
    "attempt_sequence",
    "caller",
    "decision_hash",
    "idempotency_key",
    "input_hash",
    "integrity_evidence_hash",
    "observation_hash",
    "output_hash",
    "policy_capability_hash",
    "policy_hash",
    "policy_version",
    "proposal_hash",
    "receipt_hash",
    "status",
}
_RESULT = {
    "attempt_sequence",
    "directive",
    "failure_code",
    "idempotency_key",
    "input_hash",
    "invocation_performed",
    "late_outcome_hashes",
    "output_hash",
    "policy_capability_hash",
    "producer",
    "recovered",
    "receipt_hash",
    "retryable",
    "schedule_hash",
    "schedule_version",
    "status",
}
_STATE = {
    "action_input_hash",
    "action_plan_hash",
    "action_type",
    "amount",
    "attempt_sequence",
    "child_index",
    "harness_decision_hash",
    "idempotency_key",
    "policy_hash",
    "previous_state_hash",
    "policy_capability_hash",
    "proposal_hash",
    "proposal_id",
    "resource_id",
    "value",
    "version",
}
_RECEIPT = {
    "action_input_hash",
    "action_plan_hash",
    "action_type",
    "attempt_sequence",
    "child_index",
    "harness_decision_hash",
    "idempotency_key",
    "policy_hash",
    "policy_capability_hash",
    "proposal_hash",
    "proposal_id",
    "resource_id",
    "resource_state_hash",
    "status",
}
_CAPABILITY = {
    "allowed_action_types",
    "allowed_resources",
    "policy_hash",
    "policy_version",
}
_LATE = {
    "action_type",
    "child_index",
    "idempotency_key",
    "input_hash",
    "observed_at_attempt",
    "origin_attempt_hash",
    "origin_attempt_sequence",
    "origin_directive",
    "policy_capability_hash",
    "proposal_hash",
    "schedule_hash",
    "schedule_version",
    "status",
}
_IDENTITY = {
    "config_hash",
    "input_hash",
    "policy_capability_hash",
    "policy_hash",
    "proposal_id",
}
_INTEGRITY_FAILURE = {
    "boundary_ref_set_hash",
    "candidate_transition_hash",
    "failure_code",
    "idempotency_key",
    "predecessor_transition_hash",
    "proposal_hash",
}


class HarnessBoundaryPreflightCorruption(HarnessJournalCorruption):
    """Boundary refs are unsafe, while the proposal journal remains appendable."""


def _fail(detail: str) -> NoReturn:
    raise HarnessJournalCorruption(detail)


def _exact(artifact: Artifact, schema: str, fields: set[str]) -> dict[str, object]:
    payload = cast(dict[str, object], artifact.payload)
    if artifact.schema_name != schema or artifact.schema_version != "1.0.0" or set(payload) != fields:
        _fail(f"{schema} closed-world schema or fields are invalid")
    return payload


def _hash(value: object, detail: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        _fail(detail)
    return cast(str, value)


def _integer(value: object, detail: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        _fail(detail)
    return cast(int, value)


def _read_ref(ref: Path) -> str:
    try:
        raw = ref.read_bytes()
    except OSError as error:
        raise HarnessJournalCorruption("required Harness lineage ref is missing") from error
    if len(raw) != 65 or raw[-1:] != b"\n":
        _fail("Harness lineage ref bytes are malformed")
    try:
        value = raw[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        raise HarnessJournalCorruption("Harness lineage ref is not ASCII") from error
    return _hash(value, "Harness lineage ref has no SHA-256 hash")


def _provider_observation_semantics(
    result: Artifact,
) -> tuple[str, str | None, str]:
    payload = cast(dict[str, object], result.payload)
    attempt = _integer(payload.get("attempt_sequence"), "provider result attempt is invalid", minimum=1)
    directive = payload.get("directive")
    status = payload.get("status")
    failure_code = payload.get("failure_code")
    output_value = payload.get("output_hash")
    invocation_performed = payload.get("invocation_performed")
    recovered = payload.get("recovered")
    receipt_value = payload.get("receipt_hash")
    retryable = payload.get("retryable")
    if (
        not isinstance(directive, str)
        or not isinstance(status, str)
        or (failure_code is not None and not isinstance(failure_code, str))
        or (output_value is not None and not isinstance(output_value, str))
        or not isinstance(invocation_performed, bool)
        or not isinstance(recovered, bool)
        or not isinstance(retryable, bool)
        or (receipt_value is not None and not isinstance(receipt_value, str))
    ):
        _fail("provider result semantic field types are invalid")
    output_hash = _hash(output_value, "provider output hash is invalid") if isinstance(output_value, str) else None
    receipt_hash = _hash(receipt_value, "provider receipt hash is invalid") if isinstance(receipt_value, str) else None
    if (output_hash is None) != (receipt_hash is None):
        _fail("provider state and receipt hashes must be present together")

    if recovered:
        if (
            invocation_performed
            or status != "succeeded"
            or failure_code is not None
            or retryable
            or output_hash is None
            or receipt_hash is None
        ):
            _fail("recovered provider result semantics are invalid")
    elif not invocation_performed:
        _fail("non-recovered provider result must represent one invocation")
    elif status == "conflict":
        if (
            directive not in {"applied_then_timeout", "success"}
            or failure_code != "RESOURCE_VERSION_CONFLICT"
            or retryable
            or output_hash is not None
            or receipt_hash is not None
        ):
            _fail("provider conflict result semantics are invalid")
    elif directive == "success":
        if (
            status != "succeeded"
            or failure_code is not None
            or retryable
            or output_hash is None
            or receipt_hash is None
        ):
            _fail("provider success result semantics are invalid")
    elif directive == "applied_then_timeout":
        if (
            status != "retryable"
            or failure_code != "PROVIDER_OUTCOME_UNKNOWN"
            or retryable is not True
            or output_hash is None
            or receipt_hash is None
        ):
            _fail("provider unknown-outcome result semantics are invalid")
    elif directive == "timeout":
        if (
            status != "retryable"
            or failure_code != "PROVIDER_TIMEOUT"
            or retryable is not True
            or output_hash is not None
            or receipt_hash is not None
        ):
            _fail("provider timeout result semantics are invalid")
    elif directive == "delayed":
        if (
            status != "retryable"
            or failure_code != "PROVIDER_RESULT_DELAYED"
            or retryable is not True
            or output_hash is not None
            or receipt_hash is not None
        ):
            _fail("provider delayed result semantics are invalid")
    elif directive == "permanent_failure":
        if (
            status != "failed"
            or failure_code != "PROVIDER_PERMANENT_FAILURE"
            or retryable
            or output_hash is not None
            or receipt_hash is not None
        ):
            _fail("provider permanent-failure result semantics are invalid")
    else:
        _fail("provider result directive/status combination is invalid")

    observation_status = (
        "succeeded" if status == "succeeded" else "conflict" if status == "conflict" else "provider_error"
    )
    observation_output = output_hash
    if observation_output is None:
        observation_output = sha256_hex(
            canonical_json_bytes(
                {
                    "attempt_sequence": attempt,
                    "failure_code": failure_code,
                    "provider_result_hash": result.content_hash,
                    "status": observation_status,
                }
            )
        )
    return observation_status, cast(str | None, failure_code), observation_output


def _expected_outcome_semantics(
    decision: str,
    decision_reason: object,
    observation: dict[str, object],
    attempt: int,
    schedule_length: int,
) -> tuple[str, str, bool]:
    observation_status = observation.get("status")
    if decision == "deny":
        if not isinstance(decision_reason, str):
            _fail("denied decision has no reason code")
        return "denied", decision_reason, True
    if observation_status == "succeeded":
        recovered = attempt > 1 or observation.get("recovered") is True
        return (
            "recovered" if recovered else "allowed",
            "ACTION_RECOVERED" if recovered else "ACTION_APPLIED",
            True,
        )
    failure_code = observation.get("failure_code")
    if observation_status == "conflict":
        if not isinstance(failure_code, str):
            _fail("conflict observation has no failure code")
        return "conflict", failure_code, True
    if observation_status == "provider_error":
        reason_code = failure_code if isinstance(failure_code, str) else "PROVIDER_FAILURE"
        terminal = not (observation.get("retryable") is True and attempt < schedule_length)
        return "provider_error", reason_code, terminal
    _fail("ActionObservation status is not closed-world")


def _validate_integrity_evidence(
    store: ArtifactStore,
    evidence_value: object,
    decision: Artifact,
    proposal: Artifact,
    idempotency_key: str,
    failure_code: object,
) -> None:
    if evidence_value is None:
        return
    evidence = store.read(
        _hash(evidence_value, "Harness integrity evidence hash is invalid"),
        expected_schema_name="HarnessIntegrityFailure",
    )
    payload = _exact(evidence, "HarnessIntegrityFailure", _INTEGRITY_FAILURE)
    if (
        payload.get("candidate_transition_hash") != decision.content_hash
        or payload.get("predecessor_transition_hash") != decision.payload.get("previous_transition_hash")
        or payload.get("proposal_hash") != proposal.content_hash
        or payload.get("idempotency_key") != idempotency_key
        or payload.get("failure_code") != failure_code
        or _HASH.fullmatch(cast(str, payload.get("boundary_ref_set_hash"))) is None
    ):
        _fail("Harness integrity evidence is not bound to the failed decision")


def validate_harness_history(
    root: Path,
    store: ArtifactStore,
    item: DecisionProposal,
    config: FixtureHarnessConfig,
    identity: Artifact,
    transitions: list[Artifact],
) -> None:
    """Verify exact shapes and every identity edge before state is trusted."""

    identity_payload = cast(dict[str, object], identity.payload)
    if identity.schema_version != "1.0.0" or set(identity_payload) != {
        "config_hash",
        "input_hash",
        "policy_capability_hash",
        "policy_hash",
        "proposal_id",
    }:
        _fail("Harness workflow identity fields are invalid")
    if identity_payload.get("proposal_id") != item.proposal_id:
        _fail("Harness workflow identity proposal binding is invalid")
    config_artifact = store.read(
        _hash(identity_payload.get("config_hash"), "Harness config hash is invalid"),
        expected_schema_name="FixtureHarnessConfig",
    )
    if config_artifact.schema_version != "1.0.0" or config_artifact.payload != config.artifact_payload():
        _fail("Harness fixture config identity is invalid")
    workflow_input = store.read(
        _hash(identity_payload.get("input_hash"), "Harness workflow input hash is invalid"),
        expected_schema_name="HarnessWorkflowInput",
    )
    expected_input = {
        "action": item.action.artifact_payload(),
        "caller": item.caller,
        "phase": item.phase,
        "proposal_id": item.proposal_id,
    }
    if workflow_input.schema_version != "1.0.0" or workflow_input.payload != expected_input:
        _fail("Harness workflow input identity is invalid")
    policy = store.read(
        _hash(identity_payload.get("policy_hash"), "Harness policy hash is invalid"),
        expected_schema_name="HarnessPolicy",
    )
    expected_policy = fixture_policy_payload(config)
    if policy.schema_version != "1.0.0" or policy.payload != expected_policy:
        _fail("Harness policy closed-world fields or identity are invalid")
    capability_hash = _hash(
        identity_payload.get("policy_capability_hash"),
        "Harness policy capability hash is invalid",
    )
    capability = store.read(
        capability_hash,
        expected_schema_name="HarnessPolicyCapability",
    )
    capability_payload = _exact(capability, "HarnessPolicyCapability", _CAPABILITY)
    if capability_payload != fixture_policy_capability_payload(config, policy.content_hash):
        _fail("Harness provider policy capability is not bound to the frozen policy")
    if not transitions:
        return

    proposal = transitions[0]
    proposal_payload = _exact(proposal, "DecisionProposal", _PROPOSAL)
    if (
        proposal_payload.get("action_type") != INCREMENT_ACTION_TYPE
        or proposal_payload.get("caller") != item.caller
        or proposal_payload.get("phase") != item.phase
        or proposal_payload.get("input_hash") != workflow_input.content_hash
        or proposal_payload.get("policy_capability_hash") != capability.content_hash
        or proposal_payload.get("policy_hash") != policy.content_hash
    ):
        _fail("DecisionProposal identity lineage is invalid")
    if len(transitions) == 1:
        return

    plan = transitions[1]
    plan_payload = _exact(plan, "ActionPlan", _PLAN)
    plan_child_index = _integer(plan_payload.get("child_index"), "ActionPlan child index is invalid")
    if (
        plan_child_index != 0
        or plan_payload.get("proposal_hash") != proposal.content_hash
        or plan_payload.get("policy_capability_hash") != capability.content_hash
        or plan_payload.get("policy_hash") != policy.content_hash
    ):
        _fail("ActionPlan proposal or policy lineage is invalid")
    action_hash = _hash(plan_payload.get("action_input_hash"), "ActionPlan input hash is invalid")
    action = store.read(action_hash, expected_schema_name="TypedHarnessAction")
    action_payload = _exact(action, "TypedHarnessAction", _ACTION)
    action_amount = _integer(action_payload.get("amount"), "typed action amount is invalid", minimum=1)
    action_child_index = _integer(action_payload.get("child_index"), "typed action child index is invalid")
    action_expected_version = _integer(
        action_payload.get("expected_version"),
        "typed action expected version is invalid",
    )
    expected_action = {
        **item.action.artifact_payload(),
        "child_index": 0,
        "policy_capability_hash": capability.content_hash,
        "proposal_hash": proposal.content_hash,
    }
    if (
        action_payload != expected_action
        or action_amount != item.action.amount
        or action_child_index != 0
        or action_expected_version != item.action.expected_version
    ):
        _fail("typed child action does not match its proposal")
    expected_key = sha256_hex(
        canonical_json_bytes(
            {
                "action_input_hash": action.content_hash,
                "child_index": 0,
                "domain": "harness-child-idempotency/1.0.0",
                "proposal_hash": proposal.content_hash,
            }
        )
    )
    if plan_payload.get("idempotency_key") != expected_key:
        _fail("ActionPlan idempotency key derivation is invalid")

    position = 2
    expected_attempt = 1
    previous_outcome: Artifact | None = None
    expected_late_refs: dict[str, str] = {}
    expected_result_refs: dict[str, str] = {}
    boundary_integrity_failed = False
    while position < len(transitions):
        decision = transitions[position]
        decision_payload = _exact(decision, "HarnessDecision", _DECISION)
        decision_attempt = _integer(
            decision_payload.get("attempt_sequence"),
            "HarnessDecision attempt sequence is invalid",
            minimum=1,
        )
        if (
            decision_attempt != expected_attempt
            or decision_payload.get("caller") != item.caller
            or decision_payload.get("idempotency_key") != expected_key
            or decision_payload.get("input_hash") != action.content_hash
            or decision_payload.get("plan_hash") != plan.content_hash
            or decision_payload.get("policy_capability_hash") != capability.content_hash
            or decision_payload.get("policy_hash") != policy.content_hash
            or decision_payload.get("policy_version") != config.policy_version
            or decision_payload.get("proposal_hash") != proposal.content_hash
            or decision_payload.get("previous_outcome_hash")
            != (previous_outcome.content_hash if previous_outcome is not None else None)
        ):
            _fail("HarnessDecision proposal, policy, caller, input, or retry lineage is invalid")
        expected_decision = expected_harness_decision(
            caller=item.caller,
            phase=item.phase,
            action_type=action_payload.get("action_type"),
            resource_id=action_payload.get("resource_id"),
            policy=policy.payload,
        )
        if (decision_payload.get("decision"), decision_payload.get("reason_code")) != expected_decision:
            _fail("HarnessDecision does not equal the frozen policy decision")
        if position + 1 >= len(transitions):
            try:
                _validate_pending_boundary_refs(
                    root,
                    store,
                    action,
                    expected_key,
                    capability.content_hash,
                    config,
                    expected_attempt,
                    expected_result_refs,
                    expected_late_refs,
                )
            except (ArtifactError, HarnessJournalCorruption) as error:
                raise HarnessBoundaryPreflightCorruption(
                    "pending provider boundary refs failed closed-world validation"
                ) from error
            return
        observation = transitions[position + 1]
        observation_payload = _exact(observation, "ActionObservation", _OBSERVATION)
        observation_attempt = _integer(
            observation_payload.get("attempt_sequence"),
            "ActionObservation attempt sequence is invalid",
            minimum=1,
        )
        if (
            observation_attempt != expected_attempt
            or observation_payload.get("decision_hash") != decision.content_hash
            or observation_payload.get("idempotency_key") != expected_key
            or observation_payload.get("input_hash") != action.content_hash
            or observation_payload.get("policy_capability_hash") != capability.content_hash
            or not isinstance(observation_payload.get("recovered"), bool)
            or not isinstance(observation_payload.get("retryable"), bool)
            or not isinstance(observation_payload.get("late_outcome_hashes"), list)
            or (
                observation_payload.get("receipt_hash") is not None
                and not isinstance(observation_payload.get("receipt_hash"), str)
            )
            or (
                observation_payload.get("integrity_evidence_hash") is not None
                and not isinstance(observation_payload.get("integrity_evidence_hash"), str)
            )
        ):
            _fail("ActionObservation decision, action, or provider lineage is invalid")
        provider_hash = observation_payload.get("provider_result_hash")
        if expected_decision[0] == "deny":
            expected_output = sha256_hex(
                canonical_json_bytes({"decision_hash": decision.content_hash, "status": "not_executed"})
            )
            if (
                provider_hash is not None
                or observation_payload.get("status") != "not_executed"
                or observation_payload.get("failure_code") != decision_payload.get("reason_code")
                or observation_payload.get("output_hash") != expected_output
                or observation_payload.get("recovered") is not False
                or observation_payload.get("retryable") is not False
                or observation_payload.get("late_outcome_hashes") != []
                or observation_payload.get("receipt_hash") is not None
                or observation_payload.get("integrity_evidence_hash") is not None
            ):
                _fail("denied ActionObservation is not an inert policy result")
            denied_result_ref = (
                root / "boundaries" / "harness" / "provider-results" / expected_key / f"{expected_attempt:020d}.ref"
            )
            if denied_result_ref.exists():
                _fail("denied Harness action unexpectedly has provider evidence")
        elif isinstance(provider_hash, str):
            result = store.read(
                _hash(provider_hash, "provider result hash is invalid"),
                expected_schema_name="FixtureHarnessResult",
            )
            result_payload = _exact(result, "FixtureHarnessResult", _RESULT)
            result_attempt = _integer(
                result_payload.get("attempt_sequence"),
                "provider result attempt sequence is invalid",
                minimum=1,
            )
            schedule_hash = sha256_hex(
                canonical_json_bytes(
                    {
                        "directives": list(config.provider_fault_schedule),
                        "version": HARNESS_FAULT_SCHEDULE_VERSION,
                    }
                )
            )
            if (
                result_attempt != expected_attempt
                or result_payload.get("directive") != config.provider_fault_schedule[expected_attempt - 1]
                or result_payload.get("idempotency_key") != expected_key
                or result_payload.get("input_hash") != action.content_hash
                or result_payload.get("policy_capability_hash") != capability.content_hash
                or result_payload.get("producer") != "fixture_harness"
                or result_payload.get("schedule_hash") != schedule_hash
                or result_payload.get("schedule_version") != HARNESS_FAULT_SCHEDULE_VERSION
                or result_payload.get("recovered") != observation_payload.get("recovered")
                or result_payload.get("retryable") != observation_payload.get("retryable")
                or result_payload.get("late_outcome_hashes") != observation_payload.get("late_outcome_hashes")
                or result_payload.get("receipt_hash") != observation_payload.get("receipt_hash")
                or observation_payload.get("integrity_evidence_hash") is not None
            ):
                _fail("provider attempt/result action or schedule lineage is invalid")
            result_ref = (
                root / "boundaries" / "harness" / "provider-results" / expected_key / f"{expected_attempt:020d}.ref"
            )
            if _read_ref(result_ref) != result.content_hash:
                _fail("provider attempt ref does not match ActionObservation")
            expected_result_refs[result_ref.name] = result.content_hash
            late_ref_binding = _validate_late_outcomes(
                root,
                store,
                action,
                expected_key,
                capability.content_hash,
                result,
                config,
            )
            if late_ref_binding is not None:
                late_ref_name, late_hash = late_ref_binding
                if late_ref_name in expected_late_refs:
                    _fail("late Harness ref origin attempt is duplicated")
                expected_late_refs[late_ref_name] = late_hash
            expected_observation_status, expected_failure, expected_observation_output = (
                _provider_observation_semantics(result)
            )
            if (
                observation_payload.get("status") != expected_observation_status
                or observation_payload.get("failure_code") != expected_failure
                or observation_payload.get("output_hash") != expected_observation_output
            ):
                _fail("ActionObservation does not equal the provider result semantics")
            output_hash = result_payload.get("output_hash")
            if isinstance(output_hash, str):
                _validate_state_and_receipt(
                    root,
                    store,
                    action,
                    expected_key,
                    capability.content_hash,
                    output_hash,
                    _hash(result_payload.get("receipt_hash"), "provider receipt hash is invalid"),
                    allow_target_pending=not (
                        transitions[-1].schema_name == "DecisionOutcome"
                        and transitions[-1].payload.get("terminal") is True
                    ),
                )
        elif provider_hash is not None:
            _fail("ActionObservation provider result hash type is invalid")
        else:
            failure_code = observation_payload.get("failure_code")
            if failure_code not in {"PROVIDER_ARTIFACT_ERROR", "PROVIDER_RUNTIME_ERROR"}:
                _fail("provider-less ActionObservation has an invalid sanitized failure code")
            expected_output = sha256_hex(
                canonical_json_bytes(
                    {
                        "attempt_sequence": expected_attempt,
                        "failure_code": failure_code,
                        "idempotency_key": expected_key,
                        "status": "provider_error",
                    }
                )
            )
            if (
                observation_payload.get("status") != "provider_error"
                or observation_payload.get("output_hash") != expected_output
                or observation_payload.get("recovered") is not False
                or observation_payload.get("retryable") is not False
                or observation_payload.get("late_outcome_hashes") != []
                or observation_payload.get("receipt_hash") is not None
            ):
                _fail("sanitized provider failure observation is invalid")
            _validate_integrity_evidence(
                store,
                observation_payload.get("integrity_evidence_hash"),
                decision,
                proposal,
                expected_key,
                failure_code,
            )
            if observation_payload.get("integrity_evidence_hash") is not None:
                boundary_integrity_failed = True
        if position + 2 >= len(transitions):
            return
        outcome = transitions[position + 2]
        outcome_payload = _exact(outcome, "DecisionOutcome", _OUTCOME)
        outcome_attempt = _integer(
            outcome_payload.get("attempt_sequence"),
            "DecisionOutcome attempt sequence is invalid",
            minimum=1,
        )
        expected_outcome = _expected_outcome_semantics(
            expected_decision[0],
            decision_payload.get("reason_code"),
            observation_payload,
            expected_attempt,
            len(config.provider_fault_schedule),
        )
        if (
            outcome_attempt != expected_attempt
            or outcome_payload.get("decision_hash") != decision.content_hash
            or outcome_payload.get("idempotency_key") != expected_key
            or outcome_payload.get("input_hash") != action.content_hash
            or outcome_payload.get("observation_hash") != observation.content_hash
            or outcome_payload.get("output_hash") != observation_payload.get("output_hash")
            or outcome_payload.get("policy_capability_hash") != capability.content_hash
            or outcome_payload.get("policy_hash") != policy.content_hash
            or outcome_payload.get("proposal_hash") != proposal.content_hash
            or outcome_payload.get("receipt_hash") != observation_payload.get("receipt_hash")
            or outcome_payload.get("integrity_evidence_hash") != observation_payload.get("integrity_evidence_hash")
            or outcome_payload.get("previous_outcome_hash")
            != (previous_outcome.content_hash if previous_outcome is not None else None)
            or type(outcome_payload.get("terminal")) is not bool
            or (
                outcome_payload.get("status"),
                outcome_payload.get("reason_code"),
                outcome_payload.get("terminal"),
            )
            != expected_outcome
        ):
            _fail("DecisionOutcome action, policy, observation, or previous-outcome lineage is invalid")
        audit = store.read(
            _hash(outcome_payload.get("audit_event_hash"), "DecisionOutcome audit hash is invalid"),
            expected_schema_name="AuditEvent",
        )
        audit_payload = _exact(audit, "AuditEvent", _AUDIT)
        audit_attempt = _integer(
            audit_payload.get("attempt_sequence"),
            "AuditEvent attempt sequence is invalid",
            minimum=1,
        )
        if (
            audit_attempt != expected_attempt
            or audit_payload.get("caller") != item.caller
            or audit_payload.get("decision_hash") != decision.content_hash
            or audit_payload.get("idempotency_key") != expected_key
            or audit_payload.get("input_hash") != action.content_hash
            or audit_payload.get("observation_hash") != observation.content_hash
            or audit_payload.get("output_hash") != observation_payload.get("output_hash")
            or audit_payload.get("policy_capability_hash") != capability.content_hash
            or audit_payload.get("policy_hash") != policy.content_hash
            or audit_payload.get("policy_version") != config.policy_version
            or audit_payload.get("proposal_hash") != proposal.content_hash
            or audit_payload.get("receipt_hash") != observation_payload.get("receipt_hash")
            or audit_payload.get("integrity_evidence_hash") != observation_payload.get("integrity_evidence_hash")
            or audit_payload.get("status") != outcome_payload.get("status")
        ):
            _fail("AuditEvent proposal, policy, caller, input, or output lineage is invalid")
        if expected_outcome[2] and position + 3 < len(transitions):
            _fail("terminal Harness outcome is not the final transition")
        previous_outcome = outcome
        position += 3
        expected_attempt += 1
    if not boundary_integrity_failed:
        _validate_late_ref_set(root, expected_key, expected_late_refs)


def _validate_late_outcomes(
    root: Path,
    store: ArtifactStore,
    action: Artifact,
    idempotency_key: str,
    policy_capability_hash: str,
    result: Artifact,
    config: FixtureHarnessConfig,
) -> tuple[str, str] | None:
    result_payload = cast(dict[str, object], result.payload)
    attempt = _integer(result_payload.get("attempt_sequence"), "provider attempt is invalid", minimum=1)
    raw_hashes = result_payload.get("late_outcome_hashes")
    if not isinstance(raw_hashes, list):
        _fail("provider late outcome list is invalid")
    hashes = [_hash(value, "late outcome hash is invalid") for value in raw_hashes]
    if len(hashes) != len(set(hashes)):
        _fail("provider late outcome list contains duplicates")

    expects_late = attempt > 1 and config.provider_fault_schedule[attempt - 2] == "delayed"
    if len(hashes) != (1 if expects_late else 0):
        _fail("provider late outcome evidence does not match the fault schedule")
    if not hashes:
        return None

    origin_attempt = attempt - 1
    origin_ref = root / "boundaries" / "harness" / "provider-results" / idempotency_key / f"{origin_attempt:020d}.ref"
    origin_hash = _read_ref(origin_ref)
    try:
        origin = store.read(origin_hash, expected_schema_name="FixtureHarnessResult")
        late = store.read(hashes[0], expected_schema_name="FixtureLateHarnessOutcome")
    except ArtifactError as error:
        raise HarnessJournalCorruption("required late Harness evidence is missing or corrupt") from error
    origin_payload = _exact(origin, "FixtureHarnessResult", _RESULT)
    origin_result_attempt = _integer(
        origin_payload.get("attempt_sequence"),
        "late Harness origin result attempt is invalid",
        minimum=1,
    )
    if (
        origin_result_attempt != origin_attempt
        or origin_payload.get("directive") != "delayed"
        or origin_payload.get("idempotency_key") != idempotency_key
        or origin_payload.get("input_hash") != action.content_hash
        or origin_payload.get("policy_capability_hash") != policy_capability_hash
        or origin_payload.get("schedule_hash") != result_payload.get("schedule_hash")
        or origin_payload.get("schedule_version") != HARNESS_FAULT_SCHEDULE_VERSION
    ):
        _fail("late Harness origin attempt lineage is invalid")

    action_payload = cast(dict[str, object], action.payload)
    late_payload = _exact(late, "FixtureLateHarnessOutcome", _LATE)
    late_child_index = _integer(late_payload.get("child_index"), "late Harness child index is invalid")
    late_observed_attempt = _integer(
        late_payload.get("observed_at_attempt"),
        "late Harness observed attempt is invalid",
        minimum=1,
    )
    late_origin_attempt = _integer(
        late_payload.get("origin_attempt_sequence"),
        "late Harness origin attempt is invalid",
        minimum=1,
    )
    expected_late = {
        "action_type": action_payload.get("action_type"),
        "child_index": action_payload.get("child_index"),
        "idempotency_key": idempotency_key,
        "input_hash": action.content_hash,
        "observed_at_attempt": attempt,
        "origin_attempt_hash": origin.content_hash,
        "origin_attempt_sequence": origin_attempt,
        "origin_directive": "delayed",
        "policy_capability_hash": policy_capability_hash,
        "proposal_hash": action_payload.get("proposal_hash"),
        "schedule_hash": result_payload.get("schedule_hash"),
        "schedule_version": HARNESS_FAULT_SCHEDULE_VERSION,
        "status": "late_quarantined",
    }
    if (
        late_payload != expected_late
        or late_child_index != 0
        or late_observed_attempt != attempt
        or late_origin_attempt != origin_attempt
    ):
        _fail("late Harness evidence action, proposal, attempt, or schedule lineage is invalid")
    late_ref = root / "boundaries" / "harness" / "late-outcomes" / idempotency_key / f"{origin_attempt:020d}.ref"
    if _read_ref(late_ref) != late.content_hash:
        _fail("late Harness ref does not match the provider result")
    return late_ref.name, late.content_hash


def _validate_pending_boundary_refs(
    root: Path,
    store: ArtifactStore,
    action: Artifact,
    idempotency_key: str,
    policy_capability_hash: str,
    config: FixtureHarnessConfig,
    attempt: int,
    committed_result_refs: dict[str, str],
    committed_late_refs: dict[str, str],
) -> None:
    """Preflight all existing provider refs before the adapter is constructed."""

    result_dir = root / "boundaries" / "harness" / "provider-results" / idempotency_key
    result_entries = sorted(result_dir.iterdir()) if result_dir.exists() else []
    if any(not entry.is_file() or entry.suffix != ".ref" for entry in result_entries):
        _fail("pending provider result directory contains an unexpected entry")
    current_name = f"{attempt:020d}.ref"
    names = [entry.name for entry in result_entries]
    allowed_names = set(committed_result_refs) | {current_name}
    if len(names) != len(set(names)) or any(name not in allowed_names for name in names):
        _fail("pending provider result ref set is out of order or unexpected")
    for name, expected_hash in committed_result_refs.items():
        ref = result_dir / name
        if name not in names or _read_ref(ref) != expected_hash:
            _fail("pending provider history does not preserve committed result refs")

    expected_late = dict(committed_late_refs)
    current_ref = result_dir / current_name
    if current_ref.exists():
        result = store.read(
            _read_ref(current_ref),
            expected_schema_name="FixtureHarnessResult",
        )
        payload = _exact(result, "FixtureHarnessResult", _RESULT)
        schedule_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "directives": list(config.provider_fault_schedule),
                    "version": HARNESS_FAULT_SCHEDULE_VERSION,
                }
            )
        )
        if (
            _integer(payload.get("attempt_sequence"), "pending provider attempt is invalid", minimum=1) != attempt
            or payload.get("directive") != config.provider_fault_schedule[attempt - 1]
            or payload.get("idempotency_key") != idempotency_key
            or payload.get("input_hash") != action.content_hash
            or payload.get("policy_capability_hash") != policy_capability_hash
            or payload.get("producer") != "fixture_harness"
            or payload.get("schedule_hash") != schedule_hash
            or payload.get("schedule_version") != HARNESS_FAULT_SCHEDULE_VERSION
        ):
            _fail("pending provider result binding is invalid")
        _provider_observation_semantics(result)
        output_hash = payload.get("output_hash")
        receipt_hash = payload.get("receipt_hash")
        if isinstance(output_hash, str) and isinstance(receipt_hash, str):
            _validate_state_and_receipt(
                root,
                store,
                action,
                idempotency_key,
                policy_capability_hash,
                output_hash,
                receipt_hash,
                allow_target_pending=True,
            )
        binding = _validate_late_outcomes(
            root,
            store,
            action,
            idempotency_key,
            policy_capability_hash,
            result,
            config,
        )
        if binding is not None:
            name, content_hash = binding
            if name in expected_late and expected_late[name] != content_hash:
                _fail("pending provider late outcome conflicts with committed history")
            expected_late[name] = content_hash
    elif attempt > 1 and config.provider_fault_schedule[attempt - 2] == "delayed":
        origin_attempt = attempt - 1
        late_name = f"{origin_attempt:020d}.ref"
        late_ref = root / "boundaries" / "harness" / "late-outcomes" / idempotency_key / late_name
        if late_ref.exists():
            late_hash = _validate_pending_late_outcome(
                root,
                store,
                action,
                idempotency_key,
                policy_capability_hash,
                config,
                attempt,
                late_ref,
            )
            if late_name in expected_late and expected_late[late_name] != late_hash:
                _fail("pending late outcome conflicts with committed history")
            expected_late[late_name] = late_hash
    _validate_late_ref_set(root, idempotency_key, expected_late)


def _validate_pending_late_outcome(
    root: Path,
    store: ArtifactStore,
    action: Artifact,
    idempotency_key: str,
    policy_capability_hash: str,
    config: FixtureHarnessConfig,
    attempt: int,
    late_ref: Path,
) -> str:
    origin_attempt = attempt - 1
    origin_ref = root / "boundaries" / "harness" / "provider-results" / idempotency_key / f"{origin_attempt:020d}.ref"
    origin = store.read(
        _read_ref(origin_ref),
        expected_schema_name="FixtureHarnessResult",
    )
    origin_payload = _exact(origin, "FixtureHarnessResult", _RESULT)
    late = store.read(_read_ref(late_ref), expected_schema_name="FixtureLateHarnessOutcome")
    late_payload = _exact(late, "FixtureLateHarnessOutcome", _LATE)
    action_payload = cast(dict[str, object], action.payload)
    schedule_hash = sha256_hex(
        canonical_json_bytes(
            {
                "directives": list(config.provider_fault_schedule),
                "version": HARNESS_FAULT_SCHEDULE_VERSION,
            }
        )
    )
    expected = {
        "action_type": action_payload.get("action_type"),
        "child_index": action_payload.get("child_index"),
        "idempotency_key": idempotency_key,
        "input_hash": action.content_hash,
        "observed_at_attempt": attempt,
        "origin_attempt_hash": origin.content_hash,
        "origin_attempt_sequence": origin_attempt,
        "origin_directive": "delayed",
        "policy_capability_hash": policy_capability_hash,
        "proposal_hash": action_payload.get("proposal_hash"),
        "schedule_hash": schedule_hash,
        "schedule_version": HARNESS_FAULT_SCHEDULE_VERSION,
        "status": "late_quarantined",
    }
    if (
        origin_payload.get("attempt_sequence") != origin_attempt
        or origin_payload.get("directive") != "delayed"
        or origin_payload.get("idempotency_key") != idempotency_key
        or origin_payload.get("input_hash") != action.content_hash
        or late_payload != expected
    ):
        _fail("pending late outcome is not bound to its delayed origin")
    return late.content_hash


def _validate_late_ref_set(
    root: Path,
    idempotency_key: str,
    expected: dict[str, str],
) -> None:
    late_dir = root / "boundaries" / "harness" / "late-outcomes" / idempotency_key
    entries = sorted(late_dir.iterdir()) if late_dir.exists() else []
    if any(not entry.is_file() or entry.suffix != ".ref" for entry in entries):
        _fail("late Harness evidence directory contains an unexpected entry")
    if [entry.name for entry in entries] != sorted(expected):
        _fail("late Harness ref names do not exactly match committed evidence")
    for entry in entries:
        if _read_ref(entry) != expected[entry.name]:
            _fail("late Harness ref set does not match committed evidence")


def _validate_state_and_receipt(
    root: Path,
    store: ArtifactStore,
    action: Artifact,
    idempotency_key: str,
    policy_capability_hash: str,
    state_hash: str,
    receipt_hash: str,
    *,
    allow_target_pending: bool,
) -> None:
    state = store.read(
        _hash(state_hash, "resource state hash is invalid"),
        expected_schema_name="FixtureHarnessResourceState",
    )
    state_payload = _exact(state, "FixtureHarnessResourceState", _STATE)
    action_payload = cast(dict[str, object], action.payload)
    version = _integer(state_payload.get("version"), "resource state version is invalid", minimum=1)
    expected_version = _integer(
        action_payload.get("expected_version"),
        "typed action expected version is invalid",
    )
    value = _integer(state_payload.get("value"), "resource state value is invalid", minimum=1)
    if (
        state_payload.get("action_input_hash") != action.content_hash
        or state_payload.get("action_type") != INCREMENT_ACTION_TYPE
        or state_payload.get("amount") != 1
        or state_payload.get("idempotency_key") != idempotency_key
        or state_payload.get("policy_capability_hash") != policy_capability_hash
        or state_payload.get("resource_id") != action_payload.get("resource_id")
        or version != expected_version + 1
        or value != version
    ):
        _fail("resource state action, key, or value lineage is invalid")
    _validate_resource_chain(
        root,
        store,
        cast(str, state_payload["resource_id"]),
        state,
        allow_target_pending=allow_target_pending,
    )
    receipt_ref = root / "boundaries" / "harness" / "receipts" / f"{idempotency_key}.ref"
    if _read_ref(receipt_ref) != receipt_hash:
        _fail("provider receipt ref does not match the provider result")
    receipt = store.read(receipt_hash, expected_schema_name="FixtureHarnessReceipt")
    receipt_payload = _exact(receipt, "FixtureHarnessReceipt", _RECEIPT)
    if receipt_payload != {
        "action_input_hash": action.content_hash,
        "action_plan_hash": state_payload.get("action_plan_hash"),
        "action_type": INCREMENT_ACTION_TYPE,
        "attempt_sequence": state_payload.get("attempt_sequence"),
        "child_index": state_payload.get("child_index"),
        "harness_decision_hash": state_payload.get("harness_decision_hash"),
        "idempotency_key": idempotency_key,
        "policy_hash": state_payload.get("policy_hash"),
        "policy_capability_hash": policy_capability_hash,
        "proposal_hash": state_payload.get("proposal_hash"),
        "proposal_id": state_payload.get("proposal_id"),
        "resource_id": action_payload.get("resource_id"),
        "resource_state_hash": state.content_hash,
        "status": "applied",
    }:
        _fail("provider receipt action, key, or resource lineage is invalid")


def _read_committed_transition_chain(
    root: Path,
    store: ArtifactStore,
    proposal_id: str,
) -> list[Artifact]:
    transitions_dir = root / "harness-workflows" / proposal_id / "transitions"
    if not transitions_dir.is_dir():
        _fail("historical resource proposal transition journal is missing")
    journal = HarnessJournal(root, store, proposal_id)
    artifacts = journal.strict_transition_snapshot()
    if not artifacts:
        _fail("historical resource proposal transition ref set is invalid")
    previous_hash: str | None = None
    fields_by_schema = {
        "ActionObservation": _OBSERVATION,
        "ActionPlan": _PLAN,
        "DecisionOutcome": _OUTCOME,
        "DecisionProposal": _PROPOSAL,
        "HarnessDecision": _DECISION,
    }
    for sequence, artifact in enumerate(artifacts, start=1):
        if artifact.schema_name not in _TRANSITIONS:
            _fail("historical resource proposal transition schema is invalid")
        payload = _exact(artifact, artifact.schema_name, fields_by_schema[artifact.schema_name])
        if (
            _integer(payload.get("workflow_sequence"), "historical transition sequence is invalid", minimum=1)
            != sequence
            or payload.get("previous_transition_hash") != previous_hash
            or payload.get("proposal_id") != proposal_id
        ):
            _fail("historical resource proposal transition lineage is invalid")
        previous_hash = artifact.content_hash
    return artifacts


def validate_committed_action_authorization(
    root: Path,
    store: ArtifactStore,
    state: Artifact,
    receipt: Artifact | None,
    *,
    allow_pending: bool,
    visited_proposals: set[str],
    visited_states: set[str],
) -> bool:
    """Anchor one resource version to a committed, approved Harness workflow.

    This deliberately validates only this version's authorization and terminal
    evidence. The caller walks predecessor versions exactly once, so this
    function never recursively follows resource state.
    """

    state_payload = _exact(state, "FixtureHarnessResourceState", _STATE)
    proposal_id = state_payload.get("proposal_id")
    if not isinstance(proposal_id, str) or _SAFE_ID.fullmatch(proposal_id) is None:
        _fail("historical resource proposal id is invalid")
    proposal_hash = _hash(state_payload.get("proposal_hash"), "historical resource proposal hash is invalid")
    if state.content_hash in visited_states or proposal_hash in visited_proposals:
        _fail("resource state or proposal is reused in the version chain")
    visited_states.add(state.content_hash)
    visited_proposals.add(proposal_hash)

    workflow_dir = root / "harness-workflows" / proposal_id
    identity = store.read(
        _read_ref(workflow_dir / "identity.ref"),
        expected_schema_name="HarnessWorkflowIdentity",
    )
    identity_payload = _exact(identity, "HarnessWorkflowIdentity", _IDENTITY)
    if identity_payload.get("proposal_id") != proposal_id:
        _fail("historical resource workflow identity proposal is invalid")
    config_artifact = store.read(
        _hash(identity_payload.get("config_hash"), "historical Harness config hash is invalid"),
        expected_schema_name="FixtureHarnessConfig",
    )
    try:
        config = FixtureHarnessConfig.from_artifact_payload(config_artifact.payload)
    except (TypeError, ValueError) as error:
        raise HarnessJournalCorruption("historical Harness config is invalid") from error
    policy = store.read(
        _hash(identity_payload.get("policy_hash"), "historical Harness policy hash is invalid"),
        expected_schema_name="HarnessPolicy",
    )
    if policy.schema_version != "1.0.0" or policy.payload != fixture_policy_payload(config):
        _fail("historical Harness policy is not the committed fixture policy")
    capability = store.read(
        _hash(
            identity_payload.get("policy_capability_hash"),
            "historical Harness capability hash is invalid",
        ),
        expected_schema_name="HarnessPolicyCapability",
    )
    capability_payload = _exact(capability, "HarnessPolicyCapability", _CAPABILITY)
    if capability_payload != fixture_policy_capability_payload(config, policy.content_hash):
        _fail("historical Harness capability is not bound to its policy")
    workflow_input = store.read(
        _hash(identity_payload.get("input_hash"), "historical Harness input hash is invalid"),
        expected_schema_name="HarnessWorkflowInput",
    )
    workflow_input_payload = cast(dict[str, object], workflow_input.payload)
    if workflow_input.schema_version != "1.0.0" or set(workflow_input_payload) != {
        "action",
        "caller",
        "phase",
        "proposal_id",
    }:
        _fail("historical Harness workflow input is invalid")

    transitions = _read_committed_transition_chain(root, store, proposal_id)
    if len(transitions) < 3:
        _fail("historical resource proposal has no committed allow decision")
    proposal = transitions[0]
    plan = transitions[1]
    proposal_payload = _exact(proposal, "DecisionProposal", _PROPOSAL)
    plan_payload = _exact(plan, "ActionPlan", _PLAN)
    action = store.read(
        _hash(state_payload.get("action_input_hash"), "historical resource action hash is invalid"),
        expected_schema_name="TypedHarnessAction",
    )
    action_payload = _exact(action, "TypedHarnessAction", _ACTION)
    expected_workflow_action = {
        "action_type": action_payload.get("action_type"),
        "amount": action_payload.get("amount"),
        "expected_version": action_payload.get("expected_version"),
        "resource_id": action_payload.get("resource_id"),
    }
    if (
        workflow_input_payload.get("proposal_id") != proposal_id
        or workflow_input_payload.get("action") != expected_workflow_action
        or proposal.content_hash != proposal_hash
        or proposal_payload.get("input_hash") != workflow_input.content_hash
        or proposal_payload.get("caller") != workflow_input_payload.get("caller")
        or proposal_payload.get("phase") != workflow_input_payload.get("phase")
        or proposal_payload.get("action_type") != action_payload.get("action_type")
        or proposal_payload.get("policy_hash") != policy.content_hash
        or proposal_payload.get("policy_capability_hash") != capability.content_hash
        or plan.content_hash != state_payload.get("action_plan_hash")
        or plan_payload.get("proposal_hash") != proposal.content_hash
        or plan_payload.get("action_input_hash") != action.content_hash
        or plan_payload.get("child_index") != action_payload.get("child_index")
        or plan_payload.get("idempotency_key") != state_payload.get("idempotency_key")
        or plan_payload.get("policy_hash") != policy.content_hash
        or plan_payload.get("policy_capability_hash") != capability.content_hash
        or action_payload.get("proposal_hash") != proposal.content_hash
        or action_payload.get("policy_capability_hash") != capability.content_hash
        or state_payload.get("policy_hash") != policy.content_hash
        or state_payload.get("policy_capability_hash") != capability.content_hash
        or state_payload.get("child_index") != action_payload.get("child_index")
    ):
        _fail("historical resource proposal, plan, action, or policy binding is invalid")

    authorization_hash = _hash(
        state_payload.get("harness_decision_hash"),
        "historical resource decision hash is invalid",
    )
    authorization_matches = [item for item in transitions if item.content_hash == authorization_hash]
    if len(authorization_matches) != 1:
        _fail("historical resource allow decision is not uniquely committed")
    authorization = authorization_matches[0]
    authorization_payload = _exact(authorization, "HarnessDecision", _DECISION)
    authorization_attempt = _integer(
        authorization_payload.get("attempt_sequence"),
        "historical resource decision attempt is invalid",
        minimum=1,
    )
    expected_decision = expected_harness_decision(
        caller=workflow_input_payload.get("caller"),
        phase=workflow_input_payload.get("phase"),
        action_type=action_payload.get("action_type"),
        resource_id=action_payload.get("resource_id"),
        policy=policy.payload,
    )
    if (
        (authorization_payload.get("decision"), authorization_payload.get("reason_code")) != ("allow", "POLICY_ALLOWED")
        or expected_decision != ("allow", "POLICY_ALLOWED")
        or authorization_attempt != state_payload.get("attempt_sequence")
        or authorization_payload.get("caller") != workflow_input_payload.get("caller")
        or authorization_payload.get("idempotency_key") != state_payload.get("idempotency_key")
        or authorization_payload.get("input_hash") != action.content_hash
        or authorization_payload.get("plan_hash") != plan.content_hash
        or authorization_payload.get("proposal_hash") != proposal.content_hash
        or authorization_payload.get("policy_hash") != policy.content_hash
        or authorization_payload.get("policy_capability_hash") != capability.content_hash
        or authorization_payload.get("policy_version") != config.policy_version
    ):
        _fail("historical resource state was not produced by its exact allow decision")

    expected_receipt = {
        "action_input_hash": action.content_hash,
        "action_plan_hash": plan.content_hash,
        "action_type": INCREMENT_ACTION_TYPE,
        "attempt_sequence": authorization_attempt,
        "child_index": action_payload.get("child_index"),
        "harness_decision_hash": authorization.content_hash,
        "idempotency_key": state_payload.get("idempotency_key"),
        "policy_hash": policy.content_hash,
        "policy_capability_hash": capability.content_hash,
        "proposal_hash": proposal.content_hash,
        "proposal_id": proposal_id,
        "resource_id": state_payload.get("resource_id"),
        "resource_state_hash": state.content_hash,
        "status": "applied",
    }
    if receipt is not None:
        receipt_payload = _exact(receipt, "FixtureHarnessReceipt", _RECEIPT)
        if receipt_payload != expected_receipt:
            _fail("historical resource receipt authorization lineage is invalid")

    # A provider-locked CAS loser needs only immutable authorization proof for
    # the newest mismatching state. Its owner may be publishing receipt and
    # terminal workflow refs concurrently; no successor is created on this
    # path, so those completion artifacts are deliberately not race-required.
    if allow_pending:
        return False

    terminal = transitions[-1]
    is_terminal = terminal.schema_name == "DecisionOutcome" and terminal.payload.get("terminal") is True
    if not is_terminal:
        _fail("resource predecessor workflow has no terminal outcome")
    if receipt is None:
        _fail("terminal resource workflow is missing its immutable receipt")
    receipt_ref = root / "boundaries" / "harness" / "receipts" / f"{state_payload['idempotency_key']}.ref"
    if _read_ref(receipt_ref) != receipt.content_hash:
        _fail("historical resource receipt ref does not match its artifact")

    outcome_payload = _exact(terminal, "DecisionOutcome", _OUTCOME)
    observation_hash = _hash(
        outcome_payload.get("observation_hash"),
        "terminal resource observation hash is invalid",
    )
    observations = [item for item in transitions if item.content_hash == observation_hash]
    if len(observations) != 1:
        _fail("terminal resource observation is not uniquely committed")
    observation = observations[0]
    observation_payload = _exact(observation, "ActionObservation", _OBSERVATION)
    terminal_decision_hash = _hash(
        outcome_payload.get("decision_hash"),
        "terminal resource decision hash is invalid",
    )
    terminal_decisions = [item for item in transitions if item.content_hash == terminal_decision_hash]
    if len(terminal_decisions) != 1:
        _fail("terminal resource decision is not uniquely committed")
    terminal_decision = terminal_decisions[0]
    terminal_decision_payload = _exact(terminal_decision, "HarnessDecision", _DECISION)
    result_hash = _hash(
        observation_payload.get("provider_result_hash"),
        "terminal resource provider result hash is invalid",
    )
    result = store.read(result_hash, expected_schema_name="FixtureHarnessResult")
    result_payload = _exact(result, "FixtureHarnessResult", _RESULT)
    result_attempt = _integer(
        result_payload.get("attempt_sequence"),
        "terminal resource provider result attempt is invalid",
        minimum=1,
    )
    result_ref = (
        root
        / "boundaries"
        / "harness"
        / "provider-results"
        / cast(str, state_payload["idempotency_key"])
        / f"{result_attempt:020d}.ref"
    )
    _provider_observation_semantics(result)
    audit = store.read(
        _hash(outcome_payload.get("audit_event_hash"), "terminal resource audit hash is invalid"),
        expected_schema_name="AuditEvent",
    )
    audit_payload = _exact(audit, "AuditEvent", _AUDIT)
    if (
        outcome_payload.get("terminal") is not True
        or outcome_payload.get("status") not in {"allowed", "recovered"}
        or outcome_payload.get("reason_code") not in {"ACTION_APPLIED", "ACTION_RECOVERED"}
        or outcome_payload.get("idempotency_key") != state_payload.get("idempotency_key")
        or outcome_payload.get("input_hash") != action.content_hash
        or outcome_payload.get("output_hash") != state.content_hash
        or outcome_payload.get("receipt_hash") != receipt.content_hash
        or outcome_payload.get("proposal_hash") != proposal.content_hash
        or outcome_payload.get("policy_hash") != policy.content_hash
        or outcome_payload.get("policy_capability_hash") != capability.content_hash
        or observation_payload.get("decision_hash") != terminal_decision.content_hash
        or observation_payload.get("idempotency_key") != state_payload.get("idempotency_key")
        or observation_payload.get("input_hash") != action.content_hash
        or observation_payload.get("output_hash") != state.content_hash
        or observation_payload.get("receipt_hash") != receipt.content_hash
        or observation_payload.get("status") != "succeeded"
        or terminal_decision_payload.get("decision") != "allow"
        or terminal_decision_payload.get("idempotency_key") != state_payload.get("idempotency_key")
        or terminal_decision_payload.get("input_hash") != action.content_hash
        or terminal_decision_payload.get("plan_hash") != plan.content_hash
        or result_payload.get("idempotency_key") != state_payload.get("idempotency_key")
        or result_payload.get("input_hash") != action.content_hash
        or result_payload.get("output_hash") != state.content_hash
        or result_payload.get("receipt_hash") != receipt.content_hash
        or _read_ref(result_ref) != result.content_hash
        or audit_payload.get("decision_hash") != terminal_decision.content_hash
        or audit_payload.get("observation_hash") != observation.content_hash
        or audit_payload.get("idempotency_key") != state_payload.get("idempotency_key")
        or audit_payload.get("input_hash") != action.content_hash
        or audit_payload.get("output_hash") != state.content_hash
        or audit_payload.get("receipt_hash") != receipt.content_hash
        or audit_payload.get("proposal_hash") != proposal.content_hash
        or audit_payload.get("policy_hash") != policy.content_hash
        or audit_payload.get("policy_capability_hash") != capability.content_hash
        or audit_payload.get("status") != outcome_payload.get("status")
    ):
        _fail("terminal resource observation, outcome, receipt, or audit lineage is invalid")
    return True


def _validate_resource_chain(
    root: Path,
    store: ArtifactStore,
    resource_id: str,
    target_state: Artifact,
    *,
    allow_target_pending: bool,
) -> None:
    versions_root = root / "boundaries" / "harness" / "resources" / resource_id / "versions"
    refs = sorted(versions_root.glob("*.ref"))
    target_version = _integer(
        target_state.payload.get("version"),
        "provider output state version is invalid",
        minimum=1,
    )
    if len(refs) < target_version:
        _fail("resource state version chain is incomplete")
    previous_hash: str | None = None
    target_seen = False
    visited_proposals: set[str] = set()
    visited_states: set[str] = set()
    for version, ref in enumerate(refs, start=1):
        if ref.name != f"{version:020d}.ref":
            _fail("resource state version ref ordering is invalid")
        state = store.read(
            _read_ref(ref),
            expected_schema_name="FixtureHarnessResourceState",
        )
        payload = _exact(state, "FixtureHarnessResourceState", _STATE)
        state_version = _integer(payload.get("version"), "resource chain version is invalid", minimum=1)
        state_value = _integer(payload.get("value"), "resource chain value is invalid", minimum=1)
        action_hash = _hash(payload.get("action_input_hash"), "resource chain action hash is invalid")
        capability_hash = _hash(
            payload.get("policy_capability_hash"),
            "resource chain capability hash is invalid",
        )
        state_key = _hash(payload.get("idempotency_key"), "resource chain idempotency key is invalid")
        if (
            state_version != version
            or state_value != version
            or payload.get("action_type") != INCREMENT_ACTION_TYPE
            or _integer(payload.get("amount"), "resource chain amount is invalid", minimum=1) != 1
            or payload.get("resource_id") != resource_id
            or payload.get("previous_state_hash") != previous_hash
            or _integer(payload.get("attempt_sequence"), "resource state attempt is invalid", minimum=1) < 1
            or _integer(payload.get("child_index"), "resource state child index is invalid") != 0
        ):
            _fail("resource state version/value/predecessor lineage is invalid")
        historical_action = store.read(action_hash, expected_schema_name="TypedHarnessAction")
        historical_payload = _exact(historical_action, "TypedHarnessAction", _ACTION)
        child_index = _integer(historical_payload.get("child_index"), "historical child index is invalid")
        action_expected_version = _integer(
            historical_payload.get("expected_version"),
            "historical action expected version is invalid",
        )
        proposal_hash = _hash(
            historical_payload.get("proposal_hash"),
            "historical action proposal hash is invalid",
        )
        expected_key = sha256_hex(
            canonical_json_bytes(
                {
                    "action_input_hash": historical_action.content_hash,
                    "child_index": child_index,
                    "domain": "harness-child-idempotency/1.0.0",
                    "proposal_hash": proposal_hash,
                }
            )
        )
        if (
            child_index != 0
            or action_expected_version != version - 1
            or historical_payload.get("action_type") != INCREMENT_ACTION_TYPE
            or _integer(historical_payload.get("amount"), "historical action amount is invalid", minimum=1) != 1
            or historical_payload.get("resource_id") != resource_id
            or historical_payload.get("policy_capability_hash") != capability_hash
            or payload.get("proposal_hash") != proposal_hash
            or state_key != expected_key
        ):
            _fail("resource state is not bound to its exact typed action")
        historical_capability = store.read(
            capability_hash,
            expected_schema_name="HarnessPolicyCapability",
        )
        historical_capability_payload = _exact(
            historical_capability,
            "HarnessPolicyCapability",
            _CAPABILITY,
        )
        historical_policy_hash = _hash(
            historical_capability_payload.get("policy_hash"),
            "historical capability policy hash is invalid",
        )
        historical_policy = store.read(
            historical_policy_hash,
            expected_schema_name="HarnessPolicy",
        )
        if (
            historical_policy.schema_version != "1.0.0"
            or not valid_fixture_policy_capability(
                historical_capability_payload,
                historical_policy.payload,
                policy_content_hash=historical_policy.content_hash,
            )
            or not fixture_capability_allows(
                historical_capability_payload,
                action_type=historical_payload.get("action_type"),
                resource_id=historical_payload.get("resource_id"),
            )
        ):
            _fail("historical resource state was not authorized by its frozen capability")
        if version == target_version:
            if state.content_hash != target_state.content_hash:
                _fail("resource version ref does not match provider output")
            target_seen = True
        receipt_ref = root / "boundaries" / "harness" / "receipts" / f"{state_key}.ref"
        receipt: Artifact | None = None
        if receipt_ref.exists():
            receipt = store.read(
                _read_ref(receipt_ref),
                expected_schema_name="FixtureHarnessReceipt",
            )
        validate_committed_action_authorization(
            root,
            store,
            state,
            receipt,
            allow_pending=(
                allow_target_pending and version == target_version and state.content_hash == target_state.content_hash
            ),
            visited_proposals=visited_proposals,
            visited_states=visited_states,
        )
        previous_hash = state.content_hash
    if not target_seen:
        _fail("provider output state is absent from the resource chain")
