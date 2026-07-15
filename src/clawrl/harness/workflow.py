"""Application workflow for one production-shaped synthetic typed action."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from clawrl.artifacts import (
    Artifact,
    ArtifactError,
    ArtifactStore,
    CanonicalizationError,
    canonical_json_bytes,
    sha256_hex,
)
from clawrl.harness.fixture import FixtureHarnessCorruption, PersistentFixtureHarness
from clawrl.harness.journal import (
    HarnessHeadConflict,
    HarnessIdentityConflict,
    HarnessJournal,
    HarnessJournalCorruption,
    HarnessWorkflowTerminal,
)
from clawrl.harness.models import (
    DecisionProposal,
    FixtureHarnessConfig,
    HarnessProfileConfig,
    IncrementFixtureResource,
    ProductionHarnessConfig,
    UntrustedHarnessInput,
    expected_harness_decision,
    fixture_policy_capability_payload,
    fixture_policy_payload,
)
from clawrl.harness.validation import HarnessBoundaryPreflightCorruption, validate_harness_history

_CONTENT_HASH = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_PHASE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_MAX_UNTRUSTED_INPUT_BYTES = 64 * 1024
_MAX_UNTRUSTED_INPUT_DEPTH = 64
_MAX_UNTRUSTED_INPUT_NODES = 8_192


def _opaque_input_limit_reason(value: object) -> str | None:
    """Bound traversal before canonicalization without invoking user objects."""

    stack: list[tuple[object, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    nodes = 0
    text_units = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_UNTRUSTED_INPUT_NODES:
            return "INPUT_TOO_LARGE"
        if depth > _MAX_UNTRUSTED_INPUT_DEPTH:
            return "UNENCODABLE_VALUE"
        if current is None or type(current) in {bool, int}:
            continue
        if type(current) is str:
            text_units += len(cast(str, current))
            if text_units > _MAX_UNTRUSTED_INPUT_BYTES:
                return "INPUT_TOO_LARGE"
            continue
        if type(current) is list:
            container_id = id(current)
            if container_id in seen_containers:
                return "UNENCODABLE_VALUE"
            seen_containers.add(container_id)
            values = cast(list[object], current)
            remaining_node_budget = _MAX_UNTRUSTED_INPUT_NODES - nodes - len(stack)
            if len(values) > remaining_node_budget:
                return "INPUT_TOO_LARGE"
            stack.extend((child, depth + 1) for child in reversed(values))
            continue
        if type(current) is dict:
            container_id = id(current)
            if container_id in seen_containers:
                return "UNENCODABLE_VALUE"
            seen_containers.add(container_id)
            mapping = cast(dict[object, object], current)
            remaining_node_budget = _MAX_UNTRUSTED_INPUT_NODES - nodes - len(stack)
            if len(mapping) > remaining_node_budget:
                return "INPUT_TOO_LARGE"
            for key, child in mapping.items():
                if type(key) is not str:
                    return "UNENCODABLE_VALUE"
                text_units += len(cast(str, key))
                if text_units > _MAX_UNTRUSTED_INPUT_BYTES:
                    return "INPUT_TOO_LARGE"
                stack.append((child, depth + 1))
            continue
        return "UNENCODABLE_VALUE"
    return None


class InjectedHarnessCrash(RuntimeError):
    """Test-only host crash at a durable Harness boundary."""


class HarnessBoundaryFactory(Protocol):
    def build_fixture(
        self,
        root: Path,
        store: ArtifactStore,
        config: FixtureHarnessConfig,
        policy_capability_hash: str,
    ) -> PersistentFixtureHarness: ...


class DefaultHarnessBoundaryFactory:
    def build_fixture(
        self,
        root: Path,
        store: ArtifactStore,
        config: FixtureHarnessConfig,
        policy_capability_hash: str,
    ) -> PersistentFixtureHarness:
        return PersistentFixtureHarness(
            root,
            store,
            config.provider_fault_schedule,
            policy_capability_hash=policy_capability_hash,
        )


@dataclass(frozen=True, slots=True)
class HarnessWorkflowSnapshot:
    transitions: list[Artifact]
    readiness_report: Artifact | None = None

    @property
    def proposal(self) -> Artifact | None:
        return next((item for item in self.transitions if item.schema_name == "DecisionProposal"), None)

    @property
    def decisions(self) -> list[Artifact]:
        return [item for item in self.transitions if item.schema_name == "HarnessDecision"]

    @property
    def observations(self) -> list[Artifact]:
        return [item for item in self.transitions if item.schema_name == "ActionObservation"]

    @property
    def outcomes(self) -> list[Artifact]:
        return [item for item in self.transitions if item.schema_name == "DecisionOutcome"]

    @property
    def terminal(self) -> bool:
        return bool(
            self.readiness_report is not None
            or (
                self.transitions
                and self.transitions[-1].schema_name == "DecisionOutcome"
                and self.transitions[-1].payload.get("terminal") is True
            )
            or (self.transitions and self.transitions[-1].schema_name == "InputDisposition")
        )


class TypedHarnessWorkflow:
    """Advance exactly one durable domain transition per call."""

    def __init__(
        self,
        root: str | Path,
        config: HarnessProfileConfig,
        *,
        boundary_factory: HarnessBoundaryFactory | None = None,
    ) -> None:
        self.root = Path(root)
        self.store = ArtifactStore(self.root)
        self.config = config
        self.boundary_factory = boundary_factory or DefaultHarnessBoundaryFactory()

    @classmethod
    def resume(cls, root: str | Path, proposal_id: str, *, epoch: int) -> HarnessWorkflowSnapshot:
        store = ArtifactStore(root)
        journal = HarnessJournal(root, store, proposal_id)
        identity = journal.identity()
        config = store.read(
            cast(str, identity.payload["config_hash"]),
            expected_schema_name="FixtureHarnessConfig",
        )
        fixture_config = FixtureHarnessConfig.from_artifact_payload(config.payload)
        input_artifact = store.read(
            cast(str, identity.payload["input_hash"]),
            expected_schema_name="HarnessWorkflowInput",
        )
        payload = input_artifact.payload
        action_payload = payload.get("action")
        if not isinstance(action_payload, dict):
            raise RuntimeError("persisted Harness workflow input has no typed action")
        item = DecisionProposal(
            proposal_id=proposal_id,
            caller=cast(str, payload["caller"]),
            phase=cast(str, payload["phase"]),
            action=IncrementFixtureResource(
                resource_id=cast(str, action_payload["resource_id"]),
                expected_version=cast(int, action_payload["expected_version"]),
                amount=cast(int, action_payload["amount"]),
            ),
        )
        return cls(root, fixture_config).advance(item, epoch=epoch)

    def _integrity_failure(self, decision: Artifact, proposal: Artifact) -> Artifact:
        key = cast(str, decision.payload["idempotency_key"])
        candidates = [
            self.root / "boundaries" / "harness" / "provider-results" / key,
            self.root / "boundaries" / "harness" / "late-outcomes" / key,
        ]
        ref_digests: list[dict[str, str]] = []
        for directory in candidates:
            if not directory.exists() or not directory.is_dir():
                continue
            for entry in sorted(directory.iterdir(), key=lambda path: path.name):
                name_hash = sha256_hex(entry.name.encode("utf-8", "surrogatepass"))
                try:
                    value_hash = sha256_hex(entry.read_bytes()) if entry.is_file() else sha256_hex(b"<non-file>")
                except OSError:
                    value_hash = sha256_hex(b"<unreadable>")
                ref_digests.append({"name_hash": name_hash, "value_hash": value_hash})
        receipt_ref = self.root / "boundaries" / "harness" / "receipts" / f"{key}.ref"
        if receipt_ref.exists():
            try:
                value_hash = sha256_hex(receipt_ref.read_bytes())
            except OSError:
                value_hash = sha256_hex(b"<unreadable>")
            ref_digests.append(
                {
                    "name_hash": sha256_hex(b"receipt.ref"),
                    "value_hash": value_hash,
                }
            )
        boundary_ref_set_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "domain": "harness-boundary-ref-set/1.0.0",
                    "refs": ref_digests,
                }
            )
        )
        return self.store.put(
            "HarnessIntegrityFailure",
            "1.0.0",
            {
                "boundary_ref_set_hash": boundary_ref_set_hash,
                "candidate_transition_hash": decision.content_hash,
                "failure_code": "PROVIDER_ARTIFACT_ERROR",
                "idempotency_key": key,
                "predecessor_transition_hash": decision.payload["previous_transition_hash"],
                "proposal_hash": proposal.content_hash,
            },
        )

    def advance(
        self,
        item: DecisionProposal | UntrustedHarnessInput,
        *,
        epoch: int,
        crash_after: str | None = None,
    ) -> HarnessWorkflowSnapshot:
        if type(self.config) is ProductionHarnessConfig:
            production = cast(ProductionHarnessConfig, self.config)
            checks: list[dict[str, str]] = []

            def blocked(code: str) -> None:
                checks.append({"code": code, "status": "blocked"})

            if production.execution_profile != "production":
                blocked("INVALID_EXECUTION_PROFILE")
            for field, missing_code, invalid_code in (
                (
                    "whitelist_artifact_hash",
                    "MISSING_HARNESS_WHITELIST",
                    "INVALID_HARNESS_WHITELIST",
                ),
                (
                    "audit_config_hash",
                    "MISSING_HARNESS_AUDIT_CONFIG",
                    "INVALID_HARNESS_AUDIT_CONFIG",
                ),
                (
                    "policy_artifact_hash",
                    "MISSING_HARNESS_POLICY",
                    "INVALID_HARNESS_POLICY",
                ),
            ):
                value = getattr(production, field, None)
                if value is None:
                    blocked(missing_code)
                elif not isinstance(value, str) or _CONTENT_HASH.fullmatch(value) is None:
                    blocked(invalid_code)
            caller = getattr(production, "caller_identity", None)
            if caller is None:
                blocked("MISSING_HARNESS_CALLER")
            elif not isinstance(caller, str) or _SAFE_ID.fullmatch(caller) is None:
                blocked("INVALID_HARNESS_CALLER")
            phase_value = getattr(production, "phase", None)
            if not isinstance(phase_value, str) or _SAFE_PHASE.fullmatch(phase_value) is None:
                blocked("INVALID_HARNESS_PHASE")
                safe_phase = "INVALID"
            else:
                safe_phase = phase_value
            blocked("PRODUCTION_HARNESS_CONTRACT_UNAVAILABLE")
            report = self.store.put(
                "ReadinessReport",
                "1.0.0",
                {
                    "checks": checks,
                    "execution_profile": "production",
                    "phase": safe_phase,
                    "side_effects_permitted": False,
                    "status": "blocked",
                },
            )
            return HarnessWorkflowSnapshot([], readiness_report=report)
        if type(item) is UntrustedHarnessInput and type(self.config) is FixtureHarnessConfig:
            opaque = cast(UntrustedHarnessInput, item)
            reason_code = "NON_TYPED_INPUT_INERT"
            if type(opaque.value) is bytes:
                input_type = "bytes"
                input_bytes = cast(bytes, opaque.value)
                if len(input_bytes) > _MAX_UNTRUSTED_INPUT_BYTES:
                    reason_code = "INPUT_TOO_LARGE"
            elif type(opaque.value) is str:
                input_type = "text"
                text_value = cast(str, opaque.value)
                if len(text_value) > _MAX_UNTRUSTED_INPUT_BYTES:
                    input_bytes = b""
                    reason_code = "INPUT_TOO_LARGE"
                else:
                    try:
                        input_bytes = text_value.encode("utf-8")
                    except UnicodeError:
                        input_bytes = b""
                        reason_code = "UNENCODABLE_TEXT"
                    if len(input_bytes) > _MAX_UNTRUSTED_INPUT_BYTES:
                        reason_code = "INPUT_TOO_LARGE"
            else:
                input_type = "opaque_json"
                reason_code = _opaque_input_limit_reason(opaque.value) or reason_code
                if reason_code != "NON_TYPED_INPUT_INERT":
                    input_bytes = b""
                else:
                    try:
                        input_bytes = canonical_json_bytes(opaque.value)
                    except (CanonicalizationError, MemoryError, RecursionError, UnicodeError):
                        input_bytes = b""
                        reason_code = "UNENCODABLE_VALUE"
                if len(input_bytes) > _MAX_UNTRUSTED_INPUT_BYTES:
                    reason_code = "INPUT_TOO_LARGE"
            if reason_code != "NON_TYPED_INPUT_INERT":
                input_bytes = canonical_json_bytes(
                    {
                        "domain": "sanitized-untrusted-harness-input/1.0.0",
                        "input_type": input_type,
                        "reason_code": reason_code,
                    }
                )
            disposition = self.store.put(
                "InputDisposition",
                "1.0.0",
                {
                    "caller": opaque.caller,
                    "input_byte_size": len(input_bytes),
                    "input_hash": sha256_hex(input_bytes),
                    "input_id": opaque.input_id,
                    "input_type": input_type,
                    "phase": opaque.phase,
                    "reason_code": reason_code,
                    "status": "inert",
                },
            )
            ArtifactStore._publish(
                self.root / "harness-inputs" / opaque.input_id / "disposition.ref",
                f"{disposition.content_hash}\n".encode("ascii"),
            )
            return HarnessWorkflowSnapshot([disposition])
        if type(item) is not DecisionProposal or type(self.config) is not FixtureHarnessConfig:
            raise TypeError("minimal typed Harness workflow requires an exact DecisionProposal")
        proposal_input = cast(DecisionProposal, item)
        config = cast(FixtureHarnessConfig, self.config)
        workflow_input_payload = {
            "action": proposal_input.action.artifact_payload(),
            "caller": proposal_input.caller,
            "phase": proposal_input.phase,
            "proposal_id": proposal_input.proposal_id,
        }
        policy_payload = fixture_policy_payload(config)
        journal = HarnessJournal(self.root, self.store, proposal_input.proposal_id)
        identity_ref = journal.workflow_dir / "identity.ref"
        if identity_ref.exists():
            identity = journal.identity()
            persisted_config = self.store.read(
                cast(str, identity.payload["config_hash"]),
                expected_schema_name="FixtureHarnessConfig",
            )
            persisted_input = self.store.read(
                cast(str, identity.payload["input_hash"]),
                expected_schema_name="HarnessWorkflowInput",
            )
            persisted_policy = self.store.read(
                cast(str, identity.payload["policy_hash"]),
                expected_schema_name="HarnessPolicy",
            )
            persisted_capability = self.store.read(
                cast(str, identity.payload["policy_capability_hash"]),
                expected_schema_name="HarnessPolicyCapability",
            )
            if (
                persisted_config.payload != config.artifact_payload()
                or persisted_input.payload != workflow_input_payload
                or persisted_policy.payload != policy_payload
                or persisted_capability.payload
                != fixture_policy_capability_payload(config, persisted_policy.content_hash)
            ):
                raise HarnessIdentityConflict("proposal_id is reserved for different immutable input")
            existing_transitions = journal.transitions()
            if (
                existing_transitions
                and existing_transitions[-1].schema_name == "DecisionOutcome"
                and existing_transitions[-1].payload.get("terminal") is True
            ):
                validate_harness_history(
                    self.root,
                    self.store,
                    proposal_input,
                    config,
                    identity,
                    existing_transitions,
                )
                return HarnessWorkflowSnapshot(existing_transitions)
        config_artifact = self.store.put("FixtureHarnessConfig", "1.0.0", config.artifact_payload())
        policy = self.store.put(
            "HarnessPolicy",
            "1.0.0",
            policy_payload,
        )
        policy_capability = self.store.put(
            "HarnessPolicyCapability",
            "1.0.0",
            fixture_policy_capability_payload(config, policy.content_hash),
        )
        workflow_input = self.store.put(
            "HarnessWorkflowInput",
            "1.0.0",
            workflow_input_payload,
        )
        identity = journal.reserve_identity(
            workflow_input.content_hash,
            config_artifact.content_hash,
            policy.content_hash,
            policy_capability.content_hash,
        )
        transitions = journal.transitions()
        try:
            validate_harness_history(
                self.root,
                self.store,
                proposal_input,
                config,
                identity,
                transitions,
            )
        except HarnessBoundaryPreflightCorruption:
            if not transitions or transitions[-1].schema_name != "HarnessDecision":
                raise
            journal.claim_epoch(epoch)
            decision = transitions[-1]
            failure = self._integrity_failure(decision, transitions[0])
            attempt_sequence = cast(int, decision.payload["attempt_sequence"])
            key = cast(str, decision.payload["idempotency_key"])
            integrity_failure_code = "PROVIDER_ARTIFACT_ERROR"
            output_hash = sha256_hex(
                canonical_json_bytes(
                    {
                        "attempt_sequence": attempt_sequence,
                        "failure_code": integrity_failure_code,
                        "idempotency_key": key,
                        "status": "provider_error",
                    }
                )
            )
            journal.append(
                epoch,
                "ActionObservation",
                {
                    "attempt_sequence": attempt_sequence,
                    "decision_hash": decision.content_hash,
                    "failure_code": integrity_failure_code,
                    "idempotency_key": key,
                    "input_hash": decision.payload["input_hash"],
                    "integrity_evidence_hash": failure.content_hash,
                    "late_outcome_hashes": [],
                    "output_hash": output_hash,
                    "policy_capability_hash": policy_capability.content_hash,
                    "provider_result_hash": None,
                    "receipt_hash": None,
                    "recovered": False,
                    "retryable": False,
                    "status": "provider_error",
                },
                expected_sequence=len(transitions) + 1,
                expected_previous_hash=decision.content_hash,
            )
            committed = journal.transitions()
            validate_harness_history(
                self.root,
                self.store,
                proposal_input,
                config,
                identity,
                committed,
            )
            return HarnessWorkflowSnapshot(committed)
        if (
            transitions
            and transitions[-1].schema_name == "DecisionOutcome"
            and transitions[-1].payload.get("terminal") is True
        ):
            return HarnessWorkflowSnapshot(transitions)
        journal.claim_epoch(epoch)
        previous_hash = transitions[-1].content_hash if transitions else None
        sequence = len(transitions) + 1
        if not transitions:
            journal.append(
                epoch,
                "DecisionProposal",
                {
                    "action_type": proposal_input.action.action_type,
                    "caller": proposal_input.caller,
                    "input_hash": workflow_input.content_hash,
                    "phase": proposal_input.phase,
                    "policy_capability_hash": policy_capability.content_hash,
                    "policy_hash": policy.content_hash,
                },
                expected_sequence=sequence,
                expected_previous_hash=previous_hash,
            )
        elif len(transitions) == 1:
            proposal = transitions[0]
            action_input = self.store.put(
                "TypedHarnessAction",
                "1.0.0",
                {
                    **proposal_input.action.artifact_payload(),
                    "child_index": 0,
                    "policy_capability_hash": policy_capability.content_hash,
                    "proposal_hash": proposal.content_hash,
                },
            )
            idempotency_key = sha256_hex(
                canonical_json_bytes(
                    {
                        "action_input_hash": action_input.content_hash,
                        "child_index": 0,
                        "domain": "harness-child-idempotency/1.0.0",
                        "proposal_hash": proposal.content_hash,
                    }
                )
            )
            journal.append(
                epoch,
                "ActionPlan",
                {
                    "action_input_hash": action_input.content_hash,
                    "child_index": 0,
                    "idempotency_key": idempotency_key,
                    "policy_capability_hash": policy_capability.content_hash,
                    "policy_hash": policy.content_hash,
                    "proposal_hash": proposal.content_hash,
                },
                expected_sequence=sequence,
                expected_previous_hash=previous_hash,
            )
        elif transitions[-1].schema_name == "ActionPlan":
            proposal, plan = transitions
            decision_value, reason_code = expected_harness_decision(
                caller=proposal_input.caller,
                phase=proposal_input.phase,
                action_type=proposal_input.action.action_type,
                resource_id=proposal_input.action.resource_id,
                policy=policy.payload,
            )
            journal.append(
                epoch,
                "HarnessDecision",
                {
                    "attempt_sequence": 1,
                    "caller": proposal_input.caller,
                    "decision": decision_value,
                    "idempotency_key": plan.payload["idempotency_key"],
                    "input_hash": plan.payload["action_input_hash"],
                    "plan_hash": plan.content_hash,
                    "policy_capability_hash": policy_capability.content_hash,
                    "policy_hash": policy.content_hash,
                    "policy_version": config.policy_version,
                    "previous_outcome_hash": None,
                    "proposal_hash": proposal.content_hash,
                    "reason_code": reason_code,
                },
                expected_sequence=sequence,
                expected_previous_hash=previous_hash,
            )
        elif transitions[-1].schema_name == "DecisionOutcome":
            proposal = transitions[0]
            plan = transitions[1]
            previous_outcome = transitions[-1]
            attempt_sequence = cast(int, previous_outcome.payload["attempt_sequence"]) + 1
            if attempt_sequence > len(config.provider_fault_schedule):
                raise RuntimeError("provider retry schedule exhausted without a terminal outcome")
            decision_value, reason_code = expected_harness_decision(
                caller=proposal_input.caller,
                phase=proposal_input.phase,
                action_type=proposal_input.action.action_type,
                resource_id=proposal_input.action.resource_id,
                policy=policy.payload,
            )
            journal.append(
                epoch,
                "HarnessDecision",
                {
                    "attempt_sequence": attempt_sequence,
                    "caller": proposal_input.caller,
                    "decision": decision_value,
                    "idempotency_key": plan.payload["idempotency_key"],
                    "input_hash": plan.payload["action_input_hash"],
                    "plan_hash": plan.content_hash,
                    "policy_capability_hash": policy_capability.content_hash,
                    "policy_hash": policy.content_hash,
                    "policy_version": config.policy_version,
                    "previous_outcome_hash": previous_outcome.content_hash,
                    "proposal_hash": proposal.content_hash,
                    "reason_code": reason_code,
                },
                expected_sequence=sequence,
                expected_previous_hash=previous_hash,
            )
        elif transitions[-1].schema_name == "HarnessDecision":
            decision = transitions[-1]
            key = cast(str, decision.payload["idempotency_key"])
            input_hash = cast(str, decision.payload["input_hash"])
            attempt_sequence = cast(int, decision.payload["attempt_sequence"])
            action_input = self.store.read(input_hash, expected_schema_name="TypedHarnessAction")
            expected_decision = expected_harness_decision(
                caller=proposal_input.caller,
                phase=proposal_input.phase,
                action_type=action_input.payload.get("action_type"),
                resource_id=action_input.payload.get("resource_id"),
                policy=policy.payload,
            )
            actual_decision = (decision.payload.get("decision"), decision.payload.get("reason_code"))
            if actual_decision != expected_decision:
                raise HarnessJournalCorruption("HarnessDecision does not match the frozen policy")
            provider_result_hash: str | None
            failure_code: str | None
            if actual_decision[0] == "deny":
                provider_result_hash = None
                receipt_hash = None
                integrity_evidence_hash = None
                failure_code = cast(str, decision.payload["reason_code"])
                recovered = False
                retryable = False
                observation_status = "not_executed"
                late_outcome_hashes: list[object] = []
                output_hash = sha256_hex(
                    canonical_json_bytes(
                        {
                            "decision_hash": decision.content_hash,
                            "status": "not_executed",
                        }
                    )
                )
            elif actual_decision[0] == "allow":
                integrity_evidence_hash = None
                try:
                    boundary = self.boundary_factory.build_fixture(
                        self.root,
                        self.store,
                        config,
                        policy_capability.content_hash,
                    )
                    provider_result = boundary.query(key, input_hash, attempt_sequence)
                    executed = False
                    if provider_result is None:
                        provider_result = boundary.execute(key, action_input, attempt_sequence)
                        executed = True
                    if crash_after == "external_execution" and executed:
                        raise InjectedHarnessCrash("injected crash after external execution")
                    provider_result = self.store.read(
                        provider_result.content_hash,
                        expected_schema_name="FixtureHarnessResult",
                    )
                    result_payload = provider_result.payload
                    expected_result_fields = {
                        "attempt_sequence",
                        "directive",
                        "failure_code",
                        "idempotency_key",
                        "input_hash",
                        "invocation_performed",
                        "late_outcome_hashes",
                        "output_hash",
                        "producer",
                        "policy_capability_hash",
                        "recovered",
                        "receipt_hash",
                        "retryable",
                        "schedule_hash",
                        "schedule_version",
                        "status",
                    }
                    if (
                        set(result_payload) != expected_result_fields
                        or type(result_payload.get("attempt_sequence")) is not int
                        or result_payload.get("attempt_sequence") != attempt_sequence
                        or result_payload.get("directive") != config.provider_fault_schedule[attempt_sequence - 1]
                        or result_payload.get("idempotency_key") != key
                        or result_payload.get("input_hash") != input_hash
                        or result_payload.get("producer") != "fixture_harness"
                        or result_payload.get("policy_capability_hash") != policy_capability.content_hash
                        or result_payload.get("status") not in {"conflict", "failed", "retryable", "succeeded"}
                        or not isinstance(result_payload.get("invocation_performed"), bool)
                        or not isinstance(result_payload.get("recovered"), bool)
                        or not isinstance(result_payload.get("retryable"), bool)
                        or not isinstance(result_payload.get("late_outcome_hashes"), list)
                    ):
                        raise RuntimeError("fixture Harness provider result binding is invalid")
                    provider_result_hash = provider_result.content_hash
                    receipt_value = result_payload.get("receipt_hash")
                    receipt_hash = receipt_value if isinstance(receipt_value, str) else None
                    failure_value = result_payload.get("failure_code")
                    failure_code = failure_value if isinstance(failure_value, str) else None
                    recovered = cast(bool, result_payload["recovered"])
                    retryable = cast(bool, result_payload["retryable"])
                    late_outcome_hashes = cast(list[object], result_payload["late_outcome_hashes"])
                    provider_status = result_payload["status"]
                    if provider_status == "succeeded":
                        observation_status = "succeeded"
                    elif provider_status == "conflict":
                        observation_status = "conflict"
                    else:
                        observation_status = "provider_error"
                    provider_output_hash = result_payload.get("output_hash")
                    if isinstance(provider_output_hash, str):
                        output_hash = provider_output_hash
                    else:
                        output_hash = sha256_hex(
                            canonical_json_bytes(
                                {
                                    "attempt_sequence": attempt_sequence,
                                    "failure_code": failure_code,
                                    "provider_result_hash": provider_result.content_hash,
                                    "status": observation_status,
                                }
                            )
                        )
                except InjectedHarnessCrash:
                    raise
                except FixtureHarnessCorruption:
                    failure_code = "PROVIDER_ARTIFACT_ERROR"
                    provider_result_hash = None
                    receipt_hash = None
                    recovered = False
                    retryable = False
                    late_outcome_hashes = []
                    observation_status = "provider_error"
                    integrity_evidence_hash = self._integrity_failure(
                        decision,
                        transitions[0],
                    ).content_hash
                    output_hash = sha256_hex(
                        canonical_json_bytes(
                            {
                                "attempt_sequence": attempt_sequence,
                                "failure_code": failure_code,
                                "idempotency_key": key,
                                "status": observation_status,
                            }
                        )
                    )
                except ArtifactError:
                    failure_code = "PROVIDER_ARTIFACT_ERROR"
                    provider_result_hash = None
                    receipt_hash = None
                    integrity_evidence_hash = None
                    recovered = False
                    retryable = False
                    late_outcome_hashes = []
                    observation_status = "provider_error"
                    output_hash = sha256_hex(
                        canonical_json_bytes(
                            {
                                "attempt_sequence": attempt_sequence,
                                "failure_code": failure_code,
                                "idempotency_key": key,
                                "status": observation_status,
                            }
                        )
                    )
                except Exception:
                    failure_code = "PROVIDER_RUNTIME_ERROR"
                    provider_result_hash = None
                    receipt_hash = None
                    integrity_evidence_hash = None
                    recovered = False
                    retryable = False
                    late_outcome_hashes = []
                    observation_status = "provider_error"
                    output_hash = sha256_hex(
                        canonical_json_bytes(
                            {
                                "attempt_sequence": attempt_sequence,
                                "failure_code": failure_code,
                                "idempotency_key": key,
                                "status": observation_status,
                            }
                        )
                    )
            else:  # pragma: no cover - tuple equality above makes this defensive only
                raise HarnessJournalCorruption("HarnessDecision value is not closed-world")
            journal.append(
                epoch,
                "ActionObservation",
                {
                    "attempt_sequence": attempt_sequence,
                    "decision_hash": decision.content_hash,
                    "failure_code": failure_code,
                    "idempotency_key": key,
                    "input_hash": input_hash,
                    "integrity_evidence_hash": integrity_evidence_hash,
                    "late_outcome_hashes": late_outcome_hashes,
                    "output_hash": output_hash,
                    "policy_capability_hash": policy_capability.content_hash,
                    "provider_result_hash": provider_result_hash,
                    "recovered": recovered,
                    "receipt_hash": receipt_hash,
                    "retryable": retryable,
                    "status": observation_status,
                },
                expected_sequence=sequence,
                expected_previous_hash=previous_hash,
            )
        elif transitions[-1].schema_name == "ActionObservation":
            proposal = transitions[0]
            decision = transitions[-2]
            observation = transitions[-1]
            output_hash = cast(str, observation.payload["output_hash"])
            denied = decision.payload.get("decision") == "deny"
            attempt_sequence = cast(int, observation.payload["attempt_sequence"])
            observation_status = cast(str, observation.payload.get("status"))
            receipt_hash = cast(str | None, observation.payload.get("receipt_hash"))
            integrity_evidence_hash = cast(
                str | None,
                observation.payload.get("integrity_evidence_hash"),
            )
            if denied:
                outcome_status = "denied"
                reason_code = cast(str, decision.payload["reason_code"])
                terminal = True
            elif observation_status == "succeeded":
                outcome_status = (
                    "recovered" if attempt_sequence > 1 or observation.payload.get("recovered") is True else "allowed"
                )
                reason_code = "ACTION_RECOVERED" if outcome_status == "recovered" else "ACTION_APPLIED"
                terminal = True
            elif observation_status == "conflict":
                outcome_status = "conflict"
                reason_code = cast(str, observation.payload["failure_code"])
                terminal = True
            else:
                outcome_status = "provider_error"
                failure_value = observation.payload.get("failure_code")
                reason_code = failure_value if isinstance(failure_value, str) else "PROVIDER_FAILURE"
                terminal = not (
                    observation.payload.get("retryable") is True
                    and attempt_sequence < len(config.provider_fault_schedule)
                )
            audit = self.store.put(
                "AuditEvent",
                "1.0.0",
                {
                    "attempt_sequence": attempt_sequence,
                    "caller": proposal_input.caller,
                    "decision_hash": decision.content_hash,
                    "idempotency_key": decision.payload["idempotency_key"],
                    "input_hash": decision.payload["input_hash"],
                    "integrity_evidence_hash": integrity_evidence_hash,
                    "observation_hash": observation.content_hash,
                    "output_hash": output_hash,
                    "policy_capability_hash": policy_capability.content_hash,
                    "policy_hash": policy.content_hash,
                    "policy_version": config.policy_version,
                    "proposal_hash": proposal.content_hash,
                    "receipt_hash": receipt_hash,
                    "status": outcome_status,
                },
            )
            journal.append(
                epoch,
                "DecisionOutcome",
                {
                    "attempt_sequence": attempt_sequence,
                    "audit_event_hash": audit.content_hash,
                    "decision_hash": decision.content_hash,
                    "idempotency_key": decision.payload["idempotency_key"],
                    "input_hash": decision.payload["input_hash"],
                    "integrity_evidence_hash": integrity_evidence_hash,
                    "observation_hash": observation.content_hash,
                    "output_hash": output_hash,
                    "policy_capability_hash": policy_capability.content_hash,
                    "policy_hash": policy.content_hash,
                    "previous_outcome_hash": decision.payload["previous_outcome_hash"],
                    "proposal_hash": proposal.content_hash,
                    "reason_code": reason_code,
                    "receipt_hash": receipt_hash,
                    "status": outcome_status,
                    "terminal": terminal,
                },
                expected_sequence=sequence,
                expected_previous_hash=previous_hash,
            )
        else:
            raise RuntimeError("typed Harness workflow has an unexpected transition state")
        committed = journal.transitions()
        validate_harness_history(
            self.root,
            self.store,
            proposal_input,
            config,
            identity,
            committed,
        )
        return HarnessWorkflowSnapshot(committed)

    def run_to_terminal(
        self,
        item: DecisionProposal | UntrustedHarnessInput,
        *,
        epoch: int,
    ) -> HarnessWorkflowSnapshot:
        snapshot: HarnessWorkflowSnapshot | None = None
        for _ in range(64):
            try:
                snapshot = self.advance(item, epoch=epoch)
            except (HarnessHeadConflict, HarnessWorkflowTerminal):
                continue
            if snapshot.terminal:
                return snapshot
        raise RuntimeError("typed Harness workflow did not reach a terminal outcome")
