"""Durable deterministic stateful fixture at the Harness side-effect boundary."""

from __future__ import annotations

import fcntl
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from clawrl.artifacts import (
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    canonical_json_bytes,
    sha256_hex,
)
from clawrl.harness.journal import HarnessJournal, HarnessJournalCorruption
from clawrl.harness.models import (
    HARNESS_FAULT_SCHEDULE_VERSION,
    INCREMENT_ACTION_TYPE,
    expected_harness_decision,
    fixture_capability_allows,
    valid_fixture_policy_capability,
)
from clawrl.harness.validation import validate_committed_action_authorization

_HASH = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIRECTIVES = {
    "applied_then_timeout",
    "delayed",
    "permanent_failure",
    "success",
    "timeout",
}
_RESULT_FIELDS = {
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
_STATE_FIELDS = {
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
_RECEIPT_FIELDS = {
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
_ACTION_FIELDS = {
    "action_type",
    "amount",
    "child_index",
    "expected_version",
    "policy_capability_hash",
    "proposal_hash",
    "resource_id",
}
_PROPOSAL_FIELDS = {
    "action_type",
    "caller",
    "controller_epoch",
    "input_hash",
    "phase",
    "policy_capability_hash",
    "policy_hash",
    "previous_transition_hash",
    "proposal_id",
    "workflow_sequence",
}
_PLAN_FIELDS = {
    "action_input_hash",
    "child_index",
    "controller_epoch",
    "idempotency_key",
    "policy_capability_hash",
    "policy_hash",
    "previous_transition_hash",
    "proposal_hash",
    "proposal_id",
    "workflow_sequence",
}
_DECISION_FIELDS = {
    "attempt_sequence",
    "caller",
    "controller_epoch",
    "decision",
    "idempotency_key",
    "input_hash",
    "plan_hash",
    "policy_capability_hash",
    "policy_hash",
    "policy_version",
    "previous_outcome_hash",
    "previous_transition_hash",
    "proposal_hash",
    "proposal_id",
    "reason_code",
    "workflow_sequence",
}


class FixtureHarnessError(RuntimeError):
    """Base failure raised by the synthetic external boundary."""


class FixtureHarnessConflict(FixtureHarnessError):
    """An idempotency key is bound to another input or resource CAS lost."""


class FixtureHarnessCorruption(FixtureHarnessError):
    """Durable external fixture state cannot be verified."""


class FixtureHarnessAuthorizationDenied(FixtureHarnessError):
    """The provider independently rejected an action outside its capability."""


class PersistentFixtureHarness:
    """A process-independent provider with append-only resource versions.

    A resource version ref is the fixture's external side effect. Provider
    results and receipts are immutable coordination evidence. Querying precedes
    every invocation so a host crash cannot repeat a committed increment.
    """

    def __init__(
        self,
        root: str | Path,
        store: ArtifactStore,
        fault_schedule: tuple[str, ...] = ("success",),
        *,
        policy_capability_hash: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.store = store
        self.fault_schedule = fault_schedule
        self.policy_capability_hash = policy_capability_hash
        if policy_capability_hash is not None and _HASH.fullmatch(policy_capability_hash) is None:
            raise FixtureHarnessCorruption("fixture Harness policy capability hash is invalid")
        if (
            not isinstance(fault_schedule, tuple)
            or not fault_schedule
            or any(not isinstance(item, str) or item not in _DIRECTIVES for item in fault_schedule)
        ):
            raise FixtureHarnessCorruption("fixture Harness provider schedule is invalid")
        self.schedule_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "directives": list(fault_schedule),
                    "version": HARNESS_FAULT_SCHEDULE_VERSION,
                }
            )
        )
        self.boundary_root = self.root / "boundaries" / "harness"
        self.results_root = self.boundary_root / "provider-results"
        self.receipts_root = self.boundary_root / "receipts"
        self.resources_root = self.boundary_root / "resources"
        self.late_root = self.boundary_root / "late-outcomes"
        for directory in (self.results_root, self.receipts_root, self.resources_root, self.late_root):
            ArtifactStore.durable_mkdir(directory)
        self.lock_path = self.boundary_root / "provider.lock"
        ArtifactStore.durable_touch(self.lock_path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self.lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def query(self, idempotency_key: str, input_hash: str, attempt_sequence: int) -> Artifact | None:
        self._validate_key(idempotency_key)
        self._validate_key(input_hash)
        self._validate_attempt(attempt_sequence)
        action_input = self._load_authorized_action(input_hash)
        self._validate_idempotency_binding(idempotency_key, action_input)
        action = self._validate_action_input(action_input)
        with self._locked():
            return self._query_unlocked(
                idempotency_key,
                action_input,
                action,
                attempt_sequence,
            )

    def execute(
        self,
        idempotency_key: str,
        action_input: Artifact,
        attempt_sequence: int,
    ) -> Artifact:
        self._validate_key(idempotency_key)
        self._validate_attempt(attempt_sequence)
        action = self._validate_action_input(action_input)
        self._authorize_action(action)
        self._validate_idempotency_binding(idempotency_key, action_input)
        with self._locked():
            existing = self._query_unlocked(
                idempotency_key,
                action_input,
                action,
                attempt_sequence,
            )
            if existing is not None:
                return existing
            self._validate_attempt_predecessors(idempotency_key, action_input.content_hash, attempt_sequence)
            directive = self.fault_schedule[attempt_sequence - 1]
            late_hashes = self._late_outcomes_unlocked(
                idempotency_key,
                action_input,
                action,
                attempt_sequence,
            )
            if directive in {"success", "applied_then_timeout"}:
                state_or_conflict = self._apply_unlocked(
                    idempotency_key,
                    action_input,
                    action,
                    attempt_sequence,
                )
                if isinstance(state_or_conflict, str):
                    result = self._result(
                        idempotency_key=idempotency_key,
                        input_hash=action_input.content_hash,
                        attempt_sequence=attempt_sequence,
                        directive=directive,
                        status="conflict",
                        failure_code=state_or_conflict,
                        retryable=False,
                        output_hash=None,
                        invocation_performed=True,
                        recovered=False,
                        late_outcome_hashes=late_hashes,
                    )
                elif directive == "applied_then_timeout":
                    state, receipt = state_or_conflict
                    result = self._result(
                        idempotency_key=idempotency_key,
                        input_hash=action_input.content_hash,
                        attempt_sequence=attempt_sequence,
                        directive=directive,
                        status="retryable",
                        failure_code="PROVIDER_OUTCOME_UNKNOWN",
                        retryable=True,
                        output_hash=state.content_hash,
                        receipt_hash=receipt.content_hash,
                        invocation_performed=True,
                        recovered=False,
                        late_outcome_hashes=late_hashes,
                    )
                else:
                    state, receipt = state_or_conflict
                    result = self._result(
                        idempotency_key=idempotency_key,
                        input_hash=action_input.content_hash,
                        attempt_sequence=attempt_sequence,
                        directive=directive,
                        status="succeeded",
                        failure_code=None,
                        retryable=False,
                        output_hash=state.content_hash,
                        receipt_hash=receipt.content_hash,
                        invocation_performed=True,
                        recovered=False,
                        late_outcome_hashes=late_hashes,
                    )
            elif directive in {"timeout", "delayed"}:
                result = self._result(
                    idempotency_key=idempotency_key,
                    input_hash=action_input.content_hash,
                    attempt_sequence=attempt_sequence,
                    directive=directive,
                    status="retryable",
                    failure_code=("PROVIDER_TIMEOUT" if directive == "timeout" else "PROVIDER_RESULT_DELAYED"),
                    retryable=True,
                    output_hash=None,
                    invocation_performed=True,
                    recovered=False,
                    late_outcome_hashes=late_hashes,
                )
            elif directive == "permanent_failure":
                result = self._result(
                    idempotency_key=idempotency_key,
                    input_hash=action_input.content_hash,
                    attempt_sequence=attempt_sequence,
                    directive=directive,
                    status="failed",
                    failure_code="PROVIDER_PERMANENT_FAILURE",
                    retryable=False,
                    output_hash=None,
                    invocation_performed=True,
                    recovered=False,
                    late_outcome_hashes=late_hashes,
                )
            else:  # pragma: no cover - constructor validation makes this defensive only
                raise FixtureHarnessCorruption("unknown provider directive")
            self._publish_result(idempotency_key, attempt_sequence, result)
            return result

    def resource_state(self, resource_id: str) -> Artifact | None:
        if _SAFE_ID.fullmatch(resource_id) is None:
            raise ValueError("resource_id must be a safe identifier")
        with self._locked():
            states = self._resource_states_unlocked(resource_id)
            return states[-1] if states else None

    @staticmethod
    def external_write_count(root: str | Path) -> int:
        return len(list((Path(root) / "boundaries" / "harness" / "resources").glob("*/versions/*.ref")))

    @staticmethod
    def provider_invocation_count(root: str | Path) -> int:
        root_path = Path(root)
        store = ArtifactStore(root_path)
        count = 0
        for ref in (root_path / "boundaries" / "harness" / "provider-results").glob("*/*.ref"):
            result = store.read(PersistentFixtureHarness._read_ref(ref), expected_schema_name="FixtureHarnessResult")
            if result.payload.get("invocation_performed") is True:
                count += 1
        return count

    def _query_unlocked(
        self,
        idempotency_key: str,
        action_input: Artifact,
        action: Mapping[str, object],
        attempt_sequence: int,
    ) -> Artifact | None:
        input_hash = action_input.content_hash
        result_ref = self._result_ref(idempotency_key, attempt_sequence)
        if result_ref.exists():
            return self._read_result(result_ref, idempotency_key, input_hash, attempt_sequence)

        receipt = self._receipt_unlocked(idempotency_key, input_hash)
        if receipt is None:
            receipt = self._recover_receipt_from_state_unlocked(
                idempotency_key,
                input_hash,
                cast(str, action["resource_id"]),
            )
        if receipt is None:
            return None
        output_hash = receipt.payload.get("resource_state_hash")
        if not isinstance(output_hash, str):
            raise FixtureHarnessCorruption("fixture Harness receipt output hash is invalid")
        late_hashes = self._late_outcomes_unlocked(
            idempotency_key,
            action_input,
            action,
            attempt_sequence,
        )
        result = self._result(
            idempotency_key=idempotency_key,
            input_hash=input_hash,
            attempt_sequence=attempt_sequence,
            directive=self.fault_schedule[attempt_sequence - 1],
            status="succeeded",
            failure_code=None,
            retryable=False,
            output_hash=output_hash,
            receipt_hash=receipt.content_hash,
            invocation_performed=False,
            recovered=True,
            late_outcome_hashes=late_hashes,
        )
        self._publish_result(idempotency_key, attempt_sequence, result)
        return result

    def _apply_unlocked(
        self,
        idempotency_key: str,
        action_input: Artifact,
        action: Mapping[str, object],
        attempt_sequence: int,
    ) -> tuple[Artifact, Artifact] | str:
        resource_id = cast(str, action["resource_id"])
        expected_version = cast(int, action["expected_version"])
        states = self._resource_states_unlocked(
            resource_id,
            allow_any_last_pending=True,
        )
        current = states[-1] if states else None
        current_version = cast(int, current.payload["version"]) if current is not None else 0
        current_value = cast(int, current.payload["value"]) if current is not None else 0
        if current_version != expected_version:
            return "RESOURCE_VERSION_CONFLICT"
        # A matching CAS would create a successor, so every predecessor must
        # now have terminal observation/outcome/audit evidence.
        states = self._resource_states_unlocked(resource_id)
        current = states[-1] if states else None
        current_value = cast(int, current.payload["value"]) if current is not None else 0
        version = current_version + 1
        authorization = self._resolve_current_authorization(
            idempotency_key,
            action_input,
            action,
            attempt_sequence,
        )
        state = self.store.put(
            "FixtureHarnessResourceState",
            "1.0.0",
            {
                "action_input_hash": action_input.content_hash,
                "action_plan_hash": authorization["action_plan_hash"],
                "action_type": INCREMENT_ACTION_TYPE,
                "amount": 1,
                "attempt_sequence": attempt_sequence,
                "child_index": action["child_index"],
                "harness_decision_hash": authorization["harness_decision_hash"],
                "idempotency_key": idempotency_key,
                "policy_hash": authorization["policy_hash"],
                "previous_state_hash": current.content_hash if current is not None else None,
                "policy_capability_hash": self._required_capability_hash(),
                "proposal_hash": action["proposal_hash"],
                "proposal_id": authorization["proposal_id"],
                "resource_id": resource_id,
                "value": current_value + 1,
                "version": version,
            },
        )
        version_ref = self.resources_root / resource_id / "versions" / f"{version:020d}.ref"
        self._publish_ref(version_ref, state.content_hash)
        receipt = self._ensure_receipt_unlocked(idempotency_key, action_input.content_hash, state)
        return state, receipt

    def _resource_states_unlocked(
        self,
        resource_id: str,
        *,
        pending_key: str | None = None,
        pending_input_hash: str | None = None,
        allow_any_last_pending: bool = False,
    ) -> list[Artifact]:
        refs = sorted((self.resources_root / resource_id / "versions").glob("*.ref"))
        states: list[Artifact] = []
        previous_hash: str | None = None
        previous_value = 0
        visited_proposals: set[str] = set()
        visited_states: set[str] = set()
        for version, ref in enumerate(refs, start=1):
            if ref.name != f"{version:020d}.ref":
                raise FixtureHarnessCorruption("fixture Harness resource version sequence is invalid")
            state = self.store.read(self._read_ref(ref), expected_schema_name="FixtureHarnessResourceState")
            payload = state.payload
            action_input_hash = payload.get("action_input_hash")
            policy_capability_hash = payload.get("policy_capability_hash")
            if (
                set(payload) != _STATE_FIELDS
                or state.schema_version != "1.0.0"
                or payload.get("action_type") != INCREMENT_ACTION_TYPE
                or type(payload.get("amount")) is not int
                or payload.get("amount") != 1
                or payload.get("resource_id") != resource_id
                or type(payload.get("version")) is not int
                or payload.get("version") != version
                or type(payload.get("value")) is not int
                or payload.get("value") != previous_value + 1
                or payload.get("previous_state_hash") != previous_hash
                or not isinstance(policy_capability_hash, str)
                or _HASH.fullmatch(policy_capability_hash) is None
                or not isinstance(action_input_hash, str)
                or _HASH.fullmatch(action_input_hash) is None
                or not isinstance(payload.get("idempotency_key"), str)
                or not isinstance(payload.get("proposal_id"), str)
                or _SAFE_ID.fullmatch(cast(str, payload.get("proposal_id"))) is None
                or not isinstance(payload.get("proposal_hash"), str)
                or _HASH.fullmatch(cast(str, payload.get("proposal_hash"))) is None
                or not isinstance(payload.get("action_plan_hash"), str)
                or _HASH.fullmatch(cast(str, payload.get("action_plan_hash"))) is None
                or not isinstance(payload.get("harness_decision_hash"), str)
                or _HASH.fullmatch(cast(str, payload.get("harness_decision_hash"))) is None
                or not isinstance(payload.get("policy_hash"), str)
                or _HASH.fullmatch(cast(str, payload.get("policy_hash"))) is None
                or type(payload.get("attempt_sequence")) is not int
                or cast(int, payload.get("attempt_sequence")) < 1
                or type(payload.get("child_index")) is not int
                or payload.get("child_index") != 0
            ):
                raise FixtureHarnessCorruption("fixture Harness resource state lineage is invalid")
            historical_action = self.store.read(
                action_input_hash,
                expected_schema_name="TypedHarnessAction",
            )
            historical_payload = self._validate_action_input(historical_action)
            if (
                historical_payload.get("policy_capability_hash") != policy_capability_hash
                or historical_payload.get("resource_id") != resource_id
                or historical_payload.get("amount") != payload.get("amount")
                or historical_payload.get("expected_version") != version - 1
                or historical_payload.get("proposal_hash") != payload.get("proposal_hash")
                or historical_payload.get("child_index") != payload.get("child_index")
            ):
                raise FixtureHarnessCorruption("fixture Harness resource capability lineage is invalid")
            try:
                self._validate_action_capability(historical_payload, policy_capability_hash)
            except FixtureHarnessAuthorizationDenied as error:
                raise FixtureHarnessCorruption(
                    "fixture Harness historical resource was not capability-authorized"
                ) from error
            self._validate_idempotency_binding(cast(str, payload["idempotency_key"]), historical_action)
            receipt_ref = self.receipts_root / f"{payload['idempotency_key']}.ref"
            receipt: Artifact | None = None
            if receipt_ref.exists():
                receipt = self.store.read(
                    self._read_ref(receipt_ref),
                    expected_schema_name="FixtureHarnessReceipt",
                )
            try:
                validate_committed_action_authorization(
                    self.root,
                    self.store,
                    state,
                    receipt,
                    allow_pending=(
                        version == len(refs)
                        and (
                            allow_any_last_pending
                            or (
                                payload.get("idempotency_key") == pending_key
                                and payload.get("action_input_hash") == pending_input_hash
                            )
                        )
                    ),
                    visited_proposals=visited_proposals,
                    visited_states=visited_states,
                )
            except HarnessJournalCorruption as error:
                raise FixtureHarnessCorruption("fixture Harness resource workflow authorization is corrupt") from error
            states.append(state)
            previous_hash = state.content_hash
            previous_value += 1
        return states

    def _receipt_unlocked(self, idempotency_key: str, input_hash: str) -> Artifact | None:
        ref = self.receipts_root / f"{idempotency_key}.ref"
        if not ref.exists():
            return None
        receipt = self.store.read(self._read_ref(ref), expected_schema_name="FixtureHarnessReceipt")
        payload = receipt.payload
        state_hash = payload.get("resource_state_hash")
        resource_id = payload.get("resource_id")
        if not isinstance(state_hash, str) or not isinstance(resource_id, str):
            raise FixtureHarnessCorruption("fixture Harness receipt lineage is invalid")
        version_refs = sorted((self.resources_root / resource_id / "versions").glob("*.ref"))
        if sum(self._read_ref(candidate) == state_hash for candidate in version_refs) != 1:
            raise FixtureHarnessCorruption("fixture Harness receipt state is not a committed resource version")
        if (
            receipt.schema_version != "1.0.0"
            or set(payload) != _RECEIPT_FIELDS
            or payload.get("idempotency_key") != idempotency_key
            or payload.get("policy_capability_hash") != self._required_capability_hash()
            or payload.get("status") != "applied"
        ):
            raise FixtureHarnessCorruption("fixture Harness receipt fields are invalid")
        if payload.get("action_input_hash") != input_hash:
            raise FixtureHarnessConflict("idempotency key is already bound to a different action input")
        state = self.store.read(state_hash, expected_schema_name="FixtureHarnessResourceState")
        committed_states = self._resource_states_unlocked(
            resource_id,
            pending_key=idempotency_key,
            pending_input_hash=input_hash,
        )
        committed_matches = [
            candidate for candidate in committed_states if candidate.content_hash == state.content_hash
        ]
        if len(committed_matches) != 1:
            raise FixtureHarnessCorruption("fixture Harness receipt state is not a committed resource version")
        if (
            state.payload.get("idempotency_key") != idempotency_key
            or state.payload.get("action_input_hash") != input_hash
            or state.payload.get("policy_capability_hash") != self._required_capability_hash()
            or state.payload.get("resource_id") != resource_id
        ):
            raise FixtureHarnessCorruption("fixture Harness receipt does not bind its resource state")
        return receipt

    def _recover_receipt_from_state_unlocked(
        self,
        idempotency_key: str,
        input_hash: str,
        resource_id: str,
    ) -> Artifact | None:
        """Recover only this action's crash-partial state.

        Unrelated newest versions are intentionally left to the CAS path. A
        losing action must not race-require another workflow's receipt/outcome
        merely to discover that its expected version is stale.
        """

        refs = sorted((self.resources_root / resource_id / "versions").glob("*.ref"))
        matching_hashes: list[str] = []
        for ref in refs:
            state = self.store.read(
                self._read_ref(ref),
                expected_schema_name="FixtureHarnessResourceState",
            )
            if state.payload.get("idempotency_key") == idempotency_key:
                matching_hashes.append(state.content_hash)
                if state.payload.get("action_input_hash") != input_hash:
                    raise FixtureHarnessConflict("idempotency key resource state has a different action input")
        if len(matching_hashes) > 1:
            raise FixtureHarnessCorruption("idempotency key appears in multiple resource states")
        match: Artifact | None = None
        if matching_hashes:
            states = self._resource_states_unlocked(
                resource_id,
                pending_key=idempotency_key,
                pending_input_hash=input_hash,
            )
            match = next(
                (state for state in states if state.content_hash == matching_hashes[0]),
                None,
            )
            if match is None:
                raise FixtureHarnessCorruption("idempotency state is absent from the verified resource chain")
        if match is None:
            return None
        return self._ensure_receipt_unlocked(idempotency_key, input_hash, match)

    def _ensure_receipt_unlocked(self, idempotency_key: str, input_hash: str, state: Artifact) -> Artifact:
        existing = self._receipt_unlocked(idempotency_key, input_hash)
        if existing is not None:
            if existing.payload.get("resource_state_hash") != state.content_hash:
                raise FixtureHarnessCorruption("fixture Harness receipt conflicts with resource state")
            return existing
        resource_id = state.payload.get("resource_id")
        if not isinstance(resource_id, str):
            raise FixtureHarnessCorruption("fixture Harness resource state has no resource identity")
        receipt = self.store.put(
            "FixtureHarnessReceipt",
            "1.0.0",
            {
                "action_input_hash": input_hash,
                "action_plan_hash": state.payload["action_plan_hash"],
                "action_type": state.payload["action_type"],
                "attempt_sequence": state.payload["attempt_sequence"],
                "child_index": state.payload["child_index"],
                "harness_decision_hash": state.payload["harness_decision_hash"],
                "idempotency_key": idempotency_key,
                "policy_hash": state.payload["policy_hash"],
                "policy_capability_hash": self._required_capability_hash(),
                "proposal_hash": state.payload["proposal_hash"],
                "proposal_id": state.payload["proposal_id"],
                "resource_id": resource_id,
                "resource_state_hash": state.content_hash,
                "status": "applied",
            },
        )
        self._publish_ref(self.receipts_root / f"{idempotency_key}.ref", receipt.content_hash)
        return receipt

    def _late_outcomes_unlocked(
        self,
        idempotency_key: str,
        action_input: Artifact,
        action: Mapping[str, object],
        attempt_sequence: int,
    ) -> list[str]:
        if attempt_sequence <= 1:
            return []
        previous = self._read_result(
            self._result_ref(idempotency_key, attempt_sequence - 1),
            idempotency_key,
            action_input.content_hash,
            attempt_sequence - 1,
        )
        if previous.payload.get("directive") != "delayed":
            return []
        late = self.store.put(
            "FixtureLateHarnessOutcome",
            "1.0.0",
            {
                "action_type": action["action_type"],
                "child_index": action["child_index"],
                "idempotency_key": idempotency_key,
                "input_hash": action_input.content_hash,
                "observed_at_attempt": attempt_sequence,
                "origin_attempt_hash": previous.content_hash,
                "origin_attempt_sequence": attempt_sequence - 1,
                "origin_directive": "delayed",
                "policy_capability_hash": self._required_capability_hash(),
                "proposal_hash": action["proposal_hash"],
                "schedule_hash": self.schedule_hash,
                "schedule_version": HARNESS_FAULT_SCHEDULE_VERSION,
                "status": "late_quarantined",
            },
        )
        self._publish_ref(
            self.late_root / idempotency_key / f"{attempt_sequence - 1:020d}.ref",
            late.content_hash,
        )
        return [late.content_hash]

    def _validate_attempt_predecessors(self, key: str, input_hash: str, attempt: int) -> None:
        refs = sorted((self.results_root / key).glob("*.ref"))
        if len(refs) != attempt - 1:
            raise FixtureHarnessCorruption("fixture Harness provider attempt sequence has a gap")
        for sequence, ref in enumerate(refs, start=1):
            if ref.name != f"{sequence:020d}.ref":
                raise FixtureHarnessCorruption("fixture Harness provider attempt ref is invalid")
            self._read_result(ref, key, input_hash, sequence)

    def _read_result(self, ref: Path, key: str, input_hash: str, attempt: int) -> Artifact:
        if not ref.exists():
            raise FixtureHarnessCorruption("fixture Harness provider result ref is missing")
        result = self.store.read(self._read_ref(ref), expected_schema_name="FixtureHarnessResult")
        payload = result.payload
        if (
            set(payload) != _RESULT_FIELDS
            or result.schema_version != "1.0.0"
            or type(payload.get("attempt_sequence")) is not int
            or payload.get("attempt_sequence") != attempt
            or payload.get("directive") != self.fault_schedule[attempt - 1]
            or payload.get("idempotency_key") != key
            or payload.get("input_hash") != input_hash
            or payload.get("producer") != "fixture_harness"
            or payload.get("policy_capability_hash") != self._required_capability_hash()
            or payload.get("schedule_hash") != self.schedule_hash
            or payload.get("schedule_version") != HARNESS_FAULT_SCHEDULE_VERSION
            or payload.get("status") not in {"conflict", "failed", "retryable", "succeeded"}
            or not isinstance(payload.get("invocation_performed"), bool)
            or not isinstance(payload.get("recovered"), bool)
            or not isinstance(payload.get("retryable"), bool)
            or not isinstance(payload.get("late_outcome_hashes"), list)
        ):
            raise FixtureHarnessCorruption("fixture Harness provider result binding is invalid")
        return result

    def _result(
        self,
        *,
        idempotency_key: str,
        input_hash: str,
        attempt_sequence: int,
        directive: str,
        status: str,
        failure_code: str | None,
        retryable: bool,
        output_hash: str | None,
        invocation_performed: bool,
        recovered: bool,
        late_outcome_hashes: list[str],
        receipt_hash: str | None = None,
    ) -> Artifact:
        return self.store.put(
            "FixtureHarnessResult",
            "1.0.0",
            {
                "attempt_sequence": attempt_sequence,
                "directive": directive,
                "failure_code": failure_code,
                "idempotency_key": idempotency_key,
                "input_hash": input_hash,
                "invocation_performed": invocation_performed,
                "late_outcome_hashes": late_outcome_hashes,
                "output_hash": output_hash,
                "policy_capability_hash": self._required_capability_hash(),
                "producer": "fixture_harness",
                "recovered": recovered,
                "receipt_hash": receipt_hash,
                "retryable": retryable,
                "schedule_hash": self.schedule_hash,
                "schedule_version": HARNESS_FAULT_SCHEDULE_VERSION,
                "status": status,
            },
        )

    def _publish_result(self, key: str, attempt: int, result: Artifact) -> None:
        self._publish_ref(self._result_ref(key, attempt), result.content_hash)

    def _result_ref(self, key: str, attempt: int) -> Path:
        return self.results_root / key / f"{attempt:020d}.ref"

    def _load_authorized_action(self, input_hash: str) -> Artifact:
        action = self.store.read(input_hash, expected_schema_name="TypedHarnessAction")
        payload = self._validate_action_input(action)
        self._authorize_action(payload)
        return action

    def _authorize_action(self, action: Mapping[str, object]) -> None:
        capability_hash = self._required_capability_hash()
        self._validate_action_capability(action, capability_hash)

    def _resolve_current_authorization(
        self,
        idempotency_key: str,
        action_input: Artifact,
        action: Mapping[str, object],
        attempt_sequence: int,
    ) -> dict[str, str]:
        """Resolve the exact committed allow decision before publishing a state."""

        proposal_hash = cast(str, action.get("proposal_hash"))
        try:
            proposal = self.store.read(proposal_hash, expected_schema_name="DecisionProposal")
        except ArtifactCorruption as error:
            raise FixtureHarnessCorruption("authorized proposal artifact is missing or corrupt") from error
        proposal_payload = proposal.payload
        proposal_id = proposal_payload.get("proposal_id")
        if (
            proposal.schema_version != "1.0.0"
            or set(proposal_payload) != _PROPOSAL_FIELDS
            or proposal.content_hash != proposal_hash
            or not isinstance(proposal_id, str)
            or _SAFE_ID.fullmatch(proposal_id) is None
            or proposal_payload.get("action_type") != action.get("action_type")
            or proposal_payload.get("policy_capability_hash") != self._required_capability_hash()
        ):
            raise FixtureHarnessCorruption("resource action is not bound to an exact committed proposal")
        workflow_dir = self.root / "harness-workflows" / proposal_id
        if not (workflow_dir / "identity.ref").is_file() or not (workflow_dir / "transitions").is_dir():
            raise FixtureHarnessCorruption("authorized proposal journal is missing")
        try:
            journal = HarnessJournal(self.root, self.store, proposal_id)
            journal.verify()
            identity = journal.identity()
            transitions = journal.transitions()
        except (ArtifactCorruption, HarnessJournalCorruption) as error:
            raise FixtureHarnessCorruption("authorized proposal journal is corrupt") from error
        if len(transitions) < 3 or transitions[0].content_hash != proposal.content_hash:
            raise FixtureHarnessCorruption("authorized proposal ref is not committed at journal head")
        plan = transitions[1]
        plan_payload = plan.payload
        decision = transitions[-1]
        decision_payload = decision.payload
        policy_hash = proposal_payload.get("policy_hash")
        if not isinstance(policy_hash, str) or _HASH.fullmatch(policy_hash) is None:
            raise FixtureHarnessCorruption("authorized proposal policy hash is invalid")
        try:
            policy = self.store.read(policy_hash, expected_schema_name="HarnessPolicy")
        except ArtifactCorruption as error:
            raise FixtureHarnessCorruption("authorized proposal policy is missing or corrupt") from error
        expected_decision = expected_harness_decision(
            caller=proposal_payload.get("caller"),
            phase=proposal_payload.get("phase"),
            action_type=action.get("action_type"),
            resource_id=action.get("resource_id"),
            policy=policy.payload,
        )
        if (
            plan.schema_name != "ActionPlan"
            or plan.schema_version != "1.0.0"
            or set(plan_payload) != _PLAN_FIELDS
            or plan_payload.get("proposal_hash") != proposal.content_hash
            or plan_payload.get("action_input_hash") != action_input.content_hash
            or plan_payload.get("child_index") != action.get("child_index")
            or plan_payload.get("idempotency_key") != idempotency_key
            or plan_payload.get("policy_hash") != policy_hash
            or plan_payload.get("policy_capability_hash") != self._required_capability_hash()
            or decision.schema_name != "HarnessDecision"
            or decision.schema_version != "1.0.0"
            or set(decision_payload) != _DECISION_FIELDS
            or decision_payload.get("attempt_sequence") != attempt_sequence
            or decision_payload.get("decision") != "allow"
            or decision_payload.get("reason_code") != "POLICY_ALLOWED"
            or expected_decision != ("allow", "POLICY_ALLOWED")
            or decision_payload.get("idempotency_key") != idempotency_key
            or decision_payload.get("input_hash") != action_input.content_hash
            or decision_payload.get("plan_hash") != plan.content_hash
            or decision_payload.get("proposal_hash") != proposal.content_hash
            or decision_payload.get("policy_hash") != policy_hash
            or decision_payload.get("policy_capability_hash") != self._required_capability_hash()
            or identity.payload.get("policy_hash") != policy_hash
            or identity.payload.get("policy_capability_hash") != self._required_capability_hash()
        ):
            raise FixtureHarnessCorruption("provider invocation is not an exact committed allow decision")
        return {
            "action_plan_hash": plan.content_hash,
            "harness_decision_hash": decision.content_hash,
            "policy_hash": policy_hash,
            "proposal_id": proposal_id,
        }

    def _validate_action_capability(
        self,
        action: Mapping[str, object],
        capability_hash: str,
    ) -> None:
        if action.get("policy_capability_hash") != capability_hash:
            raise FixtureHarnessAuthorizationDenied("action is not bound to the provider policy capability")
        capability = self.store.read(
            capability_hash,
            expected_schema_name="HarnessPolicyCapability",
        )
        payload = capability.payload
        policy_hash = payload.get("policy_hash")
        if capability.schema_version != "1.0.0" or not isinstance(policy_hash, str):
            raise FixtureHarnessCorruption("fixture Harness policy capability is invalid")
        policy = self.store.read(policy_hash, expected_schema_name="HarnessPolicy")
        if policy.schema_version != "1.0.0" or not valid_fixture_policy_capability(
            payload,
            policy.payload,
            policy_content_hash=policy.content_hash,
        ):
            raise FixtureHarnessCorruption("fixture Harness capability does not match its frozen policy")
        if not fixture_capability_allows(
            payload,
            action_type=action.get("action_type"),
            resource_id=action.get("resource_id"),
        ):
            raise FixtureHarnessAuthorizationDenied("action or resource is not allowed by the provider capability")

    def _required_capability_hash(self) -> str:
        if self.policy_capability_hash is None:
            raise FixtureHarnessAuthorizationDenied("provider invocation requires a policy capability")
        return self.policy_capability_hash

    @staticmethod
    def _validate_idempotency_binding(idempotency_key: str, action: Artifact) -> None:
        payload = action.payload
        expected = sha256_hex(
            canonical_json_bytes(
                {
                    "action_input_hash": action.content_hash,
                    "child_index": payload.get("child_index"),
                    "domain": "harness-child-idempotency/1.0.0",
                    "proposal_hash": payload.get("proposal_hash"),
                }
            )
        )
        if idempotency_key != expected:
            raise FixtureHarnessAuthorizationDenied("idempotency key is not bound to the authorized action")

    @staticmethod
    def _validate_action_input(action: Artifact) -> Mapping[str, object]:
        payload = action.payload
        if (
            action.schema_name != "TypedHarnessAction"
            or action.schema_version != "1.0.0"
            or set(payload) != _ACTION_FIELDS
            or payload.get("action_type") != INCREMENT_ACTION_TYPE
            or type(payload.get("amount")) is not int
            or payload.get("amount") != 1
            or type(payload.get("child_index")) is not int
            or payload.get("child_index") != 0
            or type(payload.get("expected_version")) is not int
            or cast(int, payload.get("expected_version")) < 0
            or not isinstance(payload.get("policy_capability_hash"), str)
            or _HASH.fullmatch(cast(str, payload.get("policy_capability_hash"))) is None
            or not isinstance(payload.get("proposal_hash"), str)
            or _HASH.fullmatch(cast(str, payload.get("proposal_hash"))) is None
            or not isinstance(payload.get("resource_id"), str)
            or _SAFE_ID.fullmatch(cast(str, payload.get("resource_id"))) is None
        ):
            raise FixtureHarnessCorruption("fixture Harness action input is invalid")
        return cast(Mapping[str, object], payload)

    def _validate_attempt(self, attempt: int) -> None:
        if not isinstance(attempt, int) or isinstance(attempt, bool) or not 1 <= attempt <= len(self.fault_schedule):
            raise FixtureHarnessCorruption("fixture Harness provider attempt is invalid or exhausted")

    @staticmethod
    def _validate_key(value: str) -> None:
        if _HASH.fullmatch(value) is None:
            raise ValueError("fixture Harness keys must be SHA-256 hashes")

    @staticmethod
    def _read_ref(ref: Path) -> str:
        try:
            raw = ref.read_bytes()
        except OSError as error:
            raise FixtureHarnessCorruption("fixture Harness ref cannot be read") from error
        if len(raw) != 65 or raw[-1:] != b"\n":
            raise FixtureHarnessCorruption("fixture Harness ref bytes are malformed")
        try:
            value = raw[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise FixtureHarnessCorruption("fixture Harness ref is not ASCII") from error
        if _HASH.fullmatch(value) is None:
            raise FixtureHarnessCorruption("fixture Harness ref has no SHA-256 hash")
        return value

    @staticmethod
    def _publish_ref(ref: Path, content_hash: str) -> None:
        if _HASH.fullmatch(content_hash) is None:
            raise FixtureHarnessCorruption("fixture Harness publication hash is invalid")
        try:
            ArtifactStore._publish(ref, f"{content_hash}\n".encode("ascii"))
        except Exception as error:
            if isinstance(error, ArtifactCorruption):
                raise
            raise FixtureHarnessCorruption("fixture Harness ref publication failed or conflicted") from error
