from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from clawrl.artifacts import (
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    JsonValue,
    UnknownSchemaMajor,
    canonical_json_bytes,
    sha256_hex,
)
from clawrl.harness import (
    DecisionProposal,
    FixtureHarnessAuthorizationDenied,
    FixtureHarnessConfig,
    FixtureHarnessCorruption,
    HarnessJournalCorruption,
    HarnessWorkflowSnapshot,
    IncrementFixtureResource,
    InjectedHarnessCrash,
    PersistentFixtureHarness,
    TypedHarnessWorkflow,
)
from clawrl.harness.models import fixture_policy_capability_payload, fixture_policy_payload


def completed(root: Path, proposal_id: str) -> tuple[DecisionProposal, HarnessWorkflowSnapshot]:
    item = DecisionProposal(
        proposal_id=proposal_id,
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 0),
    )
    snapshot = TypedHarnessWorkflow(root, FixtureHarnessConfig()).run_to_terminal(item, epoch=1)
    return item, snapshot


def transition_ref(root: Path, proposal_id: str, sequence: int) -> Path:
    return root / "harness-workflows" / proposal_id / "transitions" / f"{sequence:020d}.ref"


def substitute_ref(ref: Path, artifact: Artifact) -> None:
    ref.write_bytes(f"{artifact.content_hash}\n".encode("ascii"))


def test_fabricated_resource_predecessor_without_committed_workflow_is_rejected_before_v2_write(
    tmp_path: Path,
) -> None:
    """A content-consistent boundary row is not proof that Harness approved it."""

    store = ArtifactStore(tmp_path)
    config = FixtureHarnessConfig()
    policy = store.put("HarnessPolicy", "1.0.0", fixture_policy_payload(config))
    capability = store.put(
        "HarnessPolicyCapability",
        "1.0.0",
        fixture_policy_capability_payload(config, policy.content_hash),
    )
    forged_proposal_hash = "a" * 64
    action = store.put(
        "TypedHarnessAction",
        "1.0.0",
        {
            "action_type": "fixture.resource.increment.v1",
            "amount": 1,
            "child_index": 0,
            "expected_version": 0,
            "policy_capability_hash": capability.content_hash,
            "proposal_hash": forged_proposal_hash,
            "resource_id": "fixture-counter",
        },
    )
    key = sha256_hex(
        canonical_json_bytes(
            {
                "action_input_hash": action.content_hash,
                "child_index": 0,
                "domain": "harness-child-idempotency/1.0.0",
                "proposal_hash": forged_proposal_hash,
            }
        )
    )
    state = store.put(
        "FixtureHarnessResourceState",
        "1.0.0",
        {
            "action_input_hash": action.content_hash,
            "action_plan_hash": "b" * 64,
            "action_type": "fixture.resource.increment.v1",
            "amount": 1,
            "attempt_sequence": 1,
            "child_index": 0,
            "harness_decision_hash": "c" * 64,
            "idempotency_key": key,
            "policy_hash": policy.content_hash,
            "policy_capability_hash": capability.content_hash,
            "previous_state_hash": None,
            "proposal_hash": forged_proposal_hash,
            "proposal_id": "forged-proposal-with-no-journal",
            "resource_id": "fixture-counter",
            "value": 1,
            "version": 1,
        },
    )
    state_ref = (
        tmp_path / "boundaries" / "harness" / "resources" / "fixture-counter" / "versions" / "00000000000000000001.ref"
    )
    state_ref.parent.mkdir(parents=True, exist_ok=True)
    substitute_ref(state_ref, state)
    receipt = store.put(
        "FixtureHarnessReceipt",
        "1.0.0",
        {
            "action_input_hash": action.content_hash,
            "action_plan_hash": "b" * 64,
            "action_type": "fixture.resource.increment.v1",
            "attempt_sequence": 1,
            "child_index": 0,
            "harness_decision_hash": "c" * 64,
            "idempotency_key": key,
            "policy_hash": policy.content_hash,
            "policy_capability_hash": capability.content_hash,
            "proposal_hash": forged_proposal_hash,
            "proposal_id": "forged-proposal-with-no-journal",
            "resource_id": "fixture-counter",
            "resource_state_hash": state.content_hash,
            "status": "applied",
        },
    )
    receipt_ref = tmp_path / "boundaries" / "harness" / "receipts" / f"{key}.ref"
    receipt_ref.parent.mkdir(parents=True, exist_ok=True)
    substitute_ref(receipt_ref, receipt)

    real_v2 = DecisionProposal(
        proposal_id="proposal-real-v2-after-forgery",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 1),
    )
    failed = TypedHarnessWorkflow(tmp_path, config).run_to_terminal(real_v2, epoch=1)

    assert failed.outcomes[-1].payload["status"] == "provider_error"
    assert failed.outcomes[-1].payload["reason_code"] == "PROVIDER_ARTIFACT_ERROR"
    evidence_hash = failed.observations[-1].payload["integrity_evidence_hash"]
    assert isinstance(evidence_hash, str)
    ArtifactStore(tmp_path).read(evidence_hash, expected_schema_name="HarnessIntegrityFailure")
    assert PersistentFixtureHarness.external_write_count(tmp_path) == 1
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 0


def test_missing_predecessor_receipt_blocks_next_version_before_provider_write(tmp_path: Path) -> None:
    first_item, first = completed(tmp_path, "proposal-legitimate-v1-missing-receipt")
    del first_item
    first_key = str(first.observations[-1].payload["idempotency_key"])
    (tmp_path / "boundaries" / "harness" / "receipts" / f"{first_key}.ref").unlink()
    second = DecisionProposal(
        proposal_id="proposal-v2-after-missing-v1-receipt",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 1),
    )

    failed = TypedHarnessWorkflow(tmp_path, FixtureHarnessConfig()).run_to_terminal(second, epoch=1)

    assert failed.outcomes[-1].payload["status"] == "provider_error"
    assert failed.outcomes[-1].payload["reason_code"] == "PROVIDER_ARTIFACT_ERROR"
    replay = TypedHarnessWorkflow.resume(tmp_path, second.proposal_id, epoch=2)
    assert replay.outcomes[-1].content_hash == failed.outcomes[-1].content_hash
    assert PersistentFixtureHarness.external_write_count(tmp_path) == 1
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 1


def test_pending_decision_rejects_unexpected_late_ref_before_boundary_construction_or_write(
    tmp_path: Path,
) -> None:
    """Committed history is preflighted before the side-effect adapter exists."""

    item = DecisionProposal(
        proposal_id="proposal-pending-extra-late",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 0),
    )
    config = FixtureHarnessConfig()
    workflow = TypedHarnessWorkflow(tmp_path, config)
    snapshot = workflow.advance(item, epoch=1)
    snapshot = workflow.advance(item, epoch=1)
    snapshot = workflow.advance(item, epoch=1)
    assert snapshot.transitions[-1].schema_name == "HarnessDecision"
    decision = snapshot.decisions[-1]
    key = str(decision.payload["idempotency_key"])
    input_hash = str(decision.payload["input_hash"])
    store = ArtifactStore(tmp_path)
    action = store.read(input_hash, expected_schema_name="TypedHarnessAction")
    schedule_hash = sha256_hex(
        canonical_json_bytes(
            {
                "directives": list(config.provider_fault_schedule),
                "version": "fixture-harness-fault-schedule/1.0.0",
            }
        )
    )
    unexpected = store.put(
        "FixtureLateHarnessOutcome",
        "1.0.0",
        {
            "action_type": action.payload["action_type"],
            "child_index": action.payload["child_index"],
            "idempotency_key": key,
            "input_hash": action.content_hash,
            "observed_at_attempt": 1,
            "origin_attempt_hash": "b" * 64,
            "origin_attempt_sequence": 1,
            "origin_directive": "delayed",
            "policy_capability_hash": action.payload["policy_capability_hash"],
            "proposal_hash": action.payload["proposal_hash"],
            "schedule_hash": schedule_hash,
            "schedule_version": "fixture-harness-fault-schedule/1.0.0",
            "status": "late_quarantined",
        },
    )
    late_ref = tmp_path / "boundaries" / "harness" / "late-outcomes" / key / "00000000000000000001.ref"
    late_ref.parent.mkdir(parents=True, exist_ok=True)
    substitute_ref(late_ref, unexpected)

    class CountingFactory:
        def __init__(self) -> None:
            self.calls = 0

        def build_fixture(
            self,
            root: Path,
            store: ArtifactStore,
            config: FixtureHarnessConfig,
            policy_capability_hash: str,
        ) -> PersistentFixtureHarness:
            self.calls += 1
            return PersistentFixtureHarness(
                root,
                store,
                config.provider_fault_schedule,
                policy_capability_hash=policy_capability_hash,
            )

    factory = CountingFactory()
    failed = TypedHarnessWorkflow(tmp_path, config, boundary_factory=factory).advance(item, epoch=1)

    assert factory.calls == 0
    assert failed.transitions[-1].schema_name == "ActionObservation"
    assert failed.observations[-1].payload["failure_code"] == "PROVIDER_ARTIFACT_ERROR"
    assert isinstance(failed.observations[-1].payload["integrity_evidence_hash"], str)
    terminal = TypedHarnessWorkflow(tmp_path, config).run_to_terminal(item, epoch=1)
    assert terminal.outcomes[-1].payload["status"] == "provider_error"
    replay = TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=2)
    assert replay.outcomes[-1].content_hash == terminal.outcomes[-1].content_hash
    assert PersistentFixtureHarness.external_write_count(tmp_path) == 0
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 0


def test_pending_success_result_with_missing_state_and_receipt_fails_auditable_before_factory(
    tmp_path: Path,
) -> None:
    item = DecisionProposal(
        proposal_id="proposal-pending-missing-state-receipt",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 0),
    )
    config = FixtureHarnessConfig()
    workflow = TypedHarnessWorkflow(tmp_path, config)
    snapshot = workflow.advance(item, epoch=1)
    snapshot = workflow.advance(item, epoch=1)
    snapshot = workflow.advance(item, epoch=1)
    decision = snapshot.decisions[-1]
    key = str(decision.payload["idempotency_key"])
    input_hash = str(decision.payload["input_hash"])
    capability_hash = str(decision.payload["policy_capability_hash"])
    store = ArtifactStore(tmp_path)
    result = store.put(
        "FixtureHarnessResult",
        "1.0.0",
        {
            "attempt_sequence": 1,
            "directive": "success",
            "failure_code": None,
            "idempotency_key": key,
            "input_hash": input_hash,
            "invocation_performed": True,
            "late_outcome_hashes": [],
            "output_hash": "a" * 64,
            "policy_capability_hash": capability_hash,
            "producer": "fixture_harness",
            "receipt_hash": "b" * 64,
            "recovered": False,
            "retryable": False,
            "schedule_hash": sha256_hex(
                canonical_json_bytes(
                    {
                        "directives": ["success"],
                        "version": "fixture-harness-fault-schedule/1.0.0",
                    }
                )
            ),
            "schedule_version": "fixture-harness-fault-schedule/1.0.0",
            "status": "succeeded",
        },
    )
    result_ref = tmp_path / "boundaries" / "harness" / "provider-results" / key / "00000000000000000001.ref"
    result_ref.parent.mkdir(parents=True, exist_ok=True)
    substitute_ref(result_ref, result)

    class CountingFactory:
        def __init__(self) -> None:
            self.calls = 0

        def build_fixture(
            self,
            root: Path,
            store: ArtifactStore,
            config: FixtureHarnessConfig,
            policy_capability_hash: str,
        ) -> PersistentFixtureHarness:
            self.calls += 1
            return PersistentFixtureHarness(
                root,
                store,
                config.provider_fault_schedule,
                policy_capability_hash=policy_capability_hash,
            )

    factory = CountingFactory()
    failed = TypedHarnessWorkflow(tmp_path, config, boundary_factory=factory).advance(item, epoch=2)
    assert factory.calls == 0
    assert failed.transitions[-1].schema_name == "ActionObservation"
    assert failed.observations[-1].payload["failure_code"] == "PROVIDER_ARTIFACT_ERROR"
    assert isinstance(failed.observations[-1].payload["integrity_evidence_hash"], str)
    terminal = TypedHarnessWorkflow(tmp_path, config).run_to_terminal(item, epoch=2)
    assert terminal.outcomes[-1].payload["status"] == "provider_error"
    assert terminal.outcomes[-1].payload["reason_code"] == "PROVIDER_ARTIFACT_ERROR"
    replay = TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=3)
    assert replay.outcomes[-1].content_hash == terminal.outcomes[-1].content_hash
    assert PersistentFixtureHarness.external_write_count(tmp_path) == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "state_ref_missing",
        "state_ref_wrong",
        "state_artifact_missing",
        "state_schema_wrong",
        "state_key_wrong",
        "state_version_wrong",
        "state_capability_wrong",
        "receipt_ref_missing",
        "receipt_ref_wrong",
        "receipt_artifact_missing",
        "receipt_schema_wrong",
        "receipt_key_wrong",
        "receipt_capability_wrong",
    ],
)
def test_pending_result_preflight_validates_exact_state_and_receipt_lineage(
    tmp_path: Path,
    mutation: str,
) -> None:
    item = DecisionProposal(
        proposal_id=f"proposal-pending-result-{mutation}",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 0),
    )
    config = FixtureHarnessConfig()
    workflow = TypedHarnessWorkflow(tmp_path, config)
    snapshot = workflow.advance(item, epoch=1)
    snapshot = workflow.advance(item, epoch=1)
    snapshot = workflow.advance(item, epoch=1)
    assert snapshot.transitions[-1].schema_name == "HarnessDecision"
    with pytest.raises(InjectedHarnessCrash):
        workflow.advance(item, epoch=1, crash_after="external_execution")

    decision = snapshot.decisions[-1]
    key = str(decision.payload["idempotency_key"])
    result_ref = tmp_path / "boundaries" / "harness" / "provider-results" / key / "00000000000000000001.ref"
    state_ref = (
        tmp_path / "boundaries" / "harness" / "resources" / "fixture-counter" / "versions" / "00000000000000000001.ref"
    )
    receipt_ref = tmp_path / "boundaries" / "harness" / "receipts" / f"{key}.ref"
    store = ArtifactStore(tmp_path)
    result = store.read(result_ref.read_text().strip(), expected_schema_name="FixtureHarnessResult")
    state = store.read(str(result.payload["output_hash"]), expected_schema_name="FixtureHarnessResourceState")
    receipt = store.read(str(result.payload["receipt_hash"]), expected_schema_name="FixtureHarnessReceipt")

    def replace_result(*, output_hash: str = state.content_hash, receipt_hash: str = receipt.content_hash) -> None:
        payload = result.payload
        payload["output_hash"] = output_hash
        payload["receipt_hash"] = receipt_hash
        substitute_ref(result_ref, store.put("FixtureHarnessResult", "1.0.0", payload))

    def replace_receipt(
        updates: dict[str, JsonValue],
        *,
        schema_name: str = "FixtureHarnessReceipt",
    ) -> Artifact:
        payload = receipt.payload
        payload.update(updates)
        replacement = store.put(schema_name, "1.0.0", payload)
        substitute_ref(receipt_ref, replacement)
        replace_result(receipt_hash=replacement.content_hash)
        return replacement

    def replace_state(
        updates: dict[str, JsonValue],
        *,
        schema_name: str = "FixtureHarnessResourceState",
    ) -> Artifact:
        payload = state.payload
        payload.update(updates)
        replacement = store.put(schema_name, "1.0.0", payload)
        substitute_ref(state_ref, replacement)
        replacement_receipt = replace_receipt({"resource_state_hash": replacement.content_hash})
        replace_result(
            output_hash=replacement.content_hash,
            receipt_hash=replacement_receipt.content_hash,
        )
        return replacement

    wrong_schema = store.put(
        "AuxiliaryHarnessObservation",
        "1.0.0",
        {"proposal_id": item.proposal_id, "status": "not_boundary_evidence"},
    )
    if mutation == "state_ref_missing":
        state_ref.unlink()
    elif mutation == "state_ref_wrong":
        substitute_ref(state_ref, wrong_schema)
    elif mutation == "state_artifact_missing":
        state.path.unlink()
    elif mutation == "state_schema_wrong":
        replace_state({}, schema_name="NotFixtureHarnessResourceState")
    elif mutation == "state_key_wrong":
        replace_state({"idempotency_key": "f" * 64})
    elif mutation == "state_version_wrong":
        replace_state({"version": 2})
    elif mutation == "state_capability_wrong":
        replace_state({"policy_capability_hash": "f" * 64})
    elif mutation == "receipt_ref_missing":
        receipt_ref.unlink()
    elif mutation == "receipt_ref_wrong":
        substitute_ref(receipt_ref, wrong_schema)
    elif mutation == "receipt_artifact_missing":
        receipt.path.unlink()
    elif mutation == "receipt_schema_wrong":
        replace_receipt({}, schema_name="NotFixtureHarnessReceipt")
    elif mutation == "receipt_key_wrong":
        replace_receipt({"idempotency_key": "f" * 64})
    else:
        replace_receipt({"policy_capability_hash": "f" * 64})
    expected_write_count = PersistentFixtureHarness.external_write_count(tmp_path)

    class CountingFactory:
        def __init__(self) -> None:
            self.calls = 0

        def build_fixture(
            self,
            root: Path,
            store: ArtifactStore,
            config: FixtureHarnessConfig,
            policy_capability_hash: str,
        ) -> PersistentFixtureHarness:
            self.calls += 1
            return PersistentFixtureHarness(
                root,
                store,
                config.provider_fault_schedule,
                policy_capability_hash=policy_capability_hash,
            )

    factory = CountingFactory()
    failed = TypedHarnessWorkflow(tmp_path, config, boundary_factory=factory).advance(item, epoch=2)
    assert factory.calls == 0
    assert failed.observations[-1].payload["failure_code"] == "PROVIDER_ARTIFACT_ERROR"
    assert isinstance(failed.observations[-1].payload["integrity_evidence_hash"], str)
    terminal = TypedHarnessWorkflow(tmp_path, config).run_to_terminal(item, epoch=2)
    assert terminal.outcomes[-1].payload["status"] == "provider_error"
    replay = TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=3)
    assert replay.outcomes[-1].content_hash == terminal.outcomes[-1].content_hash
    assert PersistentFixtureHarness.external_write_count(tmp_path) == expected_write_count


def test_legitimate_two_version_resource_chain_revalidates_both_workflows_and_receipts(
    tmp_path: Path,
) -> None:
    config = FixtureHarnessConfig()
    first = DecisionProposal(
        proposal_id="proposal-legitimate-resource-v1",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 0),
    )
    first_terminal = TypedHarnessWorkflow(tmp_path, config).run_to_terminal(first, epoch=1)
    second = DecisionProposal(
        proposal_id="proposal-legitimate-resource-v2",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 1),
    )
    second_terminal = TypedHarnessWorkflow(tmp_path, config).run_to_terminal(second, epoch=1)
    store = ArtifactStore(tmp_path)
    states = [
        store.read(ref.read_text().strip(), expected_schema_name="FixtureHarnessResourceState")
        for ref in sorted(
            (tmp_path / "boundaries" / "harness" / "resources" / "fixture-counter" / "versions").glob("*.ref")
        )
    ]

    assert [state.payload["version"] for state in states] == [1, 2]
    assert states[1].payload["previous_state_hash"] == states[0].content_hash
    for state, terminal in zip(states, (first_terminal, second_terminal), strict=True):
        proposal_artifact = terminal.proposal
        assert proposal_artifact is not None
        plan = terminal.transitions[1]
        authorizing_decision_hash = state.payload["harness_decision_hash"]
        assert authorizing_decision_hash in {decision.content_hash for decision in terminal.decisions}
        assert state.payload["proposal_id"] == proposal_artifact.payload["proposal_id"]
        assert state.payload["proposal_hash"] == proposal_artifact.content_hash
        assert state.payload["action_plan_hash"] == plan.content_hash
        assert state.payload["policy_hash"] == proposal_artifact.payload["policy_hash"]
        key = str(state.payload["idempotency_key"])
        receipt_ref = tmp_path / "boundaries" / "harness" / "receipts" / f"{key}.ref"
        receipt = store.read(receipt_ref.read_text().strip(), expected_schema_name="FixtureHarnessReceipt")
        assert receipt.payload["resource_state_hash"] == state.content_hash
        assert receipt.payload["proposal_hash"] == proposal_artifact.content_hash
        assert terminal.observations[-1].payload["receipt_hash"] == receipt.content_hash
        assert terminal.outcomes[-1].payload["receipt_hash"] == receipt.content_hash

    replay = TypedHarnessWorkflow.resume(tmp_path, second.proposal_id, epoch=9)
    assert replay.outcomes[-1].content_hash == second_terminal.outcomes[-1].content_hash
    assert PersistentFixtureHarness.external_write_count(tmp_path) == 2


@pytest.mark.parametrize("deleted", ["ref", "artifact"])
def test_terminal_replay_revalidates_every_predecessor_receipt(
    tmp_path: Path,
    deleted: str,
) -> None:
    config = FixtureHarnessConfig()
    first = DecisionProposal(
        proposal_id=f"proposal-recert-v1-{deleted}",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 0),
    )
    first_terminal = TypedHarnessWorkflow(tmp_path, config).run_to_terminal(first, epoch=1)
    second = DecisionProposal(
        proposal_id=f"proposal-recert-v2-{deleted}",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 1),
    )
    TypedHarnessWorkflow(tmp_path, config).run_to_terminal(second, epoch=1)
    first_key = str(first_terminal.observations[-1].payload["idempotency_key"])
    receipt_ref = tmp_path / "boundaries" / "harness" / "receipts" / f"{first_key}.ref"
    if deleted == "ref":
        receipt_ref.unlink()
    else:
        receipt_hash = receipt_ref.read_text().strip()
        (tmp_path / "artifacts" / f"{receipt_hash}.json").unlink()

    with pytest.raises((HarnessJournalCorruption, ArtifactCorruption)):
        TypedHarnessWorkflow.resume(tmp_path, second.proposal_id, epoch=2)

    assert PersistentFixtureHarness.external_write_count(tmp_path) == 2
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 2


@pytest.mark.parametrize("mutation", ["missing", "malformed", "substituted", "duplicate"])
def test_pending_retry_preflights_complete_result_and_late_ref_set_before_factory(
    tmp_path: Path,
    mutation: str,
) -> None:
    item = DecisionProposal(
        proposal_id=f"proposal-pending-late-{mutation}",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 0),
    )
    config = FixtureHarnessConfig(provider_fault_schedule=("delayed", "success"))
    workflow = TypedHarnessWorkflow(tmp_path, config)
    snapshot = workflow.advance(item, epoch=1)
    for _ in range(5):
        snapshot = workflow.advance(item, epoch=1)
    assert snapshot.transitions[-1].schema_name == "HarnessDecision"
    with pytest.raises(InjectedHarnessCrash):
        workflow.advance(item, epoch=1, crash_after="external_execution")
    key = str(snapshot.decisions[-1].payload["idempotency_key"])
    late_ref = tmp_path / "boundaries" / "harness" / "late-outcomes" / key / "00000000000000000001.ref"
    original_bytes = late_ref.read_bytes()
    store = ArtifactStore(tmp_path)
    if mutation == "missing":
        late_ref.unlink()
    elif mutation == "malformed":
        late_ref.write_bytes(b"malformed\n")
    elif mutation == "substituted":
        late = store.read(original_bytes.decode("ascii").strip(), expected_schema_name="FixtureLateHarnessOutcome")
        payload = late.payload
        payload["proposal_hash"] = "f" * 64
        substitute_ref(late_ref, store.put("FixtureLateHarnessOutcome", "1.0.0", payload))
    else:
        duplicate_ref = late_ref.with_name("00000000000000000099.ref")
        duplicate_ref.write_bytes(original_bytes)

    class CountingFactory:
        def __init__(self) -> None:
            self.calls = 0

        def build_fixture(
            self,
            root: Path,
            store: ArtifactStore,
            config: FixtureHarnessConfig,
            policy_capability_hash: str,
        ) -> PersistentFixtureHarness:
            self.calls += 1
            return PersistentFixtureHarness(
                root,
                store,
                config.provider_fault_schedule,
                policy_capability_hash=policy_capability_hash,
            )

    factory = CountingFactory()
    failed = TypedHarnessWorkflow(tmp_path, config, boundary_factory=factory).advance(item, epoch=2)
    assert factory.calls == 0
    assert failed.observations[-1].payload["failure_code"] == "PROVIDER_ARTIFACT_ERROR"
    terminal = TypedHarnessWorkflow(tmp_path, config).run_to_terminal(item, epoch=2)
    assert terminal.outcomes[-1].payload["status"] == "provider_error"
    assert PersistentFixtureHarness.external_write_count(tmp_path) == 1


@pytest.mark.parametrize(
    "ref_bytes",
    [
        b"\xff\n",
        b"short\n",
        ("z" * 64 + "\n").encode("ascii"),
        ("f" * 64 + "\n").encode("ascii"),
    ],
)
def test_malformed_or_missing_terminal_ref_fails_closed_without_reexecution(
    tmp_path: Path,
    ref_bytes: bytes,
) -> None:
    item, snapshot = completed(tmp_path, "proposal-terminal-ref-corrupt")
    original_outcome = snapshot.outcomes[-1]
    transition_ref(tmp_path, item.proposal_id, 5).write_bytes(ref_bytes)

    with pytest.raises(HarnessJournalCorruption):
        TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=2)

    assert ArtifactStore(tmp_path).read(original_outcome.content_hash).content_hash == original_outcome.content_hash
    assert PersistentFixtureHarness.external_write_count(tmp_path) == 1
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 1


def test_self_consistent_downstream_substitution_with_wrong_action_input_and_key_is_rejected(
    tmp_path: Path,
) -> None:
    item, snapshot = completed(tmp_path, "proposal-downstream-substitution")
    store = ArtifactStore(tmp_path)
    proposal, plan, decision, observation, outcome = snapshot.transitions
    original_state = store.read(
        str(observation.payload["output_hash"]),
        expected_schema_name="FixtureHarnessResourceState",
    )
    original_result = store.read(
        str(observation.payload["provider_result_hash"]),
        expected_schema_name="FixtureHarnessResult",
    )
    bad_input = "a" * 64
    bad_key = "b" * 64

    state_payload = original_state.payload
    state_payload["action_input_hash"] = bad_input
    state_payload["idempotency_key"] = bad_key
    bad_state = store.put("FixtureHarnessResourceState", "1.0.0", state_payload)
    substitute_ref(
        tmp_path / "boundaries" / "harness" / "resources" / "fixture-counter" / "versions" / "00000000000000000001.ref",
        bad_state,
    )
    bad_receipt = store.put(
        "FixtureHarnessReceipt",
        "1.0.0",
        {
            "action_input_hash": bad_input,
            "idempotency_key": bad_key,
            "resource_id": "fixture-counter",
            "resource_state_hash": bad_state.content_hash,
            "status": "applied",
        },
    )
    receipt_ref = tmp_path / "boundaries" / "harness" / "receipts" / f"{bad_key}.ref"
    receipt_ref.parent.mkdir(parents=True, exist_ok=True)
    receipt_ref.write_bytes(f"{bad_receipt.content_hash}\n".encode("ascii"))

    result_payload = original_result.payload
    result_payload["idempotency_key"] = bad_key
    result_payload["input_hash"] = bad_input
    result_payload["output_hash"] = bad_state.content_hash
    bad_result = store.put("FixtureHarnessResult", "1.0.0", result_payload)
    result_ref = tmp_path / "boundaries" / "harness" / "provider-results" / bad_key / "00000000000000000001.ref"
    result_ref.parent.mkdir(parents=True, exist_ok=True)
    result_ref.write_bytes(f"{bad_result.content_hash}\n".encode("ascii"))

    decision_payload = decision.payload
    decision_payload["idempotency_key"] = bad_key
    decision_payload["input_hash"] = bad_input
    bad_decision = store.put("HarnessDecision", "1.0.0", decision_payload)
    observation_payload = observation.payload
    observation_payload.update(
        {
            "decision_hash": bad_decision.content_hash,
            "idempotency_key": bad_key,
            "input_hash": bad_input,
            "output_hash": bad_state.content_hash,
            "previous_transition_hash": bad_decision.content_hash,
            "provider_result_hash": bad_result.content_hash,
        }
    )
    bad_observation = store.put("ActionObservation", "1.0.0", observation_payload)
    audit = store.read(str(outcome.payload["audit_event_hash"]), expected_schema_name="AuditEvent")
    audit_payload = audit.payload
    audit_payload.update(
        {
            "decision_hash": bad_decision.content_hash,
            "idempotency_key": bad_key,
            "input_hash": bad_input,
            "observation_hash": bad_observation.content_hash,
            "output_hash": bad_state.content_hash,
        }
    )
    bad_audit = store.put("AuditEvent", "1.0.0", audit_payload)
    outcome_payload = outcome.payload
    outcome_payload.update(
        {
            "audit_event_hash": bad_audit.content_hash,
            "decision_hash": bad_decision.content_hash,
            "idempotency_key": bad_key,
            "input_hash": bad_input,
            "observation_hash": bad_observation.content_hash,
            "output_hash": bad_state.content_hash,
            "previous_transition_hash": bad_observation.content_hash,
        }
    )
    bad_outcome = store.put("DecisionOutcome", "1.0.0", outcome_payload)
    for sequence, artifact in enumerate((bad_decision, bad_observation, bad_outcome), start=3):
        substitute_ref(transition_ref(tmp_path, item.proposal_id, sequence), artifact)

    with pytest.raises(HarnessJournalCorruption):
        TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=2)

    assert plan.payload["action_input_hash"] != bad_input
    proposal_artifact = snapshot.proposal
    assert proposal_artifact is not None
    assert proposal.content_hash == proposal_artifact.content_hash
    assert PersistentFixtureHarness.external_write_count(tmp_path) == 1


def test_self_consistent_resource_predecessor_substitution_is_rejected(tmp_path: Path) -> None:
    item, snapshot = completed(tmp_path, "proposal-resource-predecessor-substitution")
    store = ArtifactStore(tmp_path)
    _proposal, _plan, _decision, observation, outcome = snapshot.transitions
    state = store.read(str(observation.payload["output_hash"]), expected_schema_name="FixtureHarnessResourceState")
    result = store.read(
        str(observation.payload["provider_result_hash"]),
        expected_schema_name="FixtureHarnessResult",
    )
    key = str(observation.payload["idempotency_key"])

    state_payload = state.payload
    state_payload["previous_state_hash"] = "a" * 64
    substituted_state = store.put("FixtureHarnessResourceState", "1.0.0", state_payload)
    substitute_ref(
        tmp_path / "boundaries" / "harness" / "resources" / "fixture-counter" / "versions" / "00000000000000000001.ref",
        substituted_state,
    )
    receipt_ref = tmp_path / "boundaries" / "harness" / "receipts" / f"{key}.ref"
    receipt = store.read(receipt_ref.read_text().strip(), expected_schema_name="FixtureHarnessReceipt")
    receipt_payload = receipt.payload
    receipt_payload["resource_state_hash"] = substituted_state.content_hash
    substituted_receipt = store.put("FixtureHarnessReceipt", "1.0.0", receipt_payload)
    substitute_ref(receipt_ref, substituted_receipt)

    result_payload = result.payload
    result_payload["output_hash"] = substituted_state.content_hash
    substituted_result = store.put("FixtureHarnessResult", "1.0.0", result_payload)
    substitute_ref(
        tmp_path / "boundaries" / "harness" / "provider-results" / key / "00000000000000000001.ref",
        substituted_result,
    )
    observation_payload = observation.payload
    observation_payload["output_hash"] = substituted_state.content_hash
    observation_payload["provider_result_hash"] = substituted_result.content_hash
    substituted_observation = store.put("ActionObservation", "1.0.0", observation_payload)
    audit = store.read(str(outcome.payload["audit_event_hash"]), expected_schema_name="AuditEvent")
    audit_payload = audit.payload
    audit_payload["observation_hash"] = substituted_observation.content_hash
    audit_payload["output_hash"] = substituted_state.content_hash
    substituted_audit = store.put("AuditEvent", "1.0.0", audit_payload)
    outcome_payload = outcome.payload
    outcome_payload["audit_event_hash"] = substituted_audit.content_hash
    outcome_payload["observation_hash"] = substituted_observation.content_hash
    outcome_payload["output_hash"] = substituted_state.content_hash
    outcome_payload["previous_transition_hash"] = substituted_observation.content_hash
    substituted_outcome = store.put("DecisionOutcome", "1.0.0", outcome_payload)
    substitute_ref(transition_ref(tmp_path, item.proposal_id, 4), substituted_observation)
    substitute_ref(transition_ref(tmp_path, item.proposal_id, 5), substituted_outcome)

    with pytest.raises(HarnessJournalCorruption):
        TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=2)

    assert PersistentFixtureHarness.external_write_count(tmp_path) == 1
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda audit, outcome: audit.update({"caller": "wrong-caller"}),
        lambda audit, outcome: audit.update({"policy_hash": "c" * 64}),
        lambda audit, outcome: outcome.update({"previous_outcome_hash": "d" * 64}),
        lambda audit, outcome: outcome.update({"unexpected_field": "must-fail-closed"}),
        lambda audit, outcome: outcome.update({"terminal": 1}),
        lambda audit, outcome: outcome.update({"terminal": False}),
    ],
)
def test_audit_and_outcome_lineage_or_extra_field_substitution_is_rejected(
    tmp_path: Path,
    mutate: Callable[[dict[str, JsonValue], dict[str, JsonValue]], None],
) -> None:
    item, snapshot = completed(tmp_path, "proposal-audit-outcome-substitution")
    store = ArtifactStore(tmp_path)
    original_outcome = snapshot.outcomes[-1]
    audit = store.read(str(original_outcome.payload["audit_event_hash"]), expected_schema_name="AuditEvent")
    audit_payload = audit.payload
    outcome_payload = original_outcome.payload
    mutate(audit_payload, outcome_payload)
    bad_audit = store.put("AuditEvent", "1.0.0", audit_payload)
    outcome_payload["audit_event_hash"] = bad_audit.content_hash
    bad_outcome = store.put("DecisionOutcome", "1.0.0", outcome_payload)
    substitute_ref(transition_ref(tmp_path, item.proposal_id, 5), bad_outcome)

    with pytest.raises(HarnessJournalCorruption):
        TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=2)

    assert store.read(original_outcome.content_hash).content_hash == original_outcome.content_hash
    assert PersistentFixtureHarness.external_write_count(tmp_path) == 1
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        {"workflow_sequence": 99},
        {"previous_transition_hash": "e" * 64},
    ],
)
def test_transition_sequence_or_previous_hash_substitution_is_rejected(
    tmp_path: Path,
    mutation: dict[str, JsonValue],
) -> None:
    item, snapshot = completed(tmp_path, "proposal-transition-chain-substitution")
    plan = snapshot.transitions[1]
    payload = plan.payload
    payload.update(mutation)
    bad_plan = ArtifactStore(tmp_path).put("ActionPlan", "1.0.0", payload)
    substitute_ref(transition_ref(tmp_path, item.proposal_id, 2), bad_plan)

    with pytest.raises(HarnessJournalCorruption):
        TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=2)

    assert PersistentFixtureHarness.external_write_count(tmp_path) == 1


def test_unknown_transition_schema_major_fails_closed(tmp_path: Path) -> None:
    item, snapshot = completed(tmp_path, "proposal-unknown-transition-major")
    outcome = snapshot.outcomes[-1]
    future = ArtifactStore(tmp_path).put("DecisionOutcome", "2.0.0", outcome.payload)
    substitute_ref(transition_ref(tmp_path, item.proposal_id, 5), future)

    with pytest.raises(UnknownSchemaMajor):
        TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=2)

    assert PersistentFixtureHarness.external_write_count(tmp_path) == 1


@pytest.mark.parametrize(
    ("caller", "phase", "resource_id", "decision_value", "reason_code"),
    [
        ("not-allowlisted", "TRAIN_35B", "fixture-counter", "allow", "POLICY_ALLOWED"),
        ("fixture-governor", "EVAL_35B", "fixture-counter", "allow", "POLICY_ALLOWED"),
        ("fixture-governor", "TRAIN_35B", "not-allowlisted", "allow", "POLICY_ALLOWED"),
        ("fixture-governor", "TRAIN_35B", "fixture-counter", "deny", "RESOURCE_NOT_ALLOWLISTED"),
        ("fixture-governor", "TRAIN_35B", "fixture-counter", "unknown", "POLICY_ALLOWED"),
        ("fixture-governor", "TRAIN_35B", "fixture-counter", "allow", "WRONG_REASON"),
    ],
)
def test_content_valid_policy_decision_substitution_is_rejected_before_provider(
    tmp_path: Path,
    caller: str,
    phase: str,
    resource_id: str,
    decision_value: str,
    reason_code: str,
) -> None:
    item = DecisionProposal(
        proposal_id=f"proposal-decision-substitute-{caller}-{phase}-{resource_id}-{decision_value}-{reason_code}",
        caller=caller,
        phase=phase,
        action=IncrementFixtureResource(resource_id, 0),
    )
    workflow = TypedHarnessWorkflow(tmp_path, FixtureHarnessConfig())
    snapshot = workflow.advance(item, epoch=1)
    snapshot = workflow.advance(item, epoch=1)
    snapshot = workflow.advance(item, epoch=1)
    decision = snapshot.decisions[-1]
    payload = decision.payload
    payload["decision"] = decision_value
    payload["reason_code"] = reason_code
    substituted = ArtifactStore(tmp_path).put("HarnessDecision", "1.0.0", payload)
    substitute_ref(transition_ref(tmp_path, item.proposal_id, 3), substituted)

    with pytest.raises(HarnessJournalCorruption):
        TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=2)

    assert PersistentFixtureHarness.external_write_count(tmp_path) == 0
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 0


def test_boolean_disguised_action_integers_are_rejected_before_provider(tmp_path: Path) -> None:
    item = DecisionProposal(
        proposal_id="proposal-boolean-action-fields",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 0),
    )
    workflow = TypedHarnessWorkflow(tmp_path, FixtureHarnessConfig())
    workflow.advance(item, epoch=1)
    snapshot = workflow.advance(item, epoch=1)
    plan = snapshot.transitions[-1]
    store = ArtifactStore(tmp_path)
    action = store.read(str(plan.payload["action_input_hash"]), expected_schema_name="TypedHarnessAction")
    action_payload = action.payload
    action_payload["amount"] = True
    action_payload["child_index"] = False
    substituted_action = store.put("TypedHarnessAction", "1.0.0", action_payload)
    proposal_artifact = snapshot.proposal
    assert proposal_artifact is not None
    substituted_key = sha256_hex(
        canonical_json_bytes(
            {
                "action_input_hash": substituted_action.content_hash,
                "child_index": 0,
                "domain": "harness-child-idempotency/1.0.0",
                "proposal_hash": proposal_artifact.content_hash,
            }
        )
    )
    plan_payload = plan.payload
    plan_payload["action_input_hash"] = substituted_action.content_hash
    plan_payload["idempotency_key"] = substituted_key
    substituted_plan = store.put("ActionPlan", "1.0.0", plan_payload)
    substitute_ref(transition_ref(tmp_path, item.proposal_id, 2), substituted_plan)

    with pytest.raises(HarnessJournalCorruption):
        TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=2)

    assert PersistentFixtureHarness.external_write_count(tmp_path) == 0
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 0


def test_missing_late_outcome_artifact_invalidates_terminal_without_reexecution(tmp_path: Path) -> None:
    item = DecisionProposal(
        proposal_id="proposal-missing-late-artifact",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 0),
    )
    config = FixtureHarnessConfig(provider_fault_schedule=("delayed", "success"))
    snapshot = TypedHarnessWorkflow(tmp_path, config).run_to_terminal(item, epoch=1)
    late_hashes = snapshot.observations[-1].payload["late_outcome_hashes"]
    assert isinstance(late_hashes, list) and len(late_hashes) == 1
    late_hash = late_hashes[0]
    assert isinstance(late_hash, str)
    (tmp_path / "artifacts" / f"{late_hash}.json").unlink()

    with pytest.raises(HarnessJournalCorruption):
        TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=2)

    assert PersistentFixtureHarness.external_write_count(tmp_path) == 1
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 2


@pytest.mark.parametrize("mutation", ["duplicate", "substituted"])
def test_duplicate_or_substituted_late_evidence_invalidates_terminal(
    tmp_path: Path,
    mutation: str,
) -> None:
    item = DecisionProposal(
        proposal_id=f"proposal-invalid-late-{mutation}",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 0),
    )
    config = FixtureHarnessConfig(provider_fault_schedule=("delayed", "success"))
    snapshot = TypedHarnessWorkflow(tmp_path, config).run_to_terminal(item, epoch=1)
    store = ArtifactStore(tmp_path)
    observation = snapshot.observations[-1]
    outcome = snapshot.outcomes[-1]
    result = store.read(
        str(observation.payload["provider_result_hash"]),
        expected_schema_name="FixtureHarnessResult",
    )
    raw_late_hashes = observation.payload["late_outcome_hashes"]
    assert isinstance(raw_late_hashes, list) and len(raw_late_hashes) == 1
    original_late_hash = raw_late_hashes[0]
    assert isinstance(original_late_hash, str)
    replacement_hashes: list[JsonValue]
    if mutation == "duplicate":
        replacement_hashes = [original_late_hash, original_late_hash]
    else:
        late = store.read(original_late_hash, expected_schema_name="FixtureLateHarnessOutcome")
        late_payload = late.payload
        late_payload["proposal_hash"] = "b" * 64
        substituted_late = store.put("FixtureLateHarnessOutcome", "1.0.0", late_payload)
        replacement_hashes = [substituted_late.content_hash]

    result_payload = result.payload
    result_payload["late_outcome_hashes"] = replacement_hashes
    substituted_result = store.put("FixtureHarnessResult", "1.0.0", result_payload)
    substitute_ref(
        tmp_path
        / "boundaries"
        / "harness"
        / "provider-results"
        / str(observation.payload["idempotency_key"])
        / "00000000000000000002.ref",
        substituted_result,
    )
    observation_payload = observation.payload
    observation_payload["late_outcome_hashes"] = replacement_hashes
    observation_payload["provider_result_hash"] = substituted_result.content_hash
    substituted_observation = store.put("ActionObservation", "1.0.0", observation_payload)
    audit = store.read(str(outcome.payload["audit_event_hash"]), expected_schema_name="AuditEvent")
    audit_payload = audit.payload
    audit_payload["observation_hash"] = substituted_observation.content_hash
    substituted_audit = store.put("AuditEvent", "1.0.0", audit_payload)
    outcome_payload = outcome.payload
    outcome_payload["audit_event_hash"] = substituted_audit.content_hash
    outcome_payload["observation_hash"] = substituted_observation.content_hash
    outcome_payload["previous_transition_hash"] = substituted_observation.content_hash
    substituted_outcome = store.put("DecisionOutcome", "1.0.0", outcome_payload)
    substitute_ref(transition_ref(tmp_path, item.proposal_id, 7), substituted_observation)
    substitute_ref(transition_ref(tmp_path, item.proposal_id, 8), substituted_outcome)

    with pytest.raises(HarnessJournalCorruption):
        TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=2)

    assert PersistentFixtureHarness.external_write_count(tmp_path) == 1
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 2


def test_missing_late_outcome_ref_invalidates_terminal_without_reexecution(tmp_path: Path) -> None:
    item = DecisionProposal(
        proposal_id="proposal-missing-late-ref",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 0),
    )
    config = FixtureHarnessConfig(provider_fault_schedule=("delayed", "success"))
    snapshot = TypedHarnessWorkflow(tmp_path, config).run_to_terminal(item, epoch=1)
    key = str(snapshot.observations[-1].payload["idempotency_key"])
    (tmp_path / "boundaries" / "harness" / "late-outcomes" / key / "00000000000000000001.ref").unlink()

    with pytest.raises(HarnessJournalCorruption):
        TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=2)

    assert PersistentFixtureHarness.external_write_count(tmp_path) == 1
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 2


def test_extra_out_of_order_late_ref_invalidates_terminal_without_reexecution(tmp_path: Path) -> None:
    item = DecisionProposal(
        proposal_id="proposal-extra-late-ref",
        caller="fixture-governor",
        phase="TRAIN_35B",
        action=IncrementFixtureResource("fixture-counter", 0),
    )
    config = FixtureHarnessConfig(provider_fault_schedule=("delayed", "success"))
    snapshot = TypedHarnessWorkflow(tmp_path, config).run_to_terminal(item, epoch=1)
    observation = snapshot.observations[-1]
    key = str(observation.payload["idempotency_key"])
    late_hashes = observation.payload["late_outcome_hashes"]
    assert isinstance(late_hashes, list) and len(late_hashes) == 1
    extra_ref = tmp_path / "boundaries" / "harness" / "late-outcomes" / key / "00000000000000000099.ref"
    extra_ref.write_bytes(f"{late_hashes[0]}\n".encode("ascii"))

    with pytest.raises(HarnessJournalCorruption):
        TypedHarnessWorkflow.resume(tmp_path, item.proposal_id, epoch=2)

    assert PersistentFixtureHarness.external_write_count(tmp_path) == 1
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 2


def test_provider_independently_rejects_resource_outside_immutable_capability(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    config = FixtureHarnessConfig()
    policy = store.put("HarnessPolicy", "1.0.0", fixture_policy_payload(config))
    capability = store.put(
        "HarnessPolicyCapability",
        "1.0.0",
        fixture_policy_capability_payload(config, policy.content_hash),
    )
    action = store.put(
        "TypedHarnessAction",
        "1.0.0",
        {
            "action_type": "fixture.resource.increment.v1",
            "amount": 1,
            "child_index": 0,
            "expected_version": 0,
            "policy_capability_hash": capability.content_hash,
            "proposal_hash": "a" * 64,
            "resource_id": "not-allowlisted",
        },
    )
    key = sha256_hex(
        canonical_json_bytes(
            {
                "action_input_hash": action.content_hash,
                "child_index": 0,
                "domain": "harness-child-idempotency/1.0.0",
                "proposal_hash": "a" * 64,
            }
        )
    )
    provider = PersistentFixtureHarness(
        tmp_path,
        store,
        policy_capability_hash=capability.content_hash,
    )

    with pytest.raises(FixtureHarnessAuthorizationDenied, match="resource"):
        provider.query(key, action.content_hash, 1)

    assert PersistentFixtureHarness.external_write_count(tmp_path) == 0
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 0
    assert not list((tmp_path / "boundaries" / "harness" / "provider-results").glob("*/*.ref"))


def test_provider_rejects_boolean_disguised_action_integers_before_lock(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    config = FixtureHarnessConfig()
    policy = store.put("HarnessPolicy", "1.0.0", fixture_policy_payload(config))
    capability = store.put(
        "HarnessPolicyCapability",
        "1.0.0",
        fixture_policy_capability_payload(config, policy.content_hash),
    )
    action = store.put(
        "TypedHarnessAction",
        "1.0.0",
        {
            "action_type": "fixture.resource.increment.v1",
            "amount": True,
            "child_index": False,
            "expected_version": 0,
            "policy_capability_hash": capability.content_hash,
            "proposal_hash": "d" * 64,
            "resource_id": "fixture-counter",
        },
    )
    provider = PersistentFixtureHarness(
        tmp_path,
        store,
        policy_capability_hash=capability.content_hash,
    )

    with pytest.raises(FixtureHarnessCorruption, match="action input"):
        provider.execute("e" * 64, action, 1)

    assert PersistentFixtureHarness.external_write_count(tmp_path) == 0
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 0


@pytest.mark.parametrize(
    ("historical_resources", "historical_expected_version"),
    [
        (("other-resource",), 0),
        (("fixture-counter",), 99),
    ],
)
def test_provider_rejects_unauthorized_or_cas_invalid_historical_state(
    tmp_path: Path,
    historical_resources: tuple[str, ...],
    historical_expected_version: int,
) -> None:
    store = ArtifactStore(tmp_path)
    historical_config = FixtureHarnessConfig(allowed_resources=historical_resources)
    historical_policy = store.put("HarnessPolicy", "1.0.0", fixture_policy_payload(historical_config))
    historical_capability = store.put(
        "HarnessPolicyCapability",
        "1.0.0",
        fixture_policy_capability_payload(historical_config, historical_policy.content_hash),
    )
    historical_action = store.put(
        "TypedHarnessAction",
        "1.0.0",
        {
            "action_type": "fixture.resource.increment.v1",
            "amount": 1,
            "child_index": 0,
            "expected_version": historical_expected_version,
            "policy_capability_hash": historical_capability.content_hash,
            "proposal_hash": "a" * 64,
            "resource_id": "fixture-counter",
        },
    )
    historical_key = sha256_hex(
        canonical_json_bytes(
            {
                "action_input_hash": historical_action.content_hash,
                "child_index": 0,
                "domain": "harness-child-idempotency/1.0.0",
                "proposal_hash": "a" * 64,
            }
        )
    )
    forged_state = store.put(
        "FixtureHarnessResourceState",
        "1.0.0",
        {
            "action_input_hash": historical_action.content_hash,
            "action_type": "fixture.resource.increment.v1",
            "amount": 1,
            "idempotency_key": historical_key,
            "policy_capability_hash": historical_capability.content_hash,
            "previous_state_hash": None,
            "resource_id": "fixture-counter",
            "value": 1,
            "version": 1,
        },
    )
    forged_ref = (
        tmp_path / "boundaries" / "harness" / "resources" / "fixture-counter" / "versions" / "00000000000000000001.ref"
    )
    forged_ref.parent.mkdir(parents=True, exist_ok=True)
    substitute_ref(forged_ref, forged_state)

    config = FixtureHarnessConfig()
    policy = store.put("HarnessPolicy", "1.0.0", fixture_policy_payload(config))
    capability = store.put(
        "HarnessPolicyCapability",
        "1.0.0",
        fixture_policy_capability_payload(config, policy.content_hash),
    )
    current_action = store.put(
        "TypedHarnessAction",
        "1.0.0",
        {
            "action_type": "fixture.resource.increment.v1",
            "amount": 1,
            "child_index": 0,
            "expected_version": 1,
            "policy_capability_hash": capability.content_hash,
            "proposal_hash": "b" * 64,
            "resource_id": "fixture-counter",
        },
    )
    current_key = sha256_hex(
        canonical_json_bytes(
            {
                "action_input_hash": current_action.content_hash,
                "child_index": 0,
                "domain": "harness-child-idempotency/1.0.0",
                "proposal_hash": "b" * 64,
            }
        )
    )
    provider = PersistentFixtureHarness(
        tmp_path,
        store,
        policy_capability_hash=capability.content_hash,
    )

    with pytest.raises(FixtureHarnessCorruption):
        provider.execute(current_key, current_action, 1)

    assert PersistentFixtureHarness.external_write_count(tmp_path) == 1
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 0


def test_provider_does_not_recover_from_receipt_without_committed_resource_ref(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    config = FixtureHarnessConfig()
    policy = store.put("HarnessPolicy", "1.0.0", fixture_policy_payload(config))
    capability = store.put(
        "HarnessPolicyCapability",
        "1.0.0",
        fixture_policy_capability_payload(config, policy.content_hash),
    )
    action = store.put(
        "TypedHarnessAction",
        "1.0.0",
        {
            "action_type": "fixture.resource.increment.v1",
            "amount": 1,
            "child_index": 0,
            "expected_version": 0,
            "policy_capability_hash": capability.content_hash,
            "proposal_hash": "c" * 64,
            "resource_id": "fixture-counter",
        },
    )
    key = sha256_hex(
        canonical_json_bytes(
            {
                "action_input_hash": action.content_hash,
                "child_index": 0,
                "domain": "harness-child-idempotency/1.0.0",
                "proposal_hash": "c" * 64,
            }
        )
    )
    uncommitted_state = store.put(
        "FixtureHarnessResourceState",
        "1.0.0",
        {
            "action_input_hash": action.content_hash,
            "action_type": "fixture.resource.increment.v1",
            "amount": 1,
            "idempotency_key": key,
            "policy_capability_hash": capability.content_hash,
            "previous_state_hash": None,
            "resource_id": "fixture-counter",
            "value": 1,
            "version": 1,
        },
    )
    receipt = store.put(
        "FixtureHarnessReceipt",
        "1.0.0",
        {
            "action_input_hash": action.content_hash,
            "idempotency_key": key,
            "policy_capability_hash": capability.content_hash,
            "resource_id": "fixture-counter",
            "resource_state_hash": uncommitted_state.content_hash,
            "status": "applied",
        },
    )
    receipt_ref = tmp_path / "boundaries" / "harness" / "receipts" / f"{key}.ref"
    receipt_ref.parent.mkdir(parents=True, exist_ok=True)
    substitute_ref(receipt_ref, receipt)
    provider = PersistentFixtureHarness(
        tmp_path,
        store,
        policy_capability_hash=capability.content_hash,
    )

    with pytest.raises(FixtureHarnessCorruption, match="not a committed resource version"):
        provider.query(key, action.content_hash, 1)

    assert PersistentFixtureHarness.external_write_count(tmp_path) == 0
    assert PersistentFixtureHarness.provider_invocation_count(tmp_path) == 0
