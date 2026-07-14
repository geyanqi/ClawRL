from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

import clawrl.artifacts as artifacts_module
from clawrl.artifacts import (
    ArtifactCorruption,
    ArtifactDurabilityError,
    ArtifactStore,
    CanonicalizationError,
    UnknownSchemaMajor,
    canonical_json_bytes,
    sha256_hex,
)
from clawrl.training.run_journal import (
    RunAlreadyClosed,
    RunJournal,
    StaleFencingEpoch,
)


def test_canonical_json_covers_the_supported_rfc8785_domain() -> None:
    # RFC 8785 property ordering is by UTF-16 code units, not Python code points.
    value = {
        "\ufb33": "Hebrew presentation form",
        "\U0001f600": "grinning face",
        "\u20ac": "euro",
        "\u00f6": "latin small o diaeresis",
        "\u0080": "control-range edge",
        "1": "ascii digit",
        "\r": "carriage return",
    }

    encoded = canonical_json_bytes(value)

    assert encoded == (
        b'{"\\r":"carriage return","1":"ascii digit","\xc2\x80":"control-range edge",'
        b'"\xc3\xb6":"latin small o diaeresis","\xe2\x82\xac":"euro",'
        b'"\xf0\x9f\x98\x80":"grinning face","\xef\xac\xb3":"Hebrew presentation form"}'
    )
    assert canonical_json_bytes({"s": '\b\t\n\f\r"\\/\u20ac'}) == (b'{"s":"\\b\\t\\n\\f\\r\\"\\\\/\xe2\x82\xac"}')
    assert canonical_json_bytes(
        {
            "basis_points": 10_000,
            "negative": -9_007_199_254_740_991,
            "precise_decimal": "333333333.33333329",
            "zero": 0,
        }
    ) == (b'{"basis_points":10000,"negative":-9007199254740991,"precise_decimal":"333333333.33333329","zero":0}')


@pytest.mark.parametrize(
    "unsupported",
    [
        -0.0,
        333333333.33333329,
        1e30,
        float("nan"),
        float("inf"),
        9_007_199_254_740_992,
    ],
)
def test_canonical_json_rejects_numbers_outside_the_artifact_numeric_contract(
    unsupported: float | int,
) -> None:
    with pytest.raises(CanonicalizationError, match="numeric"):
        canonical_json_bytes({"value": unsupported})


def test_canonical_json_rejects_unpaired_unicode_surrogates() -> None:
    with pytest.raises(CanonicalizationError, match="surrogate"):
        canonical_json_bytes({"value": "\ud800"})


def test_artifact_bytes_and_hash_are_stable_in_a_fresh_subprocess(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    payload = {
        "trace_id": "training-trace-0001",
        "reward_basis_points": 8_750,
        "evidence": ["checkpoint hash verified", "idempotency key reused"],
    }
    ref = store.put("EvidenceLinkedReward", "1.0.0", payload)
    expected_bytes = ref.path.read_bytes()

    script = """
import json
import sys
from clawrl.artifacts import ArtifactStore

root = sys.argv[1]
payload = json.loads(sys.argv[2])
store = ArtifactStore(root)
ref = store.put("EvidenceLinkedReward", "1.0.0", payload)
loaded = store.read(ref.content_hash, expected_schema_name="EvidenceLinkedReward")
print(json.dumps({
    "content_hash": ref.content_hash,
    "raw_hex": loaded.raw_bytes.hex(),
    "payload": loaded.payload,
}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), json.dumps(payload)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    fresh = json.loads(result.stdout)

    assert fresh["content_hash"] == ref.content_hash
    assert bytes.fromhex(fresh["raw_hex"]) == expected_bytes
    assert fresh["payload"] == payload
    assert ref.content_hash == sha256_hex(ref.digest_bytes)


def test_artifact_payload_corruption_fails_closed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put("TrainingTrace", "1.0.0", {"trace_id": "trace-1", "prompt": "safe"})
    corrupted = ref.path.read_bytes().replace(b'"safe"', b'"evil"')
    assert corrupted != ref.path.read_bytes()
    ref.path.write_bytes(corrupted)

    with pytest.raises(ArtifactCorruption, match="content hash"):
        ArtifactStore(tmp_path).read(ref.content_hash, expected_schema_name="TrainingTrace")


def test_artifact_payload_is_a_deep_snapshot_with_no_mutable_aliases(
    tmp_path: Path,
) -> None:
    caller_payload: dict[str, Any] = {
        "nested": {"labels": ["committed"], "score": 1},
        "trace_id": "immutable-trace",
    }
    store = ArtifactStore(tmp_path)
    artifact = store.put("TrainingTrace", "1.0.0", caller_payload)
    committed_bytes = artifact.raw_bytes

    caller_payload["nested"]["labels"].append("caller-mutation")
    caller_payload["nested"]["score"] = 9
    exposed = cast(dict[str, Any], artifact.payload)
    exposed["nested"]["labels"].append("artifact-view-mutation")
    exposed["nested"]["score"] = 7

    expected = {
        "nested": {"labels": ["committed"], "score": 1},
        "trace_id": "immutable-trace",
    }
    assert artifact.payload == expected
    assert artifact.raw_bytes == committed_bytes
    assert (
        ArtifactStore(tmp_path)
        .read(
            artifact.content_hash,
            expected_schema_name="TrainingTrace",
        )
        .payload
        == expected
    )


def test_durable_directory_creation_and_publish_sync_every_new_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced: list[Path] = []
    monkeypatch.setattr(
        artifacts_module,
        "_fsync_directory",
        lambda path: synced.append(Path(path)),
    )
    root = tmp_path / "new-root" / "nested"

    store = ArtifactStore(root)
    artifact = store.put("TrainingTrace", "1.0.0", {"trace_id": "durable"})

    assert artifact.path.exists()
    assert tmp_path in synced
    assert root.parent in synced
    assert root in synced
    assert store.artifact_dir in synced
    assert synced.count(store.artifact_dir) >= 2


def test_directory_fsync_failure_aborts_publication_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path)

    def fail_directory_sync(path: Path) -> None:
        raise OSError(f"injected directory durability failure at {path.name}")

    monkeypatch.setattr(artifacts_module, "_fsync_directory", fail_directory_sync)

    with pytest.raises(ArtifactDurabilityError, match="directory fsync"):
        store.put("TrainingTrace", "1.0.0", {"trace_id": "must-not-publish"})


def test_unknown_schema_major_fails_closed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put("TrainingTrace", "2.0.0", {"trace_id": "trace-future"})

    with pytest.raises(UnknownSchemaMajor, match="major"):
        ArtifactStore(tmp_path).read(ref.content_hash, expected_schema_name="TrainingTrace")


def test_event_chain_fencing_immutable_close_and_terminal_quarantine(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    journal = RunJournal(tmp_path, store, "fixture-run-ticket01")
    workflow_input = store.put(
        "TraceRunInput",
        "1.0.0",
        {"run_id": "fixture-run-ticket01"},
    )
    journal.reserve_identity(workflow_input.content_hash)
    first = journal.start_run(
        41,
        workflow_input.content_hash,
        {
            "input_hash": workflow_input.content_hash,
            "trace_id": "training-trace-0001",
        },
    )
    second = journal.append(41, "SCORING_REQUESTED", {"request_id": "score-1"})

    assert first.payload["sequence"] == 1
    assert first.payload["previous_event_hash"] is None
    assert second.payload["sequence"] == 2
    assert second.payload["previous_event_hash"] == first.content_hash

    restarted = RunJournal(tmp_path, ArtifactStore(tmp_path), "fixture-run-ticket01")
    assert [event.content_hash for event in restarted.events()] == [
        first.content_hash,
        second.content_hash,
    ]
    restarted.claim_epoch(42)
    with pytest.raises(StaleFencingEpoch, match="41"):
        journal.append(41, "STALE_WRITE", {})

    closed = restarted.close(42, status="succeeded", reason_code="TRACE_COMPLETE")
    event_hashes_at_close = [event.content_hash for event in restarted.events()]
    assert restarted.closed().content_hash == closed.content_hash
    assert restarted.close(42, status="succeeded", reason_code="TRACE_COMPLETE").content_hash == (closed.content_hash)
    with pytest.raises(RunAlreadyClosed):
        restarted.close(42, status="failed", reason_code="CONFLICTING_TERMINAL")
    with pytest.raises(RunAlreadyClosed):
        restarted.append(42, "POST_TERMINAL_MUTATION", {})

    late = restarted.record_observation(
        42,
        "LATE_CLUSTER_LOG",
        {"line": "job completed after controller close"},
    )
    assert late.quarantined is True
    assert late.observation.schema_name == "Observation"
    assert late.quarantine_ref is not None
    assert late.quarantine_ref.exists()
    assert [event.content_hash for event in restarted.events()] == event_hashes_at_close

    fresh = RunJournal(tmp_path, ArtifactStore(tmp_path), "fixture-run-ticket01")
    fresh.verify()
    assert fresh.closed().content_hash == closed.content_hash
    assert [item.observation.content_hash for item in fresh.quarantined_observations()] == [
        late.observation.content_hash
    ]


def test_boolean_epoch_cannot_alias_integer_controller_authority(tmp_path: Path) -> None:
    journal = RunJournal(tmp_path, ArtifactStore(tmp_path), "strict-epoch-run")

    with pytest.raises(ValueError, match="positive safe integer"):
        journal.claim_epoch(True)
    with pytest.raises(ValueError, match="positive safe integer"):
        journal.append(True, "BOOL_WRITE", {})
    with pytest.raises(ValueError, match="positive safe integer"):
        journal.close(True, status="failed", reason_code="BOOL_CLOSE")
    with pytest.raises(ValueError, match="positive safe integer"):
        journal.record_observation(True, "BOOL_OBSERVATION", {})

    assert journal.events() == []
