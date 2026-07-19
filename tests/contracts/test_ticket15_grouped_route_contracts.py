"""Ticket 15 bounded helper and durable grouped-route contracts."""

from pathlib import Path

from clawrl.artifacts import ArtifactStore
from clawrl.router.grouped_route import FixtureHelperSandbox, ToolRequest


def test_allowlisted_helpers_are_read_only_and_scoped() -> None:
    sandbox = FixtureHelperSandbox(uid="uid-001", global_step=7, batch=[{"score": 1}, {"score": 2}])
    assert sandbox.execute(ToolRequest("python", "len(batch)"))[0] == "allow"
    assert sandbox.execute(ToolRequest("javascript", "batch.length"))[1] == 2
    assert sandbox.execute(ToolRequest("regex", "score=>score"))[0] == "allow"
    before = sandbox.batch
    assert sandbox.execute(ToolRequest("python", "len(batch)", uid="uid-other"))[0] == "deny"
    assert sandbox.execute(ToolRequest("python", "len(batch)", global_step=8))[0] == "deny"
    assert sandbox.execute(ToolRequest("python", "open('x')"))[0] == "deny"
    assert sandbox.execute(ToolRequest("python", "network_token"))[0] == "deny"
    assert sandbox.batch == before and sandbox.scratch == {}


def test_helper_rejects_unbounded_backtracking_before_match() -> None:
    sandbox = FixtureHelperSandbox(uid="uid-001", global_step=7, batch=[{"score": "a" * 2_000}])
    for program in ("(a|aa)+$=>score", "(.*)*$=>score", "a{1,100000000}=>score"):
        status, value, reason = sandbox.execute(ToolRequest("regex", program))
        assert (status, value) == ("deny", None)
        assert reason == "HELPER_ERROR"
    assert sandbox.execute(ToolRequest("python", "1/0"))[0] == "deny"


def test_output_blob_is_content_addressed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    digest, size = store.put_blob(b"fixture")
    assert len(digest) == 64
    assert store.read_blob(digest, expected_size=size) == b"fixture"
