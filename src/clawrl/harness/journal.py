"""Fenced append-only journal for immutable Harness workflow transitions."""

from __future__ import annotations

import fcntl
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from clawrl.artifacts import MAX_SAFE_INTEGER, Artifact, ArtifactCorruption, ArtifactStore

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_TRANSITION_SCHEMAS = {
    "ActionObservation",
    "ActionPlan",
    "DecisionOutcome",
    "DecisionProposal",
    "HarnessDecision",
}


class HarnessJournalError(RuntimeError):
    """Base error for durable Harness coordination."""


class HarnessIdentityConflict(HarnessJournalError):
    """A proposal identifier was previously bound to different input."""


class HarnessJournalCorruption(HarnessJournalError):
    """Committed refs or transition lineage cannot be verified."""


class HarnessHeadConflict(HarnessJournalError):
    """Another writer advanced the expected transition head."""


class StaleHarnessEpoch(HarnessJournalError):
    """A controller lost its externally supplied fencing epoch."""


class HarnessWorkflowTerminal(HarnessJournalError):
    """A terminal DecisionOutcome makes the workflow read-only."""


class HarnessJournal:
    """Proposal-scoped CAS journal whose refs are the only committed state."""

    def __init__(self, root: str | Path, store: ArtifactStore, proposal_id: str) -> None:
        if _SAFE_ID.fullmatch(proposal_id) is None:
            raise ValueError("proposal_id must be a filesystem-safe stable identifier")
        self.root = Path(root)
        self.store = store
        self.proposal_id = proposal_id
        self.workflow_dir = self.root / "harness-workflows" / proposal_id
        self.transitions_dir = self.workflow_dir / "transitions"
        self.fences_dir = self.workflow_dir / "fences"
        ArtifactStore.durable_mkdir(self.transitions_dir)
        ArtifactStore.durable_mkdir(self.fences_dir)
        self.lock_path = self.workflow_dir / "journal.lock"
        ArtifactStore.durable_touch(self.lock_path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self.lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def reserve_identity(
        self,
        input_hash: str,
        config_hash: str,
        policy_hash: str,
        policy_capability_hash: str,
    ) -> Artifact:
        for value in (input_hash, config_hash, policy_hash, policy_capability_hash):
            self._validate_hash(value)
        payload = {
            "config_hash": config_hash,
            "input_hash": input_hash,
            "policy_capability_hash": policy_capability_hash,
            "policy_hash": policy_hash,
            "proposal_id": self.proposal_id,
        }
        reservation = self.store.put("HarnessWorkflowIdentity", "1.0.0", payload)
        with self._locked():
            ref = self.workflow_dir / "identity.ref"
            if ref.exists():
                committed_hash = self._read_ref(ref)
                if committed_hash != reservation.content_hash:
                    raise HarnessIdentityConflict("proposal_id is reserved for different immutable input")
                committed = self.store.read(committed_hash, expected_schema_name="HarnessWorkflowIdentity")
                if committed.payload != payload:
                    raise HarnessJournalCorruption("Harness identity payload conflicts with its ref")
                return committed
            self._publish_ref(ref, reservation.content_hash)
            return reservation

    def identity(self) -> Artifact:
        ref = self.workflow_dir / "identity.ref"
        if not ref.exists():
            raise HarnessJournalCorruption("Harness workflow identity ref is missing")
        identity = self.store.read(self._read_ref(ref), expected_schema_name="HarnessWorkflowIdentity")
        if (
            identity.schema_version != "1.0.0"
            or set(identity.payload)
            != {"config_hash", "input_hash", "policy_capability_hash", "policy_hash", "proposal_id"}
            or identity.payload.get("proposal_id") != self.proposal_id
        ):
            raise HarnessJournalCorruption("Harness workflow identity fields are invalid")
        for field in ("config_hash", "input_hash", "policy_capability_hash", "policy_hash"):
            value = identity.payload.get(field)
            if not isinstance(value, str) or _HASH.fullmatch(value) is None:
                raise HarnessJournalCorruption("Harness workflow identity hash is invalid")
        return identity

    def claim_epoch(self, epoch: int) -> Artifact:
        self._validate_epoch(epoch)
        with self._locked():
            current = self._current_epoch_unlocked()
            if current is not None and epoch < current:
                raise StaleHarnessEpoch(f"controller epoch {epoch} is stale; current epoch is {current}")
            claim = self.store.put(
                "HarnessFenceClaim",
                "1.0.0",
                {"epoch": epoch, "proposal_id": self.proposal_id},
            )
            self._publish_ref(self.fences_dir / f"{epoch:020d}.ref", claim.content_hash)
            return claim

    def transitions(self) -> list[Artifact]:
        with self._locked():
            return self._transitions_unlocked()

    def strict_transition_snapshot(self) -> list[Artifact]:
        """Read an exact ref-only transition set under the writer lock."""

        with self._locked():
            entries = sorted(self.transitions_dir.iterdir())
            if any(not entry.is_file() or entry.suffix != ".ref" for entry in entries):
                raise HarnessJournalCorruption("Harness transition directory contains an unexpected entry")
            return self._transitions_unlocked()

    def append(
        self,
        epoch: int,
        schema_name: str,
        details: Mapping[str, object],
        *,
        expected_sequence: int,
        expected_previous_hash: str | None,
    ) -> Artifact:
        self._validate_epoch(epoch)
        if schema_name not in _TRANSITION_SCHEMAS:
            raise ValueError("unsupported Harness transition schema")
        with self._locked():
            current = self._current_epoch_unlocked()
            if current != epoch:
                raise StaleHarnessEpoch(f"controller epoch {epoch} is not current")
            transitions = self._transitions_unlocked()
            if self._is_terminal(transitions):
                raise HarnessWorkflowTerminal("terminal Harness workflow is read-only")
            actual_sequence = len(transitions) + 1
            actual_previous_hash = transitions[-1].content_hash if transitions else None
            if actual_sequence != expected_sequence or actual_previous_hash != expected_previous_hash:
                raise HarnessHeadConflict("Harness transition head changed before CAS publication")
            payload = dict(details)
            reserved = {
                "controller_epoch",
                "previous_transition_hash",
                "proposal_id",
                "workflow_sequence",
            }
            if reserved.intersection(payload):
                raise ValueError("Harness transition details contain reserved lineage fields")
            payload.update(
                {
                    "controller_epoch": epoch,
                    "previous_transition_hash": actual_previous_hash,
                    "proposal_id": self.proposal_id,
                    "workflow_sequence": actual_sequence,
                }
            )
            transition = self.store.put(schema_name, "1.0.0", payload)
            self._publish_ref(
                self.transitions_dir / f"{actual_sequence:020d}.ref",
                transition.content_hash,
            )
            return transition

    def verify(self) -> None:
        with self._locked():
            self.identity()
            self._current_epoch_unlocked()
            transitions = self._transitions_unlocked()
            terminal_indexes = [
                index
                for index, transition in enumerate(transitions)
                if transition.schema_name == "DecisionOutcome" and transition.payload.get("terminal") is True
            ]
            if terminal_indexes and terminal_indexes != [len(transitions) - 1]:
                raise HarnessJournalCorruption("terminal Harness outcome is not the final transition")

    def _transitions_unlocked(self) -> list[Artifact]:
        refs = sorted(self.transitions_dir.glob("*.ref"))
        transitions: list[Artifact] = []
        previous_hash: str | None = None
        for sequence, ref in enumerate(refs, start=1):
            if ref.name != f"{sequence:020d}.ref":
                raise HarnessJournalCorruption("Harness transition sequence has a gap or unexpected ref")
            try:
                transition = self.store.read(self._read_ref(ref))
            except ArtifactCorruption as error:
                raise HarnessJournalCorruption("Harness transition artifact is corrupt") from error
            payload = transition.payload
            epoch = payload.get("controller_epoch")
            if (
                transition.schema_name not in _TRANSITION_SCHEMAS
                or transition.schema_version != "1.0.0"
                or payload.get("proposal_id") != self.proposal_id
                or type(payload.get("workflow_sequence")) is not int
                or payload.get("workflow_sequence") != sequence
                or payload.get("previous_transition_hash") != previous_hash
                or not isinstance(epoch, int)
                or isinstance(epoch, bool)
                or not 0 < epoch <= MAX_SAFE_INTEGER
            ):
                raise HarnessJournalCorruption("Harness transition lineage is invalid")
            transitions.append(transition)
            previous_hash = transition.content_hash
        return transitions

    def _current_epoch_unlocked(self) -> int | None:
        claims: list[int] = []
        for ref in sorted(self.fences_dir.glob("*.ref")):
            try:
                epoch = int(ref.stem)
            except ValueError as error:
                raise HarnessJournalCorruption("Harness fence ref name is invalid") from error
            if ref.name != f"{epoch:020d}.ref":
                raise HarnessJournalCorruption("Harness fence ref name is invalid")
            claim = self.store.read(self._read_ref(ref), expected_schema_name="HarnessFenceClaim")
            if (
                claim.schema_version != "1.0.0"
                or type(claim.payload.get("epoch")) is not int
                or claim.payload
                != {
                    "epoch": epoch,
                    "proposal_id": self.proposal_id,
                }
            ):
                raise HarnessJournalCorruption("Harness fencing claim conflicts with its ref")
            claims.append(epoch)
        if claims != sorted(set(claims)):
            raise HarnessJournalCorruption("Harness fencing epochs are not unique and monotonic")
        return claims[-1] if claims else None

    @staticmethod
    def _is_terminal(transitions: list[Artifact]) -> bool:
        return bool(
            transitions
            and transitions[-1].schema_name == "DecisionOutcome"
            and transitions[-1].payload.get("terminal") is True
        )

    @staticmethod
    def _validate_epoch(epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or not 0 < epoch <= MAX_SAFE_INTEGER:
            raise ValueError("controller epoch must be a positive safe integer")

    @staticmethod
    def _validate_hash(value: str) -> None:
        if _HASH.fullmatch(value) is None:
            raise ValueError("Harness identity fields must be SHA-256 hashes")

    @staticmethod
    def _read_ref(ref: Path) -> str:
        try:
            raw = ref.read_bytes()
        except OSError as error:
            raise HarnessJournalCorruption("Harness ref cannot be read") from error
        if len(raw) != 65 or raw[-1:] != b"\n":
            raise HarnessJournalCorruption("Harness ref bytes are malformed")
        try:
            value = raw[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise HarnessJournalCorruption("Harness ref is not ASCII") from error
        if _HASH.fullmatch(value) is None:
            raise HarnessJournalCorruption("Harness ref does not contain a SHA-256 hash")
        return value

    @staticmethod
    def _publish_ref(ref: Path, content_hash: str) -> None:
        try:
            ArtifactStore._publish(ref, f"{content_hash}\n".encode("ascii"))
        except Exception as error:
            if isinstance(error, HarnessJournalError):
                raise
            raise HarnessJournalCorruption("Harness ref publication failed or conflicted") from error


def required_string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise HarnessJournalCorruption(f"Harness transition {field} must be a string")
    return value


def required_integer(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise HarnessJournalCorruption(f"Harness transition {field} must be an integer")
    return cast(int, value)
