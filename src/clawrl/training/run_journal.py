"""Persistent single-writer run event journal for the walking skeleton."""

from __future__ import annotations

import fcntl
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from clawrl.artifacts import (
    MAX_SAFE_INTEGER,
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    canonical_json_bytes,
    sha256_hex,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_UNSET = object()


class RunJournalError(RuntimeError):
    """Base class for fail-closed run journal errors."""


class StaleFencingEpoch(RunJournalError):
    """A controller no longer owns the externally supplied epoch."""


class RunAlreadyClosed(RunJournalError):
    """An immutable terminal event already exists."""


class RunJournalCorruption(RunJournalError):
    """The append-only journal cannot be verified from disk."""


class JournalHeadConflict(RunJournalError):
    """The caller's persisted transition view lost an optimistic head CAS."""


class RunIdentityConflict(RunJournalError):
    """A run_id is already reserved for a different immutable input."""


class RunNotStarted(RunJournalError):
    """A journal operation cannot precede its atomic RUN_STARTED event."""


@dataclass(frozen=True, slots=True)
class ObservationReceipt:
    observation: Artifact
    quarantined: bool
    quarantine_ref: Path | None
    event: Artifact | None = None


class RunJournal:
    """Append-only event refs plus externally fenced controller writes.

    Artifact publication can precede its ref. Such an artifact is an uncommitted
    orphan and is harmless: committed journal state is derived only from the
    deterministic ref paths. ``close()`` owns terminal publication; snapshot
    readers only verify the already committed terminal ref.
    """

    def __init__(self, root: str | Path, store: ArtifactStore, run_id: str) -> None:
        if _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("run_id must be a filesystem-safe stable identifier")
        self.root = Path(root)
        self.store = store
        self.run_id = run_id
        self.run_dir = self.root / "runs" / run_id
        self.events_dir = self.run_dir / "events"
        self.fences_dir = self.run_dir / "fences"
        self.quarantine_dir = self.run_dir / "quarantine"
        self.startup_quarantine_dir = self.run_dir / "startup-quarantine"
        for directory in (
            self.events_dir,
            self.fences_dir,
            self.quarantine_dir,
            self.startup_quarantine_dir,
        ):
            ArtifactStore.durable_mkdir(directory)
        self.lock_path = self.run_dir / "journal.lock"
        ArtifactStore.durable_touch(self.lock_path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self.lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def claim_epoch(self, epoch: int) -> Artifact:
        self._validate_epoch(epoch)
        with self._locked():
            return self._claim_epoch_unlocked(epoch)

    def reserve_identity(self, input_hash: str) -> Artifact:
        """CAS-reserve this run before any input artifact or fencing write."""

        self._validate_input_hash(input_hash)
        with self._locked():
            payload = self._identity_payload(input_hash)
            expected_hash = self._artifact_hash(
                "RunIdentityReservation",
                "1.0.0",
                payload,
            )
            identity_ref = self.run_dir / "identity.ref"
            if identity_ref.exists():
                committed_hash = self._read_ref(identity_ref)
                if committed_hash != expected_hash:
                    raise RunIdentityConflict("run_id is reserved for a different immutable input")
            else:
                self._publish_ref(identity_ref, expected_hash)

            events = self._events_unlocked()
            if events:
                first_details = events[0].payload.get("details")
                if (
                    events[0].payload.get("event_type") != "RUN_STARTED"
                    or not isinstance(first_details, dict)
                    or first_details.get("input_hash") != input_hash
                ):
                    raise RunJournalCorruption("committed start conflicts with run identity reservation")

            reservation_path = self.store.artifact_dir / f"{expected_hash}.json"
            if reservation_path.exists():
                reservation = self.store.read(
                    expected_hash,
                    expected_schema_name="RunIdentityReservation",
                )
            else:
                reservation = self.store.put(
                    "RunIdentityReservation",
                    "1.0.0",
                    payload,
                )
            if reservation.content_hash != expected_hash:
                raise RunJournalCorruption("run identity reservation hash does not match its ref")
            self._validate_identity_artifact(reservation, input_hash)
            return reservation

    def reserved_input_hash(self) -> str:
        """Read the verified TraceRunInput identity without repairing state."""

        with self._locked():
            reservation = self._identity_unlocked()
            input_hash = reservation.payload.get("input_hash")
            if not isinstance(input_hash, str):
                raise RunJournalCorruption("run identity reservation has no input hash")
            self._validate_input_hash(input_hash)
            self._validate_identity_artifact(reservation, input_hash)
            return input_hash

    def start_run(
        self,
        epoch: int,
        input_hash: str,
        details: Mapping[str, object],
        *,
        after_event_commit: Callable[[], None] | None = None,
    ) -> Artifact:
        """Commit the authoritative start event, then its derivative fence."""

        self._validate_epoch(epoch)
        self._validate_input_hash(input_hash)
        if details.get("input_hash") != input_hash:
            raise ValueError("RUN_STARTED details do not match the reserved input")
        with self._locked():
            reservation = self._identity_unlocked()
            self._validate_identity_artifact(reservation, input_hash)
            events = self._events_unlocked()
            if events:
                first = events[0]
                first_details = first.payload.get("details")
                if (
                    first.payload.get("event_type") != "RUN_STARTED"
                    or not isinstance(first_details, dict)
                    or first_details.get("input_hash") != input_hash
                ):
                    raise RunJournalCorruption("journal head does not match the reserved run identity")
                if events[-1].payload.get("event_type") != "RUN_CLOSED":
                    self._claim_epoch_unlocked(epoch)
                return first
            self._quarantine_orphan_fences_unlocked()
            started = self._append_unlocked(
                epoch,
                "RUN_STARTED",
                details,
                events,
            )
            if after_event_commit is not None:
                after_event_commit()
            self._claim_epoch_unlocked(epoch)
            return started

    def append(
        self,
        epoch: int,
        event_type: str,
        details: Mapping[str, object],
        *,
        expected_sequence: int | None = None,
        expected_previous_hash: str | None | object = _UNSET,
    ) -> Artifact:
        self._validate_epoch(epoch)
        if event_type == "RUN_CLOSED":
            raise ValueError("RUN_CLOSED can only be committed by close()")
        with self._locked():
            self._assert_current_epoch_unlocked(epoch)
            events = self._events_unlocked()
            self._assert_expected_head_unlocked(events, expected_sequence, expected_previous_hash)
            self._assert_open_unlocked(events)
            return self._append_unlocked(epoch, event_type, details, events)

    def close(
        self,
        epoch: int,
        *,
        status: str,
        reason_code: str,
        expected_sequence: int | None = None,
        expected_previous_hash: str | None | object = _UNSET,
    ) -> Artifact:
        self._validate_epoch(epoch)
        if status not in {"succeeded", "failed"}:
            raise ValueError("terminal status must be 'succeeded' or 'failed'")
        if not reason_code:
            raise ValueError("reason_code is required")
        with self._locked():
            self._assert_current_epoch_unlocked(epoch)
            events = self._events_unlocked()
            if events and events[-1].payload["event_type"] == "RUN_CLOSED":
                existing = self._closed_from_terminal_unlocked(events[-1])
                if existing.payload["status"] != status or existing.payload["reason_code"] != reason_code:
                    raise RunAlreadyClosed("run already has a different immutable terminal outcome")
                self._publish_ref(self.run_dir / "closed.ref", existing.content_hash)
                return existing
            self._assert_expected_head_unlocked(events, expected_sequence, expected_previous_hash)
            self._assert_open_unlocked(events)
            previous_hash = events[-1].content_hash if events else None
            terminal_sequence = len(events) + 1
            closed = self.store.put(
                "RunClosed",
                "1.0.0",
                {
                    "controller_epoch": epoch,
                    "previous_event_hash": previous_hash,
                    "reason_code": reason_code,
                    "run_id": self.run_id,
                    "status": status,
                    "terminal_sequence": terminal_sequence,
                },
            )
            terminal_event = self._append_unlocked(
                epoch,
                "RUN_CLOSED",
                {"run_closed_hash": closed.content_hash},
                events,
            )
            if terminal_event.payload["sequence"] != terminal_sequence:
                raise RunJournalCorruption("terminal event sequence changed during close")
            self._publish_ref(self.run_dir / "closed.ref", closed.content_hash)
            return closed

    def record_observation(
        self,
        epoch: int,
        observation_type: str,
        details: Mapping[str, object],
    ) -> ObservationReceipt:
        self._validate_epoch(epoch)
        with self._locked():
            events = self._events_unlocked()
            if not events or events[0].payload.get("event_type") != "RUN_STARTED":
                raise RunNotStarted("observations cannot be recorded before RUN_STARTED")
            self._assert_current_epoch_unlocked(epoch)
            observation = self.store.put(
                "Observation",
                "1.0.0",
                {
                    "details": dict(details),
                    "observation_type": observation_type,
                    "run_id": self.run_id,
                },
            )
            if events and events[-1].payload["event_type"] == "RUN_CLOSED":
                quarantine_ref = self.quarantine_dir / f"{observation.content_hash}.ref"
                self._publish_ref(quarantine_ref, observation.content_hash)
                return ObservationReceipt(
                    observation=observation,
                    quarantined=True,
                    quarantine_ref=quarantine_ref,
                )
            self._assert_open_unlocked(events)
            event = self._append_unlocked(
                epoch,
                "OBSERVATION_RECORDED",
                {"observation_hash": observation.content_hash},
                events,
            )
            return ObservationReceipt(
                observation=observation,
                quarantined=False,
                quarantine_ref=None,
                event=event,
            )

    def events(self) -> list[Artifact]:
        return self._events_unlocked()

    def closed(self) -> Artifact:
        events = self._events_unlocked()
        if not events or events[-1].payload["event_type"] != "RUN_CLOSED":
            raise RunJournalCorruption("run has no committed terminal event")
        closed = self._closed_from_terminal_unlocked(events[-1])
        closed_ref = self.run_dir / "closed.ref"
        if not closed_ref.exists() or self._read_ref(closed_ref) != closed.content_hash:
            raise RunJournalCorruption("closed ref is missing or conflicts with terminal event")
        return closed

    def quarantined_observations(self) -> list[ObservationReceipt]:
        receipts: list[ObservationReceipt] = []
        for ref_path in sorted(self.quarantine_dir.glob("*.ref")):
            observation = self.store.read(self._read_ref(ref_path), expected_schema_name="Observation")
            receipts.append(
                ObservationReceipt(
                    observation=observation,
                    quarantined=True,
                    quarantine_ref=ref_path,
                )
            )
        return receipts

    def verify(self) -> None:
        events = self._events_unlocked()
        self._current_epoch_unlocked()
        identity_ref = self.run_dir / "identity.ref"
        if identity_ref.exists():
            identity = self._identity_unlocked()
            if events:
                first_details = events[0].payload.get("details")
                if (
                    events[0].payload.get("event_type") != "RUN_STARTED"
                    or not isinstance(first_details, dict)
                    or first_details.get("input_hash") != identity.payload.get("input_hash")
                ):
                    raise RunJournalCorruption("RUN_STARTED does not match the run identity reservation")
        terminal_positions = [
            index for index, event in enumerate(events) if event.payload["event_type"] == "RUN_CLOSED"
        ]
        if terminal_positions and terminal_positions != [len(events) - 1]:
            raise RunJournalCorruption("terminal event is not the final event")
        closed_ref = self.run_dir / "closed.ref"
        if terminal_positions:
            closed = self._closed_from_terminal_unlocked(events[-1])
            if closed_ref.exists() and self._read_ref(closed_ref) != closed.content_hash:
                raise RunJournalCorruption("closed ref conflicts with terminal event")
        elif closed_ref.exists():
            raise RunJournalCorruption("closed ref exists without terminal event")
        for receipt in self.quarantined_observations():
            if receipt.observation.payload["run_id"] != self.run_id:
                raise RunJournalCorruption("quarantined observation belongs to another run")

    def _append_unlocked(
        self,
        epoch: int,
        event_type: str,
        details: Mapping[str, object],
        events: list[Artifact],
    ) -> Artifact:
        if not event_type:
            raise ValueError("event_type is required")
        sequence = len(events) + 1
        previous_hash = events[-1].content_hash if events else None
        event = self.store.put(
            "RunEvent",
            "1.0.0",
            {
                "controller_epoch": epoch,
                "details": dict(details),
                "event_type": event_type,
                "previous_event_hash": previous_hash,
                "run_id": self.run_id,
                "sequence": sequence,
            },
        )
        self._publish_ref(self.events_dir / f"{sequence:020d}.ref", event.content_hash)
        return event

    def _events_unlocked(self) -> list[Artifact]:
        events: list[Artifact] = []
        previous_hash: str | None = None
        refs = sorted(self.events_dir.glob("*.ref"))
        for expected_sequence, ref_path in enumerate(refs, start=1):
            if ref_path.name != f"{expected_sequence:020d}.ref":
                raise RunJournalCorruption("event sequence has a gap or unexpected ref")
            try:
                event = self.store.read(self._read_ref(ref_path), expected_schema_name="RunEvent")
            except ArtifactCorruption as error:
                raise RunJournalCorruption("event artifact is corrupt") from error
            payload = event.payload
            if (
                payload.get("run_id") != self.run_id
                or payload.get("sequence") != expected_sequence
                or payload.get("previous_event_hash") != previous_hash
                or not isinstance(payload.get("event_type"), str)
                or not isinstance(payload.get("controller_epoch"), int)
                or isinstance(payload.get("controller_epoch"), bool)
                or not 0 < cast(int, payload.get("controller_epoch")) <= MAX_SAFE_INTEGER
                or not isinstance(payload.get("details"), dict)
            ):
                raise RunJournalCorruption("event chain fields are invalid")
            events.append(event)
            previous_hash = event.content_hash
        return events

    def _current_epoch_unlocked(self) -> int | None:
        events = self._events_unlocked()
        start_epoch: int | None = None
        if events:
            first = events[0]
            first_epoch = first.payload.get("controller_epoch")
            if (
                first.payload.get("event_type") != "RUN_STARTED"
                or not isinstance(first_epoch, int)
                or isinstance(first_epoch, bool)
                or not 0 < first_epoch <= MAX_SAFE_INTEGER
            ):
                raise RunJournalCorruption("first event is not an authoritative RUN_STARTED fence")
            start_epoch = first_epoch
        claims = self._fence_claims_unlocked()
        quarantined_names = self._quarantined_fence_names_unlocked(claims)
        active_claims = [epoch for ref_path, epoch, _claim in claims if ref_path.name not in quarantined_names]
        if start_epoch is None:
            return None
        if active_claims and active_claims[0] < start_epoch:
            raise RunJournalCorruption("fencing claim predates authoritative RUN_STARTED")
        return max([start_epoch, *active_claims])

    def _fence_claims_unlocked(self) -> list[tuple[Path, int, Artifact]]:
        claims: list[tuple[Path, int, Artifact]] = []
        for ref_path in sorted(self.fences_dir.glob("*.ref")):
            try:
                epoch = int(ref_path.stem)
            except ValueError as error:
                raise RunJournalCorruption("invalid fencing ref name") from error
            if ref_path.name != f"{epoch:020d}.ref":
                raise RunJournalCorruption("invalid fencing ref name")
            claim = self.store.read(self._read_ref(ref_path), expected_schema_name="FenceClaim")
            claim_epoch = claim.payload.get("epoch")
            if (
                not isinstance(claim_epoch, int)
                or isinstance(claim_epoch, bool)
                or not 0 < claim_epoch <= MAX_SAFE_INTEGER
                or claim.payload != {"epoch": epoch, "run_id": self.run_id}
            ):
                raise RunJournalCorruption("fencing claim content does not match its ref")
            claims.append((ref_path, epoch, claim))
        epochs = [epoch for _ref_path, epoch, _claim in claims]
        if epochs != sorted(set(epochs)):
            raise RunJournalCorruption("fencing epochs are not unique and monotonic")
        return claims

    def _quarantine_payload(
        self,
        ref_path: Path,
        epoch: int,
        claim: Artifact,
    ) -> dict[str, object]:
        try:
            ref_bytes = ref_path.read_bytes()
        except OSError as error:
            raise RunJournalCorruption("cannot audit orphan fencing ref") from error
        return {
            "claim_hash": claim.content_hash,
            "epoch": epoch,
            "fence_ref_hash": sha256_hex(ref_bytes),
            "fence_ref_name": ref_path.name,
            "reason_code": "ORPHAN_FENCE_WITHOUT_RUN_STARTED",
            "run_id": self.run_id,
        }

    def _quarantined_fence_names_unlocked(
        self,
        claims: list[tuple[Path, int, Artifact]],
    ) -> set[str]:
        claim_by_name = {ref_path.name: (ref_path, epoch, claim) for ref_path, epoch, claim in claims}
        quarantined: set[str] = set()
        for quarantine_ref in sorted(self.startup_quarantine_dir.glob("*.ref")):
            claim_state = claim_by_name.get(quarantine_ref.name)
            if claim_state is None:
                raise RunJournalCorruption("startup quarantine refers to a missing fencing claim")
            ref_path, epoch, claim = claim_state
            quarantine = self.store.read(
                self._read_ref(quarantine_ref),
                expected_schema_name="StartupFenceQuarantine",
            )
            if quarantine.payload != self._quarantine_payload(ref_path, epoch, claim):
                raise RunJournalCorruption("startup fencing quarantine does not match preserved evidence")
            quarantined.add(quarantine_ref.name)
        return quarantined

    def _quarantine_orphan_fences_unlocked(self) -> None:
        if self._events_unlocked():
            raise RunJournalCorruption("startup fencing quarantine is only valid before RUN_STARTED")
        claims = self._fence_claims_unlocked()
        quarantined = self._quarantined_fence_names_unlocked(claims)
        for ref_path, epoch, claim in claims:
            if ref_path.name in quarantined:
                continue
            quarantine = self.store.put(
                "StartupFenceQuarantine",
                "1.0.0",
                self._quarantine_payload(ref_path, epoch, claim),
            )
            self._publish_ref(
                self.startup_quarantine_dir / ref_path.name,
                quarantine.content_hash,
            )

    def _claim_epoch_unlocked(self, epoch: int) -> Artifact:
        if not self._events_unlocked():
            raise RunNotStarted("fencing cannot be claimed before authoritative RUN_STARTED")
        current = self._current_epoch_unlocked()
        if current is not None and epoch < current:
            raise StaleFencingEpoch(f"fencing epoch {epoch} is stale; active epoch is {current}")
        ref_path = self.fences_dir / f"{epoch:020d}.ref"
        if current == epoch and ref_path.exists():
            return self.store.read(
                self._read_ref(ref_path),
                expected_schema_name="FenceClaim",
            )
        claim = self.store.put(
            "FenceClaim",
            "1.0.0",
            {"epoch": epoch, "run_id": self.run_id},
        )
        self._publish_ref(ref_path, claim.content_hash)
        return claim

    def _identity_unlocked(self) -> Artifact:
        identity_ref = self.run_dir / "identity.ref"
        if not identity_ref.exists():
            raise RunJournalCorruption("run identity reservation is missing")
        try:
            return self.store.read(
                self._read_ref(identity_ref),
                expected_schema_name="RunIdentityReservation",
            )
        except ArtifactCorruption as error:
            raise RunJournalCorruption("run identity reservation cannot be resolved") from error

    def _validate_identity_artifact(
        self,
        reservation: Artifact,
        expected_input_hash: str,
    ) -> None:
        if (
            reservation.schema_version != "1.0.0"
            or reservation.payload != self._identity_payload(expected_input_hash)
            or self._read_ref(self.run_dir / "identity.ref") != reservation.content_hash
        ):
            raise RunJournalCorruption("run identity reservation is invalid")

    def _identity_payload(self, input_hash: str) -> dict[str, object]:
        return {
            "input_hash": input_hash,
            "input_schema_name": "TraceRunInput",
            "input_schema_version": "1.0.0",
            "run_id": self.run_id,
        }

    @staticmethod
    def _artifact_hash(
        schema_name: str,
        schema_version: str,
        payload: Mapping[str, object],
    ) -> str:
        return sha256_hex(
            canonical_json_bytes(
                {
                    "payload": dict(payload),
                    "schema_name": schema_name,
                    "schema_version": schema_version,
                }
            )
        )

    @staticmethod
    def _validate_input_hash(input_hash: str) -> None:
        if _HASH.fullmatch(input_hash) is None:
            raise ValueError("input_hash must be a SHA-256 hex digest")

    def _assert_current_epoch_unlocked(self, epoch: int) -> None:
        self._validate_epoch(epoch)
        current = self._current_epoch_unlocked()
        if current is None or current != epoch:
            raise StaleFencingEpoch(f"fencing epoch {epoch} is stale; active epoch is {current}")

    @staticmethod
    def _validate_epoch(epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or not 0 < epoch <= MAX_SAFE_INTEGER:
            raise ValueError("fencing epoch must be a positive safe integer")

    def _assert_open_unlocked(self, events: list[Artifact]) -> None:
        if events and events[-1].payload["event_type"] == "RUN_CLOSED":
            raise RunAlreadyClosed("run has an immutable terminal event")
        if (self.run_dir / "closed.ref").exists():
            raise RunJournalCorruption("closed ref exists without a terminal event")

    @staticmethod
    def _assert_expected_head_unlocked(
        events: list[Artifact],
        expected_sequence: int | None,
        expected_previous_hash: str | None | object,
    ) -> None:
        if expected_sequence is None and expected_previous_hash is _UNSET:
            return
        actual_sequence = len(events) + 1
        actual_previous_hash = events[-1].content_hash if events else None
        if (
            expected_sequence != actual_sequence
            or expected_previous_hash is _UNSET
            or expected_previous_hash != actual_previous_hash
        ):
            raise JournalHeadConflict(
                "journal head changed: "
                f"expected sequence={expected_sequence}, previous={expected_previous_hash}; "
                f"actual sequence={actual_sequence}, previous={actual_previous_hash}"
            )

    def _closed_from_terminal_unlocked(self, terminal_event: Artifact) -> Artifact:
        details = terminal_event.payload.get("details")
        if not isinstance(details, dict):
            raise RunJournalCorruption("terminal event details are invalid")
        closed_hash = details.get("run_closed_hash")
        if not isinstance(closed_hash, str):
            raise RunJournalCorruption("terminal event does not reference RunClosed")
        closed = self.store.read(closed_hash, expected_schema_name="RunClosed")
        if (
            closed.payload.get("run_id") != self.run_id
            or closed.payload.get("terminal_sequence") != terminal_event.payload["sequence"]
            or closed.payload.get("previous_event_hash") != terminal_event.payload["previous_event_hash"]
            or closed.payload.get("controller_epoch") != terminal_event.payload["controller_epoch"]
        ):
            raise RunJournalCorruption("RunClosed does not match its terminal event")
        return closed

    @staticmethod
    def _publish_ref(path: Path, content_hash: str) -> None:
        if _HASH.fullmatch(content_hash) is None:
            raise RunJournalCorruption("cannot publish an invalid artifact hash")
        ArtifactStore._publish(path, f"{content_hash}\n".encode("ascii"))

    @staticmethod
    def _read_ref(path: Path) -> str:
        try:
            content = path.read_bytes()
        except OSError as error:
            raise RunJournalCorruption(f"cannot read immutable ref {path}") from error
        if len(content) != 65 or not content.endswith(b"\n"):
            raise RunJournalCorruption(f"invalid immutable ref {path}")
        try:
            content_hash = content[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise RunJournalCorruption(f"invalid immutable ref {path}") from error
        if _HASH.fullmatch(content_hash) is None:
            raise RunJournalCorruption(f"invalid immutable ref {path}")
        return cast(str, content_hash)
