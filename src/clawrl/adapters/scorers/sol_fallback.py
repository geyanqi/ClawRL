"""Stateful deterministic Sol fallback evidence boundary for Ticket 06."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from clawrl.artifacts import Artifact, ArtifactStore, canonical_json_bytes, sha256_hex
from clawrl.judge.luna4_models import SolEvidenceMode, classify_sol_evidence


class SolFallbackBoundaryError(RuntimeError):
    """Sol fallback boundary evidence is unavailable, corrupt, or conflicted."""


class FixtureSolFallbackEvidenceAdapter:
    """Replace only the external Sol fallback re-evaluation boundary."""

    def __init__(self, root: str | Path, store: ArtifactStore) -> None:
        self.root = Path(root)
        self.store = store
        self.boundary_root = self.root / "boundaries" / "sol-fallback-evidence"
        ArtifactStore.durable_mkdir(self.boundary_root)
        self.lock_path = self.boundary_root / "adapter.lock"
        ArtifactStore.durable_touch(self.lock_path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self.lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def execute(self, *, run_id: str, teacher_label_set: Artifact, mode: SolEvidenceMode) -> tuple[Artifact, Artifact]:
        scalars = self._scalars(teacher_label_set)
        schedule_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "boundary": "sol_fallback_evidence",
                    "directive": mode,
                    "version": "fixture-sol-fallback-schedule/1.0.0",
                }
            )
        )
        request = self.store.put(
            "SolFallbackBoundaryRequest",
            "1.0.0",
            {
                "fault_directive": mode,
                "label_set_hash": teacher_label_set.content_hash,
                "requested_contract": "calibrated_scalar",
                "run_id": run_id,
                "schedule_hash": schedule_hash,
            },
        )
        classification = classify_sol_evidence(scalars, mode)
        observation = self.store.put(
            "SolFallbackBoundaryObservation",
            "1.0.0",
            {
                "classification": classification,
                "label_set_hash": teacher_label_set.content_hash,
                "request_hash": request.content_hash,
                "schedule_hash": schedule_hash,
                "status": "succeeded" if classification["calibrated_scalar_valid"] else "contract_invalid",
            },
        )
        run_root = self.boundary_root / run_id
        ArtifactStore.durable_mkdir(run_root)
        ref = run_root / "observation.ref"
        with self._locked():
            expected_ref = f"{observation.content_hash}\n".encode("ascii")
            if ref.exists():
                try:
                    committed_ref = ref.read_bytes()
                except OSError as error:
                    raise SolFallbackBoundaryError("Sol fallback observation ref is unavailable") from error
                if committed_ref != expected_ref:
                    raise SolFallbackBoundaryError("Sol fallback boundary input conflicts for stable run identity")
            else:
                ArtifactStore._publish(ref, expected_ref)
        self.verify(request=request, observation=observation, teacher_label_set=teacher_label_set, mode=mode)
        return request, observation

    def verify(
        self,
        *,
        request: Artifact,
        observation: Artifact,
        teacher_label_set: Artifact,
        mode: SolEvidenceMode,
    ) -> None:
        expected = classify_sol_evidence(self._scalars(teacher_label_set), mode)
        if (
            request.schema_name != "SolFallbackBoundaryRequest"
            or request.payload.get("label_set_hash") != teacher_label_set.content_hash
            or request.payload.get("fault_directive") != mode
            or request.payload.get("requested_contract") != "calibrated_scalar"
            or observation.schema_name != "SolFallbackBoundaryObservation"
            or observation.payload.get("request_hash") != request.content_hash
            or observation.payload.get("label_set_hash") != teacher_label_set.content_hash
            or observation.payload.get("schedule_hash") != request.payload.get("schedule_hash")
            or observation.payload.get("classification") != expected
            or observation.payload.get("status")
            != ("succeeded" if expected["calibrated_scalar_valid"] else "contract_invalid")
        ):
            raise SolFallbackBoundaryError("Sol fallback evidence cannot be freshly recertified")
        ref = self.boundary_root / cast(str, request.payload["run_id"]) / "observation.ref"
        try:
            ref_bytes = ref.read_bytes()
        except OSError as error:
            raise SolFallbackBoundaryError("Sol fallback committed observation ref is missing") from error
        if ref_bytes != f"{observation.content_hash}\n".encode("ascii"):
            raise SolFallbackBoundaryError("Sol fallback committed observation ref changed")

    @staticmethod
    def _scalars(teacher_label_set: Artifact) -> list[int]:
        if teacher_label_set.schema_name != "TeacherLabelSet":
            raise SolFallbackBoundaryError("Sol fallback requires a TeacherLabelSet")
        labels = teacher_label_set.payload.get("labels")
        if not isinstance(labels, list):
            return []
        return [
            cast(int, item["scalar_micros"])
            for item in labels
            if isinstance(item, dict) and type(item.get("scalar_micros")) is int
        ]
