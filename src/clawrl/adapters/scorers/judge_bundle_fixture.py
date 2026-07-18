"""Stateful deterministic terminal-certification source for Ticket 08 fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from clawrl.artifacts import Artifact, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.data.validation import load_training_dataset

FixtureTerminalVariant = Literal[
    "valid",
    "valid_alternate",
    "uncertifiable",
    "reward_mismatch",
    "scalarizer_mismatch",
    "scalar_incomparable",
    "scalar_nonfinite",
    "group_relative_all_low",
]


class FixtureJudgeTerminalSourceError(RuntimeError):
    """The terminal certification simulator is conflicted or malformed."""


class FixtureJudgeTerminalSource:
    """Materialize prior-ticket terminal evidence without bypassing bundle compilation."""

    def __init__(
        self,
        root: str | Path,
        *,
        dataset_version_hash: str,
        reward_schema_hash: str,
        scalarizer_hash: str,
        algorithm_contract_hash: str,
    ) -> None:
        self.root = Path(root)
        self.store = ArtifactStore(root)
        try:
            self.dataset = load_training_dataset(self.store, dataset_version_hash)
            self.store.read(reward_schema_hash, expected_schema_name="RewardSchema")
            self.store.read(scalarizer_hash, expected_schema_name="Scalarizer")
            self.store.read(algorithm_contract_hash, expected_schema_name="RLAlgorithmContract")
        except Exception as error:
            raise FixtureJudgeTerminalSourceError("fixture terminal source input is invalid") from error
        self.dataset_version_hash = dataset_version_hash
        self.reward_schema_hash = reward_schema_hash
        self.scalarizer_hash = scalarizer_hash
        self.algorithm_contract_hash = algorithm_contract_hash
        self.boundary = self.root / "boundaries" / "judge-terminal-source"
        ArtifactStore.durable_mkdir(self.boundary)

    def materialize(self, trace_id: str, *, variant: FixtureTerminalVariant = "valid") -> Artifact:
        traces = {cast(str, item.payload["trace_id"]): item for item in self.dataset.training_traces}
        if trace_id not in traces or variant not in {
            "valid",
            "valid_alternate",
            "uncertifiable",
            "reward_mismatch",
            "scalarizer_mismatch",
            "scalar_incomparable",
            "scalar_nonfinite",
            "group_relative_all_low",
        }:
            raise FixtureJudgeTerminalSourceError("terminal source request is outside the dataset contract")
        request = self.store.put(
            "FixtureJudgeTerminalRequest",
            "1.0.0",
            {
                "dataset_version_hash": self.dataset_version_hash,
                "trace_id": trace_id,
                "variant": variant,
            },
        )
        ref = self.boundary / trace_id / f"{variant}.ref"
        if ref.exists():
            digest = ref.read_text(encoding="ascii").strip()
            attempt = self.store.read(digest, expected_schema_name="FixtureJudgeTerminalAttempt")
            if attempt.payload.get("request_hash") != request.content_hash:
                raise FixtureJudgeTerminalSourceError("terminal source attempt conflicts with request")
            return self.store.read(
                cast(str, attempt.payload["outcome_hash"]), expected_schema_name="TraceCertificationOutcome"
            )
        outcome = self._build_outcome(traces[trace_id], variant)
        attempt = self.store.put(
            "FixtureJudgeTerminalAttempt",
            "1.0.0",
            {
                "outcome_hash": outcome.content_hash,
                "request_hash": request.content_hash,
                "status": "committed",
                "trace_id": trace_id,
                "variant": variant,
            },
        )
        ArtifactStore.durable_mkdir(ref.parent)
        ArtifactStore._publish(ref, f"{attempt.content_hash}\n".encode("ascii"))
        return outcome

    def _build_outcome(self, trace: Artifact, variant: FixtureTerminalVariant) -> Artifact:
        trace_id = cast(str, trace.payload["trace_id"])
        fit_refs: list[dict[str, JsonValue]] = []
        labels: list[dict[str, JsonValue]] = []
        for index in range(32):
            response = (
                f"Fixture policy response {index} for {trace_id}; content-derived token "
                f"{sha256_hex(canonical_json_bytes({'trace': trace_id, 'index': index}))[:24]}."
            )
            content = self.store.put(
                "TrajectoryContent",
                "1.0.0",
                {
                    "prompt_hash": sha256_hex(cast(str, trace.payload["prompt"]).encode()),
                    "response": response,
                    "response_hash": sha256_hex(response.encode()),
                    "trace_id": trace_id,
                    "trajectory_index": index,
                },
            )
            manifest = self.store.put(
                "TrajectoryManifest",
                "1.0.0",
                {
                    "content_hash": content.content_hash,
                    "dataset_version_hash": self.dataset_version_hash,
                    "split": "fit",
                    "trace_id": trace_id,
                    "trajectory_index": index,
                },
            )
            fit_refs.append({"manifest_hash": manifest.content_hash, "trajectory_index": index})
            scalar = 20_000_000 + int(content.content_hash[:8], 16) % 70_000_001
            labels.append(
                {
                    "evidence_hash": sha256_hex(
                        canonical_json_bytes({"content": content.content_hash, "scalar": scalar})
                    ),
                    "scalar_micros": scalar,
                    "trajectory_manifest_hash": manifest.content_hash,
                }
            )
        fit = self.store.put(
            "FitTrajectorySet",
            "1.0.0",
            {
                "dataset_version_hash": self.dataset_version_hash,
                "item_count": 32,
                "trace_id": trace_id,
                "trajectory_refs": fit_refs,
            },
        )
        teacher = self.store.put(
            "TeacherLabelSet",
            "1.0.0",
            {
                "fit_trajectory_set_hash": fit.content_hash,
                "label_count": 32,
                "labels": labels,
                "reward_schema_hash": self.reward_schema_hash,
                "scalarizer_hash": self.scalarizer_hash,
                "trace_id": trace_id,
            },
        )
        if variant in {"uncertifiable", "group_relative_all_low"}:
            group_relative = variant == "group_relative_all_low"
            evidence = self.store.put(
                "SolFallbackEvidence",
                "1.0.0",
                {
                    "all_low_count": 32 if group_relative else 0,
                    "calibrated_scalar_available": not group_relative,
                    "fit_trajectory_set_hash": fit.content_hash,
                    "item_count": 32,
                    "relative_order_available": group_relative,
                    "scalar_micros": [] if group_relative else [50_000_000] * 32,
                    "teacher_label_set_hash": teacher.content_hash,
                    "trace_id": trace_id,
                },
            )
            report = self.store.put(
                "UncertifiableJudgeReport",
                "1.0.0",
                {
                    "blocks_bundle_publication": True,
                    "diagnostic_only": group_relative,
                    "fit_trajectory_set_hash": fit.content_hash,
                    "reason_code": (
                        "SOL_GROUP_RELATIVE_ONLY_UNSUPPORTED_V1"
                        if variant == "group_relative_all_low"
                        else "SOL_CALIBRATED_SCALAR_UNCERTIFIABLE"
                    ),
                    "sol_fallback_evidence_hash": evidence.content_hash,
                    "status": "uncertifiable",
                    "teacher_label_set_hash": teacher.content_hash,
                    "trace_id": trace_id,
                    "training_authorized": False,
                },
            )
            return self.store.put(
                "TraceCertificationOutcome",
                "1.0.0",
                {"artifact_hash": report.content_hash, "kind": "uncertifiable", "trace_id": trace_id},
            )
        index = [cast(str, item.payload["trace_id"]) for item in self.dataset.training_traces].index(trace_id)
        scorer_tier = "sol" if index % 10 == 0 else "luna"
        golden: Artifact | None = None
        if scorer_tier == "sol":
            golden = self.store.put(
                "GoldenHardEntry",
                "1.0.0",
                {
                    "reason_tags": ["LUNA8_AND_LUNA4_EXHAUSTED", "CALIBRATED_SOL_FALLBACK"],
                    "status": "active",
                    "trace_id": trace_id,
                },
            )
        mismatch_reward: Artifact | None = None
        if variant == "reward_mismatch":
            mismatch_reward = self.store.put(
                "RewardSchema",
                "1.0.0",
                {"aggregation": "calibrated_scalar", "maximum_micros": 90_000_000, "minimum_micros": 0},
            )
        mismatch_scalarizer: Artifact | None = None
        if variant == "scalarizer_mismatch":
            mismatch_scalarizer = self.store.put(
                "Scalarizer",
                "1.0.0",
                {
                    "dimension_weights_micros": {
                        "correctness": 400_000,
                        "reasoning_quality": 200_000,
                        "task_completion": 200_000,
                        "tool_discipline": 200_000,
                    },
                    "scalarizer_id": "fixture-mismatched-dimension-mean-v1",
                    "schema_version": "scalarizer/1.0.0",
                },
            )
        terminal = self.store.put(
            "TraceCertificationTerminal",
            "1.0.0",
            {
                "fit_trajectory_set_hash": fit.content_hash,
                "scorer_tier": scorer_tier,
                "status": "succeeded",
                "teacher_label_set_hash": teacher.content_hash,
                "trace_id": trace_id,
            },
        )
        pack_payload: dict[str, JsonValue] = {
            "aggregation": "calibrated_scalar",
            "algorithm_contract_hash": self.algorithm_contract_hash,
            "calibrated_training_authorized": True,
            "certification_level": 16 if variant == "valid_alternate" else 8 if scorer_tier == "luna" else 0,
            "certification_mode": "student_holdout" if scorer_tier == "luna" else "teacher_fallback",
            "dataset_version_hash": self.dataset_version_hash,
            "dataset_version_id": self.dataset.dataset_version.payload["dataset_version_id"],
            "fit_trajectory_set_hash": fit.content_hash,
            "golden_hard_entry_hash": golden.content_hash if golden else None,
            "items_per_turn": 16 if variant == "valid_alternate" else 8 if scorer_tier == "luna" else 4,
            "local_tie_groups_diagnostic_only": True,
            "reward_schema_hash": (
                mismatch_reward.content_hash if mismatch_reward is not None else self.reward_schema_hash
            ),
            "scalar_comparability": {
                "finite": variant != "scalar_nonfinite",
                "maximum_micros": 100_000_000,
                "minimum_micros": 0,
                "scale_id": (
                    "incomparable-local-scale-v1" if variant == "scalar_incomparable" else "sol-calibrated-scalar-v1"
                ),
            },
            "scalarizer_hash": (
                mismatch_scalarizer.content_hash if mismatch_scalarizer is not None else self.scalarizer_hash
            ),
            "scorer_tier": scorer_tier,
            "status": "terminal",
            "teacher_label_set_hash": teacher.content_hash,
            "terminal_certification_hash": terminal.content_hash,
            "trace_id": trace_id,
            "training_trace_hash": trace.content_hash,
        }
        pack = self.store.put("JudgePack", "1.0.0", pack_payload)
        return self.store.put(
            "TraceCertificationOutcome",
            "1.0.0",
            {"artifact_hash": pack.content_hash, "kind": "judge_pack", "trace_id": trace_id},
        )
