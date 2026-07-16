"""Fenced append-only governed data-ingestion workflow."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from clawrl.adapters.sources.fixture import (
    DataProviderIngressReceipt,
    DataProviderPublicContract,
    DataSourceBoundaryError,
    FixtureDataSource,
    QueryResult,
)
from clawrl.artifacts import (
    Artifact,
    ArtifactCorruption,
    ArtifactStore,
    JsonValue,
    canonical_json_bytes,
    sha256_hex,
)
from clawrl.data.domain import dataset_version_id, selection_score, trace_set_hash, training_trace_payload
from clawrl.data.models import (
    DataContractError,
    DatasetValidationError,
    DataSourceSkill,
    FixtureDataIngestConfig,
    ProductionDataIngestConfig,
)
from clawrl.data.safety import assert_public_sentinel_free
from clawrl.data.sql import SelectAst, parse_and_validate_select
from clawrl.data.validation import (
    expected_dedupe_manifest,
    expected_selection_manifest,
    expected_validation_manifest,
    load_training_dataset,
)
from clawrl.training.run_journal import (
    JournalHeadConflict,
    RunIdentityConflict,
    RunJournal,
)

_INPUT = "DataIngestWorkflowInput"
_EVENTS = (
    "RUN_STARTED",
    "QUERY_PLANNED",
    "QUERY_REQUESTED",
    "CANDIDATES_SANITIZED",
    "ROWS_VALIDATED",
    "ROWS_DEDUPED",
    "BADCASES_SELECTED",
    "DATASET_PUBLISHED",
    "DECISION_RECORDED",
    "RUN_CLOSED",
)
_KNOWN_BOUNDARY_REASON_CODES = frozenset(
    {
        "QUERY_BYTE_LIMIT_EXCEEDED",
        "QUERY_CARDINALITY_INVALID",
        "QUERY_ROW_LIMIT_EXCEEDED",
        "QUERY_TIME_LIMIT_EXCEEDED",
        "RAW_FIELD_COMPLETENESS_FAILED",
        "RETRY_POLICY_EXHAUSTED",
        "SENTINEL_LEAK_DETECTED",
        "SENTINEL_VALIDATION_FAILED",
    }
)


class DataIngestWorkflowError(RuntimeError):
    """The governed workflow cannot verify or advance its durable state."""


class InjectedDataIngestCrash(RuntimeError):
    """Test-only process interruption after a durable boundary commit."""


@dataclass(frozen=True, slots=True)
class DataIngestSnapshot:
    events: tuple[Artifact, ...]
    dataset_version: Artifact | None
    decision_record: Artifact | None
    terminal: bool


@dataclass(frozen=True, slots=True)
class _PreparedInput:
    public_contract: DataProviderPublicContract
    effective_skill: DataSourceSkill
    effective_skill_hash: str
    input_payload: dict[str, object]
    input_hash: str


class GovernedDataIngestWorkflow:
    """One durable application transition per ``resume`` call."""

    @staticmethod
    def production_readiness(root: str | Path, config: object) -> Artifact:
        """Emit a sanitized DATA_INGEST blocker before any adapter/run construction."""

        if type(config) is ProductionDataIngestConfig:
            try:
                payload = cast(ProductionDataIngestConfig, config).readiness_payload()
            except (AttributeError, TypeError, ValueError, UnicodeError):
                payload = {
                    "checks": [
                        {"code": "INVALID_PRODUCTION_CONFIGURATION", "status": "blocked"},
                        {"code": "PRODUCTION_DATA_SOURCE_CONTRACT_UNAVAILABLE", "status": "blocked"},
                    ],
                    "execution_profile": "production",
                    "phase": "DATA_INGEST",
                    "side_effects_permitted": False,
                    "status": "blocked",
                }
        else:
            payload = {
                "checks": [
                    {"code": "INVALID_PRODUCTION_CONFIGURATION", "status": "blocked"},
                    {"code": "PRODUCTION_DATA_SOURCE_CONTRACT_UNAVAILABLE", "status": "blocked"},
                ],
                "execution_profile": "production",
                "phase": "DATA_INGEST",
                "side_effects_permitted": False,
                "status": "blocked",
            }
        return ArtifactStore(root).put("ReadinessReport", "1.0.0", payload)

    @staticmethod
    def _journal(root: Path, store: ArtifactStore, run_id: str) -> RunJournal:
        return RunJournal(root, store, run_id, input_schema_name=_INPUT, input_schema_version="1.0.0")

    @classmethod
    def bootstrap(
        cls,
        root: str | Path,
        config: FixtureDataIngestConfig,
        *,
        role_ingress: DataProviderIngressReceipt,
        epoch: int,
    ) -> DataIngestSnapshot:
        root_path = Path(root)
        try:
            prepared = cls._prepare_input(root_path, config, role_ingress)
        except Exception as error:
            if (root_path / "runs" / config.run_id / "identity.ref").exists():
                raise DataIngestWorkflowError("run_id is bound to a different immutable ingest input") from error
            raise
        store = ArtifactStore(root_path)
        journal = cls._journal(root_path, store, config.run_id)
        try:
            journal.reserve_identity(prepared.input_hash)
        except (RunIdentityConflict, RuntimeError) as error:
            raise DataIngestWorkflowError("run_id is bound to a different immutable ingest input") from error
        input_path = store.artifact_dir / f"{prepared.input_hash}.json"
        if input_path.exists():
            persisted = store.read(prepared.input_hash, expected_schema_name=_INPUT)
            if persisted.payload != prepared.input_payload:
                raise DataIngestWorkflowError("persisted ingest input conflicts with its reserved identity")
            if journal.events():
                return cls._snapshot(store, journal)
        adapter = FixtureDataSource.bootstrap(root_path, role_ingress, config)
        workflow_input = store.put(
            _INPUT,
            "1.0.0",
            prepared.input_payload,
        )
        if (
            workflow_input.content_hash != prepared.input_hash
            or adapter.blueprint_artifact.content_hash != role_ingress.blueprint_hash
            or adapter.skill_artifact.content_hash != prepared.effective_skill_hash
            or adapter.role_audit_artifact.content_hash != role_ingress.role_invocation_audit_hash
            or adapter.correction_role_audit_artifact.content_hash != role_ingress.correction_role_invocation_audit_hash
        ):
            raise DataIngestWorkflowError("adapter artifacts do not match the reserved ingest identity")
        journal.start_run(
            epoch,
            workflow_input.content_hash,
            {"input_hash": workflow_input.content_hash, "purpose": "training_allowed"},
        )
        return cls._snapshot(store, journal)

    @classmethod
    def _prepare_input(
        cls,
        root: Path,
        config: FixtureDataIngestConfig,
        role_ingress: DataProviderIngressReceipt,
    ) -> _PreparedInput:
        assert_public_sentinel_free(
            {
                "config": config.artifact_payload(),
                "role_ingress": role_ingress.artifact_payload(),
            }
        )
        contract = FixtureDataSource.load_public_contract(root, role_ingress)
        effective_skill = FixtureDataSource.effective_skill(contract.base_skill, config)
        parse_and_validate_select(config.query_sql, effective_skill)
        skill_hash = cls._artifact_hash("DataSourceSkill", effective_skill.artifact_payload())
        input_payload: dict[str, object] = {
            "blueprint_hash": role_ingress.blueprint_hash,
            "config": config.artifact_payload(),
            "data_source_skill_hash": skill_hash,
            "data_provider_ingress_hash": role_ingress.ingress_hash,
            "input_packet_hash": role_ingress.input_packet_hash,
            "raw_role_output_hash": role_ingress.raw_output_hash,
            "normalized_role_output_hash": role_ingress.normalized_output_hash,
            "role_invocation_audit_hash": role_ingress.role_invocation_audit_hash,
            "correction_input_packet_hash": role_ingress.correction_input_packet_hash,
            "correction_raw_role_output_hash": role_ingress.correction_raw_output_hash,
            "correction_normalized_role_output_hash": role_ingress.correction_normalized_output_hash,
            "correction_role_invocation_audit_hash": role_ingress.correction_role_invocation_audit_hash,
            "run_id": config.run_id,
        }
        assert_public_sentinel_free(input_payload)
        return _PreparedInput(
            public_contract=contract,
            effective_skill=effective_skill,
            effective_skill_hash=skill_hash,
            input_payload=input_payload,
            input_hash=cls._artifact_hash(_INPUT, input_payload),
        )

    @staticmethod
    def _artifact_hash(schema_name: str, payload: Mapping[str, object]) -> str:
        return sha256_hex(
            canonical_json_bytes(
                {
                    "payload": dict(payload),
                    "schema_name": schema_name,
                    "schema_version": "1.0.0",
                }
            )
        )

    @staticmethod
    def _is_sha256(value: object) -> bool:
        if type(value) is not str:
            return False
        text = cast(str, value)
        return len(text) == 64 and all(character in "0123456789abcdef" for character in text)

    @classmethod
    def resume(
        cls,
        root: str | Path,
        run_id: str,
        *,
        epoch: int,
        crash_after: str | None = None,
    ) -> DataIngestSnapshot:
        root_path = Path(root)
        store = ArtifactStore(root_path)
        cls._verify_public_sentinel_closure(root_path)
        journal = cls._journal(root_path, store, run_id)
        item, config = cls._input(store, journal, run_id)
        events = journal.events()
        cls._verify_application_events(events, expected_input_hash=item.content_hash)
        if events[-1].payload["event_type"] == "RUN_CLOSED":
            cls._verify_terminal_decision_binding(store, events)
            lifecycle_events = [event for event in events if event.payload.get("event_type") != "OBSERVATION_RECORDED"]
            adapter = FixtureDataSource.restore_read_only(
                root_path,
                expected_blueprint_hash=cast(str, item.payload["blueprint_hash"]),
                expected_role_audit_hash=cast(str, item.payload["role_invocation_audit_hash"]),
                expected_correction_role_audit_hash=cast(
                    str,
                    item.payload["correction_role_invocation_audit_hash"],
                ),
                expected_skill_hash=cast(str, item.payload["data_source_skill_hash"]),
                expected_config=config,
            )
            cls._verify_committed_public_state(store, adapter, config, events)
            cls._verify_query_attempt_evidence(store, adapter, config, lifecycle_events)
            cls._verify_committed_boundary_result(store, adapter, events)
            return cls._snapshot(store, journal)
        journal.claim_epoch(epoch)
        lifecycle_events = [event for event in events if event.payload.get("event_type") != "OBSERVATION_RECORDED"]
        if lifecycle_events[-1].payload.get("event_type") == "INGEST_FAILURE_RECORDED":
            has_query_evidence = any(
                event.payload.get("event_type")
                in {
                    "QUERY_REQUESTED",
                    "QUERY_RETRY_SCHEDULED",
                    "QUERY_LATE_QUARANTINED",
                    "CANDIDATES_SANITIZED",
                }
                for event in lifecycle_events
            )
            if has_query_evidence:
                try:
                    adapter = FixtureDataSource.restore_read_only(
                        root_path,
                        expected_blueprint_hash=cast(str, item.payload["blueprint_hash"]),
                        expected_role_audit_hash=cast(str, item.payload["role_invocation_audit_hash"]),
                        expected_correction_role_audit_hash=cast(
                            str,
                            item.payload["correction_role_invocation_audit_hash"],
                        ),
                        expected_skill_hash=cast(str, item.payload["data_source_skill_hash"]),
                        expected_config=config,
                    )
                    cls._verify_committed_public_state(store, adapter, config, events)
                    cls._verify_query_attempt_evidence(store, adapter, config, lifecycle_events)
                    cls._verify_committed_boundary_result(store, adapter, events)
                except DataIngestWorkflowError:
                    raise
                except Exception as error:
                    raise DataIngestWorkflowError("failed-run evidence is invalid") from error
            decision = cls._ref(
                store,
                lifecycle_events[-1],
                "decision_record_hash",
                "DecisionRecord",
            )
            journal.close(
                epoch,
                status="failed",
                reason_code=cast(str, decision.payload["reason_code"]),
                expected_sequence=len(events) + 1,
                expected_previous_hash=events[-1].content_hash,
            )
            return cls._snapshot(store, journal)
        try:
            adapter = FixtureDataSource.open(
                root_path,
                expected_blueprint_hash=cast(str, item.payload["blueprint_hash"]),
                expected_role_audit_hash=cast(str, item.payload["role_invocation_audit_hash"]),
                expected_correction_role_audit_hash=cast(
                    str,
                    item.payload["correction_role_invocation_audit_hash"],
                ),
                expected_skill_hash=cast(str, item.payload["data_source_skill_hash"]),
                expected_config=config,
            )
        except Exception:
            cls._record_failure(
                store,
                journal,
                epoch,
                events,
                failed_stage="BOUNDARY_FACTORY",
                reason_code="DATA_SOURCE_BOUNDARY_FACTORY_FAILED",
            )
            return cls._snapshot(store, journal)
        try:
            cls._verify_committed_public_state(store, adapter, config, events)
        except DataIngestWorkflowError:
            raise
        except Exception as error:
            raise DataIngestWorkflowError("committed ingest artifact graph is invalid") from error
        try:
            cls._verify_query_attempt_evidence(store, adapter, config, lifecycle_events)
        except (DataSourceBoundaryError, ArtifactCorruption, RuntimeError) as error:
            if lifecycle_events[-1].payload.get("event_type") == "INGEST_FAILURE_RECORDED":
                raise DataIngestWorkflowError("committed query evidence is invalid") from error
            cls._record_failure(
                store,
                journal,
                epoch,
                events,
                failed_stage="QUERY_EVIDENCE_VERIFY",
                reason_code="QUERY_EVIDENCE_INTEGRITY_FAILED",
            )
            return cls._snapshot(store, journal)
        try:
            cls._advance(store, journal, adapter, config, epoch, crash_after=crash_after)
        except JournalHeadConflict:
            # Another session committed the identical logical transition first.
            # The durable journal is authoritative; losing speculative artifacts
            # are content-identical and therefore have no orphan identity.
            pass
        return cls._snapshot(store, journal)

    @classmethod
    def run_once(
        cls,
        root: str | Path,
        config: FixtureDataIngestConfig,
        *,
        role_ingress: DataProviderIngressReceipt,
        epoch: int,
    ) -> DataIngestSnapshot:
        root_path = Path(root)
        if (root_path / "runs" / config.run_id / "identity.ref").exists():
            store = ArtifactStore(root_path)
            journal = cls._journal(root_path, store, config.run_id)
            try:
                prepared = cls._prepare_input(root_path, config, role_ingress)
            except Exception as error:
                raise RunIdentityConflict("supplied ingest input does not match the reserved run") from error
            if journal.reserved_input_hash() != prepared.input_hash:
                raise RunIdentityConflict("supplied ingest input does not match the reserved run")
            snapshot = cls.resume(root_path, config.run_id, epoch=epoch)
        else:
            snapshot = cls.bootstrap(
                root_path,
                config,
                role_ingress=role_ingress,
                epoch=epoch,
            )
        while not snapshot.terminal:
            snapshot = cls.resume(root_path, config.run_id, epoch=epoch)
        return snapshot

    @classmethod
    def _input(cls, store: ArtifactStore, journal: RunJournal, run_id: str) -> tuple[Artifact, FixtureDataIngestConfig]:
        item = store.read(journal.reserved_input_hash(), expected_schema_name=_INPUT)
        config_value = item.payload.get("config")
        if item.schema_version != "1.0.0" or item.payload.get("run_id") != run_id or not isinstance(config_value, dict):
            raise DataIngestWorkflowError("persisted ingest input is invalid")
        config = FixtureDataIngestConfig.from_mapping(cast(dict[str, object], config_value))
        return item, config

    @classmethod
    def _advance(
        cls,
        store: ArtifactStore,
        journal: RunJournal,
        adapter: FixtureDataSource,
        config: FixtureDataIngestConfig,
        epoch: int,
        *,
        crash_after: str | None,
    ) -> None:
        events = journal.events()
        lifecycle_events = [event for event in events if event.payload.get("event_type") != "OBSERVATION_RECORDED"]
        types = tuple(cast(str, event.payload["event_type"]) for event in lifecycle_events)
        cls._verify_event_types(types)
        head = lifecycle_events[-1]
        journal_head = events[-1]
        current = cast(str, head.payload["event_type"])
        if current == "RUN_CLOSED":
            # A concurrent session may close the run after this caller's initial
            # snapshot.  Terminal state is immutable, so replay is a no-op.
            return
        if current in {
            "BADCASES_SELECTED",
            "CANDIDATES_SANITIZED",
            "ROWS_DEDUPED",
            "ROWS_VALIDATED",
        }:
            try:
                cls._verify_committed_boundary_result(store, adapter, lifecycle_events)
            except DataIngestWorkflowError:
                # A committed candidate graph that no longer matches its private
                # receipt is persisted-artifact substitution, not a source fault.
                raise
            except (DataSourceBoundaryError, ArtifactCorruption, RuntimeError):
                cls._record_failure(
                    store,
                    journal,
                    epoch,
                    events,
                    failed_stage="QUERY_RESULT_VERIFY",
                    reason_code="QUERY_RESULT_INTEGRITY_FAILED",
                )
                return
        details: dict[str, object]
        next_event: str
        if current == "RUN_STARTED":
            artifact = cls._plan(store, adapter, config)
            next_event, details = "QUERY_PLANNED", {"query_plan_hash": artifact.content_hash}
        elif current == "QUERY_PLANNED":
            plan = cls._ref(store, head, "query_plan_hash", "QueryPlan")
            artifact = store.put(
                "QueryRequest",
                "1.0.0",
                {
                    "fault_schedule_hash": plan.payload["fault_schedule_hash"],
                    "idempotency_key": "query-" + plan.content_hash[:40],
                    "input_hash": plan.payload["input_hash"],
                    "parameters": plan.payload["parameters"],
                    "query_hash": plan.payload["query_hash"],
                    "query_plan_hash": plan.content_hash,
                    "source_id": plan.payload["source_id"],
                    "window_hash": plan.payload["window_hash"],
                },
            )
            next_event, details = "QUERY_REQUESTED", {"query_request_hash": artifact.content_hash}
        elif current in {"QUERY_REQUESTED", "QUERY_RETRY_SCHEDULED", "QUERY_LATE_QUARANTINED"}:
            request = cls._ref(store, head, "query_request_hash", "QueryRequest")
            try:
                candidate_artifact, result = cls._candidates(
                    store,
                    adapter,
                    request,
                    crash_after=crash_after,
                )
            except InjectedDataIngestCrash:
                raise
            except Exception as error:
                reason_code = cls._boundary_reason_code(error)
                cls._record_failure(
                    store,
                    journal,
                    epoch,
                    events,
                    failed_stage="QUERY_EXECUTION",
                    reason_code=reason_code,
                )
                return
            if candidate_artifact is not None:
                next_event, details = (
                    "CANDIDATES_SANITIZED",
                    {
                        "candidate_count": 103,
                        "candidate_set_hash": candidate_artifact.content_hash,
                    },
                )
            else:
                if result is None or result.failure_code is None:
                    raise DataIngestWorkflowError("query attempt has no typed outcome")
                observation = store.put(
                    "DataSourceQueryObservation",
                    "1.0.0",
                    {
                        "attempt_ordinal": 1
                        + sum(item in {"QUERY_RETRY_SCHEDULED", "QUERY_LATE_QUARANTINED"} for item in types),
                        "failure_code": result.failure_code,
                        "attempt_hash": result.attempt_hash,
                        "input_hash": result.input_hash,
                        "late_quarantined": result.status == "late",
                        "late_result_hash": result.late_result_hash,
                        "query_hash": result.query_hash,
                        "query_request_hash": request.content_hash,
                        "request_hash": result.request_hash,
                        "result_hash": result.result_hash,
                        "schedule_hash": result.schedule_hash,
                        "status": result.status,
                    },
                )
                if result.status == "permanent_failure":
                    cls._record_failure(
                        store,
                        journal,
                        epoch,
                        events,
                        failed_stage="QUERY_EXECUTION",
                        reason_code=result.failure_code,
                        observation=observation,
                    )
                    return
                next_event = "QUERY_LATE_QUARANTINED" if result.status == "late" else "QUERY_RETRY_SCHEDULED"
                details = {
                    "observation_hash": observation.content_hash,
                    "query_request_hash": request.content_hash,
                }
        elif current == "CANDIDATES_SANITIZED":
            candidate_set = cls._ref(store, head, "candidate_set_hash", "SanitizedCandidateSet")
            try:
                artifact = cls._validate(store, candidate_set, config)
            except (DataIngestWorkflowError, DatasetValidationError, ValueError):
                cls._record_failure(
                    store,
                    journal,
                    epoch,
                    events,
                    failed_stage="POST_QUERY_VALIDATION",
                    reason_code="POST_QUERY_VALIDATION_FAILED",
                )
                return
            next_event, details = "ROWS_VALIDATED", {"validation_manifest_hash": artifact.content_hash}
        elif current == "ROWS_VALIDATED":
            validation = cls._ref(store, head, "validation_manifest_hash", "ValidationManifest")
            candidate_set = store.read(
                cast(str, validation.payload["candidate_set_hash"]),
                expected_schema_name="SanitizedCandidateSet",
            )
            try:
                artifact = cls._dedupe(store, candidate_set, validation)
            except (DataIngestWorkflowError, DatasetValidationError):
                cls._record_failure(
                    store,
                    journal,
                    epoch,
                    events,
                    failed_stage="DEDUPE",
                    reason_code="DEDUPE_INVARIANT_FAILED",
                )
                return
            next_event, details = "ROWS_DEDUPED", {"dedupe_manifest_hash": artifact.content_hash}
        elif current == "ROWS_DEDUPED":
            dedupe = cls._ref(store, head, "dedupe_manifest_hash", "DedupeManifest")
            artifact = cls._select(
                store,
                dedupe,
                config.selection_policy_version,
                adapter.exchange.skill,
            )
            next_event, details = "BADCASES_SELECTED", {"selection_manifest_hash": artifact.content_hash}
        elif current == "BADCASES_SELECTED":
            selection = cls._ref(store, head, "selection_manifest_hash", "BadcaseSelectionManifest")
            dataset, publication = cls._dataset(store, adapter, config, selection, events)
            next_event, details = (
                "DATASET_PUBLISHED",
                {
                    "dataset_publication_hash": publication.content_hash,
                    "dataset_version_hash": dataset.content_hash,
                    "trace_count": 100,
                },
            )
        elif current == "DATASET_PUBLISHED":
            dataset = cls._ref(store, head, "dataset_version_hash", "DatasetVersion")
            publication = cls._ref(store, head, "dataset_publication_hash", "DatasetPublicationManifest")
            artifact = store.put(
                "DecisionRecord",
                "1.0.0",
                {
                    "action": "publish_training_dataset",
                    "dataset_publication_hash": publication.content_hash,
                    "dataset_version_hash": dataset.content_hash,
                    "evidence_hashes": [dataset.content_hash, publication.content_hash],
                    "outcome": "published",
                    "purpose": "training_allowed",
                    "reason_code": "GOVERNED_100_TRACE_DATASET_COMPLETE",
                },
            )
            next_event, details = "DECISION_RECORDED", {"decision_record_hash": artifact.content_hash}
        elif current == "DECISION_RECORDED":
            journal.close(
                epoch,
                status="succeeded",
                reason_code="GOVERNED_100_TRACE_DATASET_COMPLETE",
                expected_sequence=len(events) + 1,
                expected_previous_hash=journal_head.content_hash,
            )
            return
        elif current == "INGEST_FAILURE_RECORDED":
            decision = cls._ref(store, head, "decision_record_hash", "DecisionRecord")
            journal.close(
                epoch,
                status="failed",
                reason_code=cast(str, decision.payload["reason_code"]),
                expected_sequence=len(events) + 1,
                expected_previous_hash=journal_head.content_hash,
            )
            return
        else:
            raise DataIngestWorkflowError("terminal workflow cannot advance")
        journal.append(
            epoch,
            next_event,
            details,
            expected_sequence=len(events) + 1,
            expected_previous_hash=journal_head.content_hash,
        )

    @staticmethod
    def _verify_event_types(types: tuple[str, ...]) -> None:
        if not types or types[0] != "RUN_STARTED":
            raise DataIngestWorkflowError("workflow has no authoritative start")
        transitions = {
            "RUN_STARTED": {"INGEST_FAILURE_RECORDED", "QUERY_PLANNED"},
            "QUERY_PLANNED": {"QUERY_REQUESTED"},
            "QUERY_REQUESTED": {
                "CANDIDATES_SANITIZED",
                "INGEST_FAILURE_RECORDED",
                "QUERY_LATE_QUARANTINED",
                "QUERY_RETRY_SCHEDULED",
            },
            "QUERY_RETRY_SCHEDULED": {
                "CANDIDATES_SANITIZED",
                "INGEST_FAILURE_RECORDED",
                "QUERY_LATE_QUARANTINED",
                "QUERY_RETRY_SCHEDULED",
            },
            "QUERY_LATE_QUARANTINED": {
                "CANDIDATES_SANITIZED",
                "INGEST_FAILURE_RECORDED",
                "QUERY_LATE_QUARANTINED",
                "QUERY_RETRY_SCHEDULED",
            },
            "CANDIDATES_SANITIZED": {"INGEST_FAILURE_RECORDED", "ROWS_VALIDATED"},
            "ROWS_VALIDATED": {"INGEST_FAILURE_RECORDED", "ROWS_DEDUPED"},
            "ROWS_DEDUPED": {"BADCASES_SELECTED", "INGEST_FAILURE_RECORDED"},
            "BADCASES_SELECTED": {"DATASET_PUBLISHED", "INGEST_FAILURE_RECORDED"},
            "DATASET_PUBLISHED": {"DECISION_RECORDED"},
            "DECISION_RECORDED": {"RUN_CLOSED"},
            "INGEST_FAILURE_RECORDED": {"RUN_CLOSED"},
            "RUN_CLOSED": set(),
        }
        for previous, current in zip(types, types[1:], strict=False):
            if current not in transitions.get(previous, set()):
                raise DataIngestWorkflowError("workflow event transition is invalid")

    @classmethod
    def _verify_terminal_decision_binding(cls, store: ArtifactStore, events: list[Artifact]) -> None:
        """Bind RunClosed's generic journal outcome to the application decision path."""

        lifecycle = [event for event in events if event.payload.get("event_type") != "OBSERVATION_RECORDED"]
        if len(lifecycle) < 2 or lifecycle[-1].payload.get("event_type") != "RUN_CLOSED":
            raise DataIngestWorkflowError("terminal application lifecycle is invalid")
        terminal = lifecycle[-1]
        previous = lifecycle[-2]
        closed = cls._ref(store, terminal, "run_closed_hash", "RunClosed")
        previous_type = previous.payload.get("event_type")
        expected_status: str
        expected_reason: str
        if previous_type == "DECISION_RECORDED":
            decision = cls._ref(store, previous, "decision_record_hash", "DecisionRecord")
            expected_status = "succeeded"
            expected_reason = "GOVERNED_100_TRACE_DATASET_COMPLETE"
            if decision.payload.get("reason_code") != expected_reason:
                raise DataIngestWorkflowError("successful terminal decision reason is invalid")
        elif previous_type == "INGEST_FAILURE_RECORDED":
            decision = cls._ref(store, previous, "decision_record_hash", "DecisionRecord")
            expected_status = "failed"
            reason_value = decision.payload.get("reason_code")
            if type(reason_value) is not str or not reason_value:
                raise DataIngestWorkflowError("failed terminal decision reason is invalid")
            expected_reason = cast(str, reason_value)
        else:
            raise DataIngestWorkflowError("RUN_CLOSED does not follow an application decision")
        if closed.payload.get("status") != expected_status or closed.payload.get("reason_code") != expected_reason:
            raise DataIngestWorkflowError("RunClosed contradicts the application decision")

    @classmethod
    def _verify_application_events(
        cls,
        events: list[Artifact],
        *,
        expected_input_hash: str,
    ) -> None:
        """Validate the application lifecycle and every event detail closed-world."""

        if not events:
            raise DataIngestWorkflowError("workflow has no application events")
        lifecycle_types: list[str] = []
        hash_detail_fields: dict[str, set[str]] = {
            "QUERY_PLANNED": {"query_plan_hash"},
            "QUERY_REQUESTED": {"query_request_hash"},
            "ROWS_VALIDATED": {"validation_manifest_hash"},
            "ROWS_DEDUPED": {"dedupe_manifest_hash"},
            "BADCASES_SELECTED": {"selection_manifest_hash"},
            "DECISION_RECORDED": {"decision_record_hash"},
            "RUN_CLOSED": {"run_closed_hash"},
            "OBSERVATION_RECORDED": {"observation_hash"},
        }
        for index, event in enumerate(events):
            event_type = event.payload.get("event_type")
            details = event.payload.get("details")
            if type(event_type) is not str or type(details) is not dict:
                raise DataIngestWorkflowError("application event type or details are invalid")
            name = cast(str, event_type)
            values = cast(dict[str, object], details)
            if name == "RUN_STARTED":
                if values != {"input_hash": expected_input_hash, "purpose": "training_allowed"}:
                    raise DataIngestWorkflowError("RUN_STARTED details are invalid")
            elif name in hash_detail_fields:
                fields = hash_detail_fields[name]
                if set(values) != fields or any(not cls._is_sha256(values.get(field)) for field in fields):
                    raise DataIngestWorkflowError(f"{name} details are invalid")
            elif name in {"QUERY_RETRY_SCHEDULED", "QUERY_LATE_QUARANTINED"}:
                fields = {"observation_hash", "query_request_hash"}
                if set(values) != fields or any(not cls._is_sha256(values.get(field)) for field in fields):
                    raise DataIngestWorkflowError(f"{name} details are invalid")
            elif name == "CANDIDATES_SANITIZED":
                if (
                    set(values) != {"candidate_count", "candidate_set_hash"}
                    or type(values.get("candidate_count")) is not int
                    or values.get("candidate_count") != 103
                    or not cls._is_sha256(values.get("candidate_set_hash"))
                ):
                    raise DataIngestWorkflowError("CANDIDATES_SANITIZED details are invalid")
            elif name == "DATASET_PUBLISHED":
                if (
                    set(values) != {"dataset_publication_hash", "dataset_version_hash", "trace_count"}
                    or type(values.get("trace_count")) is not int
                    or values.get("trace_count") != 100
                    or not cls._is_sha256(values.get("dataset_publication_hash"))
                    or not cls._is_sha256(values.get("dataset_version_hash"))
                ):
                    raise DataIngestWorkflowError("DATASET_PUBLISHED details are invalid")
            elif name == "INGEST_FAILURE_RECORDED":
                fields = {"decision_record_hash", "observation_hash"}
                if set(values) != fields or any(not cls._is_sha256(values.get(field)) for field in fields):
                    raise DataIngestWorkflowError("INGEST_FAILURE_RECORDED details are invalid")
            else:
                raise DataIngestWorkflowError("application event type is not allowlisted")
            if name == "RUN_CLOSED" and index != len(events) - 1:
                raise DataIngestWorkflowError("RUN_CLOSED is not terminal")
            if name != "OBSERVATION_RECORDED":
                lifecycle_types.append(name)
        cls._verify_event_types(tuple(lifecycle_types))

    @classmethod
    def _record_failure(
        cls,
        store: ArtifactStore,
        journal: RunJournal,
        epoch: int,
        events: list[Artifact],
        *,
        failed_stage: str,
        reason_code: str,
        observation: Artifact | None = None,
    ) -> None:
        if not reason_code or not all(
            character.isupper() or character.isdigit() or character == "_" for character in reason_code
        ):
            reason_code = "DATA_INGEST_BOUNDARY_FAILED"
        if observation is None:
            failure_observation_payload = {
                "failure_code": reason_code,
                "failed_stage": failed_stage,
                "raw_values_exported": False,
            }
            assert_public_sentinel_free(failure_observation_payload)
            observation = store.put(
                "DataIngestFailureObservation",
                "1.0.0",
                failure_observation_payload,
            )
        else:
            assert_public_sentinel_free(observation.payload)
        decision_payload = {
            "action": "abort_data_ingest",
            "dataset_version_hash": None,
            "evidence_hashes": [observation.content_hash],
            "failed_stage": failed_stage,
            "outcome": "failed",
            "purpose": "training_allowed",
            "reason_code": reason_code,
        }
        assert_public_sentinel_free(decision_payload)
        decision = store.put(
            "DecisionRecord",
            "1.0.0",
            decision_payload,
        )
        head = events[-1]
        journal.append(
            epoch,
            "INGEST_FAILURE_RECORDED",
            {
                "decision_record_hash": decision.content_hash,
                "observation_hash": observation.content_hash,
            },
            expected_sequence=len(events) + 1,
            expected_previous_hash=head.content_hash,
        )

    @staticmethod
    def _boundary_reason_code(error: Exception) -> str:
        if type(error) is DataSourceBoundaryError and len(error.args) == 1:
            code = error.args[0]
            if type(code) is str and code in _KNOWN_BOUNDARY_REASON_CODES:
                return code
        return "DATA_SOURCE_BOUNDARY_FAILED"

    @staticmethod
    def _verify_public_sentinel_closure(root: Path) -> None:
        artifact_dir = root / "artifacts"
        if not artifact_dir.exists():
            return
        for path in artifact_dir.glob("*.json"):
            try:
                value = json.loads(path.read_bytes())
                assert_public_sentinel_free(value)
            except (OSError, UnicodeError, json.JSONDecodeError, DataContractError) as error:
                raise DataIngestWorkflowError("public artifact violates the sentinel closure policy") from error

    @staticmethod
    def _plan(
        store: ArtifactStore,
        adapter: FixtureDataSource,
        config: FixtureDataIngestConfig,
    ) -> Artifact:
        ast = parse_and_validate_select(config.query_sql, adapter.exchange.skill)
        return store.put(
            "QueryPlan",
            "1.0.0",
            GovernedDataIngestWorkflow._plan_payload(adapter, config, ast),
        )

    @staticmethod
    def _plan_payload(
        adapter: FixtureDataSource,
        config: FixtureDataIngestConfig,
        ast: SelectAst,
    ) -> dict[str, object]:
        if not hasattr(ast, "artifact_payload"):
            raise DataIngestWorkflowError("query AST cannot be serialized")
        parameters = {
            "end_utc": config.window.end_utc,
            "purpose": "training_allowed",
            "start_utc": config.window.start_utc,
        }
        query_hash = FixtureDataSource.query_contract_hash(ast, parameters)
        boundary_config = config.artifact_payload()
        boundary_config.pop("run_id")
        input_hash = sha256_hex(
            canonical_json_bytes(
                {
                    "blueprint_hash": adapter.blueprint_artifact.content_hash,
                    "config": boundary_config,
                    "data_source_skill_hash": adapter.skill_artifact.content_hash,
                    "domain": "governed-query-input/1.0.0",
                    "correction_role_invocation_audit_hash": (adapter.correction_role_audit_artifact.content_hash),
                    "role_invocation_audit_hash": adapter.role_audit_artifact.content_hash,
                }
            )
        )
        return {
            "approved_data_mix_hash": adapter.exchange.skill.approved_data_mix.content_hash,
            "ast": ast.artifact_payload(),
            "blueprint_hash": adapter.blueprint_artifact.content_hash,
            "correction_role_invocation_audit_hash": (adapter.correction_role_audit_artifact.content_hash),
            "data_source_skill_hash": adapter.skill_artifact.content_hash,
            "fault_schedule_hash": FixtureDataSource.fault_schedule_hash(config),
            "input_hash": input_hash,
            "parameters": parameters,
            "purpose": "training_allowed",
            "query_hash": query_hash,
            "query_source_hash": sha256_hex(config.query_sql.encode()),
            "role_invocation_audit_hash": adapter.role_audit_artifact.content_hash,
            "sanitizer_policy_version": config.sanitizer_policy_version,
            "selection_policy_version": config.selection_policy_version,
            "source_id": adapter.exchange.skill.source_id,
            "window_hash": config.window.content_hash,
        }

    @classmethod
    def _candidates(
        cls,
        store: ArtifactStore,
        adapter: FixtureDataSource,
        request: Artifact,
        *,
        crash_after: str | None,
    ) -> tuple[Artifact | None, QueryResult | None]:
        plan = store.read(cast(str, request.payload["query_plan_hash"]), expected_schema_name="QueryPlan")
        result = adapter.query(
            query_key=cast(str, request.payload["idempotency_key"]),
            ast=parse_and_validate_select(adapter.config.query_sql, adapter.exchange.skill),
            parameters=cast(dict[str, object], request.payload["parameters"]),
            input_hash=cast(str, request.payload["input_hash"]),
            query_hash=cast(str, request.payload["query_hash"]),
        )
        if result.status != "success":
            return None, result
        if crash_after == "PRIVATE_QUERY_RESULT_COMMITTED":
            raise InjectedDataIngestCrash("PRIVATE_QUERY_RESULT_COMMITTED")
        refs: list[dict[str, str]] = []
        for row in result.sanitized_rows:
            candidate = store.put(
                "SanitizedCandidate",
                "1.0.0",
                {
                    "event_time_utc": row["event_time_utc"],
                    "ingestion_time_utc": row["ingestion_time_utc"],
                    "model_id": row["model_id"],
                    "prompt": row["prompt"],
                    "purpose": row["purpose"],
                    "query_plan_hash": plan.content_hash,
                    "report_id": row["report_id"],
                    "response": row["response"],
                    "sanitizer_version": adapter.config.sanitizer_policy_version,
                    "source_id": adapter.exchange.skill.source_id,
                    "source_trace_key": row["trace_pk"],
                    "tool_name": row["tool_name"],
                },
            )
            refs.append(
                {
                    "artifact_hash": candidate.content_hash,
                    "report_id": cast(str, row["report_id"]),
                    "source_trace_key": cast(str, row["trace_pk"]),
                }
            )
        return store.put(
            "SanitizedCandidateSet",
            "1.0.0",
            {
                "candidate_count": len(refs),
                "candidate_ref_set_hash": sha256_hex(canonical_json_bytes(refs)),
                "candidate_refs": refs,
                "boundary_attempt_ordinal": result.attempt_ordinal,
                "boundary_attempt_hash": result.attempt_hash,
                "boundary_input_hash": result.input_hash,
                "boundary_private_result_content_hash": result.private_result_content_hash,
                "boundary_query_hash": result.query_hash,
                "boundary_request_hash": result.request_hash,
                "boundary_result_hash": result.result_hash,
                "boundary_result_size": result.result_size,
                "boundary_schedule_hash": result.schedule_hash,
                "boundary_raw_result_hash": result.raw_result_hash,
                "query_plan_hash": plan.content_hash,
                "query_request_hash": request.content_hash,
                "sanitizer_policy_version": adapter.config.sanitizer_policy_version,
            },
        ), None

    @classmethod
    def _verify_committed_boundary_result(
        cls,
        store: ArtifactStore,
        adapter: FixtureDataSource,
        lifecycle_events: list[Artifact],
    ) -> None:
        event_by_type = {
            cast(str, event.payload["event_type"]): event
            for event in lifecycle_events
            if event.payload.get("event_type") != "OBSERVATION_RECORDED"
        }
        candidate_event = event_by_type.get("CANDIDATES_SANITIZED")
        request_event = event_by_type.get("QUERY_REQUESTED")
        if candidate_event is None:
            return
        if request_event is None:
            raise DataIngestWorkflowError("committed candidates have no QueryRequest")
        request = cls._ref(store, request_event, "query_request_hash", "QueryRequest")
        candidate_set = cls._ref(
            store,
            candidate_event,
            "candidate_set_hash",
            "SanitizedCandidateSet",
        )
        query_key = request.payload.get("idempotency_key")
        boundary_request_hash = candidate_set.payload.get("boundary_request_hash")
        if not isinstance(query_key, str) or not isinstance(boundary_request_hash, str):
            raise DataIngestWorkflowError("candidate boundary receipt identity is invalid")
        result = adapter.read_committed_result(
            query_key=query_key,
            expected_request_hash=boundary_request_hash,
            expected_query_hash=cast(str, request.payload.get("query_hash")),
            expected_input_hash=cast(str, request.payload.get("input_hash")),
            expected_schedule_hash=cast(str, request.payload.get("fault_schedule_hash")),
        )
        if (
            result.request_hash != boundary_request_hash
            or candidate_set.payload.get("boundary_attempt_hash") != result.attempt_hash
            or candidate_set.payload.get("boundary_input_hash") != result.input_hash
            or candidate_set.payload.get("boundary_private_result_content_hash") != result.private_result_content_hash
            or candidate_set.payload.get("boundary_query_hash") != result.query_hash
            or candidate_set.payload.get("boundary_raw_result_hash") != result.raw_result_hash
            or candidate_set.payload.get("boundary_result_hash") != result.result_hash
            or candidate_set.payload.get("boundary_result_size") != result.result_size
            or candidate_set.payload.get("boundary_schedule_hash") != result.schedule_hash
            or candidate_set.payload.get("boundary_attempt_ordinal") != result.attempt_ordinal
        ):
            raise DataIngestWorkflowError("candidate set does not match the committed boundary receipt")
        refs = candidate_set.payload.get("candidate_refs")
        rows = result.sanitized_rows
        if not isinstance(refs, list) or len(refs) != len(rows):
            raise DataIngestWorkflowError("candidate set cardinality does not match boundary receipt")
        for ref, row in zip(refs, rows, strict=True):
            if not isinstance(ref, dict) or not isinstance(ref.get("artifact_hash"), str):
                raise DataIngestWorkflowError("candidate ref is invalid")
            candidate = store.read(
                cast(str, ref["artifact_hash"]),
                expected_schema_name="SanitizedCandidate",
            )
            expected = {
                "event_time_utc": row.get("event_time_utc"),
                "ingestion_time_utc": row.get("ingestion_time_utc"),
                "model_id": row.get("model_id"),
                "prompt": row.get("prompt"),
                "purpose": row.get("purpose"),
                "report_id": row.get("report_id"),
                "response": row.get("response"),
                "source_trace_key": row.get("trace_pk"),
                "tool_name": row.get("tool_name"),
            }
            for key, value in expected.items():
                if candidate.payload.get(key) != value:
                    raise DataIngestWorkflowError("candidate content differs from boundary receipt")

    @classmethod
    def _verify_query_attempt_evidence(
        cls,
        store: ArtifactStore,
        adapter: FixtureDataSource,
        config: FixtureDataIngestConfig,
        lifecycle_events: list[Artifact],
    ) -> None:
        """Close every private attempt over its exact public lifecycle evidence."""

        request_event = next(
            (event for event in lifecycle_events if event.payload.get("event_type") == "QUERY_REQUESTED"),
            None,
        )
        if request_event is None:
            return
        request = cls._ref(store, request_event, "query_request_hash", "QueryRequest")
        query_key = request.payload.get("idempotency_key")
        input_hash = request.payload.get("input_hash")
        query_hash = request.payload.get("query_hash")
        schedule_hash = request.payload.get("fault_schedule_hash")
        parameters = request.payload.get("parameters")
        if (
            type(query_key) is not str
            or not cls._is_sha256(input_hash)
            or not cls._is_sha256(query_hash)
            or not cls._is_sha256(schedule_hash)
            or type(parameters) is not dict
        ):
            raise DataIngestWorkflowError("QueryRequest boundary identity is invalid")
        ast = parse_and_validate_select(config.query_sql, adapter.exchange.skill)
        request_hash = sha256_hex(
            canonical_json_bytes(
                FixtureDataSource.boundary_request_payload(
                    query_key=cast(str, query_key),
                    ast=ast,
                    parameters=cast(dict[str, object], parameters),
                    input_hash=cast(str, input_hash),
                    query_hash=cast(str, query_hash),
                    schedule_hash=cast(str, schedule_hash),
                    source_id=adapter.exchange.skill.source_id,
                    window_hash=adapter.exchange.skill.approved_window.content_hash,
                )
            )
        )
        request_path = adapter.private_root / "queries" / cast(str, query_key) / "request.canonical.json"
        outcome_events = [
            event
            for event in lifecycle_events
            if event.payload.get("event_type") in {"QUERY_RETRY_SCHEDULED", "QUERY_LATE_QUARANTINED"}
        ]
        failure_event = next(
            (event for event in lifecycle_events if event.payload.get("event_type") == "INGEST_FAILURE_RECORDED"),
            None,
        )
        failure_observation: Artifact | None = None
        if failure_event is not None:
            failure_details = failure_event.payload.get("details")
            if type(failure_details) is not dict:
                raise DataIngestWorkflowError("failure event details are invalid")
            failure_hash = cast(dict[str, object], failure_details).get("observation_hash")
            if type(failure_hash) is not str:
                raise DataIngestWorkflowError("failure observation identity is invalid")
            failure_observation = store.read(cast(str, failure_hash))
            if failure_observation.schema_name == "DataSourceQueryObservation":
                outcome_events.append(failure_event)

        has_candidate = any(event.payload.get("event_type") == "CANDIDATES_SANITIZED" for event in lifecycle_events)
        if not request_path.exists():
            if outcome_events or has_candidate:
                raise DataIngestWorkflowError("committed query outcome has no private request evidence")
            return
        attempts = adapter.verify_attempt_chain(
            query_key=cast(str, query_key),
            expected_request_hash=request_hash,
            expected_query_hash=cast(str, query_hash),
            expected_input_hash=cast(str, input_hash),
            expected_schedule_hash=cast(str, schedule_hash),
        )
        if len(attempts) < len(outcome_events):
            raise DataIngestWorkflowError("public query outcomes exceed the private attempt chain")
        for ordinal, (event, attempt) in enumerate(zip(outcome_events, attempts, strict=False), start=1):
            details = event.payload.get("details")
            if type(details) is not dict:
                raise DataIngestWorkflowError("query outcome event details are invalid")
            details_value = cast(dict[str, object], details)
            observation_hash = details_value.get("observation_hash")
            if event.payload.get("event_type") in {"QUERY_RETRY_SCHEDULED", "QUERY_LATE_QUARANTINED"}:
                if set(details_value) != {"observation_hash", "query_request_hash"}:
                    raise DataIngestWorkflowError("query outcome event fields are invalid")
                if details_value.get("query_request_hash") != request.content_hash:
                    raise DataIngestWorkflowError("query outcome request lineage is invalid")
            elif failure_observation is not None:
                observation_hash = failure_observation.content_hash
            if type(observation_hash) is not str:
                raise DataIngestWorkflowError("query outcome observation hash is invalid")
            observation = store.read(
                cast(str, observation_hash),
                expected_schema_name="DataSourceQueryObservation",
            )
            expected_observation = {
                "attempt_hash": attempt.attempt_hash,
                "attempt_ordinal": ordinal,
                "failure_code": attempt.failure_code,
                "input_hash": attempt.input_hash,
                "late_quarantined": attempt.status == "late",
                "late_result_hash": attempt.late_result_hash,
                "query_hash": attempt.query_hash,
                "query_request_hash": request.content_hash,
                "request_hash": attempt.request_hash,
                "result_hash": attempt.result_hash,
                "schedule_hash": attempt.schedule_hash,
                "status": attempt.status,
            }
            if observation.schema_version != "1.0.0" or observation.payload != expected_observation:
                raise DataIngestWorkflowError("query outcome observation differs from its private attempt")
        extra = attempts[len(outcome_events) :]
        if has_candidate:
            if len(extra) != 1 or extra[0].status != "success":
                raise DataIngestWorkflowError("candidate commit does not close exactly one success attempt")
        elif failure_event is not None:
            if len(extra) > 1 or (extra and extra[0].status != "boundary_failure"):
                raise DataIngestWorkflowError("failure commit has unexpected private attempts")
            if extra:
                failure_details = failure_event.payload.get("details")
                if not isinstance(failure_details, dict):
                    raise DataIngestWorkflowError("failure event details are invalid")
                decision = cls._ref(store, failure_event, "decision_record_hash", "DecisionRecord")
                observation = store.read(
                    cast(str, failure_details.get("observation_hash")),
                    expected_schema_name="DataIngestFailureObservation",
                )
                attempt = extra[0]
                if (
                    decision.payload.get("failed_stage") != "QUERY_EXECUTION"
                    or decision.payload.get("reason_code") != attempt.failure_code
                    or observation.payload.get("failed_stage") != "QUERY_EXECUTION"
                    or observation.payload.get("failure_code") != attempt.failure_code
                ):
                    raise DataIngestWorkflowError("boundary failure decision contradicts private attempt evidence")
        elif len(extra) > 1 or (extra and extra[0].status not in {"success", "boundary_failure"}):
            raise DataIngestWorkflowError("uncommitted query attempt tail is invalid")

    @classmethod
    def _verify_committed_public_state(
        cls,
        store: ArtifactStore,
        adapter: FixtureDataSource,
        config: FixtureDataIngestConfig,
        events: list[Artifact],
    ) -> None:
        """Re-read every committed public stage and its closed-world cross-refs."""

        lifecycle = [event for event in events if event.payload.get("event_type") != "OBSERVATION_RECORDED"]
        by_type: dict[str, Artifact] = {}
        for event in lifecycle:
            event_type = event.payload.get("event_type")
            if not isinstance(event_type, str):
                raise DataIngestWorkflowError("lifecycle event type is invalid")
            if event_type not in {"QUERY_RETRY_SCHEDULED", "QUERY_LATE_QUARANTINED"} and event_type in by_type:
                raise DataIngestWorkflowError("lifecycle stage is duplicated")
            by_type[event_type] = event
        plan: Artifact | None = None
        if "QUERY_PLANNED" in by_type:
            start_details = by_type["RUN_STARTED"].payload.get("details")
            if not isinstance(start_details, dict) or not cls._is_sha256(start_details.get("input_hash")):
                raise DataIngestWorkflowError("RUN_STARTED input identity is invalid")
            plan = cls._ref(store, by_type["QUERY_PLANNED"], "query_plan_hash", "QueryPlan")
            expected_plan = cls._plan_payload(
                adapter,
                config,
                parse_and_validate_select(config.query_sql, adapter.exchange.skill),
            )
            if plan.schema_version != "1.0.0" or plan.payload != expected_plan:
                raise DataIngestWorkflowError("QueryPlan differs from the frozen approved query")
        request: Artifact | None = None
        if "QUERY_REQUESTED" in by_type:
            if plan is None:
                raise DataIngestWorkflowError("QueryRequest has no verified QueryPlan")
            request = cls._ref(store, by_type["QUERY_REQUESTED"], "query_request_hash", "QueryRequest")
            expected_request = {
                "fault_schedule_hash": plan.payload["fault_schedule_hash"],
                "idempotency_key": "query-" + plan.content_hash[:40],
                "input_hash": plan.payload["input_hash"],
                "parameters": plan.payload["parameters"],
                "query_hash": plan.payload["query_hash"],
                "query_plan_hash": plan.content_hash,
                "source_id": plan.payload["source_id"],
                "window_hash": plan.payload["window_hash"],
            }
            if request.schema_version != "1.0.0" or request.payload != expected_request:
                raise DataIngestWorkflowError("QueryRequest differs from its verified QueryPlan")
        candidate_set: Artifact | None = None
        candidates: list[Artifact] = []
        if "CANDIDATES_SANITIZED" in by_type:
            if request is None or plan is None:
                raise DataIngestWorkflowError("SanitizedCandidateSet has no query lineage")
            candidate_set = cls._ref(
                store,
                by_type["CANDIDATES_SANITIZED"],
                "candidate_set_hash",
                "SanitizedCandidateSet",
            )
            expected_fields = {
                "boundary_attempt_hash",
                "boundary_attempt_ordinal",
                "boundary_input_hash",
                "boundary_private_result_content_hash",
                "boundary_query_hash",
                "boundary_raw_result_hash",
                "boundary_request_hash",
                "boundary_result_hash",
                "boundary_result_size",
                "boundary_schedule_hash",
                "candidate_count",
                "candidate_ref_set_hash",
                "candidate_refs",
                "query_plan_hash",
                "query_request_hash",
                "sanitizer_policy_version",
            }
            refs = candidate_set.payload.get("candidate_refs")
            candidate_set_hash_fields = (
                candidate_set.payload.get("boundary_attempt_hash"),
                candidate_set.payload.get("boundary_input_hash"),
                candidate_set.payload.get("boundary_private_result_content_hash"),
                candidate_set.payload.get("boundary_query_hash"),
                candidate_set.payload.get("boundary_raw_result_hash"),
                candidate_set.payload.get("boundary_request_hash"),
                candidate_set.payload.get("boundary_result_hash"),
                candidate_set.payload.get("boundary_schedule_hash"),
                candidate_set.payload.get("candidate_ref_set_hash"),
                candidate_set.payload.get("query_plan_hash"),
                candidate_set.payload.get("query_request_hash"),
            )
            candidate_set_integer_fields = (
                candidate_set.payload.get("boundary_attempt_ordinal"),
                candidate_set.payload.get("boundary_result_size"),
                candidate_set.payload.get("candidate_count"),
            )
            if (
                candidate_set.schema_version != "1.0.0"
                or set(candidate_set.payload) != expected_fields
                or not all(cls._is_sha256(value) for value in candidate_set_hash_fields)
                or any(type(value) is not int or cast(int, value) <= 0 for value in candidate_set_integer_fields)
                or candidate_set.payload.get("candidate_count") != 103
                or candidate_set.payload.get("query_plan_hash") != plan.content_hash
                or candidate_set.payload.get("query_request_hash") != request.content_hash
                or candidate_set.payload.get("boundary_input_hash") != request.payload.get("input_hash")
                or candidate_set.payload.get("boundary_query_hash") != request.payload.get("query_hash")
                or candidate_set.payload.get("boundary_schedule_hash") != request.payload.get("fault_schedule_hash")
                or candidate_set.payload.get("sanitizer_policy_version") != config.sanitizer_policy_version
                or not isinstance(refs, list)
                or len(refs) != 103
                or candidate_set.payload.get("candidate_ref_set_hash") != sha256_hex(canonical_json_bytes(refs))
            ):
                raise DataIngestWorkflowError("SanitizedCandidateSet fields or lineage are invalid")
            seen: set[tuple[str, str, str]] = set()
            for ref in refs:
                if not isinstance(ref, dict) or set(ref) != {"artifact_hash", "report_id", "source_trace_key"}:
                    raise DataIngestWorkflowError("SanitizedCandidate ref is invalid")
                values = (ref.get("artifact_hash"), ref.get("report_id"), ref.get("source_trace_key"))
                if (
                    not cls._is_sha256(values[0])
                    or type(values[1]) is not str
                    or type(values[2]) is not str
                    or cast(tuple[str, str, str], values) in seen
                ):
                    raise DataIngestWorkflowError("SanitizedCandidate ref is duplicated")
                seen.add(cast(tuple[str, str, str], values))
                candidate = store.read(cast(str, values[0]), expected_schema_name="SanitizedCandidate")
                string_fields = (
                    "event_time_utc",
                    "ingestion_time_utc",
                    "model_id",
                    "prompt",
                    "purpose",
                    "query_plan_hash",
                    "report_id",
                    "response",
                    "sanitizer_version",
                    "source_id",
                    "source_trace_key",
                    "tool_name",
                )
                if (
                    candidate.schema_version != "1.0.0"
                    or set(candidate.payload)
                    != {
                        "event_time_utc",
                        "ingestion_time_utc",
                        "model_id",
                        "prompt",
                        "purpose",
                        "query_plan_hash",
                        "report_id",
                        "response",
                        "sanitizer_version",
                        "source_id",
                        "source_trace_key",
                        "tool_name",
                    }
                    or any(type(candidate.payload.get(field)) is not str for field in string_fields)
                    or not cls._is_sha256(candidate.payload.get("query_plan_hash"))
                    or candidate.payload.get("report_id") != values[1]
                    or candidate.payload.get("source_trace_key") != values[2]
                    or candidate.payload.get("query_plan_hash") != plan.content_hash
                    or candidate.payload.get("sanitizer_version") != config.sanitizer_policy_version
                    or candidate.payload.get("source_id") != plan.payload.get("source_id")
                ):
                    raise DataIngestWorkflowError("SanitizedCandidate fields are invalid")
                candidates.append(candidate)
            boundary_verified = False
            try:
                cls._verify_committed_boundary_result(store, adapter, lifecycle)
            except DataIngestWorkflowError:
                # Same-schema, same-type value substitution is detected by the
                # immutable private receipt/content binding.
                raise
            except DataSourceBoundaryError:
                # Missing/corrupt source receipts are handled by _advance as an
                # auditable boundary failure; they never authorize bad content.
                pass
            else:
                boundary_verified = True
            if (
                any(candidate.payload.get("purpose") != "training_allowed" for candidate in candidates)
                and not boundary_verified
            ):
                raise DataIngestWorkflowError("candidate purpose is outside the frozen policy")
        validation: Artifact | None = None
        if "ROWS_VALIDATED" in by_type:
            if candidate_set is None:
                raise DataIngestWorkflowError("ValidationManifest has no candidates")
            validation = cls._ref(
                store,
                by_type["ROWS_VALIDATED"],
                "validation_manifest_hash",
                "ValidationManifest",
            )
            if (
                validation.schema_version != "1.0.0"
                or set(validation.payload)
                != {
                    "candidate_set_hash",
                    "checks",
                    "sanitized_artifact_bytes",
                    "status",
                    "validated_count",
                }
                or validation.payload.get("candidate_set_hash") != candidate_set.content_hash
                or validation.payload.get("validated_count") != 103
                or validation.payload.get("status") != "passed"
                or validation.payload.get("sanitized_artifact_bytes")
                != sum(len(candidate.raw_bytes) for candidate in candidates)
            ):
                raise DataIngestWorkflowError("ValidationManifest is invalid")
            if validation.payload != expected_validation_manifest(
                candidate_set,
                candidates,
                window_start_utc=config.window.start_utc,
                window_end_utc=config.window.end_utc,
            ):
                raise DataIngestWorkflowError("ValidationManifest is not recomputed from candidates")
        dedupe: Artifact | None = None
        if "ROWS_DEDUPED" in by_type:
            if candidate_set is None or validation is None:
                raise DataIngestWorkflowError("DedupeManifest has incomplete lineage")
            dedupe = cls._ref(store, by_type["ROWS_DEDUPED"], "dedupe_manifest_hash", "DedupeManifest")
            winners = dedupe.payload.get("winner_refs")
            if (
                dedupe.schema_version != "1.0.0"
                or set(dedupe.payload)
                != {
                    "candidate_set_hash",
                    "duplicate_count",
                    "duplicate_groups",
                    "input_count",
                    "semantics",
                    "unique_count",
                    "validation_manifest_hash",
                    "winner_refs",
                }
                or dedupe.payload.get("candidate_set_hash") != candidate_set.content_hash
                or dedupe.payload.get("validation_manifest_hash") != validation.content_hash
                or dedupe.payload.get("input_count") != 103
                or dedupe.payload.get("duplicate_count") != 3
                or dedupe.payload.get("unique_count") != 100
                or not isinstance(winners, list)
                or len(winners) != 100
            ):
                raise DataIngestWorkflowError("DedupeManifest is invalid")
            if dedupe.payload != expected_dedupe_manifest(candidate_set, validation, candidates):
                raise DataIngestWorkflowError("DedupeManifest is not recomputed from candidates")
        if "BADCASES_SELECTED" in by_type:
            if dedupe is None:
                raise DataIngestWorkflowError("BadcaseSelectionManifest has no dedupe lineage")
            selection = cls._ref(
                store,
                by_type["BADCASES_SELECTED"],
                "selection_manifest_hash",
                "BadcaseSelectionManifest",
            )
            rankings = selection.payload.get("rankings")
            if (
                selection.schema_version != "1.0.0"
                or set(selection.payload)
                != {
                    "approved_data_mix_hash",
                    "approved_data_mix_policy_version",
                    "dedupe_manifest_hash",
                    "eligible_count",
                    "mix",
                    "rankings",
                    "score_contract",
                    "selected_count",
                    "selection_policy_version",
                }
                or selection.payload.get("dedupe_manifest_hash") != dedupe.content_hash
                or selection.payload.get("eligible_count") != 100
                or selection.payload.get("selected_count") != 100
                or selection.payload.get("selection_policy_version") != config.selection_policy_version
                or selection.payload.get("approved_data_mix_hash")
                != adapter.exchange.skill.approved_data_mix.content_hash
                or selection.payload.get("approved_data_mix_policy_version")
                != adapter.exchange.skill.approved_data_mix.policy_version
                or selection.payload.get("score_contract") != "sha256-over-sanitized-semantic-content"
                or not isinstance(rankings, list)
                or len(rankings) != 100
            ):
                raise DataIngestWorkflowError("BadcaseSelectionManifest is invalid")
            previous: tuple[str, str] | None = None
            seen_hashes: set[str] = set()
            observed_tools: Counter[str] = Counter()
            observed_models: Counter[str] = Counter()
            observed_purposes: Counter[str] = Counter()
            for ordinal, ranking in enumerate(rankings, start=1):
                if not isinstance(ranking, dict):
                    raise DataIngestWorkflowError("badcase ranking is invalid")
                candidate_hash = ranking.get("artifact_hash")
                source_key = ranking.get("source_trace_key")
                if (
                    set(ranking) != {"artifact_hash", "rank", "selected", "selection_score", "source_trace_key"}
                    or type(ranking.get("rank")) is not int
                    or ranking.get("rank") != ordinal
                    or type(ranking.get("selected")) is not bool
                    or ranking.get("selected") is not True
                    or type(candidate_hash) is not str
                    or candidate_hash in seen_hashes
                    or type(source_key) is not str
                ):
                    raise DataIngestWorkflowError("badcase ranking identity is invalid")
                candidate = store.read(candidate_hash, expected_schema_name="SanitizedCandidate")
                score = selection_score(candidate.payload, config.selection_policy_version)
                if ranking.get("selection_score") != score:
                    raise DataIngestWorkflowError("badcase score is not derived from content")
                ordering = (score, source_key)
                if previous is not None and ordering >= previous:
                    raise DataIngestWorkflowError("badcase ranking order is invalid")
                previous = ordering
                seen_hashes.add(candidate_hash)
                observed_tools[cast(str, candidate.payload["tool_name"])] += 1
                observed_models[cast(str, candidate.payload["model_id"])] += 1
                observed_purposes[cast(str, candidate.payload["purpose"])] += 1
            observed_mix = {
                "model": dict(sorted(observed_models.items())),
                "purpose": dict(sorted(observed_purposes.items())),
                "tool": dict(sorted(observed_tools.items())),
            }
            approved_mix = adapter.exchange.skill.approved_data_mix.artifact_payload()
            if selection.payload.get("mix") != observed_mix or observed_mix != {
                "model": approved_mix["model"],
                "purpose": approved_mix["purpose"],
                "tool": approved_mix["tool"],
            }:
                raise DataIngestWorkflowError("BadcaseSelectionManifest mix is invalid")
            if selection.payload != expected_selection_manifest(
                store,
                dedupe,
                policy=config.selection_policy_version,
                approved_mix=adapter.exchange.skill.approved_data_mix,
            ):
                raise DataIngestWorkflowError("BadcaseSelectionManifest is not fully recomputed")
        dataset: Artifact | None = None
        publication: Artifact | None = None
        if "DATASET_PUBLISHED" in by_type:
            published_event = by_type["DATASET_PUBLISHED"]
            dataset = cls._ref(store, published_event, "dataset_version_hash", "DatasetVersion")
            publication = cls._ref(
                store,
                published_event,
                "dataset_publication_hash",
                "DatasetPublicationManifest",
            )
            loaded = load_training_dataset(store, dataset.content_hash)
            dataset_lineage = dataset.payload.get("lineage")
            if not isinstance(dataset_lineage, dict):
                raise DataIngestWorkflowError("DatasetVersion lineage is invalid")
            expected_publication = {
                "dataset_version_hash": dataset.content_hash,
                "lineage_hash": sha256_hex(canonical_json_bytes(dataset_lineage)),
                "purpose": "training_allowed",
                "status": "published",
                "trace_count": 100,
                "trace_set_hash": dataset.payload["trace_set_hash"],
            }
            details = published_event.payload.get("details")
            if (
                publication.schema_version != "1.0.0"
                or publication.payload != expected_publication
                or not isinstance(details, dict)
                or details.get("trace_count") != 100
                or len(loaded.training_traces) != 100
            ):
                raise DataIngestWorkflowError("DatasetPublicationManifest is invalid")
        if "DECISION_RECORDED" in by_type:
            if dataset is None or publication is None:
                raise DataIngestWorkflowError("successful DecisionRecord has no publication")
            decision = cls._ref(
                store,
                by_type["DECISION_RECORDED"],
                "decision_record_hash",
                "DecisionRecord",
            )
            if decision.schema_version != "1.0.0" or decision.payload != {
                "action": "publish_training_dataset",
                "dataset_publication_hash": publication.content_hash,
                "dataset_version_hash": dataset.content_hash,
                "evidence_hashes": [dataset.content_hash, publication.content_hash],
                "outcome": "published",
                "purpose": "training_allowed",
                "reason_code": "GOVERNED_100_TRACE_DATASET_COMPLETE",
            }:
                raise DataIngestWorkflowError("successful DecisionRecord is invalid")
        if "INGEST_FAILURE_RECORDED" in by_type:
            failure_event = by_type["INGEST_FAILURE_RECORDED"]
            failure_details = failure_event.payload.get("details")
            if not isinstance(failure_details, dict):
                raise DataIngestWorkflowError("failure event details are invalid")
            decision = cls._ref(store, failure_event, "decision_record_hash", "DecisionRecord")
            observation_hash = failure_details.get("observation_hash")
            evidence = decision.payload.get("evidence_hashes")
            if (
                set(failure_details) != {"decision_record_hash", "observation_hash"}
                or decision.schema_version != "1.0.0"
                or set(decision.payload)
                != {
                    "action",
                    "dataset_version_hash",
                    "evidence_hashes",
                    "failed_stage",
                    "outcome",
                    "purpose",
                    "reason_code",
                }
                or decision.payload.get("action") != "abort_data_ingest"
                or decision.payload.get("dataset_version_hash") is not None
                or decision.payload.get("outcome") != "failed"
                or decision.payload.get("purpose") != "training_allowed"
                or type(decision.payload.get("failed_stage")) is not str
                or not cast(str, decision.payload.get("failed_stage"))
                or type(decision.payload.get("reason_code")) is not str
                or not cast(str, decision.payload.get("reason_code"))
                or not cls._is_sha256(failure_details.get("decision_record_hash"))
                or not cls._is_sha256(observation_hash)
                or evidence != [observation_hash]
            ):
                raise DataIngestWorkflowError("failed DecisionRecord is invalid")
            observation = store.read(cast(str, observation_hash))
            if observation.schema_version != "1.0.0" or observation.schema_name not in {
                "DataIngestFailureObservation",
                "DataSourceQueryObservation",
            }:
                raise DataIngestWorkflowError("failure observation is invalid")
            if observation.schema_name == "DataIngestFailureObservation" and observation.payload != {
                "failure_code": decision.payload["reason_code"],
                "failed_stage": decision.payload["failed_stage"],
                "raw_values_exported": False,
            }:
                raise DataIngestWorkflowError("generic failure observation is invalid")

    @staticmethod
    def _candidate_list(store: ArtifactStore, candidate_set: Artifact) -> list[Artifact]:
        refs = cast(list[object], candidate_set.payload["candidate_refs"])
        return [
            store.read(
                cast(str, cast(dict[str, object], ref)["artifact_hash"]),
                expected_schema_name="SanitizedCandidate",
            )
            for ref in refs
        ]

    @classmethod
    def _validate(cls, store: ArtifactStore, candidate_set: Artifact, config: FixtureDataIngestConfig) -> Artifact:
        candidates = cls._candidate_list(store, candidate_set)
        return store.put(
            "ValidationManifest",
            "1.0.0",
            expected_validation_manifest(
                candidate_set,
                candidates,
                window_start_utc=config.window.start_utc,
                window_end_utc=config.window.end_utc,
            ),
        )

    @classmethod
    def _dedupe(cls, store: ArtifactStore, candidate_set: Artifact, validation: Artifact) -> Artifact:
        return store.put(
            "DedupeManifest",
            "1.0.0",
            expected_dedupe_manifest(
                candidate_set,
                validation,
                cls._candidate_list(store, candidate_set),
            ),
        )

    @staticmethod
    def _select(store: ArtifactStore, dedupe: Artifact, policy: str, skill: DataSourceSkill) -> Artifact:
        return store.put(
            "BadcaseSelectionManifest",
            "1.0.0",
            expected_selection_manifest(
                store,
                dedupe,
                policy=policy,
                approved_mix=skill.approved_data_mix,
            ),
        )

    @classmethod
    def _dataset(
        cls,
        store: ArtifactStore,
        adapter: FixtureDataSource,
        config: FixtureDataIngestConfig,
        selection: Artifact,
        events: list[Artifact],
    ) -> tuple[Artifact, Artifact]:
        trace_refs: list[dict[str, str]] = []
        for rank in cast(list[dict[str, object]], selection.payload["rankings"]):
            candidate_hash = cast(str, rank["artifact_hash"])
            candidate = store.read(candidate_hash, expected_schema_name="SanitizedCandidate")
            payload = training_trace_payload(candidate_hash, candidate.payload)
            trace = store.put("TrainingTrace", "1.0.0", payload)
            trace_refs.append({"artifact_hash": trace.content_hash, "trace_id": cast(str, payload["trace_id"])})
        by_type = {cast(str, event.payload["event_type"]): event for event in events}

        def detail(event_type: str, key: str) -> str:
            values = cast(dict[str, object], by_type[event_type].payload["details"])
            return cast(str, values[key])

        plan_hash = detail("QUERY_PLANNED", "query_plan_hash")
        plan = store.read(plan_hash, expected_schema_name="QueryPlan")
        lineage: dict[str, JsonValue] = {
            "approved_data_mix_hash": adapter.exchange.skill.approved_data_mix.content_hash,
            "approved_data_mix_policy_version": adapter.exchange.skill.approved_data_mix.policy_version,
            "blueprint_hash": adapter.blueprint_artifact.content_hash,
            "candidate_set_hash": detail("CANDIDATES_SANITIZED", "candidate_set_hash"),
            "correction_role_invocation_audit_hash": (adapter.correction_role_audit_artifact.content_hash),
            "data_source_skill_hash": adapter.skill_artifact.content_hash,
            "dedupe_manifest_hash": detail("ROWS_DEDUPED", "dedupe_manifest_hash"),
            "query_plan_hash": plan_hash,
            "query_source_hash": cast(str, plan.payload["query_source_hash"]),
            "role_invocation_audit_hash": adapter.role_audit_artifact.content_hash,
            "sanitizer_policy_version": config.sanitizer_policy_version,
            "selection_manifest_hash": selection.content_hash,
            "selection_policy_version": config.selection_policy_version,
            "validation_manifest_hash": detail("ROWS_VALIDATED", "validation_manifest_hash"),
        }
        without_id: dict[str, object] = {
            "lineage": lineage,
            "purpose": "training_allowed",
            "source_id": adapter.exchange.skill.source_id,
            "trace_count": 100,
            "trace_refs": trace_refs,
            "trace_set_hash": trace_set_hash(trace_refs),
        }
        dataset = store.put(
            "DatasetVersion",
            "1.0.0",
            {"dataset_version_id": dataset_version_id(without_id), **without_id},
        )
        publication = store.put(
            "DatasetPublicationManifest",
            "1.0.0",
            {
                "dataset_version_hash": dataset.content_hash,
                "lineage_hash": sha256_hex(canonical_json_bytes(lineage)),
                "purpose": "training_allowed",
                "status": "published",
                "trace_count": 100,
                "trace_set_hash": without_id["trace_set_hash"],
            },
        )
        return dataset, publication

    @staticmethod
    def _ref(store: ArtifactStore, event: Artifact, key: str, schema: str) -> Artifact:
        details = cast(dict[str, object], event.payload["details"])
        return store.read(cast(str, details[key]), expected_schema_name=schema)

    @classmethod
    def _snapshot(cls, store: ArtifactStore, journal: RunJournal) -> DataIngestSnapshot:
        journal.verify()
        events = journal.events()
        cls._verify_application_events(events, expected_input_hash=journal.reserved_input_hash())
        dataset = None
        decision = None
        for event in events:
            if event.payload["event_type"] == "DATASET_PUBLISHED":
                dataset = cls._ref(store, event, "dataset_version_hash", "DatasetVersion")
            elif event.payload["event_type"] in {"DECISION_RECORDED", "INGEST_FAILURE_RECORDED"}:
                decision = cls._ref(store, event, "decision_record_hash", "DecisionRecord")
        terminal = events[-1].payload["event_type"] == "RUN_CLOSED"
        if terminal:
            journal.closed()
            if dataset is not None:
                load_training_dataset(store, dataset.content_hash)
        return DataIngestSnapshot(tuple(events), dataset, decision, terminal)
