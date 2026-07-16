"""First red contracts for governed Ticket 03 data ingestion."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from clawrl.adapters.sources.fixture import FixtureDataSource
from clawrl.data.models import DataContractError, FixtureDataIngestConfig, QueryWindow
from clawrl.data.sql import SqlPolicyError, parse_and_validate_select
from tests.fixtures.ticket03_data import (
    DATA_PROVIDER_RAW,
    PRIVATE_SENTINEL,
    canonical_provider_output,
    stage_data_provider,
)

APPROVED_SQL = (
    "SELECT trace_pk, report_id, event_time_utc, ingestion_time_utc, purpose, prompt, response, "
    "tool_name, model_id, private_sentinel FROM fixture.online_trace "
    "WHERE event_time_utc >= :start_utc AND event_time_utc < :end_utc AND purpose = :purpose "
    "ORDER BY trace_pk ASC, ingestion_time_utc ASC"
)


def fixture_config(*, fault_schedule: tuple[str, ...] = ("success",)) -> FixtureDataIngestConfig:
    return FixtureDataIngestConfig(
        run_id="ticket03-contract",
        query_sql=APPROVED_SQL,
        window=QueryWindow("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
        fault_schedule=fault_schedule,
    )


def test_role_bytes_are_exact_canonical_and_normalized_blueprint_is_public_safe(tmp_path: Path) -> None:
    assert hashlib.sha256(DATA_PROVIDER_RAW).hexdigest() == (
        "e38025307baa56c095df829a7a501cbf8ecf844435e7f3dab873a4cae55add74"
    )
    assert canonical_provider_output() != DATA_PROVIDER_RAW
    receipt = stage_data_provider(tmp_path)
    contract = FixtureDataSource.load_public_contract(tmp_path, receipt)
    assert contract.base_skill.expected_raw_rows == 103
    assert contract.base_skill.expected_unique_rows == 100
    assert contract.role_audit_artifact.payload["raw_output_size"] == len(DATA_PROVIDER_RAW)
    assert receipt.raw_output_hash != contract.role_audit_artifact.payload["normalized_output_hash"]
    assert PRIVATE_SENTINEL.encode() not in contract.blueprint_artifact.raw_bytes


def test_closed_world_parser_builds_ast_for_exact_approved_query(tmp_path: Path) -> None:
    skill = FixtureDataSource.load_public_contract(tmp_path, stage_data_provider(tmp_path)).base_skill
    ast = parse_and_validate_select(APPROVED_SQL, skill)
    assert ast.table == "fixture.online_trace"
    assert ast.columns == skill.approved_columns
    assert [(item.column, item.operator, item.parameter) for item in ast.filters] == [
        ("event_time_utc", ">=", "start_utc"),
        ("event_time_utc", "<", "end_utc"),
        ("purpose", "=", "purpose"),
    ]
    assert [(item.column, item.direction) for item in ast.order_by] == [
        ("trace_pk", "ASC"),
        ("ingestion_time_utc", "ASC"),
    ]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM fixture.online_trace WHERE purpose = :purpose",
        "SELECT trace_pk FROM fixture.secret",
        "SELECT trace_pk FROM fixture.online_trace; DROP TABLE fixture.online_trace",
        "SELECT upper(prompt) FROM fixture.online_trace",
        "SELECT trace_pk FROM fixture.online_trace -- approved",
        "UPDATE fixture.online_trace SET purpose = :purpose",
        "SELECT trace_pk FROM fixture.online_trace WHERE model_id = :model_id",
        "SELECT trace_pk FROM fixture.online_trace WHERE purpose != :purpose",
        "SELECT trace_pk FROM fixture.online_trace JOIN fixture.other ON trace_pk = trace_pk",
        "SELECT trace_pk FROM fixture.online_trace WHERE purpose = 'training_allowed'",
    ],
)
def test_sql_policy_rejects_every_unapproved_shape_before_query(sql: str, tmp_path: Path) -> None:
    skill = FixtureDataSource.load_public_contract(tmp_path, stage_data_provider(tmp_path)).base_skill
    with pytest.raises(SqlPolicyError):
        parse_and_validate_select(sql, skill)


def test_adapter_queries_103_sanitizes_inside_boundary_and_dedupes_to_exactly_100(tmp_path: Path) -> None:
    receipt = stage_data_provider(tmp_path)
    skill = FixtureDataSource.load_public_contract(tmp_path, receipt).base_skill
    ast = parse_and_validate_select(APPROVED_SQL, skill)
    adapter = FixtureDataSource.bootstrap(tmp_path, receipt, fixture_config())
    parameters = {
        "end_utc": "2026-01-02T00:00:00Z",
        "purpose": "training_allowed",
        "start_utc": "2026-01-01T00:00:00Z",
    }
    result = adapter.query(
        query_key="query-contract-001",
        ast=ast,
        parameters=parameters,
        input_hash="a" * 64,
        query_hash=FixtureDataSource.query_contract_hash(ast, parameters),
    )
    assert result.status == "success"
    assert result.raw_row_count == 103
    assert len(result.sanitized_rows) == 103
    assert all("private_sentinel" not in row for row in result.sanitized_rows)
    deduped = result.deduplicate()
    assert len(deduped) == 100
    assert len({row["trace_pk"] for row in deduped}) == 100
    assert {row["report_id"] for row in deduped} >= {"report-100", "report-101", "report-102"}


def test_fixture_config_rejects_credentials_and_bool_limits() -> None:
    with pytest.raises(DataContractError):
        FixtureDataIngestConfig.from_mapping(
            {
                "credential_ref": "must-not-be-accepted",
                "query_sql": APPROVED_SQL,
                "run_id": "bad",
                "window": {
                    "end_utc": "2026-01-02T00:00:00Z",
                    "start_utc": "2026-01-01T00:00:00Z",
                },
            }
        )
    with pytest.raises(DataContractError):
        FixtureDataIngestConfig(
            run_id="bad-bool",
            query_sql=APPROVED_SQL,
            window=QueryWindow("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
            retry_limit=True,
        )
