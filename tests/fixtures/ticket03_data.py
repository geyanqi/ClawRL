"""Allowlisted Data Provider exchange for Ticket 03 tests."""

from __future__ import annotations

import json
from pathlib import Path

from clawrl.adapters.sources.fixture import DataProviderIngressReceipt, FixtureDataSource
from clawrl.artifacts import canonical_json_bytes

PRIVATE_SENTINEL = "T03_PRIVATE_SENTINEL_NEVER_EXPORT"

# Exact role bytes returned by isolated task /root/ticket03_data_provider.
DATA_PROVIDER_RAW = (
    b'{"schema_version":"data-provider-output/1.0.0","packet_id":"dp-ticket03-0001","seed":3001,'
    b'"source_id":"fixture-online-traces-v1","table":"fixture.online_trace","columns":["trace_pk",'
    b'"report_id","event_time_utc","ingestion_time_utc","purpose","prompt","response","tool_name",'
    b'"model_id","private_sentinel"],"row_count":103,"unique_key_count":100,"duplicate_count":3,'
    b'"purpose":"training_allowed","field_mapping":{"prompt":"prompt","response":"response",'
    b'"tool":"tool_name","model":"model_id","event_time":"event_time_utc","ingestion_time":'
    b'"ingestion_time_utc","primary_key":"trace_pk","dedupe_key":"trace_pk"},"window":{"start_utc":'
    b'"2026-01-01T00:00:00Z","end_utc":"2026-01-02T00:00:00Z"},"dedupe_semantics":'
    b'"Group by trace_pk and retain the row with the greatest ingestion_time_utc; mapped content after '
    b'removing private_sentinel must match across reports for the same trace_pk.","sanitizer_rule":'
    b'"Validate private_sentinel equals the declared sentinel on every raw row, then delete '
    b'private_sentinel before producing any adapter output.","sentinel":"T03_PRIVATE_SENTINEL_NEVER_EXPORT",'
    b'"row_formula":{"language":"javascript","source":"({seed,duplicate_overrides})=>{const pad=n=>String(n)'
    b".padStart(3,'0');const iso=s=>new Date(Date.parse('2026-01-01T00:00:00Z')+s*1000).toISOString();"
    b"const make=(i,report_id,lag)=>{const z=pad(i);const event=(i+1)*600;return {trace_pk:`trace-${z}`,"
    b"report_id,event_time_utc:iso(event),ingestion_time_utc:iso(event+lag),purpose:'training_allowed',"
    b"prompt:`seed-${seed}-prompt-${z}-code-${seed+i*17}`,response:`seed-${seed}-response-${z}-value-"
    b"${seed*3+i*29}`,tool_name:['python','regex','none','javascript'][i%4],model_id:['fixture-model-a',"
    b"'fixture-model-b'][i%2],private_sentinel:'T03_PRIVATE_SENTINEL_NEVER_EXPORT'};};return "
    b"[...Array.from({length:100},(_,i)=>make(i,`report-${pad(i)}`,30+(i%7))),...duplicate_overrides"
    b'.map(d=>make(d.logical_index,d.report_id,d.ingestion_lag_seconds))];}"},"duplicate_overrides":'
    b'[{"logical_index":7,"trace_pk":"trace-007","report_id":"report-100","ingestion_lag_seconds":3600},'
    b'{"logical_index":42,"trace_pk":"trace-042","report_id":"report-101","ingestion_lag_seconds":3601},'
    b'{"logical_index":99,"trace_pk":"trace-099","report_id":"report-102","ingestion_lag_seconds":3602}],'
    b'"invariants":["Executing row_formula with seed=3001 and duplicate_overrides yields exactly 103 raw '
    b'rows and uses no randomness.","The raw rows contain exactly 100 distinct trace_pk values, trace-000 '
    b'through trace-099, and only trace-007, trace-042, and trace-099 occur twice.","Every raw row has '
    b'purpose equal to training_allowed and private_sentinel equal to T03_PRIVATE_SENTINEL_NEVER_EXPORT.",'
    b'"For logical indices 0 through 99, event_time_utc is valid UTC, lies strictly inside the declared '
    b'window, and is strictly increasing by logical index.","Every ingestion_time_utc is later than its '
    b"event_time_utc; each duplicate ingestion_time_utc is later than the original row for its trace_pk, "
    b'so highest-ingestion deduplication selects the duplicate report.","For duplicate trace_pk values, '
    b"purpose, prompt, response, tool_name, model_id, and event_time_utc are identical after "
    b'private_sentinel is removed; report_id and ingestion_time_utc are operational report metadata.",'
    b'"After validation and sanitization, private_sentinel is absent from every adapter output.",'
    b'"Prompts and responses are deterministic, nonconstant, and unique across the 100 distinct trace_pk '
    b'values."]}'
)

DATA_PROVIDER_INPUT_PACKET = {
    "packet_id": "dp-ticket03-0001",
    "request": {
        "required_purpose": "training_allowed",
        "required_raw_rows": 103,
        "required_unique_keys": 100,
        "window_end_utc": "2026-01-02T00:00:00Z",
        "window_start_utc": "2026-01-01T00:00:00Z",
    },
    "schema_version": "data-provider-input/1.0.0",
    "seed": 3001,
}

DATA_PROVIDER_CORRECTION_RAW = (
    b'{"schema_version":"data-provider-correction-output/1.0.0","packet_id":"dp-ticket03-correction-0001",'
    b'"prior_packet_id":"dp-ticket03-0001","seed":3001,"source_id":"fixture-online-traces-v1",'
    b'"corrected_field_mapping":{"primary_key":"report_id","logical_trace_key":"trace_pk",'
    b'"dedupe_key":"trace_pk"},"report_identity_rule":"report_id is unique for every raw report; trace_pk may '
    b'repeat and is the logical TrainingTrace identity after highest-ingestion dedupe","dedupe_semantics":'
    b'"highest ingestion_time_utc wins for the same trace_pk; sanitized semantic content must match",'
    b'"reason_code":"SOURCE_KEY_CONTRACT_CORRECTED","invariants":{"unique_report_id_count":103,'
    b'"unique_trace_pk_count":100,"duplicate_logical_keys":["trace-007","trace-042","trace-099"],'
    b'"raw_row_content_unchanged":true,"window_unchanged":true,"sanitizer_unchanged":true,'
    b'"formula_unchanged":true}}'
)

DATA_PROVIDER_CORRECTION_INPUT_PACKET = {
    "packet_id": "dp-ticket03-correction-0001",
    "prior_output_hash": "e38025307baa56c095df829a7a501cbf8ecf844435e7f3dab873a4cae55add74",
    "request": {
        "dedupe_key": "trace_pk",
        "logical_trace_key": "trace_pk",
        "primary_key": "report_id",
        "reason_code": "SOURCE_KEY_CONTRACT_CORRECTION_REQUIRED",
    },
    "schema_version": "data-provider-correction-input/1.0.0",
    "seed": 3001,
}


def canonical_provider_output() -> bytes:
    """Return the canonical normalization while preserving exact role bytes separately."""

    decoded = json.loads(DATA_PROVIDER_RAW)
    return canonical_json_bytes(decoded)


def stage_data_provider(root: Path) -> DataProviderIngressReceipt:
    return FixtureDataSource.stage_private_role_exchange(
        root,
        input_packet=DATA_PROVIDER_INPUT_PACKET,
        raw_role_output=DATA_PROVIDER_RAW,
        role_session_lineage="/root/ticket03_data_provider",
        correction_input_packet=DATA_PROVIDER_CORRECTION_INPUT_PACKET,
        correction_raw_role_output=DATA_PROVIDER_CORRECTION_RAW,
        correction_role_session_lineage="/root/ticket03_data_provider_correction",
    )
