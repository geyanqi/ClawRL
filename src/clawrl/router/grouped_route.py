"""Bounded fixture Judge-tool execution for Ticket 15.

The implementation intentionally exposes a tiny, read-only helper surface.  A
tool request is an untrusted proposal; only the allowlisted helper languages
are evaluated against a de-identified batch and ephemeral scratch.  Every
decision, output, reward and trace is an immutable :class:`ArtifactStore`
record so a fresh process can resume without session memory.
"""

from __future__ import annotations

import ast
import copy
import json
import re
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from clawrl.artifacts import Artifact, ArtifactCorruption, ArtifactStore, JsonValue, canonical_json_bytes, sha256_hex
from clawrl.router.trace_router import (
    FixtureTraceRouter,
    RouterCapacityConfig,
    TraceRouterConfig,
)

_HASH = re.compile(r"^[0-9a-f]{64}$")
_TOOLS = {"python", "javascript", "regex"}
_DENY_WORDS = ("credential", "secret", "token", "password", "network", "socket", "mount", "raw_payload")
_MAX_REGEX_PATTERN_BYTES = 4 * 1024
_QUANTIFIER_CHARS = frozenset("*+?")
_UNSAFE_REGEX_CHARS = frozenset("*+?|{}")


class GroupedToolRouteError(RuntimeError):
    """A malformed request or unverifiable durable route failed closed."""


class InjectedGroupedRouteCrash(GroupedToolRouteError):
    """Synthetic crash after a durable group commit."""


@dataclass(frozen=True, slots=True)
class ToolSandboxLimits:
    cpu_ms: int = 50
    memory_bytes: int = 64 * 1024
    wall_time_ms: int = 100
    output_bytes: int = 16 * 1024

    def __post_init__(self) -> None:
        if any(
            type(v) is not int or v <= 0 for v in (self.cpu_ms, self.memory_bytes, self.wall_time_ms, self.output_bytes)
        ):
            raise GroupedToolRouteError("sandbox limits must be positive integers")


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """Untrusted helper proposal.  ``program`` is never executed as a shell."""

    tool: str
    program: str
    uid: str | None = None
    global_step: int | None = None
    scratch_key: str = "default"
    request_id: str = "tool-request"

    def __post_init__(self) -> None:
        if type(self.tool) is not str or type(self.program) is not str:
            raise GroupedToolRouteError("tool and program must be strings")
        if type(self.request_id) is not str or type(self.scratch_key) is not str:
            raise GroupedToolRouteError("request identity fields must be strings")

    @property
    def language(self) -> str:
        return self.tool


@dataclass(frozen=True, slots=True)
class GroupedRouteConfig:
    run_id: str
    global_step: int
    uid: str
    resolver_epoch: int = 1
    capacity: RouterCapacityConfig = field(default_factory=lambda: RouterCapacityConfig(5))
    limits: ToolSandboxLimits = field(default_factory=ToolSandboxLimits)
    fault_group: int | None = None

    def router_config(self) -> TraceRouterConfig:
        return TraceRouterConfig(self.run_id, self.global_step, self.uid, self.resolver_epoch, self.capacity)


def _safe_json(value: object) -> JsonValue:
    try:
        encoded = canonical_json_bytes(value)
        decoded = json.loads(encoded.decode("utf-8"))
    except Exception as error:  # canonicalization is the boundary
        raise GroupedToolRouteError("helper output is not canonical JSON") from error
    if not isinstance(decoded, (dict, list, str, int, bool)) and decoded is not None:
        raise GroupedToolRouteError("helper output is not canonical JSON")
    return cast(JsonValue, decoded)


def _has_nested_quantifier(pattern: str) -> bool:
    """Reject a quantified expression containing another quantifier.

    This small scanner avoids compiling potentially catastrophic expressions and
    handles escaped characters, character classes, and nested groups without
    depending on CPython's private regex parser.
    """

    groups: list[bool] = []
    escaped = False
    in_class = False
    for index, character in enumerate(pattern):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "[" and not in_class:
            in_class = True
            continue
        if character == "]" and in_class:
            in_class = False
            continue
        if in_class:
            continue
        if character == "(":
            groups.append(False)
            continue
        if character == ")" and groups:
            contained_quantifier = groups.pop()
            # A group prefix such as (?:...) is not a quantifier.  Quantifiers
            # immediately following a closed group are, however, dangerous.
            next_character = pattern[index + 1] if index + 1 < len(pattern) else ""
            quantified_group = next_character in _QUANTIFIER_CHARS or next_character == "{"
            if contained_quantifier and quantified_group:
                return True
            if groups:
                groups[-1] = groups[-1] or quantified_group or contained_quantifier
            continue
        if character in _QUANTIFIER_CHARS or (
            character == "{" and index + 1 < len(pattern) and pattern[index + 1].isdigit()
        ):
            # Ignore a non-capturing/lookaround group prefix (?..., which is
            # syntactic metadata rather than a repetition operator.
            if groups and not (character == "?" and index > 0 and pattern[index - 1] == "("):
                groups[-1] = True
    return False


def _has_unsafe_regex_construct(pattern: str) -> bool:
    """Reject constructs whose backtracking cost cannot be bounded safely.

    The fixture adapter deliberately supports a conservative, literal-oriented
    regex subset.  Unbounded/ambiguous repetition and alternation are denied
    before compilation; this is fail-closed when the host has no regex timeout
    primitive and prevents a malicious helper from monopolising the process.
    """

    escaped = False
    for character in pattern:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character in _UNSAFE_REGEX_CHARS:
            return True
    return False


def _validated_hash_list(payload: dict[str, JsonValue], field: str) -> list[str]:
    value = payload.get(field)
    if (
        not isinstance(value, list)
        or any(type(item) is not str or _HASH.fullmatch(item) is None for item in value)
        or len(set(value)) != len(value)
    ):
        raise GroupedToolRouteError(f"grouped tool {field} lineage is invalid")
    return cast(list[str], value)


class FixtureHelperSandbox:
    """Read-only deterministic helper evaluator with bounded resource checks."""

    def __init__(
        self, *, uid: str, global_step: int, batch: list[dict[str, JsonValue]], limits: ToolSandboxLimits | None = None
    ) -> None:
        self.uid = uid
        self.global_step = global_step
        self.batch = copy.deepcopy(batch)
        self.scratch: dict[str, JsonValue] = {}
        self.limits = limits or ToolSandboxLimits()

    def execute(self, request: ToolRequest) -> tuple[str, JsonValue, str]:
        started = time.monotonic_ns()
        if not isinstance(request.tool, str):
            return "deny", None, "TOOL_NOT_ALLOWLISTED"
        tool = request.tool.lower()
        if tool not in _TOOLS:
            return "deny", None, "TOOL_NOT_ALLOWLISTED"
        if request.uid is not None and request.uid != self.uid:
            return "deny", None, "UID_NOT_CURRENT"
        if request.global_step is not None and request.global_step != self.global_step:
            return "deny", None, "STEP_NOT_CURRENT"
        if not isinstance(request.program, str) or len(request.program.encode("utf-8")) > self.limits.memory_bytes:
            return "deny", None, "PROGRAM_TOO_LARGE"
        lowered = request.program.lower()
        if any(word in lowered for word in _DENY_WORDS):
            return "deny", None, "CAPABILITY_NOT_GRANTED"
        try:
            if tool == "python":
                value = self._python(request.program)
            elif tool == "javascript":
                value = self._javascript(request.program)
            else:
                value = self._regex(request.program)
            value = _safe_json(value)
            output_size = len(canonical_json_bytes(value))
            elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
            if output_size > self.limits.output_bytes:
                return "deny", None, "OUTPUT_LIMIT"
            if elapsed_ms > self.limits.wall_time_ms:
                return "deny", None, "TIME_LIMIT"
            return "allow", value, "POLICY_ALLOWED"
        except GroupedToolRouteError as error:
            reason = (
                str(error)
                if str(error) in {"CPU_LIMIT", "OUTPUT_LIMIT", "TIME_LIMIT", "PROGRAM_TOO_LARGE"}
                else "HELPER_ERROR"
            )
            return "deny", None, reason
        except Exception:
            return "deny", None, "HELPER_ERROR"

    def put_scratch(self, key: str, value: JsonValue) -> None:
        """Store one bounded ephemeral value; it is never published as a trace."""

        if not isinstance(key, str) or not key or len(key) > 128:
            raise GroupedToolRouteError("scratch key is invalid")
        candidate = dict(self.scratch)
        candidate[key] = copy.deepcopy(value)
        if len(canonical_json_bytes(candidate)) > self.limits.memory_bytes:
            raise GroupedToolRouteError("MEMORY_LIMIT")
        self.scratch = candidate

    def _python(self, source: str) -> object:
        tree = ast.parse(source, mode="eval")
        if len(list(ast.walk(tree))) * max(1, len(self.batch)) > self.limits.cpu_ms * 1_000:
            raise GroupedToolRouteError("CPU_LIMIT")
        allowed_names = {"batch", "scratch", "len", "sum", "min", "max", "sorted", "str", "int", "bool"}
        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.Attribute,
                    ast.Lambda,
                    ast.NamedExpr,
                    ast.ListComp,
                    ast.DictComp,
                    ast.SetComp,
                    ast.GeneratorExp,
                    ast.BinOp,
                ),
            ):
                raise GroupedToolRouteError("python construct is not allowlisted")
            if isinstance(node, ast.Name) and node.id not in allowed_names:
                raise GroupedToolRouteError("python name is not allowlisted")
            if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
                raise GroupedToolRouteError("python call is not allowlisted")
        return eval(
            compile(tree, "<fixture-helper>", "eval"),
            {"__builtins__": {}},
            {  # noqa: S307 - AST allowlist above
                "batch": copy.deepcopy(self.batch),
                "scratch": copy.deepcopy(self.scratch),
                "len": len,
                "sum": sum,
                "min": min,
                "max": max,
                "sorted": sorted,
                "str": str,
                "int": int,
                "bool": bool,
            },
        )

    def _javascript(self, source: str) -> object:
        text = source.strip().rstrip(";")
        if text in {"batch.length", "batch.length()"}:
            return len(self.batch)
        if text in {"JSON.stringify(batch)", "JSON.stringify(scratch)"}:
            return json.dumps(self.batch if "batch" in text else self.scratch, sort_keys=True, separators=(",", ":"))
        match = re.fullmatch(r"(?:batch|scratch)\[['\"]([A-Za-z0-9_.-]+)['\"]\]", text)
        if match:
            key = match.group(1)
            if text.startswith("batch"):
                return [row.get(key) for row in self.batch]
            return self.scratch.get(key)
        raise GroupedToolRouteError("javascript expression is not allowlisted")

    def _regex(self, source: str) -> object:
        # Regex proposals use ``pattern`` or ``pattern => field`` and only see
        # the de-identified batch JSON, never the raw request payload.
        pattern = source.strip()
        field = None
        if "=>" in pattern:
            pattern, field = (item.strip() for item in pattern.split("=>", 1))
        pattern_size = len(pattern.encode("utf-8"))
        if pattern_size > min(self.limits.memory_bytes, _MAX_REGEX_PATTERN_BYTES):
            raise GroupedToolRouteError("PROGRAM_TOO_LARGE")
        if _has_unsafe_regex_construct(pattern) or _has_nested_quantifier(pattern):
            raise GroupedToolRouteError("regex nested quantifier is not allowlisted")
        if len(pattern.encode("utf-8")) * max(1, len(self.batch)) > self.limits.cpu_ms * 1_000:
            raise GroupedToolRouteError("CPU_LIMIT")
        rx = re.compile(pattern)
        values = []
        for row in self.batch:
            target = row if field is None else row.get(field)
            values.append(bool(rx.search(json.dumps(target, sort_keys=True))))
        return values


class FixtureGroupedToolRouter:
    """Execute bounded helper proposals alongside Ticket 14 grouped scoring."""

    @staticmethod
    def _compressed_fields(payload: dict[str, JsonValue], store: ArtifactStore) -> dict[str, JsonValue]:
        blob_hash, blob_size = store.put_blob(zlib.compress(canonical_json_bytes(payload), level=9))
        return {
            "compressed": True,
            "compression": "zlib",
            "compressed_blob_hash": blob_hash,
            "compressed_size": blob_size,
        }

    @staticmethod
    def _verify_compressed_payload(
        store: ArtifactStore,
        artifact: Artifact,
        expected_payload: JsonValue,
    ) -> None:
        blob_hash = artifact.payload.get("compressed_blob_hash")
        blob_size = artifact.payload.get("compressed_size")
        if type(blob_hash) is not str or type(blob_size) is not int or blob_size <= 0:
            raise GroupedToolRouteError("grouped tool compression lineage is invalid")
        try:
            compressed = store.read_blob(blob_hash, expected_size=blob_size)
            decoded = zlib.decompress(compressed)
        except (ArtifactCorruption, zlib.error) as error:
            raise GroupedToolRouteError("grouped tool compressed payload is unavailable") from error
        if decoded != canonical_json_bytes(expected_payload):
            raise GroupedToolRouteError("grouped tool compressed payload does not match artifact")

    @classmethod
    def run(
        cls,
        root: str | Path,
        *,
        config: GroupedRouteConfig | TraceRouterConfig,
        expected_set: Artifact,
        judge_pack: Artifact,
        requests: tuple[ToolRequest, ...] | list[ToolRequest] | None = None,
        crash_after_group: int | None = None,
    ) -> Artifact:
        if isinstance(config, TraceRouterConfig):
            config = GroupedRouteConfig(
                config.run_id,
                config.global_step,
                config.uid,
                config.resolver_epoch,
                config.capacity,
            )
        store = ArtifactStore(root)
        namespace = (
            Path(root)
            / "grouped-tool-routes"
            / config.run_id
            / str(config.global_step)
            / config.uid
            / str(config.resolver_epoch)
        )
        report_ref = namespace / "report.ref"
        if report_ref.exists():
            recovered = cls.resume(root, config.run_id, config.global_step, config.uid, config.resolver_epoch)
            base_report = store.read(
                cast(str, recovered.payload["result_hash"]), expected_schema_name="TraceRouterReport"
            )
            if (
                base_report.payload.get("expected_set_hash") != expected_set.content_hash
                or base_report.payload.get("judge_pack_hash") != judge_pack.content_hash
            ):
                raise GroupedToolRouteError("grouped tool request identity changed")
            if requests is not None:
                existing_decisions = [
                    store.read(value, expected_schema_name="HarnessDecision")
                    for value in cast(list[str], recovered.payload["decision_hashes"])
                ]
                existing_proposals = [
                    store.read(cast(str, item.payload["proposal_hash"]), expected_schema_name="ToolProposal")
                    for item in existing_decisions
                ]
                if len(existing_proposals) != len(requests):
                    raise GroupedToolRouteError("grouped tool request identity changed")
                for request, proposal in zip(requests, existing_proposals, strict=True):
                    program_hash = (
                        sha256_hex(request.program.encode())
                        if isinstance(request.program, str)
                        else sha256_hex(canonical_json_bytes(request.program))
                    )
                    if any(
                        proposal.payload.get(field) != expected
                        for field, expected in {
                            "global_step": config.global_step,
                            "program_hash": program_hash,
                            "request_id": request.request_id,
                            "requested_global_step": request.global_step,
                            "requested_uid": request.uid,
                            "run_id": config.run_id,
                            "scratch_key": request.scratch_key,
                            "tool": request.tool,
                            "uid": config.uid,
                        }.items()
                    ):
                        raise GroupedToolRouteError("grouped tool request identity changed")
            return recovered
        base = FixtureTraceRouter.run(
            root, config=config.router_config(), expected_set=expected_set, judge_pack=judge_pack
        )
        batch = [
            {
                "trajectory_hash": result.payload["trajectory_manifest_hash"],
                "scalar_micros": result.payload["scalar_micros"],
            }
            for result in base.results
        ]
        sandbox = FixtureHelperSandbox(
            uid=config.uid, global_step=config.global_step, batch=batch, limits=config.limits
        )
        proposals = (
            tuple(requests)
            if requests is not None
            else (
                ToolRequest("python", "len(batch)", request_id="approved-python"),
                ToolRequest("javascript", "batch.length", request_id="approved-javascript"),
                ToolRequest("regex", "scalar_micros=>scalar_micros", request_id="approved-regex"),
                ToolRequest("python", "len(batch)", uid="other-uid", request_id="injected-overreach"),
            )
        )

        output_hashes: list[str] = []
        decision_hashes: list[str] = []
        for index, request in enumerate(proposals):
            if not isinstance(request.program, str):
                # Defensive boundary for deserialised/untyped callers that
                # bypass ToolRequest.__post_init__.
                program_hash = sha256_hex(canonical_json_bytes(request.program))
            else:
                program_hash = sha256_hex(request.program.encode())
            proposal_payload: dict[str, JsonValue] = {
                "global_step": config.global_step,
                "program_hash": program_hash,
                "request_id": request.request_id,
                "run_id": config.run_id,
                "requested_global_step": request.global_step,
                "requested_uid": request.uid,
                "scratch_key": request.scratch_key,
                "tool": request.tool,
                "uid": config.uid,
            }
            proposal_payload.update(cls._compressed_fields(proposal_payload, store))
            proposal = store.put("ToolProposal", "1.0.0", proposal_payload)
            status, value, reason = sandbox.execute(request)
            decision_payload: dict[str, JsonValue] = {
                "decision": status,
                "policy": "fixture-judge-tools/1.0.0",
                "proposal_hash": proposal.content_hash,
                "reason_code": reason,
                "side_effects": False,
                "uid": config.uid,
            }
            decision_payload.update(cls._compressed_fields(decision_payload, store))
            decision = store.put("HarnessDecision", "1.0.0", decision_payload)
            decision_hashes.append(decision.content_hash)
            output_payload: dict[str, JsonValue] = {
                "decision_hash": decision.content_hash,
                "output": value,
                "status": status,
                "output_hash": sha256_hex(canonical_json_bytes(value)),
                "proposal_hash": proposal.content_hash,
            }
            compressed_output_hash, compressed_output_size = store.put_blob(
                zlib.compress(canonical_json_bytes(value), level=9)
            )
            output_payload.update(
                {
                    "compressed": True,
                    "compression": "zlib",
                    "compressed_blob_hash": compressed_output_hash,
                    "compressed_size": compressed_output_size,
                }
            )
            output = store.put("ToolOutput", "1.0.0", output_payload)
            output_hashes.append(output.content_hash)
            if crash_after_group is not None and index == crash_after_group:
                raise InjectedGroupedRouteCrash("injected crash after durable helper group")
        rewards: list[str] = []
        for result in base.results:
            reward_payload: dict[str, JsonValue] = {
                "reward_micros": result.payload["scalar_micros"],
                "valid": True,
                "judge_result_hash": result.content_hash,
                "tool_output_hashes": cast(JsonValue, output_hashes),
            }
            reward_payload.update(cls._compressed_fields(reward_payload, store))
            reward = store.put("GroupedReward", "1.0.0", reward_payload)
            rewards.append(reward.content_hash)
        trace_data = canonical_json_bytes({"decisions": decision_hashes, "outputs": output_hashes, "rewards": rewards})
        trace_blob, trace_blob_size = store.put_blob(zlib.compress(trace_data, level=9))
        trace = store.put(
            "CodexTrace",
            "1.0.0",
            {
                "compressed": True,
                "compression": "zlib",
                "content_hash": trace_blob,
                "content_size": trace_blob_size,
                "decision_hashes": decision_hashes,
                "output_hashes": output_hashes,
                "reward_hashes": rewards,
            },
        )
        report = store.put(
            "GroupedToolRouteReport",
            "1.0.0",
            {
                "codex_trace_hash": trace.content_hash,
                "decision_hashes": decision_hashes,
                "expected_set_hash": expected_set.content_hash,
                "judge_pack_hash": judge_pack.content_hash,
                "output_hashes": output_hashes,
                "reward_hashes": rewards,
                "result_hash": base.report.content_hash,
                "run_id": config.run_id,
                "global_step": config.global_step,
                "uid": config.uid,
                "status": "resolved",
                "production_readiness": "blocked",
            },
        )
        ArtifactStore.durable_mkdir(namespace)
        # Publish-if-absent: a concurrent request with a different proposal
        # set must surface an immutable conflict instead of replacing the
        # durable pointer underneath a running/fresh process.
        ArtifactStore._publish(report_ref, f"{report.content_hash}\n".encode("ascii"))
        return report

    @classmethod
    def resume(cls, root: str | Path, run_id: str, global_step: int, uid: str, resolver_epoch: int = 1) -> Artifact:
        store = ArtifactStore(root)
        namespace = Path(root) / "grouped-tool-routes" / run_id / str(global_step) / uid / str(resolver_epoch)
        try:
            report = store.read(
                namespace.joinpath("report.ref").read_text().strip(), expected_schema_name="GroupedToolRouteReport"
            )
            expected_fields = {
                "codex_trace_hash",
                "decision_hashes",
                "expected_set_hash",
                "global_step",
                "judge_pack_hash",
                "output_hashes",
                "production_readiness",
                "result_hash",
                "reward_hashes",
                "run_id",
                "status",
                "uid",
            }
            if report.schema_version != "1.0.0" or set(report.payload) != expected_fields:
                raise GroupedToolRouteError("grouped tool report schema is invalid")
            if (
                report.payload.get("run_id") != run_id
                or report.payload.get("global_step") != global_step
                or report.payload.get("uid") != uid
                or report.payload.get("status") != "resolved"
                or report.payload.get("production_readiness") != "blocked"
            ):
                raise GroupedToolRouteError("grouped tool report step lineage is invalid")
            decision_hashes = _validated_hash_list(report.payload, "decision_hashes")
            output_hashes = _validated_hash_list(report.payload, "output_hashes")
            reward_hashes = _validated_hash_list(report.payload, "reward_hashes")
            trace = store.read(cast(str, report.payload["codex_trace_hash"]), expected_schema_name="CodexTrace")
            base_report = store.read(cast(str, report.payload["result_hash"]), expected_schema_name="TraceRouterReport")
            try:
                base_snapshot = FixtureTraceRouter.resume(root, run_id, global_step, uid)
            except Exception as error:
                raise GroupedToolRouteError("grouped tool base route cannot be recovered") from error
            if base_snapshot.report.content_hash != base_report.content_hash:
                raise GroupedToolRouteError("grouped tool base route identity changed")
            expected_set_hash = report.payload.get("expected_set_hash")
            judge_pack_hash = report.payload.get("judge_pack_hash")
            if (
                type(expected_set_hash) is not str
                or _HASH.fullmatch(expected_set_hash) is None
                or type(judge_pack_hash) is not str
                or _HASH.fullmatch(judge_pack_hash) is None
                or base_report.payload.get("expected_set_hash") != expected_set_hash
                or base_report.payload.get("judge_pack_hash") != judge_pack_hash
            ):
                raise GroupedToolRouteError("grouped tool expected/judge identity is invalid")
            expected_set = store.read(expected_set_hash, expected_schema_name="ExpectedTrajectorySet")
            judge_pack = store.read(judge_pack_hash, expected_schema_name="JudgePack")
            if (
                expected_set.payload.get("run_id") != run_id
                or expected_set.payload.get("global_step") != global_step
                or judge_pack.payload.get("uid") != uid
            ):
                raise GroupedToolRouteError("grouped tool expected/judge identity changed")
            decisions = [store.read(value, expected_schema_name="HarnessDecision") for value in decision_hashes]
            proposals = [
                store.read(cast(str, item.payload["proposal_hash"]), expected_schema_name="ToolProposal")
                for item in decisions
            ]
            outputs = [store.read(value, expected_schema_name="ToolOutput") for value in output_hashes]
            rewards = [store.read(value, expected_schema_name="GroupedReward") for value in reward_hashes]
            if len(decisions) != len(outputs) or len(rewards) != cast(int, base_report.payload.get("result_count", -1)):
                raise GroupedToolRouteError("grouped tool artifact cardinality is invalid")
            expected_decision_fields = {
                "compressed",
                "compression",
                "compressed_blob_hash",
                "compressed_size",
                "decision",
                "policy",
                "proposal_hash",
                "reason_code",
                "side_effects",
                "uid",
            }
            expected_proposal_fields = {
                "global_step",
                "program_hash",
                "request_id",
                "requested_global_step",
                "requested_uid",
                "run_id",
                "scratch_key",
                "tool",
                "uid",
                "compressed",
                "compression",
                "compressed_blob_hash",
                "compressed_size",
            }
            expected_output_fields = {
                "compressed",
                "compression",
                "compressed_blob_hash",
                "compressed_size",
                "decision_hash",
                "output",
                "output_hash",
                "proposal_hash",
                "status",
            }
            expected_reward_fields = {
                "compressed",
                "compression",
                "compressed_blob_hash",
                "compressed_size",
                "judge_result_hash",
                "reward_micros",
                "tool_output_hashes",
                "valid",
            }
            if (
                any(
                    item.schema_version != "1.0.0" or set(item.payload) != expected_proposal_fields
                    for item in proposals
                )
                or any(item.schema_version != "1.0.0" for item in decisions + outputs + rewards)
                or any(set(item.payload) != expected_decision_fields for item in decisions)
                or any(set(item.payload) != expected_output_fields for item in outputs)
                or any(set(item.payload) != expected_reward_fields for item in rewards)
            ):
                raise GroupedToolRouteError("grouped tool artifact schema is invalid")
            decisions_by_hash = {item.content_hash: item for item in decisions}
            if any(
                item.payload.get("run_id") != run_id
                or item.payload.get("global_step") != global_step
                or item.payload.get("uid") != uid
                or type(item.payload.get("program_hash")) is not str
                or _HASH.fullmatch(cast(str, item.payload.get("program_hash"))) is None
                or type(item.payload.get("request_id")) is not str
                or type(item.payload.get("scratch_key")) is not str
                or item.payload.get("requested_uid") is not None
                and type(item.payload.get("requested_uid")) is not str
                or item.payload.get("requested_global_step") is not None
                and type(item.payload.get("requested_global_step")) is not int
                or type(item.payload.get("tool")) is not str
                for item in proposals
            ):
                raise GroupedToolRouteError("grouped tool proposal lineage is invalid")
            if any(
                item.payload.get("proposal_hash") != proposals[index].content_hash or item.payload.get("uid") != uid
                for index, item in enumerate(decisions)
            ):
                raise GroupedToolRouteError("grouped tool decision lineage is invalid")
            if any(
                item.payload.get("decision") not in {"allow", "deny"}
                or item.payload.get("policy") != "fixture-judge-tools/1.0.0"
                or type(item.payload.get("reason_code")) is not str
                or item.payload.get("side_effects") is not False
                for item in decisions
            ):
                raise GroupedToolRouteError("grouped tool decision policy is invalid")
            expected_trace_fields = {
                "compressed",
                "compression",
                "content_hash",
                "content_size",
                "decision_hashes",
                "output_hashes",
                "reward_hashes",
            }
            if (
                set(trace.payload) != expected_trace_fields
                or trace.payload.get("decision_hashes") != [item.content_hash for item in decisions]
                or trace.payload.get("output_hashes") != [item.content_hash for item in outputs]
                or trace.payload.get("reward_hashes") != [item.content_hash for item in rewards]
                or trace.payload.get("compressed") is not True
                or trace.payload.get("compression") != "zlib"
                or base_report.payload.get("status") != "resolved"
                or not isinstance(base_report.payload.get("expected_set_hash"), str)
                or not isinstance(base_report.payload.get("judge_pack_hash"), str)
                or any(item.payload.get("decision_hash") not in {x.content_hash for x in decisions} for item in outputs)
                or any(item.payload.get("proposal_hash") is None for item in decisions + outputs)
            ):
                raise GroupedToolRouteError("grouped tool artifact lineage is invalid")
            if any(
                item.payload.get("proposal_hash")
                != decisions_by_hash[cast(str, item.payload["decision_hash"])].payload.get("proposal_hash")
                or item.payload.get("status")
                != decisions_by_hash[cast(str, item.payload["decision_hash"])].payload.get("decision")
                for item in outputs
            ):
                raise GroupedToolRouteError("grouped tool output proposal lineage is invalid")
            if any(
                item.payload.get("output_hash") != sha256_hex(canonical_json_bytes(item.payload.get("output")))
                for item in outputs
            ):
                raise GroupedToolRouteError("grouped tool output hash lineage is invalid")
            result_hashes: list[str] = []
            for wave_hash in _validated_hash_list(base_report.payload, "wave_hashes"):
                wave = store.read(wave_hash, expected_schema_name="RouterWave")
                result_hashes.extend(_validated_hash_list(wave.payload, "result_hashes"))
            if (
                len(result_hashes) != cast(int, base_report.payload.get("result_count", -1))
                or len(set(result_hashes)) != len(result_hashes)
                or [item.payload.get("judge_result_hash") for item in rewards] != result_hashes
                or any(
                    item.payload.get("tool_output_hashes") != output_hashes
                    or item.payload.get("judge_result_hash") not in result_hashes
                    or item.payload.get("valid") is not True
                    or type(item.payload.get("reward_micros")) is not int
                    for item in rewards
                )
                or any(
                    reward.payload.get("reward_micros") != result.payload.get("scalar_micros")
                    for reward, result in zip(rewards, base_snapshot.results, strict=True)
                )
            ):
                raise GroupedToolRouteError("grouped tool reward lineage is invalid")
            trace_blob_hash = trace.payload.get("content_hash")
            trace_blob_size = trace.payload.get("content_size")
            if (
                type(trace_blob_hash) is not str
                or _HASH.fullmatch(trace_blob_hash) is None
                or type(trace_blob_size) is not int
                or trace_blob_size <= 0
            ):
                raise GroupedToolRouteError("grouped tool trace compression lineage is invalid")
            try:
                compressed_trace = store.read_blob(trace_blob_hash, expected_size=trace_blob_size)
                trace_bytes = zlib.decompress(compressed_trace)
            except (ArtifactCorruption, zlib.error) as error:
                raise GroupedToolRouteError("grouped tool trace blob is unavailable") from error
            decoded_trace = json.loads(trace_bytes.decode("utf-8"))
            if decoded_trace != {
                "decisions": decision_hashes,
                "outputs": output_hashes,
                "rewards": reward_hashes,
            }:
                raise GroupedToolRouteError("grouped tool trace payload is invalid")
            compression_fields = {"compressed", "compression", "compressed_blob_hash", "compressed_size"}
            for artifact in proposals + decisions + rewards:
                expected_payload = {
                    key: value for key, value in artifact.payload.items() if key not in compression_fields
                }
                self_payload = cast(JsonValue, expected_payload)
                cls._verify_compressed_payload(store, artifact, self_payload)
            for output in outputs:
                cls._verify_compressed_payload(store, output, output.payload["output"])
        except (ArtifactCorruption, OSError, ValueError, TypeError, KeyError) as error:
            raise GroupedToolRouteError("grouped tool route cannot be recovered") from error
        if (
            report.payload.get("status") != "resolved"
            or report.payload.get("run_id") != run_id
            or report.payload.get("global_step") != global_step
            or report.payload.get("uid") != uid
        ):
            raise GroupedToolRouteError("grouped tool report lineage is invalid")
        return report

    @staticmethod
    def production_readiness(root: str | Path, phase: str = "TRAIN_35B") -> Artifact:
        return ArtifactStore(root).put(
            "ReadinessReport",
            "1.0.0",
            {
                "phase": phase,
                "execution_profile": "production",
                "status": "blocked",
                "side_effects_permitted": False,
                "checks": [
                    {"code": "TOOL_SANDBOX_WHITELIST_UNAVAILABLE", "status": "blocked"},
                    {"code": "PERMANENT_TRACE_ACL_UNAVAILABLE", "status": "blocked"},
                    {"code": "TRACE_ENCRYPTION_UNAVAILABLE", "status": "blocked"},
                    {"code": "TRACE_CAPACITY_POLICY_UNAVAILABLE", "status": "blocked"},
                    {"code": "TRACE_DELETION_POLICY_UNAVAILABLE", "status": "blocked"},
                    {"code": "INCIDENT_APPROVAL_UNAVAILABLE", "status": "blocked"},
                ],
            },
        )


# Friendly aliases used by integrations and earlier Ticket 15 drafts.
GroupedToolRouter = FixtureGroupedToolRouter
JudgeToolGroupedRoute = FixtureGroupedToolRouter
FixtureJudgeToolRoute = FixtureGroupedToolRouter
FixtureGroupedRoute = FixtureGroupedToolRouter
ToolProposal = ToolRequest
HarnessDecision = Artifact

__all__ = [
    "FixtureHelperSandbox",
    "FixtureGroupedToolRouter",
    "GroupedToolRouter",
    "JudgeToolGroupedRoute",
    "FixtureJudgeToolRoute",
    "FixtureGroupedRoute",
    "GroupedRouteConfig",
    "ToolSandboxLimits",
    "ToolRequest",
    "ToolProposal",
    "GroupedToolRouteError",
    "InjectedGroupedRouteCrash",
]
