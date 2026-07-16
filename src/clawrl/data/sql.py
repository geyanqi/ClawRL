"""Closed-world tokenizer, parser, and AST policy for governed SELECT queries.

This is intentionally a small SQL language, not a partial regex match.  Every
input byte is tokenized, parsed to an AST, and compared with the frozen
DataSourceSkill policy before a source adapter can be constructed.
"""

from __future__ import annotations

from dataclasses import dataclass

from clawrl.artifacts import JsonValue
from clawrl.data.models import DataSourceSkill


class SqlPolicyError(ValueError):
    """SQL is syntactically invalid or outside the closed-world policy."""


@dataclass(frozen=True, slots=True)
class Token:
    kind: str
    value: str
    offset: int


@dataclass(frozen=True, slots=True)
class FilterAst:
    column: str
    operator: str
    parameter: str

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {"column": self.column, "operator": self.operator, "parameter": self.parameter}


@dataclass(frozen=True, slots=True)
class OrderAst:
    column: str
    direction: str

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {"column": self.column, "direction": self.direction}


@dataclass(frozen=True, slots=True)
class SelectAst:
    columns: tuple[str, ...]
    table: str
    filters: tuple[FilterAst, ...]
    order_by: tuple[OrderAst, ...]

    def artifact_payload(self) -> dict[str, JsonValue]:
        return {
            "columns": list(self.columns),
            "filters": [item.artifact_payload() for item in self.filters],
            "order_by": [item.artifact_payload() for item in self.order_by],
            "statement_type": "SELECT",
            "table": self.table,
        }


_KEYWORDS = {"AND", "ASC", "BY", "DESC", "FROM", "ORDER", "SELECT", "WHERE"}


def tokenize_sql(sql: str) -> tuple[Token, ...]:
    """Tokenize the entire approved SQL subset; unknown bytes fail closed."""

    if not isinstance(sql, str) or not sql:
        raise SqlPolicyError("SQL must be a non-empty string")
    try:
        encoded = sql.encode("utf-8")
    except UnicodeEncodeError as error:
        raise SqlPolicyError("SQL must be valid UTF-8 text") from error
    if len(encoded) > 16_384:
        raise SqlPolicyError("SQL exceeds the governed byte limit")
    tokens: list[Token] = []
    cursor = 0
    while cursor < len(sql):
        character = sql[cursor]
        if character in " \t\r\n":
            cursor += 1
            continue
        if character.isascii() and (character.isalpha() or character == "_"):
            start = cursor
            cursor += 1
            while cursor < len(sql) and sql[cursor].isascii() and (sql[cursor].isalnum() or sql[cursor] == "_"):
                cursor += 1
            value = sql[start:cursor]
            upper = value.upper()
            tokens.append(Token(upper if upper in _KEYWORDS else "IDENT", value, start))
            continue
        if character == ":":
            start = cursor
            cursor += 1
            if cursor >= len(sql) or not sql[cursor].isascii() or not (sql[cursor].isalpha() or sql[cursor] == "_"):
                raise SqlPolicyError(f"invalid parameter token at byte {start}")
            name_start = cursor
            cursor += 1
            while cursor < len(sql) and sql[cursor].isascii() and (sql[cursor].isalnum() or sql[cursor] == "_"):
                cursor += 1
            tokens.append(Token("PARAM", sql[name_start:cursor], start))
            continue
        if character == ">" and cursor + 1 < len(sql) and sql[cursor + 1] == "=":
            tokens.append(Token("OP", ">=", cursor))
            cursor += 2
            continue
        if character in "<=":
            tokens.append(Token("OP", character, cursor))
            cursor += 1
            continue
        if character == ",":
            tokens.append(Token("COMMA", character, cursor))
            cursor += 1
            continue
        if character == ".":
            tokens.append(Token("DOT", character, cursor))
            cursor += 1
            continue
        # This rejects *, ;, quotes, comments, parentheses/functions, and every
        # other operator before parsing or adapter construction.
        raise SqlPolicyError(f"unsupported SQL token at byte {cursor}")
    tokens.append(Token("EOF", "", len(sql)))
    return tuple(tokens)


class _Parser:
    def __init__(self, tokens: tuple[Token, ...]) -> None:
        self.tokens = tokens
        self.index = 0

    def parse(self) -> SelectAst:
        self._take("SELECT")
        columns = self._identifier_list()
        self._take("FROM")
        first = self._identifier()
        self._take("DOT")
        table = f"{first}.{self._identifier()}"
        self._take("WHERE")
        filters = [self._filter()]
        while self._peek().kind == "AND":
            self._take("AND")
            filters.append(self._filter())
        self._take("ORDER")
        self._take("BY")
        order = [self._order_item()]
        while self._peek().kind == "COMMA":
            self._take("COMMA")
            order.append(self._order_item())
        self._take("EOF")
        return SelectAst(tuple(columns), table, tuple(filters), tuple(order))

    def _identifier_list(self) -> list[str]:
        values = [self._identifier()]
        while self._peek().kind == "COMMA":
            self._take("COMMA")
            values.append(self._identifier())
        return values

    def _filter(self) -> FilterAst:
        column = self._identifier()
        operator = self._take("OP").value
        parameter = self._take("PARAM").value
        return FilterAst(column, operator, parameter)

    def _order_item(self) -> OrderAst:
        column = self._identifier()
        direction = self._peek().kind
        if direction not in {"ASC", "DESC"}:
            raise SqlPolicyError(f"ORDER BY direction is required at byte {self._peek().offset}")
        self._take(direction)
        return OrderAst(column, direction)

    def _identifier(self) -> str:
        token = self._take("IDENT")
        return token.value

    def _peek(self) -> Token:
        return self.tokens[self.index]

    def _take(self, kind: str) -> Token:
        token = self._peek()
        if token.kind != kind:
            raise SqlPolicyError(f"expected {kind}, found {token.kind} at byte {token.offset}")
        self.index += 1
        return token


def parse_and_validate_select(sql: str, skill: DataSourceSkill) -> SelectAst:
    """Build an AST and compare every node with the frozen source policy."""

    if not isinstance(skill, DataSourceSkill):
        raise SqlPolicyError("a validated DataSourceSkill is required")
    ast = _Parser(tokenize_sql(sql)).parse()
    if ast.table != skill.table:
        raise SqlPolicyError("query table is not approved")
    if ast.columns != skill.approved_columns or len(set(ast.columns)) != len(ast.columns):
        raise SqlPolicyError("query columns must exactly match the approved projection")
    expected_filters = tuple(_parse_policy_filter(item) for item in skill.allowed_filters)
    if ast.filters != expected_filters:
        raise SqlPolicyError("query filters or parameters do not match the approved policy")
    if ast.order_by != (OrderAst(skill.dedupe_key, "ASC"), OrderAst(skill.ingestion_time_column, "ASC")):
        raise SqlPolicyError("query ordering must make duplicate resolution explicit")
    if skill.allowed_joins:
        raise SqlPolicyError("Ticket 03 source declares no executable join grammar")
    return ast


def _parse_policy_filter(value: str) -> FilterAst:
    """Parse policy declarations without routing user SQL through string matching."""

    parts = value.split(" ")
    if len(parts) != 3 or not parts[2].startswith(":") or len(parts[2]) == 1:
        raise SqlPolicyError("DataSourceSkill contains an invalid filter declaration")
    column, operator, parameter = parts[0], parts[1], parts[2][1:]
    if operator not in {"=", "<", ">="}:
        raise SqlPolicyError("DataSourceSkill contains an unsupported filter operator")
    return FilterAst(column, operator, parameter)
