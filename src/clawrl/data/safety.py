"""Closed-world public-data safety guards for governed ingestion."""

from __future__ import annotations

import unicodedata

from clawrl.data.models import DataContractError

_SENTINEL_TOKEN = "t03privatesentinelneverexport"


def _normalized_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def assert_public_sentinel_free(value: object) -> None:
    """Reject literal/case/separator variants before they enter public state."""

    pending = [value]
    nodes = 0
    while pending:
        item = pending.pop()
        nodes += 1
        if nodes > 200_000:
            raise DataContractError("public sentinel closure input exceeds structural limits")
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif type(item) is str and _SENTINEL_TOKEN in _normalized_token(item):
            raise DataContractError("public sentinel closure policy rejected a protected value")
