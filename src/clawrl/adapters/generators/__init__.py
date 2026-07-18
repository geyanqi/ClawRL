"""Generator boundary adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clawrl.adapters.generators.fixture import FixtureFitGenerator, GeneratorBoundaryError
    from clawrl.adapters.generators.holdout import FixtureHoldoutGenerator, HoldoutBoundaryError

__all__ = [
    "FixtureFitGenerator",
    "FixtureHoldoutGenerator",
    "GeneratorBoundaryError",
    "HoldoutBoundaryError",
]


def __getattr__(name: str) -> Any:
    """Keep package-level compatibility without eager circular imports."""

    if name in {"FixtureFitGenerator", "GeneratorBoundaryError"}:
        from clawrl.adapters.generators import fixture

        return getattr(fixture, name)
    if name in {"FixtureHoldoutGenerator", "HoldoutBoundaryError"}:
        from clawrl.adapters.generators import holdout

        return getattr(holdout, name)
    raise AttributeError(name)
