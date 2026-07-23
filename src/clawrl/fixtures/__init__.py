"""Deterministic fake adapters and closed-loop fixture support."""

from clawrl.fixtures.closed_loop import (
    NEGATIVE_CONSTRAINTS,
    PHASES,
    ClosedLoopCampaign,
    ClosedLoopCampaignWorkflow,
    ClosedLoopConfig,
    ClosedLoopError,
    ClosedLoopFixtureConfig,
    ClosedLoopReadinessError,
    ClosedLoopSnapshot,
    FixtureClosedLoop,
    FixtureClosedLoopConfig,
    FixtureClosedLoopWorkflow,
)

__all__ = [
    "PHASES",
    "NEGATIVE_CONSTRAINTS",
    "ClosedLoopConfig",
    "ClosedLoopCampaign",
    "ClosedLoopCampaignWorkflow",
    "ClosedLoopError",
    "ClosedLoopFixtureConfig",
    "ClosedLoopReadinessError",
    "ClosedLoopSnapshot",
    "FixtureClosedLoop",
    "FixtureClosedLoopConfig",
    "FixtureClosedLoopWorkflow",
]
