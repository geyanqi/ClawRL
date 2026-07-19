"""Compatibility exports for the Ticket 15 grouped Judge tool route."""

from clawrl.router.grouped_route import (
    FixtureGroupedToolRouter,
    FixtureHelperSandbox,
    GroupedRouteConfig,
    GroupedToolRouteError,
    InjectedGroupedRouteCrash,
    ToolRequest,
    ToolSandboxLimits,
)

FixtureJudgeToolRoute = FixtureGroupedToolRouter

__all__ = [
    "FixtureGroupedToolRouter",
    "FixtureJudgeToolRoute",
    "FixtureHelperSandbox",
    "GroupedRouteConfig",
    "GroupedToolRouteError",
    "InjectedGroupedRouteCrash",
    "ToolRequest",
    "ToolSandboxLimits",
]
