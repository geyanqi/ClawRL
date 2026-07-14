"""Stable public description of the ClawRL automation surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AutomationDomain(StrEnum):
    """End-to-end RL lifecycle domains managed by ClawRL."""

    DATA = "data-curation"
    JUDGE = "judge-alignment"
    REWARD = "agentic-reward-generation"
    TRAINING = "training-orchestration"
    MONITORING = "training-monitoring"
    EXPERIMENTS = "experiment-iteration"
    EVALUATION = "model-evaluation"


@dataclass(frozen=True, slots=True)
class SystemManifest:
    """A serializable manifest for the initial framework boundary."""

    name: str
    version: str
    description: str
    domains: tuple[AutomationDomain, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "domains": [domain.value for domain in self.domains],
        }


def default_manifest() -> SystemManifest:
    """Return the canonical manifest for this framework version."""
    return SystemManifest(
        name="ClawRL",
        version="0.1.0",
        description=(
            "An autonomous RL training system powered by coding agents, "
            "automating data curation, judge alignment, agentic reward generation, "
            "training orchestration, monitoring, experimentation, and evaluation."
        ),
        domains=tuple(AutomationDomain),
    )
