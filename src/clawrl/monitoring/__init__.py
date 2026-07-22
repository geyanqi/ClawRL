"""Deterministic hard stops and Governor-driven two-phase soft stops."""

from clawrl.training.hard_failure_stop import (
    FailureKind,
    FencedRunController,
    FixtureHardFailureSimulator,
    HardFailureClass,
    HardFailureKind,
    HardFailureMonitor,
    HardFailureObservation,
    HardFailureStopConfig,
    HardFailureStopError,
    HardFailureStopSnapshot,
    HardFailureStopWorkflow,
    HardFailureTerminalStop,
    InjectedHardFailureControllerCrash,
    StaleHardFailureController,
    classify_hard_failure,
)

__all__ = [
    "FailureKind",
    "HardFailureClass",
    "HardFailureKind",
    "HardFailureObservation",
    "HardFailureMonitor",
    "HardFailureStopConfig",
    "HardFailureStopError",
    "HardFailureTerminalStop",
    "HardFailureStopSnapshot",
    "HardFailureStopWorkflow",
    "FencedRunController",
    "FixtureHardFailureSimulator",
    "InjectedHardFailureControllerCrash",
    "StaleHardFailureController",
    "classify_hard_failure",
]
