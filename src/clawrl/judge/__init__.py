"""Teacher labels, Luna certification state machines, and JudgeBundle assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clawrl.judge.certification_models import (
        AlignmentPolicy,
        BaseJudgePrompt,
        FixtureLuna8Config,
        LunaInferenceConfig,
        ProductionLuna8Config,
        RewardSchema,
        Scalarizer,
        TraceJudgePrompt,
    )
    from clawrl.judge.certification_workflow import Luna8CertificationWorkflow, Luna8WorkflowSnapshot
    from clawrl.judge.fit_models import FixtureFitConfig, GeneratorPlan, InitialEvalRubric, SolInferenceConfig
    from clawrl.judge.fit_workflow import FitTrajectoryWorkflow, FitWorkflowSnapshot

__all__ = [
    "AlignmentPolicy",
    "BaseJudgePrompt",
    "FitTrajectoryWorkflow",
    "FitWorkflowSnapshot",
    "FixtureFitConfig",
    "FixtureLuna8Config",
    "GeneratorPlan",
    "InitialEvalRubric",
    "Luna8CertificationWorkflow",
    "Luna8WorkflowSnapshot",
    "LunaInferenceConfig",
    "ProductionLuna8Config",
    "RewardSchema",
    "Scalarizer",
    "SolInferenceConfig",
    "TraceJudgePrompt",
]

_CERTIFICATION_MODELS = {
    "AlignmentPolicy",
    "BaseJudgePrompt",
    "FixtureLuna8Config",
    "LunaInferenceConfig",
    "ProductionLuna8Config",
    "RewardSchema",
    "Scalarizer",
    "TraceJudgePrompt",
}
_CERTIFICATION_WORKFLOW = {"Luna8CertificationWorkflow", "Luna8WorkflowSnapshot"}
_FIT_MODELS = {"FixtureFitConfig", "GeneratorPlan", "InitialEvalRubric", "SolInferenceConfig"}
_FIT_WORKFLOW = {"FitTrajectoryWorkflow", "FitWorkflowSnapshot"}


def __getattr__(name: str) -> Any:
    """Resolve public APIs lazily so boundary adapters can import model modules safely."""

    if name in _CERTIFICATION_MODELS:
        from clawrl.judge import certification_models

        return getattr(certification_models, name)
    if name in _CERTIFICATION_WORKFLOW:
        from clawrl.judge import certification_workflow

        return getattr(certification_workflow, name)
    if name in _FIT_MODELS:
        from clawrl.judge import fit_models

        return getattr(fit_models, name)
    if name in _FIT_WORKFLOW:
        from clawrl.judge import fit_workflow

        return getattr(fit_workflow, name)
    raise AttributeError(name)
