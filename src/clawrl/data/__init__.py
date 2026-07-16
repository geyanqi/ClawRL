"""Governed data ingestion, sanitization, selection, and DatasetVersion assembly."""

from clawrl.data.models import (
    DataContractError,
    DatasetValidationError,
    DataSourceSkill,
    FixtureDataIngestConfig,
    ProductionDataIngestConfig,
    QueryWindow,
)
from clawrl.data.validation import (
    LoadedTrainingDataset,
    load_training_dataset,
    validate_dataset_for_experiment,
)

__all__ = [
    "DataContractError",
    "DataIngestSnapshot",
    "DataSourceSkill",
    "DatasetValidationError",
    "FixtureDataIngestConfig",
    "GovernedDataIngestWorkflow",
    "LoadedTrainingDataset",
    "ProductionDataIngestConfig",
    "QueryWindow",
    "load_training_dataset",
    "validate_dataset_for_experiment",
]


def __getattr__(name: str) -> object:
    """Load workflow exports lazily so source adapters can import data models."""

    if name in {"DataIngestSnapshot", "GovernedDataIngestWorkflow"}:
        from clawrl.data.workflow import DataIngestSnapshot, GovernedDataIngestWorkflow

        return {
            "DataIngestSnapshot": DataIngestSnapshot,
            "GovernedDataIngestWorkflow": GovernedDataIngestWorkflow,
        }[name]
    raise AttributeError(name)
