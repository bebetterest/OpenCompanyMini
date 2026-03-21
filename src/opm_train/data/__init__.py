"""Data adapters and contracts for dataset-driven batch execution."""

from opm_train.data.contracts import (
    BatchItemResult,
    BatchSummary,
    DatasetAdapter,
    DatasetSample,
    PreparedTask,
    ValidationResult,
)
from opm_train.data.gsm8k import GSM8KDatasetAdapter
from opm_train.data.math_verify import MathVerifyDatasetAdapter
from opm_train.data.simple_math import SimpleMathDatasetAdapter
from opm_train.data.registry import get_dataset_adapter, list_dataset_adapters, register_dataset_adapter

register_dataset_adapter(GSM8KDatasetAdapter())
register_dataset_adapter(SimpleMathDatasetAdapter())

__all__ = [
    "BatchItemResult",
    "BatchSummary",
    "DatasetAdapter",
    "DatasetSample",
    "PreparedTask",
    "ValidationResult",
    "get_dataset_adapter",
    "list_dataset_adapters",
    "register_dataset_adapter",
    "GSM8KDatasetAdapter",
    "MathVerifyDatasetAdapter",
    "SimpleMathDatasetAdapter",
]
