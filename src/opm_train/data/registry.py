"""Dataset adapter registry."""

from __future__ import annotations

from typing import Final

from opm_train.data.contracts import DatasetAdapter


_DATASET_ADAPTERS: Final[dict[str, DatasetAdapter]] = {}


def register_dataset_adapter(adapter: DatasetAdapter, *, replace: bool = True) -> None:
    """Register one dataset adapter by normalized dataset name."""
    if not isinstance(adapter, DatasetAdapter):
        raise ValueError("dataset adapter must implement DatasetAdapter protocol")
    name = str(getattr(adapter, "name", "")).strip().lower()
    if not name:
        raise ValueError("dataset adapter must define a non-empty 'name'")
    if name in _DATASET_ADAPTERS and not replace:
        raise ValueError(f"dataset adapter already registered: {name}")
    _DATASET_ADAPTERS[name] = adapter


def get_dataset_adapter(name: str) -> DatasetAdapter:
    """Resolve one dataset adapter by name."""
    key = str(name).strip().lower()
    adapter = _DATASET_ADAPTERS.get(key)
    if adapter is None:
        available = ", ".join(sorted(_DATASET_ADAPTERS.keys())) or "<none>"
        raise ValueError(f"unknown dataset adapter '{name}', available: {available}")
    return adapter


def list_dataset_adapters() -> list[str]:
    """List registered dataset adapter names."""
    return sorted(_DATASET_ADAPTERS.keys())
