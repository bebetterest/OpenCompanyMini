"""SFT backend registry."""

from __future__ import annotations

from typing import Final

from opm_train.sft.contracts import SFTBackend


_SFT_BACKENDS: Final[dict[str, SFTBackend]] = {}


def register_sft_backend(backend: SFTBackend, *, replace: bool = True) -> None:
    """Register one SFT backend by normalized backend name."""
    if not isinstance(backend, SFTBackend):
        raise ValueError("SFT backend must implement SFTBackend protocol")
    name = str(getattr(backend, "name", "")).strip().lower()
    if not name:
        raise ValueError("SFT backend must define a non-empty 'name'")
    if name in _SFT_BACKENDS and not replace:
        raise ValueError(f"SFT backend already registered: {name}")
    _SFT_BACKENDS[name] = backend


def get_sft_backend(name: str) -> SFTBackend:
    """Resolve one SFT backend by backend name."""
    key = str(name).strip().lower()
    backend = _SFT_BACKENDS.get(key)
    if backend is None:
        available = ", ".join(sorted(_SFT_BACKENDS.keys())) or "<none>"
        raise ValueError(f"unknown SFT backend '{name}', available: {available}")
    return backend


def list_sft_backends() -> list[str]:
    """List registered SFT backend names."""
    return sorted(_SFT_BACKENDS.keys())
