"""Contracts for supervised fine-tuning backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


MetricCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True, frozen=True)
class SFTExample:
    """One normalized supervised fine-tuning sample."""

    example_id: str
    prompt: str
    completion: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SFTBackendConfig:
    """Backend-agnostic SFT execution settings."""

    base_model: str
    output_model: str
    steps: int
    batch_size: int
    learning_rate: float
    rank: int
    sample_prompt: str | None = None
    sample_max_tokens: int = 64
    sample_temperature: float = 0.0


@dataclass(slots=True, frozen=True)
class SFTBackendResult:
    """Terminal payload returned by one SFT backend execution."""

    backend: str
    base_model: str
    output_model: str
    losses: list[float]
    checkpoint_path: str | None = None
    sample_output: str | None = None


@runtime_checkable
class SFTBackend(Protocol):
    """Backend protocol for supervised fine-tuning implementations."""

    name: str

    def train(
        self,
        *,
        config: SFTBackendConfig,
        examples: list[SFTExample],
        on_metric: MetricCallback | None = None,
    ) -> SFTBackendResult:
        """Run supervised fine-tuning and return terminal summary payload."""
