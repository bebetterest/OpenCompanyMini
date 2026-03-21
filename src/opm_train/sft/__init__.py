"""SFT backends, runners, and contracts."""

from opm_train.sft.backends import TinkerSFTBackend
from opm_train.sft.contracts import SFTBackend, SFTBackendConfig, SFTBackendResult, SFTExample
from opm_train.sft.jsonl import load_sft_examples
from opm_train.sft.registry import get_sft_backend, list_sft_backends, register_sft_backend
from opm_train.sft.runner import SFTRunConfig, SFTRunOutput, run_sft

register_sft_backend(TinkerSFTBackend())

__all__ = [
    "SFTBackend",
    "SFTBackendConfig",
    "SFTBackendResult",
    "SFTExample",
    "SFTRunConfig",
    "SFTRunOutput",
    "get_sft_backend",
    "list_sft_backends",
    "load_sft_examples",
    "register_sft_backend",
    "run_sft",
    "TinkerSFTBackend",
]
