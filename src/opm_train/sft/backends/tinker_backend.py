"""Tinker-backed supervised fine-tuning implementation."""

from __future__ import annotations

import importlib
from typing import Any

from opm_train.sft.contracts import MetricCallback, SFTBackendConfig, SFTBackendResult, SFTExample


class TinkerSFTBackend:
    """SFT backend built on top of the official Tinker Python SDK."""

    name = "tinker"

    def train(
        self,
        *,
        config: SFTBackendConfig,
        examples: list[SFTExample],
        on_metric: MetricCallback | None = None,
    ) -> SFTBackendResult:
        """Execute one supervised fine-tuning run through Tinker SDK."""
        if not examples:
            raise ValueError("SFT examples must not be empty")

        tinker = _import_tinker_module()
        types = getattr(tinker, "types", None)
        if types is None:
            raise RuntimeError("tinker SDK missing 'types' module")

        service_client = tinker.ServiceClient()
        create_kwargs: dict[str, Any] = {"base_model": config.base_model}
        if int(config.rank) > 0:
            create_kwargs["rank"] = int(config.rank)
        training_client = service_client.create_lora_training_client(**create_kwargs)
        tokenizer = training_client.get_tokenizer()

        processed_examples = [
            _to_tinker_datum(
                example=example,
                tokenizer=tokenizer,
                types=types,
            )
            for example in examples
        ]

        losses: list[float] = []
        cursor = 0
        for step_index in range(max(1, int(config.steps))):
            batch, cursor = _next_batch(processed_examples, batch_size=int(config.batch_size), cursor=cursor)
            fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(types.AdamParams(learning_rate=float(config.learning_rate)))
            fwdbwd_result = fwdbwd_future.result()
            _ = optim_future.result()
            loss = _weighted_loss(
                batch=batch,
                loss_outputs=getattr(fwdbwd_result, "loss_fn_outputs", []),
            )
            losses.append(loss)
            if on_metric is not None:
                on_metric({"step": step_index + 1, "loss": loss})

        sampling_client = training_client.save_weights_and_get_sampling_client(name=config.output_model)
        checkpoint_path = _extract_checkpoint_path(sampling_client)
        sample_output = _maybe_sample_output(
            config=config,
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            types=types,
        )

        return SFTBackendResult(
            backend=self.name,
            base_model=config.base_model,
            output_model=config.output_model,
            losses=losses,
            checkpoint_path=checkpoint_path,
            sample_output=sample_output,
        )


def _import_tinker_module() -> Any:
    """Import Tinker SDK lazily so runtime can stay dependency-light by default."""
    try:
        return importlib.import_module("tinker")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tinker backend requires the 'tinker' package. "
            "Install it with `conda run -n OpenCompany pip install tinker`."
        ) from exc


def _to_tinker_datum(*, example: SFTExample, tokenizer: Any, types: Any) -> Any:
    """Convert one normalized example into one Tinker Datum."""
    prompt_tokens = _as_int_list(tokenizer.encode(example.prompt, add_special_tokens=True))
    completion_tokens = _as_int_list(tokenizer.encode(example.completion, add_special_tokens=False))
    if not completion_tokens:
        raise ValueError(f"completion tokenization produced empty output for example '{example.example_id}'")

    tokens = [*prompt_tokens, *completion_tokens]
    if len(tokens) < 2:
        raise ValueError(f"example '{example.example_id}' must produce at least two tokens")

    weights = ([0.0] * len(prompt_tokens)) + ([1.0] * len(completion_tokens))
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    loss_weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "weights": loss_weights,
            "target_tokens": target_tokens,
        },
    )


def _next_batch(datums: list[Any], *, batch_size: int, cursor: int) -> tuple[list[Any], int]:
    """Build one cyclic batch and return the next cursor position."""
    if not datums:
        raise ValueError("datums must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    selected: list[Any] = []
    index = cursor % len(datums)
    for _ in range(batch_size):
        selected.append(datums[index])
        index = (index + 1) % len(datums)
    return selected, index


def _weighted_loss(*, batch: list[Any], loss_outputs: Any) -> float:
    """Compute weighted token-level cross-entropy from Tinker loss outputs."""
    output_rows = _as_list(loss_outputs)
    if len(output_rows) != len(batch):
        raise RuntimeError(
            "loss output count mismatch: "
            f"expected {len(batch)} rows, got {len(output_rows)}"
        )

    total_weight = 0.0
    weighted_logprob_sum = 0.0
    for datum, output in zip(batch, output_rows):
        logprobs = _as_float_list(_extract_logprobs(output))
        weights = _as_float_list(_extract_weights(datum))
        if len(logprobs) != len(weights):
            raise RuntimeError(
                "logprob/weight length mismatch: "
                f"{len(logprobs)} vs {len(weights)}"
            )
        for logprob, weight in zip(logprobs, weights):
            weighted_logprob_sum += float(logprob) * float(weight)
            total_weight += float(weight)

    if total_weight <= 0:
        raise RuntimeError("non-positive aggregate loss weight")
    return -weighted_logprob_sum / total_weight


def _extract_logprobs(output: Any) -> Any:
    """Extract logprob payload from one Tinker loss output row."""
    if isinstance(output, dict):
        return output.get("logprobs", [])
    return getattr(output, "logprobs", [])


def _extract_weights(datum: Any) -> Any:
    """Extract per-token weights from one Tinker datum."""
    loss_fn_inputs = getattr(datum, "loss_fn_inputs", None)
    if isinstance(loss_fn_inputs, dict):
        return loss_fn_inputs.get("weights", [])
    return []


def _extract_checkpoint_path(sampling_client: Any) -> str | None:
    """Best-effort extraction of checkpoint path from Tinker sampling client."""
    for attr_name in ("sampler_path", "model_path", "path", "checkpoint_path", "weights_path"):
        value = getattr(sampling_client, attr_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for method_name in (
        "get_sampler_path",
        "get_model_path",
        "get_path",
        "sampler_path",
        "model_path",
    ):
        method = getattr(sampling_client, method_name, None)
        if callable(method):
            try:
                candidate = method()
            except Exception:  # pragma: no cover - defensive path
                continue
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _maybe_sample_output(*, config: SFTBackendConfig, sampling_client: Any, tokenizer: Any, types: Any) -> str | None:
    """Optionally sample one completion from saved sampler checkpoint."""
    if not config.sample_prompt:
        return None
    prompt_tokens = _as_int_list(tokenizer.encode(config.sample_prompt, add_special_tokens=True))
    prompt = types.ModelInput.from_ints(prompt_tokens)
    sampling_params = types.SamplingParams(
        max_tokens=max(1, int(config.sample_max_tokens)),
        temperature=float(config.sample_temperature),
    )
    sample_future = sampling_client.sample(prompt=prompt, sampling_params=sampling_params, num_samples=1)
    sample_result = sample_future.result()
    sequences = _as_list(getattr(sample_result, "sequences", []))
    if not sequences:
        return None
    first_sequence = sequences[0]
    tokens = first_sequence.get("tokens") if isinstance(first_sequence, dict) else getattr(first_sequence, "tokens", [])
    return str(tokenizer.decode(_as_int_list(tokens)))


def _as_int_list(value: Any) -> list[int]:
    """Normalize an integer sequence from list/tuple/array-like values."""
    return [int(item) for item in _as_list(value)]


def _as_float_list(value: Any) -> list[float]:
    """Normalize a float sequence from list/tuple/array-like values."""
    return [float(item) for item in _as_list(value)]


def _as_list(value: Any) -> list[Any]:
    """Normalize arbitrary array-like values into a plain list."""
    candidate = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(candidate, list):
        return candidate
    if isinstance(candidate, tuple):
        return list(candidate)
    if candidate is None:
        return []
    return [candidate]
