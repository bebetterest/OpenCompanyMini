from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from opm_train.sft import SFTBackendConfig, SFTBackendResult, SFTExample, load_sft_examples, register_sft_backend, run_sft
from opm_train.sft.backends.tinker_backend import TinkerSFTBackend
from opm_train.sft.runner import SFTRunConfig


class _FakeBackend:
    name = "fake-test"

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def train(self, *, config: SFTBackendConfig, examples: list[SFTExample], on_metric=None) -> SFTBackendResult:
        self.state["config"] = config
        self.state["examples"] = examples
        if on_metric is not None:
            on_metric({"step": 1, "loss": 0.9})
            on_metric({"step": 2, "loss": 0.4})
        return SFTBackendResult(
            backend=self.name,
            base_model=config.base_model,
            output_model=config.output_model,
            losses=[0.9, 0.4],
            checkpoint_path="fake://checkpoint/0002",
            sample_output="sample",
        )


def test_load_sft_examples_auto_detects_common_key_pairs() -> None:
    with TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "sft.jsonl"
        input_path.write_text(
            "\n".join(
                [
                    '{"id":"a","prompt":"P1","completion":"C1"}',
                    '{"id":"b","input":"P2","output":"C2"}',
                ]
            ),
            encoding="utf-8",
        )

        examples = load_sft_examples(input_path=input_path)

    assert [item.example_id for item in examples] == ["a", "b"]
    assert [item.prompt for item in examples] == ["P1", "P2"]
    assert [item.completion for item in examples] == ["C1", "C2"]


def test_run_sft_writes_result_and_metrics_artifacts() -> None:
    with TemporaryDirectory() as temp_dir:
        app_dir = Path(temp_dir)
        input_path = app_dir / "sft.jsonl"
        input_path.write_text('{"id":"x1","prompt":"hello","completion":"world"}\n', encoding="utf-8")

        state: dict[str, Any] = {}
        register_sft_backend(_FakeBackend(state), replace=True)

        output = run_sft(
            SFTRunConfig(
                backend="fake-test",
                input_path=input_path,
                project_dir=app_dir,
                app_dir=app_dir,
                base_model="base-model",
                run_id="sft-unit-run",
            )
        )

        assert output.run_id == "sft-unit-run"
        assert output.total_examples == 1
        assert output.result.losses == [0.9, 0.4]
        assert output.result_path.exists()
        assert output.metrics_path.exists()
        result_payload = json.loads(output.result_path.read_text(encoding="utf-8"))
        assert result_payload["backend"] == "fake-test"
        metrics_rows = [
            json.loads(line)
            for line in output.metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [row["step"] for row in metrics_rows] == [1, 2]
        assert [round(float(row["loss"]), 3) for row in metrics_rows] == [0.9, 0.4]

        captured_config = state["config"]
        assert isinstance(captured_config, SFTBackendConfig)
        assert captured_config.output_model == "sft-unit-run-model"
        assert captured_config.steps == 6
        captured_examples = state["examples"]
        assert isinstance(captured_examples, list)
        assert captured_examples[0].prompt == "hello"


def test_tinker_backend_raises_clear_error_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = TinkerSFTBackend()

    def _missing_module(_: str) -> Any:
        raise ModuleNotFoundError("No module named tinker")

    monkeypatch.setattr("opm_train.sft.backends.tinker_backend.importlib.import_module", _missing_module)

    with pytest.raises(RuntimeError, match="pip install tinker"):
        backend.train(
            config=SFTBackendConfig(
                base_model="Qwen/Qwen3",
                output_model="demo",
                steps=1,
                batch_size=1,
                learning_rate=1e-4,
                rank=32,
            ),
            examples=[SFTExample(example_id="1", prompt="a", completion="b")],
        )


def test_tinker_backend_executes_with_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeFuture:
        def __init__(self, payload: Any) -> None:
            self.payload = payload

        def result(self) -> Any:
            return self.payload

    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            _ = add_special_tokens
            return [ord(ch) for ch in text]

        def decode(self, tokens: list[int]) -> str:
            return "".join(chr(token) for token in tokens)

    class FakeModelInput:
        def __init__(self, tokens: list[int]) -> None:
            self.tokens = list(tokens)

        @classmethod
        def from_ints(cls, tokens: list[int]) -> "FakeModelInput":
            return cls(tokens)

    class FakeDatum:
        def __init__(self, *, model_input: FakeModelInput, loss_fn_inputs: dict[str, Any]) -> None:
            self.model_input = model_input
            self.loss_fn_inputs = loss_fn_inputs

    class FakeSamplingParams:
        def __init__(self, *, max_tokens: int, temperature: float) -> None:
            self.max_tokens = max_tokens
            self.temperature = temperature

    class FakeAdamParams:
        def __init__(self, *, learning_rate: float) -> None:
            self.learning_rate = learning_rate

    class FakeSequence:
        def __init__(self, tokens: list[int]) -> None:
            self.tokens = tokens

    class FakeSamplingResult:
        def __init__(self) -> None:
            self.sequences = [FakeSequence(tokens=[111, 107])]

    class FakeSamplingClient:
        sampler_path = "tinker://fake/train:0/sampler_weights/000002"

        def sample(self, *, prompt: Any, sampling_params: Any, num_samples: int) -> FakeFuture:
            captured["sample"] = {
                "prompt_tokens": getattr(prompt, "tokens", []),
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "num_samples": num_samples,
            }
            return FakeFuture(FakeSamplingResult())

    class FakeTrainingResult:
        def __init__(self, loss_fn_outputs: list[dict[str, list[float]]]) -> None:
            self.loss_fn_outputs = loss_fn_outputs

    class FakeTrainingClient:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()

        def get_tokenizer(self) -> FakeTokenizer:
            return self.tokenizer

        def forward_backward(self, batch: list[FakeDatum], loss_name: str) -> FakeFuture:
            captured["loss_name"] = loss_name
            captured["batch_size"] = len(batch)
            outputs = []
            for datum in batch:
                weights = list(datum.loss_fn_inputs["weights"])
                outputs.append({"logprobs": [-1.0 for _ in weights]})
            return FakeFuture(FakeTrainingResult(outputs))

        def optim_step(self, params: FakeAdamParams) -> FakeFuture:
            captured["learning_rate"] = params.learning_rate
            return FakeFuture({"ok": True})

        def save_weights_and_get_sampling_client(self, *, name: str) -> FakeSamplingClient:
            captured["output_model"] = name
            return FakeSamplingClient()

    class FakeServiceClient:
        def create_lora_training_client(self, **kwargs: Any) -> FakeTrainingClient:
            captured["create_kwargs"] = kwargs
            return FakeTrainingClient()

    class FakeTypes:
        Datum = FakeDatum
        ModelInput = FakeModelInput
        SamplingParams = FakeSamplingParams
        AdamParams = FakeAdamParams

    class FakeTinkerModule:
        ServiceClient = FakeServiceClient
        types = FakeTypes

    monkeypatch.setattr(
        "opm_train.sft.backends.tinker_backend.importlib.import_module",
        lambda _: FakeTinkerModule,
    )

    backend = TinkerSFTBackend()
    result = backend.train(
        config=SFTBackendConfig(
            base_model="Qwen/Qwen3",
            output_model="demo-model",
            steps=2,
            batch_size=1,
            learning_rate=2e-4,
            rank=64,
            sample_prompt="hi",
            sample_max_tokens=12,
            sample_temperature=0.3,
        ),
        examples=[SFTExample(example_id="ex-1", prompt="a", completion="b")],
    )

    assert result.backend == "tinker"
    assert result.output_model == "demo-model"
    assert result.checkpoint_path == "tinker://fake/train:0/sampler_weights/000002"
    assert result.sample_output == "ok"
    assert result.losses == [1.0, 1.0]
    assert captured["create_kwargs"] == {"base_model": "Qwen/Qwen3", "rank": 64}
    assert captured["learning_rate"] == 2e-4
    assert captured["batch_size"] == 1
    assert captured["sample"]["num_samples"] == 1
