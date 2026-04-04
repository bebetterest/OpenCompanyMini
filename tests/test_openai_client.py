from __future__ import annotations

from unittest import mock

import httpx
import pytest

from opm_train.llm.openai_compatible import OpenAICompatibleClient, SseParser


class FakeStreamResponse:
    def __init__(self, *, chunks: list[str], status_code: int = 200, error_body: str = "") -> None:
        request = httpx.Request("POST", "https://example.com/chat/completions")
        self.request = request
        self._chunks = chunks
        self.status_code = status_code
        self.reason_phrase = "Bad Request" if status_code == 400 else "OK"
        self._error_body = error_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def aiter_text(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self) -> bytes:
        return self._error_body.encode("utf-8")


class FakeAsyncClient:
    response_chunks: list[str] = []
    response_status_code: int = 200
    response_error_body: str = ""
    last_request: dict[str, object] | None = None
    last_timeout: float | None = None

    def __init__(self, *, timeout: float) -> None:
        type(self).last_timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def stream(self, method: str, url: str, *, headers: dict[str, str], json: dict[str, object]):
        type(self).last_request = {
            "method": method,
            "url": url,
            "headers": headers,
            "json": json,
        }
        return FakeStreamResponse(
            chunks=type(self).response_chunks,
            status_code=type(self).response_status_code,
            error_body=type(self).response_error_body,
        )


def test_sse_parser_handles_chunked_events() -> None:
    parser = SseParser()
    first = parser.feed('data: {"choices":[{"delta":{"content":"Hel')
    assert first == []
    second = parser.feed('lo"}}]}\n\ndata: [DONE]\n\n')
    assert second[0] == '{"choices":[{"delta":{"content":"Hello"}}]}'
    assert second[1] == "[DONE]"


def test_sse_parser_handles_crlf_and_mixed_chunk_boundaries() -> None:
    parser = SseParser()
    first = parser.feed('data: {"choices":[{"delta":{"content":"Hello"}}]}\r')
    assert first == []
    second = parser.feed("\n\r\ndata: [DONE]\r\n")
    assert second[0] == '{"choices":[{"delta":{"content":"Hello"}}]}'
    assert len(second) == 1
    third = parser.feed("\r\n")
    assert third[0] == "[DONE]"


def test_sse_parser_keeps_unicode_line_separator_inside_data_json() -> None:
    parser = SseParser()
    payload = '{"choices":[{"delta":{"content":"Line A\u2028Line B"}}]}'
    events = parser.feed(f"data: {payload}\n\n")
    assert events == [payload]


@pytest.mark.asyncio
async def test_stream_chat_sends_tool_fields_and_parses_calls() -> None:
    client = OpenAICompatibleClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        timeout_seconds=30,
        max_retries=0,
        retry_backoff_seconds=0.0,
        headers={"X-Test": "yes"},
    )
    FakeAsyncClient.response_chunks = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"shell","arguments":"{\\"command\\":\\"ec"}}]}}]}\n\n',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ho hi\\"}"}}]}}]}\n\n',
        "data: [DONE]\n\n",
    ]
    FakeAsyncClient.response_status_code = 200
    FakeAsyncClient.response_error_body = ""

    with mock.patch("httpx.AsyncClient", FakeAsyncClient):
        result = await client.stream_chat(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.1,
            max_tokens=128,
            tools=[{"type": "function", "function": {"name": "shell", "parameters": {"type": "object"}}}],
            tool_choice="auto",
            parallel_tool_calls=False,
        )

    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["id"] == "call_1"
    assert result.tool_calls[0]["type"] == "function"
    assert result.tool_calls[0]["function"]["name"] == "shell"
    assert result.tool_calls[0]["function"]["arguments"] == '{"command":"echo hi"}'
    assert set(result.tool_calls[0].keys()) == {"id", "type", "function"}

    payload = FakeAsyncClient.last_request["json"]  # type: ignore[index]
    assert payload["tools"][0]["function"]["name"] == "shell"  # type: ignore[index]
    assert payload["tool_choice"] == "auto"  # type: ignore[index]
    assert payload["parallel_tool_calls"] is False  # type: ignore[index]


@pytest.mark.asyncio
async def test_stream_chat_http_status_error_includes_response_body() -> None:
    client = OpenAICompatibleClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        timeout_seconds=30,
        max_retries=0,
        retry_backoff_seconds=0.0,
    )
    FakeAsyncClient.response_chunks = []
    FakeAsyncClient.response_status_code = 400
    FakeAsyncClient.response_error_body = '{"error":{"message":"invalid_function_parameters"}}'

    with mock.patch("httpx.AsyncClient", FakeAsyncClient):
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.stream_chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.1,
                max_tokens=128,
            )

    message = str(exc_info.value)
    assert "response_body=" in message
    assert "invalid_function_parameters" in message


@pytest.mark.asyncio
async def test_stream_chat_retries_on_empty_stream() -> None:
    class EmptyThenSuccessAsyncClient:
        last_request: dict[str, object] | None = None
        call_count: int = 0

        def __init__(self, *, timeout: float) -> None:
            _ = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict[str, str], json: dict[str, object]):
            type(self).last_request = {
                "method": method,
                "url": url,
                "headers": headers,
                "json": json,
            }
            type(self).call_count += 1
            if type(self).call_count == 1:
                return FakeStreamResponse(chunks=[], status_code=200, error_body="")
            return FakeStreamResponse(
                chunks=[
                    'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
                    "data: [DONE]\n\n",
                ],
                status_code=200,
                error_body="",
            )

    client = OpenAICompatibleClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        timeout_seconds=30,
        max_retries=1,
        retry_backoff_seconds=0.0,
    )
    client._retry_delay = lambda attempt: 0.0  # type: ignore[method-assign]
    retries: list[dict[str, object]] = []

    with mock.patch("httpx.AsyncClient", EmptyThenSuccessAsyncClient):
        result = await client.stream_chat(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.1,
            max_tokens=64,
            on_retry=lambda payload: retries.append(payload),
        )

    assert result.content == "ok"
    assert EmptyThenSuccessAsyncClient.call_count == 2
    assert len(retries) == 1
    assert retries[0]["reason"] == "empty_stream"
    assert retries[0]["had_partial_output"] is False


@pytest.mark.asyncio
async def test_stream_chat_retries_on_partial_stream_transport_error_and_discards_partial_output() -> None:
    class PartialThenErrorResponse(FakeStreamResponse):
        async def aiter_text(self):
            yield 'data: {"choices":[{"delta":{"content":"partial-"}}]}\n\n'
            raise httpx.ReadError("connection dropped", request=self.request)

    class PartialThenSuccessAsyncClient:
        call_count: int = 0

        def __init__(self, *, timeout: float) -> None:
            _ = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict[str, str], json: dict[str, object]):
            _ = (method, url, headers, json)
            type(self).call_count += 1
            if type(self).call_count == 1:
                return PartialThenErrorResponse(chunks=[], status_code=200, error_body="")
            return FakeStreamResponse(
                chunks=[
                    'data: {"choices":[{"delta":{"content":"final"}}]}\n\n',
                    "data: [DONE]\n\n",
                ],
                status_code=200,
                error_body="",
            )

    client = OpenAICompatibleClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        timeout_seconds=30,
        max_retries=1,
        retry_backoff_seconds=0.0,
    )
    client._retry_delay = lambda attempt: 0.0  # type: ignore[method-assign]
    retries: list[dict[str, object]] = []

    with mock.patch("httpx.AsyncClient", PartialThenSuccessAsyncClient):
        result = await client.stream_chat(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.1,
            max_tokens=64,
            on_retry=lambda payload: retries.append(payload),
        )

    assert result.content == "final"
    assert PartialThenSuccessAsyncClient.call_count == 2
    assert len(retries) == 1
    assert retries[0]["reason"] == "api_or_network"
    assert retries[0]["had_partial_output"] is True
    assert len(result.raw_events) == 1
    assert result.raw_events[0]["choices"][0]["delta"]["content"] == "final"
