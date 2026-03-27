from __future__ import annotations

from unittest import mock

import pytest

from opm_train.llm.openai_compatible import OpenAICompatibleClient, SseParser


class FakeStreamResponse:
    def __init__(self, *, chunks: list[str], status_code: int = 200) -> None:
        self._chunks = chunks
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            request = httpx.Request("POST", "https://example.com")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    async def aiter_text(self):
        for chunk in self._chunks:
            yield chunk


class FakeAsyncClient:
    response_chunks: list[str] = []
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
        return FakeStreamResponse(chunks=type(self).response_chunks)


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
