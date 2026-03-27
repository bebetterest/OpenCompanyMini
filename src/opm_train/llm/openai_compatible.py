"""OpenAI-compatible streaming chat client with retry and SSE parsing."""

from __future__ import annotations

import asyncio
import inspect
import json
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx


TokenCallback = Callable[[str], Awaitable[None] | None]
RetryCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


class SseParser:
    """Incremental parser for ``data:`` framed SSE payloads."""

    def __init__(self) -> None:
        """Initialize parser with an empty chunk buffer."""
        self._buffer = ""

    def feed(self, chunk: str) -> list[str]:
        """Consume chunk text and emit complete SSE data events."""
        self._buffer += chunk
        # Normalize line endings so both LF and CRLF-framed SSE streams are supported.
        self._buffer = self._buffer.replace("\r\n", "\n").replace("\r", "\n")
        events: list[str] = []
        while "\n\n" in self._buffer:
            raw, self._buffer = self._buffer.split("\n\n", 1)
            lines = []
            # Use explicit LF splitting only; splitlines() treats Unicode line
            # separators (for example U+2028) as hard breaks inside JSON strings.
            for line in raw.split("\n"):
                if line.startswith("data:"):
                    lines.append(line[5:].strip())
            if lines:
                events.append("\n".join(lines))
        return events


@dataclass(slots=True)
class ChatResult:
    """Normalized streaming response payload used by orchestrator."""

    content: str
    raw_events: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    usage: dict[str, Any] | None = None


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible streaming chat client."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: int,
        max_retries: int,
        retry_backoff_seconds: float,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Store connection and retry settings."""
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key)
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if isinstance(headers, dict):
            self.headers.update({str(k): str(v) for k, v in headers.items()})

    async def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        on_token: TokenCallback | None = None,
        on_reasoning: TokenCallback | None = None,
        on_retry: RetryCallback | None = None,
    ) -> ChatResult:
        """Stream one chat completion and aggregate content/tool deltas."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls

        max_attempts = self.max_retries + 1
        for attempt in range(max_attempts):
            parser = SseParser()
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            raw_events: list[dict[str, Any]] = []
            tool_call_parts: dict[int, dict[str, str]] = {}
            usage: dict[str, Any] | None = None
            received_event = False
            try:
                async with httpx.AsyncClient(timeout=float(self.timeout_seconds)) as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_text():
                            for event_data in parser.feed(chunk):
                                if event_data == "[DONE]":
                                    continue
                                received_event = True
                                event = json.loads(event_data)
                                raw_events.append(event)
                                if isinstance(event.get("usage"), dict):
                                    usage = event.get("usage")
                                choices = event.get("choices")
                                if not isinstance(choices, list) or not choices:
                                    continue
                                choice = choices[0]
                                delta = choice.get("delta")
                                if not isinstance(delta, dict):
                                    continue
                                text = delta.get("content")
                                if isinstance(text, str) and text:
                                    content_parts.append(text)
                                    if on_token is not None:
                                        await _maybe_await(on_token(text))
                                reasoning = _extract_reasoning(delta)
                                if reasoning:
                                    reasoning_parts.append(reasoning)
                                    if on_reasoning is not None:
                                        await _maybe_await(on_reasoning(reasoning))
                                tool_calls = delta.get("tool_calls")
                                if isinstance(tool_calls, list):
                                    for raw_call in tool_calls:
                                        _merge_tool_call_delta(tool_call_parts, raw_call)
                return ChatResult(
                    content="".join(content_parts),
                    reasoning="".join(reasoning_parts),
                    raw_events=raw_events,
                    tool_calls=_tool_calls_payload(tool_call_parts),
                    usage=usage,
                )
            except Exception as exc:
                should_retry = self._should_retry(
                    exc=exc,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    has_partial_output=received_event,
                )
                if not should_retry:
                    raise
                wait_seconds = self._retry_delay(attempt)
                if on_retry is not None:
                    payload_retry = {
                        "attempt": attempt + 1,
                        "max_attempts": max_attempts,
                        "wait_seconds": wait_seconds,
                        "error": str(exc),
                    }
                    await _maybe_await(on_retry(payload_retry))
                await asyncio.sleep(wait_seconds)
        raise RuntimeError("stream_chat exhausted retry budget")

    def _should_retry(
        self,
        *,
        exc: Exception,
        attempt: int,
        max_attempts: int,
        has_partial_output: bool,
    ) -> bool:
        """Return whether failed request should be retried."""
        if attempt + 1 >= max_attempts:
            return False
        if has_partial_output:
            return False
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            code = int(exc.response.status_code)
            return code >= 429
        return False

    def _retry_delay(self, attempt: int) -> float:
        """Compute exponential backoff with bounded random jitter."""
        base = self.retry_backoff_seconds * (2**attempt)
        jitter = random.uniform(0.0, 0.3)
        return base + jitter


def _extract_reasoning(delta: dict[str, Any]) -> str:
    """Extract reasoning fragments from provider-specific delta fields."""
    for key in ("reasoning", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value
    reasoning_entries = delta.get("reasoning_details")
    if isinstance(reasoning_entries, list):
        fragments = []
        for entry in reasoning_entries:
            if isinstance(entry, dict):
                text = entry.get("text")
                if isinstance(text, str) and text:
                    fragments.append(text)
        if fragments:
            return "".join(fragments)
    return ""


def _merge_tool_call_delta(parts: dict[int, dict[str, str]], raw_call: Any) -> None:
    """Merge one streamed tool-call delta into indexed accumulation map."""
    if not isinstance(raw_call, dict):
        return
    index = int(raw_call.get("index", 0) or 0)
    current = parts.setdefault(index, {"id": "", "type": "function", "name": "", "arguments": ""})
    if raw_call.get("id"):
        current["id"] = str(raw_call["id"])
    call_type = str(raw_call.get("type", "")).strip()
    if call_type:
        current["type"] = call_type
    function = raw_call.get("function")
    if not isinstance(function, dict):
        return
    if function.get("name"):
        current["name"] += str(function["name"])
    if function.get("arguments"):
        current["arguments"] += str(function["arguments"])


def _tool_calls_payload(parts: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    """Convert accumulated call fragments into normalized call payloads."""
    payloads: list[dict[str, Any]] = []
    for index, value in sorted(parts.items()):
        arguments_json = value["arguments"] or "{}"
        name = value["name"]
        call_type = str(value.get("type", "")).strip() or "function"
        payloads.append(
            {
                "id": value["id"] or f"tool-call-{index}",
                "type": call_type,
                "function": {"name": name, "arguments": arguments_json},
            }
        )
    return payloads


async def _maybe_await(value: Any) -> None:
    """Await callback return value only when it is awaitable."""
    if inspect.isawaitable(value):
        await value
