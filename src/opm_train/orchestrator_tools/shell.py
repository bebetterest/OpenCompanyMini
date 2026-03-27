"""Shell tool handler mixin."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from opm_train.models import ToolRun
from opm_train.tools import TERMINAL_TOOL_RUN_STATUSES
from opm_train.utils import json_ready


class ShellToolMixin:
    """Attach shell tool execution behavior."""

    async def _tool_shell(self, run: ToolRun, action: dict[str, Any]) -> dict[str, Any]:
        """Execute shell command with inline wait budget and background fallback."""
        command = str(action.get("command", "")).strip()
        if not command:
            raise ValueError("shell requires non-empty command")
        cwd = self.project_dir / str(action.get("cwd", "."))
        cwd = cwd.resolve()
        if self.project_dir not in [cwd, *cwd.parents]:
            raise ValueError("shell cwd escapes project directory")
        timeout_seconds = _parse_shell_timeout_seconds(
            action.get("timeout_seconds"),
            default=float(self.config.runtime.tools.shell_timeout_seconds),
        )
        inline_wait_seconds = max(0.0, float(self.config.runtime.tools.shell_inline_wait_seconds))
        blocking = run.blocking
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        pid: int | None = None

        def snapshot(*, exit_code: int | None = None, timed_out: bool = False) -> dict[str, Any]:
            """Build latest shell execution snapshot payload."""
            return {
                "command": command,
                "cwd": str(cwd),
                "pid": pid,
                "timeout_seconds": timeout_seconds,
                "exit_code": exit_code,
                "stdout": "".join(stdout_parts),
                "stderr": "".join(stderr_parts),
                "timed_out": timed_out,
            }

        def publish(*, exit_code: int | None = None, timed_out: bool = False) -> dict[str, Any]:
            """Update run.result and return the same snapshot object."""
            value = snapshot(exit_code=exit_code, timed_out=timed_out)
            run.result = value
            return value

        publish()

        async def worker() -> dict[str, Any]:
            nonlocal pid
            process: asyncio.subprocess.Process | None = None
            stdout_task: asyncio.Task[Any] | None = None
            stderr_task: asyncio.Task[Any] | None = None
            try:
                process = await asyncio.create_subprocess_shell(
                    command,
                    cwd=str(cwd),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                pid = process.pid
                publish()

                async def read_stream(stream: asyncio.StreamReader | None, sink: list[str]) -> None:
                    if stream is None:
                        return
                    while True:
                        chunk = await stream.read(4096)
                        if not chunk:
                            return
                        sink.append(chunk.decode("utf-8", errors="replace"))
                        publish()

                stdout_task = asyncio.create_task(read_stream(process.stdout, stdout_parts), name=f"shell-stdout:{run.id}")
                stderr_task = asyncio.create_task(read_stream(process.stderr, stderr_parts), name=f"shell-stderr:{run.id}")
                await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=False)
                result = publish(exit_code=int(process.returncode or 0))
                self._complete_tool_run(run, result=result)
                return result
            except asyncio.TimeoutError:
                if process is not None:
                    process.kill()
                    await process.wait()
                if stdout_task is not None and stderr_task is not None:
                    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                publish(exit_code=int(process.returncode or -1) if process is not None else -1, timed_out=True)
                timeout_error = f"shell command timed out after {timeout_seconds:g}s"
                self._fail_tool_run(run, error=timeout_error)
                raise ValueError(timeout_error)
            except asyncio.CancelledError:
                if process is not None:
                    with suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
                if stdout_task is not None and stderr_task is not None:
                    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                if run.status.value not in TERMINAL_TOOL_RUN_STATUSES:
                    publish(exit_code=int(process.returncode or -1) if process is not None else -1)
                    self._cancel_tool_run_obj(run, reason="shell command cancelled")
                raise
            except Exception:
                if process is not None and process.returncode is None:
                    with suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
                if stdout_task is not None and stderr_task is not None:
                    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                publish(exit_code=int(process.returncode or -1) if process is not None else -1)
                raise

        task_observed = False

        def _consume_task_exception(done_task: asyncio.Task[Any]) -> None:
            """Consume unobserved background exceptions and keep run state consistent."""
            nonlocal task_observed
            with suppress(asyncio.CancelledError):
                exc = done_task.exception()
                if exc is None or task_observed:
                    return
                if run.status.value not in TERMINAL_TOOL_RUN_STATUSES:
                    publish(exit_code=-1)
                    self._fail_tool_run(run, error=str(exc))
                agent = self.agents.get(run.agent_id) if isinstance(getattr(self, "agents", None), dict) else None
                self._record_exception(
                    stage="shell_task",
                    exc=exc,
                    agent=agent,
                    payload={
                        "tool_name": "shell",
                        "tool_run_id": run.id,
                        "action": json_ready(action),
                    },
                )
                self._persist_snapshot()

        task = asyncio.create_task(worker(), name=f"shell:{run.id}")
        task.add_done_callback(_consume_task_exception)
        self.tool_tasks[run.id] = task
        if not blocking or inline_wait_seconds <= 0.0:
            return _running_payload(run_id=run.id, result=run.result, inline_wait_seconds=inline_wait_seconds)
        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout=inline_wait_seconds)
            task_observed = True
            return result
        except asyncio.TimeoutError:
            return _running_payload(run_id=run.id, result=run.result, inline_wait_seconds=inline_wait_seconds)
        except Exception:
            task_observed = True
            raise


def _running_payload(*, run_id: str, result: dict[str, Any] | None, inline_wait_seconds: float) -> dict[str, Any]:
    """Build standardized background-running shell response."""
    payload: dict[str, Any] = {
        "status": "running",
        "background": True,
        "tool_run_id": run_id,
        "result": json_ready(result) if isinstance(result, dict) else result,
    }
    if inline_wait_seconds > 0:
        payload["inline_wait_seconds"] = inline_wait_seconds
    return payload


def _parse_shell_timeout_seconds(value: Any, *, default: float) -> float:
    """Parse shell timeout from action payload with config default fallback."""
    raw_seconds = default if value is None else value
    try:
        seconds = float(raw_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("shell timeout_seconds must be numeric") from exc
    if seconds < 1.0:
        raise ValueError("shell timeout_seconds must be >= 1")
    return seconds
