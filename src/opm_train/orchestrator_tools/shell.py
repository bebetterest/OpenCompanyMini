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
        timeout_seconds = float(action.get("timeout_seconds") or 60.0)
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
            try:
                await asyncio.wait_for(process.wait(), timeout=max(1.0, timeout_seconds))
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=False)
                result = publish(exit_code=int(process.returncode or 0))
                self._complete_tool_run(run, result=result)
                return result
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                publish(exit_code=int(process.returncode or -1), timed_out=True)
                self._fail_tool_run(run, error="shell command timed out")
                raise ValueError("shell command timed out")
            except asyncio.CancelledError:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                if run.status.value not in TERMINAL_TOOL_RUN_STATUSES:
                    publish(exit_code=int(process.returncode or -1))
                    self._cancel_tool_run_obj(run, reason="shell command cancelled")
                raise

        task = asyncio.create_task(worker(), name=f"shell:{run.id}")
        self.tool_tasks[run.id] = task
        if not blocking or inline_wait_seconds <= 0.0:
            return _running_payload(run_id=run.id, result=run.result, inline_wait_seconds=inline_wait_seconds)
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=inline_wait_seconds)
        except asyncio.TimeoutError:
            return _running_payload(run_id=run.id, result=run.result, inline_wait_seconds=inline_wait_seconds)


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
