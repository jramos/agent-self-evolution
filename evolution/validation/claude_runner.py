"""ClaudeCodeAgentRunner — drive ``claude -p`` headlessly and map to AgentRunResult.

Implements the AgentRunner protocol so the agnostic scorer/validator consume it
unchanged. The candidate prompt-section under test is injected via
``--append-system-prompt-file`` (the installer writes the candidate to that file;
``claude`` reads it fresh each invocation), so the user's real CLAUDE.md is never
touched during validation.

Invocation recipe (verified against claude 2.1.169):
  - subscription auth via CLAUDE_CODE_OAUTH_TOKEN env (NOT --bare; bare ignores the token)
  - hermetic per-run: HOME=fresh tmp (no ambient ~/.claude CLAUDE.md / plugins / memory),
    --strict-mcp-config (no MCP servers), --no-session-persistence
  - cwd + --add-dir = fixture_dir; an OS sandbox (settings.json) confines writes to the
    sandbox — defense against an agent editing real repos via absolute paths
  - --output-format stream-json --verbose streams assistant tool_use blocks + a final
    result event carrying total_cost_usd + token usage
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from evolution.core.lm_timing_callback import (
    COST_LEDGER,
    CostCeilingExceeded,
    CostLedger,
)
from evolution.validation.agent_runner import AgentRunResult, TaskRunContext

DEFAULT_CLAUDE_TIMEOUT_SECONDS = 300
_DEFAULT_ALLOWED_TOOLS = ["Read", "Edit", "Write", "Bash", "Glob", "Grep"]


class ClaudeCodeAgentRunner:
    """Invoke ``claude -p`` and parse the resulting stream-json."""

    def __init__(
        self,
        *,
        append_prompt_file: Optional[Path] = None,
        model: str = "sonnet",
        timeout_seconds: int = DEFAULT_CLAUDE_TIMEOUT_SECONDS,
        allowed_tools: Optional[list[str]] = None,
        claude_command: str = "claude",
        cost_ledger: CostLedger = COST_LEDGER,
    ) -> None:
        # The installer owns ``append_prompt_file``; ``claude`` reads it via
        # --append-system-prompt-file each run, so re-installing a new candidate
        # is picked up without reconstructing the runner.
        self.append_prompt_file = append_prompt_file
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.allowed_tools = allowed_tools or list(_DEFAULT_ALLOWED_TOOLS)
        self.claude_command = claude_command
        self.cost_ledger = cost_ledger

    def run(self, ctx: TaskRunContext) -> AgentRunResult:
        home = Path(tempfile.mkdtemp(prefix="cl_claude_home_"))
        try:
            self._write_sandbox_settings(home, ctx.fixture_dir)
            env = {
                **os.environ,
                "HOME": str(home),
                "CLAUDE_CODE_OAUTH_TOKEN": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", ""),
                **ctx.extra_env,
            }
            argv = [
                self.claude_command, "-p", ctx.user_message,
                "--output-format", "stream-json", "--verbose",
                "--model", self.model,
                "--no-session-persistence",
                "--strict-mcp-config",
                "--permission-mode", "bypassPermissions",
                "--add-dir", str(ctx.fixture_dir),
                "--allowedTools", ",".join(self.allowed_tools),
            ]
            if self.append_prompt_file is not None:
                argv += ["--append-system-prompt-file", str(self.append_prompt_file)]
            start = time.time()
            try:
                proc = subprocess.run(
                    argv, env=env, cwd=str(ctx.fixture_dir),
                    capture_output=True, text=True,
                    timeout=self.timeout_seconds, check=False,
                )
            except subprocess.TimeoutExpired:
                result = AgentRunResult(
                    tool_calls_seq=[], final_text_tail="",
                    duration_seconds=time.time() - start,
                    error=f"claude -p timed out after {self.timeout_seconds}s",
                )
            except FileNotFoundError as exc:
                result = AgentRunResult(
                    tool_calls_seq=[], final_text_tail="",
                    duration_seconds=time.time() - start,
                    error=f"claude command not found: {exc}",
                )
            else:
                result = _parse_stream_json(
                    proc.stdout, duration_seconds=time.time() - start,
                    stderr_tail=proc.stderr[-500:],
                )
        finally:
            shutil.rmtree(home, ignore_errors=True)
        # Record + enforce the cost ceiling eagerly: Layer-1/convention scoring
        # makes no in-process LM call, so the BaseLM.__call__ guard would never
        # fire for an agent-cost overrun. Check + raise here instead.
        self.cost_ledger.record_agent_cost(result.agent_cost_usd)
        state = self.cost_ledger.get_abort_state()
        if state is not None:
            raise CostCeilingExceeded(*state)
        return result

    @staticmethod
    def _write_sandbox_settings(home: Path, fixture_dir: Path) -> None:
        """OS-level containment: confine filesystem writes to the fixture sandbox."""
        claude_dir = home / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        allow = [f"{fixture_dir}/**", f"{home}/**", "/tmp/**", "/private/var/folders/**"]
        settings = {
            "sandbox": {
                "enabled": True,
                "filesystem": {"allowRead": allow, "allowWrite": allow},
                "network": {"allowedDomains": []},
                "autoAllowBashIfSandboxed": True,
            }
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings, indent=2))


def _parse_stream_json(
    stdout: str, *, duration_seconds: float, stderr_tail: str = ""
) -> AgentRunResult:
    """Parse ``claude -p --output-format stream-json`` events into an AgentRunResult.

    Events: ``system/init`` (model), ``assistant`` (content blocks incl. ``tool_use``),
    and a final ``result`` (total_cost_usd, usage, final text, is_error). A $0 cost
    alongside any run is treated as uncaptured (a billed run is not free), mirroring
    the Hermes runner's $0-distrust.
    """
    tool_calls_seq: list[str] = []
    tool_calls_with_args: list[dict] = []
    final_text = ""
    cost: Optional[float] = None
    tokens: dict = {}
    model_name: Optional[str] = None
    err: Optional[str] = None
    saw_result = False

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = ev.get("type")
        if etype == "system" and ev.get("subtype") == "init":
            model_name = ev.get("model") or model_name
        elif etype == "assistant":
            for block in (ev.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name")
                    if name:
                        tool_calls_seq.append(str(name))
                        args = block.get("input")
                        tool_calls_with_args.append(
                            {"name": str(name),
                             "arguments": args if isinstance(args, dict) else {}}
                        )
        elif etype == "result":
            saw_result = True
            final_text = str(ev.get("result") or "")
            cost = ev.get("total_cost_usd")
            u = ev.get("usage") or {}
            tokens = {
                "input_tokens": u.get("input_tokens", 0),
                "output_tokens": u.get("output_tokens", 0),
                "cache_read_tokens": u.get("cache_read_input_tokens", 0),
                "cache_write_tokens": u.get("cache_creation_input_tokens", 0),
            }
            if ev.get("is_error"):
                err = f"claude result is_error: {final_text[:200]}"

    if not saw_result and err is None:
        err = f"no result event in claude stream-json output (stderr: {stderr_tail[:200]})"

    captured = cost is not None and cost > 0
    return AgentRunResult(
        tool_calls_seq=tool_calls_seq,
        final_text_tail=final_text[-4096:],
        duration_seconds=duration_seconds,
        model_name=model_name,
        error=err,
        tool_calls_with_args=tool_calls_with_args,
        agent_cost_usd=cost if captured else None,
        agent_cost_source="actual" if captured else "uncaptured",
        agent_tokens=tokens,
    )
