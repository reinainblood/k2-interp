"""Harbor adapter: run a K2 Horizon checkpoint as an agent system and freeze every trajectory.

Harbor owns the task container, timeouts, artifact collection, and the separate
verifier. This adapter owns the agent loop. It talks to one deployed k2-interp
checkpoint over ``/v1/chat/completions``, executes one fenced ``bash`` block per
assistant turn inside the container, and records, per trial, under the agent
logs directory Harbor already collects:

    k2trace/trajectory.jsonl   one record per turn; native reasoning verbatim
    k2trace/frozen.json        exact prompt/completion token ids per turn, plus
                               the full final-turn sequence for teacher-forced replay
    k2trace/adapter.json       endpoint, checkpoint, sampling, limits, adapter version

It also fills Harbor's ``AgentContext.rollout_details`` with the same token
identity, so the trajectory is first-class in Harbor's own result files.

The loop itself (:class:`AgentLoop`) depends on nothing but two callables, so it
is tested without Harbor. ``harbor`` is imported lazily inside :class:`K2Agent`;
this module imports without it, and the package never imports torch. Harbor
needs Python 3.12+, so run Harbor from its own environment with this package
installed beside it, and select the adapter with
``harbor run ... -a k2_interp.harbor:K2Agent --ak checkpoint=sft_3_1500``.

Tool protocol (text, because the chat endpoint has no native tool calling): the
assistant may put at most one fenced ``bash`` block in a turn; its stdout and
stderr come back as the next user message. A turn whose content contains the
submit token on its own line ends the run. No block and no submit token counts
as one idle turn; ``max_idle_turns`` idle turns in a row end the run.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ADAPTER_VERSION = "0.1.0"
TRACE_DIR = "k2trace"
SUBMIT_TOKEN = "SUBMIT"
FENCE = re.compile(r"```(?:bash|sh|shell)\s*\n(.*?)```", re.DOTALL)

SYSTEM_PROMPT = """You are working in a Linux container at {workdir}. You cannot see the \
container; you act on it only by putting exactly one fenced ```bash block in each reply. \
The block is executed with bash, and its stdout and stderr are returned to you in the next \
message (truncated to {output_chars} characters). Commands run from {workdir}; state such as \
files persists between turns, shell variables do not.

When you are finished, or when you have nothing further you can do, write the single word \
{submit} on its own line (no fenced block in that reply). Do not write {submit} before you \
have written any deliverable the task asks for. If you cannot complete the task, say so plainly \
and then write {submit}."""


def extract_command(content: str) -> str | None:
    """Return the first fenced bash block, or None. A second block is ignored on purpose."""
    match = FENCE.search(content or "")
    return match.group(1).strip() if match else None


def is_submit(content: str, token: str = SUBMIT_TOKEN) -> bool:
    return any(line.strip() == token for line in (content or "").splitlines())


def truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    head = limit // 2
    return text[:head] + f"\n...[{len(text) - limit} characters truncated]...\n" + text[-head:]


def post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode())


class FrozenTrajectory:
    """Per-turn token identity, written after every turn so a timeout still leaves a record."""

    def __init__(self, path: Path):
        self.path = path
        self.turns: list[dict] = []

    def add(self, turn: int, block: dict) -> None:
        self.turns.append(
            {
                "turn": turn,
                "run_id": block["run_id"],
                "revision": block["revision"],
                "prompt_token_ids": block["input_ids"],
                "completion_token_ids": block["output_ids"],
                "logprobs": block["output_logprobs"],
                "finish_reason": block["finish_reason"],
                "captured": block.get("captured", False),
            }
        )
        self.write()

    def payload(self) -> dict:
        final = self.turns[-1] if self.turns else None
        return {
            "schema": "k2-interp/harbor-frozen/1",
            "turns": self.turns,
            # The sequence the model saw on its last turn with that turn's completion
            # appended. Replay it teacher-forced to capture every position once.
            "final_sequence": (
                final["prompt_token_ids"] + final["completion_token_ids"] if final else []
            ),
            "final_prompt_length": len(final["prompt_token_ids"]) if final else 0,
        }

    def write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.payload()))
        os.replace(tmp, self.path)

    def rollout_detail(self) -> dict:
        return {
            "prompt_token_ids": [t["prompt_token_ids"] for t in self.turns],
            "completion_token_ids": [t["completion_token_ids"] for t in self.turns],
            "logprobs": [t["logprobs"] for t in self.turns],
            "extra": {
                "k2_interp_run_id": [t["run_id"] for t in self.turns],
                "finish_reason": [t["finish_reason"] for t in self.turns],
            },
        }


@dataclass
class LoopSettings:
    workdir: str = "/workspace"
    max_turns: int = 60
    max_idle_turns: int = 3
    exec_timeout_sec: int = 600
    output_chars: int = 8000
    submit_token: str = SUBMIT_TOKEN


@dataclass
class ExecOutcome:
    stdout: str | None
    stderr: str | None
    return_code: int


@dataclass
class LoopResult:
    stop: str
    turns: int
    n_input_tokens: int
    n_output_tokens: int
    messages: list[dict] = field(default_factory=list)


class AgentLoop:
    """The text-protocol loop, independent of Harbor.

    ``chat(messages) -> response`` must return the ``/v1/chat/completions`` JSON
    (with its ``k2_interp`` block). ``execute(command) -> ExecOutcome`` runs a
    shell command in the task environment.
    """

    def __init__(
        self,
        chat: Callable[[list[dict]], dict],
        execute: Callable[[str], Awaitable[ExecOutcome]],
        trace_dir: Path,
        settings: LoopSettings | None = None,
    ):
        self.chat = chat
        self.execute = execute
        self.trace_dir = Path(trace_dir)
        self.settings = settings or LoopSettings()
        self.frozen = FrozenTrajectory(self.trace_dir / "frozen.json")

    def system_prompt(self) -> str:
        s = self.settings
        return SYSTEM_PROMPT.format(
            workdir=s.workdir, output_chars=s.output_chars, submit=s.submit_token
        )

    async def run(self, instruction: str) -> LoopResult:
        s = self.settings
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        system = self.system_prompt()
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
        n_input = n_output = 0
        idle = 0
        stop = "max_turns"
        turns_done = 0
        with (self.trace_dir / "trajectory.jsonl").open("a") as log:
            log.write(json.dumps({"turn": 0, "role": "system", "content": system}) + "\n")
            log.write(json.dumps({"turn": 0, "role": "user", "content": instruction}) + "\n")
            for turn in range(1, s.max_turns + 1):
                started = time.time()
                response = self.chat(messages)
                block = response["k2_interp"]
                message = response["choices"][0]["message"]
                usage = response.get("usage", {})
                n_input += usage.get("prompt_tokens", 0)
                n_output += usage.get("completion_tokens", 0)
                self.frozen.add(turn, block)
                turns_done = turn
                content = message.get("content", "") or ""
                command = extract_command(content)
                submitted = is_submit(content, s.submit_token)
                record = {
                    "turn": turn,
                    "role": "assistant",
                    "content": content,
                    "reasoning": message.get("reasoning_content", ""),
                    "tool_call": None if submitted else command,
                    "submitted": submitted,
                    "finish_reason": block["finish_reason"],
                    "usage": usage,
                    "run_id": block["run_id"],
                    "started_at": started,
                    "finished_at": time.time(),
                }
                # Keep the assistant turn exactly as the server rendered it; K2's
                # template requires the reasoning field on every history turn.
                messages.append(message)
                if submitted:
                    log.write(json.dumps(record) + "\n")
                    stop = "submitted"
                    break
                if command is None:
                    idle += 1
                    log.write(json.dumps(record) + "\n")
                    if idle >= s.max_idle_turns:
                        stop = "idle"
                        break
                    nudge = (
                        "No fenced bash block was found. Put exactly one ```bash block in "
                        f"your reply to act, or write {s.submit_token} on its own line to finish."
                    )
                    messages.append({"role": "user", "content": nudge})
                    log.write(json.dumps({"turn": turn, "role": "user", "content": nudge}) + "\n")
                    continue
                idle = 0
                result = await self.execute(command)
                out = (result.stdout or "") + (
                    "\n[stderr]\n" + result.stderr if result.stderr else ""
                )
                observation = f"[exit {result.return_code}]\n{truncate(out, s.output_chars)}"
                record["tool_result"] = {
                    "return_code": result.return_code,
                    "stdout_chars": len(result.stdout or ""),
                    "stderr_chars": len(result.stderr or ""),
                }
                log.write(json.dumps(record) + "\n")
                messages.append({"role": "user", "content": observation})
                log.write(json.dumps({"turn": turn, "role": "user", "content": observation}) + "\n")
            log.write(json.dumps({"turn": -1, "role": "adapter", "stop": stop}) + "\n")
        return LoopResult(stop, turns_done, n_input, n_output, messages)


def chat_payload(
    messages: list[dict],
    *,
    checkpoint: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    max_context: int,
    reasoning_effort: str,
    capture_router: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": checkpoint,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "max_context": max_context,
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
    }
    if capture_router:
        from .schema import ROUTER_SUMMARY_FIELDS

        payload["capture"] = {
            "router": True,
            "router_fields": list(ROUTER_SUMMARY_FIELDS),
            "layers": [],
            "sites": [],
        }
    return payload


def _options_model():
    from typing import Annotated

    from harbor.agents.options import AgentOptions, Env
    from pydantic import Field

    class K2AgentOptions(AgentOptions):
        endpoint: Annotated[
            str,
            Env("K2_INTERP_ENDPOINT", fallback="K2_INTERP_ENDPOINT"),
            Field(description="Base URL of the deployed checkpoint, e.g. https://...modal.run"),
        ]
        checkpoint: str = Field(default="sft_3_1500", description="Registry checkpoint tag")
        token_id: Annotated[
            str | None,
            Env("MODAL_PROXY_TOKEN_ID", fallback="MODAL_PROXY_TOKEN_ID"),
            Field(default=None, description="Modal proxy-auth token id (Modal-Key header)"),
        ]
        token_secret: Annotated[
            str | None,
            Env("MODAL_PROXY_TOKEN_SECRET", fallback="MODAL_PROXY_TOKEN_SECRET"),
            Field(default=None, description="Modal proxy-auth token secret (Modal-Secret header)"),
        ]
        max_turns: int = Field(default=60, ge=1)
        max_idle_turns: int = Field(default=3, ge=1)
        max_tokens: int = Field(default=8192, ge=1, le=8192)
        max_context: int = Field(default=262144, ge=2, le=524288)
        # IFM's recommended sampling. Greedy decoding produces repetition loops
        # that look exactly like impasse, so it is not the default here.
        temperature: float = Field(default=1.0, ge=0, le=5)
        top_p: float = Field(default=0.95, gt=0, le=1)
        seed: int = Field(default=7, ge=0)
        reasoning_effort: str = Field(default="high")
        exec_timeout_sec: int = Field(default=600, ge=1)
        output_chars: int = Field(default=8000, ge=256)
        request_timeout_sec: int = Field(default=3600, ge=1)
        workdir: str = Field(default="/workspace")
        submit_token: str = Field(default=SUBMIT_TOKEN)
        capture_router: bool = Field(
            default=False,
            description="Capture summary routing fields live on every turn (adds trace cost)",
        )

    return K2AgentOptions


def _agent_class():
    from harbor.agents.base import BaseAgent
    from harbor.agents.capabilities import AgentCapabilities
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext

    class K2Agent(BaseAgent):
        """Harbor wrapper around :class:`AgentLoop` for one deployed checkpoint."""

        capabilities = AgentCapabilities()
        options_model = _options_model()

        @staticmethod
        def name() -> str:
            return "k2-interp"

        def version(self) -> str:
            return ADAPTER_VERSION

        async def setup(self, environment: BaseEnvironment) -> None:
            await environment.exec(f"mkdir -p {self.options.workdir}")

        def _headers(self) -> dict:
            o = self.options
            if not (o.token_id and o.token_secret):
                raise ValueError(
                    "Modal proxy credentials missing: set MODAL_PROXY_TOKEN_ID and "
                    "MODAL_PROXY_TOKEN_SECRET (or --ak token_id= --ak token_secret=)"
                )
            return {"Modal-Key": o.token_id, "Modal-Secret": o.token_secret}

        def _chat(self, messages: list[dict]) -> dict:
            o = self.options
            payload = chat_payload(
                messages,
                checkpoint=o.checkpoint,
                max_tokens=o.max_tokens,
                temperature=o.temperature,
                top_p=o.top_p,
                seed=o.seed,
                max_context=o.max_context,
                reasoning_effort=o.reasoning_effort,
                capture_router=o.capture_router,
            )
            url = o.endpoint.rstrip("/") + "/v1/chat/completions"
            try:
                return post_json(url, payload, self._headers(), o.request_timeout_sec)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:2000]
                raise RuntimeError(f"k2-interp chat failed: HTTP {exc.code}: {detail}") from exc

        async def run(
            self, instruction: str, environment: BaseEnvironment, context: AgentContext
        ) -> None:
            o = self.options
            trace_dir = self.logs_dir / TRACE_DIR
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / "adapter.json").write_text(
                json.dumps(
                    {
                        "adapter_version": ADAPTER_VERSION,
                        "endpoint": o.endpoint,
                        "checkpoint": o.checkpoint,
                        "sampling": {
                            "temperature": o.temperature,
                            "top_p": o.top_p,
                            "seed": o.seed,
                            "max_tokens": o.max_tokens,
                            "max_context": o.max_context,
                            "reasoning_effort": o.reasoning_effort,
                        },
                        "limits": {
                            "max_turns": o.max_turns,
                            "max_idle_turns": o.max_idle_turns,
                            "exec_timeout_sec": o.exec_timeout_sec,
                            "output_chars": o.output_chars,
                        },
                        "submit_token": o.submit_token,
                        "capture_router": o.capture_router,
                        "session_id": self.session_id,
                        "context_id": str(self.context_id) if self.context_id else None,
                    },
                    indent=2,
                )
            )

            async def execute(command: str) -> ExecOutcome:
                result = await environment.exec(
                    command, cwd=o.workdir, timeout_sec=o.exec_timeout_sec
                )
                return ExecOutcome(result.stdout, result.stderr, result.return_code)

            loop = AgentLoop(
                self._chat,
                execute,
                trace_dir,
                LoopSettings(
                    workdir=o.workdir,
                    max_turns=o.max_turns,
                    max_idle_turns=o.max_idle_turns,
                    exec_timeout_sec=o.exec_timeout_sec,
                    output_chars=o.output_chars,
                    submit_token=o.submit_token,
                ),
            )
            outcome = await loop.run(instruction)
            context.n_input_tokens = outcome.n_input_tokens
            context.n_output_tokens = outcome.n_output_tokens
            context.rollout_details = [loop.frozen.rollout_detail()]
            context.metadata = {
                "k2_interp": {
                    "checkpoint": o.checkpoint,
                    "turns": outcome.turns,
                    "stop": outcome.stop,
                    "trace_dir": str(trace_dir),
                }
            }

    return K2Agent


def __getattr__(name: str):
    # Lazy so `k2_interp.harbor` imports without harbor installed; Harbor resolves
    # `k2_interp.harbor:K2Agent` through this hook.
    if name == "K2Agent":
        return _agent_class()
    if name == "K2AgentOptions":
        return _options_model()
    raise AttributeError(name)
