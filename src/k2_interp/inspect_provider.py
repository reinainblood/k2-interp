"""Inspect AI model provider: run a deployed K2 checkpoint under any Inspect solver.

Registered as provider ``k2interp`` (see :mod:`k2_interp.inspect_registry` and the
``inspect_ai`` entry point), so ``inspect eval task.py --model k2interp/sft_3_1500``
talks to one deployed checkpoint over ``/v1/chat/completions`` with the same Modal
proxy credentials and payload the Harbor adapter uses.

Every turn's exact token identity (the gateway's ``k2_interp`` block) is appended
to a per-sample :class:`~k2_interp.harbor.FrozenTrajectory`::

    <trace_dir>/<eval_id>/<sample_id>/epoch-<n>/frozen.json   schema k2-interp/harbor-frozen/1
    <trace_dir>/<eval_id>/<sample_id>/epoch-<n>/provider.json endpoint, checkpoint, sampling

The path, the per-turn server ``run_id``s, and the checkpoint revision are written
to the sample store under ``k2_interp`` (logged as ``EvalSample.store``), and each
turn's ``ModelOutput.metadata["k2_interp"]`` names its run id, turn number, and
frozen path. ``k2-interp inspect-replay`` reads them back.

Limits, all loud: no streaming, no native tools (tool emulation is not implemented
yet), one choice per call, no stop sequences. Proxy credentials come only from the
environment so they never land in ``EvalSpec.model_args`` inside the log.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
from pathlib import Path
from typing import Any

import anyio
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model._model_output import ChatCompletionChoice, StopReason
from inspect_ai.tool import ToolChoice, ToolInfo

from . import harbor
from .harbor import FrozenTrajectory, chat_payload
from .registry import CHECKPOINTS

PROVIDER_VERSION = "0.1.0"
PROVIDER_NAME = "k2interp"
STORE_KEY = "k2_interp"
MESSAGE_KEY = "k2_interp"  # ChatMessageAssistant.metadata key holding the raw server message
GATEWAY_MAX_TOKENS = 8192  # ChatRequest.max_tokens upper bound
TOKEN_FIELDS = ("input_ids", "output_ids", "output_logprobs")
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class GatewayError(RuntimeError):
    """Non-2xx response from the k2-interp gateway."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"k2-interp chat failed: HTTP {status}: {detail}")
        self.status = status


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def safe_name(value: Any) -> str:
    """Filesystem- and RunSpec-safe rendering of an Inspect sample id."""
    text = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value)).strip("._") or "sample"
    return text.replace("..", "_")


def trace_path(trace_dir: str | Path, eval_id: str, sample_id: Any, epoch: int) -> Path:
    return Path(trace_dir) / safe_name(eval_id) / safe_name(sample_id) / f"epoch-{epoch}"


def _reasoning(message: ChatMessageAssistant) -> str:
    if isinstance(message.content, str):
        return ""
    return "".join(part.reasoning for part in message.content if isinstance(part, ContentReasoning))


def to_gateway_messages(messages: list[ChatMessage]) -> list[dict]:
    """Inspect messages -> the gateway's chat message dicts.

    Assistant turns produced by this provider carry the server's raw message in
    ``metadata["k2_interp"]["message"]``; it is reused verbatim (including any
    ``think_fast`` keys) as long as the visible text and reasoning are unchanged,
    so history re-renders exactly as the model produced it. Any other assistant
    turn is rebuilt with ``reasoning_content`` (empty when absent), which K2's
    chat template requires on every assistant history turn.
    """
    out: list[dict] = []
    for message in messages:
        if isinstance(message, ChatMessageTool):
            raise NotImplementedError(
                "k2interp: tool messages are not supported; tool emulation is not implemented yet"
            )
        if isinstance(message, ChatMessageAssistant):
            if message.tool_calls:
                raise NotImplementedError(
                    "k2interp: assistant tool calls are not supported; "
                    "tool emulation is not implemented yet"
                )
            text, reasoning = message.text, _reasoning(message)
            raw = ((message.metadata or {}).get(MESSAGE_KEY) or {}).get("message")
            if (
                isinstance(raw, dict)
                and raw.get("content", "") == text
                and raw.get("reasoning_content", "") == reasoning
            ):
                out.append(dict(raw))
            else:
                out.append({"role": "assistant", "content": text, "reasoning_content": reasoning})
        elif isinstance(message, ChatMessageSystem | ChatMessageUser):
            if not isinstance(message.content, str) and any(
                part.type not in ("text", "reasoning") for part in message.content
            ):
                raise NotImplementedError("k2interp: only text content is supported")
            out.append({"role": message.role, "content": message.text})
        else:  # pragma: no cover - ChatMessage is a closed union
            raise TypeError(f"Unsupported message type {type(message).__name__}")
    return out


def stop_reason(finish_reason: str | None) -> StopReason:
    return {"eos": "stop", "stop": "stop", "length": "max_tokens"}.get(
        finish_reason or "", "unknown"
    )


def to_model_output(response: dict, checkpoint: str, extra: dict | None = None) -> ModelOutput:
    """Gateway response -> ModelOutput with reasoning, usage, stop reason, and token receipt."""
    raw = response["choices"][0]["message"]
    block = response["k2_interp"]
    content: list = []
    reasoning = raw.get("reasoning_content") or ""
    if reasoning:
        content.append(ContentReasoning(reasoning=reasoning))
    content.append(ContentText(text=raw.get("content") or ""))
    receipt = {
        "run_id": block["run_id"],
        "revision": block["revision"],
        "finish_reason": block["finish_reason"],
        "mode": block.get("mode"),
        "captured": block.get("captured", False),
        **(extra or {}),
    }
    message = ChatMessageAssistant(
        content=content,
        model=checkpoint,
        source="generate",
        metadata={MESSAGE_KEY: {"message": raw, **receipt}},
    )
    usage = response.get("usage") or {}
    return ModelOutput(
        model=checkpoint,
        choices=[
            ChatCompletionChoice(message=message, stop_reason=stop_reason(block["finish_reason"]))
        ],
        usage=ModelUsage(
            input_tokens=usage.get("prompt_tokens", len(block["input_ids"])),
            output_tokens=usage.get("completion_tokens", len(block["output_ids"])),
            total_tokens=usage.get(
                "total_tokens", len(block["input_ids"]) + len(block["output_ids"])
            ),
        ),
        metadata={MESSAGE_KEY: receipt},
    )


def _filter_tokens(key: Any, value: Any) -> Any:
    # Keep the logged ModelCall small: token arrays live in frozen.json.
    if key in TOKEN_FIELDS and isinstance(value, list):
        return f"<{len(value)} values; see frozen.json>"
    return value


class K2InterpAPI(ModelAPI):
    """``k2interp/<checkpoint>``: one deployed k2-interp checkpoint gateway."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig | None = None,
        endpoint: str | None = None,
        capture_router: bool = False,
        max_context: int = 262144,
        trace_dir: str | None = None,
        request_timeout: int = 3600,
        stream: bool = False,
        **model_args: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, [], config or GenerateConfig())
        if model_args:
            raise ValueError(
                f"k2interp: unknown model args {sorted(model_args)}; proxy credentials are "
                "read from MODAL_PROXY_TOKEN_ID / MODAL_PROXY_TOKEN_SECRET only"
            )
        if model_name not in CHECKPOINTS:
            raise ValueError(
                f"k2interp: unknown checkpoint {model_name!r}; use one of {sorted(CHECKPOINTS)}"
            )
        if _bool(stream):
            raise ValueError("k2interp: streaming is not supported by the k2-interp gateway")
        self.checkpoint = model_name
        self.revision = CHECKPOINTS[model_name]
        self.endpoint = endpoint or base_url or os.environ.get("K2_INTERP_ENDPOINT")
        if not self.endpoint:
            raise ValueError("k2interp: set K2_INTERP_ENDPOINT or -M endpoint=https://...")
        self.capture_router = _bool(capture_router)
        self.max_context = int(max_context)
        self.trace_dir = Path(
            trace_dir or os.environ.get("K2_INTERP_TRACE_DIR") or "logs/k2trace"
        ).resolve()
        self.request_timeout = int(request_timeout)
        self._traces: dict[str, tuple[Path, FrozenTrajectory]] = {}

    # -- Inspect hooks ----------------------------------------------------------------

    def max_tokens(self) -> int | None:
        return GATEWAY_MAX_TOKENS

    def force_reasoning_history(self):
        # K2's template needs reasoning on every assistant history turn.
        return "all"

    def connection_key(self) -> str:
        return f"{self.endpoint}|{self.checkpoint}"

    def should_retry(self, ex: Exception) -> bool:
        if isinstance(ex, GatewayError):
            return ex.status in RETRY_STATUS
        return isinstance(ex, urllib.error.URLError | TimeoutError | ConnectionError)

    # -- request building ---------------------------------------------------------------

    def _headers(self) -> dict:
        token_id = os.environ.get("MODAL_PROXY_TOKEN_ID")
        token_secret = os.environ.get("MODAL_PROXY_TOKEN_SECRET")
        if not (token_id and token_secret):
            raise ValueError(
                "Modal proxy credentials missing: set MODAL_PROXY_TOKEN_ID and "
                "MODAL_PROXY_TOKEN_SECRET"
            )
        return {"Modal-Key": token_id, "Modal-Secret": token_secret}

    def sampling(self, config: GenerateConfig) -> dict:
        # IFM's recommended sampling unless the eval overrides it.
        max_tokens = config.max_tokens or GATEWAY_MAX_TOKENS
        if max_tokens > GATEWAY_MAX_TOKENS:
            raise ValueError(
                f"k2interp: max_tokens {max_tokens} > gateway cap {GATEWAY_MAX_TOKENS}"
            )
        return {
            "max_tokens": max_tokens,
            "temperature": 1.0 if config.temperature is None else config.temperature,
            "top_p": 0.95 if config.top_p is None else config.top_p,
            "seed": 7 if config.seed is None else config.seed,
            "reasoning_effort": config.reasoning_effort or "high",
        }

    def payload(self, input: list[ChatMessage], config: GenerateConfig) -> dict:
        if config.num_choices not in (None, 1):
            raise NotImplementedError("k2interp: num_choices > 1 is not supported")
        if config.stop_seqs:
            raise NotImplementedError("k2interp: stop sequences are not supported")
        return chat_payload(
            to_gateway_messages(input),
            checkpoint=self.checkpoint,
            max_context=self.max_context,
            capture_router=self.capture_router,
            **self.sampling(config),
        )

    def _check_identity(self, response: dict) -> None:
        served = response.get("model")
        block = response.get("k2_interp")
        if served != self.checkpoint:
            raise RuntimeError(
                f"k2interp: checkpoint mismatch: requested {self.checkpoint!r}, "
                f"endpoint {self.endpoint} served {served!r}"
            )
        if not isinstance(block, dict):
            raise RuntimeError("k2interp: response has no k2_interp token block")
        if block.get("revision") != self.revision:
            raise RuntimeError(
                f"k2interp: checkpoint mismatch: {self.checkpoint} is pinned to "
                f"{self.revision}, endpoint served revision {block.get('revision')!r}"
            )

    def _post(self, payload: dict) -> dict:
        url = self.endpoint.rstrip("/") + "/v1/chat/completions"
        try:
            # Resolved through the harbor module so tests patch one function for both.
            return harbor.post_json(url, payload, self._headers(), self.request_timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            raise GatewayError(exc.code, detail) from exc

    # -- traces -------------------------------------------------------------------------

    def _trace(self, config: GenerateConfig) -> tuple[Path, FrozenTrajectory] | None:
        from inspect_ai.log._samples import sample_active

        active = sample_active()
        if active is None:
            return None
        key = active.id  # unique per sample attempt, so a retried sample starts fresh
        if key not in self._traces:
            directory = trace_path(self.trace_dir, active.eval_id, active.sample.id, active.epoch)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "provider.json").write_text(
                json.dumps(
                    {
                        "provider": PROVIDER_NAME,
                        "provider_version": PROVIDER_VERSION,
                        "endpoint": self.endpoint,
                        "checkpoint": self.checkpoint,
                        "revision": self.revision,
                        "sampling": self.sampling(config),
                        "max_context": self.max_context,
                        "capture_router": self.capture_router,
                        "eval_id": active.eval_id,
                        "task": active.task,
                        "sample_id": active.sample.id,
                        "epoch": active.epoch,
                    },
                    indent=2,
                )
            )
            self._traces[key] = (directory, FrozenTrajectory(directory / "frozen.json"))
        return self._traces[key]

    @staticmethod
    def _record_in_store(directory: Path, frozen: FrozenTrajectory, revision: str) -> None:
        from inspect_ai.util import store

        store().set(
            STORE_KEY,
            {
                "frozen": str(directory / "frozen.json"),
                "trace_dir": str(directory),
                "revision": revision,
                "run_ids": [t["run_id"] for t in frozen.turns],
                "turns": len(frozen.turns),
            },
        )

    # -- generate ----------------------------------------------------------------------

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput | Exception, ModelCall]:
        if tools:
            raise NotImplementedError(
                "k2interp: tools were offered but tool emulation is not supported yet; "
                "use a text-only solver (tools would otherwise be silently dropped)"
            )
        if config.stream_idle_timeout is not None or _stream_requested():
            raise NotImplementedError("k2interp: streaming is not supported")
        payload = self.payload(input, config)
        call = ModelCall.create(payload, None, filter=_filter_tokens)
        response = await anyio.to_thread.run_sync(self._post, payload)
        call.set_response(response)
        self._check_identity(response)
        extra: dict = {}
        trace = self._trace(config)
        if trace is not None:
            directory, frozen = trace
            turn = len(frozen.turns) + 1
            frozen.add(turn, response["k2_interp"])
            self._record_in_store(directory, frozen, self.revision)
            extra = {"turn": turn, "frozen": str(directory / "frozen.json")}
        return to_model_output(response, self.checkpoint, extra), call


def _stream_requested() -> bool:
    try:
        from inspect_ai.model._stream import model_stream_requested
    except ImportError:  # pragma: no cover - older/newer inspect without the hook
        return False
    return model_stream_requested()
