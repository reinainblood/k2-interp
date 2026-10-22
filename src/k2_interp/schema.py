"""Versioned, fail-closed experiment contracts shared by local and remote execution."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "k2-interp/1"
SITES = ("resid_pre", "attn_out", "resid_mid", "moe_in", "moe_out", "shared_out", "resid_post")
ROUTER_FIELDS = (
    "router_logits",
    "router_scores",
    "selection_scores",
    "expert_ids",
    "expert_weights",
    "routing_margin",
    "router_entropy",
)
# Routing summary without the three 192-wide score vectors: ~6 KB/token across 58 layers.
ROUTER_SUMMARY_FIELDS = ("expert_ids", "expert_weights", "routing_margin", "router_entropy")


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def identifier(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,119}", value) or ".." in value:
        raise ValueError(f"Invalid artifact identifier: {value!r}")
    return value


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class CaptureSpec(Contract):
    router: bool = True
    # None keeps every router field. A subset (e.g. ROUTER_SUMMARY_FIELDS) drops the
    # 192-wide score vectors so whole-trajectory router traces stay small.
    router_fields: list[str] | None = None
    layers: list[int] = Field(default_factory=list)
    sites: list[str] = Field(default_factory=lambda: ["resid_post"])
    positions: list[int] | None = None
    # All intermediate rows are saved only at explicitly selected layers/sites.
    expert_outputs: bool = False
    # Exact L2 norms of weighted native expert outputs, without dense vector storage.
    expert_output_norms: bool = False
    tensor_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    max_bytes: int = Field(default=2_000_000_000, gt=0)

    @model_validator(mode="after")
    def validate_sites(self):
        if any(s not in SITES for s in self.sites):
            raise ValueError(f"sites must be drawn from {SITES}")
        if self.router_fields is not None and (
            not self.router_fields
            or len(set(self.router_fields)) != len(self.router_fields)
            or any(f not in ROUTER_FIELDS for f in self.router_fields)
        ):
            raise ValueError(f"router_fields must be a unique subset of {ROUTER_FIELDS}")
        if len(set(self.layers)) != len(self.layers) or any(i < 0 for i in self.layers):
            raise ValueError("layers must be unique, nonnegative indices")
        if self.positions is not None and (
            len(set(self.positions)) != len(self.positions) or any(p < 0 for p in self.positions)
        ):
            raise ValueError("positions must be unique absolute nonnegative token positions")
        return self


class GenerationSpec(Contract):
    max_new_tokens: int = Field(default=64, ge=0, le=8192)
    temperature: float = Field(default=0, ge=0, le=5)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int = Field(default=0, ge=0)


class Sample(Contract):
    id: str
    text: str | None = None
    messages: list[dict[str, Any]] | None = None
    input_ids: list[int] | None = None
    continuation_ids: list[int] | None = None
    continuation: str | None = None
    labels: dict[str, str | int | float | bool] = Field(default_factory=dict)
    template_group: str = "unspecified"
    split: Literal["train", "validation", "test", "exploratory"] = "exploratory"
    chat_template_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"reasoning_effort": "high"}
    )

    @model_validator(mode="after")
    def validate_input(self):
        identifier(self.id)
        if {"tokenize", "add_generation_prompt", "return_dict", "return_tensors"}.intersection(
            self.chat_template_kwargs
        ):
            raise ValueError("Tokenization/return-format options are controlled by the runtime")
        if sum(x is not None for x in (self.text, self.messages, self.input_ids)) != 1:
            raise ValueError("Supply exactly one of text, messages, input_ids")
        if self.continuation is not None and self.continuation_ids is not None:
            raise ValueError("Supply at most one continuation representation")
        if self.input_ids is not None and (not self.input_ids or min(self.input_ids) < 0):
            raise ValueError("input_ids must be nonempty and nonnegative")
        if self.continuation_ids is not None and (
            not self.continuation_ids or min(self.continuation_ids) < 0
        ):
            raise ValueError("continuation_ids must be nonempty and nonnegative")
        if self.continuation == "":
            raise ValueError("Empty continuation")
        return self


class RunSpec(Contract):
    run_id: str
    checkpoint: str = "sft_3_1500"
    samples: list[Sample] = Field(min_length=1)
    capture: CaptureSpec = Field(default_factory=CaptureSpec)
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    max_context: int = Field(default=8192, ge=2, le=524288)
    max_seconds: int = Field(default=3600, ge=1, le=86400)
    protocol: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_run(self):
        identifier(self.run_id)
        if len({s.id for s in self.samples}) != len(self.samples):
            raise ValueError("Sample ids must be unique")
        return self


class ChatRequest(Contract):
    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0, ge=0, le=5)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int = Field(default=0, ge=0)
    stream: bool = False
    max_context: int = Field(default=8192, ge=2, le=524288)
    chat_template_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"reasoning_effort": "high"}
    )
    # Optional live capture for this turn. The response's `k2_interp` block then
    # names the run_id holding the trace. Omit for plain, capture-free chat.
    capture: CaptureSpec | None = None


class Intervention(Contract):
    """A patch on absolute token positions. All tensor values come from safetensors."""

    layer: int = Field(ge=0)
    site: str
    positions: list[int] = Field(min_length=1)
    operation: Literal["replace", "add", "scale", "zero"]
    value_key: str | None = None
    scale: float = 1.0

    @model_validator(mode="after")
    def validate_patch(self):
        if self.site not in (*SITES, "router_logits", "expert_ids", "expert_weights"):
            raise ValueError("Unknown intervention site")
        if self.site == "expert_ids" and self.operation != "replace":
            raise ValueError("Expert ids support replacement only")
        if self.operation in ("replace", "add") and not self.value_key:
            raise ValueError("Replacement/addition requires a tensor value_key")
        if len(set(self.positions)) != len(self.positions) or min(self.positions) < 0:
            raise ValueError("Patch positions must be unique nonnegative absolute indices")
        return self
