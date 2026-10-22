"""Live projections of captured tensors. No interpolation, simulated progress, or GPU RPCs.

The publisher coalesces intermediate UI frames on a background thread. Raw captures
and one projection per completed forward are durable, independent of the browser.
Prefill's projection is its LAST input position; prefill processes many tokens in
parallel. Other positions are inspected from the committed raw shard.
"""

from __future__ import annotations

import copy
import threading
import time
from datetime import UTC, datetime

from .artifacts import atomic_json, file_hash


def utc_now():
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def project_layer(layer: int, tensors: dict, row: int = -1) -> dict:
    """Numerically literal projection, also used to inspect arbitrary stored positions."""
    prefix = f"layers.{layer}."
    result = {}
    for key, tensor in tensors.items():
        if not key.startswith(prefix):
            continue
        site = key[len(prefix) :]
        value = tensor[row].float()
        if site in (
            "expert_ids",
            "expert_weights",
            "router_logits",
            "router_scores",
            "selection_scores",
        ):
            result[site] = tensor[row].tolist()
        elif site in ("router_entropy", "routing_margin"):
            result[site] = float(value.item())
        elif site.startswith("expert."):
            expert = site.split(".")[1]
            result.setdefault("expert_norms", {})[expert] = (
                float(value.item()) if site.endswith("weighted_norm") else float(value.norm())
            )
        else:
            result.setdefault("norms", {})[site] = float(value.norm())
            result.setdefault("rms", {})[site] = float(value.square().mean().sqrt())
    ids = tensors.get(prefix + "expert_ids")
    weights = tensors.get(prefix + "expert_weights")
    scores = tensors.get(prefix + "router_scores")
    if ids is not None and weights is not None:
        import torch

        n = scores.shape[-1] if scores is not None else int(ids.max()) + 1
        result["forward_expert_counts"] = torch.bincount(ids.flatten(), minlength=n).tolist()
        mass = torch.zeros(n, dtype=torch.float32)
        mass.scatter_add_(0, ids.flatten(), weights.float().flatten())
        result["forward_expert_mass"] = mass.tolist()
    return result


class CaptureCancelled(RuntimeError):
    pass


class LiveTelemetry:
    def __init__(self, run_id, publish, cancelled=lambda: False, interval=0.4):
        self.run_id, self.publish, self.cancelled = run_id, publish, cancelled
        self.interval = interval
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.cancel = threading.Event()
        self.state = {
            "schema": "k2-workbench/1",
            "run_id": run_id,
            "sequence": 0,
            "status": "loading",
            "updated_at": utc_now(),
            "layers": {},
            "projection": "last processed input token",
            "transport_errors": 0,
        }
        self.thread = threading.Thread(target=self._loop, daemon=True, name="k2-telemetry")

    def update(self, **values):
        with self.lock:
            self.state.update(copy.deepcopy(values))
            self.state["sequence"] += 1
            self.state["updated_at"] = utc_now()

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def _loop(self):
        while not self.stop.wait(self.interval):
            try:
                if self.cancelled():
                    self.cancel.set()
                self.publish(self.snapshot())
            except Exception:
                # A UI transport failure must not silently alter model execution.
                with self.lock:
                    self.state["transport_errors"] += 1

    def __enter__(self):
        self.thread.start()
        return self

    def check_cancel(self):
        if self.cancel.is_set():
            raise CaptureCancelled("Capture cancelled at a native layer boundary")

    def prompt(self, ids, tokenizer, identity):
        self.check_cancel()
        self.tokenizer = tokenizer
        self.update(
            status="prefill",
            input_ids=ids,
            token_text=tokenizer.convert_ids_to_tokens(ids),
            identity=identity,
            input_tokens=len(ids),
            output_ids=[],
            output_logprobs=[],
            output_text="",
        )

    def begin(self, positions, step, phase, token_ids):
        self.check_cancel()
        text = self.snapshot().get("token_text", [])
        if token_ids and positions[-1] >= len(text):
            text.extend(self.tokenizer.convert_ids_to_tokens(token_ids))
        self.update(
            status=phase,
            phase=phase,
            step=step,
            positions=positions,
            position=positions[-1],
            current_token_ids=token_ids,
            layers={},
            forward_started_at=utc_now(),
            forward_complete=False,
            token_text=text,
        )

    def capture(self, layer, positions, tensors):
        self.check_cancel()
        projection = project_layer(layer, tensors)
        with self.lock:
            entry = self.state["layers"].setdefault(str(layer), {})
            for key, value in projection.items():
                if key in ("norms", "rms", "expert_norms"):
                    entry.setdefault(key, {}).update(value)
                else:
                    entry[key] = value
            self.state["position"] = positions[-1]
            self.state["last_layer"] = layer
            self.state["sequence"] += 1
            self.state["updated_at"] = utc_now()

    def forward_complete(
        self, sink, prompt, generated, scores, tokenizer, generation_config, progress
    ):
        from .runtime import response_message

        self.update(
            forward_complete=False,
            committed_at=utc_now(),
            output_ids=generated,
            output_logprobs=scores,
            token_text=tokenizer.convert_ids_to_tokens(prompt + generated),
            output_text=tokenizer.decode(generated, skip_special_tokens=False),
            assistant_message=response_message(tokenizer, generation_config, prompt, generated),
            tensor_bytes=sink.bytes,
            attempt=str(sink.path.relative_to(sink.parent.root)),
        )
        # The full source of every projection is the adjacent checksummed tensor shard.
        path = sink.path / f"workbench-step-{self.state['step']:06d}.json"
        atomic_json(path, {**self.snapshot(), "forward_complete": True})
        sink.files[str(path.relative_to(sink.parent.root))] = file_hash(path)
        sink.checkpoint(progress)
        self.update(forward_complete=True, committed_at=utc_now())

    def finish(self, response):
        self.update(status="complete", result=response)

    def fail(self, exc):
        self.update(
            status="cancelled" if isinstance(exc, CaptureCancelled) else "failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=10)
        # Last publication failure remains recoverable from durable trace artifacts.
        for attempt in range(3):
            try:
                self.publish(self.snapshot())
                break
            except Exception:
                if attempt < 2:
                    time.sleep(0.2)
        return False
