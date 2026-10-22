"""Temporary native hooks. Handles are always removed, including after exceptions."""

from __future__ import annotations

import types
from contextlib import AbstractContextManager

import torch
import torch.nn.functional as F

from .routing import route, sparse_forward
from .schema import CaptureSpec, Intervention


class NativeTrace(AbstractContextManager):
    def __init__(
        self,
        model,
        spec: CaptureSpec,
        sink=None,
        interventions: list[Intervention] | None = None,
        values: dict[str, torch.Tensor] | None = None,
        observers: dict | None = None,
        transforms: dict | None = None,
    ):
        self.model, self.spec, self.sink = model, spec, sink
        self.interventions = interventions or []
        self.values = values or {}
        self.observers = observers or {}
        self.transforms = transforms or {}
        self.handles, self.originals = [], []
        self.positions = []
        self.token_ids = None
        self.step = 0
        self.phase = "prefill"
        self.routes = {}
        self.effective_routes = {}
        self.applied = set()
        self.applied_positions = [set() for _ in self.interventions]
        self.pending_tensors = {}
        self.pending_records = []
        self.pending_bytes = 0
        self.layers = model.model.layers
        wanted = spec.layers + [p.layer for p in self.interventions]
        if any(i >= len(self.layers) for i in wanted):
            raise ValueError("Requested layer is outside model")
        for patch in self.interventions:
            if patch.site.startswith("expert_") or patch.site == "router_logits":
                if not hasattr(self.layers[patch.layer].mlp, "experts"):
                    raise ValueError("Routing intervention requested on a dense layer")
            if patch.value_key and patch.value_key not in self.values:
                raise ValueError(f"Missing intervention tensor {patch.value_key}")
            if patch.value_key:
                value = self.values[patch.value_key]
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Intervention values must be finite")
                if patch.site == "expert_ids" and value.is_floating_point():
                    raise ValueError("Expert-id replacement tensors must have integer dtype")

    def begin(
        self, positions: list[int], *, step: int, phase: str, token_ids: list[int] | None = None
    ) -> None:
        self.flush()
        if token_ids is not None and len(token_ids) != len(positions):
            raise ValueError("Token IDs and capture positions differ in length")
        self.token_ids = token_ids
        self.positions, self.step, self.phase = positions, step, phase
        self.routes.clear()
        self.effective_routes.clear()

    def _selected(self, tensor):
        if tensor.shape[0] != 1:
            raise ValueError("Canonical capture requires batch size one")
        if tensor.shape[1] != len(self.positions):
            raise ValueError("Capture/token alignment mismatch")
        keep = (
            list(range(len(self.positions)))
            if self.spec.positions is None
            else [i for i, p in enumerate(self.positions) if p in self.spec.positions]
        )
        return keep, tensor[0, keep]

    def emit(self, layer: int, tensors: dict[str, torch.Tensor]) -> None:
        if self.sink is None:
            return
        selected, values = None, {}
        for name, tensor in tensors.items():
            selected, value = self._selected(tensor)
            if value.is_floating_point() and name not in (
                "router_logits",
                "router_scores",
                "selection_scores",
                "expert_weights",
                "router_entropy",
                "routing_margin",
            ):
                value = value.to(getattr(torch, self.spec.tensor_dtype))
            values[f"layers.{layer}.{name}"] = value
        if not selected:
            return
        positions = [self.positions[i] for i in selected]
        values["positions"] = torch.tensor(positions, dtype=torch.int64)
        keys = []
        for key, value in values.items():
            if key in self.pending_tensors:
                if key == "positions" and torch.equal(self.pending_tensors[key], value.cpu()):
                    continue
                raise ValueError("Duplicate capture key; call begin() for each model forward")
            added = value.numel() * value.element_size()
            if self.sink.bytes + self.pending_bytes + added > self.spec.max_bytes:
                raise ValueError("Capture byte budget exceeded")
            self.pending_tensors[key] = value.detach().cpu().contiguous().clone()
            self.pending_bytes += added
            keys.append(key)
        self.pending_records.append(
            {
                "step": self.step,
                "phase": self.phase,
                "layer": layer,
                "positions": positions,
                "token_ids": [self.token_ids[i] for i in selected]
                if self.token_ids is not None
                else None,
                "keys": keys,
            }
        )

    def flush(self):
        if self.pending_tensors:
            self.sink.record(
                self.step,
                {"step": self.step, "phase": self.phase, "records": self.pending_records},
                self.pending_tensors,
            )
            self.pending_tensors = {}
            self.pending_records = []
            self.pending_bytes = 0

    def edit(self, layer: int, site: str, tensor: torch.Tensor) -> torch.Tensor:
        result = tensor
        for i, patch in enumerate(self.interventions):
            if patch.layer != layer or patch.site != site:
                continue
            overlap = [
                (j, patch.positions.index(p))
                for j, p in enumerate(self.positions)
                if p in patch.positions
            ]
            if not overlap:
                continue
            indices, source_rows = zip(*overlap, strict=True)
            result = result.clone()
            current = result[:, list(indices)]
            if patch.operation == "zero":
                value = torch.zeros_like(current)
            elif patch.operation == "scale":
                value = current * patch.scale
            else:
                replacement = self.values[patch.value_key].to(result.device, dtype=result.dtype)
                if replacement.shape != (len(patch.positions), *result.shape[2:]):
                    raise ValueError(f"Patch tensor shape mismatch at {site}")
                value = replacement[list(source_rows)].unsqueeze(0)
                if patch.operation == "add":
                    value = current + value * patch.scale
            result[:, list(indices)] = value
            self.applied.add(i)
            self.applied_positions[i].update(self.positions[j] for j in indices)
        transform = self.transforms.get((layer, site))
        if transform:
            result = transform(result, self.positions)
        observer = self.observers.get((layer, site))
        if observer:
            observer(result, self.positions)
        if layer in self.spec.layers and site in self.spec.sites:
            self.emit(layer, {site: result})
        return result

    def _pre(self, layer, site):
        def hook(module, args, kwargs):
            if args:
                return (self.edit(layer, site, args[0]), *args[1:]), kwargs
            key = "hidden_states"
            return args, {**kwargs, key: self.edit(layer, site, kwargs[key])}

        return hook

    def _post(self, layer, site):
        def hook(module, args, result):
            if isinstance(result, tuple):
                return self.edit(layer, site, result[0]), *result[1:]
            return self.edit(layer, site, result)

        return hook

    def _moe_pre(self, layer):
        def hook(block, args):
            if self.spec.expert_outputs and layer in self.spec.layers:
                x = args[0]
                logits = F.linear(x.reshape(-1, x.shape[-1]), block.gate.weight)
                self.routes[layer] = route(logits, block)

        return hook

    def _moe_post(self, layer):
        def hook(block, args, result):
            if self.spec.router:
                routes = self.effective_routes.get(layer)
                if routes is None:
                    routes = route(result[1], block)
                batch, length = args[0].shape[:2]
                self.emit(
                    layer, {k: v.reshape(batch, length, *v.shape[1:]) for k, v in routes.items()}
                )

        return hook

    def _expert_post(self, layer, expert):
        def hook(module, args, result):
            routes = self.routes[layer]
            # The native block orders rows by top-k slot then original token index.
            slot, row = torch.where((routes["expert_ids"] == expert).T)
            if row.numel() != result.shape[0]:
                raise ValueError("Expert output row alignment failed")
            tensor = torch.zeros(
                (1, len(self.positions), result.shape[-1]), device=result.device, dtype=result.dtype
            )
            tensor[0, row] = result * routes["expert_weights"][row, slot, None].to(result.dtype)
            self.emit(layer, {f"expert.{expert}.weighted_out": tensor})

        return hook

    def __enter__(self):
        try:
            route_layers = {
                p.layer
                for p in self.interventions
                if p.site in ("router_logits", "expert_ids", "expert_weights")
            }
            if route_layers and self.spec.expert_outputs:
                raise ValueError(
                    "Expert-output capture with edited routing requires a separate baseline trace"
                )
            for i, block in enumerate(self.layers):
                for module, site in (
                    (block, "resid_pre"),
                    (block.post_attention_layernorm, "resid_mid"),
                    (block.mlp, "moe_in"),
                ):
                    self.handles.append(
                        module.register_forward_pre_hook(self._pre(i, site), with_kwargs=True)
                    )
                for module, site in (
                    (block.self_attn, "attn_out"),
                    (block.mlp, "moe_out"),
                    (block, "resid_post"),
                ):
                    self.handles.append(module.register_forward_hook(self._post(i, site)))
                if hasattr(block.mlp, "shared_experts"):
                    self.handles.append(
                        block.mlp.shared_experts.register_forward_hook(self._post(i, "shared_out"))
                    )
                if hasattr(block.mlp, "experts"):
                    self.handles.append(block.mlp.register_forward_pre_hook(self._moe_pre(i)))
                    self.handles.append(block.mlp.register_forward_hook(self._moe_post(i)))
                    if self.spec.expert_outputs and i in self.spec.layers:
                        for e, expert in enumerate(block.mlp.experts):
                            self.handles.append(
                                expert.register_forward_hook(self._expert_post(i, e))
                            )
                    if i in route_layers:
                        self.originals.append((block.mlp, block.mlp.forward))

                        def forward(m, x, layer=i):
                            return sparse_forward(
                                m,
                                x,
                                lambda kind, t: self.edit(layer, kind, t),
                                lambda routed: self.effective_routes.__setitem__(layer, routed),
                            )

                        block.mlp.forward = types.MethodType(forward, block.mlp)
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        for handle in reversed(self.handles):
            handle.remove()
        for module, forward in self.originals:
            module.forward = forward
        self.handles.clear()
        self.originals.clear()
        self.routes.clear()
        self.effective_routes.clear()
        if not args or args[0] is None:
            self.flush()
        self.pending_tensors.clear()
        self.pending_records.clear()
        self.pending_bytes = 0
        return False
