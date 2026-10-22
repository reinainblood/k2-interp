"""K2 native routing semantics, including selection-only correction bias.

Matches the upstream Apache-2.0 K2HorizonSparseMoeBlock computation. Expert names
are logical module-list indices, never expert-parallel physical locations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def route(logits: torch.Tensor, block) -> dict[str, torch.Tensor]:
    if block.router_score_func == "sigmoid":
        scores = torch.sigmoid(logits.float())
    elif block.router_score_func == "softmax":
        scores = F.softmax(logits, dim=-1, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported router score function {block.router_score_func}")
    choice = scores if block.gate.bias is None else scores + block.gate.bias.to(scores.dtype)
    _, ids = torch.topk(choice, block.top_k, dim=-1)
    weights = scores.gather(-1, ids)
    if block.norm_topk_prob:
        weights = weights / weights.sum(-1, keepdim=True)
    weights = weights * block.router_scaling_factor
    ordered = choice.topk(min(block.top_k + 1, choice.shape[-1]), dim=-1).values
    margin = (
        ordered[:, block.top_k - 1] - ordered[:, block.top_k]
        if choice.shape[-1] > block.top_k
        else torch.zeros_like(ordered[:, 0])
    )
    # Sigmoid scores are not a categorical distribution. Normalize explicitly for entropy.
    probability = scores / scores.sum(-1, keepdim=True)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
    return {
        "router_logits": logits.float(),
        "router_scores": scores,
        "selection_scores": choice,
        "expert_ids": ids,
        "expert_weights": weights,
        "routing_margin": margin,
        "router_entropy": entropy,
    }


def sparse_forward(block, hidden_states: torch.Tensor, edit) -> tuple[torch.Tensor, torch.Tensor]:
    """Intervention-only reference forward; identity is tested against native K2.

    We preserve native expert accumulation order and shared-expert computation.
    Frozen weights + autograd through inputs are supported for attribution.
    """
    batch, length, width = hidden_states.shape
    flat = hidden_states.reshape(-1, width)
    logits = F.linear(flat, block.gate.weight)
    logits = edit("router_logits", logits.reshape(batch, length, -1)).reshape_as(logits)
    routes = route(logits, block)
    ids = (
        edit("expert_ids", routes["expert_ids"].reshape(batch, length, -1))
        .reshape(-1, block.top_k)
        .long()
    )
    if bool(((ids < 0) | (ids >= block.num_experts)).any()):
        raise ValueError("Expert id outside native expert set")
    if bool((ids.sort(-1).values.diff(dim=-1) == 0).any()):
        raise ValueError("Duplicate expert selections are invalid")
    # If ids are replaced, recalculate weights from the original unbiased scores.
    weights = routes["router_scores"].gather(-1, ids)
    if block.norm_topk_prob:
        weights = weights / weights.sum(-1, keepdim=True)
    weights = weights * block.router_scaling_factor
    weights = edit("expert_weights", weights.reshape(batch, length, -1)).reshape_as(weights)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("Expert weights must be finite and nonnegative")
    weights = weights.to(hidden_states.dtype)
    output = torch.zeros_like(flat)
    mask = F.one_hot(ids, num_classes=block.num_experts).permute(2, 1, 0)
    for expert_idx in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        slot, rows = torch.where(mask[expert_idx].squeeze(0))
        value = block.experts[expert_idx](flat[None, rows].reshape(-1, width))
        output.index_add_(0, rows, (value * weights[rows, slot, None]).to(output.dtype))
    output = output.reshape(batch, length, width)
    if block.num_shared_experts > 0:
        output = output + block.shared_experts(hidden_states)
    return output, logits
