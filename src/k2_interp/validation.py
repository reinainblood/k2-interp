"""Bounded full-weight verification. No scores or findings about any study protocol."""

from __future__ import annotations

import time

import torch

from .artifacts import atomic_json, iter_records
from .causal import intervention_sweep
from .hooks import NativeTrace
from .runtime import token_inputs
from .schema import CaptureSpec, GenerationSpec, RunSpec, Sample


def full_model_smoke(runtime, run_id):
    sample = Sample(id="factual", text="The capital of France is", continuation=" Paris.")
    prompt, continuation = token_inputs(runtime.tokenizer, sample)
    x = torch.tensor([prompt], device=runtime.device)
    with torch.no_grad():
        base = runtime.model(x, use_cache=False, logits_to_keep=1).logits.cpu()
        with NativeTrace(
            runtime.model,
            CaptureSpec(router=True, layers=[3, 30, 60], sites=["resid_post", "moe_in", "moe_out"]),
        ) as trace:
            trace.begin(list(range(len(prompt))), step=0, phase="prefill")
            captured = runtime.model(x, use_cache=False, logits_to_keep=1).logits.cpu()
    if not torch.equal(base, captured):
        raise ValueError("Capture changed model logits")
    started = time.monotonic()
    receipt = runtime.run(
        RunSpec(
            run_id=run_id,
            checkpoint=runtime.identity["checkpoint"],
            samples=[sample],
            generation=GenerationSpec(max_new_tokens=0),
            capture=CaptureSpec(layers=[3, 30, 60], sites=["resid_post", "moe_in", "moe_out"]),
            max_context=256,
            max_seconds=600,
        )
    )
    routes = {}
    observed_positions = set()
    for _, meta, tensors in iter_records(runtime.root / run_id):
        observed_positions.update(meta["positions"])
        for key, value in tensors.items():
            if key.endswith(".router_logits"):
                routes.setdefault(meta["layer"], set()).update(meta["positions"])
                if value.shape[-1] != 192:
                    raise ValueError("Unexpected router width")
            if key.endswith(".expert_weights"):
                torch.testing.assert_close(value.sum(-1), torch.full_like(value.sum(-1), 2.5))
    expected = set(range(len(prompt) + len(continuation)))
    if (
        set(routes) != set(range(3, 61))
        or any(p != expected for p in routes.values())
        or observed_positions != expected
    ):
        raise ValueError("Missing router layers or token positions")
    again = runtime.run(
        RunSpec(
            run_id=run_id,
            checkpoint=runtime.identity["checkpoint"],
            samples=[sample],
            generation=GenerationSpec(max_new_tokens=0),
            capture=CaptureSpec(layers=[3, 30, 60], sites=["resid_post", "moe_in", "moe_out"]),
            max_context=256,
            max_seconds=600,
        )
    )
    assert again["samples"]["factual"]["resumed"]
    causal = intervention_sweep(
        runtime,
        run_id + "-causal",
        sample.model_dump(),
        [
            [
                {
                    "layer": 60,
                    "site": "resid_post",
                    "positions": [len(prompt) - 1],
                    "operation": "scale",
                    "scale": 1,
                }
            ],
            [
                {
                    "layer": 60,
                    "site": "resid_post",
                    "positions": [len(prompt) - 1],
                    "operation": "zero",
                }
            ],
        ],
        max_context=256,
        max_seconds=600,
    )
    if causal["interventions"][0]["delta_sum_logprob"] != 0:
        raise ValueError("Identity intervention changed scores")
    if abs(causal["interventions"][1]["delta_sum_logprob"]) < 1e-6:
        raise ValueError("Ablation did not change target score")
    report = {
        "status": "passed",
        "model": runtime.identity,
        "capture_logit_identity": True,
        "router_layers": len(routes),
        "all_token_positions_covered": True,
        "resume_passed": True,
        "identity_intervention_passed": True,
        "ablation_delta": causal["interventions"][1]["delta_sum_logprob"],
        "intervention_cleanup_passed": causal["cleanup_identity_passed"],
        "elapsed_test_seconds": time.monotonic() - started,
        "trace_path": receipt["path"],
        "scope": "runtime verification; not a scientific study result",
    }
    atomic_json(runtime.root / run_id / "verification.json", report)
    runtime.commit()
    return report
