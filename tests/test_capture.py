import copy
import json
import time

import pytest
import torch

from k2_interp.artifacts import TraceWriter, iter_records
from k2_interp.hooks import NativeTrace
from k2_interp.routing import route, sparse_forward
from k2_interp.runtime import Runtime
from k2_interp.schema import CaptureSpec, GenerationSpec, Intervention, RunSpec, Sample


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_router_semantics_and_identity(native_model, dtype):
    block = copy.deepcopy(native_model.model.layers[1].mlp).to(dtype)
    x = torch.randn(1, 7, 16, dtype=dtype)
    original, logits = block(x)
    actual, other = sparse_forward(block, x, lambda site, t: t)
    torch.testing.assert_close(original, actual, rtol=0, atol=0)
    torch.testing.assert_close(logits, other, rtol=0, atol=0)
    routing = route(logits, block)
    torch.testing.assert_close(routing["expert_weights"].sum(-1), torch.full((7,), 2.5))
    assert routing["expert_ids"].shape == (7, 2)
    # Bias affects selection, never the gate values used for weighting.
    with torch.no_grad():
        saved = block.gate.bias.clone()
        block.gate.bias.zero_()
        block.gate.bias[0] = 100
        biased = route(logits, block)
        assert (biased["expert_ids"][:, 0] == 0).all()
        torch.testing.assert_close(biased["router_scores"], routing["router_scores"])
        block.gate.bias.copy_(saved)


def test_capture_preserves_logits_and_removes_hooks(native_model, tmp_path):
    ids = torch.tensor([[4, 5, 6]])
    baseline = native_model(ids).logits
    writer = TraceWriter(tmp_path, {"run_id": "r"}, {"model": "tiny-native"})
    sink = writer.sample("x", 1000000)
    with NativeTrace(
        native_model,
        CaptureSpec(layers=[1, 2], sites=["moe_in", "moe_out", "resid_post"], expert_outputs=True),
        sink,
    ) as trace:
        trace.begin([0, 1, 2], step=0, phase="prefill")
        observed = native_model(ids).logits
    sink.finish({"ok": True})
    torch.testing.assert_close(baseline, observed, rtol=0, atol=0)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in native_model.modules())
    records = list(iter_records(tmp_path / "r"))
    assert any("layers.1.router_logits" in t for _, _, t in records)
    assert any("layers.1.moe_in" in t for _, _, t in records)
    weighted = sum(t[k] for _, _, t in records for k in t if k.startswith("layers.1.expert."))
    assert weighted.shape == (3, 16)


def test_patch_effect_cleanup_and_validation(native_model):
    ids = torch.tensor([[4, 5, 6]])
    baseline = native_model(ids).logits
    patch = Intervention(layer=2, site="resid_post", positions=[2], operation="zero")
    with NativeTrace(native_model, CaptureSpec(router=False), interventions=[patch]) as trace:
        trace.begin([0, 1, 2], step=0, phase="prefill")
        edited = native_model(ids).logits
        assert trace.applied == {0}
    assert not torch.equal(edited[:, -1], baseline[:, -1])
    torch.testing.assert_close(native_model(ids).logits, baseline, rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        with NativeTrace(native_model, CaptureSpec()) as trace:
            raise RuntimeError("deliberate failure")
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in native_model.modules())


def test_generation_positions_teacher_forcing_and_resume(native_model, tokenizer, tmp_path):
    runtime = Runtime(native_model, tokenizer, {"checkpoint": "test"}, tmp_path)
    spec = RunSpec(
        run_id="test-run",
        checkpoint="test",
        samples=[Sample(id="sample", input_ids=[4, 5, 6], continuation_ids=[7, 8])],
        capture=CaptureSpec(layers=[1], sites=["resid_post"]),
    )
    receipt = runtime.run(spec)
    result_path = tmp_path / "test-run" / receipt["samples"]["sample"]["attempt"] / "result.json"
    result = json.loads(result_path.read_text())
    # Cached incremental likelihoods match a single full forward with the same prefix.
    full = native_model(torch.tensor([[4, 5, 6, 7, 8]])).logits[0].log_softmax(-1)
    assert result["output_logprobs"] == pytest.approx(
        [float(full[2, 7]), float(full[3, 8])], abs=1e-5
    )
    positions = {p for _, meta, _ in iter_records(tmp_path / "test-run") for p in meta["positions"]}
    assert positions == {0, 1, 2, 3, 4}
    assert runtime.run(spec)["samples"]["sample"]["resumed"]
    altered = spec.model_copy(deep=True)
    altered.samples[0].input_ids = [5, 5, 6]
    with pytest.raises(ValueError, match="different request"):
        runtime.run(altered)
    result_path.write_text("{}")
    with pytest.raises(ValueError, match="Corrupt"):
        runtime.run(spec)


def test_reject_overlong_and_expired(native_model, tokenizer, tmp_path):
    runtime = Runtime(native_model, tokenizer, {"checkpoint": "test"}, tmp_path)
    with NativeTrace(native_model, CaptureSpec(router=False)) as trace:
        with pytest.raises(ValueError, match="context"):
            runtime.execute(
                Sample(id="s", input_ids=[2, 3]),
                GenerationSpec(max_new_tokens=2),
                trace,
                3,
                time.monotonic() + 10,
            )
        with pytest.raises(TimeoutError):
            runtime.execute(Sample(id="s", input_ids=[2, 3]), GenerationSpec(), trace, 100, 0)


def test_greedy_generation_matches_upstream(native_model, tokenizer, tmp_path):
    runtime = Runtime(native_model, tokenizer, {"checkpoint": "test"}, tmp_path)
    with NativeTrace(native_model, CaptureSpec(router=False)) as trace:
        result = runtime.execute(
            Sample(id="s", input_ids=[4, 5, 6]),
            GenerationSpec(max_new_tokens=3),
            trace,
            100,
            time.monotonic() + 10,
        )
    expected = native_model.generate(torch.tensor([[4, 5, 6]]), max_new_tokens=3, do_sample=False)
    assert result["output_ids"] == expected[0, 3:].tolist()
