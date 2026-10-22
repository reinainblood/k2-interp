import json

import pytest

from k2_interp.runtime import Runtime
from k2_interp.schema import GenerationSpec, RunSpec, Sample


def test_interruption_retains_aligned_partial_tokens(native_model, tokenizer, tmp_path):
    runtime = Runtime(native_model, tokenizer, {"checkpoint": "test"}, tmp_path)
    calls = 0

    def interrupt(module, args, kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")

    handle = native_model.register_forward_pre_hook(interrupt, with_kwargs=True)
    try:
        with pytest.raises(RuntimeError, match="simulated interruption"):
            runtime.run(
                RunSpec(
                    run_id="partial",
                    checkpoint="test",
                    samples=[Sample(id="s", input_ids=[4, 5, 6])],
                    generation=GenerationSpec(max_new_tokens=5),
                )
            )
    finally:
        handle.remove()
    base = tmp_path / "partial/samples/s"
    assert not (base / "complete.json").exists()
    attempt = next(base.glob("attempt-*"))
    progress = json.loads((attempt / "progress.json").read_text())["generation"]
    assert progress["processed_tokens"] == 4
    assert len(progress["output_ids"]) == 1
    assert len(progress["output_logprobs"]) == 1
    for meta_path in attempt.glob("step-*.json"):
        meta = json.loads(meta_path.read_text())
        for record in meta["records"]:
            assert record["token_ids"] == [
                progress["all_token_ids"][i] for i in record["positions"]
            ]
