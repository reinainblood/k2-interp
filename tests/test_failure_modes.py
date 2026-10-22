import json

import pytest
import torch
from test_analysis import synthetic_runs

from k2_interp.artifacts import TraceWriter, iter_records
from k2_interp.causal import score
from k2_interp.datasets import build_dataset
from k2_interp.dictionary import train_from_config
from k2_interp.hooks import NativeTrace
from k2_interp.runtime import Runtime
from k2_interp.schema import CaptureSpec, Intervention, Sample


def test_partial_patch_and_nonfinite_rejected(native_model, tokenizer, tmp_path):
    runtime = Runtime(native_model, tokenizer, {"checkpoint": "test"}, tmp_path)
    sample = Sample(id="s", input_ids=[4, 5, 6], continuation_ids=[7])
    with pytest.raises(ValueError, match="every requested"):
        score(
            runtime,
            sample,
            patches=[
                Intervention(layer=2, site="resid_post", positions=[0, 900], operation="zero")
            ],
        )
    with pytest.raises(ValueError, match="finite"):
        NativeTrace(
            native_model,
            CaptureSpec(),
            interventions=[
                Intervention(
                    layer=2, site="resid_post", positions=[0], operation="add", value_key="v"
                )
            ],
            values={"v": torch.full((1, 16), float("nan"))},
        )
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in native_model.modules())


def test_edited_routes_are_the_actual_captured_routes(native_model, tmp_path):
    writer = TraceWriter(tmp_path, {"run_id": "routes"}, {"checkpoint": "tiny"})
    sink = writer.sample("s", 1_000_000)
    patch = Intervention(
        layer=1, site="expert_ids", positions=[0, 1], operation="replace", value_key="ids"
    )
    ids = torch.tensor([[2, 3], [2, 3]])
    with NativeTrace(
        native_model, CaptureSpec(), sink, interventions=[patch], values={"ids": ids}
    ) as trace:
        trace.begin([0, 1], step=0, phase="prefill")
        native_model(torch.tensor([[4, 5]]))
    sink.finish({})
    found = [
        t["layers.1.expert_ids"]
        for _, _, t in iter_records(tmp_path / "routes")
        if "layers.1.expert_ids" in t
    ]
    assert len(found) == 1
    torch.testing.assert_close(found[0], ids)


def test_training_interrupt_resume_equals_uninterrupted(tmp_path):
    synthetic_runs(tmp_path)
    build_dataset(
        [
            {"name": role, "role": role, "run": str(tmp_path / "a"), "site": "layers.1.resid_post"}
            for role in ("input", "target")
        ],
        tmp_path / "data",
    )
    cfg = {
        "kind": "sae",
        "dataset": "data",
        "output": "interrupted",
        "width": 16,
        "k": 4,
        "epochs": 3,
        "batch_size": 64,
        "checkpoint_steps": 2,
        "device": "cpu",
    }

    class Interruption(Exception):
        pass

    def interrupt():
        raise Interruption()

    with pytest.raises(Interruption):
        train_from_config(cfg, root=tmp_path, commit=interrupt)
    resumed = train_from_config(cfg, root=tmp_path)
    complete = train_from_config({**cfg, "output": "whole"}, root=tmp_path)
    assert resumed["weights_sha256"] == complete["weights_sha256"]


def test_split_leakage_and_sequence_mismatch_rejected(tmp_path):
    synthetic_runs(tmp_path)
    # Create legitimate-but-misaligned artifacts, not merely a corrupt file.
    writer = TraceWriter(tmp_path, {"run_id": "different"}, {"checkpoint": "other"})
    for i in range(4):
        sample = Sample(
            id=f"s{i}",
            input_ids=[9] * 96,
            split="train" if i < 3 else "validation",
            template_group=f"g{i}",
        )
        sink = writer.sample(sample.id, 1_000_000)
        sink.record(
            0,
            {"positions": list(range(96)), "layer": 1},
            {"layers.1.resid_post": torch.randn(96, 8)},
        )
        sink.finish({"all_token_ids": [9] * 96, "sample": sample.model_dump()})
    views = [
        {"name": "x", "role": "input", "run": str(tmp_path / "a"), "site": "layers.1.resid_post"},
        {
            "name": "y",
            "role": "target",
            "run": str(tmp_path / "different"),
            "site": "layers.1.resid_post",
        },
    ]
    with pytest.raises(ValueError, match="Token sequences"):
        build_dataset(views, tmp_path / "invalid")


def test_example_schema():
    from pathlib import Path

    from k2_interp.schema import RunSpec

    spec = RunSpec.model_validate(
        json.loads((Path(__file__).parents[1] / "examples/capture.json").read_text())
    )
    assert spec.samples[0].continuation
