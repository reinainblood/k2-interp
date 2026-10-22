import json

import pytest
import torch
from safetensors.torch import save_file

from k2_interp.analysis import feature_report, localize, router_comparison
from k2_interp.artifacts import TraceWriter, atomic_json, file_hash
from k2_interp.causal import (
    attribute_features,
    feature_graph,
    intervention_sweep,
    reconstruction_fidelity,
)
from k2_interp.datasets import ActivationDataset, build_dataset
from k2_interp.dictionary import (
    SparseDictionary,
    evaluate,
    fit_normalization,
    load_dictionary,
    train_from_config,
)
from k2_interp.runtime import Runtime
from k2_interp.schema import CaptureSpec, GenerationSpec, RunSpec, Sample


def synthetic_runs(root):
    torch.manual_seed(12)
    transform = torch.randn(8, 8)
    for name in ("a", "b"):
        writer = TraceWriter(root, {"run_id": name}, {"checkpoint": name, "revision": name})
        for i in range(4):
            torch.manual_seed(100 + i)
            x = torch.randn(96, 8)
            if name == "b":
                x = x @ transform
            y = x @ transform + 0.2 * x.relu()
            sample = Sample(
                id=f"s{i}",
                input_ids=list(range(96)),
                split="train" if i < 3 else "validation",
                template_group=f"g{i}",
            )
            sink = writer.sample(sample.id, 1_000_000)
            sink.record(
                0,
                {"positions": list(range(96)), "layer": 1, "phase": "prefill"},
                {"layers.1.resid_post": x, "layers.1.moe_in": x, "layers.1.moe_out": y},
            )
            sink.finish({"all_token_ids": list(range(96)), "sample": sample.model_dump()})


@pytest.mark.parametrize("kind", ["sae", "skip_transcoder", "crosscoder"])
def test_dictionary_end_to_end(tmp_path, kind):
    synthetic_runs(tmp_path)
    views = []
    for role in ("input", "target"):
        names = ["a", "b"] if kind == "crosscoder" else ["a"]
        for name in names:
            site = (
                "resid_post"
                if kind != "skip_transcoder"
                else ("moe_in" if role == "input" else "moe_out")
            )
            views.append(
                {
                    "name": f"{role}_{name}",
                    "run": str(tmp_path / name),
                    "site": f"layers.1.{site}",
                    "role": role,
                }
            )
    build_dataset(views, tmp_path / "data")
    ds = ActivationDataset(tmp_path / "data")
    torch.manual_seed(0)
    before = SparseDictionary(ds.input_dims, ds.target_dims, 32, 8, kind)
    fit_normalization(before, ds, 64)
    initial = evaluate(before, ds)["normalized_mse"]
    cfg = {
        "kind": kind,
        "dataset": "data",
        "output": "dict",
        "device": "cpu",
        "width": 32,
        "k": 8,
        "epochs": 15,
        "batch_size": 64,
        "learning_rate": 0.01,
        "max_steps": 200,
    }
    meta = train_from_config(cfg, root=tmp_path)
    assert meta["metrics"]["normalized_mse"] < initial
    dictionary, _ = load_dictionary(tmp_path / "dict")
    x, _ = next(ds.batches("validation", 10))
    torch.testing.assert_close(dictionary.encode(x)[0], dictionary.encode(x[:1])[0])
    assert train_from_config(cfg, root=tmp_path)["weights_sha256"] == meta["weights_sha256"]
    with pytest.raises(ValueError, match="configuration"):
        train_from_config({**cfg, "k": 4}, root=tmp_path)
    assert feature_report(tmp_path / "dict", tmp_path / "data", tmp_path / "features.json")[
        "features"
    ]


def test_alignment_and_integrity(tmp_path):
    synthetic_runs(tmp_path)
    views = [
        {"name": "x", "run": str(tmp_path / "a"), "site": "layers.1.resid_post", "role": "input"},
        {"name": "y", "run": str(tmp_path / "b"), "site": "layers.1.resid_post", "role": "target"},
    ]
    build_dataset(views, tmp_path / "data")
    ds = ActivationDataset(tmp_path / "data")
    shard = tmp_path / "data" / ds.manifest["shards"][0]["file"]
    shard.write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash"):
        next(ds.batches("train", 16))
    # Mutating a completed capture invalidates its receipt even if it still parses.
    complete = json.loads((tmp_path / "b/samples/s0/complete.json").read_text())
    p = tmp_path / "b" / complete["attempt"] / "result.json"
    data = json.loads(p.read_text())
    data["all_token_ids"][0] = 999
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash|Unverified"):
        build_dataset(views, tmp_path / "invalid")


def test_causal_and_graph(native_model, tokenizer, tmp_path):
    runtime = Runtime(
        native_model, tokenizer, {"checkpoint": "test", "revision": "fixture"}, tmp_path
    )
    sample = Sample(id="s", input_ids=[4, 5, 6], continuation_ids=[7, 8])
    report = intervention_sweep(
        runtime,
        "causal",
        sample.model_dump(),
        [
            [
                {
                    "layer": 2,
                    "site": "resid_post",
                    "positions": [2],
                    "operation": "scale",
                    "scale": 1,
                }
            ],
            [{"layer": 2, "site": "resid_post", "positions": [2], "operation": "zero"}],
            [
                {
                    "layer": 1,
                    "site": "router_logits",
                    "positions": [2],
                    "operation": "scale",
                    "scale": 0,
                }
            ],
        ],
    )
    assert report["interventions"][0]["delta_sum_logprob"] == 0
    assert report["interventions"][1]["kl_from_baseline"] > 0
    # Construct a small valid dictionary with known provenance, independent of a training claim.
    dictionary = SparseDictionary([16], [16], 16, 8, "sae")
    with torch.no_grad():
        dictionary.encoder.bias.fill_(2)
        dictionary.threshold.fill_(0)
    d = tmp_path / "dict"
    d.mkdir()
    save_file(dictionary.state_dict(), str(d / "dictionary.safetensors"))
    views = [
        {"role": r, "site": "layers.1.resid_post", "source_identity": {"revision": "fixture"}}
        for r in ("input", "target")
    ]
    atomic_json(
        d / "dictionary.json",
        {
            **dictionary.metadata(),
            "views": views,
            "weights_sha256": file_hash(d / "dictionary.safetensors"),
        },
    )
    nodes = [
        {
            "id": "f0",
            "layer": 1,
            "site": "resid_post",
            "dictionary": "dict",
            "feature": 0,
            "position": 2,
        }
    ]
    graph = feature_graph(runtime, "graph", sample.model_dump(), nodes, [0, 1, 1.001])
    assert graph["cleanup_identity_passed"]
    assert next(e for e in graph["edges"] if e["factor"] == 1)["effect"] == 0
    attr = attribute_features(runtime, "attribute", sample.model_dump(), nodes)
    epsilon_effect = next(e for e in graph["edges"] if e["factor"] == 1.001)["effect"]
    assert epsilon_effect / 0.001 == pytest.approx(
        attr["edges"][0]["attribution"], rel=0.08, abs=0.002
    )
    fidelity = reconstruction_fidelity(
        runtime, "fidelity", sample.model_dump(), "dict", 1, "resid_post"
    )
    assert fidelity["kl_from_baseline"] >= 0


def test_localize_and_routes(native_model, tokenizer, tmp_path):
    runtime = Runtime(native_model, tokenizer, {"checkpoint": "test"}, tmp_path)
    samples = [Sample(id="a", input_ids=[2, 3, 4]), Sample(id="b", input_ids=[2, 5, 4])]
    spec = RunSpec(
        run_id="r",
        checkpoint="test",
        samples=samples,
        generation=GenerationSpec(max_new_tokens=0),
        capture=CaptureSpec(layers=[1], sites=["resid_post"]),
    )
    runtime.run(spec)
    report = localize(
        tmp_path / "r",
        [
            {
                "clean_id": "a",
                "control_id": "b",
                "clean_position": 2,
                "control_position": 2,
                "group": "g",
            }
        ],
        ["layers.1.resid_post"],
        tmp_path / "localize",
    )
    assert report["metrics"][0]["mean_pair_l2"] > 0
    comparison = router_comparison(tmp_path / "r", tmp_path / "r", [1, 2], tmp_path / "routes.json")
    assert all(r["js_divergence"] == 0 and r["topk_jaccard"] == 1 for r in comparison["records"])
