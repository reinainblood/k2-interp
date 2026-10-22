"""Small neutral full-weight integration exercise; not a scientific training corpus."""

import argparse
import json
import uuid
from pathlib import Path

import modal

from k2_interp.artifacts import atomic_json
from k2_interp.cli import app_name, remote
from k2_interp.schema import RunSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="sft_3_1500")
    parser.add_argument("--output", default=".private/pipeline-verification.json")
    args = parser.parse_args()
    worker = remote(args.checkpoint)
    analyze = modal.Function.from_name(app_name(args.checkpoint), "analyze", environment_name="FDR")
    train = modal.Function.from_name(
        app_name(args.checkpoint), "train_dictionary", environment_name="FDR"
    )
    name = "pipeline-check-" + uuid.uuid4().hex[:12]
    request = json.loads((Path(__file__).parents[1] / "examples/capture.json").read_text())
    request.update(run_id=name, checkpoint=args.checkpoint)
    request["capture"]["layers"] = [30]
    spec = RunSpec.model_validate(request)
    capture = worker.run.remote(spec.model_dump())
    print("full-model captures complete", flush=True)
    base = f"{args.checkpoint}/{name}"
    evaluations = analyze.remote(
        {"operation": "evaluate", "run": base, "output": base + "/evaluation.json"}
    )
    training = {}
    for kind in ("sae", "skip_transcoder"):
        x = "resid_post" if kind == "sae" else "moe_in"
        y = "resid_post" if kind == "sae" else "moe_out"
        views = [
            {"name": "input", "run": base, "site": f"layers.30.{x}", "role": "input"},
            {"name": "target", "run": base, "site": f"layers.30.{y}", "role": "target"},
        ]
        analyze.remote({"operation": "dataset", "views": views, "output": base + f"/{kind}-data"})
        training[kind] = train.remote(
            {
                "kind": kind,
                "dataset": base + f"/{kind}-data",
                "output": base + f"/{kind}",
                "width": 256,
                "k": 16,
                "epochs": 2,
                "batch_size": 32,
                "max_steps": 4,
                "checkpoint_steps": 1,
                "device": "cuda",
                "seed": 17,
            }
        )
        print(f"{kind} training/reload artifacts complete", flush=True)
    sample = spec.samples[0].model_dump()
    dictionary = f"{name}/sae"
    node = {
        "id": "test-feature",
        "layer": 30,
        "site": "resid_post",
        "dictionary": dictionary,
        "feature": 0,
        "position": 0,
        "operation": "add",
    }
    fidelity = worker.fidelity.remote(
        {
            "run_id": name + "-fidelity",
            "sample": sample,
            "dictionary": dictionary,
            "layer": 30,
            "site": "resid_post",
            "max_context": 512,
        }
    )
    attribution = worker.attribute.remote(
        {"run_id": name + "-attribute", "sample": sample, "nodes": [node], "max_context": 512}
    )
    graph = worker.graph.remote(
        {
            "run_id": name + "-graph",
            "sample": sample,
            "nodes": [node],
            "factors": [0, 1],
            "max_context": 512,
            "max_seconds": 900,
        }
    )
    assert graph["cleanup_identity_passed"]
    assert all(e["effect"] == 0 for e in graph["edges"] if e["factor"] == 0)
    result = {
        "status": "passed",
        "scope": "software integration on neutral examples, not scientific dictionary quality",
        "checkpoint": args.checkpoint,
        "run_id": name,
        "capture": capture,
        "evaluation": evaluations,
        "training": training,
        "fidelity": fidelity,
        "attribution": attribution,
        "graph": graph,
        "crosscoder_status": "covered by numerical tests; multi-checkpoint study not run",
    }
    atomic_json(Path(args.output), result)
    print(
        json.dumps(
            {
                "status": "passed",
                "checkpoint": args.checkpoint,
                "run_id": name,
                "completed_samples": evaluations["completed_samples"],
                "dictionary_kinds": list(training),
                "attribution_edges": len(attribution["edges"]),
                "causal_edges": len(graph["edges"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
