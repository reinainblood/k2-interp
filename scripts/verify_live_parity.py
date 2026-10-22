"""Compare full-weight logits with and without live capture's expert-norm hooks."""

import json
import uuid
from pathlib import Path

import modal

from k2_interp.artifacts import atomic_json


def main():
    worker = modal.Cls.from_name("k2-interp-sft-3-1500", "NativeWorker", environment_name="FDR")()
    volume = modal.Volume.from_name("k2-interp-artifacts", environment_name="FDR")
    suffix = uuid.uuid4().hex[:12]
    results = []
    for capture in (False, True):
        spec = {
            "run_id": f"live-parity-{suffix}-{'captured' if capture else 'baseline'}",
            "checkpoint": "sft_3_1500",
            "samples": [
                {"id": "control", "text": "The capital of France is", "continuation": " Paris."}
            ],
            "capture": {
                "router": capture,
                "layers": [10, 30, 50] if capture else [],
                "expert_output_norms": capture,
            },
            "generation": {"max_new_tokens": 2, "temperature": 0, "seed": 19},
            "max_context": 128,
            "max_seconds": 300,
        }
        receipt = worker.run.remote(spec)
        path = f"sft_3_1500/{spec['run_id']}/{receipt['samples']['control']['attempt']}/result.json"
        result = json.loads(b"".join(volume.read_file(path)))
        results.append({"receipt": receipt, "result": result})
    delta = max(
        abs(a - b)
        for a, b in zip(
            results[0]["result"]["output_logprobs"],
            results[1]["result"]["output_logprobs"],
            strict=True,
        )
    )
    assert delta == 0, f"Capture changed logits: {delta}"
    atomic_json(
        Path(".private/workbench-parity.json"),
        {"status": "passed", "max_logprob_delta": delta, "runs": results},
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "max_logprob_delta": delta,
                "run_ids": [x["receipt"]["run_id"] for x in results],
            }
        )
    )


if __name__ == "__main__":
    main()
