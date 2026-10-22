"""Verify every pinned checkpoint's public metadata without downloading model weights."""

import argparse
import concurrent.futures
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from k2_interp.artifacts import atomic_json, file_hash
from k2_interp.registry import CHECKPOINTS, REPO_ID, resolve


def verify(checkpoint):
    lock = resolve(checkpoint)
    path = hf_hub_download(REPO_ID, "config.json", revision=lock["revision"])
    cfg = json.loads(Path(path).read_text())
    expected = {
        "num_hidden_layers": 61,
        "hidden_size": 6144,
        "num_experts": 192,
        "num_experts_per_tok": 8,
    }
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Unsupported geometry for {checkpoint}")
    weights = [m for p, m in lock["files"].items() if p.endswith(".safetensors")]
    return {
        "checkpoint": checkpoint,
        "revision": lock["revision"],
        "weight_shards": len(weights),
        "weight_bytes": sum(w["size"] for w in weights),
        "config_sha256": file_hash(path),
        "weight_manifest_sha256": lock["weight_manifest_sha256"],
        "model_code_blob": lock["files"]["modeling_k2_horizon.py"]["git_blob"],
        "tokenizer_sha256": lock["files"]["tokenizer.json"]["sha256"],
        "geometry": expected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".private/registry-verification.json")
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(verify, CHECKPOINTS))
    result = {
        "status": "passed",
        "scope": "Hub metadata and config only; no model-weight downloads",
        "checkpoint_count": len(rows),
        "model_code_variants": len({r["model_code_blob"] for r in rows}),
        "tokenizer_variants": len({r["tokenizer_sha256"] for r in rows}),
        "checkpoints": rows,
    }
    atomic_json(Path(args.output), result)
    print(json.dumps({k: v for k, v in result.items() if k != "checkpoints"}, indent=2))


if __name__ == "__main__":
    main()
