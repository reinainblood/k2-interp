"""Prepare immutable snapshots on CPU; never download weights on paid GPU startup."""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from .artifacts import atomic_json, file_hash
from .registry import CHECKPOINTS, REPO_ID, resolve


def prepare(
    checkpoint: str,
    root: str | Path,
    *,
    reuse: str | Path | None = None,
    verify_hashes: bool = True,
    commit=lambda: None,
) -> dict:
    """Snapshot all executable source + weights. Reuse is validated by full LFS hashes.

    A reuse directory is mounted read-only. Only matching weights are linked; model
    source/tokenizer/config always come from the requested immutable Hub commit.
    """
    lock = resolve(checkpoint)
    path = Path(root) / lock["revision"]
    path.mkdir(parents=True, exist_ok=True)
    receipt = path / "snapshot.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["weight_manifest_sha256"] != lock["weight_manifest_sha256"]:
            raise ValueError("Existing snapshot has a different weight manifest")
    weights = [
        (name, meta) for name, meta in lock["files"].items() if name.endswith(".safetensors")
    ]

    def obtain(item):
        name, meta = item
        target = path / name
        source = Path(reuse) / name if reuse else target
        if source.is_file() and source.stat().st_size == meta["size"]:
            observed = file_hash(source) if verify_hashes else None
            if observed != meta["sha256"]:
                if reuse:
                    raise ValueError(f"Reuse source does not match requested checkpoint: {name}")
            elif reuse and not target.exists():
                target.symlink_to(source)
        if not target.is_file():
            hf_hub_download(REPO_ID, name, revision=lock["revision"], local_dir=path)
        if target.stat().st_size != meta["size"]:
            raise ValueError(f"Wrong shard size: {name}")
        if verify_hashes and file_hash(target) != meta["sha256"]:
            raise ValueError(f"Wrong shard hash: {name}")
        return name

    # Bounded IO and per-shard persistence make a failed download resumable.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(obtain, weights):
            print(f"verified {name}", flush=True)
            commit()
    for name in lock["files"]:
        if name.endswith((".py", ".json", ".jinja")) and "/" not in name:
            hf_hub_download(REPO_ID, name, revision=lock["revision"], local_dir=path)
    lock["local_files"] = {
        name: file_hash(path / name)
        for name in lock["files"]
        if name.endswith((".py", ".json", ".jinja")) and "/" not in name
    }
    lock["weights_verified"] = verify_hashes
    lock["reuse_root"] = str(reuse) if reuse else None
    atomic_json(receipt, lock)
    commit()
    return lock


def ready(checkpoint: str, root: str | Path) -> tuple[Path, dict]:
    path = Path(root) / CHECKPOINTS[checkpoint]
    receipt = path / "snapshot.json"
    if not receipt.exists():
        raise FileNotFoundError("Prepare checkpoint on CPU before allocating a GPU worker")
    lock = json.loads(receipt.read_text())
    if lock["revision"] != CHECKPOINTS[checkpoint] or not lock["weights_verified"]:
        raise ValueError("Snapshot identity/verification failed")
    for name, meta in lock["files"].items():
        if name.endswith(".safetensors"):
            p = path / name
            if not p.is_file() or p.stat().st_size != meta["size"]:
                raise ValueError(f"Snapshot incomplete: {name}")
    for name, sha in lock["local_files"].items():
        if file_hash(path / name) != sha:
            raise ValueError(f"Snapshot source/config drift: {name}")
    return path, lock


def native_load(checkpoint: str, root: str | Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path, lock = ready(checkpoint, root)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("Full K2 reference worker requires eight CUDA GPUs")
    # Layers stay whole. This is sequential model parallelism, not TP/EP.
    # Avoid automatic CPU offload or silently loading a quantized fallback.
    memory = {i: int(torch.cuda.get_device_properties(i).total_memory * 0.88) for i in range(8)}
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=memory,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    if any(str(d) in ("cpu", "disk") for d in model.hf_device_map.values()):
        raise RuntimeError("Reference worker must fit entirely in GPU memory")
    model.requires_grad_(False)
    identity = {
        "repo_id": REPO_ID,
        "checkpoint": checkpoint,
        "revision": lock["revision"],
        "weight_manifest_sha256": lock["weight_manifest_sha256"],
        "source_files": lock["local_files"],
        "precision": "bfloat16",
        "attention": "sdpa",
        "parallelism": "native-whole-layer-model-parallel",
        "devices": {str(k): str(v) for k, v in model.hf_device_map.items()},
        "gpus": [torch.cuda.get_device_name(i) for i in range(8)],
        "code_revision": os.environ.get("K2_INTERP_CODE_REVISION", "unrecorded"),
    }
    return model, tokenizer, identity
