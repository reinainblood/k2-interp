"""Checkpoint identity and immutable Hub resolution; no GPU allocation."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi

from .schema import digest

REPO_ID = "IFM/K2-Horizon-375B-A23B"
# Captured from the official Hub refs. resolve() verifies these against current metadata.
CHECKPOINTS = {
    # Last pretraining phase: the form-vs-content baseline for self-report studies.
    # Same 61-shard geometry as the post-training releases; verified 2026-10-22.
    "pretrain_ph2_240000": "9640a9fbd5dab74c8778d3690bf6c92fd5ee02e8",
    "mid_3_3500": "a02ec93a1c6d0f3766e8262a48ee8a846a851574",
    "mid_4_6000": "7d22cc0ac26a91003d19482efa71070244239b11",
    "rl_knowledge_work": "becd3c065be2f373bb3529c84637bf2b4742080a",
    "rl_if": "f834a8871e685392fff8f2147a3d63671f1e0f68",
    "rl_search": "c0de5c46bb3c9e96a464c68475ae423dd9fa99c4",
    "rl_tool-use": "1c3378f283358951af43c85ffb3888b3e9926592",
    "rl_reasoning": "de63d895d8f9233429fd2359f5f66ec5d4060a37",
    "rl_merged": "a6175eb52940f3811e3249e7fa634208c98fe698",
    "sft_1_2400": "6cfac56ebd59375196a1dd41da1decb439f24220",
    "sft_2_6000": "380ba6da151399f20192ae3f76b4aebf409dd059",
    "sft_3_1500": "cfd157f11804b290c47521311874fa24870c5b1e",
}
SPINE = ("mid_3_3500", "mid_4_6000", "rl_merged", "sft_1_2400", "sft_2_6000", "sft_3_1500")


def resolve(checkpoint: str, *, api=None) -> dict:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint {checkpoint!r}; choose from {list(CHECKPOINTS)}")
    revision = CHECKPOINTS[checkpoint]
    info = (api or HfApi()).model_info(REPO_ID, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("Hub returned a different immutable revision")
    files = {}
    for item in info.siblings:
        lfs = item.lfs
        files[item.rfilename] = {
            "size": item.size,
            "sha256": getattr(lfs, "sha256", None) if lfs else None,
            "git_blob": item.blob_id,
        }
    weights = {k: v for k, v in files.items() if k.endswith(".safetensors")}
    if len(weights) != 61 or any(not f["sha256"] for f in weights.values()):
        raise ValueError("Expected 61 hash-addressed K2 weight shards")
    return {
        "repo_id": REPO_ID,
        "checkpoint": checkpoint,
        "revision": revision,
        "files": files,
        "weight_manifest_sha256": digest(weights),
    }


def read_lock(path: str | Path) -> dict:
    lock = json.loads(Path(path).read_text())
    if lock["revision"] != CHECKPOINTS.get(lock["checkpoint"]):
        raise ValueError("Checkpoint lock is not in the pinned registry")
    return lock
