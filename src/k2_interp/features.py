"""Flatten router traces into per-token feature rows.

A ``b1`` replay stores, per forward, one safetensors shard with
``layers.<L>.<field>`` tensors for every MoE layer. Analysis wants the opposite
shape: one row per (sample, position, layer) with the scalar routing summaries
and the selected experts, plus the sample's behavioral labels. This module
produces that table as JSONL and as a compact ``.npz``, verifying shard hashes
through :func:`k2_interp.artifacts.iter_records` on the way.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import iter_records

SCALARS = ("router_entropy", "routing_margin")
VECTORS = ("expert_ids", "expert_weights")


def _sample_labels(run: Path) -> dict[str, dict]:
    manifest = json.loads((run / "manifest.json").read_text())
    spec = manifest.get("spec") or {}
    return {s["id"]: s.get("labels", {}) for s in spec.get("samples", [])}


def router_rows(run: str | Path, sample_id: str | None = None):
    """Yield one dict per (sample, position, layer) from a router capture."""
    run = Path(run)
    labels = _sample_labels(run)
    wanted = {f"layers.{layer}.{f}" for layer in range(0, 64) for f in SCALARS + VECTORS}
    for sid, meta, tensors in iter_records(run, sample_id, fields=wanted):
        layer = meta["layer"]
        prefix = f"layers.{layer}."
        present = {k[len(prefix) :]: v for k, v in tensors.items() if k.startswith(prefix)}
        if not present:
            continue
        positions = meta["positions"]
        token_ids = meta.get("token_ids") or [None] * len(positions)
        for i, position in enumerate(positions):
            row = {
                "sample": sid,
                "position": int(position),
                "token_id": None if token_ids[i] is None else int(token_ids[i]),
                "layer": int(layer),
                "step": meta.get("step"),
                "phase": meta.get("phase"),
            }
            for name in SCALARS:
                if name in present:
                    row[name] = float(present[name][i].reshape(-1)[0])
            for name in VECTORS:
                if name in present:
                    values = present[name][i].reshape(-1)
                    row[name] = (
                        [int(x) for x in values]
                        if name == "expert_ids"
                        else [float(x) for x in values]
                    )
            row["labels"] = labels.get(sid, {})
            yield row


def write_router_features(run: str | Path, output: str | Path) -> dict:
    """Write ``<output>.jsonl`` and ``<output>.npz``; return a small summary."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl = output.with_suffix(".jsonl")
    count = 0
    samples: dict[str, int] = {}
    layers: set[int] = set()
    cols: dict[str, list] = {
        "sample_index": [],
        "position": [],
        "layer": [],
        "router_entropy": [],
        "routing_margin": [],
        "expert_ids": [],
        "expert_weights": [],
    }
    with jsonl.open("w") as f:
        for row in router_rows(run):
            f.write(json.dumps(row) + "\n")
            count += 1
            layers.add(row["layer"])
            index = samples.setdefault(row["sample"], len(samples))
            cols["sample_index"].append(index)
            cols["position"].append(row["position"])
            cols["layer"].append(row["layer"])
            cols["router_entropy"].append(row.get("router_entropy", np.nan))
            cols["routing_margin"].append(row.get("routing_margin", np.nan))
            cols["expert_ids"].append(row.get("expert_ids") or [])
            cols["expert_weights"].append(row.get("expert_weights") or [])
    if count == 0:
        raise ValueError(f"No router records found under {run}")
    width = max((len(v) for v in cols["expert_ids"]), default=0)
    ids = np.full((count, width), -1, dtype=np.int16)
    weights = np.full((count, width), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(zip(cols["expert_ids"], cols["expert_weights"], strict=True)):
        ids[i, : len(a)] = a
        weights[i, : len(b)] = b
    np.savez_compressed(
        output.with_suffix(".npz"),
        sample_index=np.asarray(cols["sample_index"], dtype=np.int32),
        sample_names=np.asarray(list(samples), dtype=object),
        position=np.asarray(cols["position"], dtype=np.int32),
        layer=np.asarray(cols["layer"], dtype=np.int16),
        router_entropy=np.asarray(cols["router_entropy"], dtype=np.float32),
        routing_margin=np.asarray(cols["routing_margin"], dtype=np.float32),
        expert_ids=ids,
        expert_weights=weights,
    )
    return {
        "rows": count,
        "samples": len(samples),
        "layers": sorted(layers),
        "jsonl": str(jsonl),
        "npz": str(output.with_suffix(".npz")),
    }


def window_summary(run: str | Path, positions: dict[str, list[int]]) -> dict:
    """Mean entropy and margin per layer inside named position windows of one sample run.

    ``positions`` maps a window name to absolute positions. Returns
    ``{window: {layer: {"entropy": mean, "margin": mean, "n": count}}}``.
    """
    sets = {name: set(ps) for name, ps in positions.items()}
    acc: dict[str, dict[int, list]] = {name: {} for name in sets}
    for row in router_rows(run):
        for name, ps in sets.items():
            if row["position"] in ps:
                bucket = acc[name].setdefault(row["layer"], [[], []])
                if "router_entropy" in row:
                    bucket[0].append(row["router_entropy"])
                if "routing_margin" in row:
                    bucket[1].append(row["routing_margin"])
    return {
        name: {
            layer: {
                "entropy": float(np.mean(e)) if e else None,
                "margin": float(np.mean(m)) if m else None,
                "n": max(len(e), len(m)),
            }
            for layer, (e, m) in sorted(layers.items())
        }
        for name, layers in acc.items()
    }
