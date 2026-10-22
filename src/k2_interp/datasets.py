"""Aligned activation datasets with explicit checkpoint, token, split, and group identity."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .artifacts import atomic_json, file_hash, iter_records
from .schema import SCHEMA_VERSION, digest


def site_rows(run: str | Path, site: str, selected_sample: str | None = None) -> dict:
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    rows = {}
    for sample_id, meta, tensors in iter_records(run, selected_sample, fields={site}):
        if site not in tensors:
            continue
        sample = rows.setdefault(sample_id, {"rows": {}, "manifest": manifest})
        for i, position in enumerate(meta["positions"]):
            if position in sample["rows"]:
                raise ValueError(f"Repeated position {position} in {sample_id}/{site}")
            sample["rows"][position] = tensors[site][i].float()
    for sample_id, item in rows.items():
        receipt = json.loads((run / "samples" / sample_id / "complete.json").read_text())
        result_path = run / receipt["attempt"] / "result.json"
        if receipt["files"].get(str(result_path.relative_to(run))) != file_hash(result_path):
            raise ValueError("Unverified sample result")
        item["result"] = json.loads(result_path.read_text())
    return rows


def build_dataset(views: list[dict], output: str | Path) -> dict:
    """views = [{name, run, site, role: input|target}]. Align by sample and position.

    A crosscoder's input and target views can refer to the same checkpoint/site.
    Repeated views are intentional. Outputs from divergent generations are rejected.
    """
    if not views or len({v["name"] for v in views}) != len(views):
        raise ValueError("Dataset view names must be unique")
    for view in views:
        if view["role"] not in ("input", "target"):
            raise ValueError("view role must be input or target")
    if {v["role"] for v in views} != {"input", "target"}:
        raise ValueError("Dataset requires input and target views")
    output = Path(output)
    if (output / "dataset.json").exists():
        raise FileExistsError("Dataset already exists; use a new immutable output directory")
    output.mkdir(parents=True, exist_ok=True)
    sample_sets = [
        {p.parent.name for p in (Path(v["run"]) / "samples").glob("*/complete.json")} for v in views
    ]
    if any(s != sample_sets[0] for s in sample_sets):
        raise ValueError("View sample sets differ; explicitly select aligned samples first")
    shards = []
    dimensions = {}
    groups = {}
    for sample_id in sorted(sample_sets[0]):
        # Bound peak RAM by a single sample, not the entire activation corpus.
        loaded = [site_rows(v["run"], v["site"], sample_id) for v in views]
        if any(sample_id not in rows for rows in loaded):
            raise ValueError(f"Requested site missing from sample {sample_id}")
        results = [r[sample_id]["result"] for r in loaded]
        if any(r["all_token_ids"] != results[0]["all_token_ids"] for r in results):
            raise ValueError(
                f"Token sequences differ across views: {sample_id}; use teacher forcing"
            )
        position_sets = [set(r[sample_id]["rows"]) for r in loaded]
        if any(p != position_sets[0] for p in position_sets):
            raise ValueError("Capture positions differ between views")
        positions = sorted(position_sets[0])
        if not positions:
            continue
        sample = results[0]["sample"]
        group, split = sample["template_group"], sample["split"]
        if group in groups and groups[group] != split:
            raise ValueError(f"Template group leaks across splits: {group}")
        groups[group] = split
        if any(
            r["sample"]["split"] != split or r["sample"]["template_group"] != group for r in results
        ):
            raise ValueError("Split/group metadata differs across views")
        if any(r["sample"]["labels"] != sample["labels"] for r in results):
            raise ValueError("Sample labels differ across checkpoint views")
        tensors = {
            v["name"]: torch.stack([r[sample_id]["rows"][p] for p in positions])
            for v, r in zip(views, loaded, strict=True)
        }
        for name, tensor in tensors.items():
            if tensor.ndim != 2:
                raise ValueError("Dictionary datasets require token-by-feature matrices")
            if name in dimensions and dimensions[name] != tensor.shape[1]:
                raise ValueError("Feature dimensions changed between samples")
            dimensions[name] = tensor.shape[1]
        tensors["positions"] = torch.tensor(positions)
        shard = output / f"{len(shards):06d}.safetensors"
        save_file(tensors, str(shard))
        shards.append(
            {
                "file": shard.name,
                "sha256": file_hash(shard),
                "sample_id": sample_id,
                "positions": positions,
                "token_ids_sha256": digest(results[0]["all_token_ids"]),
                "rows": len(positions),
                "split": split,
                "template_group": group,
                "labels": sample["labels"],
            }
        )
    if not shards:
        raise ValueError("No aligned activation rows")
    provenance_views = [
        {
            **v,
            "source_identity": json.loads((Path(v["run"]) / "manifest.json").read_text())[
                "identity"
            ],
        }
        for v in views
    ]
    manifest = {
        "schema": SCHEMA_VERSION,
        "views": provenance_views,
        "dimensions": dimensions,
        "shards": shards,
        "source_requests": [
            json.loads((Path(v["run"]) / "manifest.json").read_text())["request_sha256"]
            for v in views
        ],
    }
    manifest["dataset_sha256"] = digest(manifest)
    atomic_json(output / "dataset.json", manifest)
    return manifest


class ActivationDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "dataset.json").read_text())
        if (
            digest({k: v for k, v in self.manifest.items() if k != "dataset_sha256"})
            != self.manifest["dataset_sha256"]
        ):
            raise ValueError("Dataset manifest hash mismatch")
        self.input_views = [v["name"] for v in self.manifest["views"] if v["role"] == "input"]
        self.target_views = [v["name"] for v in self.manifest["views"] if v["role"] == "target"]
        self.input_dims = [self.manifest["dimensions"][k] for k in self.input_views]
        self.target_dims = [self.manifest["dimensions"][k] for k in self.target_views]

    def batches(self, split: str, batch_size: int, *, seed: int = 0, shuffle: bool = False):
        generator = torch.Generator().manual_seed(seed)
        shards = [s for s in self.manifest["shards"] if s["split"] == split]
        if not shards:
            raise ValueError(f"No {split} shards; provide independent template-group splits")
        if shuffle:
            shards = [shards[i] for i in torch.randperm(len(shards), generator=generator)]
        pending_x, pending_y = [], []
        pending_rows = 0
        buffer_rows = batch_size * 8
        for shard in shards:
            path = self.path / shard["file"]
            if file_hash(path) != shard["sha256"]:
                raise ValueError("Dataset shard hash mismatch")
            tensors = load_file(path)
            x = torch.cat([tensors[v] for v in self.input_views], dim=-1).float()
            y = torch.cat([tensors[v] for v in self.target_views], dim=-1).float()
            for start in range(0, len(x), buffer_rows):
                pending_x.append(x[start : start + buffer_rows])
                pending_y.append(y[start : start + buffer_rows])
                pending_rows += len(pending_x[-1])
                if pending_rows < buffer_rows:
                    continue
                bx, by = torch.cat(pending_x), torch.cat(pending_y)
                order = (
                    torch.randperm(len(bx), generator=generator)
                    if shuffle
                    else torch.arange(len(bx))
                )
                full = len(order) // batch_size * batch_size
                for indices in order[:full].split(batch_size):
                    yield bx[indices], by[indices]
                remaining = order[full:]
                pending_x, pending_y = [bx[remaining]], [by[remaining]]
                pending_rows = len(remaining)
        if pending_rows:
            bx, by = torch.cat(pending_x), torch.cat(pending_y)
            order = (
                torch.randperm(len(bx), generator=generator) if shuffle else torch.arange(len(bx))
            )
            for indices in order.split(batch_size):
                yield bx[indices], by[indices]
