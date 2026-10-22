"""Descriptive router and residual localization. Causal evidence lives in causal.py."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from .artifacts import atomic_json, file_hash
from .datasets import site_rows


def _bootstrap(values: torch.Tensor, groups: list[str], seed=0, repeats=1000):
    # Resample prompt families, retaining all pairs within each selected family.
    keys = sorted(set(groups))
    by_group = [values[torch.tensor([g == key for g in groups])].float() for key in keys]
    generator = torch.Generator().manual_seed(seed)
    means = []
    for _ in range(repeats):
        chosen = torch.randint(len(keys), (len(keys),), generator=generator)
        means.append(torch.cat([by_group[int(i)] for i in chosen]).mean())
    lo, hi = torch.stack(means).quantile(torch.tensor([0.025, 0.975])).tolist()
    return [lo, hi]


def localize(run: str | Path, pairs: list[dict], sites: list[str], output: str | Path, seed=0):
    """pairs specify clean/control ids and explicit positions; no token alignment guessing."""
    if not pairs:
        raise ValueError("At least one matched pair is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    directions, metrics = {}, []
    for site in sites:
        rows = site_rows(run, site)
        deltas = []
        groups = []
        for pair in pairs:
            clean = rows[pair["clean_id"]]["rows"][pair["clean_position"]]
            control = rows[pair["control_id"]]["rows"][pair["control_position"]]
            deltas.append(clean - control)
            groups.append(pair["group"])
        matrix = torch.stack(deltas)
        norms = matrix.norm(dim=-1)
        directions[site] = matrix.mean(0)
        metrics.append(
            {
                "site": site,
                "pairs": len(pairs),
                "groups": len(set(groups)),
                "mean_pair_l2": float(norms.mean()),
                "group_bootstrap_95ci": _bootstrap(norms, groups, seed),
                "mean_direction_l2": float(matrix.mean(0).norm()),
                "interpretation": "descriptive localization, not causal importance",
            }
        )
    save_file(directions, str(output / "directions.safetensors"))
    result = {
        "schema": "k2-interp/localization/1",
        "pairs": pairs,
        "metrics": metrics,
        "seed": seed,
        "source_manifest_sha256": file_hash(Path(run) / "manifest.json"),
        "directions_sha256": file_hash(output / "directions.safetensors"),
    }
    atomic_json(output / "localization.json", result)
    return result


def router_comparison(run_a: str | Path, run_b: str | Path, layers: list[int], output: str | Path):
    result = []
    for layer in layers:
        prefix = f"layers.{layer}."
        a, b = (
            site_rows(run_a, prefix + "router_scores"),
            site_rows(run_b, prefix + "router_scores"),
        )
        ai, bi = site_rows(run_a, prefix + "expert_ids"), site_rows(run_b, prefix + "expert_ids")
        if set(a) != set(b):
            raise ValueError("Router comparison sample sets differ")
        for sample in sorted(a):
            if a[sample]["result"]["all_token_ids"] != b[sample]["result"]["all_token_ids"]:
                raise ValueError(
                    "Cross-checkpoint router comparison requires identical token sequences"
                )
            if set(a[sample]["rows"]) != set(b[sample]["rows"]):
                raise ValueError("Router positions differ")
            for pos in sorted(a[sample]["rows"]):
                p, q = a[sample]["rows"][pos], b[sample]["rows"][pos]
                p, q = p / p.sum(), q / q.sum()
                m = (p + q) / 2
                js = 0.5 * (
                    (p * (p.clamp_min(1e-30).log() - m.log())).sum()
                    + (q * (q.clamp_min(1e-30).log() - m.log())).sum()
                )
                ids_a = set(ai[sample]["rows"][pos].int().tolist())
                ids_b = set(bi[sample]["rows"][pos].int().tolist())
                result.append(
                    {
                        "sample_id": sample,
                        "layer": layer,
                        "position": pos,
                        "js_divergence": float(js),
                        "topk_jaccard": len(ids_a & ids_b) / len(ids_a | ids_b),
                        "route_changed": ids_a != ids_b,
                    }
                )
    report = {
        "schema": "k2-interp/router-comparison/1",
        "records": result,
        "normalization": "sigmoid scores explicitly normalized to categorical distributions for JS",
        "sources": [file_hash(Path(r) / "manifest.json") for r in (run_a, run_b)],
    }
    atomic_json(Path(output), report)
    return report


def feature_report(
    dictionary_path: str | Path, dataset_path: str | Path, output: str | Path, top_n=10
):
    from .datasets import ActivationDataset
    from .dictionary import load_dictionary

    dictionary, meta = load_dictionary(dictionary_path)
    dataset = ActivationDataset(dataset_path)
    if dataset.input_dims != dictionary.input_dims:
        raise ValueError("Dataset and dictionary inputs differ")
    counts = torch.zeros(dictionary.width, dtype=torch.int64)
    maxima = torch.zeros(dictionary.width)
    with torch.no_grad():
        for x, _ in dataset.batches("validation", 512):
            acts = dictionary.encode(x)
            counts += (acts > 0).sum(0)
            maxima = torch.maximum(maxima, acts.amax(0))
    best = counts.topk(min(top_n, len(counts))).indices.tolist()
    result = {
        "schema": "k2-interp/features/1",
        "dictionary_sha256": meta["weights_sha256"],
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "features": [
            {
                "id": i,
                "validation_active_rows": int(counts[i]),
                "max_activation": float(maxima[i]),
                "label": "unlabeled",
            }
            for i in best
        ],
    }
    atomic_json(Path(output), result)
    return result
