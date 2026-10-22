"""BatchTopK dictionaries for residual SAEs, skip-transcoders and checkpoint crosscoders.

Training uses batch-level top-k; evaluation uses a calibrated scalar threshold so
the same token receives the same sparse code regardless of evaluation batch size.
Crosscoders encode concatenated checkpoint views (acausal); they do not establish
a direction of causal influence between checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field
from safetensors.torch import load_file, save_file
from torch import nn

from .artifacts import atomic_json, file_hash
from .datasets import ActivationDataset
from .schema import Contract, digest


class TrainingConfig(Contract):
    kind: Literal["sae", "skip_transcoder", "crosscoder"]
    dataset: str
    output: str
    width: int = Field(default=16384, ge=2)
    k: int = Field(default=64, ge=1)
    epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=512, ge=1)
    learning_rate: float = Field(default=0.0003, gt=0)
    seed: int = Field(default=0, ge=0)
    device: str = "cuda"
    checkpoint_steps: int = Field(default=100, ge=1)
    max_steps: int = Field(default=10000, ge=1)


class SparseDictionary(nn.Module):
    def __init__(
        self, input_dims: list[int], target_dims: list[int], width: int, k: int, kind: str
    ):
        super().__init__()
        if k > width:
            raise ValueError("k exceeds dictionary width")
        if kind == "sae" and input_dims != target_dims:
            raise ValueError("SAE input/target dimensions must match")
        if kind == "crosscoder" and (len(input_dims) < 2 or input_dims != target_dims):
            raise ValueError("Crosscoder requires aligned input/target checkpoint views")
        self.input_dims, self.target_dims, self.width, self.k, self.kind = (
            input_dims,
            target_dims,
            width,
            k,
            kind,
        )
        self.encoder = nn.Linear(sum(input_dims), width)
        self.decoder = nn.Linear(width, sum(target_dims))
        self.skip = (
            nn.Linear(sum(input_dims), sum(target_dims), bias=False)
            if kind == "skip_transcoder"
            else None
        )
        if self.skip is not None:
            nn.init.zeros_(self.skip.weight)
        self.register_buffer("input_mean", torch.zeros(sum(input_dims)))
        self.register_buffer("target_mean", torch.zeros(sum(target_dims)))
        self.register_buffer("input_scale", torch.ones(sum(input_dims)))
        self.register_buffer("target_scale", torch.ones(sum(target_dims)))
        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("feature_counts", torch.zeros(width, dtype=torch.int64))
        self.register_buffer("training_rows", torch.tensor(0, dtype=torch.int64))
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def encode(self, x: torch.Tensor, *, batch_topk: bool = False):
        pre = self.encoder((x - self.input_mean) / self.input_scale).relu()
        if batch_topk:
            values, indices = pre.flatten().topk(min(self.k * len(x), pre.numel()), sorted=False)
            acts = torch.zeros_like(pre).flatten().scatter(0, indices, values).reshape_as(pre)
            with torch.no_grad():
                cutoff = values.min()
                self.threshold.lerp_(cutoff, 0.1)
                self.feature_counts.add_((acts > 0).sum(0))
                self.training_rows.add_(len(x))
            return acts
        return pre * (pre > self.threshold)

    def decode(self, acts: torch.Tensor, x: torch.Tensor):
        value = self.decoder(acts)
        if self.skip is not None:
            value = value + self.skip((x - self.input_mean) / self.input_scale)
        return value * self.target_scale + self.target_mean

    def forward(self, x: torch.Tensor, *, batch_topk: bool = False):
        acts = self.encode(x, batch_topk=batch_topk)
        return self.decode(acts, x), acts

    def metadata(self):
        return {
            "input_dims": self.input_dims,
            "target_dims": self.target_dims,
            "width": self.width,
            "k": self.k,
            "kind": self.kind,
        }


@torch.no_grad()
def fit_normalization(model: SparseDictionary, dataset: ActivationDataset, batch_size: int):
    count = 0
    sums = [
        torch.zeros_like(model.input_mean, device="cpu", dtype=torch.float64),
        torch.zeros_like(model.target_mean, device="cpu", dtype=torch.float64),
    ]
    squares = [torch.zeros_like(s) for s in sums]
    for x, y in dataset.batches("train", batch_size):
        count += len(x)
        for j, t in enumerate((x, y)):
            sums[j] += t.double().sum(0)
            squares[j] += t.double().square().sum(0)
    for j, (dims, mean, scale) in enumerate(
        (
            (model.input_dims, model.input_mean, model.input_scale),
            (model.target_dims, model.target_mean, model.target_scale),
        )
    ):
        mu = sums[j] / count
        var = (squares[j] / count - mu.square()).clamp_min(0)
        offset = 0
        for width in dims:
            mean[offset : offset + width] = mu[offset : offset + width].to(mean)
            scale[offset : offset + width] = (
                var[offset : offset + width].mean().sqrt().clamp_min(1e-5).to(scale)
            )
            offset += width


@torch.no_grad()
def evaluate(
    model: SparseDictionary, dataset: ActivationDataset, split="validation", batch_size=512
):
    errors, variance, l0, rows = 0.0, 0.0, 0.0, 0
    counts = torch.zeros(model.width, device=model.threshold.device)
    for x, y in dataset.batches(split, batch_size):
        x, y = x.to(model.threshold.device), y.to(model.threshold.device)
        pred, acts = model(x)
        errors += float(((pred - y) / model.target_scale).square().sum())
        variance += float(((y - model.target_mean) / model.target_scale).square().sum())
        l0 += float((acts > 0).sum())
        counts += (acts > 0).sum(0)
        rows += len(x)
    return {
        "split": split,
        "rows": rows,
        "normalized_mse": errors / (rows * sum(model.target_dims)),
        "variance_explained": 1 - errors / max(variance, 1e-30),
        "mean_l0": l0 / rows,
        "inactive_fraction": float((counts == 0).float().mean()),
        "threshold": float(model.threshold),
        "behavioral_fidelity": "not_measured_by_reconstruction; run model interventions",
    }


def train_from_config(config: dict, *, root: str | Path = ".", commit=lambda: None):
    cfg = TrainingConfig.model_validate(config)
    root = Path(root).resolve()
    dataset_path, output = (root / cfg.dataset).resolve(), (root / cfg.output).resolve()
    if root != Path(".").resolve() and (
        not dataset_path.is_relative_to(root) or not output.is_relative_to(root)
    ):
        raise ValueError("Training paths must remain within artifact root")
    ds = ActivationDataset(dataset_path)
    # Fail before a training job if held-out data are absent.
    next(ds.batches("validation", cfg.batch_size))
    view_pairs = [(v["run"], v["site"]) for v in ds.manifest["views"]]
    if cfg.kind == "sae":
        if len(ds.input_views) != 1 or len(ds.target_views) != 1 or view_pairs[0] != view_pairs[1]:
            raise ValueError("An SAE must reconstruct the identical activation view")
    if cfg.kind == "crosscoder":
        inputs = [(v["run"], v["site"]) for v in ds.manifest["views"] if v["role"] == "input"]
        targets = [(v["run"], v["site"]) for v in ds.manifest["views"] if v["role"] == "target"]
        if inputs != targets or len({v[0] for v in inputs}) < 2:
            raise ValueError(
                "Checkpoint crosscoder must reconstruct the same ordered checkpoint views"
            )
    torch.manual_seed(cfg.seed)
    model = SparseDictionary(ds.input_dims, ds.target_dims, cfg.width, cfg.k, cfg.kind).to(
        cfg.device
    )
    fit_normalization(model, ds, cfg.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    identity = digest({"config": config, "dataset": ds.manifest["dataset_sha256"]})
    if (output / "dictionary.json").exists():
        previous = json.loads((output / "dictionary.json").read_text())
        if previous.get("training_identity") != identity:
            raise ValueError("Completed dictionary belongs to another training configuration")
        load_dictionary(output)
        return previous
    checkpoint = output / "training-state.pt"
    step, start_epoch, start_batch = 0, 0, 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=cfg.device, weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Existing training checkpoint belongs to a different experiment")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, start_epoch, start_batch = state["step"], state["epoch"], state["next_batch"]
    elif (output / "dictionary.json").exists():
        raise FileExistsError("Completed dictionary already exists")

    def save_state(epoch, batch):
        temp = checkpoint.with_suffix(".tmp")
        torch.save(
            {
                "identity": identity,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "epoch": epoch,
                "next_batch": batch,
            },
            temp,
        )
        temp.replace(checkpoint)
        commit()

    next_epoch, next_batch = start_epoch, start_batch
    for epoch in range(start_epoch, cfg.epochs):
        for batch, (x, y) in enumerate(
            ds.batches("train", cfg.batch_size, seed=cfg.seed + epoch, shuffle=True)
        ):
            if epoch == start_epoch and batch < start_batch:
                continue
            if step >= cfg.max_steps:
                break
            x, y = x.to(cfg.device), y.to(cfg.device)
            pred, acts = model(x, batch_topk=True)
            loss = ((pred - y) / model.target_scale).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Dictionary loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.normalize_decoder()
            step += 1
            next_epoch, next_batch = epoch, batch + 1
            if step % cfg.checkpoint_steps == 0:
                save_state(epoch, batch + 1)
        if step >= cfg.max_steps:
            break
        next_epoch, next_batch = epoch + 1, 0
        save_state(epoch + 1, 0)
    save_state(next_epoch, next_batch)
    # Calibrate inference threshold on a bounded training-only reservoir, not held-out labels.
    with torch.no_grad():
        candidates = []
        calibration_rows = 0
        for i, (x, _) in enumerate(
            ds.batches("train", cfg.batch_size, seed=cfg.seed, shuffle=True)
        ):
            x = x.to(cfg.device)
            calibration_rows += len(x)
            pre = model.encoder((x - model.input_mean) / model.input_scale).relu()
            candidates.append(pre.flatten().topk(min(pre.numel(), cfg.k * len(x) * 2)).values.cpu())
            if i >= 15:
                break
        reservoir = torch.cat(candidates)
        wanted = min(cfg.k * calibration_rows, reservoir.numel())
        cutoff = reservoir.kthvalue(max(1, reservoir.numel() - wanted + 1)).values
        model.threshold.copy_(cutoff.to(cfg.device))
    metrics = evaluate(model, ds, batch_size=cfg.batch_size)
    state_path = output / "dictionary.safetensors"
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(state_path)
    )
    meta = {
        "schema": "k2-interp/dictionary/1",
        **model.metadata(),
        "training": cfg.model_dump(),
        "training_identity": identity,
        "dataset_sha256": ds.manifest["dataset_sha256"],
        "views": ds.manifest["views"],
        "steps": step,
        "metrics": metrics,
        "weights_sha256": file_hash(state_path),
        "crosscoder_semantics": "joint acausal encoding of aligned checkpoint activations"
        if cfg.kind == "crosscoder"
        else None,
    }
    atomic_json(output / "dictionary.json", meta)
    commit()
    return meta


def load_dictionary(path: str | Path, device="cpu"):
    path = Path(path)
    meta = json.loads((path / "dictionary.json").read_text())
    if file_hash(path / "dictionary.safetensors") != meta["weights_sha256"]:
        raise ValueError("Dictionary hash mismatch")
    model = SparseDictionary(
        **{k: meta[k] for k in ("input_dims", "target_dims", "width", "k", "kind")}
    )
    model.load_state_dict(load_file(path / "dictionary.safetensors"))
    return model.to(device).eval(), meta
