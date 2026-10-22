"""Atomic, checksummed trace shards. Completion is a durable receipt, never a log line."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .schema import SCHEMA_VERSION, digest, identifier


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class TraceWriter:
    def __init__(
        self, root: str | Path, spec: dict, identity: dict, commit: Callable = lambda: None
    ):
        self.root = Path(root) / identifier(spec["run_id"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.commit = commit
        self.identity = identity
        self.spec_hash = digest({"spec": spec, "identity": identity})
        manifest = self.root / "manifest.json"
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if old["request_sha256"] != self.spec_hash:
                raise ValueError("Run id already exists with a different request/model identity")
        else:
            atomic_json(
                manifest,
                {
                    "schema": SCHEMA_VERSION,
                    "request_sha256": self.spec_hash,
                    "spec": spec,
                    "identity": identity,
                    "created_unix": time.time(),
                },
            )
            self.commit()

    def completed(self, sample_id: str) -> dict | None:
        path = self.root / "samples" / identifier(sample_id) / "complete.json"
        if not path.exists():
            return None
        receipt = json.loads(path.read_text())
        for relative, expected in receipt["files"].items():
            target = self.root / relative
            if not target.is_file() or file_hash(target) != expected:
                raise ValueError(f"Corrupt completed sample artifact: {relative}")
        return receipt

    def sample(self, sample_id: str, max_bytes: int) -> SampleWriter:
        return SampleWriter(self, sample_id, max_bytes)


class SampleWriter:
    def __init__(self, parent: TraceWriter, sample_id: str, max_bytes: int):
        self.parent = parent
        self.base = parent.root / "samples" / identifier(sample_id)
        self.path = self.base / ("attempt-" + uuid.uuid4().hex)
        self.path.mkdir(parents=True)
        self.max_bytes = max_bytes
        self.bytes = 0
        self.files: dict[str, str] = {}
        self.shards = 0
        self.generation_progress = None

    def record(self, step: int, metadata: dict, tensors: dict[str, torch.Tensor]) -> None:
        # Copy before the model reuses storage. Shape/dtype are preserved, including BF16.
        tensors = {k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()}
        added = sum(x.numel() * x.element_size() for x in tensors.values())
        if self.bytes + added > self.max_bytes:
            raise ValueError("Capture byte budget exceeded; select fewer positions/layers")
        stem = f"step-{step:06d}-shard-{self.shards:06d}"
        self.shards += 1
        tensor_path = self.path / f"{stem}.safetensors"
        tmp = tensor_path.with_suffix(".tmp")
        save_file(tensors, str(tmp), metadata={"schema": SCHEMA_VERSION})
        os.replace(tmp, tensor_path)
        metadata = {
            **metadata,
            "schema": SCHEMA_VERSION,
            "tensor_file": tensor_path.name,
            "tensors": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()
            },
        }
        meta_path = self.path / f"{stem}.json"
        atomic_json(meta_path, metadata)
        for p in (tensor_path, meta_path):
            self.files[str(p.relative_to(self.parent.root))] = file_hash(p)
        self.bytes += added

    def checkpoint(self, progress: dict | None = None) -> None:
        if progress is not None:
            self.generation_progress = copy.deepcopy(progress)
        atomic_json(
            self.path / "progress.json",
            {"files": self.files, "bytes": self.bytes, "generation": self.generation_progress},
        )
        self.parent.commit()

    def finish(self, result: dict) -> dict:
        result_path = self.path / "result.json"
        atomic_json(result_path, result)
        self.files[str(result_path.relative_to(self.parent.root))] = file_hash(result_path)
        receipt = {
            "schema": SCHEMA_VERSION,
            "status": "complete",
            "files": self.files,
            "attempt": str(self.path.relative_to(self.parent.root)),
            "tensor_bytes": self.bytes,
        }
        atomic_json(self.base / "complete.json", receipt)
        self.parent.commit()
        return receipt

    def fail(self, exc: Exception) -> None:
        atomic_json(self.path / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        self.checkpoint()


def iter_records(run: str | Path, sample_id: str | None = None, fields: set[str] | None = None):
    root = Path(run)
    receipts = (
        [root / "samples" / identifier(sample_id) / "complete.json"]
        if sample_id
        else sorted((root / "samples").glob("*/complete.json"))
    )
    for path in receipts:
        receipt = json.loads(path.read_text())
        for relative, expected in sorted(receipt["files"].items()):
            target = root / relative
            if target.name.startswith("step-") and target.suffix == ".json":
                if file_hash(target) != expected:
                    raise ValueError(f"Artifact hash mismatch: {relative}")
                meta = json.loads(target.read_text())
                if fields is not None and not fields.intersection(meta["tensors"]):
                    continue
                tensor_path = target.parent / meta["tensor_file"]
                tensor_relative = str(tensor_path.relative_to(root))
                if receipt["files"].get(tensor_relative) != file_hash(tensor_path):
                    raise ValueError("Unverified tensor shard")
                with safe_open(tensor_path, framework="pt", device="cpu") as f:
                    tensors = {
                        k: f.get_tensor(k) for k in f.keys() if fields is None or k in fields
                    }
                for record in meta.get("records", [meta]):
                    keys = set(record.get("keys", tensors)) | {"positions"}
                    selected = {k: v for k, v in tensors.items() if k in keys}
                    if fields is not None and not fields.intersection(selected):
                        continue
                    yield path.parent.name, record, selected
