"""Deploy with K2_CHECKPOINT=<registered tag> modal deploy --env=FDR -m k2_interp.modal_app.

No worker is kept warm by deployment. Preparation uses CPU; a model loads only
when an authenticated run/chat request arrives. Each app owns one checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

from k2_interp.registry import CHECKPOINTS

ENVIRONMENT = "FDR"
CHECKPOINT = os.environ.get("K2_CHECKPOINT", "sft_3_1500")
if CHECKPOINT not in CHECKPOINTS:
    raise ValueError("K2_CHECKPOINT is not in the immutable checkpoint registry")
APP_NAME = "k2-interp-" + CHECKPOINT.replace("_", "-")
PROJECT = Path(__file__).resolve().parents[2]
app = modal.App(APP_NAME)
weights = modal.Volume.from_name(
    "k2-interp-weights", environment_name=ENVIRONMENT, create_if_missing=True
)
artifacts = modal.Volume.from_name(
    "k2-interp-artifacts", environment_name=ENVIRONMENT, create_if_missing=True
)
# Optional existing cache is read-only and must pass full shard hashes before reuse.
legacy = modal.Volume.from_name("k2-horizon-bf16-weights", environment_name=ENVIRONMENT)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(PROJECT / "requirements-modal.txt"))
    .add_local_dir(str(PROJECT / "src"), "/opt/k2-interp/src", copy=True)
    .add_local_file(str(PROJECT / "pyproject.toml"), "/opt/k2-interp/pyproject.toml", copy=True)
    .add_local_file(str(PROJECT / "README.md"), "/opt/k2-interp/README.md", copy=True)
    .run_commands("pip install --no-deps /opt/k2-interp")
    .env(
        {
            "HF_HOME": "/tmp/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "K2_CHECKPOINT": CHECKPOINT,
            "K2_INTERP_CODE_REVISION": os.environ.get("K2_INTERP_CODE_REVISION", "unrecorded"),
        }
    )
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=21600,
    max_containers=1,
    volumes={"/weights": weights, "/legacy": legacy.with_mount_options(read_only=True)},
)
def prepare_checkpoint(reuse_subdir: str = ""):
    from k2_interp.weights import prepare

    if reuse_subdir and ("/" in reuse_subdir or ".." in reuse_subdir):
        raise ValueError("reuse_subdir must be a single checkpoint directory name")
    return prepare(
        CHECKPOINT,
        "/weights",
        reuse=f"/legacy/{reuse_subdir}" if reuse_subdir else None,
        commit=weights.commit,
    )


@app.function(
    image=image,
    cpu=1,
    timeout=60,
    volumes={
        "/weights": weights.with_mount_options(read_only=True),
        "/legacy": legacy.with_mount_options(read_only=True),
    },
)
def preflight():
    from k2_interp.weights import ready

    path, lock = ready(CHECKPOINT, "/weights")
    return {
        "checkpoint": CHECKPOINT,
        "revision": lock["revision"],
        "ready": True,
        "path": str(path),
        "weight_manifest_sha256": lock["weight_manifest_sha256"],
    }


@app.cls(
    image=image,
    gpu="H200:8",
    cpu=16,
    memory=262144,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=60,
    timeout=7200,
    startup_timeout=1800,
    retries=0,
    volumes={
        "/weights": weights.with_mount_options(read_only=True),
        "/legacy": legacy.with_mount_options(read_only=True),
        "/artifacts": artifacts,
    },
)
class NativeWorker:
    @modal.enter()
    def load(self):
        from k2_interp.runtime import Runtime
        from k2_interp.weights import native_load

        model, tokenizer, identity = native_load(CHECKPOINT, "/weights")
        self.runtime = Runtime(
            model, tokenizer, identity, "/artifacts/" + CHECKPOINT, artifacts.commit
        )

    @modal.method()
    def run(self, request: dict):
        from k2_interp.schema import RunSpec

        artifacts.reload()
        return self.runtime.run(RunSpec.model_validate(request))

    @modal.method()
    def chat(self, request: dict):
        artifacts.reload()
        return self.runtime.chat(**request)

    @modal.method()
    def intervene(self, request: dict):
        from k2_interp.causal import intervention_sweep

        artifacts.reload()
        return intervention_sweep(self.runtime, **request)

    @modal.method()
    def smoke(self, run_id: str):
        from k2_interp.validation import full_model_smoke

        return full_model_smoke(self.runtime, run_id)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=86400,
    max_containers=1,
    min_containers=0,
    retries=0,
    volumes={"/artifacts": artifacts},
)
def train_dictionary(config: dict):
    from k2_interp.dictionary import train_from_config

    artifacts.reload()
    result = train_from_config(config, root="/artifacts", commit=artifacts.commit)
    return result


@app.function(image=image, timeout=7200)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    from fastapi import FastAPI, HTTPException

    from k2_interp.schema import RunSpec

    web = FastAPI(title="K2 Interp", version="0.1.0")

    @web.get("/health")
    def health():
        return {
            "status": "gateway_ready",
            "checkpoint": CHECKPOINT,
            "revision": CHECKPOINTS[CHECKPOINT],
            "model_loaded": "query worker separately",
        }

    @web.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": CHECKPOINT,
                    "object": "model",
                    "owned_by": "IFM",
                    "revision": CHECKPOINTS[CHECKPOINT],
                }
            ],
        }

    @web.post("/runs")
    async def run(request: RunSpec):
        if request.checkpoint != CHECKPOINT:
            raise HTTPException(400, "Checkpoint does not match this deployment")
        await preflight.remote.aio()
        call = await NativeWorker().run.spawn.aio(request.model_dump())
        return {"call_id": call.object_id, "run_id": request.run_id}

    @web.post("/v1/chat/completions")
    async def chat(request: dict):
        import time
        import uuid

        if request.get("stream"):
            raise HTTPException(400, "Streaming is not supported; use non-streaming chat")
        if request.get("model", CHECKPOINT) != CHECKPOINT:
            raise HTTPException(400, "Checkpoint mismatch")
        await preflight.remote.aio()
        run_id = "chat-" + uuid.uuid4().hex
        response = await NativeWorker().chat.remote.aio(
            {
                "messages": request["messages"],
                "run_id": run_id,
                "max_new_tokens": request.get("max_tokens", 256),
                "temperature": request.get("temperature", 0),
                "seed": request.get("seed", 0),
            }
        )
        result = response["result"]
        return {
            "id": run_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": CHECKPOINT,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["output_text"]},
                    "finish_reason": result["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": len(result["input_ids"]),
                "completion_tokens": len(result["output_ids"]),
                "total_tokens": len(result["all_token_ids"]),
            },
            "k2_interp": {"revision": CHECKPOINTS[CHECKPOINT], "run_id": run_id},
        }

    return web


@app.local_entrypoint()
def main(action: str = "preflight", reuse_subdir: str = "", run_id: str = "runtime-smoke"):
    import json

    if action == "prepare":
        result = prepare_checkpoint.remote(reuse_subdir)
        result = {k: v for k, v in result.items() if k not in ("files", "local_files")}
    elif action == "preflight":
        result = preflight.remote()
    elif action == "smoke":
        preflight.remote()
        result = NativeWorker().smoke.remote(run_id)
    else:
        raise ValueError("action must be prepare, preflight, or smoke")
    print(json.dumps(result, indent=2))
