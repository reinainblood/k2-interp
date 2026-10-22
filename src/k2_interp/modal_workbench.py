"""Independent CPU web controller. Deploy updates without touching NativeWorker."""

from __future__ import annotations

import os
from pathlib import Path

import modal

from .modal_app import artifacts, image
from .registry import CHECKPOINTS

app = modal.App("k2-interp-workbench")
sessions = modal.Volume.from_name(
    "k2-interp-workbench", environment_name="FDR", create_if_missing=True
)
live = modal.Dict.from_name("k2-interp-live", environment_name="FDR", create_if_missing=True)
enabled = os.environ.get("K2_WORKBENCH_CHECKPOINTS", "sft_3_1500").split(",")
if not enabled or set(enabled) - set(CHECKPOINTS):
    raise ValueError("Unknown workbench checkpoint")
assets = Path(__file__).resolve().parents[2] / "web" / "dist"
web_image = image.add_local_dir(str(assets), "/web", copy=True).env(
    {"K2_WORKBENCH_CHECKPOINTS": ",".join(enabled)}
)


@app.function(
    image=web_image,
    cpu=2,
    memory=8192,
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    timeout=3600,
    secrets=[modal.Secret.from_name("k2-interp-workbench-auth", environment_name="FDR")],
    volumes={"/sessions": sessions, "/artifacts": artifacts.with_mount_options(read_only=True)},
)
@modal.concurrent(max_inputs=40)
@modal.asgi_app()
def web():
    from .capture_view import latest_projection
    from .workbench import create_workbench
    from .workbench_auth import BrowserAuth
    from .workbench_store import SessionStore

    class Backend:
        checkpoints = enabled

        def start(self, checkpoint, request):
            worker = modal.Cls.from_name(
                "k2-interp-" + checkpoint.replace("_", "-"), "NativeWorker", environment_name="FDR"
            )()
            return worker.live_chat.spawn(request).object_id

        def state(self, checkpoint, turn):
            value = live.get("live:" + turn["run_id"], None)
            if value is not None:
                return value
            try:
                artifacts.reload()
                return latest_projection(Path("/artifacts"), checkpoint, turn["run_id"])
            except FileNotFoundError:
                pass
            # Surface dispatch/runtime failures even when telemetry never started.
            if turn.get("call_id"):
                try:
                    result = modal.FunctionCall.from_id(turn["call_id"]).get(timeout=0)
                    from .telemetry import utc_now

                    return {
                        "status": "complete",
                        "result": result,
                        "sequence": 0,
                        "updated_at": utc_now(),
                        "run_id": turn["run_id"],
                    }
                except (TimeoutError, modal.exception.TimeoutError):
                    return None
                except Exception as exc:
                    from .telemetry import utc_now

                    return {
                        "status": "failed",
                        "sequence": 0,
                        "updated_at": utc_now(),
                        "run_id": turn["run_id"],
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
            return None

        def cancel(self, run_id):
            live.put("cancel:" + run_id, True)

        def reload_artifacts(self):
            artifacts.reload()

    return create_workbench(
        SessionStore("/sessions", sessions.commit),
        Backend(),
        BrowserAuth(os.environ["K2_WORKBENCH_SIGNING_KEY"], live),
        Path("/artifacts"),
        Path("/web"),
    )
