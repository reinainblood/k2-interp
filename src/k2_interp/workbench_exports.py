"""Session-wide export route, kept separate from live dashboard development."""

from __future__ import annotations

import copy

from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .conversation_export import conversation_bundle
from .telemetry import utc_now


def register_session_exports(web, store, backend, artifact_root, trace_lock, session_or_404):
    @web.get("/api/sessions/{session_id}/export")
    def export_session(session_id: str):
        with store.lock:
            session = copy.deepcopy(session_or_404(session_id))
        with trace_lock:
            backend.reload_artifacts()
            path, _ = conversation_bundle(artifact_root, session["checkpoint"], session, utc_now())
        return FileResponse(
            path,
            media_type="application/zip",
            filename=f"k2-{session_id}-conversation.zip",
            background=BackgroundTask(path.unlink, missing_ok=True),
        )
