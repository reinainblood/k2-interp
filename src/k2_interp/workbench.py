"""Private research workbench API. The controller never hosts or restarts model weights."""

from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from starlette.background import BackgroundTask

from .capture_view import capture_bundle, inspect_token
from .registry import CHECKPOINTS
from .schema import CaptureSpec, Contract, GenerationSpec
from .telemetry import utc_now
from .workbench_auth import COOKIE, BrowserAuth
from .workbench_store import TERMINAL, SessionStore


class NewSession(Contract):
    checkpoint: str = "sft_3_1500"
    system: str = Field(default="", max_length=32000)


class NewTurn(Contract):
    text: str = Field(min_length=1, max_length=100000)
    request_id: str = Field(pattern=r"^[a-f0-9-]{16,40}$")
    generation: GenerationSpec = Field(default_factory=lambda: GenerationSpec(max_new_tokens=512))
    layers: list[int] = Field(default_factory=lambda: [10, 30, 50], max_length=8)
    expert_norms: bool = True
    max_context: int = Field(default=16384, ge=512, le=65536)
    reasoning_effort: str = Field(default="high", pattern=r"^(high|medium|low)$")


class Bookmark(Contract):
    position: int = Field(ge=0)
    layer: int = Field(ge=0, le=60)
    tag: str = Field(default="", max_length=120)


def create_workbench(
    store: SessionStore, backend, auth: BrowserAuth, artifact_root: Path, static_root: Path
):
    web = FastAPI(title="K2 Interp Workbench", docs_url=None, redoc_url=None, openapi_url=None)
    trace_lock = threading.RLock()
    login_attempts = {}

    @web.middleware("http")
    async def protect(request: Request, call_next):
        path = request.url.path
        if request.method not in ("GET", "HEAD"):
            limit = 2048 if path == "/api/login" else 2_000_000
            try:
                if int(request.headers.get("content-length", "0")) > limit:
                    return JSONResponse({"detail": "Request too large"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Invalid content length"}, status_code=400)
            origin = request.headers.get("origin")
            expected = f"{request.url.scheme}://{request.url.netloc}"
            if (origin and origin != expected) or request.headers.get("x-k2-intent") != "workbench":
                return JSONResponse({"detail": "Invalid request origin"}, status_code=403)
        if path.startswith("/api/") and path != "/api/login":
            if not auth.valid(request.cookies.get(COOKIE)):
                return JSONResponse({"detail": "Unauthenticated"}, status_code=401)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' blob: data:; font-src 'self'; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        return response

    @web.post("/api/login")
    async def login(request: Request):
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        recent = [t for t in login_attempts.get(ip, []) if now - t < 60]
        login_attempts[ip] = recent + [now]
        if len(recent) >= 10:
            raise HTTPException(429, "Login rate limit")
        body = await request.json()
        if not auth.redeem(body.get("ticket")):
            raise HTTPException(401, "Expired or invalid access ticket")
        response = JSONResponse({"authenticated": True})
        response.set_cookie(
            COOKIE,
            auth.issue(),
            secure=auth.secure,
            httponly=True,
            samesite="strict",
            max_age=43200,
        )
        return response

    @web.post("/api/logout")
    def logout():
        response = JSONResponse({"authenticated": False})
        response.delete_cookie(COOKIE)
        return response

    @web.get("/api/config")
    def config():
        return {
            "checkpoints": backend.checkpoints,
            "registry": CHECKPOINTS,
            "environment": "FDR",
            "schema": "k2-workbench/1",
        }

    def session_or_404(session_id):
        try:
            return store.get(session_id)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "Session not found") from None

    def turn_or_404(session_id, run_id):
        session = session_or_404(session_id)
        turn = next((t for t in session["turns"] if t["run_id"] == run_id), None)
        if turn is None:
            raise HTTPException(404, "Turn not found")
        return session, turn

    def state_for(session, turn):
        with trace_lock:
            state = backend.state(session["checkpoint"], turn)
        if state:
            store.update_turn(session["id"], turn["run_id"], state)
        return state or {"status": turn["status"], "sequence": -1, "run_id": turn["run_id"]}

    @web.get("/api/sessions")
    def sessions():
        return store.list()

    @web.post("/api/sessions")
    def create_session(body: NewSession):
        if body.checkpoint not in backend.checkpoints:
            raise HTTPException(422, "Checkpoint is not enabled on this controller")
        return store.create(body.checkpoint, body.system)

    @web.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        session = session_or_404(session_id)
        for turn in session["turns"]:
            if turn["status"] not in TERMINAL:
                state_for(session, turn)
        return store.get(session_id)

    @web.post("/api/sessions/{session_id}/settings")
    def edit_session(session_id: str, body: NewSession):
        with store.lock:
            session = session_or_404(session_id)
            if body.checkpoint != session["checkpoint"]:
                raise HTTPException(422, "Create a new session to change checkpoint")
            if any(t["status"] not in TERMINAL for t in session["turns"]):
                raise HTTPException(409, "Turn still active")
            session["system"] = body.system
            session["system_updated_at"] = utc_now()
            return store.save(session)

    @web.post("/api/sessions/{session_id}/turns")
    def send(session_id: str, body: NewTurn):
        if not body.text.strip() or body.generation.max_new_tokens < 1:
            raise HTTPException(422, "Empty turn")
        if any(layer < 0 or layer > 60 for layer in body.layers):
            raise HTTPException(422, "Layer outside model")
        with store.lock:
            session = session_or_404(session_id)
            if old := next(
                (t for t in session["turns"] if t["request_id"] == body.request_id), None
            ):
                return old
            for turn in session["turns"]:
                if turn["status"] not in TERMINAL:
                    state_for(session, turn)
            session = store.get(session_id)
            if any(t["status"] not in TERMINAL for t in session["turns"]):
                raise HTTPException(409, "Turn still active")
            request = {
                "run_id": "chat-" + uuid.uuid4().hex,
                "messages": store.history(session) + [{"role": "user", "content": body.text}],
                **body.generation.model_dump(),
                "max_context": body.max_context,
                "chat_template_kwargs": {"reasoning_effort": body.reasoning_effort},
                "capture": CaptureSpec(
                    layers=body.layers, expert_output_norms=body.expert_norms
                ).model_dump(),
            }
            turn = {
                "run_id": request["run_id"],
                "request_id": body.request_id,
                "user": body.text,
                "status": "queued",
                "created_at": utc_now(),
                "settings": body.model_dump(exclude={"text"}),
                "assistant": None,
                "system": session["system"],
            }
            # Persist intent before dispatch. A retry with the same request ID cannot duplicate a run.
            session["turns"].append(turn)
            store.save(session)
            try:
                turn["call_id"] = backend.start(session["checkpoint"], request)
            except Exception as exc:
                turn.update(
                    status="failed", error={"type": type(exc).__name__, "message": str(exc)}
                )
                store.save(session)
                raise HTTPException(503, "Model dispatch failed") from None
            store.save(session)
            return turn

    @web.get("/api/sessions/{session_id}/turns/{run_id}/state")
    def state(session_id: str, run_id: str):
        return state_for(*turn_or_404(session_id, run_id))

    @web.get("/api/sessions/{session_id}/turns/{run_id}/events")
    def events(session_id: str, run_id: str, request: Request):
        session, turn = turn_or_404(session_id, run_id)

        def generate():
            previous, started, heartbeat = "", time.monotonic(), 0.0
            while time.monotonic() - started < 55:
                value = state_for(session, turn)
                signature = str(value.get("sequence")) + value["status"]
                if signature != previous:
                    yield "data: " + json.dumps(value, allow_nan=False) + "\n\n"
                    previous = signature
                elif time.monotonic() - heartbeat > 10:
                    yield ": heartbeat\n\n"
                    heartbeat = time.monotonic()
                if value["status"] in TERMINAL:
                    break
                time.sleep(0.5)

        return StreamingResponse(
            generate(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    @web.post("/api/sessions/{session_id}/turns/{run_id}/cancel")
    def cancel(session_id: str, run_id: str):
        session, turn = turn_or_404(session_id, run_id)
        if turn["status"] not in TERMINAL:
            backend.cancel(run_id)
        return {"status": "cancel_requested"}

    @web.get("/api/sessions/{session_id}/turns/{run_id}/tokens/{position}")
    def token(session_id: str, run_id: str, position: int):
        session, _ = turn_or_404(session_id, run_id)
        with trace_lock:
            backend.reload_artifacts()
            try:
                return inspect_token(artifact_root, session["checkpoint"], run_id, position)
            except FileNotFoundError:
                raise HTTPException(409, "Capture not committed yet") from None

    @web.post("/api/sessions/{session_id}/turns/{run_id}/captures")
    def capture(session_id: str, run_id: str, body: Bookmark):
        session, turn = turn_or_404(session_id, run_id)
        prefix = copy.deepcopy(session)
        prefix["captures"] = []
        with trace_lock:
            backend.reload_artifacts()
            try:
                path, index = capture_bundle(
                    artifact_root,
                    session["checkpoint"],
                    run_id,
                    body.position,
                    body.layer,
                    prefix,
                    body.tag,
                )
            except FileNotFoundError:
                raise HTTPException(409, "Capture not committed yet") from None
        with store.lock:
            current = store.get(session_id)
            current["captures"].append({k: v for k, v in index.items() if k != "files"})
            store.save(current)
        return FileResponse(
            path,
            media_type="application/zip",
            filename=f"k2-{index['capture_id']}.zip",
            background=BackgroundTask(path.unlink, missing_ok=True),
        )

    from .workbench_exports import register_session_exports

    register_session_exports(web, store, backend, artifact_root, trace_lock, session_or_404)
    web.mount("/", StaticFiles(directory=static_root, html=True), name="workbench")
    return web
