"""Single-controller, durable private session state; tensor artifacts stay in their own volume."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from .artifacts import atomic_json
from .capture_view import read_json
from .schema import identifier
from .telemetry import utc_now

TERMINAL = {"complete", "failed", "cancelled"}


class SessionStore:
    def __init__(self, root, commit=lambda: None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.commit = commit
        self.lock = threading.RLock()

    def get(self, session_id):
        return read_json(self.root / (identifier(session_id) + ".json"))

    def save(self, session):
        atomic_json(self.root / (identifier(session["id"]) + ".json"), session)
        self.commit()
        return session

    def create(self, checkpoint, system=""):
        session = {
            "id": "session-" + uuid.uuid4().hex,
            "checkpoint": checkpoint,
            "created_at": utc_now(),
            "system": system,
            "turns": [],
            "captures": [],
        }
        with self.lock:
            return self.save(session)

    def list(self):
        result = []
        with self.lock:
            for path in self.root.glob("session-*.json"):
                session = read_json(path)
                result.append(
                    {
                        "id": session["id"],
                        "checkpoint": session["checkpoint"],
                        "created_at": session["created_at"],
                        "turns": len(session["turns"]),
                        "title": session["turns"][0]["user"][:64] if session["turns"] else "",
                    }
                )
        return sorted(result, key=lambda x: x["created_at"], reverse=True)

    def update_turn(self, session_id, run_id, state):
        with self.lock:
            session = self.get(session_id)
            turn = next(t for t in session["turns"] if t["run_id"] == run_id)
            if state["status"] in TERMINAL and turn["status"] not in TERMINAL:
                turn["status"] = state["status"]
                turn["finished_at"] = state["updated_at"]
                turn["assistant"] = state.get("result", {}).get("assistant_message")
                turn["error"] = state.get("error")
                turn["finish_reason"] = (
                    state.get("result", {}).get("result", {}).get("finish_reason")
                )
                self.save(session)

    def history(self, session):
        messages = [{"role": "system", "content": session["system"]}] if session["system"] else []
        for turn in session["turns"]:
            # A failed or cancelled turn never silently enters the next model context.
            if turn["status"] == "complete" and turn.get("assistant"):
                messages.extend([{"role": "user", "content": turn["user"]}, turn["assistant"]])
        return messages
