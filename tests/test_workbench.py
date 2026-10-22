import hashlib
import json
import time
import zipfile

import pytest
import torch
from fastapi.testclient import TestClient

from k2_interp.capture_view import capture_bundle, inspect_token, latest_projection
from k2_interp.hooks import NativeTrace
from k2_interp.runtime import Runtime, token_inputs
from k2_interp.schema import CaptureSpec, Sample
from k2_interp.telemetry import CaptureCancelled, LiveTelemetry
from k2_interp.workbench import create_workbench
from k2_interp.workbench_auth import BrowserAuth
from k2_interp.workbench_store import SessionStore


@pytest.fixture
def traced(tmp_path, native_model, tokenizer):
    published = []
    root = tmp_path / "artifacts"
    runtime = Runtime(native_model, tokenizer, {"checkpoint": "sft_3_1500"}, root / "sft_3_1500")
    with LiveTelemetry("trace", published.append, interval=0.001) as telemetry:
        response = runtime.chat(
            [{"role": "system", "content": "old system"}, {"role": "user", "content": "abc"}],
            run_id="trace",
            max_new_tokens=2,
            capture=CaptureSpec(layers=[1, 2], expert_output_norms=True),
            telemetry=telemetry,
        )
        telemetry.finish(response)
    return root, response, published


def test_live_projection_matches_native_tensors(traced):
    root, response, events = traced
    result = response["result"]
    position = len(result["all_token_ids"]) - 1
    view = inspect_token(root, "sft_3_1500", "trace", position)
    last = events[-1]
    assert last["status"] == "complete"
    assert last["position"] == position
    assert last["token_text"] == result["token_text"]
    assert last["output_ids"] == result["output_ids"]
    for layer in ("1", "2"):
        row = last["layers"][layer]
        for key in ("expert_ids", "expert_weights", "router_scores", "expert_norms", "norms"):
            assert row[key] == view["layers"][layer][key]
        assert sum(row["forward_expert_counts"]) == 2
        assert sum(row["expert_weights"]) == pytest.approx(2.5, rel=1e-5)
        assert all(row["expert_norms"][str(i)] > 0 for i in row["expert_ids"])
    first = inspect_token(root, "sft_3_1500", "trace", 0)
    assert first["token_id"] == result["input_ids"][0]
    assert "forward_expert_counts" not in first["layers"]["1"]
    assert latest_projection(root, "sft_3_1500", "trace")["status"] == "complete"


def test_expert_norms_do_not_change_logits(native_model):
    x = torch.tensor([[2, 3, 4]])
    expected = native_model(x).logits
    with NativeTrace(native_model, CaptureSpec(layers=[1], expert_output_norms=True)) as trace:
        trace.begin([0, 1, 2], step=0, phase="prefill", token_ids=[2, 3, 4])
        observed = native_model(x).logits
    torch.testing.assert_close(expected, observed, rtol=0, atol=0)


def test_capture_bundle_integrity_and_asof_state(traced):
    root, response, _ = traced
    session = {
        "id": "session-test",
        "system": "edited system",
        "turns": [
            {
                "run_id": "trace",
                "user": "abc",
                "status": "complete",
                "assistant": response["assistant_message"],
            },
            {"run_id": "later", "user": "later message", "status": "queued"},
        ],
    }
    path, index = capture_bundle(root, "sft_3_1500", "trace", 0, 1, session, "interesting")
    try:
        with zipfile.ZipFile(path) as archive:
            assert "router.svg" in archive.namelist()
            for name, expected in index["files"].items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == expected
            state = json.loads(archive.read("capture-state.json"))
            assert len(state["turns"]) == 1
            assert state["turns"][0]["assistant"]["content"] == ""
            assert state["system"] == "old system"
            assert len(json.loads(archive.read("conversation.json"))["turns"]) == 2
            assert index["captured_output_tokens"] == 0
    finally:
        path.unlink()


def test_capture_rejects_corrupt_shard(traced):
    root, _, _ = traced
    target = next(root.glob("sft_3_1500/trace/samples/chat/attempt-*/step-000000*.safetensors"))
    with target.open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="Unverified"):
        inspect_token(root, "sft_3_1500", "trace", 0)


def test_cancellation_and_hook_cleanup(tmp_path, native_model, tokenizer):
    runtime = Runtime(native_model, tokenizer, {"checkpoint": "test"}, tmp_path)
    with LiveTelemetry("cancel", lambda _: None) as telemetry:
        telemetry.cancel.set()
        with pytest.raises(CaptureCancelled):
            runtime.chat([{"role": "user", "content": "abc"}], run_id="cancel", telemetry=telemetry)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in native_model.modules())


def test_native_reasoning_efforts_preserve_history_tag(native_tokenizer):
    for effort, key in (("medium", "think_fast"), ("low", "think_faster")):
        messages = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two", "reasoning_content": "reason", key: "reason"},
            {"role": "user", "content": "three"},
        ]
        ids, _ = token_inputs(
            native_tokenizer,
            Sample(id="test", messages=messages, chat_template_kwargs={"reasoning_effort": effort}),
        )
        text = native_tokenizer.decode(ids, skip_special_tokens=False)
        assert f"<ifm|{key}>" in text
        assert "reasoning_content" in messages[1]


@pytest.fixture
def web_client(tmp_path):
    class Backend:
        checkpoints = ["sft_3_1500"]
        requests = []
        states = {}

        def start(self, checkpoint, request):
            self.requests.append(request)
            return "fc-test"

        def state(self, checkpoint, turn):
            return self.states.get(turn["run_id"])

        def reload_artifacts(self):
            pass

        def cancel(self, run_id):
            self.states[run_id] = {"status": "cancelled", "updated_at": "now", "sequence": 1}

    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("workbench")
    tickets = {}
    auth = BrowserAuth("a" * 48, tickets, secure=False)
    backend = Backend()
    store = SessionStore(tmp_path / "sessions")
    client = TestClient(create_workbench(store, backend, auth, tmp_path / "artifacts", static))
    client.headers["X-K2-Intent"] = "workbench"
    return client, auth, backend, store


def authorize(client, auth):
    ticket = "t" * 40
    auth.tickets["ticket:" + hashlib.sha256(ticket.encode()).hexdigest()] = {
        "expires": time.time() + 10
    }
    assert client.post("/api/login", json={"ticket": ticket}).status_code == 200
    return ticket


def test_auth_is_required_for_all_private_routes(web_client):
    client, auth, _, _ = web_client
    assert client.get("/").status_code == 200
    for url in ("/api/config", "/api/sessions", "/api/sessions/session-test/export"):
        assert client.get(url).status_code == 401
    ticket = authorize(client, auth)
    assert client.post("/api/login", json={"ticket": ticket}).status_code == 401
    assert client.get("/api/config").status_code == 200
    assert (
        client.post("/api/sessions", json={}, headers={"Origin": "https://evil.test"}).status_code
        == 403
    )
    assert not auth.valid(auth.issue() + "tamper")
    client.post("/api/logout", json={})
    assert client.get("/api/config").status_code == 401


def test_dispatch_idempotency_system_edit_and_reasoning_history(web_client):
    client, auth, backend, store = web_client
    authorize(client, auth)
    session = client.post("/api/sessions", json={"system": "first"}).json()
    prefix = f"/api/sessions/{session['id']}"
    body = {"text": "abc", "request_id": "a" * 32, "reasoning_effort": "low"}
    turn = client.post(prefix + "/turns", json=body).json()
    assert client.post(prefix + "/turns", json=body).json()["run_id"] == turn["run_id"]
    assert len(backend.requests) == 1
    assert backend.requests[0]["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert backend.requests[0]["capture"]["expert_output_norms"]
    assert client.post(prefix + "/settings", json={"system": "second"}).status_code == 409
    message = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "",
        "think_faster": "thought",
    }
    backend.states[turn["run_id"]] = {
        "status": "complete",
        "updated_at": "now",
        "result": {"assistant_message": message},
    }
    client.get(prefix)
    assert client.post(prefix + "/settings", json={"system": "second"}).status_code == 200
    assert client.post(prefix + "/turns", json={**body, "request_id": "b" * 32}).status_code == 200
    assert backend.requests[-1]["messages"][0]["content"] == "second"
    assert backend.requests[-1]["messages"][2] == message
    assert store.get(session["id"])["turns"][0]["assistant"] == message
