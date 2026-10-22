import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from k2_interp.artifacts import TraceWriter, atomic_json, file_hash
from k2_interp.capture_view import capture_bundle
from k2_interp.conversation_export import collect_conversation, conversation_bundle


def make_turn(root, index, complete=True):
    run_id = f"chat-{index}"
    message = {
        "role": "assistant",
        "content": f"Answer {index}: café",
        "reasoning_content": f"Reason {index}\n```python\nprint('kept')\n```",
        "think_fast": "",
        "custom_native_field": {"retained": True},
    }
    spec = {
        "run_id": run_id,
        "checkpoint": "test",
        "samples": [{"id": "chat", "messages": [{"role": "user", "content": f"Question {index}"}]}],
    }
    writer = TraceWriter(root / "test", spec, {"checkpoint": "test"})
    sink = writer.sample("chat", 100000)
    tensors = {
        "positions": torch.tensor([0, 1]),
        "layers.3.expert_ids": torch.tensor([[0, 1], [1, 0]]),
        "layers.3.expert_weights": torch.tensor([[1.0, 1.5], [1.5, 1.0]]),
    }
    sink.record(
        0,
        {
            "step": 0,
            "phase": "prefill",
            "records": [
                {
                    "step": 0,
                    "phase": "prefill",
                    "layer": 3,
                    "positions": [0, 1],
                    "token_ids": [10, 11],
                    "keys": list(tensors),
                }
            ],
        },
        tensors,
    )
    raw = {
        "assistant_message": message,
        "input_ids": [10, 11],
        "output_ids": [12],
        "all_token_ids": [10, 11, 12],
        "output_text": "<ifm|think>Raw reasoning</ifm|think>Raw answer",
        "finish_reason": "eos",
        "output_logprobs": [-0.2],
    }
    if complete:
        sink.finish(raw)
    else:
        projection = sink.path / "workbench-step-000000.json"
        atomic_json(projection, {"assistant_message": message, "output_text": "<ifm|think>partial"})
        sink.files[str(projection.relative_to(writer.root))] = file_hash(projection)
        sink.checkpoint({"step": 0, "output_ids": [12], "input_ids": [10, 11]})
    # Deliberately stale controller state: exports must recover durable assistant reasoning.
    return {"run_id": run_id, "user": f"Question {index}", "status": "queued", "assistant": None}


def test_all_twenty_turns_and_reasoning_survive_old_state_capture(tmp_path):
    session = {
        "id": "session-test",
        "checkpoint": "test",
        "system": "System text",
        "turns": [make_turn(tmp_path, index) for index in range(20)],
        "captures": [],
    }
    bundle, index = capture_bundle(tmp_path, "test", "chat-0", 0, 3, session, "early-state")
    try:
        with zipfile.ZipFile(bundle) as archive:
            conversation = json.loads(archive.read("conversation.json"))
            context = json.loads(archive.read("capture-context.json"))
            assert len(conversation["turns"]) == 20
            assert conversation["export"]["archive_complete"] is True
            assert context["messages"] == [{"role": "user", "content": "Question 0"}]
            assert index["run_id"] == "chat-0"
            for i, turn in enumerate(conversation["turns"]):
                assert turn["assistant"]["reasoning_content"].startswith(f"Reason {i}")
                assert turn["assistant"]["think_fast"] == ""
                assert turn["assistant"]["custom_native_field"] == {"retained": True}
                assert "<ifm|think>Raw reasoning" in turn["raw_result"]["output_text"]
            text = archive.read("conversation.md").decode()
            assert "Reason 19" in text and "café" in text
            assert set(index["files"]) == set(archive.namelist()) - {"capture.json"}
            from hashlib import sha256

            for name, expected in index["files"].items():
                assert sha256(archive.read(name)).hexdigest() == expected
        assert session["turns"][-1]["assistant"] is None  # export never mutates live state
    finally:
        bundle.unlink()


def test_partial_and_missing_turns_are_preserved_and_labeled(tmp_path):
    partial = make_turn(tmp_path, 0, complete=False)
    session = {
        "id": "session-test",
        "checkpoint": "test",
        "turns": [
            partial,
            {"run_id": "chat-missing", "user": "Kept despite failure", "status": "failed"},
        ],
    }
    result = collect_conversation(tmp_path, "test", session, "2026-10-22T00:00:00Z")
    assert result["export"]["archive_complete"] is False
    assert [item["status"] for item in result["export"]["coverage"]] == ["partial", "missing"]
    assert result["turns"][0]["partial_assistant"]["reasoning_content"].startswith("Reason 0")
    assert result["turns"][0]["partial_output_text"] == "<ifm|think>partial"
    assert result["turns"][1]["user"] == "Kept despite failure"


def test_corrupt_result_refuses_export(tmp_path):
    turn = make_turn(tmp_path, 0)
    path = next((tmp_path / "test" / "chat-0").glob("samples/chat/attempt-*/result.json"))
    atomic_json(path, {"assistant_message": {"content": "changed"}})
    with pytest.raises(ValueError, match="failed verification"):
        conversation_bundle(tmp_path, "test", {"turns": [turn]}, "2026-10-22")


def test_conversation_archive_contains_raw_ids_and_json_hashes(tmp_path):
    session = {"id": "session-test", "checkpoint": "test", "turns": [make_turn(tmp_path, 0)]}
    path, metadata = conversation_bundle(tmp_path, "test", session, "2026-10-22")
    try:
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read("conversation.json"))
            assert data["turns"][0]["raw_result"]["output_ids"] == [12]
            assert metadata["includes_all_tensor_shards"] is False
            assert set(archive.namelist()) == {
                "conversation.json",
                "conversation.md",
                "export.json",
            }
    finally:
        Path(path).unlink()


def test_export_verifier_checks_all_files_without_extracting(tmp_path):
    session = {"id": "session-test", "checkpoint": "test", "turns": [make_turn(tmp_path, 0)]}
    path, _ = conversation_bundle(tmp_path, "test", session, "2026-10-22")
    try:
        script = Path(__file__).resolve().parents[1] / "scripts" / "verify_export.py"
        result = subprocess.run(
            [sys.executable, str(script), str(path)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["hashes_verified"] is True
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("unexpected.txt", "not in inventory")
        result = subprocess.run(
            [sys.executable, str(script), str(path)], capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "inventory" in result.stderr
    finally:
        path.unlink()


def test_raw_trace_verifier_checks_tensors_and_manifest(tmp_path):
    make_turn(tmp_path, 0)
    root = tmp_path / "test" / "chat-0"
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_trace.py"
    result = subprocess.run(
        [sys.executable, str(script), str(root)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["run_complete"] is True
    shard = next(root.glob("samples/chat/attempt-*/*.safetensors"))
    with shard.open("ab") as stream:
        stream.write(b"corrupt")
    result = subprocess.run(
        [sys.executable, str(script), str(root)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "checksum" in result.stderr


def test_session_export_route_uses_auth_and_preserves_reasoning(tmp_path):
    from fastapi.testclient import TestClient

    from k2_interp.workbench import create_workbench
    from k2_interp.workbench_auth import COOKIE, BrowserAuth
    from k2_interp.workbench_store import SessionStore

    class Backend:
        checkpoints = ["test"]

        def reload_artifacts(self):
            pass

    store = SessionStore(tmp_path / "sessions")
    session = store.create("test")
    session["turns"].append(make_turn(tmp_path / "artifacts", 0))
    store.save(session)
    static = tmp_path / "static"
    static.mkdir()
    auth = BrowserAuth("k" * 48, {}, secure=False)
    client = TestClient(create_workbench(store, Backend(), auth, tmp_path / "artifacts", static))
    url = f"/api/sessions/{session['id']}/export"
    assert client.get(url).status_code == 401
    client.cookies.set(COOKIE, auth.issue())
    response = client.get(url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    from io import BytesIO

    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        data = json.loads(archive.read("conversation.json"))
        assert data["turns"][0]["assistant"]["reasoning_content"].startswith("Reason 0")
        assert data["export"]["archive_complete"]
