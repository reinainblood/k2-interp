from types import SimpleNamespace

from fastapi.testclient import TestClient


def test_api_schema_rejects_ignored_sampling_and_checkpoint(monkeypatch):
    from k2_interp import modal_app

    requests = []

    async def ready():
        return {"ready": True}

    async def chat(request):
        requests.append(request)
        return {
            "result": {
                "input_ids": [2, 3],
                "output_ids": [4],
                "all_token_ids": [2, 3, 4],
                "finish_reason": "eos",
            },
            "assistant_message": {
                "role": "assistant",
                "content": "OK",
                "reasoning_content": "brief reasoning",
            },
        }

    monkeypatch.setattr(modal_app, "preflight", SimpleNamespace(remote=SimpleNamespace(aio=ready)))
    monkeypatch.setattr(
        modal_app,
        "NativeWorker",
        lambda: SimpleNamespace(chat=SimpleNamespace(remote=SimpleNamespace(aio=chat))),
    )
    client = TestClient(modal_app.api.local())
    assert client.get("/health").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    payload = {
        "model": "sft_3_1500",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
        "top_p": 0.8,
        "seed": 9,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert requests[0]["temperature"] == 0.7 and requests[0]["top_p"] == 0.8
    assert response.json()["choices"][0]["message"]["reasoning_content"] == "brief reasoning"
    # Token identity travels with every turn so clients can freeze trajectories.
    block = response.json()["k2_interp"]
    assert block["input_ids"] == [2, 3] and block["output_ids"] == [4]
    assert block["captured"] is False and "capture" not in requests[0]
    captured = client.post(
        "/v1/chat/completions",
        json={**payload, "capture": {"router": True, "router_fields": ["expert_ids"], "sites": []}},
    )
    assert captured.status_code == 200 and captured.json()["k2_interp"]["captured"] is True
    assert requests[1]["capture"]["router_fields"] == ["expert_ids"]
    assert (
        client.post(
            "/v1/chat/completions", json={**payload, "capture": {"router_fields": ["bogus"]}}
        ).status_code
        == 422
    )
    assert (
        client.post("/v1/chat/completions", json={**payload, "logit_bias": {"5": 100}}).status_code
        == 422
    )
    assert (
        client.post("/v1/chat/completions", json={**payload, "model": "mid_4_6000"}).status_code
        == 400
    )
    assert client.post("/v1/chat/completions", json={**payload, "stream": True}).status_code == 400
