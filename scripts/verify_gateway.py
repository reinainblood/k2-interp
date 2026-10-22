"""Verify proxy authentication and optional bounded two-turn chat; never print credentials."""

import argparse
import json
import os
from pathlib import Path

import httpx

from k2_interp.artifacts import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--checkpoint", default="sft_3_1500")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--token-file", default=".private/modal-proxy.json")
    parser.add_argument("--output", default=".private/gateway-verification.json")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    if not base.startswith("https://") or not base.endswith(".modal.run"):
        raise ValueError("Use the exact HTTPS Modal gateway URL")
    result = {"gateway": base, "checkpoint": args.checkpoint}
    with httpx.Client(timeout=1800) as client:
        denied = client.get(base + "/health")
        result["anonymous_status"] = denied.status_code
        if denied.status_code not in (401, 403):
            raise AssertionError("Anonymous gateway access was not rejected")
        credentials = (
            json.loads(Path(args.token_file).read_text()) if Path(args.token_file).exists() else {}
        )
        headers = {
            "Modal-Key": credentials.get("token_id") or os.environ.get("MODAL_PROXY_TOKEN_ID"),
            "Modal-Secret": credentials.get("token_secret")
            or os.environ.get("MODAL_PROXY_TOKEN_SECRET"),
        }
        if not all(headers.values()):
            raise ValueError(
                "Provide an FDR proxy token; CLI API tokens do not authenticate web requests"
            )
        for endpoint in ("/health", "/v1/models", "/openapi.json"):
            response = client.get(base + endpoint, headers=headers)
            response.raise_for_status()
            result[endpoint] = response.status_code
        if args.chat:
            messages = [{"role": "user", "content": "What is two plus two? Answer briefly."}]
            result["chat_turns"] = []
            for i in range(2):
                response = client.post(
                    base + "/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": args.checkpoint,
                        "messages": messages,
                        "max_tokens": 2,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "seed": 17,
                    },
                )
                response.raise_for_status()
                body = response.json()
                message = body["choices"][0]["message"]
                assert isinstance(message.get("reasoning_content"), str)
                assert 0 < body["usage"]["completion_tokens"] <= 2
                result["chat_turns"].append(
                    {
                        "turn": i,
                        "status": response.status_code,
                        "usage": body["usage"],
                        "run_id": body["id"],
                        "finish_reason": body["choices"][0]["finish_reason"],
                        "thinking_field_present": True,
                    }
                )
                messages.extend([message, {"role": "user", "content": "Thank you."}])
    result["status"] = "passed"
    atomic_json(Path(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
