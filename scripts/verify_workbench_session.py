"""Read-only receipt verification for a browser-created session in FDR."""

import argparse
import hashlib
import json
from pathlib import Path

import modal

from k2_interp.artifacts import atomic_json
from k2_interp.registry import CHECKPOINTS
from k2_interp.schema import identifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id")
    parser.add_argument("--output", default=".private/browser-workbench-verification.json")
    args = parser.parse_args()
    sessions = modal.Volume.from_name("k2-interp-workbench", environment_name="FDR")
    artifacts = modal.Volume.from_name("k2-interp-artifacts", environment_name="FDR")
    session = json.loads(b"".join(sessions.read_file(identifier(args.session_id) + ".json")))
    verified = []
    for turn in session["turns"]:
        base = session["checkpoint"] + "/" + turn["run_id"]
        manifest = json.loads(b"".join(artifacts.read_file(base + "/manifest.json")))
        receipt = json.loads(b"".join(artifacts.read_file(base + "/samples/chat/complete.json")))
        relative = receipt["attempt"] + "/result.json"
        body = b"".join(artifacts.read_file(base + "/" + relative))
        assert hashlib.sha256(body).hexdigest() == receipt["files"][relative]
        result = json.loads(body)
        assert manifest["identity"]["revision"] == CHECKPOINTS[session["checkpoint"]]
        assert manifest["spec"]["generation"] == turn["settings"]["generation"]
        assert (
            manifest["spec"]["samples"][0]["chat_template_kwargs"]["reasoning_effort"]
            == turn["settings"]["reasoning_effort"]
        )
        if "system" in turn:
            assert manifest["spec"]["samples"][0]["messages"][0]["content"] == turn["system"]
        verified.append(
            {
                "run_id": turn["run_id"],
                "status": receipt["status"],
                "input_tokens": len(result["input_ids"]),
                "output_tokens": len(result["output_ids"]),
                "finish_reason": result["finish_reason"],
                "answer_nonempty": bool(result["assistant_message"]["content"]),
                "reasoning_effort": turn["settings"]["reasoning_effort"],
                "system_and_settings_match": True,
                "result_hash_verified": True,
                "identity": manifest["identity"],
            }
        )
    record = {
        "session_id": session["id"],
        "turns": verified,
        "capture_count": len(session["captures"]),
    }
    atomic_json(Path(args.output), record)
    print(
        json.dumps(
            {
                **record,
                "turns": [{k: v for k, v in t.items() if k != "identity"} for t in verified],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
