"""Two real chat turns through existing Modal SDK auth; results remain private."""

import argparse
import json
import uuid
from pathlib import Path

from k2_interp.artifacts import atomic_json
from k2_interp.cli import remote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="sft_3_1500")
    parser.add_argument("--output", default=".private/sdk-chat-verification.json")
    args = parser.parse_args()
    worker = remote(args.checkpoint)
    messages = [{"role": "user", "content": "What is two plus two? Answer briefly."}]
    turns = []
    for i in range(2):
        response = worker.chat.remote(
            {
                "messages": messages,
                "run_id": "sdk-check-" + uuid.uuid4().hex,
                "max_new_tokens": 128,
                "seed": 19,
                "temperature": 0.0,
            }
        )
        message = response["assistant_message"]
        assert isinstance(message.get("reasoning_content"), str)
        assert response["result"]["output_ids"]
        turns.append(
            {
                "turn": i,
                "output_tokens": len(response["result"]["output_ids"]),
                "finish_reason": response["result"]["finish_reason"],
                "run_id": response["receipt"]["run_id"],
                "message": message,
            }
        )
        messages.extend([message, {"role": "user", "content": "And what is three plus three?"}])
    result = {"status": "passed", "checkpoint": args.checkpoint, "turns": turns}
    atomic_json(Path(args.output), result)
    print(
        json.dumps(
            {
                "status": "passed",
                "checkpoint": args.checkpoint,
                "turns": [{k: v for k, v in t.items() if k != "message"} for t in turns],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
