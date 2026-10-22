"""Bounded full-weight web/SSE/capture validation. All conversations and receipts stay private."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import modal
from open_workbench import login_ticket
from verify_export import verify

from k2_interp.artifacts import atomic_json
from k2_interp.telemetry import utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--output", default=".private/workbench-verification.json")
    args = parser.parse_args()
    url = (
        args.url
        or modal.Function.from_name(
            "k2-interp-workbench", "web", environment_name="FDR"
        ).get_web_url()
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"started_at": utc_now(), "url": url, "turns": [], "status": "running"}
    with httpx.Client(base_url=url, timeout=120, headers={"X-K2-Intent": "workbench"}) as client:
        assert client.get("/api/config").status_code == 401
        assert client.get("/").status_code == 200
        response = client.post("/api/login", json={"ticket": login_ticket()})
        response.raise_for_status()
        assert "HttpOnly" in response.headers["set-cookie"]
        assert "Secure" in response.headers["set-cookie"]
        assert client.get("/api/config").status_code == 200
        response = client.post("/api/sessions", json={"system": "Answer briefly and accurately."})
        response.raise_for_status()
        session = response.json()
        receipt["session_id"] = session["id"]
        atomic_json(output, receipt)
        print(json.dumps({"session_id": session["id"], "status": "authenticated"}), flush=True)
        for i, (text, effort) in enumerate(
            [
                ("What is two plus two? Answer briefly.", "medium"),
                ("And what is three plus three?", "low"),
            ]
        ):
            if i:
                response = client.post(
                    f"/api/sessions/{session['id']}/settings",
                    json={"system": "Answer with the number only. Be accurate."},
                )
                response.raise_for_status()
            response = client.post(
                f"/api/sessions/{session['id']}/turns",
                json={
                    "text": text,
                    "request_id": uuid.uuid4().hex,
                    "generation": {"max_new_tokens": 128, "temperature": 0, "seed": 19},
                    "reasoning_effort": effort,
                    "layers": [10, 30, 50],
                    "expert_norms": True,
                },
            )
            response.raise_for_status()
            turn = response.json()
            prefix = f"/api/sessions/{session['id']}/turns/{turn['run_id']}"
            deadline = time.monotonic() + 1800
            phases, steps, frames, last, log_time = set(), set(), 0, None, 0
            partial_layers = False
            while time.monotonic() < deadline:
                with client.stream("GET", prefix + "/events") as stream:
                    stream.raise_for_status()
                    for line in stream.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        last = json.loads(line[6:])
                        frames += 1
                        phases.add(last["status"])
                        if last.get("step") is not None:
                            steps.add(last["step"])
                        count = len(
                            [
                                row
                                for row in last.get("layers", {}).values()
                                if row.get("expert_ids")
                            ]
                        )
                        partial_layers |= 0 < count < 58
                        if time.monotonic() - log_time > 25:
                            print(
                                json.dumps(
                                    {
                                        "turn": i,
                                        "status": last["status"],
                                        "frames": frames,
                                        "step": last.get("step"),
                                        "router_layers": count,
                                    }
                                ),
                                flush=True,
                            )
                            log_time = time.monotonic()
                        if last["status"] in ("complete", "failed", "cancelled"):
                            break
                if last and last["status"] in ("complete", "failed", "cancelled"):
                    break
            if not last or last["status"] != "complete":
                receipt.update(status="failed", last_state=last)
                atomic_json(output, receipt)
                raise RuntimeError(
                    f"Full-weight turn incomplete: {last.get('error') if last else 'timeout'}"
                )
            assert len(last["layers"]) == 58
            assert all(len(row["expert_ids"]) == 8 for row in last["layers"].values())
            assert all(
                abs(sum(row["expert_weights"]) - 2.5) < 1e-5 for row in last["layers"].values()
            )
            for layer in ("10", "30", "50"):
                assert last["layers"][layer]["norms"]["resid_post"] > 0
                assert last["layers"][layer]["expert_norms"]
            position = last["position"]
            response = client.get(prefix + f"/tokens/{position}")
            response.raise_for_status()
            view = response.json()
            for layer in last["layers"]:
                assert view["layers"][layer]["expert_ids"] == last["layers"][layer]["expert_ids"]
                assert (
                    view["layers"][layer]["expert_weights"]
                    == last["layers"][layer]["expert_weights"]
                )
            response = client.post(
                prefix + "/captures",
                json={"position": position, "layer": 30, "tag": f"validation-turn-{i + 1}"},
            )
            response.raise_for_status()
            archive = output.parent / f"workbench-capture-{i + 1}.zip"
            archive.write_bytes(response.content)
            checked = verify(archive)
            with zipfile.ZipFile(archive) as zipped:
                assert "router.svg" in zipped.namelist()
                assert json.loads(zipped.read("capture-context.json"))["messages"][0][
                    "content"
                ] == (
                    "Answer briefly and accurately."
                    if i == 0
                    else "Answer with the number only. Be accurate."
                )
            result = {
                "run_id": turn["run_id"],
                "frames": frames,
                "phases": sorted(phases),
                "observed_steps": sorted(steps),
                "observed_partial_layer_frame": partial_layers,
                "identity": last["identity"],
                "final_state": last,
                "router_layers": len(view["layers"]),
                "archive": str(archive),
                "archive_sha256": hashlib.sha256(response.content).hexdigest(),
                "archive_verification": checked,
                "reasoning_effort": effort,
            }
            receipt["turns"].append(result)
            atomic_json(output, receipt)
            print(
                json.dumps(
                    {
                        "turn": i,
                        "status": "passed",
                        "frames": frames,
                        "router_layers": 58,
                        "output_tokens": len(last["output_ids"]),
                        "archive_verified": True,
                    }
                ),
                flush=True,
            )
        response = client.get(f"/api/sessions/{session['id']}/export")
        response.raise_for_status()
        path = output.parent / "workbench-conversation.zip"
        path.write_bytes(response.content)
        receipt["conversation_verification"] = verify(path)
        receipt.update(status="passed", finished_at=utc_now())
        atomic_json(output, receipt)
        print(
            json.dumps({"status": "passed", "receipt": str(output), "session_id": session["id"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
