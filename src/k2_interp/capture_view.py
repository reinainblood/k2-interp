"""Verified private trace inspection and bounded, immutable capture bundles."""

from __future__ import annotations

import copy
import json
import tempfile
import zipfile
from pathlib import Path

from safetensors import safe_open

from .artifacts import file_hash
from .capture_svg import router_svg
from .conversation_export import collect_conversation, conversation_markdown
from .schema import digest, identifier
from .telemetry import project_layer, utc_now


def read_json(path):
    return json.loads(Path(path).read_text())


def trace_attempt(root: Path, checkpoint: str, run_id: str):
    run = root / identifier(checkpoint) / identifier(run_id)
    if not run.resolve().is_relative_to(root.resolve()):
        raise ValueError("Trace path escapes artifact root")
    manifest = read_json(run / "manifest.json")
    if (
        digest({"spec": manifest["spec"], "identity": manifest["identity"]})
        != manifest["request_sha256"]
    ):
        raise ValueError("Run manifest hash mismatch")
    base = run / "samples" / "chat"
    if (base / "complete.json").exists():
        receipt = read_json(base / "complete.json")
        attempt = run / receipt["attempt"]
        files = receipt["files"]
        complete = True
    else:
        attempts = sorted(base.glob("attempt-*"), key=lambda p: p.stat().st_mtime)
        if not attempts:
            raise FileNotFoundError("No committed trace yet")
        attempt = attempts[-1]
        files = read_json(attempt / "progress.json")["files"]
        complete = False
    if not attempt.resolve().is_relative_to(run.resolve()):
        raise ValueError("Invalid attempt path")
    return run, attempt, manifest, files, complete


def verified(run, path, files):
    if not path.resolve().is_relative_to(run.resolve()):
        raise ValueError("Artifact path escapes run")
    relative = str(path.relative_to(run))
    if files.get(relative) != file_hash(path):
        raise ValueError(f"Unverified capture artifact: {relative}")


def inspect_token(root, checkpoint, run_id, position):
    run, attempt, manifest, files, complete = trace_attempt(root, checkpoint, run_id)
    layers, step, phase, token_id = {}, None, None, None
    for path in sorted(attempt.glob("step-*.json")):
        if str(path.relative_to(run)) not in files:
            continue
        verified(run, path, files)
        meta = read_json(path)
        records = [r for r in meta.get("records", [meta]) if position in r["positions"]]
        if not records:
            continue
        tensor_path = attempt / meta["tensor_file"]
        verified(run, tensor_path, files)
        with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
            for record in records:
                row = record["positions"].index(position)
                selected = {}
                for key in record["keys"]:
                    if key != "positions":
                        # Slice exactly one token; never materialize the whole prompt for UI inspection.
                        selected[key] = tensors.get_slice(key)[row : row + 1]
                layer = str(record["layer"])
                projection = project_layer(record["layer"], selected, 0)
                # Aggregate metrics from a single-row slice are explicitly absent.
                projection.pop("forward_expert_counts", None)
                projection.pop("forward_expert_mass", None)
                entry = layers.setdefault(layer, {})
                for key, value in projection.items():
                    if key in ("norms", "rms", "expert_norms"):
                        entry.setdefault(key, {}).update(value)
                    else:
                        entry[key] = value
                step, phase = record["step"], record["phase"]
                token_id = record["token_ids"][row] if record.get("token_ids") else None
    if not layers:
        raise FileNotFoundError("Token is not in a committed capture")
    return {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "position": position,
        "token_id": token_id,
        "step": step,
        "phase": phase,
        "layers": layers,
        "projection": "selected input token",
        "forward_complete": True,
        "trace_complete": complete,
        "identity": manifest["identity"],
    }


def latest_projection(root, checkpoint, run_id):
    run, attempt, manifest, files, complete = trace_attempt(root, checkpoint, run_id)
    paths = [
        p for p in sorted(attempt.glob("workbench-step-*.json")) if str(p.relative_to(run)) in files
    ]
    if not paths:
        raise FileNotFoundError("No committed projection yet")
    verified(run, paths[-1], files)
    state = read_json(paths[-1])
    if complete:
        verified(run, attempt / "result.json", files)
        result = read_json(attempt / "result.json")
        state.update(
            status="complete",
            result={"result": result, "assistant_message": result["assistant_message"]},
        )
    elif (attempt / "failure.json").exists():
        state.update(status="failed", error=read_json(attempt / "failure.json"))
    state["recovered_from_artifacts"] = True
    return state


def capture_bundle(root, checkpoint, run_id, position, layer, session, tag=""):
    """Capture one source forward and preserve the entire chat available at export time.

    During prefill, one forward contains the entire prompt. That is recorded, never
    misrepresented as a sequential time slice inside parallel prefill computation.
    """
    view = inspect_token(root, checkpoint, run_id, position)
    run, attempt, manifest, files, complete = trace_attempt(root, checkpoint, run_id)
    created = utc_now()
    conversation = collect_conversation(root, checkpoint, session, created)
    stamp = created.replace(":", "").replace("-", "").replace("+0000", "Z")
    capture_id = f"{stamp}-p{position}-l{layer}"
    chosen = []
    forward_state = None
    for relative in files:
        path = run / relative
        if path.name.startswith(f"step-{view['step']:06d}-"):
            chosen.append(path)
        if path.name == f"workbench-step-{view['step']:06d}.json":
            chosen.append(path)
            verified(run, path, files)
            forward_state = read_json(path)
    session = copy.deepcopy(session)
    turn_index = next(
        (i for i, t in enumerate(session.get("turns", [])) if t["run_id"] == run_id), None
    )
    if turn_index is not None:
        session["turns"] = session["turns"][: turn_index + 1]
    if session.get("turns") and forward_state:
        session["turns"][-1]["assistant"] = forward_state.get("assistant_message")
        session["turns"][-1]["status"] = "captured_forward"
    # The exact rendered history uses the system prompt at dispatch, even after later edits.
    session["model_messages"] = manifest["spec"]["samples"][0].get("messages")
    if session["model_messages"]:
        session["system"] = next(
            (m["content"] for m in session["model_messages"] if m["role"] == "system"), ""
        )
    # Freeze the exact selection and current chat prefix into the export.
    index = {
        "schema": "k2-workbench-capture/1",
        "capture_id": capture_id,
        "tag": tag,
        "created_at": created,
        "checkpoint": checkpoint,
        "run_id": run_id,
        "position": position,
        "layer": layer,
        "step": view["step"],
        "phase": view["phase"],
        "trace_complete": complete,
        "source_captured_at": forward_state.get("updated_at") if forward_state else None,
        "captured_output_tokens": len(forward_state.get("output_ids", []))
        if forward_state
        else None,
        "scope": "source forward; all captured layers and positions",
        "conversation_scope": "entire session at export request",
        "conversation_archive_complete": conversation["export"]["archive_complete"],
        "capture_alignment": "input position i processes token i and predicts token i+1",
        "files": {},
    }
    handle = tempfile.NamedTemporaryFile(prefix="k2-capture-", suffix=".zip", delete=False)
    handle.close()
    path = Path(handle.name)
    try:
        with zipfile.ZipFile(
            path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            for source in chosen:
                verified(run, source, files)
                relative = str(source.relative_to(run))
                archive.write(source, "trace/" + relative)
                index["files"]["trace/" + relative] = files[relative]
            for name, value in {
                "run-manifest.json": manifest,
                "selection.json": view,
                "conversation.json": conversation,
                "capture-state.json": session,
                "capture-context.json": {
                    "scope": "exact input messages for selected run",
                    "messages": manifest["spec"]["samples"][0].get("messages"),
                },
            }.items():
                from hashlib import sha256

                body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()
                archive.writestr(name, body)
                index["files"][name] = sha256(body).hexdigest()
            body = conversation_markdown(conversation).encode()
            archive.writestr("conversation.md", body)
            index["files"]["conversation.md"] = sha256(body).hexdigest()
            body = router_svg(view)
            archive.writestr("router.svg", body)
            index["files"]["router.svg"] = sha256(body).hexdigest()
            archive.writestr("capture.json", json.dumps(index, indent=2, allow_nan=False))
        return path, index
    except BaseException:
        path.unlink(missing_ok=True)
        raise
