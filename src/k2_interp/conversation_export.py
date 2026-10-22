"""Lossless conversation exports, independent of the live dashboard transport.

Only committed artifact files are trusted. The session is copied, never edited.
Missing output is reported as missing; partial reasoning is never reconstructed
from an answer or invented when only token IDs survived an interruption.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path

from .artifacts import file_hash
from .schema import identifier


def _json(path):
    return json.loads(path.read_text())


def _checked(run, relative, hashes):
    path = run / relative
    if not path.resolve().is_relative_to(run.resolve()):
        raise ValueError("Conversation artifact path escapes its run")
    if not path.is_file() or hashes.get(relative) != file_hash(path):
        raise ValueError(f"Conversation artifact failed verification: {relative}")
    return _json(path)


def collect_conversation(root: Path, checkpoint: str, session: dict, exported_at: str) -> dict:
    """Keep every turn and every assistant field, hydrating durable results and partials.

    `archive_complete` means every listed turn has a verified completed result.
    It does not mean full tensors, model weights, or a resumable KV cache are included.
    """
    result = copy.deepcopy(session)
    coverage = []
    for turn in result.get("turns", []):
        run_id = identifier(turn["run_id"])
        run = Path(root) / identifier(checkpoint) / run_id
        if not run.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("Conversation run path escapes artifact root")
        item = {"run_id": run_id, "status": "missing", "verified_files": {}}
        manifest_path = run / "manifest.json"
        if not manifest_path.exists():
            item["reason"] = "No committed run manifest; session entry preserved"
            coverage.append(item)
            continue
        manifest = _json(manifest_path)
        turn["run_manifest"] = manifest
        item["verified_files"]["manifest.json"] = file_hash(manifest_path)
        base = run / "samples" / "chat"
        receipt_path = base / "complete.json"
        if receipt_path.exists():
            receipt = _json(receipt_path)
            relative = str(Path(receipt["attempt"]) / "result.json")
            raw = _checked(run, relative, receipt["files"])
            turn["raw_result"] = raw
            turn["assistant"] = copy.deepcopy(raw["assistant_message"])
            turn["finish_reason"] = raw.get("finish_reason")
            turn["status"] = "complete"
            item.update(status="complete", attempt=receipt["attempt"])
            item["verified_files"][relative] = receipt["files"][relative]
        else:
            attempts = sorted(base.glob("attempt-*"), key=lambda p: p.stat().st_mtime)
            if not attempts or not (attempts[-1] / "progress.json").is_file():
                item["reason"] = "No committed forward checkpoint; session entry preserved"
                coverage.append(item)
                continue
            attempt = attempts[-1]
            progress = _json(attempt / "progress.json")
            turn["partial_generation"] = copy.deepcopy(progress.get("generation"))
            item.update(status="partial", attempt=str(attempt.relative_to(run)))
            item["verified_files"][str((attempt / "progress.json").relative_to(run))] = file_hash(
                attempt / "progress.json"
            )
            # Projection text is paired with the same committed generation step.
            step = (progress.get("generation") or {}).get("step")
            relative = (
                str((attempt / f"workbench-step-{step:06d}.json").relative_to(run))
                if isinstance(step, int)
                else None
            )
            if relative and relative in progress["files"]:
                projection = _checked(run, relative, progress["files"])
                turn["partial_output_text"] = projection.get("output_text")
                turn["partial_assistant"] = copy.deepcopy(projection.get("assistant_message"))
                item["verified_files"][relative] = progress["files"][relative]
            else:
                item["reason"] = "Committed token IDs retained; parsed partial text unavailable"
            failure_path = attempt / "failure.json"
            if failure_path.is_file():
                turn["failure"] = _json(failure_path)
        coverage.append(item)
    result["export"] = {
        "schema": "k2-conversation-export/1",
        "scope": "entire session at export request",
        "exported_at": exported_at,
        "turn_count": len(result.get("turns", [])),
        "archive_complete": all(item["status"] == "complete" for item in coverage),
        "includes_reasoning": True,
        "includes_raw_results": True,
        "includes_all_tensor_shards": False,
        "coverage": coverage,
    }
    return result


def conversation_markdown(conversation: dict) -> str:
    """Readable companion; JSON remains authoritative and preserves all unknown fields."""
    chunks = ["# Conversation", f"\nSession: {conversation.get('id', '')}"]

    def literal(label, value):
        if value is None:
            return
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        # Avoid accidental closing fences in model output or code conversations.
        fence = "`" * max(3, max((len(x) for x in re.findall(r"`+", text)), default=0) + 1)
        chunks.extend([f"\n### {label}\n", fence + "text\n" + text + "\n" + fence])

    literal("System", conversation.get("system"))
    for index, turn in enumerate(conversation.get("turns", []), 1):
        chunks.append(f"\n## Turn {index}\n\nRun: {turn['run_id']}\n\nStatus: {turn['status']}")
        literal("User", turn.get("user"))
        for label, message in (
            ("Assistant", turn.get("assistant")),
            ("Partial assistant", turn.get("partial_assistant")),
        ):
            if message is None:
                continue
            literal(label, message.get("content"))
            for key, value in message.items():
                if key not in ("role", "content"):
                    literal(f"{label}: {key}", value)
        literal("Raw generated output", turn.get("raw_result", {}).get("output_text"))
        literal("Partial raw output", turn.get("partial_output_text"))
        literal("Failure", turn.get("failure", turn.get("error")))
    chunks.append("\n## Export coverage\n")
    literal("Coverage", conversation["export"])
    return "\n".join(chunks) + "\n"


def conversation_bundle(root, checkpoint, session, exported_at):
    conversation = collect_conversation(root, checkpoint, session, exported_at)
    handle = tempfile.NamedTemporaryFile(prefix="k2-conversation-", suffix=".zip", delete=False)
    handle.close()
    path = Path(handle.name)
    try:
        contents = {
            "conversation.json": json.dumps(
                conversation, ensure_ascii=False, indent=2, allow_nan=False
            ).encode(),
            "conversation.md": conversation_markdown(conversation).encode(),
        }
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, body in contents.items():
                archive.writestr(name, body)
            archive.writestr(
                "export.json",
                json.dumps(
                    {
                        **conversation["export"],
                        "files": {
                            name: hashlib.sha256(body).hexdigest()
                            for name, body in contents.items()
                        },
                    },
                    indent=2,
                ),
            )
        return path, conversation["export"]
    except BaseException:
        path.unlink(missing_ok=True)
        raise
