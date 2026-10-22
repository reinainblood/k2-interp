"""Protocol-neutral evaluation summaries; task-specific graders stay explicit."""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import atomic_json, file_hash


def summarize_run(run: str | Path, output: str | Path):
    root = Path(run)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    missing = []
    for sample in manifest["spec"]["samples"]:
        receipt_path = root / "samples" / sample["id"] / "complete.json"
        if not receipt_path.exists():
            missing.append(sample["id"])
            continue
        receipt = json.loads(receipt_path.read_text())
        result_path = root / receipt["attempt"] / "result.json"
        if file_hash(result_path) != receipt["files"][str(result_path.relative_to(root))]:
            raise ValueError("Evaluation result hash mismatch")
        result = json.loads(result_path.read_text())
        row = {
            "sample_id": sample["id"],
            "labels": sample["labels"],
            "split": sample["split"],
            "template_group": sample["template_group"],
            "mode": result["mode"],
            "finish_reason": result["finish_reason"],
            "output_tokens": len(result["output_ids"]),
            "continuation_logprob_sum": result["continuation_logprob_sum"],
            "continuation_logprob_mean": result["continuation_logprob_mean"],
        }
        if "expected_exact" in sample["labels"]:
            final = result["output_text"].split("</think>", 1)[-1].strip()
            row["exact_match"] = final == str(sample["labels"]["expected_exact"]).strip()
        rows.append(row)
    summary = {
        "schema": "k2-interp/evaluation/1",
        "expected_samples": len(manifest["spec"]["samples"]),
        "completed_samples": len(rows),
        "missing_sample_ids": missing,
        "rows": rows,
        "source_manifest_sha256": file_hash(root / "manifest.json"),
        "grading": "only exact_match when explicitly requested by labels.expected_exact",
    }
    exact = [row["exact_match"] for row in rows if "exact_match" in row]
    if exact:
        summary["exact_match_accuracy"] = sum(exact) / len(exact)
        summary["exact_match_denominator"] = len(exact)
    atomic_json(Path(output), summary)
    return summary
