"""Turn an Inspect ``.eval`` log run with the ``k2interp`` provider into capture RunSpecs.

Each logged sample is paired with the frozen trajectory the provider wrote for it
and with its score, converted to the trial shape :func:`k2_interp.harbor_replay.
load_trials` produces, and handed to the unchanged :func:`~k2_interp.harbor_replay.
build_specs`. Stages, windows, and teacher forcing are therefore identical to
Harbor replay.

Scores are read generically so real task sets plug in without code changes:

* ``outcome`` comes from the score value: ``C``/true/1 -> ``solved``,
  ``I``/false/0 -> ``wrong``, ``P`` or a fraction -> ``partial``, ``N`` ->
  ``bailed``, no score -> ``unscored``.
* ``first_failing_subproblem`` (score metadata) selects the ``commit`` window.
  Its answering turn comes from ``subproblem_turns[id]``, else the ``turn`` of the
  matching entry in ``subproblems``, else (assuming one assistant turn per
  subproblem) that entry's position in ``subproblems`` plus one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .harbor_replay import build_specs, repeated_ngram_ratio
from .inspect_provider import STORE_KEY, safe_name, trace_path

LOOP_NGRAM_RATIO = 0.5
LABEL_TYPES = (str, int, float, bool)


def score_outcome(value: Any) -> str:
    """Collapse an Inspect score value onto the study's outcome vocabulary."""
    if value is None:
        return "unscored"
    if isinstance(value, bool):
        return "solved" if value else "wrong"
    if isinstance(value, str):
        return {"C": "solved", "I": "wrong", "P": "partial", "N": "bailed"}.get(
            value.upper(), "unscored"
        )
    if isinstance(value, int | float):
        if value >= 1:
            return "solved"
        return "wrong" if value <= 0 else "partial"
    return "unscored"  # dict/list values: pick a scalar scorer with --scorer


def pick_score(sample, scorer: str | None):
    scores = sample.scores or {}
    if scorer:
        if scorer not in scores and scores:
            raise ValueError(f"sample {sample.id}: no score named {scorer!r} ({sorted(scores)})")
        return scorer, scores.get(scorer)
    if not scores:
        return None, None
    name = next(iter(scores))
    return name, scores[name]


def commit_point(metadata: dict | None) -> dict | None:
    """``{subproblem, turn, turn_source}`` for the first failing subproblem, or None."""
    metadata = metadata or {}
    failing = metadata.get("first_failing_subproblem")
    if failing is None:
        return None
    turns = metadata.get("subproblem_turns") or {}
    if str(failing) in {str(k) for k in turns}:
        turn = next(v for k, v in turns.items() if str(k) == str(failing))
        return {"subproblem": str(failing), "turn": turn, "turn_source": "subproblem_turns"}
    subs = metadata.get("subproblems") or []
    for index, sub in enumerate(subs):
        sid = sub.get("id") if isinstance(sub, dict) else sub
        if str(sid) == str(failing):
            if isinstance(sub, dict) and sub.get("turn") is not None:
                return {"subproblem": str(failing), "turn": sub["turn"], "turn_source": "turn"}
            return {"subproblem": str(failing), "turn": index + 1, "turn_source": "position"}
    return {"subproblem": str(failing), "turn": None, "turn_source": "unresolved"}


def find_frozen(sample, log, trace_dir: str | Path | None) -> Path | None:
    """Frozen path recorded in the sample store, else the default layout under trace_dir."""
    record = (sample.store or {}).get(STORE_KEY) or {}
    recorded = record.get("frozen")
    if recorded and Path(recorded).is_file():
        return Path(recorded)
    root = trace_dir or (log.eval.model_args or {}).get("trace_dir")
    if root:
        candidate = trace_path(root, log.eval.eval_id, sample.id, sample.epoch) / "frozen.json"
        if candidate.is_file():
            return candidate
    return None


def sample_stop(sample) -> str:
    if sample.error is not None:
        return "error"
    if sample.limit is not None:
        return f"limit_{sample.limit.type}"
    return "completed"


def load_inspect_trials(
    log_path: str | Path, *, scorer: str | None = None, trace_dir: str | Path | None = None
) -> tuple[list[dict], dict]:
    """Trials in build_specs' shape, plus a summary of what was skipped and why."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path))
    task = log.eval.task
    trials, skipped = [], []
    for sample in log.samples or []:
        frozen_path = find_frozen(sample, log, trace_dir)
        if frozen_path is None:
            skipped.append({"sample": sample.id, "epoch": sample.epoch, "reason": "no frozen.json"})
            continue
        frozen = json.loads(frozen_path.read_text())
        if not frozen.get("final_sequence"):
            skipped.append({"sample": sample.id, "epoch": sample.epoch, "reason": "empty trace"})
            continue
        score_name, score = pick_score(sample, scorer)
        meta = dict(sample.metadata or {})
        score_meta = dict((score.metadata if score else None) or {})
        commit = commit_point(score_meta)
        final = frozen["final_sequence"][frozen.get("final_prompt_length", 0) :]
        ratio = repeated_ngram_ratio(final)
        extra = {
            "source": "inspect",
            "sample_id": str(sample.id),
            "epoch": sample.epoch,
            "score": score.value if score and isinstance(score.value, LABEL_TYPES) else "",
            "scorer": score_name or "",
            "first_failing_subproblem": str(score_meta.get("first_failing_subproblem") or ""),
        }
        for key in ("variant", "pair_id", "convention_stated", "synthetic"):
            if isinstance(meta.get(key), LABEL_TYPES):
                extra[key] = meta[key]
        for key in ("passed", "total", "all_passed"):
            if isinstance(score_meta.get(key), LABEL_TYPES):
                extra[f"score_{key}"] = score_meta[key]
        trials.append(
            {
                "trial_name": f"{safe_name(sample.id)}-e{sample.epoch}"[:100],
                "trial_dir": str(frozen_path.parent),
                "stop": sample_stop(sample),
                "length_hits": sum(
                    1 for t in frozen["turns"] if t.get("finish_reason") == "length"
                ),
                "loop": {
                    "repeat_calls": 0,  # no tool calls under the text protocol
                    "repeat_ngram_ratio": round(ratio, 4),
                    "looped": ratio >= LOOP_NGRAM_RATIO,
                },
                "task_name": task,
                "benchmark": meta.get("benchmark") or task,
                "split": meta.get("split"),
                # Twins (e.g. trap/control) share a template group so they never straddle splits.
                "case_id": str(meta.get("pair_id") or meta.get("case_id") or sample.id),
                "checkpoint": {"name": log.eval.model},
                "outcome": score_outcome(score.value if score else None),
                "verifier": None,
                "frozen": frozen,
                "frozen_path": str(frozen_path),
                "turns": len(frozen["turns"]),
                "length": len(frozen["final_sequence"]),
                "extra_labels": extra,
                "commit": commit,
                "commit_turn": commit["turn"] if commit else None,
            }
        )
    summary = {
        "log": str(Path(log_path).resolve()),
        "eval_id": log.eval.eval_id,
        "task": task,
        "model": log.eval.model,
        "samples": len(log.samples or []),
        "trials": len(trials),
        "skipped": skipped,
    }
    if not trials:
        raise ValueError(f"No samples with a k2interp frozen trace in {log_path}: {skipped}")
    return trials, summary


def build_inspect_specs(
    log_path: str | Path,
    *,
    stage: str,
    checkpoints: list[str],
    run_prefix: str,
    scorer: str | None = None,
    trace_dir: str | Path | None = None,
    **kwargs: Any,
):
    trials, summary = load_inspect_trials(log_path, scorer=scorer, trace_dir=trace_dir)
    specs = build_specs(
        log_path,
        stage=stage,
        checkpoints=checkpoints,
        run_prefix=run_prefix,
        trials=trials,
        source="inspect",
        protocol_extra={
            "inspect": {
                "eval_id": summary["eval_id"],
                "task": summary["task"],
                "model": summary["model"],
                "scorer": scorer,
            },
            "window_kinds": ["impasse", "early", "midpoint", "commit"] if stage == "b2" else None,
        },
        **kwargs,
    )
    return specs, summary
