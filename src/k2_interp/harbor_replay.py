"""Turn a Harbor job of K2Agent trials into teacher-forced capture requests.

Pass A (Harbor + :mod:`k2_interp.harbor`) produces behavior: trajectories, token
identity, and a verifier label per trial. Pass B replays each frozen trajectory
through a checkpoint with capture on, so internals come from exactly the tokens
the model produced. Because every checkpoint replays the same token ids, the
captures align across checkpoints and can feed crosscoders.

Three capture stages keep storage proportional to what each hypothesis reads:

    b1   router summary fields at every MoE layer, every position
    b2   residual sites at every MoE layer, windowed positions only
    b3   residual sites at a chosen layer set, every position

Each stage becomes one ``RunSpec`` per checkpoint with one ``Sample`` per trial.
The sample carries the Harbor verifier outcome in ``labels`` so datasets keep
the behavioral label beside the activations.
"""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import atomic_json
from .registry import CHECKPOINTS
from .schema import ROUTER_SUMMARY_FIELDS, CaptureSpec, GenerationSpec, RunSpec, Sample

MOE_LAYERS = list(range(3, 61))
FROZEN = Path("agent") / "k2trace" / "frozen.json"
TRAJECTORY = Path("agent") / "k2trace" / "trajectory.jsonl"


# A schema-valid submission that does no work: AstroAgentBench's "valid-but-zero"
# class. Any of these metrics at exactly zero marks the trajectory as `empty`.
EMPTY_METRICS = ("coverage_ratio", "CR", "n_tracks", "computed_selected", "service_fraction")


def outcome_label(report: dict | None) -> str:
    """Collapse Harbor's verifier report to the study classes.

    bailed  no solution file           wrong   rejected by the verifier, or crashed
    empty   valid but does nothing     pushed  valid with nonzero work
    """
    if report is None:
        return "unscored"
    status = report.get("status")
    if status == "missing_solution":
        return "bailed"
    if report.get("valid") is not True:
        return "wrong"
    metrics = report.get("metrics") or {}
    if any(metrics.get(name) == 0 for name in EMPTY_METRICS):
        return "empty"
    return "pushed"


def harbor_split(split: str | None) -> str:
    """Map a benchmark split name onto k2-interp's dataset split vocabulary."""
    if not split:
        return "exploratory"
    if split.startswith("train"):
        return "train"
    if split.startswith("test"):
        return "test"
    if split.startswith("valid"):
        return "validation"
    return "exploratory"


def find_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> int:
    """Index of `needle` inside `haystack` at or after `start`, or -1."""
    n = len(needle)
    if n == 0:
        return -1
    first = needle[0]
    i = start
    end = len(haystack) - n
    while i <= end:
        if haystack[i] == first and haystack[i : i + n] == needle:
            return i
        i += 1
    return -1


MIN_SEARCH_TOKENS = 8


def turn_spans(frozen: dict) -> list[dict]:
    """Locate each turn's completion inside the final-turn sequence.

    The final turn's span is exact by construction: its completion follows the
    final prompt. Earlier assistant turns are re-rendered by the chat template when
    they become history, so their tokens are searched for inside the final prompt,
    in order, and only when long enough that a chance match is implausible. A turn
    that cannot be located is reported with span None and is not used for windows.
    """
    sequence = frozen["final_sequence"]
    prompt_length = frozen.get("final_prompt_length", len(sequence))
    turns = frozen["turns"]
    spans, cursor = [], 0
    for index, turn in enumerate(turns):
        ids = turn["completion_token_ids"]
        if index == len(turns) - 1:
            spans.append({"turn": turn["turn"], "start": prompt_length, "end": len(sequence)})
            continue
        at = -1
        if len(ids) >= MIN_SEARCH_TOKENS:
            at = find_subsequence(sequence[:prompt_length], ids, cursor)
        if at >= 0:
            spans.append({"turn": turn["turn"], "start": at, "end": at + len(ids)})
            cursor = at + len(ids)
        else:
            spans.append({"turn": turn["turn"], "start": None, "end": None})
    return spans


def windows(frozen: dict, size: int = 256) -> dict[str, list[int]]:
    """Absolute positions of the impasse, early, and (if available) midpoint windows."""
    sequence = frozen["final_sequence"]
    spans = [s for s in turn_spans(frozen) if s["start"] is not None]
    out: dict[str, list[int]] = {}
    if not spans:
        return out
    last = spans[-1]
    out["impasse"] = list(range(max(last["start"], last["end"] - size), last["end"]))
    first = spans[0]
    early_start = first["start"] + size
    early_end = min(first["end"], early_start + size)
    if early_end > early_start:
        out["early"] = list(range(early_start, early_end))
    mid = len(sequence) // 2
    out["midpoint"] = list(range(max(0, mid - size // 2), min(len(sequence), mid + size // 2)))
    return out


def load_trials(job: str | Path) -> list[dict]:
    """Every trial in a Harbor job directory that carries a frozen trajectory."""
    job = Path(job)
    trials = []
    for result_path in sorted(job.glob("*/result.json")):
        trial_dir = result_path.parent
        frozen_path = trial_dir / FROZEN
        if not frozen_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        frozen = json.loads(frozen_path.read_text())
        report_path = trial_dir / "verifier" / "report.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else None
        config_path = trial_dir / "verifier" / "task-config.json"
        metadata = (
            json.loads(config_path.read_text()).get("metadata", {}) if config_path.is_file() else {}
        )
        if not frozen.get("final_sequence"):
            continue
        trials.append(
            {
                "trial_name": result["trial_name"],
                "trial_dir": str(trial_dir),
                "task_name": result.get("task_name"),
                "benchmark": metadata.get("benchmark"),
                "split": metadata.get("split"),
                "case_id": metadata.get("case_id"),
                "checkpoint": (result.get("agent_info") or {}).get("model_info") or {},
                "outcome": outcome_label(report),
                "verifier": report,
                "frozen": frozen,
                "turns": len(frozen["turns"]),
                "length": len(frozen["final_sequence"]),
            }
        )
    if not trials:
        raise ValueError(f"No trials with k2trace/frozen.json under {job}")
    return trials


def trial_sample(trial: dict) -> Sample:
    sequence = trial["frozen"]["final_sequence"]
    labels = {
        "outcome": trial["outcome"],
        "benchmark": trial["benchmark"] or "",
        "split": trial["split"] or "",
        "case_id": trial["case_id"] or "",
        "turns": trial["turns"],
        "final_prompt_length": trial["frozen"]["final_prompt_length"],
    }
    if trial["verifier"]:
        labels["verifier_status"] = str(trial["verifier"].get("status"))
        labels["valid"] = bool(trial["verifier"].get("valid"))
    return Sample(
        id=trial["trial_name"],
        # Prefill captures every prompt row in one forward; the final token is the
        # single teacher-forced step. All positions are processed exactly once.
        input_ids=sequence[:-1],
        continuation_ids=sequence[-1:],
        labels=labels,
        template_group=f"{trial['benchmark']}-{trial['split']}",
        split=harbor_split(trial["split"]),
    )


def stage_capture(
    stage: str,
    *,
    positions: list[int] | None,
    layers: list[int] | None,
    sites: list[str],
    max_bytes: int,
) -> CaptureSpec:
    if stage == "b1":
        return CaptureSpec(
            router=True,
            router_fields=list(ROUTER_SUMMARY_FIELDS),
            layers=[],
            sites=[],
            positions=positions,
            max_bytes=max_bytes,
        )
    if stage == "b2":
        return CaptureSpec(
            router=False,
            layers=layers or MOE_LAYERS,
            sites=sites,
            positions=positions,
            max_bytes=max_bytes,
        )
    if stage == "b3":
        if not layers:
            raise ValueError(
                "stage b3 needs an explicit --layers set (the frozen signature layers)"
            )
        return CaptureSpec(
            router=False, layers=layers, sites=sites, positions=positions, max_bytes=max_bytes
        )
    raise ValueError("stage must be b1, b2, or b3")


def build_specs(
    job: str | Path,
    *,
    stage: str,
    checkpoints: list[str],
    run_prefix: str,
    window: int = 256,
    layers: list[int] | None = None,
    sites: list[str] | None = None,
    max_bytes: int = 2_000_000_000,
    max_seconds: int = 86400,
    max_context: int = 262144,
    per_trial: bool = False,
    outcomes: list[str] | None = None,
) -> list[RunSpec]:
    """One RunSpec per checkpoint (or per checkpoint and trial with per_trial)."""
    for checkpoint in checkpoints:
        if checkpoint not in CHECKPOINTS:
            raise ValueError(f"Unknown checkpoint {checkpoint!r}")
    sites = sites or ["resid_post", "moe_out"]
    trials = load_trials(job)
    if outcomes:
        trials = [t for t in trials if t["outcome"] in outcomes]
    if not trials:
        raise ValueError("No trials left after the outcome filter")
    # Capture positions are a RunSpec-level field, and b2 windows differ per
    # trajectory, so b2 is always one spec per trial.
    per_trial = per_trial or stage == "b2"
    specs = []
    for checkpoint in checkpoints:
        groups = [[t] for t in trials] if per_trial else [trials]
        for group in groups:
            samples, positions, window_index = [], None, None
            for trial in group:
                if trial["length"] > max_context:
                    raise ValueError(
                        f"{trial['trial_name']} has {trial['length']} tokens > max_context"
                    )
                samples.append(trial_sample(trial))
            if stage == "b2":
                found = windows(group[0]["frozen"], window)
                if not found:
                    raise ValueError(f"{group[0]['trial_name']}: no turn could be located")
                positions = sorted({p for ps in found.values() for p in ps})
                window_index = {k: [v[0], v[-1]] for k, v in found.items()}
            capture = stage_capture(
                stage, positions=positions, layers=layers, sites=sites, max_bytes=max_bytes
            )
            suffix = group[0]["trial_name"] if per_trial else "all"
            run_id = f"{run_prefix}-{stage}-{suffix}"[:120]
            specs.append(
                RunSpec(
                    run_id=run_id,
                    checkpoint=checkpoint,
                    samples=samples,
                    capture=capture,
                    generation=GenerationSpec(max_new_tokens=0, temperature=0, seed=0),
                    max_context=max_context,
                    max_seconds=max_seconds,
                    protocol={
                        "source": "harbor",
                        "job": str(Path(job).resolve()),
                        "stage": stage,
                        "window": window if stage == "b2" else None,
                        "windows": window_index,
                        "trials": [t["trial_name"] for t in group],
                    },
                )
            )
    return specs


def write_specs(specs: list[RunSpec], output: str | Path) -> list[Path]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in specs:
        path = output / f"{spec.checkpoint}--{spec.run_id}.json"
        atomic_json(path, spec.model_dump())
        paths.append(path)
    index = output / "index.json"
    atomic_json(
        index,
        {
            "specs": [
                {
                    "path": p.name,
                    "checkpoint": s.checkpoint,
                    "run_id": s.run_id,
                    "samples": len(s.samples),
                    "stage": s.protocol.get("stage"),
                }
                for p, s in zip(paths, specs, strict=True)
            ]
        },
    )
    return paths


def summarize(job: str | Path) -> dict:
    """Behavioral summary of a Harbor job, before any activation is read."""
    trials = load_trials(job)
    counts: dict[str, dict[str, int]] = {}
    for t in trials:
        counts.setdefault(t["benchmark"] or "?", {}).setdefault(t["outcome"], 0)
        counts[t["benchmark"] or "?"][t["outcome"]] += 1
    return {
        "job": str(Path(job).resolve()),
        "trials": len(trials),
        "by_benchmark": counts,
        "lengths": {
            "min": min(t["length"] for t in trials),
            "max": max(t["length"] for t in trials),
            "mean": sum(t["length"] for t in trials) / len(trials),
        },
        "turns": {
            "min": min(t["turns"] for t in trials),
            "max": max(t["turns"] for t in trials),
        },
    }
