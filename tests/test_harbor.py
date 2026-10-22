"""Harbor adapter loop, frozen trajectories, replay spec generation, router features."""

import asyncio
import json
from pathlib import Path

import pytest

from k2_interp.artifacts import iter_records
from k2_interp.harbor import (
    AgentLoop,
    ExecOutcome,
    FrozenTrajectory,
    LoopSettings,
    chat_payload,
    extract_command,
    is_submit,
    truncate,
)
from k2_interp.harbor_replay import (
    build_specs,
    find_subsequence,
    harbor_split,
    load_trials,
    outcome_label,
    summarize,
    turn_spans,
    windows,
    write_specs,
)
from k2_interp.schema import ROUTER_SUMMARY_FIELDS, CaptureSpec, ChatRequest, RunSpec, Sample


def test_protocol_parsing():
    assert extract_command("look\n```bash\nls -la\n```\nmore") == "ls -la"
    assert extract_command("```sh\necho hi\n```") == "echo hi"
    assert extract_command("no block here") is None
    assert extract_command("```python\nprint(1)\n```") is None
    assert is_submit("done\nSUBMIT\n")
    assert is_submit("  SUBMIT  ")
    assert not is_submit("I will SUBMIT later")
    assert truncate("x" * 100, 50).count("truncated") == 1
    assert truncate(None, 10) == ""


def test_chat_payload_capture_fields():
    plain = chat_payload(
        [{"role": "user", "content": "hi"}],
        checkpoint="sft_3_1500",
        max_tokens=8,
        temperature=0,
        top_p=1,
        seed=7,
        max_context=64,
        reasoning_effort="high",
        capture_router=False,
    )
    assert "capture" not in plain
    captured = chat_payload(
        [{"role": "user", "content": "hi"}],
        checkpoint="sft_3_1500",
        max_tokens=8,
        temperature=0,
        top_p=1,
        seed=7,
        max_context=64,
        reasoning_effort="high",
        capture_router=True,
    )
    request = ChatRequest.model_validate(captured)
    assert request.capture is not None
    assert request.capture.router_fields == list(ROUTER_SUMMARY_FIELDS)


def test_capture_spec_router_fields_validation():
    CaptureSpec(router_fields=["expert_ids", "router_entropy"], sites=[])
    with pytest.raises(ValueError, match="router_fields"):
        CaptureSpec(router_fields=[])
    with pytest.raises(ValueError, match="router_fields"):
        CaptureSpec(router_fields=["expert_ids", "expert_ids"])
    with pytest.raises(ValueError, match="router_fields"):
        CaptureSpec(router_fields=["not_a_field"])


class FakeServer:
    """Scripted assistant turns; echoes exact token identity like the HTTP endpoint."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        content = self.turns.pop(0)
        n = len(self.calls)
        prompt = list(range(100 * n, 100 * n + 5 + n))
        completion = [1000 + n, 1001 + n]
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": f"thinking {n}",
                    },
                    "finish_reason": "eos",
                }
            ],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(completion)},
            "k2_interp": {
                "revision": "abc",
                "run_id": f"chat-{n}",
                "captured": False,
                "input_ids": prompt,
                "output_ids": completion,
                "output_logprobs": [-0.1, -0.2],
                "finish_reason": "eos",
                "mode": "generation",
            },
        }


def run_loop(tmp_path, turns, commands=None, settings=None):
    seen = []

    async def execute(command):
        seen.append(command)
        if commands:
            return commands.pop(0)
        return ExecOutcome("ok\n", "", 0)

    server = FakeServer(turns)
    loop = AgentLoop(server, execute, tmp_path / "k2trace", settings)
    result = asyncio.run(loop.run("Solve the case."))
    return result, loop, server, seen


def test_loop_runs_commands_then_submits(tmp_path):
    result, loop, server, seen = run_loop(
        tmp_path,
        ["```bash\nls\n```", "```bash\ncat x\n```", "Cannot continue.\nSUBMIT"],
        [ExecOutcome("a b c", "", 0), ExecOutcome("", "no such file", 1)],
    )
    assert result.stop == "submitted"
    assert result.turns == 3
    assert seen == ["ls", "cat x"]
    # The history sent back carries the server's rendered assistant message intact.
    third = server.calls[2]
    assert third[2]["reasoning_content"] == "thinking 1"
    assert third[3]["content"].startswith("[exit 0]")
    assert "[stderr]" in third[5]["content"]
    records = [
        json.loads(line)
        for line in (tmp_path / "k2trace/trajectory.jsonl").read_text().splitlines()
    ]
    roles = [r["role"] for r in records]
    assert roles[:2] == ["system", "user"]
    assert records[-1] == {"turn": -1, "role": "adapter", "stop": "submitted"}
    assistant = [r for r in records if r["role"] == "assistant"]
    assert assistant[0]["tool_call"] == "ls"
    assert assistant[0]["tool_result"]["return_code"] == 0
    assert assistant[2]["submitted"] and assistant[2]["tool_call"] is None
    frozen = json.loads((tmp_path / "k2trace/frozen.json").read_text())
    assert [t["turn"] for t in frozen["turns"]] == [1, 2, 3]
    assert frozen["final_sequence"] == frozen["turns"][-1]["prompt_token_ids"] + [1003, 1004]
    assert frozen["final_prompt_length"] == len(frozen["turns"][-1]["prompt_token_ids"])
    detail = loop.frozen.rollout_detail()
    assert detail["completion_token_ids"][1] == [1002, 1003]
    assert detail["extra"]["k2_interp_run_id"] == ["chat-1", "chat-2", "chat-3"]
    assert result.n_output_tokens == 6


def test_loop_idle_nudge_and_stop(tmp_path):
    result, _, server, seen = run_loop(
        tmp_path,
        ["thinking out loud", "still nothing", "nope"],
        settings=LoopSettings(max_idle_turns=2),
    )
    assert result.stop == "idle"
    assert result.turns == 2
    assert seen == []
    assert "No fenced bash block" in server.calls[1][-1]["content"]


def test_loop_max_turns(tmp_path):
    result, *_ = run_loop(tmp_path, ["```bash\ntrue\n```"] * 3, settings=LoopSettings(max_turns=2))
    assert result.stop == "max_turns" and result.turns == 2


def test_loop_truncates_long_output(tmp_path):
    result, _, server, _ = run_loop(
        tmp_path,
        ["```bash\nyes\n```", "SUBMIT"],
        [ExecOutcome("y\n" * 10000, "", 0)],
        settings=LoopSettings(output_chars=400),
    )
    observation = server.calls[1][-1]["content"]
    assert "characters truncated" in observation and len(observation) < 700


def test_frozen_trajectory_empty_payload(tmp_path):
    frozen = FrozenTrajectory(tmp_path / "f.json")
    assert frozen.payload()["final_sequence"] == []
    assert frozen.rollout_detail()["prompt_token_ids"] == []


# --- replay --------------------------------------------------------------------------


def fake_trial(job: Path, name: str, benchmark: str, split: str, report: dict | None, turns):
    trial = job / name
    (trial / "agent" / "k2trace").mkdir(parents=True)
    (trial / "verifier").mkdir()
    (trial / "result.json").write_text(
        json.dumps(
            {
                "trial_name": name,
                "task_name": f"{benchmark}-{split}-x",
                "agent_info": {"name": "k2-interp", "model_info": {"name": "sft_3_1500"}},
            }
        )
    )
    if report is not None:
        (trial / "verifier" / "report.json").write_text(json.dumps(report))
    (trial / "verifier" / "task-config.json").write_text(
        json.dumps({"metadata": {"benchmark": benchmark, "split": split, "case_id": "c1"}})
    )
    frozen = FrozenTrajectory(trial / "agent" / "k2trace" / "frozen.json")
    for i, (prompt, completion) in enumerate(turns, 1):
        frozen.add(
            i,
            {
                "run_id": f"r{i}",
                "revision": "abc",
                "input_ids": prompt,
                "output_ids": completion,
                "output_logprobs": [0.0] * len(completion),
                "finish_reason": "eos",
            },
        )
    return trial


def test_outcome_label_and_split():
    assert outcome_label(None) == "unscored"
    assert outcome_label({"status": "missing_solution", "valid": False}) == "bailed"
    assert outcome_label({"status": "invalid", "valid": False}) == "wrong"
    assert outcome_label({"status": "error"}) == "wrong"
    assert outcome_label({"status": "valid", "valid": True}) == "solved"
    assert (
        outcome_label({"status": "valid", "valid": True, "metrics": {"coverage_ratio": 0.0}})
        == "empty"
    )
    assert outcome_label({"status": "valid", "valid": True, "metrics": {"CR": 0.4}}) == "solved"
    assert harbor_split("train") == "train"
    assert harbor_split("test_hard") == "test"
    assert harbor_split("test_horizon_2022") == "test"
    assert harbor_split(None) == "exploratory"


def test_find_subsequence_and_turn_spans():
    assert find_subsequence([1, 2, 3, 2, 3, 4], [2, 3, 4]) == 3
    assert find_subsequence([1, 2, 3], [2, 3], start=2) == -1
    assert find_subsequence([1, 2], []) == -1
    # Turn 1's completion reappears verbatim inside turn 2's prompt; turn 2 (final)
    # is placed by construction; turn 3's two-token completion is too short to search.
    c1 = list(range(20, 30))
    t1 = ([10, 11, 12], c1)
    t2 = ([10, 11, 12, *c1, 30], [40, 41, 42])
    frozen = {
        "turns": [
            {"turn": 1, "prompt_token_ids": t1[0], "completion_token_ids": t1[1]},
            {"turn": 2, "prompt_token_ids": t2[0], "completion_token_ids": t2[1]},
        ],
        "final_sequence": t2[0] + t2[1],
        "final_prompt_length": len(t2[0]),
    }
    spans = turn_spans(frozen)
    assert spans[0] == {"turn": 1, "start": 3, "end": 13}
    assert spans[1] == {"turn": 2, "start": 14, "end": 17}
    # A short final completion that also occurs inside the prompt is not mislocated.
    collide = {
        "turns": [
            {"turn": 1, "prompt_token_ids": [1], "completion_token_ids": [5, 6]},
            {"turn": 2, "prompt_token_ids": [1, 5, 6, 7, 5, 6, 8], "completion_token_ids": [5, 6]},
        ],
        "final_sequence": [1, 5, 6, 7, 5, 6, 8, 5, 6],
        "final_prompt_length": 7,
    }
    spans = turn_spans(collide)
    assert spans[0]["start"] is None  # too short to search safely
    assert spans[1] == {"turn": 2, "start": 7, "end": 9}


def test_windows_cover_impasse_early_and_midpoint():
    first = list(range(1000, 1600))
    frozen = {
        "turns": [
            {"turn": 1, "prompt_token_ids": [1, 2], "completion_token_ids": first},
            {
                "turn": 2,
                "prompt_token_ids": [1, 2] + first + [3],
                "completion_token_ids": list(range(2000, 2100)),
            },
        ],
        "final_sequence": [1, 2] + first + [3] + list(range(2000, 2100)),
        "final_prompt_length": 2 + len(first) + 1,
    }
    w = windows(frozen, size=64)
    n = len(frozen["final_sequence"])
    assert w["impasse"] == list(range(n - 64, n))
    assert w["early"] == list(range(2 + 64, 2 + 128))
    assert len(w["midpoint"]) == 64
    assert all(0 <= p < n for ps in w.values() for p in ps)


def test_build_specs_b1_b2_b3(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    fake_trial(
        job,
        "stereo-test-case_0001__A",
        "stereo_imaging",
        "test",
        {"status": "missing_solution", "valid": False},
        [([1, 2, 3], [4, 5]), ([1, 2, 3, 4, 5, 6], list(range(7, 40)))],
    )
    fake_trial(
        job,
        "aeossp-train-case_0002__B",
        "aeossp_standard",
        "train",
        {"status": "valid", "valid": True, "metrics": {"WCR": 0.7}},
        [([9, 8, 7], list(range(50, 90)))],
    )
    # A trial without a frozen trajectory (e.g. the nop agent) is skipped.
    (job / "nop__C").mkdir()
    (job / "nop__C" / "result.json").write_text(json.dumps({"trial_name": "nop__C"}))

    trials = load_trials(job)
    assert [t["outcome"] for t in trials] == ["solved", "bailed"]
    summary = summarize(job)
    assert summary["trials"] == 2 and summary["by_benchmark"]["stereo_imaging"] == {"bailed": 1}

    b1 = build_specs(
        job, stage="b1", checkpoints=["sft_3_1500", "mid_3_3500"], run_prefix="impasse-a0"
    )
    assert len(b1) == 2 and {s.checkpoint for s in b1} == {"sft_3_1500", "mid_3_3500"}
    spec = b1[0]
    assert spec.run_id == "impasse-a0-b1-all"
    assert spec.capture.router and spec.capture.router_fields == list(ROUTER_SUMMARY_FIELDS)
    assert spec.capture.sites == [] and spec.capture.positions is None
    assert spec.generation.max_new_tokens == 0
    sample = next(s for s in spec.samples if s.id.startswith("stereo"))
    assert sample.input_ids + sample.continuation_ids == [1, 2, 3, 4, 5, 6] + list(range(7, 40))
    assert sample.labels["outcome"] == "bailed" and sample.split == "test"
    assert sample.template_group == "stereo_imaging-test-c1"
    assert sample.labels["stop"] == "timeout" and sample.labels["length_hits"] == 0
    train = next(s for s in spec.samples if s.id.startswith("aeossp"))
    assert train.split == "train" and train.labels["valid"] is True
    RunSpec.model_validate(spec.model_dump())

    b2 = build_specs(job, stage="b2", checkpoints=["sft_3_1500"], run_prefix="impasse-a0", window=8)
    assert len(b2) == 2  # b2 is always per trial
    stereo = next(s for s in b2 if "stereo" in s.run_id)
    assert stereo.capture.positions and max(stereo.capture.positions) < 39
    assert stereo.capture.layers == list(range(3, 61))
    assert stereo.capture.sites == ["resid_post", "moe_out"]
    assert stereo.protocol["windows"]["impasse"][1] == 38

    with pytest.raises(ValueError, match="layers"):
        build_specs(job, stage="b3", checkpoints=["sft_3_1500"], run_prefix="p")
    b3 = build_specs(job, stage="b3", checkpoints=["sft_3_1500"], run_prefix="p", layers=[12, 30])
    assert b3[0].capture.layers == [12, 30] and not b3[0].capture.router

    only = build_specs(
        job, stage="b1", checkpoints=["sft_3_1500"], run_prefix="p", outcomes=["bailed"]
    )
    assert [s.id for s in only[0].samples] == ["stereo-test-case_0001__A"]
    with pytest.raises(ValueError, match="Unknown checkpoint"):
        build_specs(job, stage="b1", checkpoints=["nope"], run_prefix="p")
    with pytest.raises(ValueError, match="max_context"):
        build_specs(job, stage="b1", checkpoints=["sft_3_1500"], run_prefix="p", max_context=10)

    paths = write_specs(b2, tmp_path / "specs")
    assert len(paths) == 2 and (tmp_path / "specs" / "index.json").exists()
    reloaded = RunSpec.model_validate(json.loads(paths[0].read_text()))
    assert reloaded.protocol["source"] == "harbor"


# --- router features on the tiny native model --------------------------------------------


def test_router_fields_filter_and_features(native_model, tokenizer, tmp_path):
    from k2_interp.features import router_rows, window_summary, write_router_features
    from k2_interp.runtime import Runtime

    runtime = Runtime(native_model, tokenizer, {"checkpoint": "test"}, tmp_path)
    spec = RunSpec(
        run_id="b1-test",
        checkpoint="test",
        samples=[
            Sample(
                id="trial-a",
                input_ids=[4, 5, 6, 7],
                continuation_ids=[8],
                labels={"outcome": "bailed"},
            )
        ],
        capture=CaptureSpec(router=True, router_fields=list(ROUTER_SUMMARY_FIELDS), sites=[]),
    )
    runtime.run(spec)
    run = tmp_path / "b1-test"
    keys = {k for _, _, t in iter_records(run) for k in t}
    assert any(k.endswith(".router_entropy") for k in keys)
    assert not any(k.endswith(".router_logits") for k in keys)
    assert not any(k.endswith(".selection_scores") for k in keys)

    rows = list(router_rows(run))
    assert rows and all(r["sample"] == "trial-a" for r in rows)
    assert {r["position"] for r in rows} == {0, 1, 2, 3, 4}
    assert all(len(r["expert_ids"]) == 2 for r in rows)  # tiny model routes top-2
    assert rows[0]["labels"] == {"outcome": "bailed"}
    summary = write_router_features(run, tmp_path / "feat" / "b1")
    assert summary["rows"] == len(rows) and summary["samples"] == 1
    import numpy as np

    data = np.load(summary["npz"], allow_pickle=True)
    assert data["expert_ids"].shape == (len(rows), 2)
    assert data["sample_names"][0] == "trial-a"
    assert np.isfinite(data["router_entropy"]).all()
    per_window = window_summary(run, {"late": [3, 4], "early": [0, 1]})
    assert set(per_window) == {"late", "early"}
    assert all(v["n"] == 2 for v in per_window["late"].values())
