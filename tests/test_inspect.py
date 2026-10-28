"""Inspect AI runner: provider mapping, trace sidecar, synthetic task, inspect-replay. Offline."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import (  # noqa: E402
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    get_model,
)
from inspect_ai.tool import ToolInfo  # noqa: E402

from k2_interp import harbor  # noqa: E402
from k2_interp.harbor import FrozenTrajectory  # noqa: E402
from k2_interp.harbor_replay import load_trials, turn_spans, windows  # noqa: E402
from k2_interp.inspect_provider import (  # noqa: E402
    K2InterpAPI,
    to_gateway_messages,
    to_model_output,
)
from k2_interp.inspect_replay import (  # noqa: E402
    build_inspect_specs,
    commit_point,
    load_inspect_trials,
    score_outcome,
)
from k2_interp.registry import CHECKPOINTS  # noqa: E402
from k2_interp.schema import ChatRequest, RunSpec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TASK_FILE = ROOT / "examples" / "inspect" / "scicode_synth" / "task.py"


def load_task_module():
    spec = importlib.util.spec_from_file_location("scicode_synth_task", TASK_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


synth = load_task_module()

# Known-good answers per subproblem, and the trap-shaped wrong ones.
GOOD = {
    "spectral.1": "import numpy as np\ndef spectrum(x):\n    return np.fft.fft(x, norm='ortho')",
    "spectral.2": "import numpy as np\ndef band_energy(x, lo, hi):\n"
    "    return float(np.sum(np.abs(spectrum(x)[lo:hi]) ** 2))",
    "spectral.3": "import numpy as np\ndef dominant_frequency(x, fs):\n"
    "    X = np.abs(spectrum(x))\n    n = len(x)\n    k = 1 + int(np.argmax(X[1 : n // 2 + 1]))\n"
    "    return k * fs / n",
    "crosstrack.1": "import numpy as np\ndef path_direction(a, b):\n"
    "    d = b - a\n    return d / np.linalg.norm(d)",
    "crosstrack.2": "import numpy as np\ndef cross_track_error(p, a, b):\n"
    "    d = path_direction(a, b)\n    r = p - a\n    return float(d[0] * r[1] - d[1] * r[0])",
    "crosstrack.3": "import numpy as np\ndef side_counts(points, a, b):\n"
    "    e = np.array([cross_track_error(p, a, b) for p in points])\n"
    "    return int(np.sum(e > 0)), int(np.sum(e < 0))",
}
BAD = dict(GOOD)
BAD["spectral.1"] = "import numpy as np\ndef spectrum(x):\n    return np.fft.fft(x)"
BAD["crosstrack.2"] = (
    "import numpy as np\ndef cross_track_error(p, a, b):\n"
    "    d = path_direction(a, b)\n    r = p - a\n    return float(d[1] * r[0] - d[0] * r[1])"
)


def fenced(code: str) -> str:
    return f"Here is the step.\n\n```python\n{code}\n```"


def response(checkpoint, content, reasoning, prompt, completion, run_id, finish="eos", rev=None):
    return {
        "id": run_id,
        "object": "chat.completion",
        "model": checkpoint,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                },
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(completion),
            "total_tokens": len(prompt) + len(completion),
        },
        "k2_interp": {
            "revision": rev or CHECKPOINTS[checkpoint],
            "run_id": run_id,
            "captured": False,
            "input_ids": prompt,
            "output_ids": completion,
            "output_logprobs": [-0.5] * len(completion),
            "finish_reason": finish,
            "mode": "generation",
        },
    }


class FakeGateway:
    """Stands in for harbor.post_json. Answers each subproblem prompt from a table and
    emits a token stream where every prompt extends the previous prompt + completion,
    the way a chat template re-renders history, so turn_spans can locate every turn."""

    def __init__(self, answers, checkpoint="sft_3_1500", served=None, revision=None):
        self.answers = answers
        self.checkpoint = checkpoint
        self.served = served or checkpoint
        self.revision = revision
        self.calls = []
        self.streams: dict[str, list[int]] = {}
        self.counter = 0

    def __call__(self, url, payload, headers, timeout):
        assert url.endswith("/v1/chat/completions")
        assert headers == {"Modal-Key": "tid", "Modal-Secret": "tsecret"}
        ChatRequest.model_validate(payload)
        self.calls.append(payload)
        messages = payload["messages"]
        for m in messages:
            if m["role"] == "assistant":
                assert "reasoning_content" in m  # K2 template requirement
        last = messages[-1]["content"]
        sub = next(s for s in self.answers if f"`{synth_function(s)}(" in last)
        key = next(m["content"][:40] for m in messages if m["role"] == "user")
        history = self.streams.get(key, [7, 7, 7])
        self.counter += 1
        user_tokens = [3000 + self.counter, 3001, 3002]
        prompt = history + user_tokens
        completion = [10_000 + 100 * self.counter + i for i in range(12)]
        self.streams[key] = prompt + completion
        return response(
            self.served,
            fenced(self.answers[sub]),
            f"thinking about {sub}",
            prompt,
            completion,
            f"chat-{self.counter:04d}",
            rev=self.revision,
        )


def synth_function(sub_id):
    for problem in synth.PROBLEMS.values():
        for s in problem["subproblems"]:
            if s["id"] == sub_id:
                return s["function"]
    raise KeyError(sub_id)


@pytest.fixture
def creds(monkeypatch, tmp_path):
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "tid")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "tsecret")
    monkeypatch.setenv("K2_INTERP_ENDPOINT", "https://fake.example")
    return tmp_path


def api(tmp_path, **kwargs):
    return K2InterpAPI("sft_3_1500", trace_dir=str(tmp_path / "traces"), **kwargs)


# --- provider mapping ---------------------------------------------------------------------


def test_payload_and_reasoning_round_trip(creds):
    model = api(creds)
    raw = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "deep thought",
        "think_fast": "deep thought",
    }
    out = to_model_output(
        {
            **response("sft_3_1500", "answer", "deep thought", [1, 2], [3, 4, 5], "chat-1"),
            "choices": [{"index": 0, "message": raw, "finish_reason": "eos"}],
        },
        "sft_3_1500",
    )
    message = out.message
    assert out.stop_reason == "stop"
    assert out.usage.input_tokens == 2 and out.usage.output_tokens == 3
    assert out.completion == "answer"
    assert [type(c) for c in message.content] == [ContentReasoning, ContentText]
    assert message.content[0].reasoning == "deep thought"
    assert out.metadata["k2_interp"]["run_id"] == "chat-1"
    # The server message comes back verbatim, including template-specific keys.
    history = [
        ChatMessageSystem(content="sys"),
        ChatMessageUser(content="q"),
        message,
        ChatMessageUser(content="next"),
    ]
    payload = model.payload(history, GenerateConfig())
    assert payload["messages"][2] == raw
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    # IFM sampling defaults, the shared chat_payload shape, and the gateway contract.
    assert payload["temperature"] == 1.0 and payload["top_p"] == 0.95 and payload["seed"] == 7
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert payload["model"] == "sft_3_1500" and payload["max_context"] == 262144
    assert "capture" not in payload
    ChatRequest.model_validate(payload)
    # Eval-level overrides apply.
    tuned = model.payload(history, GenerateConfig(temperature=0, seed=3, max_tokens=64))
    assert tuned["temperature"] == 0 and tuned["seed"] == 3 and tuned["max_tokens"] == 64
    assert "capture" in api(creds, capture_router=True).payload(history, GenerateConfig())


def test_foreign_assistant_turns_get_reasoning_field():
    edited = ChatMessageAssistant(
        content=[ContentReasoning(reasoning="r"), ContentText(text="changed")],
        metadata={"k2_interp": {"message": {"content": "orig", "reasoning_content": "r"}}},
    )
    plain = ChatMessageAssistant(content="hello")
    out = to_gateway_messages([ChatMessageUser(content="u"), edited, plain])
    assert out[1] == {"role": "assistant", "content": "changed", "reasoning_content": "r"}
    assert out[2] == {"role": "assistant", "content": "hello", "reasoning_content": ""}
    assert to_model_output(
        response("sft_3_1500", "x", "", [1], [2], "c", finish="length"), "sft_3_1500"
    ).stop_reason == ("max_tokens")


def test_frozen_schema_parity_with_harbor(creds, monkeypatch):
    gateway = FakeGateway(GOOD)
    monkeypatch.setattr(harbor, "post_json", gateway)
    log = run_eval(creds, sample_id="spectral-control")
    sample = log.samples[0]
    record = sample.store["k2_interp"]
    frozen_path = Path(record["frozen"])
    assert frozen_path.parent.name == "epoch-1"
    assert frozen_path.parent.parent.name == "spectral-control"
    assert frozen_path.parent.parent.parent.name == log.eval.eval_id
    ours = json.loads(frozen_path.read_text())
    # Rebuild the same turns with Harbor's own writer and compare byte-for-byte shape.
    reference = FrozenTrajectory(creds / "reference.json")
    for turn, t in enumerate(ours["turns"], 1):
        reference.add(
            turn,
            {
                "run_id": t["run_id"],
                "revision": t["revision"],
                "input_ids": t["prompt_token_ids"],
                "output_ids": t["completion_token_ids"],
                "output_logprobs": t["logprobs"],
                "finish_reason": t["finish_reason"],
                "captured": t["captured"],
            },
        )
    assert ours == reference.payload()
    assert ours["schema"] == "k2-interp/harbor-frozen/1"
    assert set(ours) == {"schema", "turns", "final_sequence", "final_prompt_length"}
    assert (
        record["run_ids"]
        == [t["run_id"] for t in ours["turns"]]
        == [
            "chat-0001",
            "chat-0002",
            "chat-0003",
        ]
    )
    assert record["revision"] == CHECKPOINTS["sft_3_1500"]
    provider = json.loads((frozen_path.parent / "provider.json").read_text())
    assert provider["checkpoint"] == "sft_3_1500" and provider["sample_id"] == "spectral-control"
    # Per-turn receipts ride on the assistant messages and the logged model calls.
    assistant = [m for m in sample.messages if m.role == "assistant"]
    assert [m.metadata["k2_interp"]["turn"] for m in assistant] == [1, 2, 3]
    assert all(m.content[0].reasoning.startswith("thinking about") for m in assistant)
    # A Harbor-style job directory reads the same file with the unchanged loader.
    job = creds / "job" / "trial"
    (job / "agent" / "k2trace").mkdir(parents=True)
    (job / "agent" / "k2trace" / "frozen.json").write_text(frozen_path.read_text())
    (job / "result.json").write_text(json.dumps({"trial_name": "trial"}))
    assert load_trials(creds / "job")[0]["turns"] == 3


# --- loud failures ------------------------------------------------------------------------


def test_checkpoint_mismatch_tools_and_streaming(creds, monkeypatch):
    with pytest.raises(ValueError, match="unknown checkpoint"):
        K2InterpAPI("not_a_checkpoint")
    with pytest.raises(ValueError, match="streaming"):
        api(creds, stream=True)
    with pytest.raises(ValueError, match="credentials"):
        api(creds, token_secret="x")
    model = api(creds)
    msgs = [ChatMessageUser(content="Write `spectrum(x)` please")]

    monkeypatch.setattr(harbor, "post_json", FakeGateway(GOOD, served="mid_3_3500"))
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        asyncio.run(model.generate(msgs, [], "auto", GenerateConfig()))
    monkeypatch.setattr(harbor, "post_json", FakeGateway(GOOD, revision="deadbeef"))
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        asyncio.run(model.generate(msgs, [], "auto", GenerateConfig()))

    tool = ToolInfo(name="bash", description="run bash")
    with pytest.raises(NotImplementedError, match="tool emulation is not supported"):
        asyncio.run(model.generate(msgs, [tool], "auto", GenerateConfig()))
    with pytest.raises(NotImplementedError, match="tool emulation"):
        to_gateway_messages([ChatMessageTool(content="out", tool_call_id="1")])
    with pytest.raises(NotImplementedError, match="streaming"):
        asyncio.run(model.generate(msgs, [], "auto", GenerateConfig(stream_idle_timeout=5)))
    with pytest.raises(NotImplementedError, match="num_choices"):
        model.payload(msgs, GenerateConfig(num_choices=2))

    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET")
    monkeypatch.setattr(harbor, "post_json", FakeGateway(GOOD))
    with pytest.raises(ValueError, match="MODAL_PROXY_TOKEN_SECRET"):
        asyncio.run(model.generate(msgs, [], "auto", GenerateConfig()))


def test_provider_resolves_through_entry_point(creds):
    model = get_model("k2interp/sft_3_1500", trace_dir=str(creds / "t"), memoize=False)
    assert isinstance(model.api, K2InterpAPI)
    assert model.api.endpoint == "https://fake.example"


# --- synthetic task: scorer ---------------------------------------------------------------


def local_execute(tmp_path):
    import subprocess

    async def execute(program):
        path = tmp_path / "check.py"
        path.write_text(program)
        proc = subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True, timeout=60
        )
        return proc.returncode == 0, (proc.stdout + proc.stderr).strip() or "ok"

    return execute


@pytest.mark.parametrize("problem", ["spectral", "crosstrack"])
def test_scorer_metadata_known_good_and_bad(tmp_path, problem):
    subs = synth.PROBLEMS[problem]["subproblems"]
    tests = synth.hidden_tests(problem)
    good = {s["id"]: {"turn": i + 1, "code": GOOD[s["id"]]} for i, s in enumerate(subs)}
    bad = {s["id"]: {"turn": i + 1, "code": BAD[s["id"]]} for i, s in enumerate(subs)}
    ok = asyncio.run(synth.run_hidden_tests(good, subs, tests, local_execute(tmp_path)))
    assert ok["all_passed"] and ok["first_failing_subproblem"] is None
    assert [r["passed"] for r in ok["subproblems"]] == [True, True, True]
    wrong = asyncio.run(synth.run_hidden_tests(bad, subs, tests, local_execute(tmp_path)))
    trap = {"spectral": "spectral.1", "crosstrack": "crosstrack.2"}[problem]
    assert wrong["first_failing_subproblem"] == trap
    assert wrong["subproblem_turns"][trap] == [s["id"] for s in subs].index(trap) + 1
    failing = [r["id"] for r in wrong["subproblems"] if not r["passed"]]
    assert failing[0] == trap
    missing = asyncio.run(synth.run_hidden_tests({}, subs, tests, local_execute(tmp_path)))
    assert missing["passed"] == 0 and missing["answered"] == 0
    assert "no python block" in missing["subproblems"][0]["detail"]


def test_dataset_pairs_trap_and_control():
    samples = synth.build_samples()
    by_id = {s.id: s for s in samples}
    assert set(by_id) == {
        "spectral-trap",
        "spectral-control",
        "crosstrack-trap",
        "crosstrack-control",
    }
    for problem in ("spectral", "crosstrack"):
        trap, control = by_id[f"{problem}-trap"], by_id[f"{problem}-control"]
        assert trap.metadata["pair_id"] == control.metadata["pair_id"] == problem
        assert trap.metadata["synthetic"] and trap.metadata["canary"] == synth.CANARY
        assert not trap.metadata["convention_stated"] and control.metadata["convention_stated"]
        t_prompts = [s["prompt"] for s in trap.metadata["subproblems"]]
        c_prompts = [s["prompt"] for s in control.metadata["subproblems"]]
        assert sum(a != b for a, b in zip(t_prompts, c_prompts, strict=True)) == 1
        assert 2 <= len(t_prompts) <= 3


# --- end to end ---------------------------------------------------------------------------


def run_eval(tmp_path, sample_id=None):
    task = synth.scicode_synth()
    logs = inspect_eval(
        task,
        model="k2interp/sft_3_1500",
        model_args={"trace_dir": str(tmp_path / "traces")},
        log_dir=str(tmp_path / "logs"),
        display="none",
        sample_id=sample_id,
    )
    log = logs[0]
    assert log.status == "success", log.error
    return log


def test_end_to_end_eval_and_inspect_replay(creds, monkeypatch):
    # The fake model makes the convention error unless the convention is stated, so
    # trap variants fail at their ambiguous step and control twins pass.
    controls = [
        s["control"] for p in synth.PROBLEMS.values() for s in p["subproblems"] if "control" in s
    ]
    gateway = FakeGateway(GOOD)

    def routed(url, payload, headers, timeout):
        stated = any(c in m["content"] for m in payload["messages"] for c in controls)
        gateway.answers = GOOD if stated else BAD
        return gateway(url, payload, headers, timeout)

    monkeypatch.setattr(harbor, "post_json", routed)
    log = run_eval(creds)
    log_file = Path(log.location)
    assert log_file.suffix == ".eval" and log_file.is_file()
    assert len(gateway.calls) == 12  # 4 samples x 3 subproblems, one turn each

    by_id = {s.id: s for s in log.samples}
    for sid, sample in by_id.items():
        meta = next(iter(sample.scores.values())).metadata
        assert meta["total"] == 3 and meta["subproblem_turns"]
        if sid.endswith("control"):
            assert meta["all_passed"] and meta["first_failing_subproblem"] is None
    assert (
        next(iter(by_id["spectral-trap"].scores.values())).metadata["first_failing_subproblem"]
        == "spectral.1"
    )
    assert (
        next(iter(by_id["crosstrack-trap"].scores.values())).metadata["first_failing_subproblem"]
        == "crosstrack.2"
    )

    trials, summary = load_inspect_trials(log_file)
    assert summary["trials"] == 4 and summary["skipped"] == []
    outcomes = {t["extra_labels"]["sample_id"]: t["outcome"] for t in trials}
    assert outcomes["spectral-control"] == "solved"
    assert outcomes["crosstrack-trap"] == "partial"  # 1/3 or 2/3 passed
    cross = next(t for t in trials if t["extra_labels"]["sample_id"] == "crosstrack-trap")
    assert cross["commit"] == {
        "subproblem": "crosstrack.2",
        "turn": 2,
        "turn_source": "subproblem_turns",
    }
    # Every turn is locatable because each prompt extends the previous sequence.
    assert all(s["start"] is not None for s in turn_spans(cross["frozen"]))

    b1, _ = build_inspect_specs(
        log_file, stage="b1", checkpoints=["sft_3_1500", "mid_3_3500"], run_prefix="synth"
    )
    assert len(b1) == 2 and len(b1[0].samples) == 4
    spec = b1[0]
    assert spec.protocol["source"] == "inspect"
    assert spec.protocol["inspect"]["eval_id"] == log.eval.eval_id
    assert spec.protocol["commit"]["crosstrack-trap-e1"]["subproblem"] == "crosstrack.2"
    sample = next(s for s in spec.samples if s.id == "crosstrack-trap-e1")
    assert sample.labels["outcome"] == "partial" and sample.labels["variant"] == "trap"
    assert sample.labels["pair_id"] == "crosstrack" and sample.labels["source"] == "inspect"
    assert sample.labels["first_failing_subproblem"] == "crosstrack.2"
    assert sample.template_group.endswith("-crosstrack")  # trap and control twin share a group
    assert sample.input_ids + sample.continuation_ids == cross["frozen"]["final_sequence"]
    RunSpec.model_validate(spec.model_dump())

    b2, _ = build_inspect_specs(
        log_file, stage="b2", checkpoints=["sft_3_1500"], run_prefix="synth", window=8
    )
    assert len(b2) == 4
    trap_spec = next(s for s in b2 if "crosstrack-trap" in s.run_id)
    w = trap_spec.protocol["windows"]
    assert {"impasse", "midpoint", "commit"} <= set(w) and trap_spec.protocol["window"] == 8
    assert trap_spec.protocol["window_kinds"] == ["impasse", "early", "midpoint", "commit"]
    turn2 = next(s for s in turn_spans(cross["frozen"]) if s["turn"] == 2)
    assert w["commit"] == [turn2["start"] - 4, turn2["start"] + 3]
    assert set(range(w["commit"][0], w["commit"][1] + 1)) <= set(trap_spec.capture.positions)
    control_spec = next(s for s in b2 if "spectral-control" in s.run_id)
    assert "commit" not in control_spec.protocol["windows"]

    only = build_inspect_specs(
        log_file, stage="b1", checkpoints=["sft_3_1500"], run_prefix="s", outcomes=["solved"]
    )[0]
    assert {s.labels["variant"] for s in only[0].samples} == {"control"}

    # CLI round trip.
    from k2_interp.cli import main

    out = creds / "specs"
    main(
        [
            "inspect-replay",
            str(log_file),
            "--stage",
            "b2",
            "--window",
            "8",
            "--checkpoints",
            "sft_3_1500",
            "--run-prefix",
            "synth",
            "--output",
            str(out),
        ]
    )
    index = json.loads((out / "index.json").read_text())
    assert len(index["specs"]) == 4 and all(s["stage"] == "b2" for s in index["specs"])


def test_trace_dir_fallback_when_store_path_moved(creds, monkeypatch):
    monkeypatch.setattr(harbor, "post_json", FakeGateway(GOOD))
    log = run_eval(creds, sample_id="crosstrack-control")
    moved = creds / "moved"
    (creds / "traces").rename(moved)
    with pytest.raises(ValueError, match="no frozen.json"):
        load_inspect_trials(log.location)
    trials, _ = load_inspect_trials(log.location, trace_dir=moved)
    assert trials[0]["frozen_path"].startswith(str(moved))


# --- generic score reading ----------------------------------------------------------------


def test_score_outcome_and_commit_point_generic():
    assert score_outcome("C") == "solved" and score_outcome("I") == "wrong"
    assert score_outcome("P") == "partial" and score_outcome("N") == "bailed"
    assert score_outcome(True) == "solved" and score_outcome(0) == "wrong"
    assert score_outcome(0.5) == "partial" and score_outcome(1.0) == "solved"
    assert score_outcome(None) == "unscored" and score_outcome({"a": 1}) == "unscored"
    assert commit_point({}) is None
    assert commit_point({"first_failing_subproblem": None}) is None
    # Real task sets may report only an ordered list of subproblem ids.
    assert commit_point(
        {"first_failing_subproblem": "13.2", "subproblems": ["13.1", "13.2", "13.3"]}
    ) == {"subproblem": "13.2", "turn": 2, "turn_source": "position"}
    assert commit_point({"first_failing_subproblem": 4, "subproblems": [{"id": 4, "turn": 6}]}) == {
        "subproblem": "4",
        "turn": 6,
        "turn_source": "turn",
    }
    assert commit_point({"first_failing_subproblem": "x"})["turn"] is None


def test_commit_window_is_absent_when_turn_unlocatable():
    frozen = {
        "turns": [
            {"turn": 1, "prompt_token_ids": [1], "completion_token_ids": [5, 6]},
            {"turn": 2, "prompt_token_ids": [1, 5, 6, 7], "completion_token_ids": [8, 9]},
        ],
        "final_sequence": [1, 5, 6, 7, 8, 9],
        "final_prompt_length": 4,
    }
    assert "commit" not in windows(frozen, 4, commit_turn=1)  # too short to search
    assert windows(frozen, 4, commit_turn=2)["commit"] == [2, 3, 4, 5]
    assert "commit" not in windows(frozen, 4)
