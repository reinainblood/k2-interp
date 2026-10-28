"""SYNTHETIC SciCode-style Inspect task for testing the k2-interp Inspect pipeline.

NOT A BENCHMARK. Written by the k2-interp authors to exercise the runner, the trace
sidecar, and inspect-replay end to end. The problems are deliberately tiny; results
on them mean nothing about model capability. Do not train on this file.

    k2-interp-canary: 7c1f0e2a-scicode-synth-do-not-train

Two problems, three subproblems each, answered one subproblem per assistant turn
in one carried-forward conversation (so the frozen trace has a distinct turn per
subproblem). Each problem comes as a ``trap`` variant that leaves a convention
unstated (numpy FFT normalization; the sign of a cross-track error) and a
``control`` twin that states it. The pair shares ``pair_id`` in sample metadata.

Hidden unit tests run in an Inspect sandbox (``local`` by default, no docker).
Score metadata, read generically by ``k2-interp inspect-replay``:

    subproblems               [{id, function, turn, passed, detail}] in order
    subproblem_turns          {subproblem id: assistant turn that answered it}
    first_failing_subproblem  id of the first failing subproblem, or None
    passed / total / all_passed

Run:  inspect eval examples/inspect/scicode_synth/task.py --model k2interp/sft_3_1500
"""

from __future__ import annotations

import re
import sys

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import sandbox

CANARY = "k2-interp-canary: 7c1f0e2a-scicode-synth-do-not-train"
STORE_KEY = "scicode_synth"
CODE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)

SYSTEM = (
    "You are solving a multi-step scientific programming problem, one step per reply. "
    "In each reply, write exactly one fenced ```python block that defines the requested "
    "function. `numpy` is available; import what you use. Functions you wrote in earlier "
    "steps stay defined and may be called."
)

# Each subproblem: id, function name, prompt, hidden test (asserts; earlier answers
# and this answer are executed first). ``trap`` / ``control`` override one prompt.
PROBLEMS: dict[str, dict] = {
    "spectral": {
        "title": "Spectral energy of a sampled signal",
        "intro": (
            "A real signal x of length n is sampled at rate fs (Hz). We will build a small "
            "spectral-analysis toolkit in three steps."
        ),
        "subproblems": [
            {
                "id": "spectral.1",
                "function": "spectrum",
                "trap": (
                    "Step 1. Write `spectrum(x)` returning the discrete Fourier transform of "
                    "the 1-D array x as a complex numpy array of the same length."
                ),
                "control": (
                    "Step 1. Write `spectrum(x)` returning the discrete Fourier transform of "
                    "the 1-D array x as a complex numpy array of the same length. Use the "
                    "unitary normalization (numpy `norm='ortho'`, i.e. scale by 1/sqrt(n)) so "
                    "that the spectrum has the same energy as the signal (Parseval)."
                ),
                "test": (
                    "import numpy as np\n"
                    "rng = np.random.default_rng(0)\n"
                    "x = rng.normal(size=64)\n"
                    "X = spectrum(x)\n"
                    "assert X.shape == (64,)\n"
                    "assert np.allclose(X, np.fft.fft(x, norm='ortho'))\n"
                    "assert np.isclose(np.sum(np.abs(X) ** 2), np.sum(x ** 2))\n"
                ),
            },
            {
                "id": "spectral.2",
                "function": "band_energy",
                "prompt": (
                    "Step 2. Write `band_energy(x, lo, hi)` returning the sum of |X_k|^2 over "
                    "bins lo <= k < hi, where X = spectrum(x) from step 1."
                ),
                "test": (
                    "import numpy as np\n"
                    "rng = np.random.default_rng(1)\n"
                    "x = rng.normal(size=32)\n"
                    "X = np.fft.fft(x, norm='ortho')\n"
                    "assert np.isclose(band_energy(x, 0, 32), np.sum(x ** 2))\n"
                    "assert np.isclose(band_energy(x, 3, 9), np.sum(np.abs(X[3:9]) ** 2))\n"
                ),
            },
            {
                "id": "spectral.3",
                "function": "dominant_frequency",
                "prompt": (
                    "Step 3. Write `dominant_frequency(x, fs)` returning the frequency in Hz "
                    "of the largest-magnitude non-DC bin among the non-negative frequencies "
                    "(bins 1 .. n//2)."
                ),
                "test": (
                    "import numpy as np\n"
                    "fs = 100.0\n"
                    "t = np.arange(200) / fs\n"
                    "x = np.sin(2 * np.pi * 7.0 * t) + 0.2 * np.sin(2 * np.pi * 21.0 * t) + 3\n"
                    "assert np.isclose(dominant_frequency(x, fs), 7.0)\n"
                ),
            },
        ],
    },
    "crosstrack": {
        "title": "Cross-track error for path following",
        "intro": (
            "A vehicle follows the straight path from waypoint a to waypoint b in the plane "
            "(2-D numpy arrays). We will compute its cross-track error in three steps."
        ),
        "subproblems": [
            {
                "id": "crosstrack.1",
                "function": "path_direction",
                "prompt": (
                    "Step 1. Write `path_direction(a, b)` returning the unit vector pointing "
                    "from a to b."
                ),
                "test": (
                    "import numpy as np\n"
                    "d = path_direction(np.array([1.0, 1.0]), np.array([4.0, 5.0]))\n"
                    "assert np.allclose(d, [0.6, 0.8])\n"
                ),
            },
            {
                "id": "crosstrack.2",
                "function": "cross_track_error",
                "trap": (
                    "Step 2. Write `cross_track_error(p, a, b)` returning the signed "
                    "perpendicular distance of point p from the line through a and b."
                ),
                "control": (
                    "Step 2. Write `cross_track_error(p, a, b)` returning the signed "
                    "perpendicular distance of point p from the line through a and b. The "
                    "sign is positive when p lies to the LEFT of the direction of travel "
                    "a -> b (counter-clockwise side) and negative to the right."
                ),
                "test": (
                    "import numpy as np\n"
                    "a, b = np.array([0.0, 0.0]), np.array([10.0, 0.0])\n"
                    "assert np.isclose(cross_track_error(np.array([3.0, 2.0]), a, b), 2.0)\n"
                    "assert np.isclose(cross_track_error(np.array([3.0, -1.5]), a, b), -1.5)\n"
                    "c = np.array([0.0, 5.0])\n"
                    "assert np.isclose(cross_track_error(np.array([1.0, 1.0]), a, c), -1.0)\n"
                ),
            },
            {
                "id": "crosstrack.3",
                "function": "side_counts",
                "prompt": (
                    "Step 3. Write `side_counts(points, a, b)` taking an (m, 2) array and "
                    "returning a tuple (left, right): how many points have positive and "
                    "negative cross_track_error from step 2 (points exactly on the line "
                    "count for neither)."
                ),
                "test": (
                    "import numpy as np\n"
                    "a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0])\n"
                    "pts = np.array([[0, 1], [2, 3], [5, -1], [1, 0], [3, 0.5]], dtype=float)\n"
                    "assert tuple(int(v) for v in side_counts(pts, a, b)) == (3, 1)\n"
                ),
            },
        ],
    },
}

CONVENTIONS = {
    "spectral": "numpy FFT normalization (expects norm='ortho')",
    "crosstrack": "cross-track sign (expects positive to the left of travel)",
}


def subproblem_prompt(sub: dict, variant: str) -> str:
    return sub.get(variant) or sub["prompt"]


def build_samples() -> list[Sample]:
    samples = []
    for problem_id, problem in PROBLEMS.items():
        for variant in ("trap", "control"):
            subs = problem["subproblems"]
            samples.append(
                Sample(
                    id=f"{problem_id}-{variant}",
                    input=f"{problem['title']}\n\n{problem['intro']}",
                    metadata={
                        "synthetic": True,
                        "canary": CANARY,
                        "problem": problem_id,
                        "variant": variant,
                        "pair_id": problem_id,
                        "ambiguous_convention": CONVENTIONS[problem_id],
                        "convention_stated": variant == "control",
                        "subproblems": [
                            {
                                "id": s["id"],
                                "function": s["function"],
                                "prompt": subproblem_prompt(s, variant),
                            }
                            for s in subs
                        ],
                    },
                )
            )
    return samples


def extract_code(text: str) -> str | None:
    match = CODE.search(text or "")
    return match.group(1).strip() if match else None


@solver
def stepwise():
    """One subproblem per turn; the conversation (with reasoning) carries forward."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        intro = state.input_text
        state.messages = [ChatMessageSystem(content=SYSTEM)]
        answers: dict[str, dict] = {}
        turn = 0
        for index, sub in enumerate(state.metadata["subproblems"]):
            text = sub["prompt"] if index else f"{intro}\n\n{sub['prompt']}"
            state.messages.append(ChatMessageUser(content=text))
            state = await generate(state)
            turn += 1
            receipt = (state.output.metadata or {}).get("k2_interp") or {}
            answers[sub["id"]] = {
                "turn": receipt.get("turn", turn),
                "code": extract_code(state.output.completion),
            }
            state.store.set(STORE_KEY, {"answers": answers})
            if state.completed:
                break
        return state

    return solve


async def run_hidden_tests(answers: dict, subproblems: list[dict], tests: dict, execute) -> dict:
    """Run each subproblem's hidden test after every earlier answer; return score metadata.

    ``execute(program) -> (ok, detail)`` runs a Python program in the sandbox.
    """
    results, prefix, first_fail = [], [], None
    for sub in subproblems:
        answer = answers.get(sub["id"]) or {}
        code = answer.get("code")
        if code:
            prefix.append(code)
        if not code:
            ok, detail = False, "no python block in the answering turn"
        else:
            ok, detail = await execute("\n\n".join([*prefix, tests[sub["id"]]]))
        results.append(
            {
                "id": sub["id"],
                "function": sub["function"],
                "turn": answer.get("turn"),
                "passed": ok,
                "detail": detail[-800:],
            }
        )
        if not ok and first_fail is None:
            first_fail = sub["id"]
    passed = sum(r["passed"] for r in results)
    return {
        "subproblems": results,
        "subproblem_turns": {r["id"]: r["turn"] for r in results if r["turn"] is not None},
        "first_failing_subproblem": first_fail,
        "passed": passed,
        "total": len(results),
        "all_passed": passed == len(results),
        "answered": sum(1 for r in results if r["turn"] is not None),
    }


def hidden_tests(problem_id: str) -> dict[str, str]:
    return {s["id"]: s["test"] for s in PROBLEMS[problem_id]["subproblems"]}


@scorer(metrics=[mean()])
def subproblem_tests(python: str = "python3", timeout: int = 60):
    async def score(state: TaskState, target: Target) -> Score:
        answers = (state.store.get(STORE_KEY) or {}).get("answers", {})

        async def execute(program: str):
            env = sandbox()
            await env.write_file("check.py", program)
            result = await env.exec([python, "check.py"], timeout=timeout)
            return result.success, (result.stdout + result.stderr).strip() or "ok"

        meta = await run_hidden_tests(
            answers,
            state.metadata["subproblems"],
            hidden_tests(state.metadata["problem"]),
            execute,
        )
        meta.update(
            variant=state.metadata["variant"],
            pair_id=state.metadata["pair_id"],
            synthetic=True,
        )
        return Score(
            value=meta["passed"] / meta["total"],
            answer=meta["first_failing_subproblem"] or "all_passed",
            explanation=f"{meta['passed']}/{meta['total']} subproblems passed",
            metadata=meta,
        )

    return score


@task
def scicode_synth(python: str | None = None, sandbox_type: str = "local"):
    """SYNTHETIC trap/control SciCode-style task; a pipeline fixture, not a benchmark."""
    interpreter = python or (sys.executable if sandbox_type == "local" else "python3")
    return Task(
        dataset=MemoryDataset(build_samples(), name="scicode_synth"),
        solver=stepwise(),
        scorer=subproblem_tests(python=interpreter),
        sandbox=sandbox_type,
        metadata={"synthetic": True, "canary": CANARY},
    )
