# Inspect AI: eval traces with frozen token identity

k2-interp ships an [Inspect AI](https://inspect.aisi.org.uk/) model provider, so
any text-only Inspect eval (SciCode-style multi-step coding tasks in particular)
can run against a deployed K2 checkpoint and leave the same frozen token-identity
trace the [Harbor adapter](harbor.md) leaves. Those traces feed the existing
replay-with-capture pipeline unchanged.

```
Pass A   inspect eval task.py --model k2interp/sft_3_1500   -> .eval log + frozen.json per sample
Pass B   k2-interp inspect-replay <log.eval> --stage b1|b2|b3 -> RunSpecs; k2-interp run ...
```

## Install

```sh
uv sync --python 3.11 --extra dev --extra inspect     # inspect-ai is pinned (0.3.286)
```

The provider registers through the `inspect_ai` entry point in `pyproject.toml`;
no import is needed in task files.

## Pass A: the `k2interp` provider

```sh
export K2_INTERP_ENDPOINT=https://<your-app>.modal.run
export MODAL_PROXY_TOKEN_ID=... MODAL_PROXY_TOKEN_SECRET=...
uv run inspect eval examples/inspect/scicode_synth/task.py \
  --model k2interp/sft_3_1500 \
  -M trace_dir=logs/k2trace -M max_context=262144 \
  --log-dir logs
```

The model name is the registry checkpoint tag. Model args (`-M`):

| Arg | Default | Meaning |
| --- | --- | --- |
| `endpoint` | `$K2_INTERP_ENDPOINT` | gateway base URL (`--model-base-url` also works) |
| `capture_router` | `false` | ask for summary routing fields live on every turn |
| `max_context` | `262144` | sent as `max_context` on every request |
| `trace_dir` | `$K2_INTERP_TRACE_DIR` or `logs/k2trace` | root for frozen traces |
| `request_timeout` | `3600` | seconds per HTTP request |

Proxy credentials are read from `MODAL_PROXY_TOKEN_ID` / `MODAL_PROXY_TOKEN_SECRET`
only (sent as `Modal-Key` / `Modal-Secret`, as in Harbor), so they never appear in
`EvalSpec.model_args` inside the log.

Requests are built with the Harbor adapter's own `chat_payload`. Sampling follows
IFM's recommendation unless the eval overrides it with Inspect's generate config:
temperature 1.0, top-p 0.95, seed 7, `reasoning_effort` high, `max_tokens` 8192
(the gateway cap). `--temperature`, `--top-p`, `--seed`, `--max-tokens`, and
`--reasoning-effort` map straight through.

Reasoning round-trips exactly. The server's `reasoning_content` becomes a
`ContentReasoning` block on the assistant message, and the raw server message is
kept in that message's metadata. When the conversation is sent back, an unedited
assistant turn is re-sent verbatim; an edited or foreign assistant turn is rebuilt
with `reasoning_content` (empty if it has none), because K2's chat template needs
the field on every assistant history turn. The provider forces Inspect's
`reasoning_history` to `all` for the same reason.

Fails loudly, rather than degrading:

* a response whose `model` or pinned `revision` differs from the requested
  checkpoint (`checkpoint mismatch`), or that has no `k2_interp` block;
* tools offered to `generate`, or tool calls/messages in the history
  (`NotImplementedError`: tool emulation is not supported yet);
* streaming (`stream=true`, an `on_stream` consumer, or `stream_idle_timeout`);
* `num_choices > 1`, stop sequences, non-text content, unknown model args.

HTTP 408/425/429/5xx and connection errors are retried by Inspect's usual policy.

## Trace layout

Per sample attempt:

```
<trace_dir>/<eval_id>/<sample_id>/epoch-<n>/frozen.json     k2-interp/harbor-frozen/1
<trace_dir>/<eval_id>/<sample_id>/epoch-<n>/provider.json   endpoint, checkpoint, revision, sampling
```

`frozen.json` is written by Harbor's `FrozenTrajectory` after every turn, so it is
byte-for-byte the same schema: per turn `run_id`, `revision`, `prompt_token_ids`,
`completion_token_ids`, `logprobs`, `finish_reason`, `captured`; plus
`final_sequence` and `final_prompt_length`. Sample ids are made filesystem-safe
(characters outside `[A-Za-z0-9_.-]` become `_`).

Where the Inspect log points at it:

| Location in the log | Contents |
| --- | --- |
| `EvalSample.store["k2_interp"]` | `frozen` path, `trace_dir`, `revision`, all server `run_ids`, `turns` |
| assistant `ChatMessage.metadata["k2_interp"]` | the raw server message, `run_id`, `turn`, `frozen`, `finish_reason` |
| `ModelEvent.output.metadata["k2_interp"]` | the same per-turn receipt |
| `ModelEvent.call` | the request payload and gateway response (token arrays elided; they live in `frozen.json`) |

## Pass B: `inspect-replay`

```sh
uv run k2-interp inspect-replay logs/<file>.eval --stage b1 \
  --checkpoints sft_3_1500 rl_merged --run-prefix synth-a0 --output specs/b1
uv run k2-interp inspect-replay logs/<file>.eval --stage b2 --window 256 \
  --checkpoints sft_3_1500 --run-prefix synth-a0 --output specs/b2
uv run k2-interp run specs/b2/sft_3_1500--synth-a0-b2-spectral-trap-e1.json --detach
```

It reads the log, pairs each sample with its `frozen.json` (the path in the store;
if traces were moved, pass `--trace-dir` to resolve the default layout under a new
root) and its score, and hands the result to the same `build_specs` Harbor replay
uses. Stages, `--layers`, `--sites`, `--per-trial`, `--outcomes`, `--max-*` all
behave as in [Harbor replay](harbor.md#pass-b-replay-with-capture). Samples without
a trace (errored before the first turn, or another provider) are reported as
`skipped`.

`outcome` comes from the score value (first scorer, or `--scorer NAME`):

| Score value | `outcome` |
| --- | --- |
| `C`, `true`, `>= 1` | `solved` |
| `I`, `false`, `<= 0` | `wrong` |
| `P`, a fraction in (0, 1) | `partial` |
| `N` | `bailed` |
| none / non-scalar | `unscored` |

Other labels: `source=inspect`, `sample_id`, `epoch`, `score`, `scorer`,
`first_failing_subproblem`, `stop` (`completed`, `error`, `limit_<type>`),
`length_hits`, n-gram loop signals, and when present in sample/score metadata
`variant`, `pair_id`, `convention_stated`, `synthetic`, `score_passed`,
`score_total`, `score_all_passed`. The template group uses `pair_id` (else
`case_id`, else the sample id), so trap/control twins never straddle splits.

### The `commit` window

Stage b2 adds a `commit` window beside `impasse`, `early`, and `midpoint`: `--window`
tokens centred on the start of the completion of the turn that answered the first
failing subproblem, located with the same `turn_spans` logic. It is read
generically from score metadata:

1. `first_failing_subproblem` names the subproblem (absent or null: no window);
2. its turn is `subproblem_turns[id]`, else the `turn` of the matching entry in
   `subproblems`, else, assuming one assistant turn per subproblem, its 1-based
   position in `subproblems`.

The protocol records `windows` (first/last position of each), `window`,
`window_kinds`, and `commit` (`{trial: {subproblem, turn, turn_source}}`), so the
choice is auditable later. A turn that cannot be located (short completions are
never searched for) gets no `commit` window rather than a guessed one.

## The synthetic task

`examples/inspect/scicode_synth/task.py` is a **synthetic** SciCode-style task we
wrote to test this pipeline. It is not a benchmark, measures nothing, and must not
be trained on (it carries a canary string).

* Two problems with three subproblems each: a small spectral toolkit (FFT, band
  energy, dominant frequency) and cross-track error (direction, signed distance,
  side counts). Later steps call earlier answers.
* Each problem has a **trap** variant that leaves one convention unstated (numpy FFT
  normalization; the sign of the cross-track error) and a **control** twin that
  states it. The twins share `pair_id`; `variant` and `convention_stated` mark them.
* The `stepwise` solver asks one subproblem per turn and carries the whole
  conversation (with reasoning) forward, so the frozen trace has one turn per
  subproblem.
* The `subproblem_tests` scorer runs each hidden test, after all earlier answers,
  in the Inspect sandbox (`local` by default; `-T sandbox_type=docker` works if you
  have it). Score value is the fraction passed; metadata has per-subproblem
  pass/fail, `subproblem_turns`, and `first_failing_subproblem`.

## Limitations

* **No tool emulation yet.** Agentic solvers (`basic_agent`, `react`, anything that
  offers tools) fail immediately with `NotImplementedError`. Use text-protocol
  solvers like `stepwise`.
* **No streaming**, one choice per call, no stop sequences, text content only.
* Replay is token-exact only for what the gateway reports; a sample that errors
  mid-turn keeps every completed turn but not the failed one.
* The loop signal `repeat_calls` is always 0 (there are no tool calls); the n-gram
  ratio is computed as in Harbor.

## Next steps

* Tool emulation for agentic solvers: render `ToolInfo` into the text protocol
  (as Harbor's fenced bash block does), parse calls back into `ToolCall`s.
* Real SciCode task sets: they plug in as long as their scorer writes
  `first_failing_subproblem` (and ideally `subproblem_turns`) into score metadata.
