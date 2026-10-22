# Harbor: agentic trajectories with frozen token identity

k2-interp can act as the model behind a [Harbor](https://docs.harborframework.com/)
agent, and can turn the Harbor job that results into teacher-forced capture
requests. This is the two-pass design for studying behavior that only appears in
long agentic runs: first collect behavior, then collect internals on exactly the
tokens the model produced.

```
Pass A   harbor run -a k2_interp.harbor:K2Agent ...   -> trajectories + verifier labels
Pass B   k2-interp harbor-replay <job> --stage b1|b2|b3 -> RunSpecs; k2-interp run ...
         k2-interp router-features <run> --output ...  -> per-token routing table
```

Pass B replays identical token ids on every checkpoint, which is what
[datasets](artifacts.md) and crosscoders require. Nothing is regenerated.

## Environments

Harbor needs Python 3.12+; this package's lock is 3.11. Run Harbor from its own
environment with k2-interp installed beside it (`pip install -e .` is enough; the
adapter imports no torch). Run `harbor-replay` and `router-features` from the
k2-interp environment. The two sides meet only through the Harbor job directory.

## Pass A: the adapter

`K2Agent` is a Harbor `BaseAgent`. Harbor owns the task container, the timeout,
artifact collection, and the separate verifier; the adapter owns the loop. It
calls one deployed checkpoint over `/v1/chat/completions` with Modal proxy
credentials and drives the container through Harbor's environment API.

```sh
export K2_INTERP_ENDPOINT=https://<your-app>.modal.run
export MODAL_PROXY_TOKEN_ID=... MODAL_PROXY_TOKEN_SECRET=...
harbor run -p tasks -a k2_interp.harbor:K2Agent \
  --ak checkpoint=sft_3_1500 --ak seed=7 --ak max_tokens=8192 \
  --ak temperature=1.0 --ak top_p=0.95 \
  -n 1 --jobs-dir results --job-name a0-sft3
harbor agent schema k2_interp.harbor:K2Agent    # every option and its default
```

The chat endpoint has no native tool calling, so the protocol is text: at most
one fenced `bash` block per assistant turn, executed in the container, with
stdout and stderr returned as the next user message (truncated to
`output_chars`). The run ends when the assistant writes the submit token
(default `SUBMIT`) on its own line, after `max_turns`, after `max_idle_turns`
consecutive turns with neither a block nor the token, or at Harbor's timeout.

Per trial, under the agent logs Harbor collects (`<trial>/agent/k2trace/`):

| File | Contents |
| --- | --- |
| `trajectory.jsonl` | one record per turn: visible content, native reasoning verbatim, tool call, exit code and output sizes, usage, the server `run_id`, timestamps; a final `adapter` record with the stop reason |
| `frozen.json` | per turn: exact prompt and completion token ids, logprobs, finish reason; plus `final_sequence` (the last turn's prompt with its completion appended) and `final_prompt_length` |
| `adapter.json` | endpoint, checkpoint, sampling, limits, adapter version, Harbor session and context ids |

The same token identity is written to Harbor's `AgentContext.rollout_details`, so
`result.json` carries it too. `frozen.json` is rewritten after every turn; a trial
that times out still has every completed turn.

Token ids come from a new `k2_interp` block in every chat response:
`input_ids`, `output_ids`, `output_logprobs`, `finish_reason`, `mode`, `run_id`,
and `captured`. `ChatRequest` also accepts an optional `capture`
(`CaptureSpec`); with `--ak capture_router=true` the adapter asks for summary
routing fields on every turn, which costs trace storage but gives routing during
free generation rather than replay.

Assistant turns are appended to the history exactly as the server rendered
them, including `reasoning_content`; K2's template requires that field on every
history turn.

Defaults follow IFM's recommended sampling (temperature 1.0, top-p 0.95, high
reasoning effort) with a fixed seed. Greedy decoding is available but produces
repetition loops that look exactly like an impasse; use it knowingly.

Checkpoints without a trained reasoning channel (mid-training ones) will not
close a think tag the template opened, so their "reasoning" is ordinary
continuation and their `content` may be empty; they may also not follow the bash
protocol at all and stop as `idle`. That is a finding about those checkpoints,
not an adapter failure. Replay (Pass B) is unaffected because it only needs tokens.

## Pass B: replay with capture

`harbor-replay` reads a Harbor job directory, keeps every trial that has
`k2trace/frozen.json`, attaches the verifier outcome as sample labels, and
writes one `RunSpec` per checkpoint (or per checkpoint and trial):

```sh
k2-interp harbor-summary results/a0-sft3
k2-interp harbor-replay results/a0-sft3 --stage b1 \
  --checkpoints sft_3_1500 rl_merged mid_4_6000 --run-prefix impasse-a0 --output specs/b1
k2-interp harbor-replay results/a0-sft3 --stage b2 --window 256 \
  --checkpoints sft_3_1500 --run-prefix impasse-a0 --output specs/b2
k2-interp harbor-replay results/a0-sft3 --stage b3 --layers 12 18 30 \
  --checkpoints sft_3_1500 --run-prefix impasse-a0 --output specs/b3
k2-interp run specs/b1/sft_3_1500--impasse-a0-b1-all.json --detach
```

`--submit` spawns each spec on its checkpoint app directly. `--outcomes bailed wrong`
restricts to those labels. `--checkpoints spine` and `all` expand as in `batch`.

| Stage | Capture | Positions | Use |
| --- | --- | --- | --- |
| `b1` | router only, `router_fields` = `expert_ids`, `expert_weights`, `routing_margin`, `router_entropy` | all | cheap whole-trajectory routing features |
| `b2` | `resid_post`, `moe_out` at all 58 MoE layers (or `--layers`) | the impasse, early, and midpoint windows (`--window`) | direction finding, dictionaries, crosscoders |
| `b3` | `resid_post`, `moe_out` at `--layers` | all | per-token curves on a frozen layer set |

Each sample teacher-forces `final_sequence[:-1]` with the last token as the
continuation, so prefill captures every position in one forward. Windows are
absolute positions in `final_sequence`: `impasse` is the last `--window` tokens
of the final completion; `early` is tokens `window..2*window` of the first
completion when it can be located in the final prompt; `midpoint` straddles the
middle of the sequence. Earlier turns are searched for inside the final prompt
only when at least 8 tokens long, because short completions recur by chance.

Labels on every sample: `outcome`, `benchmark`, `split`, `case_id`, `turns`,
`final_prompt_length`, `stop` (`submitted`, `idle`, `max_turns` from the adapter;
`timeout` or `error` from Harbor, so infrastructure endings never pass for the
model's own decision), `length_hits` (turns that hit the per-request token cap),
and the verifier's `verifier_status` and `valid`. Template groups are one per
task, so resampled runs of the same case never straddle dataset splits.

| `outcome` | Meaning |
| --- | --- |
| `bailed` | no solution file was written |
| `wrong` | the verifier rejected the submission, or crashed on it |
| `empty` | schema-valid but did no work (a coverage, completion, track, selection, or service metric is exactly zero) |
| `solved` | valid with nonzero work |
| `unscored` | no verifier report |

Benchmark `train*` splits map to k2-interp `train`, `test*` to `test`, so the
dataset builder's split rules apply unchanged.

## Router features

`router-features` flattens a router capture into one row per (sample, position,
layer) with `router_entropy`, `routing_margin`, `expert_ids`, `expert_weights`,
the token id, and the sample labels, as JSONL and a compressed `.npz`:

```sh
k2-interp download sft_3_1500/impasse-a0-b1-all artifacts/b1
k2-interp router-features artifacts/b1 --output features/sft3-b1
```

Shard hashes are verified while reading. `k2_interp.features.window_summary`
gives per-window, per-layer means of entropy and margin for quick comparisons.

## Storage

Full router capture (`router_fields` unset) stores three 192-wide vectors per
token per MoE layer, roughly 140 KB per token; the summary fields are about 6 KB
per token. `b2` residual windows cost `3 × window × 2 sites × 58 layers × d_model × 2 bytes`
per trajectory and checkpoint. Set `--max-bytes` from a pilot measurement.
