# scicode_synth — SYNTHETIC pipeline fixture

**Not a benchmark. Do not train on this.** `k2-interp-canary: 7c1f0e2a-scicode-synth-do-not-train`

A tiny SciCode-style Inspect task written by the k2-interp authors to test the
`k2interp` provider, its frozen traces, and `k2-interp inspect-replay`. Two problems ×
three subproblems, each with a convention-ambiguous `trap` variant and a `control`
twin that states the convention (shared `pair_id`). See [docs/inspect.md](../../../docs/inspect.md).

```sh
uv run inspect eval examples/inspect/scicode_synth/task.py --model k2interp/sft_3_1500
```
