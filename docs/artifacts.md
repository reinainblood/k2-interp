# Artifact contract

`k2-interp/1` is the trace contract. JSON contains metadata; safetensors contains
numeric arrays. Raw conversations and generated text are private artifact data.

```
<checkpoint>/<run-id>/
  manifest.json
  samples/<sample-id>/
    attempt-<uuid>/
      step-000000-shard-000000.json
      step-000000-shard-000000.safetensors
      progress.json
      result.json
    complete.json
```

The manifest hashes the complete request and model identity. Reusing a run ID with
changed inputs, code, or checkpoint fails. Completion is recorded after all shards
and result metadata are written. The Modal Volume commits after each forward pass
and completed sample. Resume validates hashes and skips only completed samples.
An interrupted sample starts a new attempt; partial attempts remain available.
Generation does not resume halfway through a KV cache.

All layer/site records for one forward are packed into one safetensors shard and
one JSON sidecar, with per-record layer/position metadata. This avoids creating
thousands of files per response. `iter_records` exposes the same logical records
for packed shards and the earlier one-record-per-shard format. Capture byte limits
are enforced while buffering, before serialization; each completed forward is flushed.

## Token alignment

Each row records the token **being processed** at absolute position `i`, which
contributes to predicting token `i+1`. Prefill records requested prompt rows. Decode
and teacher forcing process one continuation token per step. A final pass captures
the last output token even though its next prediction is not sampled.

Positions include chat-template tokens and are nonnegative absolute indices.
Results retain input/output IDs, token text, mode, and output log probabilities.
Text continuations are tokenized separately with `add_special_tokens=False`, then
appended. This may differ from tokenizing concatenated raw strings; use explicit
`input_ids` and `continuation_ids` for exact boundary control.

## Router tensors

Keys are `layers.<L>.<field>`, with zero-based layers. Dense layers 0–2 have no router.
Layers 3–60 select eight of 192 routed experts per token.

| Field | Per-row shape | Meaning |
| --- | --- | --- |
| router_logits | 192 | Native unbiased projection, promoted to float32 |
| router_scores | 192 | Sigmoid scores |
| selection_scores | 192 | Scores plus selection-only correction bias |
| expert_ids | 8 | Logical expert indices |
| expert_weights | 8 | Unbiased selected scores, normalized and scaled by 2.5 |
| routing_margin | scalar | Selection-score rank 8 minus rank 9 |
| router_entropy | scalar | Entropy after normalizing sigmoid scores to a distribution |

Observation reconstructs selected IDs/weights from returned logits with the native
selection algorithm. Optional expert-output hooks check execution row mapping.
Routing interventions record their effective IDs/weights. The shared expert is an
always-executed additional MLP, not a ninth routed expert; no routing gate is invented.

## Activation sites

- `resid_pre`: block input before attention normalization.
- `attn_out`: attention output before residual addition.
- `resid_mid`: attention residual sum before MLP normalization.
- `moe_in`: normalized MLP/MoE input.
- `moe_out`: full MLP/MoE output, including the shared expert.
- `shared_out`: shared-expert output alone.
- `resid_post`: block output.

Weighted expert outputs are dense selected-position matrices, zero where an expert
was not selected. They are expensive and opt-in. Their sum plus `shared_out` matches
the native sparse MLP output within accumulation precision.

## Datasets and dictionaries

Views join by sample ID and exact positions. Entire token sequences must match
across checkpoint views, rejecting divergent free generations. Source identities,
request hashes, splits, template groups, dimensions, and shard hashes are retained.
A template group in multiple splits is rejected.

Dictionaries contain safetensors weights and JSON provenance, normalization,
calibrated threshold, training steps, and held-out metrics. Optimizer resume uses
`torch.load(weights_only=True)`. Hashes establish identity, not the trustworthiness
of an external artifact's origin.
