# Methods and interpretation

The reference path uses the original IFM model in BF16, distributing whole decoder
layers across eight H200s. This is sequential model parallelism, not TP/EP. Automatic
CPU/disk offload is rejected. Observation uses temporary hooks. Routing interventions
use an identity-tested sparse forward because native routing calls functional linear
operations. Selection bias, expert order, casting, and shared-expert addition are
preserved.

Residual localization describes matched-pair differences with prompt-family bootstrap
intervals. It does not establish causality. Router divergence uses explicitly
normalized sigmoid scores. Expert IDs identify computations within each checkpoint;
unchanged IDs do not establish unchanged semantics across training.

## Dictionaries

The implemented architecture is **BatchTopK**: training allocates `k × rows` positive
activations across the batch; inference uses a fixed scalar threshold calibrated
on training rows. This is not a learned JumpReLU threshold or Matryoshka architecture.
Inputs/targets are centered per dimension and scaled by their view's training RMS.
Decoder columns are unit-normalized after updates. Training uses Adam, gradient
clipping, deterministic shuffling, and resumable optimizer/cursor state.

- SAE: reconstruct one activation view.
- Skip-transcoder: predict MoE output from normalized MoE input through sparse and
  affine skip paths.
- Crosscoder: jointly encode aligned checkpoint views and reconstruct separate
  decoder slices. This acausal description is not an intervention across two models.

Reports include MSE, variance explained, mean L0, and inactive latent fraction.
`fidelity` inserts reconstructions into the original model and measures continuation
loss increase and output KL. Reconstruction accuracy alone does not establish
behavioral or mechanistic faithfulness.

## Attribution and causal graphs

`attribute` differentiates fixed-continuation log probability with respect to native
sites, then projects gradients onto dictionary decoder directions. Signed attribution
is activation × sensitivity: a first-order local estimate with attention free to vary.

`graph` measures finite interventions on selected features. Its edges are changes
in downstream feature activations and output log probability. A node's operation is
`scale`, `set`, or `add`; the `factors` list supplies multipliers/values/increments.
The decoded delta is added to the original activation, preserving reconstruction
error. Transcoder codes read `moe_in` and write `moe_out`.

These are **total effects**, potentially including indirect paths. They are not
direct mediation edges or an Anthropic-style fixed-attention CLT replacement graph.
Nonorthogonal decoder directions can affect other re-encoded features. Crosscoders
are excluded from original-model causal-node APIs.

`intervene` independently patches residuals, raw router logits, expert choices,
or mixture weights. Changed choices recalculate native weights; weights can then
be explicitly replaced. User-supplied weights are not silently renormalized.
Every sweep repeats an unmodified baseline to check hook cleanup. Replication,
held-out prompts, random controls, dose response, and semantic specificity belong
in the study protocol; the runtime does not infer labels from heatmaps.

## References

- [IFM K2 model](https://huggingface.co/IFM/K2-Horizon-375B-A23B)
- [SAELens architecture guidance](https://github.com/decoderesearch/SAELens/blob/main/docs/training_saes.md)
- [BatchTopK](https://github.com/bartbussmann/BatchTopK)
- [Sparse crosscoders](https://transformer-circuits.pub/2024/crosscoders/index.html)
- [Circuit tracing](https://www.transformer-circuits.pub/2025/attribution-graphs/methods.html)
- [Gemma Scope 2](https://deepmind.google/models/gemma/gemma-scope/)
