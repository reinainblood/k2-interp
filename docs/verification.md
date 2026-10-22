# Verification

The local suite uses the actual pinned IFM model source at random, reduced dimensions.
It checks capture/logit parity, router semantics, hook cleanup, cached teacher forcing,
token alignment, resume/integrity, all dictionary modes, batch-independent inference,
gradient-versus-finite effects, reconstruction insertion, and router comparison.

Environment: Python 3.11, PyTorch 2.13.0, Transformers 5.13.0, repository `uv.lock`.
GitHub Actions runs these checks plus lint, formatting, packaging, and CLI inspection.

## Full-weight reference receipt

On October 21, 2026, SFT3 revision
`cfd157f11804b290c47521311874fa24870c5b1e` passed the full-weight reference smoke
at runtime commit `6f8371361d37b50268c1db6c310432a07cec9c24`, using eight H200 GPUs,
BF16, and native whole-layer model parallelism.

- Capture preserved output logits exactly.
- All 58 MoE layers and every processed token position were captured.
- Hash-verified completed samples resumed without rerunning.
- The identity intervention changed the registered log-probability metric by zero.
- A final-layer ablation changed the fixed factual-continuation metric by
  approximately -12.0379 nats.
- Unmodified output returned exactly after intervention cleanup.
- Seeded generation replayed the same output tokens.

The checks after loading took about 25.8 seconds. This is a runtime test on a neutral
factual prompt, not a study result or a throughput benchmark.

## Packed-format and pipeline verification

The packed-format smoke passed on the same full SFT3 checkpoint at runtime commit
`cad37e7c48c41e36f8d30af9f42e95d3b29748e7`, with sequential weight transfers and
CUDA 13.0. All checks above passed again, including the identical -12.0379-nat
ablation delta. Post-load checks took about 19.7 seconds.

The neutral integration exercise also passed on that deployed runtime:

- Three factual teacher-forced examples captured and summarized.
- Matched residual localization on the actual layer-30 residuals.
- SAE and skip-transcoder datasets built, trained on an A100-80GB, and reloaded.
- Reconstruction insertion evaluated inside the original 375B model.
- Gradient × feature attribution computed through the original model.
- Finite feature interventions produced a total-effect graph, with an exactly zero
  identity-dose effect and successful unmodified-model cleanup.

The dictionaries received only **two training steps**. This verifies the software
path, not dictionary quality or scientific interpretability. The cohort has three
examples and is unsuitable for a scientific conclusion. Checkpoint crosscoders are
covered by numerical/integration tests on aligned synthetic views; a full-size
multi-checkpoint crosscoder study has not been run.

## Native chat verification

Two live SDK chat turns passed at runtime commit
`51d848be277fb0bbb22153add721b19379c8b34a`. The responses contained 42 and 27 tokens,
both ending at EOS. The second turn accepted the first turn's native thinking field.
The tokenizer boundary explicitly requests a flat token-ID sequence, and native
IFM thinking delimiters are preserved across history rendering.

The final retention change at `13ea557a6e915adfc029c3c3a7e659c5b2372404` adds exact
processed-token IDs/log probabilities to partial checkpoints. It is covered by a
native-code interruption regression test. The full-weight receipts above predate
that metadata-only change; no additional full-weight replay is claimed for it.

## Current checks and scope

- 26 tests pass against pinned native model/tokenizer source and synthetic data.
- Ruff, formatting, wheel/source builds, and GitHub CI pass.
- HTTP anonymous denial is verified. Authenticated HTTP is not verified in the
  execution workspace because no new proxy token could be created at its quota.
  SDK/CLI authentication and real model calls are verified independently.
- Raw full-model receipts and outputs remain in private artifact storage. This page
  reports only reviewed software-verification metadata.
- No private behavioral study has been run, and no concept/circuit discovery is claimed.

## Checkpoint inventory

[The metadata report](checkpoint-verification.json) verifies all eleven immutable
checkpoint pins, complete 61-shard inventories, and compatible geometry. Their model
implementation and tokenizer hashes agree. Only metadata/configuration are checked
by that report; it does not claim inference or full weight hashing for all eleven.
