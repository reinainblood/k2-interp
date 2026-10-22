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

Final verification of the packed trace format and the neutral dictionary/graph
integration pipeline is in progress. Full-model receipts are kept in private artifact
storage; this page reports only reviewed software-verification metadata.

## Checkpoint inventory

[The metadata report](checkpoint-verification.json) verifies all eleven immutable
checkpoint pins, complete 61-shard inventories, and compatible geometry. Their model
implementation and tokenizer hashes agree. Only metadata/configuration are checked
by that report; it does not claim inference or full weight hashing for all eleven.
