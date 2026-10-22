# K2 Interp

[![Runtime validation](https://github.com/reinainblood/k2-interp/actions/workflows/ci.yml/badge.svg)](https://github.com/reinainblood/k2-interp/actions/workflows/ci.yml)

A reproducible interpretability runtime for **IFM K2 Horizon 375B**. Run chat,
teacher-forced evaluations, and interventions across the released checkpoint tree,
then collect aligned router traces and activations for sparse dictionary analysis.

The runtime is experiment-agnostic. Prompts, chats, traces, trained dictionaries,
and private study results stay outside Git. Examples use public factual questions.
No scientific finding is implied by a successful software test.

## Implemented methods

| Stage | Runtime support | Evidence produced |
| --- | --- | --- |
| Native routing | Sigmoid scores, selection bias, top-8 logical IDs and weights, optional expert outputs | Per-token/per-layer safetensors and JSON |
| Residual localization | Selected residual/MoE/attention sites, explicit matched pairs | Difference directions, grouped bootstrap intervals, route divergence |
| Targeted SAE | BatchTopK training and fixed-threshold inference | Dictionary, resume state, held-out reconstruction/L0 metrics |
| Skip-transcoder | Sparse predictor plus affine skip from MoE input to output | Dictionary and original-model fidelity evaluation |
| Checkpoint crosscoder | Joint sparse encoding of aligned checkpoint views | Shared codes and separate decoder slices |
| Attribution/interventions | Gradient × feature attribution, finite feature effects, router/residual/expert patching | Signed graphs, log-probability deltas, KL, cleanup receipt |

Crosscoders jointly describe multiple checkpoint views; their encoding is acausal.
These graphs operate on the original model. They do **not** implement a full-model
CLT replacement graph or establish direct mediation. See [methodology](docs/methodology.md).

## Install

```sh
git clone https://github.com/reinainblood/k2-interp.git
cd k2-interp
uv sync --python 3.11 --extra modal --extra dev
uv run k2-interp checkpoints
uv run pytest -q
```

The lock uses PyTorch 2.13.0 and Transformers 5.13.0. Tests download the pinned IFM
model source/config and instantiate random, reduced-size models with that exact
implementation. Full 375B weights are required for scientific conclusions.

## Deploy one checkpoint or all eleven

Configure Modal credentials first. Every operation explicitly uses `FDR`.

```sh
# Registers an authenticated API and scale-to-zero worker. No model loads yet.
uv run k2-interp deploy --checkpoints sft_3_1500

# Downloads and hashes weights on CPU; interrupted downloads resume.
uv run k2-interp prepare --checkpoint sft_3_1500 --detach
uv run k2-interp call <returned-call-id>

# Bounded full-weight runtime verification on eight H200s.
uv run k2-interp smoke --checkpoint sft_3_1500 --detach

# Register all independent checkpoint apps. Prepare each before invoking it.
uv run k2-interp deploy --checkpoints all
```

Each worker uses **eight H200 GPUs**, native BF16, and whole-layer model parallelism.
This reference path is slower than production tensor/expert-parallel serving.
No vLLM/NNsight parity claim is made. Listing checkpoints or importing the package
never allocates GPUs. Deployment requires committed source for reproducible provenance.

Each checkpoint permits one model worker. Idle workers scale down after five minutes;
change this with `deploy --idle-seconds N` (2–3600). Active jobs use their run timeout.
The package does not stop other apps or delete model snapshots.

To reuse a cache, set `K2_REUSE_VOLUME=<existing-volume>` when deploying and use
`prepare --reuse-subdir <snapshot-directory>`. All 61 weight hashes must match.
The cache is read-only; model code and configuration still come from the pinned
revision. Keep that volume setting on subsequent deployments because the prepared
snapshot references it. Without reuse, preparation downloads a self-contained snapshot.

## Chat, evaluations, and standardized captures

```sh
uv run k2-interp chat --checkpoint sft_3_1500 --max-new-tokens 512
uv run k2-interp run examples/capture.json --detach

# Same protocol across the six-checkpoint spine, sequentially.
uv run k2-interp batch examples/capture.json --checkpoints spine

# Explicit concurrency for multiple full-size model workers.
uv run k2-interp batch examples/capture.json --checkpoints all --parallel 3

uv run k2-interp download sft_3_1500/example-facts artifacts/example-facts
uv run k2-interp evaluate artifacts/example-facts --output artifacts/evaluation.json
```

CPU preflight rejects incomplete snapshots before starting a model worker. `RunSpec`
accepts plain text, chat messages, or exact token IDs. A continuation selects teacher
forcing; otherwise the recorded sampling configuration controls generation. Context
is never silently truncated. Exact-match grading is enabled only by an explicit
`labels.expected_exact` value. Arbitrary task grading is not inferred from text.

Chat turns are saved privately, with raw output and reasoning preserved. `:quit`
ends the client; the worker follows its idle policy. The HTTP API uses Modal proxy
authentication and supports non-streaming `/v1/chat/completions` plus asynchronous
`/runs` requests. SDK calls use existing Modal credentials.

## The analysis pipeline

1. `run` captures router data at all sparse layers and selected residual/MLP sites.
2. `localize` analyzes explicit clean/control pairs; `compare-routes` compares
   identical token sequences across checkpoints.
3. `dataset` joins input/target views; `train` with `kind: sae` reconstructs one site.
4. `kind: skip_transcoder` predicts `moe_out` from `moe_in` with an affine skip path.
5. `kind: crosscoder` jointly encodes two or more aligned checkpoint views.
6. `fidelity`, `attribute`, `graph`, and `intervene` test hypotheses on the original model.

CPU analysis and dictionary training can run beside the private traces:

```sh
uv run k2-interp analyze examples/dataset-sae.json
uv run k2-interp train examples/train-sae.json --remote --detach
uv run k2-interp intervene examples/intervene.json --checkpoint sft_3_1500
uv run k2-interp fidelity examples/fidelity.json --checkpoint sft_3_1500
uv run k2-interp attribute examples/attribute.json --checkpoint sft_3_1500
uv run k2-interp graph examples/feature-graph.json --checkpoint sft_3_1500
```

The factual example exercises software plumbing; it is far too small for scientific
dictionary training. Use broad training data and independent prompt families for
validation. See [protocol](docs/protocol.md), [artifacts](docs/artifacts.md), and
[operations](docs/operations.md).

The feature examples deliberately select an arbitrary latent/token to demonstrate
the API. Select actual candidate latents and positions from your discovery data;
latent 0 and token 0 have no preassigned semantic meaning.

## Checkpoints and reproducibility

The pinned registry contains Mid3, Mid4, all five RL specialists, merged RL, and
SFT1/2/3. Immutable commits, all LFS weight hashes, tokenizer/config/source hashes,
runtime source hash, and dependency versions travel with each run.

IFM documents Mid4 → RL specialists → merged RL → SFT1 → SFT2 → SFT3. Mid4 already
mixes in agentic/reasoning SFT-style data. Mid4–SFT3 measures the combined later
training path, not SFT alone. [Official training overview](https://huggingface.co/IFM/K2-Horizon-375B-A23B).

See [verification](docs/verification.md) for tested scope and full-model receipts.
Source is Apache-2.0; upstream weights/code retain their own attribution. See
[NOTICE](NOTICE). No command publishes study data to GitHub or Hugging Face.
