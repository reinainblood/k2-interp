# Deploying the checkpoints

This guide starts with an ordinary clone and a Modal account. You do not need access
to the maintainers' workspace, caches, or conversations. You do need capacity for
one eight-H200 node for each checkpoint you want to run at the same time.

Start with one checkpoint. Registration, weight preparation, and inference are
three separate operations. A deployed URL is not evidence that a model has loaded.

## What comes from IFM, and what this project adds

The upstream model is [`IFM/K2-Horizon-375B-A23B`](https://huggingface.co/IFM/K2-Horizon-375B-A23B).
IFM's model card is the authority for its training chronology and checkpoint names.
Their linked [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/IFM/K2-Horizon)
and [vLLM recipes](https://recipes.vllm.ai/IFM) are the starting points for optimized
serving. Read the recipes' version and validation requirements; installing a
package with a familiar name does not establish K2 support.

K2 Interp is an independent instrumentation runtime, not an IFM-maintained serving
recipe. Its native Hugging Face implementation makes routing, activations, and
interventions accessible. These implementation differences are deliberate:

| Setting | Upstream guidance / recipe | K2 Interp reference worker |
| --- | --- | --- |
| Model identity | Select an upstream revision | A named registry entry resolves to an immutable commit; no floating `main` |
| Model weights | BF16 | BF16; no automatic quantized or smaller-model fallback |
| Placement | Eight-way tensor parallelism for 375B in the SGLang base recipe | Eight H200s with complete decoder layers assigned to devices; not TP/EP |
| Attention | FlashAttention-3 in the validated SGLang recipe | PyTorch SDPA |
| Loading | Multithreaded loading disabled for 375B in SGLang | Sequential materialization through `HF_DEACTIVATE_ASYNC_LOAD=1` |
| Transformers | Model card records validation with 4.57.6 | This repository locks 5.13.0 and tests its own loader; see `uv.lock` |
| Thinking | Native K2 reasoning format | Exact raw output plus parsed assistant thinking fields |

Do not copy an SGLang or vLLM command into this project and expect its internal
captures to appear. Those are different execution paths. Equally, do not attribute
this runtime's latency or numerical behavior to IFM's optimized recipe. The
[verification record](verification.md) identifies what has actually been tested.

## Choose a checkpoint

The current registry covers eleven named releases, not every intermediate
checkpoint IFM has published:

| Registry name | Comparison role |
| --- | --- |
| `mid_3_3500` | Earlier context-extended checkpoint |
| `mid_4_6000` | Final midtraining checkpoint |
| `rl_knowledge_work` | Knowledge-work RL specialist |
| `rl_if` | Instruction-following RL specialist |
| `rl_search` | Search RL specialist |
| `rl_tool-use` | Tool-use RL specialist; preserve the hyphen |
| `rl_reasoning` | Reasoning RL specialist |
| `rl_merged` | Merged RL checkpoint |
| `sft_1_2400` | First SFT phase endpoint |
| `sft_2_6000` | Second SFT phase endpoint |
| `sft_3_1500` | Third SFT phase endpoint; first deployment recommended here |

The lineage is Mid3 → Mid4 → five RL specialists → merged RL → SFT1 → SFT2 → SFT3.
`rl_merged` precedes SFT3. Mid4 already incorporates agentic/reasoning SFT-style
data, so calling it an entirely SFT-data-free baseline would be misleading.
[IFM training overview](https://huggingface.co/IFM/K2-Horizon-375B-A23B#training-overview).

The live Hub refs were checked on October 22, 2026: all eleven named refs matched
the commits in [`registry.py`](../src/k2_interp/registry.py). The Hub API returned
these names as tags, although the model card calls its inventory entries branches.
Both are revision selectors; this project uses the immutable commit underneath.
Never resolve a wildcard such as `sft_3_*` as if it were a real checkpoint name.

```sh
uv run k2-interp checkpoints
uv run k2-interp lock --checkpoint sft_3_1500 --output .private/sft3-lock.json
```

The lock command reads Hub metadata; it does not download the full model or start
GPUs. The [checked inventory](checkpoint-verification.json) records 61 weight shards
and approximately 758 GB of weight files per checkpoint. Budget storage separately
from GPU memory, and leave room for captures and temporary download state. Eleven
independent snapshots are a multi-terabyte storage decision.

## Set up your workspace

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
git clone https://github.com/reinainblood/k2-interp.git
cd k2-interp
uv sync --python 3.11 --extra modal --extra dev
uv run pytest -q
uv run modal setup --profile my-research
uv run modal profile list
MODAL_PROFILE=my-research uv run modal environment list
```

Replace `my-research` with your own profile name. Confirm that it points at the
workspace whose compute and storage you intend to use. No credentials belong in
request JSON, shell transcripts, source files, or Git.

The current implementation explicitly selects an environment named `FDR` for all
its resources. That is a namespace in **your** workspace; it does not grant access
to anyone else's environment. If your chosen workspace does not already have it:

```sh
MODAL_PROFILE=my-research uv run modal environment create FDR
```

Environment and workspace are different selectors. Changing the active default
environment does not override the explicit `FDR` lookups in this package.
[Modal environment behavior](https://modal.com/docs/guide/environments).

## Register, prepare, then verify

From a clean, committed checkout:

```sh
git status --short
MODAL_PROFILE=my-research uv run k2-interp deploy --checkpoints sft_3_1500
MODAL_PROFILE=my-research uv run k2-interp status
```

Deployment registers `k2-interp-sft-3-1500`, its functions, and storage references.
It does not load model weights. A modified checkout is rejected by the deployment
CLI so a run can name the source commit that produced it. Commit your intended
source changes before deploying; do not discard unrelated work to get a clean tree.

Prepare the full checkpoint on CPU:

```sh
MODAL_PROFILE=my-research uv run k2-interp prepare --checkpoint sft_3_1500 --detach
MODAL_PROFILE=my-research uv run k2-interp call <prepare-call-id> --timeout 30
```

Replace the bracketed value with the returned call ID. `state: running` is not a
failure or a preparation receipt. Repeat the read until the call returns its result.
Preparation downloads the pinned code, tokenizer, configuration, and all weight
shards, verifies every weight hash, and commits `snapshot.json`. Interrupted
downloads can reuse verified files when preparation is run again.

Only after preparation succeeds, run a bounded full-model check:

```sh
MODAL_PROFILE=my-research uv run k2-interp smoke --checkpoint sft_3_1500 --detach
MODAL_PROFILE=my-research uv run k2-interp call <smoke-call-id> --timeout 30
```

This step starts the eight-GPU worker and is billable. Initial loading takes time;
inspect the exact app's logs before deciding it has failed. A successful smoke
checks capture parity, token alignment, deterministic replay, and intervention
cleanup. Run it for each checkpoint you intend to claim as validated. The repository's
historical full-weight results cover SFT3, not every entry in the registry.

The worker scales to zero after five idle minutes by default. Registering a model,
loading it, finishing a request, and releasing its GPU container are distinct states.
`deploy --idle-seconds N` configures an idle window between 2 and 3,600 seconds.
The model is not evicted halfway through a request merely because that interval passes.

## Match the intended chat conditions

IFM recommends high reasoning effort, temperature `1.0`, and top-p `0.95`.
`medium` and `low` are research variations, not their recommended chat settings.
[IFM best practices](https://huggingface.co/IFM/K2-Horizon-375B-A23B#best-practices).

K2 Interp's `GenerationSpec` defaults to greedy decoding (`temperature: 0`) for
reproducible software checks. That default is **not** the upstream sampling recipe.
For a bounded chat capture with IFM's recommended sampling settings:

```sh
MODAL_PROFILE=my-research uv run k2-interp run examples/chat-ifm.json --detach
```

Inspect [the request](../examples/chat-ifm.json) before using it. It records high
reasoning effort, temperature, top-p, seed, context bound, output limit, and capture
sites. Give changed prompts or settings a new `run_id`; an existing ID cannot be
quietly repurposed.

The example's 1,024-token output limit is deliberately bounded. The current runtime
permits up to 8,192 newly generated tokens per request. Those limits differ from the
longer output limit in IFM's example. A response that reaches the limit can contain
only unfinished reasoning; `finish_reason: length` must remain visible in the data.
The model's advertised context length is not a claim that this eight-GPU capture
configuration has been validated at that length. Input plus reserved output must
fit the request's explicit `max_context`; the runtime raises instead of silently
discarding conversation history.

For plain completion-style studies, provide `text` or exact `input_ids` rather
than `messages`. For chat, preserve the assistant's full native message, including
an empty `reasoning_content` when present. Do not rebuild history from visible
answer text alone. This interface is not an automatic tool-execution harness;
upstream tool-parser support does not imply this project executes tool calls.

## Add more checkpoints

Register a specific comparison panel:

```sh
MODAL_PROFILE=my-research uv run k2-interp deploy --checkpoints mid_3_3500 mid_4_6000 rl_merged sft_3_1500
```

Or register the entire current registry:

```sh
MODAL_PROFILE=my-research uv run k2-interp deploy --checkpoints all
```

Prepare each checkpoint separately using its exact name. `all` is supported by
deployment and batch selection; it is not a checkpoint accepted by `prepare`.
Each app selects its own pinned snapshot. This does not replace another checkpoint's
worker or rewrite an existing chat's checkpoint.

```sh
MODAL_PROFILE=my-research uv run k2-interp prepare --checkpoint mid_3_3500 --detach
MODAL_PROFILE=my-research uv run k2-interp prepare --checkpoint rl_merged --detach
```

After preparation and individual validation, `batch --checkpoints spine` compares
the six-stage spine sequentially. `--parallel 3` can start three full model workers,
or 24 H200 GPUs. Preparing eleven models is not permission to assume eleven active
workers fit your account's quotas or budget.

## Deploy the web controller independently

The workbench is a separate CPU app, `k2-interp-workbench`. Its enabled-checkpoint
list should contain only checkpoint apps you have prepared and validated. Build
the frontend with Node.js and npm installed:

```sh
npm --prefix web ci
npm --prefix web run build
MODAL_PROFILE=my-research uv run python scripts/setup_workbench.py
MODAL_PROFILE=my-research K2_WORKBENCH_CHECKPOINTS=sft_3_1500 uv run modal deploy --env=FDR -m k2_interp.modal_workbench
MODAL_PROFILE=my-research uv run python scripts/open_workbench.py
```

Use a comma-separated list to enable several checkpoints, for example
`K2_WORKBENCH_CHECKPOINTS=mid_3_3500,rl_merged,sft_3_1500`. Enabling a name does not
prepare or validate its weights. The login script uses your Modal SDK credentials
to issue a short-lived, one-use ticket and opens the browser. It does not print a
long-lived password. The static page can be public while session data and model
requests remain authenticated.

Changing the controller does not require redeploying the model app. Finish active
turns and make a backup before changing the model runtime itself. Deploying over
an active GPU worker is not a safe way to refresh a graph's styling.

## Reusing an existing weight cache

Fresh preparation is self-contained. Cache reuse is an optional optimization for
operators who already have a trusted checkpoint volume in the same environment:

```sh
MODAL_PROFILE=my-research K2_REUSE_VOLUME=my-existing-k2-weights uv run k2-interp deploy --checkpoints sft_3_1500
MODAL_PROFILE=my-research uv run k2-interp prepare --checkpoint sft_3_1500 --reuse-subdir <snapshot-directory> --detach
```

The source is mounted read-only at `/legacy`; all 61 shard hashes must match the
requested checkpoint. The new snapshot can contain symlinks into that mount.
Keep the same `K2_REUSE_VOLUME` setting on every subsequent **model** deployment,
and do not delete, rename, or modify the source snapshot while those links are used.
A backup of the new volume alone may not contain the linked weights. Record both
volume names and the snapshot directory in your operator notes.

## Before a real study

Confirm the immutable model revision, source commit, preparation receipt, full-model
smoke result, and actual generation settings. Make one private test conversation,
export it with reasoning, verify its checksums, and open the readable transcript.
Then follow [storage, exports, and recovery](data-safety.md) to back up both the
conversation and its raw captures. Establish that path before collecting data you
cannot afford to lose.
