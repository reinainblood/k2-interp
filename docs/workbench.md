# Private workbench

The React workbench is a separate CPU Modal app, `k2-interp-workbench`, in **FDR**.
Its authenticated requests call the existing full-weight `NativeWorker.live_chat`
method. Updating the web app does not redeploy the model worker. The controller
uses one container so session-file updates are serialized. Model workers remain
one-per-checkpoint, scale-to-zero, and independently deployable.

## Deploy

```bash
uv sync --extra modal --extra dev
npm --prefix web ci
npm --prefix web run build
uv run python scripts/setup_workbench.py
# Commit runtime changes before deploying the worker so captures record its revision.
# Retain K2_REUSE_VOLUME on installations prepared from a read-only legacy cache.
K2_REUSE_VOLUME=k2-horizon-bf16-weights uv run k2-interp deploy --checkpoints sft_3_1500
K2_INTERP_CODE_REVISION=$(git rev-parse HEAD) uv run modal deploy --env=FDR -m k2_interp.modal_workbench
uv run python scripts/open_workbench.py
```

`setup_workbench.py` creates one project-specific signing secret in FDR and
preserves an existing secret. `open_workbench.py` uses your existing Modal SDK
credentials to issue a random, one-use, 90-second ticket. The browser consumes it
from the URL fragment, immediately removes the fragment, and obtains a signed
12-hour HttpOnly/Secure/SameSite=Strict cookie. No Modal credential reaches the
browser. No workspace proxy token is created. Unauthenticated clients can only
load static application assets; every data, streaming, export, and model route
requires a valid session. This is a single-owner research application, not a
multi-tenant authorization system. Anyone authorized to issue tickets with the
workspace SDK can access this workbench's private research data.

To expose additional **already prepared and deployed** checkpoints in the picker:

```bash
K2_WORKBENCH_CHECKPOINTS=sft_3_1500,rl_merged,mid_4_6000 \
  K2_INTERP_CODE_REVISION=$(git rev-parse HEAD) \
  uv run modal deploy --env=FDR -m k2_interp.modal_workbench
```

Do not redeploy a model worker during active runs. Web-only changes can be
deployed separately; a reconnect reattaches to the detached model call.

## Interaction

- Create a session, choose its checkpoint and system prompt, then send a message.
- **System / Reasoning / Capture** edits the system prompt and native reasoning
  effort (`high`, `medium`, `low`), output-token limit, temperature, top-p, seed,
  context limit, residual layers, and exact expert-output L2 capture. Apply a
  system edit before the next turn. Active-turn settings are locked. Every run
  manifest preserves the effective system prompt, full message history, thinking
  fields, and generation configuration.
  Browser defaults use IFM's high effort, temperature 1.0, and top-p 0.95;
  the 512-token output limit is a separate, editable operational bound.
- A turn selects its own trace. **Follow** tracks the last input token being
  processed. Clicking a token loads that exact position from verified tensor
  artifacts; the layer input and heatmap select the expert detail view.
- **Capture** downloads a ZIP with source safetensors, metadata and hashes,
  exact token/layer selection, a real data-derived `router.svg`, full conversation
  archive, and separate as-of capture context. Tags are optional; UTC timestamps,
  run IDs, token positions and layer IDs are automatic. The raw capture scope is
  one model forward (all captured layers and positions), not every tensor from
  an entire multi-turn session. Existing CLI downloads remain available.
- **Stop** requests cooperative cancellation at a native layer boundary. Closing
  a tab does not cancel work. Loading weights is not immediately interruptible;
  a stop requested while loading takes effect before inference begins.

## Measurement semantics

The live publisher coalesces UI snapshots at about 400 ms; the server sends
updates at about 500 ms. These are delivery intervals, not a promised token
latency. Cold starts can take several minutes. There is no synthetic progress
percentage, animation of unobserved tokens, or fabricated routing data.

**Prefill is parallel over prompt positions.** During a live forward, the matrix
shows the last input position and fills as real layer hooks return. After that
forward commits, every captured input position can be inspected. Decoding is
one token per forward. A captured row at position i processes input token i and
predicts the next token. The final generated token is also processed and saved.

Router plots display logical expert IDs, native selected weights, or unbiased
sigmoid scores. Weights include the native scaling factor (sum approximately
2.5); sigmoid scores are not categorical probabilities. H-sigma is entropy after
explicitly normalizing sigmoid scores. Delta-k is the corrected selection-score
gap at the top-k boundary. Forward load counts all input positions of the current
forward and is omitted for single-position historical slices.

Expert output L2 is the norm of the **weighted native expert output**, calculated
in float32 from the native weighted vector. It is collected only at selected
layers. It is neither a routing probability nor causal importance. Residual L2
is computed from captured residual tensors; only measured layers have points.
Uncollected values remain absent, distinct from measured zeros. Dense layers
0-2 have no MoE routing data. No SAE features or causal edges are invented for
these views; those require the separate analysis/training/intervention pipeline.

Raw tensors and one projection per completed forward persist on the private
artifact volume. Live transport state uses a private Modal Dict; its loss does
not delete captures. Per-forward commits are required for browser inspection and
downloads. Session history lives on a separate private volume. Failed/cancelled
turns are retained in the UI/archive but do not silently enter subsequent model
context. Context and capture-byte bounds fail explicitly; no history truncation
or tensor downsampling is silently applied.

## Validation

```bash
uv run pytest -q
uv run ruff check src tests scripts
npm --prefix web test
npm --prefix web run build
```

Unit fixtures use small random instances of the pinned native model code. They
validate software behavior, not full-checkpoint scientific findings. Full-weight
HTTP/browser verification must be reported separately with its actual run IDs,
source revision, checkpoint revision, and private receipts.
