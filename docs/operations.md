# Operations

Every new resource is explicitly scoped to Modal `FDR`. Each checkpoint app has one
maximum model worker, CPU preparation and analysis functions, an A100-80GB training
function, and a protected HTTP API. Deploying all apps does not load all weights;
concurrent model invocations can allocate eight H200s per checkpoint.

Weights live on `k2-interp-weights`, traces on `k2-interp-artifacts`. Both are private
workspace Volumes. Cache reuse is opt-in/read-only. CPU preparation hashes all shards.
GPU startup checks the receipt, sizes, and source/config hashes without rehashing
758 GB. External modifications to a reused cache invalidate its immutability;
use a dedicated snapshot for archival work.

## Access and privacy

SDK calls use existing Modal credentials. The HTTP API requires `Modal-Key` and
`Modal-Secret` proxy-auth headers. Never put them in request files or Git. Gateway
health means the API is available, not that model weights are loaded. The API accepts
structured requests and does not execute client-supplied Python.

`/v1/chat/completions` supports non-streaming responses with parsed assistant content
and reasoning. Raw output remains private. `/runs` returns a call ID, polled with
`k2-interp call` and existing Modal credentials.

## Resume and lifetime

Complete samples are skipped only after hash validation. Incomplete attempts remain
and rerun from their start. Causal sweeps checkpoint completed patches. Training saves
optimizer, model, epoch, step, and deterministic batch cursor. Idle timeout defaults
to five minutes and can be adjusted for chat. The package contains no fleet cleanup.

Use `k2-interp status` to identify exact app IDs. If you stop an app through Modal,
committed Volumes remain. Avoid broad stop patterns that could affect unrelated services.

Remote CPU/training paths are relative to `/artifacts`; model-specific causal paths
are relative to `/artifacts/<checkpoint>`.

```sh
uv run k2-interp download sft_3_1500/example-facts artifacts/example-facts
uv run modal volume put --env=FDR k2-interp-artifacts artifacts/my-dictionary /sft_3_1500/dictionaries/my-dictionary
```

Use new IDs for changed inputs or code. Export reviewed artifacts with checksums
before deleting private storage. Record current billing rates and container receipts
before increasing concurrency; this package does not assume a fixed GPU price.
