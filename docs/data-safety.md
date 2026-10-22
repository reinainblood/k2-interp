# Storage, exports, and recovery

The conversation is research data. So are its reasoning, token IDs, generation
settings, and failed attempts. A screenshot is useful, but it cannot replace them.

The safest working habit is simple: **export the chat, copy the raw runs, verify
both, then keep a second copy outside the workspace**. Do that before deleting
anything, migrating accounts, or changing a runtime during a study. The software
retains committed data; it cannot promise recovery of data that was never committed
or that an operator later deletes.

## Where each kind of data lives

These names are in the selected Modal workspace's `FDR` environment. The same name
in another workspace or environment is a different resource.

| Resource | Contents | Role |
| --- | --- | --- |
| Volume `k2-interp-workbench`, mounted at `/sessions` | `session-<id>.json`: system prompt, ordered turns, assistant fields, settings, run IDs, capture references | Durable conversation index |
| Volume `k2-interp-artifacts`, mounted at `/artifacts` | `<checkpoint>/<run-id>/`: requests, identity, raw generated output, tokens, tensors, progress, failures, receipts; also analysis/training outputs | Durable research artifacts |
| Volume `k2-interp-weights`, mounted at `/weights` | `<immutable-revision>/`: weight shards, tokenizer/config/source, `snapshot.json` | Prepared model snapshots |
| Optional reused volume, mounted read-only at `/legacy` | Original weight shards referenced by prepared snapshot links | Required dependency when cache reuse is enabled |
| Dict `k2-interp-live` | Latest live projections, cancellation flags, one-use login tickets | Transient coordination, **not a backup** |
| Local `.private/` and `artifacts/` | Operator receipts and downloaded data | Private local files ignored by this repository's Git rules |
| Browser download directory | Downloaded chat/capture ZIPs | An independent copy only after the download finishes and is verified |

The workbench app and model app are separate. Their Volumes hold the durable data;
the browser and GPU container are not the archive. Model GPU memory and the KV cache
are transient. Closing a page does not turn those into a resumable model checkpoint.

Modal requires writers to commit Volume changes and readers to reload to see new
commits. This runtime commits at explicit boundaries; the workbench reloads before
reading capture files. Background persistence alone is not the completion signal.
[Modal Volume semantics](https://modal.com/docs/guide/volumes#volume-commits-and-reloads).

Modal Dict entries expire after seven days without reads or writes. Even before
expiry, this project's live channel coalesces intermediate updates into its latest
state. Do not use a live connection, Dict value, function call ID, or provider log as
your only record of a session. [Modal Dict retention](https://modal.com/docs/guide/dicts).

## What is committed, and when

1. **Turn accepted:** the controller stores the user message and run identity before
   dispatch. That records intent, not successful model execution.
2. **Run created:** `manifest.json` stores the request, checkpoint identity, source
   and dependency provenance, and request hash.
3. **Forward completed:** tensor shards and sidecars are written, hashed, and listed
   in `progress.json`; a Volume commit makes that checkpoint available outside the
   worker. Workbench projection files retain the corresponding output and reasoning
   available at that committed forward.
4. **Sample completed:** `result.json` contains raw output, token IDs, log probabilities,
   parsed assistant content/reasoning, finish reason, and input details. `complete.json`
   records the file inventory and hashes; the writer commits the completed sample.
5. **Controller catches up:** the session index records the final assistant message.
   If it lags, export reads the durable run result instead of dropping the reasoning.

An interrupted forward can lose its in-flight tensors and unsaved text. Earlier
committed forwards remain available. A token that has been sampled but not yet
processed by the next forward is not claimed as a committed activation row.

`complete.json` is the completion receipt, not a green indicator on a page. A local
or downloaded copy must also pass its listed hashes. A failed hash check is a reason
to preserve and investigate the files, not to overwrite them with a fresh run.

## Three different exports

| Export | Conversation and reasoning | Numeric captures | Intended use |
| --- | --- | --- | --- |
| Full chat ZIP | Entire session known at export time, including every turn, system prompt, native assistant fields, raw outputs and token IDs available in committed results | No full tensor shards | Readable, portable conversation archive |
| Selected-state capture ZIP | Entire session at export time **plus** the exact input context of the selected run | The selected source forward's captured tensors and projections | Bookmark an interesting token/layer without losing the surrounding chat |
| Raw artifact backup | Original run manifests, results, partial attempts, receipts, and reasoning | Every downloaded shard, analysis output, dictionary, and training state | Preserve the experiment for reanalysis and recovery |

A selected-state capture is not a full tensor backup of the conversation. For a
20-turn chat, it may contain all 20 turns of text and reasoning while containing
only one forward's numeric tensors. Download the relevant raw run directories to
preserve the other forwards.

### Full chat ZIP

The authenticated endpoint is:

```text
GET /api/sessions/<session-id>/export
```

It includes:

- `conversation.json`: authoritative structured session, all turns, unmodified
  assistant message fields, verified raw results, run manifests, and export coverage.
- `conversation.md`: a human-readable companion with separate answer, reasoning,
  other native assistant fields, and raw-output sections.
- `export.json`: schema, timestamp, scope, completeness, and SHA-256 hashes for the
  two conversation files.

Reasoning is included, not summarized. This means the actual text generated by K2,
including `reasoning_content` and any additional native thinking fields, not a
claim to export computations the runtime never observed. Empty thinking fields
remain empty fields. Exact raw output retains native delimiters and token IDs so
future parser changes need not destroy the original record. Unknown assistant
fields are preserved in JSON rather than dropped by a display-oriented schema.

If the model stops inside reasoning, the available reasoning still belongs in the
export. A verified final result ending with `finish_reason: length` is a completed
runtime request with truncated generation, not evidence of a finished answer.

### Selected-state capture ZIP

Use the workbench's capture action after selecting a turn, token position, and
layer. An optional tag is descriptive; it does not change the underlying run.
The bundle records a UTC timestamp, run ID, checkpoint, token position, layer,
forward step, phase, trace-completion state, and checksummed inventory.

Core files are:

```text
capture.json
selection.json
run-manifest.json
capture-context.json
capture-state.json
conversation.json
conversation.md
router.svg
trace/samples/chat/attempt-<id>/step-<step>-shard-<id>.json
trace/samples/chat/attempt-<id>/step-<step>-shard-<id>.safetensors
trace/samples/chat/attempt-<id>/workbench-step-<step>.json
```

`capture-state.json` retains the conversation prefix and generated output available
at the selected committed forward. `router.svg` renders the selected token's actual
captured routing weights. The manifest inventories what was actually written; the
numbers and raw tensors are the evidence behind any rendered image.

There are two time boundaries, deliberately named:

- **Selected state:** `capture-context.json` is the exact input message list for
  that run. `capture-state.json` retains the corresponding historical output. The
  shard and its position metadata identify the actual computation.
- **Export time:** `conversation.json` preserves the entire session snapshot then
  available, even if you selected a token from an earlier turn. Later conversation
  is archival context, not evidence that the model had seen it at the selected state.

Prefill processes multiple prompt tokens in a single forward. Selecting one of
those tokens does not create a physical time slice in which the model had read
only the preceding words. The capture includes the source forward and states its
scope. See [token alignment](artifacts.md#token-alignment).

Capture references in the session index are not a second copy of the ZIP. The
download is assembled in temporary controller storage and cleaned up after the
response. Confirm that your browser actually saved it; a bookmark label alone is
not proof of a successful download.

### Partial and missing output

Exports retain failed, cancelled, and still-running turns instead of silently
filtering them out. Their `export.coverage` entries distinguish `complete`,
`partial`, and `missing` source results. `archive_complete` is false when any listed
turn lacks a verified completed result.

For an interrupted run, `partial_generation` retains committed token IDs and log
probabilities. When a committed workbench projection exists for that same step,
`partial_assistant` and `partial_output_text` preserve the available parsed reasoning
and raw text. Older or interrupted traces may have IDs without parsed partial text;
the export says so. It does not invent a transcription or replace missing reasoning
with a summary.

## Verify a downloaded ZIP

Keep downloaded files outside the public source tree, or inside its ignored
`.private/` directory. Then run:

```sh
uv run python scripts/verify_export.py .private/my-session-conversation.zip
uv run python scripts/verify_export.py .private/my-interesting-state.zip
```

The verifier checks the inventory and SHA-256 for every archive member without
extracting or executing its contents. It rejects duplicate, unsafe, missing,
unexpected, and corrupted entries. It also reports turn count, chat completeness,
and tensor scope. Checksums establish that the files match the bundle manifest;
they do not authenticate an unknown sender or make an imported artifact safe to execute.

A valid partial export is still valuable. After reviewing its coverage, verify it
explicitly with `--allow-partial`. Without that flag, partial chat coverage returns
a nonzero exit status even when its file hashes pass. Do not confuse integrity with
completeness.

Open `conversation.md` and spot-check the first and last turns, the system prompt,
and a reasoning section you recognize. Then confirm the structured JSON has the
expected turn count and checkpoint. A finished browser transfer and a passing hash
check should both happen before you call something backed up.

## Back up the raw data

Use a new destination for every backup, with enough free space for the selected
runs. Large safetensors files are expected; a tiny ZIP does not imply tiny source
data. The following shell commands are examples to run in your own workspace,
not a schedule or an automatic retention policy:

```sh
# Restrict permissions for the files this shell creates.
umask 077
mkdir -p .private/backup-20261022/sessions .private/backup-20261022/artifacts

# Session index and every artifact currently committed in the two source Volumes.
MODAL_PROFILE=my-research uv run modal volume get --env=FDR k2-interp-workbench / .private/backup-20261022/sessions
MODAL_PROFILE=my-research uv run modal volume get --env=FDR k2-interp-artifacts / .private/backup-20261022/artifacts
```

Replace the profile and dated directory. Both local destinations are created first.
There is deliberately no `--force`: a backup should not quietly overwrite its
predecessor. The recursive download preserves paths below the Volume root. The
artifact copy contains checkpoint directories; session JSON files live in the
session copy's root.

For a smaller study, download only its named runs. Start with the run IDs in its
session JSON and retain their checkpoint directory structure:

```sh
mkdir -p .private/study-backup/artifacts/sft_3_1500
MODAL_PROFILE=my-research uv run modal volume get --env=FDR k2-interp-artifacts /sft_3_1500/<run-id> .private/study-backup/artifacts/sft_3_1500
```

In the installed Modal CLI, downloading a directory into an existing destination
retains that directory's basename. Verify that the resulting run contains
`manifest.json`, `samples/`, and its listed files before feeding the path to another
command. Never use `-` as the destination for a private conversation: it prints
file contents to the terminal.

Verify each downloaded run against its original manifest and receipts:

```sh
uv run python scripts/verify_trace.py .private/study-backup/artifacts/sft_3_1500/<run-id>
```

This checks the request/identity hash, every completed sample's receipted files,
and files listed in retained partial attempts' progress receipts. It reports
partial attempts separately and returns a nonzero exit status for a run with
missing completed samples, unless you explicitly use `--allow-partial`. Files
written after the last progress receipt are not claimed as verified. Keep the
entire downloaded attempt directory for diagnosis; do not silently discard
unreceipted material because the checker cannot certify it.

For a coherent finished-session backup, let the active turn finish before copying.
The two Volumes are not a cross-volume transaction. A copy taken during generation
is a useful partial snapshot, not a promise that both copies represent the same
instant. Keep it and make a second backup after completion.

## Export from a downloaded backup, offline

Once you have the session JSON and its corresponding run directories locally:

```sh
uv run python scripts/export_session.py \
  --session .private/backup-20261022/sessions/<session-id>.json \
  --artifacts .private/backup-20261022/artifacts \
  --output .private/backup-20261022/conversation.zip

uv run python scripts/verify_export.py .private/backup-20261022/conversation.zip
```

This does not contact Modal, launch a model, or modify your source backup. The
output path must not already exist. The exporter creates it with mode 0600 and
reports partial/missing coverage with a nonzero exit status after preserving the
available data. That makes the original conversation recoverable even when the
browser or controller is unavailable.

Keep the code commit and dependency lock used to produce the study alongside your
operator notes. Copy the verified backup to a second protected location outside the
same Modal workspace and outside the same laptop's failure domain. No scheduled,
cross-account, or off-site backup is enabled by this repository; arrange that
explicitly for your study.

## What survives common interruptions

| Event | What to expect | Safe next action |
| --- | --- | --- |
| Browser refresh, closed tab, expired login | The detached model request can continue; committed files remain | Reopen the same session and inspect its run; do not immediately submit a duplicate |
| Idle GPU scale-down | Committed Volume files remain; model RAM/KV cache disappears | Next request reloads the model; preserve original run IDs and files |
| Model error, cancellation, wall-time limit | Previously committed forwards remain; in-flight work may be absent | Export partial data and inspect `progress.json` / `failure.json` |
| App stop or model redeploy | Committed Volume data is separate from the running container; active work may be interrupted | Wait for active work before planned redeploys; back up first |
| Live Dict entry expires | Latest live state can disappear independently of durable files | Recover from run artifacts; do not treat disappearance as deletion of the study |
| Local disk fills during download | Download may be incomplete | Keep the remote source, make space in a separate destination, redownload and verify |
| Volume, source cache, account access, or laptop is lost | Recovery depends on independent copies and credentials you retained | Restore from a verified backup; no universal undelete guarantee exists |

Resume is deliberately conservative. A completed sample is skipped only after
hash verification. An incomplete generation starts a new attempt from the prompt;
it does not resume a saved KV cache. Earlier attempts stay available. Dictionary
training has its own optimizer/model/cursor resume state; preserve all of it, not
just the final dictionary weights.

## Restore without overwriting the surviving copy

First verify and inspect the local backup. Create new, clearly named recovery
Volumes in `FDR`, upload there without force-overwriting existing files, and check
the restored inventory. Keep the original Volumes untouched until recovery is
verified. The current apps refer to fixed Volume names, so mounting a recovery
volume is an explicit configuration/deployment step; there is no automatic
one-click session-import promise.

Restoring model weights is separate from restoring a conversation. For a reused
cache, a symlink is not the weight data it names. Restore both dependencies or
prepare a fresh self-contained snapshot at the recorded immutable revision.
Never repair a broken link by pointing it at a different checkpoint with a similar
directory name.

## Privacy and deletion

These archives can contain intimate text, system instructions, reasoning, and raw
activations. The Git ignore rules prevent common accidental commits; they are not
encryption, access control, or a guarantee against a manual `git add -f`. Browser
Downloads folders may also be indexed or synced by your operating system. Choose
your storage location accordingly.

The workbench is currently a shared research-workspace application, not a multi-tenant
service with per-person session isolation. An authenticated operator can access its
sessions. Modal workspace members with appropriate storage access may read the
underlying Volumes. Review membership before collecting private data, and review
exports before sharing them. No export command uploads study data to GitHub or
Hugging Face.

Before deletion, identify the exact workspace, environment, Volume, checkpoint,
run IDs, and independent backup location. Check completeness and hashes, then test
opening the restored data. Stopping a paid worker and deleting its persistent data
are different decisions. The project does not include broad fleet cleanup or an
automatic data-deletion policy; do not invent one while freeing GPU capacity.
