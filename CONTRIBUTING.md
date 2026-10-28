# Contributing

Use a feature branch and keep model identity, tensor semantics, and behavior claims
explicit. Every change that affects captured numerics needs a test against the pinned
native IFM implementation. Reduced random-weight models are appropriate for software
tests; scientific claims require the intended full checkpoint and protocol.

```sh
uv sync --python 3.11 --extra dev --extra modal --extra inspect
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
uv build
```

Preserve source licensing and attribution. New checkpoint tags must resolve to
immutable commits and complete shard manifests. Never silently replace a registry
entry when an upstream tag moves; add an explicit reviewed registry migration.

Keep backward compatibility for artifact schemas or increment their versions.
Model-serving optimizations must demonstrate logit/router parity before becoming
the canonical capture path. Changes to interventions need identity and cleanup
tests, not just tests that the forward pass completes.

Do not commit prompts from private conversations, study data, credentials, trained
weights, or unreviewed telemetry. Public CI uses ordinary synthetic/factual fixtures
and never requests a paid model worker.
