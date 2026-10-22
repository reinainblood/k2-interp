# Verification

The local suite uses the actual pinned IFM model source at random, reduced dimensions.
It checks capture/logit parity, router semantics, hook cleanup, cached teacher forcing,
token alignment, resume/integrity, all dictionary modes, batch-independent inference,
gradient-versus-finite effects, reconstruction insertion, and router comparison.

Environment: Python 3.11, PyTorch 2.13.0, Transformers 5.13.0, repository `uv.lock`.
GitHub Actions runs these checks plus lint, formatting, packaging, and CLI inspection.

Full 375B verification is in progress. A complete eight-H200 receipt will be recorded
here when available. Unit tests do not establish that full weights load or complete
an intervention, and no runtime test is a scientific study result.
