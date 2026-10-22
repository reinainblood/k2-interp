# K2 Interp

A research runtime for IFM K2 Horizon 375B checkpoints. The project connects native
router capture, residual localization, sparse autoencoders, skip-transcoders,
checkpoint crosscoders, and measured causal interventions through a versioned
artifact format.

Implementation and verification are in progress. No scientific finding is implied
by this repository. Model weights remain subject to their upstream license.

## Design

- The canonical model is the pinned native Hugging Face K2 implementation in BF16.
- Each checkpoint has its own identity and Modal worker pool in environment FDR.
- Every token capture records the token being processed, its absolute position,
  and the prediction it precedes. Router scores are captured before expert dispatch.
- Traces and user prompts are private by default and excluded from Git.
- Analysis artifacts retain checkpoint, tokenizer, input, code, and tensor hashes.
- No process deletes checkpoints, stops other apps, or cleans up user sessions.

The runtime is independent of any particular experiment. Examples use ordinary
factual and text-completion prompts. See the final documentation and verification
report as they land in subsequent commits.
