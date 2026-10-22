# Experiment protocol

Define the question and primary behavioral metric before inspecting activations.
Freeze the prompt dataset and retain rendered token sequences, system context,
template settings, and sampling parameters. Begin checkpoint comparisons with
teacher forcing on identical sequences; report natural free generation separately.

Use independent prompt families for train/validation/test. Hold out vocabulary and
syntax where the hypothesis requires generalization. Randomly splitting near-duplicate
sentences by row does not produce an independent test.

1. Qualify target behavior.
2. Collect routers and selected residual sites.
3. Localize candidate layers/positions on discovery prompts.
4. Validate on held-out families.
5. Train dictionaries on broad independent activations; disclose any task enrichment.
6. Measure reconstruction and original-model fidelity.
7. Attribute features, then test finite interventions, random controls, dose response,
   necessity, and sufficiency on held-out examples.
8. Compare checkpoints and locate the transition, retaining negative results.

Mid4–SFT3 combines RL, merging, and SFT. Adding `rl_merged` isolates the later SFT
path. Mid4 already shifts toward agentic/reasoning SFT-style data; Mid3 is earlier,
but is not evidence of zero instruction-like exposure throughout its training history.

Label latents using activation distributions, exemplars, counterexamples, multiple
training seeds, and causal effects. Routing can predict behavior without carrying
its computation; distributed computations may lack a single clean latent.

Personal protocols belong in `.private/`; outputs belong in `artifacts/`. Both are
ignored by Git. Review and explicitly redact any package chosen for publication.
