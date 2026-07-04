# Ideas backlog (deferred / future)

## High-value, not yet tried on this benchmark
- ~~Selffeed with gradient (BPTT)~~: WORSE. Tested as selffeed_ms_grad (gradient
  through the multi-step loss pred): ED 690->950, sharpness 0.91->0.75 (mode
  collapse - model rewarded for 'easy-to-predict-from' outputs). DESIGN
  PRINCIPLE: DETACH the model's own predictions in both input-context selffeed
  AND the multi-step rollout loss. Detached = trains robustness; gradient = collapse.
- ~~3-step rollout loss~~: WORSE (ED 690->870, slower). 2-step is the sweet spot;
  deeper compounding adds noise.
- **Restart / renoise during ODE sampling** (stochastic sampler) at each AR step —
  sampling-side mechanism, orthogonal to training aug. UNTESTED (likely marginal:
  within-sample quality isn't the bottleneck; context drift is, already addressed).
- **Classifier-free guidance** on the context (drop context w.p. during training,
  guide at inference). UNTESTED.

## Tried and did NOT help (do not retry without a new reason)
- ~~Curriculum on selffeed probability (ramp 0→0.3)~~: WORSE (ED 747→2908) —
  reduces total exposure; early predictions don't poison.
- ~~EMA teacher for the surrogate~~: WORSE (747→1218). Live model is the correct
  surrogate (matches inference distribution); lagging EMA mismatches.
- ~~Spectral/TV HF-penalty loss~~: TIED (747→750), sharpness slightly worse.
- ~~2-step compounded selffeed~~: WORSE (747→988) + 1.6× slower.
- ~~Inference-time context blur (on champion)~~: WORSE (747→1231).
- ~~selffeed prob 0.2 / 0.5~~: both worse; optimum is 0.3.
- ~~diff_forcing (per-frame pixel noise)~~: fails like pixnoise (ED 5540).

## Generalization checks (do before trusting for production)
- DONE: multichannel C=2 + SEED=1 — BOTH selffeed_m (3x pixnoise, sharp 0.88)
  AND selffeed_ms (2.9x pixnoise, sharp 0.91) generalize. VALID, no overfit.
- DONE: fast-motion regime (dt=0.024) — ranking FLIPS (regime-dependent; documented).
- INCONCLUSIVE: IMG=64 needs ~4000 steps to train the model; at 1000 steps the
  MODEL is undertrained (train_loss 1.6 vs 0.12) so no technique works. Re-run
  with more steps + maybe more model capacity before drawing conclusions.
- TODO: 2 independent crossing blobs (test single-blob structure exploitation).
- TODO: K_CTX=3 / K_CTX=4 (more context) — does selffeed still help?

## Metric improvements
- The energy distance is dominated by blob-position (pooled features). Add a
  texture-only ED (gradient/Laplacian features alone) as a secondary to separate
  "wrong position" from "wrong texture" failure modes.
