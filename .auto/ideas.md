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
- ~~Classifier-free guidance (CFG)~~: FAILS both directions. GUIDANCE=2.0 ->
  ED 690->2338 (artifacts, amplifies drifted context); GUIDANCE=0.5 -> ED 23827
  (catastrophic blur, mean regression toward unconditional). Root cause: forecasting
  is highly context-determined so the unconditional prior is uninformative. CFG
  suits high-uncertainty generation, not forecasting.

## STATUS: technique space EXHAUSTED
All major stabilization mechanisms tested. Champions:
- selffeed_ms (self-feed + 2-step rollout loss): ED 690@2000, 156@3500 (closed-loop).
- vae_noise_blur (no-self-feed substitute): ED 1186, sharp 0.93 (static AE proxy).
Failed: pixel noise, blur-alone, manifold/diff-forcing, 3-step loss, BPTT, EMA,
  spectral, curriculum, inference-blur, CFG, selffeed2, jitter-on-VAE.
Remaining low-prior: restart/stochastic sampler (within-sample quality isn't the
  bottleneck; context drift is).

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

## 2026-07-05 session (env re-baselined; manifold_noise champion)

### ENV SHIFT (important)
Documented numbers (vae_ms 1106, selffeed_ms 690) are NOT reproducible in the
current environment (PyTorch/CUDA drift). Re-baselined at dt=0.012: ALL techniques
fail blurry now (sharp ~0.5) due to intrinsic AR exposure bias. `none`=5622,
`pixnoise`=8494, `vae_noise_blur`=4092, `vae_ms`=4919 (MS loss now UNSTABLE,
spikes to 5.48). mmd stuck at 0.404 ceiling (saturated, useless).

### NEW CHAMPION (no-self-feed): manifold_noise = context blur + amplitude jitter
- Simplest effective technique. No VAE, no self-feed, no model outputs.
- Mechanism: model's rollout error = wider, lower-amplitude blob. Blur+ampjitter
  on the context emulates this directly; teaches the model to deblur/sharpen.
- TUNING (seed=0, dt=0.012, sharp optimum):
  - MANIFOLD_BLUR_FRAC: 0.4->blurry(8801), **0.5->ideal(739-921)**, 0.6->artifacty, 1.2->ED13682
  - MANIFOLD_JITTER: 0.04->1589, 0.08->1170, 0.12->988, 0.16->921(sharp0.98), **0.20->739(sharp1.01)**, 0.24->742. (non-monotonic at 0.18=881)
  - MANIFOLD_LAST_ONLY=1 (corrupt ONLY the most-recent context frame): strictly
    better — matches inference (only recent frame is the model's own output),
    keeps older clean frame for dynamics (lower train_loss). ED 921->824 @j0.16.
- **BEST: LAST_ONLY=1, frac=0.5, jitter=0.20 -> ED=739, sharp=1.01, mass=0.001.**
  Beats vae_noise_blur (4092) by 5.5x; approaches old-env selffeed (690).

### FAILED this session (do not retry without new reason)
- ~~manifold_ms (self-feed-free multi-step loss with manifold-corrupted real frame)~~:
  HURTS (MS_WEIGHT 1.0->1280, 0.3->2140). The 1/(1-t)^2 weight clamp 200
  destabilizes training in this env. MS loss only helped in old env.
- ~~MANIFOLD_ANNEAL (cosine decay blur to track shrinking error)~~: catastrophic
  (0.2 -> ED15188 blurry). Model needs CONSISTENT deblurring regularizer; reducing
  it late reverts to mean regression. Static corruption is a regularizer, not an
  error-matcher — annealing is the wrong frame.
- ~~MANIFOLD_SHIFT (sub-pixel spatial jitter for position drift)~~: marginal/mixed
  (j0.16+shift1.0 -> 870 sharp1.10; j0.20+shift0.5 -> 800 sharp1.07). Trades
  sharpness for ED, no clear win. Reverted.

### KEY LIMITATION (same as static VAE): plateaus with training
- manifold_noise @3500 steps -> ED 1373, train_loss 0.70 (WORSE than 921@2000).
  Static corruption over-corrupts a better-trained model. Does NOT compound like
  self-feed (690->156 in old env). Best at moderate budget (~2000 steps).

### GENERALIZATION (seed dependence — important for weather transfer)
- SEED=1: model fails to train for ALL techniques (train_loss 2.6) — bad seed for
  the model itself, not a technique issue. Unusable.
- SEED=2: harder regime (rollout diverges more: none=12867). manifold helps
  directionally (j0.16 corrupt-all=9659 < none 12867) but jitter=0.20
  OVERCORRUPTS there (15898 > none). The optimal magnitude is DATA-SCALE
  DEPENDENT — must light-tune per dataset. Conservative jitter=0.16 more robust.

### TOP UNTRIED IDEA: snapshot/bank "pseudo-self-feed" (most promising)
Close the compounding gap WITHOUT live self-feed: maintain a FIFO bank of the
model's own predictions (collected cheaply during training — e.g. from x_pred at
high-t, or a periodic full-ODE sample every N steps), and use bank samples
(blended with the real last-context-frame + mild blur) as the corruption for the
last context slot. This captures the model's ACTUAL current error distribution
(its specific blur/amp pattern) and UPDATES as the model improves -> approximate
self-feed compounding, without per-batch live sampling.
- Applicable to the user's 4ctx->3fut weather model: their model already produces
  3 future-frame predictions during training; bank those, use to corrupt context.
- Variant: weight recent bank entries (model improves over time).
- Risk: bank lag (stale predictions); cheap high-t x_pred may be too noisy.
