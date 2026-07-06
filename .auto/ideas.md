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

### TOP UNTRIED IDEA: snapshot/bank "pseudo-self-feed" ~~(most promising)~~
~~Close the compounding gap WITHOUT live self-feed~~ — DEPRIORITIZED: in the
current env live self-feed UNDERPERFORMS (1284 vs manifold_noise 630), so there is
no gap to close. The bank idea (bank the model's training-time predictions, use
as context corruption) is only worth revisiting if moving to a regime where
self-feed is strong AND live self-feed is impractical. Kept here for reference.
- Applicable to the user's 4ctx->3fut weather model: their model already produces
  3 future-frame predictions during training; bank those, use to corrupt context.
- Variant: weight recent bank entries (model improves over time).
- Risk: bank lag (stale predictions); cheap high-t x_pred may be too noisy.

## 2026-07-05 session: JiT-3D production architecture + latent-noise corruption
Full writeup: JIT3D_FINDINGS.md. Summary:

### ANSWERED (do not retry on this benchmark)
- ~~Latent-noise corruption (production LatentContextCorruptor) on jit3d~~: FAILS.
  Monotonic collapse to blur (sharp 0.10->0.075 as corruption grows). Mechanism:
  context-token noise makes the UNDERTRAINED ViT ignore context -> mean
  regression. ED appears to improve but it's a metric-gaming artifact (blurry
  frames' pooled features regress to the reference mean); sharpness+mass_drift
  tell the truth. Trust sharpness, not ED, for the ViT.
- ~~bigger jit3d via more TRAIN_STEPS~~: DEGRADES past ~3000 steps (loss 0.42->1.28
  @4000). The t~=0 200x loss-weight instability accumulates. "Train more" doesn't
  help this ViT here.

### CRITICAL training fixes (infrastructure, committed)
- jit3d REQUIRES fp32 (AMP=0): bf16 diverges under the 1/(1-t)^2 RF loss.
- LR 1e-4 stable; 2e-4 knife-edge; 4e-4 diverge. grad_clip must stay 1.0 (0.5
  causes Adam amplification). NO warmup (peak-LR hold worsens divergence).

### BEST jit3d config on this benchmark
- manifold_noise (INPUT-space aug) LAST_ONLY jitter=0.20 blur_frac=0.5,
  3000 steps fp32, SAMPLE_AVG=4: sharp 0.32, mass_drift 0.002. Real forecast
  quality (sharp, correct mass) but high ED from positional divergence (chaos).

### Transferable to production (the user's real model)
- The corruption magnitude MUST be validated on a WELL-TRAINED checkpoint on real
  data. It fails here only because the ViT is undertrained on 220 trajectories.
  Failure mode to watch in production: blur collapse via context-downweighting.
  Monitor rollout sharpness + mass/energy drift, not only distributional ED.
- Consider input-space context aug (blur + amplitude jitter) as a robustness-
  friendly complement to the latent corruptor.
- Use fp32 (or loss-scaled mixed precision) for this ViT + RF-velocity loss.

### Untried (low priority, weak prior now)
- Latent corruption at a DEEP stage only (not embed) on a well-trained checkpoint
  — could avoid the context-ignoring collapse. Needs a way to train the ViT
  better first (more data, not more steps).
- A larger patch (JIT_PATCH_HW=8) to cut token count and overfitting — may let the
  ViT train better on tiny data. Not tested.

### FINAL jit3d corruption result (2026-07-05, 24 experiments)
Best config: **PROB=0.5 + production defaults (EMBED=0.10, BLOCK0=0.05) +
SAMPLE_AVG=4 + 3000 steps + fp32 (AMP=0) + RF_WCLAMP=50 + LR 1e-4 →
ED 20334, sharp 0.19** (-22% vs well-trained no-aug 26104).

EXHAUSTED / disproven this session:
- ~~manifold_noise + latent corruption combo~~: CONFLICT. Input aug raises ED
  (23933->36717) even though it fixes mass_drift (0.57->0.17). manifold_noise
  causes positional divergence on the ViT. Corruption-only is best for ED.
- ~~longer training (3500+ steps)~~: now stable under WCLAMP=50 but ED gets WORSE
  (sharper per-frame but more trajectory divergence in chaotic regime).
- ~~corruption with gentler magnitude (EMBED=0.05/no-block0)~~: 2x worse ED.
  Production magnitudes are load-bearing.
- ~~lower PROB (0.3) + AVG4~~: AVG4 only helps at high PROB (non-monotonic).

REMAINING (low prior on THIS benchmark; the gap is data/architecture, not aug):
- The ViT is ~45x behind conv (20334 vs 451) due to data-hungry architecture on
  220 trajectories. NOT fixable via aug — needs production-scale data where the
  ViT's scaling/long-range advantage appears.
- For PRODUCTION: corruption mechanism is validated (works on well-trained ViT);
  the failure mode to monitor is blur-collapse via context-downweighting early in
  training. Keep defaults, tune PROB on the ED/sharpness Pareto (0.3=sharp 0.23,
  0.5=low-ED 0.19).

## 2026-07-05 RF_WCLAMP breakthrough + full lever characterization (41 experiments)
Headline: the dominant lever for the ViT was the RF loss-weight clamp
(RF_WCLAMP), NOT the corruption. Default 200 catastrophic; optimum ~6.

### CONFIRMED OPTIMA (robust, cross-seed)
- **RF_WCLAMP=6** (THE lever): ViT 45x->2.7x behind conv. Sweep: 200 diverge,
  100 collapse, 50->21636/0.21, 25->7224/0.49, 12->3909/0.65, **6->1470/0.83**,
  3->3449 (reverses). Verified SEED=0 AND SEED=2. Principle: clamp over-emphasizes
  easy t~=1 denoising; lowering shifts weight to t~=0 forecasting direction.
- AMP=0 (fp32): required, bf16 diverges.
- SAMPLE_AVG=4: free 17%, cross-seed. AVG8 ghosts (worse).
- ODE_STEPS=32: optimal (48 regresses - chaos amplification; same as conv).
- TRAIN_STEPS 3000-4500: longer now viable (WCLAMP=6 stabilizes); seed-dependent
  sharpness tradeoff (~3000 robust default, 4500 squeezes ED).

### CLOSED NEGATIVES (do not retry)
- Latent corruption: hurts in ALL regimes (undertrained=collapse; WCLAMP50=seed-
  brittle; WCLAMP6=hurts every metric). Definitively not worth it.
- manifold_noise (input aug): worse than no-aug on ViT ED (positional divergence).
- grad_clip 0.5, warmup: both poisonous (Adam amplification / peak-LR hold).

### REMAINING (architectural, not aug-addressable)
- ViT ED 1210 (S0) vs conv 451 - 2.7x gap is data-scale/inductive-bias (ViT must
  learn Lorenz dynamics from 220 traj; conv's locality is ideal). Closes with
  production-scale data, not tricks.

## 2026-07-06 session: power-norm latent corruption + learned adversarial augmentor + 2k data

### METHODOLOGY: benchmark ED is NOISY (critical)
- No-corruption baseline ED swings 1553<->1825 (+17.5%) run-to-run while
  train_loss is rock-stable (0.0074). The ~17% ED noise is INHERENT to the
  chaotic Lorenz rollout (not from corruption). train_loss/sharpness/mass are
  stable; rollout_ed is NOT. Single-run ED comparisons are unreliable; need
  >5-6% to trust. For the ViT, trust sharpness+mass, not ED, when any
  perturbation/aug is active (blur regresses pooled features to the reference
  mean -> ED drops while quality drops = metric-gaming).

### Power-norm (per-token L2) latent corruption = FAILS like old corruption
- jit3d LatentContextCorruptor now uses per-token L2 power-norm (user spec).
- Strong/sparse (PROB=0.3, EMBED>=0.10): HURTS (sharp 0.84->0.81, ED worse).
- Gentle/frequent (PROB>=0.7, EMBED~0.03, embed-only): NEUTRAL on ED (mean 1595
  ~ baseline 1689, within noise), MODEST tendency to improve sharpness (0.87 vs
  0.84) + mass (0.065 vs 0.080) but buried in noise. The ONLY non-harmful variant.
- block0-stage corruption is the HARMFUL one (corrupting post-attention features
  worse than raw patch features). embed-only ~neutral.
- Partial vindication of user's hypothesis: power-norm blurs LESS than old global
  mean/std corruption (sharpness hit -3.6% vs -8.4%). But no ED gain on a
  well-calibrated model (corruption is pure robustness tax there).
- VERDICT: corruption doesn't help the well-calibrated ViT regardless of
  normalization. For production: validate magnitude on well-trained checkpoint
  on real data; treat as optional brittle regularizer, not a default.

### Learned adversarial conv augmentor (TECHNIQUE=learned_adv) = DEAD
- Implemented: small conv G(context,noise)->per-frame power-normalized residual,
  trained adversarially (maximize RF loss) within eps-ball. Input-space (sound,
  model can't cancel). Decoupled at inference.
- Unconstrained (eps=0.05): HURTS sharp 0.63, mass 0.22, ED 2248. Adversary
  attacks BLOB POSITION first (highest-loss direction in position-critical task)
  -> positional divergence. Same root cause as manifold_noise.
- Texture-CONSTRAINED (high-pass hp=2.5): HURTS WORSE sharp 0.33. Constraining
  to high-freq makes adversary attack TEXTURE directly (=what sharpness measures).
- ROOT CAUSE (definitive): classic robustness-accuracy tradeoff. The adversarial
  WORST-CASE input is out-of-distribution for CLEAN inference (rollout uses clean
  decoupled context). Every attackable direction (position OR texture) is one the
  model needs. Training on OOD-hard examples degrades clean-inference performance.
  Adversary's best case (disabled) = neutral = no-aug.
- VERDICT: input-space adversarial robustness is the WRONG tool for clean-inference
  forecasting. "Robustify from noise" is better served by matching the ACTUAL
  inference drift distribution (self-feed / snapshot-bank), NOT the worst-case.

### 2k trajectories = NOT benchmarkable in 4-min budget
- 2k @ fixed compute (BATCH=96/4500) UNDERTRAINS: ED worse, train_loss 0.014 vs
  0.0075 (2.8 epochs vs 25). 2k @ BATCH=256: train_loss 0.029 (LR too low for
  batch), sharp 0.77 (blurry undertraining -> ED 1258 is metric-gaming).
- Fully converging on 2k needs ~9x compute (~12-26 min/run). Impractical for the
  fast loop. 2k is a PRODUCTION-scale lever, not benchmarkable here.
- For production: more data WILL help the ViT (scaling advantage) given real compute.

### SESSION CONCLUSION (technique space re-confirmed exhausted)
- Latent corruption (any norm), learned adversarial augmentor (any constraint),
  and 2k-data-at-fixed-compute all FAIL or are impractical here.
- Champions STAND: RF_WCLAMP=6 + fp32 + SAMPLE_AVG=4 + ODE_STEPS=32 + ~4500 steps.
- Remaining ViT-vs-conv gap is inductive-bias/data-scale (closes only with conv
  architecture or production-scale data + compute).
- HIGHEST-VALUE untried (if user wants to keep going on THIS benchmark): JIT_PATCH_HW=8
  (fewer tokens, less overfit on 220 traj) and K_CTX=3/4 (more context frames).
