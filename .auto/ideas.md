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

## 2026-07-06 session: ARCHITECTURAL BREAKTHROUGH (PATCH_HW + capacity) — ViT now ~1.8x conv

### THE WIN: JIT_PATCH_HW=8 (coarser patches -> fewer tokens -> less overfitting)
- Prior-session hypothesis VERIFIED: "larger patch -> cut token count -> less
  overfit on tiny data". PATCH_HW 4->8 cuts tokens 192->48 (4x fewer).
- ROBUST across seeds (clean 2x2 within-env):
  SEED=0: 170k PHW4 sharp0.84/mass0.080/ED1689 -> PHW8 sharp0.93/mass0.024/ED1512
  SEED=2: 170k PHW4 sharp0.87/mass0.065/ED1610 -> PHW8 sharp0.90/mass0.056/ED886
  PATCH_HW=8 >= PATCH_HW=4 on EVERY metric at BOTH seeds. Direction unanimous.
- Operates on capacity/overfitting axis, NOT a perturbation tax -> WHY IT WORKS
  where ALL augmentation/robustness mechanisms failed this session.

### CAPACITY: 800k (embed=128/depth=4) + PATCH_HW=8 = CHAMPION
- Compounds two wins (capacity + coarse patches):
  170k(64/3) PHW8: ED1512/sharp0.93/mass0.024 (S0)
  800k(128/4) PHW8: ED803/sharp1.03/mass0.018 (S0); ED805/sharp0.86/mass0.081 (S2)
- ViT now ~1.8x conv (ED 804 avg vs conv 451; sharp ~1.0 vs conv 1.11). Was 45x.
- CAVEAT (overfitting discipline): 800k sharpness win is SEED-DEPENDENT. At SEED=2
  more capacity HURTS sharp/mass: 170k sharp0.90/mass0.056 > 800k 0.86/0.081 > 96/4 0.81/0.119.
  Averaged across seeds 800k wins ED+sharp, 170k wins mass. Monitor mass per-seed.
- 96/4 (middle capacity) ELIMINATED: SEED=0 'best mass' 0.002 was a fluke
  (SEED=2 mass 0.119, worst). Seed-specific, not robust.

### CAPACITY AXIS EXHAUSTED at PATCH_HW=8
- 800k is the ED/sharpness sweet spot. 170k is the mass-robust alternative.
- Remaining ViT-vs-conv gap (~1.8x) is data-scale, not capacity/aug.

### METHODOLOGY (re-confirmed): trust sharpness+mass, not rollout_ed, for the ViT
- rollout_ed is +-17% noisy run-to-run (no-corruption baseline 1553<->1825,
  train_loss rock-stable). Noise is inherent to chaotic Lorenz rollout.
- When ANY perturbation/aug is active, ED metric-games (blur regresses pooled
  features to reference mean -> ED drops while sharp/mass drop). Trust sharp/mass.

### PRODUCTION RECOMMENDATIONS (this session)
- Use PATCH_HW=8 (or coarser) for the ViT on small-data regimes -> big robust win.
- 800k-class capacity + coarse patches gets the ViT near conv. Watch mass per-seed.
- Latent corruption (any normalization) and learned adversarial augmentation are
  NOT worth it for clean-inference forecasting (adversarial worst-case is OOD).
- 2k+ trajectories need ~9x compute to converge; viable only at production scale.

### K_CTX=3 (more context) — SEED-DEPENDENT, doesn't beat champion
- K=3 on 800k champion: OVERSHOOTS (sharp 1.18 = artifacts, mass 0.104, ed_late
  1508). High capacity + more context = confident extrapolation -> artifacts.
- K=3 on 170k: HELPS at SEED=0 (ED 1512->920, sharp 0.98, mass 0.0036) but HURTS
  at SEED=2 (ED 886->1061, worse than K=2 on all metrics). Seed-dependent.
- Hypothesis confirmed: more context helps a SMALLER/less-confident model but lets
  a high-capacity model overshoot. Capacity x context interaction.
- Averaged: K=3/170k = best mass (0.031) but ED 990 > 800k/K=2 champion's 804.
- Does NOT beat the champion on primary (ED). Mass-best alternative only.

### OVERFITTING FRONTIER REACHED (stop tuning hyperparameters)
- Every further hyperparameter (capacity 96/4, K_CTX=3, width/depth) now produces
  SEED-DEPENDENT noise (S0 and S2 disagree), not robust gains. This is the
  signature of the overfitting frontier -> continuing to tune risks OVERFITTING
  THE BENCHMARK (the prompt's warning).
- SESSION CHAMPIONS (both verified S0+S2):
  - 800k (embed128/depth4) + K=2 + PATCH_HW=8: ED 803/805, sharp 1.03/0.86,
    mass 0.018/0.081. BEST ED + sharpness. (committed)
  - 170k (embed64/depth3) + K=2 + PATCH_HW=8: ED 1512/886, sharp 0.93/0.90,
    mass 0.024/0.056. MOST ROBUST on mass/sharp (no SEED=2 regression).
- ViT now ~1.8x conv (was 45x). Remaining gap is DATA-SCALE (production compute),
  not architecture/augmentation.

### WHAT WOULD ACTUALLY MOVE THE NEEDLE NEXT (production-scale, not this benchmark)
- More trajectories (2k+) WITH proportionally more compute (~9x) -> ViT scaling
  advantage appears. Not benchmarkable in the 4-min loop.
- Generalization check at different LORENZ_DT (validates champion isn't overfit
  to dt=0.012) - a VALIDATION, not an optimization.

### GENERALIZATION VALIDATION: PATCH_HW=8 win is dt=0.012-SPECIFIC
- At dt=0.024 (fast motion): PATCH_HW=8 ED4326 < PATCH_HW=4 ED5423 (better early
  ED) BUT worse ed_late (39817 vs 23924), worse mass (0.60 vs 0.27), artifact
  sharpness (2.87 vs 0.50). MIXED - win does NOT cleanly generalize.
- Fast motion (dt=0.024) is an unstable regime for this RF forecaster regardless
  of patch (both ED ~5000 vs 803 at dt=0.012; sharpness far from 1.0). Matches
  prior session 'rankings flip in fast motion'.
- CAVEAT for production: validate PATCH_HW on the target motion regime. The
  coarse-patch win holds in moderate motion (dt=0.012, robust SEED=0/2) but fast
  motion needs separate treatment.

### FINAL SESSION STATE (committed champion)
- 800k (embed=128/depth=4) + K=2 + PATCH_HW=8 + WCLAMP=6 + fp32 + AVG4 + ODE32 +
  4500 steps. ED 803 (S0) / 805 (S2), sharp 1.03/0.86, mass 0.018/0.081.
- ViT ~1.8x conv (was 45x). Gap is data-scale (production), not arch/aug.
- run.env restored to this champion.

### NOISE_INJECT (stochastic sampler) — MARGINAL (confirmed), + latent bug
- NOISE_INJECT=0.5 (inference-only re-noising during ODE) on champion: ED 774 vs
  803 (within noise), ed_late slightly worse. NEUTRAL. Confirms prior 'likely
  marginal' - within-sample stochasticity doesn't address the context-drift
  bottleneck.
- LATENT BUG (now known): NOISE_INJECT code had (1.0-(i+1)*dt).clamp() on a
  Python float -> AttributeError. Fix: max(0.0, 1.0-(i+1)*dt). The fix reverts on
  discard (feature marginal); if ever re-testing NOISE_INJECT, re-apply the fix
  at sample_step lines ~881/~898.

### SESSION FULLY EXHAUSTED (71 experiments)
All principled levers tested/confirmed. Champion (800k/PHW8/WCLAMP6/AVG4/ODE32/
fp32) is optimal on every major axis. Remaining ViT-vs-conv gap (~1.8x) is
DATA-SCALE (production compute), not addressable on this 4-min benchmark without
overfitting. The benchmark's AR-stability PURPOSE was addressed via architecture
(less overfitting -> better generalization -> less rollout drift), NOT via
augmentation/robustness (all failed: corruption, adversarial, stochastic sampler).

## 2026-07-06 session: Brownian-bridge flow matching (FLOW=bridge) — EXPLORED

### RESULT: bridge helps WHERE RF STRUGGLES, not on easy data
- Implemented FLOW=bridge (z_t=(1-t)eps+t y+c_t eta, c_t^2=sigma^2 t(1-t)+sigma_min^2)
  + exact closed-form ODE sampler + 3 loss modes (vloss/ivar/uniform). Composes
  with any TECHNIQUE (context aug orthogonal to target flow path).
- EASY regime (dt=0.012, single blob): bridge ~1.4x RF ED (1174 vs 803), competitive
  sharpness. RF already well-tuned -> bridge doesn't help.
- HARD/unstable regime (dt=0.024 fast motion): bridge DRAMATICALLY better — ED 1541
  vs RF 4326 (2.8x), mass 0.086 vs 0.60 (7x), ed_late 2662 vs 39817 (15x), tames
  artifacts (sharp 2.87->1.13). Clean 2x2 (same env/arch). This is the bridge's
  stated AR-stability purpose DEMONSTRATED: clean endpoint landing -> minimal drift
  injection -> stable long rollout.
- Matches hypersphere result (bridge helped there — harder data).

### KEY LESSONS (transferable to production)
- **Loss type matters (COLLAPSE):** uniform-weighted x-pred (user's production
  reference 'v-loss') COLLAPSES to "predict from context, ignore z_t" on EASY data
  (train_loss anomalously low, sharp>1.1 artifacts). Needs hard data to avoid.
  The `(1-t c'/c)^2` vloss weight (one-sided data-end upweight) prevents collapse.
  For the user's HARD production data, uniform may be fine (context insufficient) —
  but vloss is the robust choice regardless.
- **train_loss NOT comparable across loss types** (weight scale differs: ivar
  ~30-50x, vloss ~0.5x, uniform 1x). Trust sharpness/mass, not train_loss magnitude.
- **RF hyperparams don't transfer to bridge:** RF's optimal RF_WCLAMP=6 -> bridge
  artifacts; bridge wants wclamp=10. sigma=0.3 U-shaped optimum (0.2 too low -> RF-like,
  0.5 too high -> blur).
- **Bridge + manifold_noise CONFLICT:** manifold over-corrupts (bridge already
  regularizes target). Use one or the other.

### OPTIMAL bridge config (committed): FLOW=bridge BRIDGE_LOSS=vloss
BRIDGE_SIGMA=0.3 BRIDGE_WCLAMP=10 BRIDGE_SIGMA_MIN=1e-3, jit3d 800k/PHW8/none.
ED 1174/ed_late 1222/sharp 1.06/mass 0.025 (dt=0.012).

### DONE / not worth retrying
- ~~manifold + bridge~~: conflicts (1732 > 1103).
- ~~sigma 0.2/0.5~~: U-shape, 0.3 optimal.
- ~~wclamp 6~~: artifacts (metric-gaming).
- ~~sigma_min 1e-4~~: too-sharp landing, artifacts.

### UNTRIED (lower priority)
- Bridge at dt=0.024 with sigma=0.5 (harder regime may want more mid-path reg).
  Niche; the 2.8x win over RF at dt=0.024 already demonstrated the mechanism.
- Bridge on C_CHAN=2 multichannel (more production-like; bridge may help more).
- Stratified t sampling (user ref 32-bin) for better t coverage — minor.

## 2026-07-08 session: Gaussian/Brownian-bridge CONTEXT augmentation (ctx_bridge) — WIN

### USER HYPOTHESIS (validated): add a gaussian bridge to the context information
Bridge champion (FLOW=bridge + TECHNIQUE=none + jit3d 800k/PHW8) regularizes the
TARGET path. User asked: also corrupt the CONTEXT with bridge-shaped gaussian noise
to robustify AR rollout. Implemented TECHNIQUE=ctx_bridge: noise std follows the
SAME bridge variance schedule c_s^2 = sigma^2 s(1-s) + sigma_min^2 (or flat/uniform,
or peak). CTX_BRIDGE_LAST_ONLY=1 corrupts only the most-recent context frame (the
slot holding the model's own output at inference). Decoupled (clean ctx) at inference.

### RESULT: gaussian context noise HELPS where AR drift exists (robust, cross-seed)
Cross-seed 2x2 at dt=0.024 (drift regime), TRAIN_STEPS=4500, jit3d 800k/PHW8:
| config               | S0 ED | S2 ED | avg ED | avg sharp | avg mass |
|----------------------|-------|-------|--------|-----------|----------|
| baseline none        | 951   | 1242  | 1097   | 0.90      | 0.067    |
| bridge-sched sample  | 855   | 1009  | 932    | 1.05      | 0.018    |
| flat gaussian std.06 | 792   | 1097  | 945    | 1.03      | 0.014    |
Both techniques robustly beat baseline ~14% avg ED. HEADLINE: at the HARD S2 seed
(baseline blurry sharp 0.78, mass drift 0.125), ctx_bridge CURES blur (sharp->1.04+)
and mass drift (->0.02) — exactly the AR-stability failure modes the project targets.

### KEY ABLATION: bridge SCHEDULE is NOT the active ingredient — MAGNITUDE is
Flat/uniform gaussian at MATCHED magnitude (std~0.06) BEATS bridge-scheduled sample
mode at S0 (792<855) but LOSES at S2 (1097>1009). The flip = overfitting-frontier
signature => schedule choice is within seed noise. Mechanism: sample mode gives a wide
variance range (many samples get near-zero noise at s~0,1 = under-regularized);
flat gaussian applies consistent reg to every sample. ROBUST finding: it's the
gaussian noise MAGNITUDE (~std 0.06 on the last ctx frame) that matters, not the
bridge shape. For production: just add N(0, ~0.06) to the most-recent context frame.
NOTE this is much smaller than the pixnoise that failed (std 0.4 on all frames).

### REGIME-DEPENDENT (correct robustness-mechanism behavior)
- dt=0.024 (fast/hard, AR drift exists): HELPS robustly (-14% avg ED, cures blur/mass).
- dt=0.012 (easy, minimal drift): NEUTRAL-to-worse (0.3 neutral, 0.5 neutral-ED but
  worse ed_late, 1.0 worse). No drift to fix => noise is pure late-rollout tax.
This is the RIGHT behavior: a robustness mechanism should help only where the failure
mode exists. Do NOT apply ctx_bridge at slow/easy regimes.

### CHAMPION (committed): ctx_bridge uniform flat gaussian, std~0.06, last-frame-only,
dt=0.024. Best single (792 S0) + most robust on mass (trustworthy metric). Bridge-
scheduled sample mode (scale=0.5) is comparable and matches target's noise shape
(the user's literal idea) — also validated. run.env restored to flat-gaussian champion.

### CLOSED (do not retry without new reason)
- ~~scale sweep @dt=0.024 S0~~: U-shape 0.3->866, 0.5->855, 0.7->974(artifacts). Opt ~0.5.
- ~~bridge vs flat schedule~~: within seed noise (no robust winner); magnitude is key.
- ~~dt=0.012~~: neutral-to-worse across all magnitudes (no drift). Don't apply here.

### PRODUCTION RECOMMENDATION (for the user's 4ctx->3fut weather bridge model)
- ADD input-space gaussian noise to the most-recent context frame(s) at training time,
  magnitude ~std 0.06 in frame-std units (validate/tune on a well-trained checkpoint).
- The bridge schedule is NOT required — flat gaussian is simpler and comparable. Match
  the magnitude to the model's actual per-step rollout error (larger error => more noise).
- Decouple at inference (clean context). Monitor rollout sharpness + mass/energy drift.
- This is the FIRST input-space context-aug that helps the bridge forecaster (prior:
  manifold_noise/blur CONFLICTED, latent corruption COLLAPSED, pixnoise FAILED).
  Why it works where others didn't: SMALL magnitude + last-frame-only + bridge flow
  already handles endpoint drift, leaving only residual per-step drift to address.

### REMAINING (low priority — avoid overfitting frontier)
- Magnitude finer-tune is seed-noise now (overfitting frontier reached for this knob).
- ctx_bridge + more context (K_CTX=3/4): untested; more context may change drift profile.
- Flat gaussian at dt=0.012 with a TINY magnitude (<0.03): marginal, likely neutral.

### MULTICHANNEL GENERALIZATION (C=2) — soft positive (2026-07-08)
ctx_bridge flat gaussian on C_CHAN=2 @dt=0.024: ED neutral (2054~2033, within noise)
BUT quality metrics improve: ed_late -12% (1803<2040), sharp 0.97->1.02, mass 0.040->0.021
(2x better). Consistent with trust-sharp+mass methodology -> benefit transfers to
multichannel (production-relevant: user's model is 4ctx->3fut multichannel). ED neutral
here likely because C=2's failure mode is under-fit/blur (different from C=1 overshoot);
ctx_bridge sharpens without moving the position-dominated ED much. Did NOT tune magnitude
for C=2 (overfitting frontier) — production should re-tune magnitude per channel-set.

### STRUCTURAL VARIANTS — all FALSIFIED (champion design confirmed) (2026-07-08)
- ~~K=3 + ctx_bridge synergy~~: FALSIFIED. K=3+ctx_bridge ED 1246 >> K=2+ctx_bridge 792.
  Extra context frame adds divergence the small gaussian can't tame. K=2 optimal.
- ~~ctx_bridge at K=3~~: HURTS (K=3 none 1016 < K=3 ctx_bridge 1246) — OPPOSITE of K=2.
  Mechanism refined: ctx_bridge benefit is K-DEPENDENT. Helps when context is SCARCE
  (K=2, model over-relies on the few drifted frames -> mean-regression blur); with more
  context (K=3) the model is naturally robust and the noise is pure degradation.
- ~~LAST_ONLY=0 (corrupt BOTH context frames)~~: WORSE (ED 917 > 792, mass 0.029>0.005).
  Older frame[0] is CLEAN in training; corrupting it degrades the dynamics signal the
  model needs. Last-frame-only confirmed optimal for gaussian too (matches manifold).

### COMPOUNDING TEST — ctx_bridge is an ACCELERATOR, NOT a compounder (2026-07-08)
Matched-compute comparison @dt=0.024 S0:
| TRAIN_STEPS | none ED | ctx_bridge ED | winner        |
|-------------|---------|---------------|---------------|
| 4500        | 951     | 792           | ctx -16.7%    |
| 6000        | 646     | 755           | none -14.2%   |
KEY: ctx_bridge's advantage REVERSES with more training. At 4500 it wins big; by 6000
the base model has learned robust dynamics alone and the context noise becomes a pure
tax (only mass stays better: ctx 0.0016 vs none 0.017). So ctx_bridge is a COMPUTE-
EFFICIENT ACCELERATOR (gets you to good rollout stability faster at moderate budget),
NOT a self-feed-style compounder (which widened its lead with training). Static regularizer
the model outgrows — same class as manifold_noise (which plateaued), NOT selffeed.
NOTE: none@6000 ED 646 is achievable in-budget (~207s) but it's a "train more" result,
not a technique. At MATCHED compute, ctx_bridge only wins at moderate budget (<=4500).

### PRODUCTION RECOMMENDATION (refined)
- ctx_bridge = add N(0,~0.06) to the most-recent context frame, train-only, decoupled infer.
- Use it when COMPUTE IS LIMITED (moderate training budget) and dynamics are drift-prone
  (fast/hard). It accelerates reaching stable rollout (cures early blur + mass drift).
- If you can train LONGER, the benefit fades — the base model learns robustness on its own.
  Don't expect it to compound like scheduled sampling; it's a bootstrap regularizer.
- The magnitude (~0.06) and last-frame-only are load-bearing; K=2 is where it helps.

### ANNEAL — DEAD (static is correct) (2027-07-08)
Implemented CTX_BRIDGE_ANNEAL (linear decay of ctx noise to 0 over training).
- anneal@6000 (0.8): ED 678 < static 755 (recovers late-tax penalty), best ed_late 623,
  but does NOT beat none@6000 (646, within 5% noise). At high budget none is optimal.
- anneal@4500 (0.8): ED 894 > static 792, sharp 0.91, mass 0.056 (WORSE on everything).
  At moderate budget the model still NEEDS the regularizer; annealing it off -> revert
  to blur/mass-drift (the exact failure modes ctx_bridge prevents). SAME failure as
  MANIFOLD_ANNEAL ("model needs CONSISTENT regularizer").
CONCLUSION: ctx_bridge must be STATIC. The model either needs it (moderate budget: keep
ON) or has outgrown it (high budget: none wins anyway). Annealing combines the worst of
both (removes regularizer while still needed). CTX_BRIDGE_ANNEAL=0 (off) is the default;
feature kept in code as a documented negative result.

### FULL ctx_bridge CHARACTERIZATION (technique space exhausted)
Magnitude (0.06 optimal, U-shape), schedule (flat≈bridge, within noise), K (K=2 optimal,
hurts K=3), frames (last-only optimal), compounding (accelerator not compounder, reverses
>6000), anneal (dead), regime (helps dt=0.024, neutral dt=0.012), multichannel (soft
positive). CHAMPION: static flat gaussian std~0.06, last-frame-only, K=2, @4500 drift regime.

## 2026-07-08 session: RANDOM non-conditional pixel-noise sigma on context (ctx_bridge TIME=randsigma)

### USER HYPOTHESIS (the literal request this session)
User asked: add RANDOM (non-conditional) gaussian/pixel noise on the context = `ctx += sigma*eps`
with `sigma ~ U(0,1)` per sample (the noise STD itself sampled uniformly, NOT tied to the
bridge schedule t). This is a flow/diffusion-style forward process applied to the CONTEXT.
Implemented as `CTX_BRIDGE_TIME=randsigma` in augment_context (video_rollout_experiment.py,
JiT-3D backbone + Brownian-bridge target flow). NOTE: this is PLAIN PIXEL NOISE with random
scale — not "bridge" noise on context (user clarified). It reuses the ctx_bridge plumbing.

### MAGNITUDE SWEEP (SCALE = upper bound of sigma~U(0,SCALE); mean std = SCALE/2)
U-shape, optimum at SCALE=0.1 (mean std 0.05). @4500/220 SEED=0 dt=0.024:
| SCALE | mean std | ED    |
|-------|----------|-------|
| 0.05  | 0.025    | 914   |
| 0.1   | 0.05     | 875   | <- optimum (beats baseline 951)
| 0.15  | 0.075    | 1052  |
| 0.2   | 0.10     | 1051  |
| 1.0   | 0.50     | 1223  | (over-corrupts: mass 0.098, sharp 1.14 overshoot)
Literal `sigma~U(0,1)` (SCALE=1.0) over-corrupts — high-sigma samples destroy mass.
Magnitude ~0.05 mean matches the prior flat-gaussian champion (fixed std 0.06). KEY: the
random SPREAD (many near-zero samples) slightly under-regularizes vs fixed-flat, so the
optimum mean (0.05) is a touch below flat's (0.06).

### THE CORE FINDING: randsigma is a REGIME-DEPENDENT ROBUSTNESS MECHANISM
It is NOT a universal win. It is a TAX where the base model is already good and a CURE
where the base model FAILS. Cross-seed @6000/500 dt=0.024:
| config            | S0 ED | S2 ED | avg ED | avg sharp | avg mass |
|-------------------|-------|-------|--------|-----------|----------|
| none (base)       | 634   | 862   | 748    | 0.91      | 0.063    |
| randsigma SCALE=.1| 789   | 763   | 776    | 1.00      | 0.024    |
- SEED=0 (lucky/easy): base already sharp(1.00)/stable(0.0075) -> noise is a TAX (789>634).
- SEED=2 (hard): base FAILS blurry (sharp 0.82, mass 0.119) -> noise CURES it (sharp 0.95,
  mass 0.028, 4x; ED 763<862 -11.5%). EXACTLY the AR-stability failure mode the project targets.
- Cross-seed ED ties (~4%, within noise); QUALITY favors randsigma (avg sharp 1.00 vs 0.91,
  avg mass 0.024 vs 0.063). It is robustness INSURANCE: small cost on easy seeds, prevents
  catastrophic failure on hard seeds. Same class/mechanism as flat-gaussian ctx_bridge
  (validated SEED=2 cure earlier this week).

### 3-SEED GENERALIZATION (honest, 4500/220 dt=0.024) - NOT a free lunch on easy seeds
| config            | S0 ED | S2 ED  | S3 ED | 3-seed avg |
|-------------------|-------|--------|-------|------------|
| none              | 951   | 1242   | 988   | 1060       |
| randsigma SCALE=.1| 875   | 947    | 1099  | 974 (-8%)  |
Seed landscape: S0 easy (951), S2 hard-FAILS (1242, sharp 0.78), S3 easy (988). ~1/3 of
inits catastrophically fail (blur+mass-drift). KEY HONESTY: on EASY seeds randsigma is
neutral-to-HARMFUL — S0 marginal help (875, within 17% noise), S3 OVERSHOOTS (sharp 1.16,
mass 0.0815, ED 1099>988). So randsigma's ONLY ROBUST effect is CURING catastrophic hard-
seed failure; on easy seeds it can overshoot. 3-seed avg still -8% (the S2 catastrophe
prevention dominates). VERDICT: robustness INSURANCE (cheap catastrophe prevention), not a
free quality win. For production (one init, can't seed-tune): worth it if catastrophic AR
failure is costlier than slight easy-case overshoot. The S2 cure is the load-bearing result.

### COMPUTE / DATA SCALING (user's 'train longer + bigger dataset' directive)
Sweet spot at none@6000 steps/500 traj (~15 epochs) = ED 634 (S0) / 862 (S2). NON-monotonic:
| config          | epochs | ED (S0) | note                       |
|-----------------|--------|---------|----------------------------|
| none 4500/220   | 25     | 951     | undertrained-ish (orig)    |
| none 6000/500   | 15     | 634     | <- SWEET SPOT              |
| none 6000/1000  | 7      | 787     | UNDERTRAINS (blur 0.91)    |
| none 9000/750   | 15     | 877     | OVERTRAINS (train_loss 1.8e-5 tiny, rollout degrades; sharp 1.08 overshoot) |
LESSONS: (1) bigger dataset at fixed compute UNDERTRAINS (7 epochs) -> blur/mass-drift.
Needs proportionally more compute (confirms prior '2k needs ~9x compute'). (2) MORE STEPS
past ~6000 OVERTRAINS the ViT on bridge+vloss (low train_loss but worse AR rollout drift) —
the 1/(1-t c'/c)^2 weighting accumulates instability. (3) 6000/500 is the practical optimum:
enough epochs to learn, not so many steps it overfits/destabilizes.

### MULTICHANNEL (C=2) TRANSFER - SEED-DEPENDENT, NOT robust (2026-07-08)
Tested randsigma on production-like multichannel. C=2 @4500/220 dt=0.024:
| config   | S0 ED (blurry base) | S2 ED (sharp base) | avg ED |
|----------|---------------------|--------------------|--------|
| none     | 2033 (sharp 0.97)   | 1795 (sharp 0.997) | 1914   |
| randsigma| 1899 (sharp 1.03)   | 2170 (sharp 1.02)  | 2035 (+6%) |
- S0 (slightly-blurry base): randsigma HELPS (-7% ED, sharpens 0.97->1.03, mass 10x).
- S2 (already-sharp base): randsigma HURTS (+21% ED, overshoots).
- Cross-seed avg: randsigma WORSE (+6%). The S0 'win' was a single-seed artifact
  (the model happened to be slightly blurry). Prior session's flat-gaussian C=2 'soft
  positive' was also single-seed (S0) - same artifact.
- NOTE: at C=2 the HARD seed flips (S2 is sharp here, S0 is the blurry one) - seed-
  difficulty is CONFIG-DEPENDENT, not intrinsic to a seed number.

### FINAL HONEST VERDICT (20 experiments, C=1 + C=2, 3 seeds)
Random non-conditional pixel noise on context (sigma~U(0,0.1)) is **CATASTROPHE
INSURANCE ONLY**. The CONSISTENT pattern across EVERY config tested:
  - Helps ONLY where the base model is already drifted/blurry (C=1 S2 catastrophic
    blur+mass-drift; C=2 S0 mild blur).
  - HURTS (overshoots) where the base model is already sharp (C=1 S0/S3; C=2 S2).
Its SOLE robust value: preventing the ~1/3 of random inits that collapse into
blur+mass-drift failure. It is NOT a reliable ED/quality improver cross-seed
(C=1 3-seed avg -8% only because the S2 catastrophe dominates; C=2 2-seed +6% worse).
PRODUCTION RECOMMENDATION: add context pixel noise ONLY if you observe catastrophic
AR failure (blur+mass-drift) in your model. On a well-trained/stable model it can
HURT (overshoot). Do NOT treat it as a default quality booster. Validate on YOUR data.
Root cause of the value: the noise forces robustness to degraded context, which
prevents mean-regression collapse - but a model that isn't collapsing doesn't need it.

### MOST PROMISING NEXT DIRECTION (root-cause, not band-aid)
Instead of INSURING against the catastrophic seed-failure, FIX the training so no seed
fails. The C=1 S2 collapse (blur+mass-drift = mean regression) is a training instability
affecting ~1/3 of inits. Candidates: BRIDGE_WCLAMP tuning (the prior stability lever -
maybe a different value prevents the S2 collapse without context noise), LR schedule,
init scheme. A training recipe that makes ALL seeds stable would beat the insurance
approach (no overshoot cost on good seeds). UNTESTED - high value if it works.

### CHAMPION / run.env
- Best primary metric (C=1 SEED=0): none@6000/500 = ED 634, sharp 1.00, mass 0.0075
  (the compute/data sweet spot). run.env restored to this.
- Technique (randsigma) is documented as catastrophe insurance, not the ED champion.
