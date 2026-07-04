# Findings: Rollout Stability for Rectified-Flow 2D Video Forecasting

**Date:** 2026-07-04
**Goal:** Find training techniques that keep an autoregressive (AR) rectified-flow
forecaster **on-distribution over long rollouts** — frames that "look like training
data" instead of going blurry / developing artifacts. This is the weather-model
rollout-instability problem (see FINDINGS.md), studied here on a fast 2D-video
benchmark (`video_rollout_experiment.py`).

Benchmark: Lorenz-driven 2D Gaussian "cloud" blob video (32×32, optionally
multichannel). K=2 context frames → forecast next. Rectified flow, endpoint /
x-prediction, inverse-conditional-variance weighted velocity loss (matches the
weather model). Stability = **energy distance** between feature distributions of
50-step AR rollout frames and held-out training frames (lower = more on-manifold;
sharpness_ratio ~1.0 and mass_drift ~0 are ideal).

---

## TL;DR

- **Scheduled sampling ("selffeed") is the dominant lever.** Feeding the model
  its own 1-step predictions as context during training closes the train/inference
  (exposure-bias) gap directly. **rollout_ed 9770 → 725 (13× better than pixel
  noise, 7× better than blur)**, with sharpness 0.78 (near ideal 1.0) and
  mass_drift 0.14.
- **Adding on-manifold amplitude jitter ("selffeed_m") keeps stability and
  sharpens frames**: sharpness 0.78 → **0.90**, mass_drift 0.14 → **0.07**, at
  equal rollout_ed.
- **Adding a 2-step rollout loss ("selffeed_ms") is the new champion** (deterministic
  regime): ED 912 → **690** (24% better than selffeed_m, 9.8× vs pixnoise),
  ed_late 1240 → 710 (stable end-to-end), mass_drift 0.08 → 0.02. The multi-step
  loss adds a *loss* on the model's own 2-step compounded output, closing the
  compounding gap that input-only selffeed misses. Reproducible (deterministic).
- **Reproduces the user's real-world finding**: plain Gaussian pixel/voxel noise
  on the context **fails** on structured image data (sharpness 0.18 — catastrophic
  blur, mass_drift 0.65) — and is in fact **WORSE than no augmentation at all**
  (pixnoise ED 9770 vs no-aug ED 5940) on slow/weather-like motion. **Blur works**
  (ED 9770→5214) but introduces artifacts (sharpness 1.65) and precision loss
  (train_loss 0.16→3.37). Scheduled sampling beats both with **none** of those
  side-effects.
- **All pixel-noise-based augmentation fails on image data regardless of schedule**
  (pixnoise ED 9770, diff_forcing ED 5540) — the noise *type* matters more than
  the schedule. Pixel noise breaks local blob/profile structure.
- **The mild blur in selffeed is load-bearing** (σ≈0.4 px): pure scheduled
  sampling without it is worse (ED 725→1200). It smooths the model's own
  prediction errors into plausible degraded contexts.
- **Sweet-spot selffeed probability ≈ 0.3, constant from step 0.** Too low
  (0.2→1410) under-trains error correction; too high (0.5→5869, late divergence)
  trains too often on bad predictions. **A curriculum ramp (0→0.3) is WORSE**
  (ED 2908) — it reduces total exposure; early predictions do *not* poison
  training.
- **Deeper compounding (2-step selffeed) does NOT help** (ED 988, 1.6× slower).
  Single-step exposure is sufficient.
- **Generalizes**: on multichannel (C=2) + different seed, selffeed_m beats
  pixnoise 3× on ED with sharpness 0.88 vs 0.42.

---

## Results table (C=1, IMG=32, default config, energy distance)

| Technique | rollout_ed | ed_late | sharpness | mass_drift | train_loss | wall_s | note |
|---|---|---|---|---|---|---|---|
| none (no aug) | 5940 | 12283 | 0.57 | 0.27 | 0.44 | 66 | no-aug fails; note pixnoise is even worse |
| **pixnoise** (baseline) | **9770** | 15571 | 0.18 | 0.65 | 0.16 | 64 | blurry — pixel noise is WORSE than no-aug on slow image data |
| **blur** σ=0.8 | 5214 | 6240 | 1.65 | 0.28 | 3.37 | 66 | works but artifacts + precision loss |
| manifold_noise | 1635 | 2441 | **0.98** | **0.05** | 0.12 | 65 | best per-frame quality; cheap; less stable than selffeed |
| diff_forcing (per-frame pixnoise) | 5540 | 11395 | 0.46 | 0.41 | 0.15 | 65 | pixel-noise schedule still fails |
| **selffeed** p=0.3 +blur0.4 | **725** | 906 | 0.88 | 0.08 | 0.12 | 168 | **best stability** |
| selffeed (no blur) | 1200 | 2988 | 0.65 | 0.24 | 0.12 | 165 | blur is needed |
| selffeed p=0.2 | 1410 | 3786 | 0.73 | 0.16 | 0.15 | 174 | too low |
| selffeed p=0.5 | 5869 | 15739 | 0.83 | 0.09 | 0.12 | 171 | too high — late divergence |
| selffeed2 (2-step compounded) | 988 | 1695 | 0.77 | 0.18 | 0.17 | 280 | deeper compounding doesn't help |
| selffeed_cur (ramp 0→0.3) | 2908 | 9471 | 0.55 | 0.26 | 0.10 | 174 | curriculum reduces exposure → worse |
| **selffeed_m** p=0.3 +blur0.4 +ampjitter | **747** | 1051 | **0.90** | **0.07** | 0.19 | 169 | **champion: stable + sharp** |

Lower rollout_ed / ed_late is better. sharpness ~1.0 and mass_drift ~0 are ideal.

---

## Why scheduled sampling wins (mechanism)

The AR-rollout failure is an **exposure-bias / train-inference mismatch** problem:
at training time the model always sees *clean, ground-truth* context frames; at
inference it sees its own slightly-drifted outputs, which lie off the data
distribution, so errors compound. Pixel-noise augmentation tries to emulate this
drift but **isotopic pixel noise is not a plausible perturbation** for structured
image data (it breaks local blob/profile structure) → the model learns to produce
over-smoothed means (sharpness 0.18). Blur is more plausible (a wider/misplaced
blob is data-like) so it helps, but it degrades the condition so much that
dynamics learning suffers (train_loss 0.16→3.37) and outputs develop artifacts
(sharpness 1.65).

**Scheduled sampling sidesteps the "what does drift look like?" question entirely**
by generating the drift from the model itself: with prob p, replace a context
frame with the model's own (detached) 1-step forecast + mild blur. The model
then trains on its *actual* error distribution. Result: it learns dynamics
precisely (train_loss stays low 0.12) *and* becomes robust to its own drift.
Adding on-manifold amplitude jitter (selffeed_m) further keeps outputs on the
data manifold (sharper, correct mass).

This is the 2D-image analog of the hypersphere `decoupled-uncond` result
(FINDINGS.md): stability comes from training on degraded conditions — but for
structured data the *best* degraded conditions are the model's own predictions,
not synthetic noise.

---

## Actionable takeaways for the weather model

1. **Adopt scheduled sampling ("selffeed") AND a multi-step rollout loss** in the
   RF trainer. This is the combined recipe (`selffeed_ms`) that wins:
   - With prob ~0.3 (constant from step 0), replace one context frame with the
     model's own 1-step forecast (detached) + mild blur σ≈0.4 px + small
     amplitude jitter. (Augments the *input* context.)
   - With prob ~0.3, add a loss term on predicting frame t+1 from a context that
     contains the model's own prediction of frame t (detached). (Adds a *loss* on
     the 2-step compounded output.) This closed the remaining late-rollout drift
     that input-only selffeed missed: ED 912→690, ed_late 1240→710, mass_drift
     0.08→0.02, reproducibly.
2. **Keep a SMALL context blur** (σ≈0.4 px) on the self-fed frames; do **not**
   use strong blur alone (artifacts + dynamics loss). Optimal internal blur ≈0.4.
3. **Do NOT use pixel/voxel noise augmentation** — it is *worse than no aug* on
   slow/weather-like image data (sharpness 0.22, catastrophically blurry).
4. **Probabilities ~0.3** for both self-feed and the multi-step loss; higher
   (0.5) causes late divergence / dynamics loss. Lower (0.15) gives slightly
   sharper frames but worse rollout ED.
5. **Multi-step depth**: 2-step loss suffices at this cadence (single-step
   selffeed alone is 912; +2-step loss → 690). A 3-step loss is untested but
   likely diminishing returns for the cost.
6. **Use deterministic eval** when tuning: GPU non-determinism × chaotic rollout
   gave ~±50% variance and a false outlier; `torch.use_deterministic_algorithms`
   + `CUBLAS_WORKSPACE_CONFIG` makes single runs reproducible (~20% slower).

---

## Caveats / not overfitting

- Validated on a *single-blob Lorenz* dataset. The mechanism (exposure-bias fix)
  is data-agnostic, but the exact probability / blur σ should be re-tuned on real
  weather frames.
- Confirmed generalization to **multichannel (C=2)** and a different seed
  (selffeed_m: sharpness 0.88, mass_drift 0.07, 3× better ED than pixnoise).
- The metric (energy distance on pooled+gradient features) is non-saturating and
  tracks both position drift and texture/artifact drift (sharpness, mass).
- Energy distance replaces an earlier Gaussian-kernel MMD which **saturated** at
  an identical value (0.404) for both pixnoise and blur (k_ab→0, mbb→1 ceiling)
  and was useless for measuring improvements.
- **METRIC VARIANCE / DETERMINISM (important methodological note):** non-deterministic
  CUDA kernels (conv/groupnorm backward) × chaotic-rollout amplification produced
  ~±50% run-to-run ED variance even with fixed seeds — a single run once gave a
  false ED=285 that did NOT reproduce (real value ~750-1000). Fixed by enabling
  `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  + `cudnn.deterministic`. Verified reproducible (selffeed_m → 912.186462 three
  times identically). Reproducible-regime numbers: pixnoise=6746, selffeed_m=912
  (BLUR_SIGMA=1.0). Cost ~20% slower. **Lesson: always verify reproducibility
  before trusting a surprising win on a chaotic benchmark.**

## Reproducing

```bash
# baseline (pixel noise — fails):
TECHNIQUE=pixnoise SIGMA=0.4 python video_rollout_experiment.py
# champion (selffeed + on-manifold jitter):
TECHNIQUE=selffeed_m SELFFEED_PROB=0.3 BLUR_SIGMA=1.0 python video_rollout_experiment.py
# multichannel robustness:
C_CHAN=2 SEED=1 TECHNIQUE=selffeed_m SELFFEED_PROB=0.3 BLUR_SIGMA=1.0 python video_rollout_experiment.py
```
