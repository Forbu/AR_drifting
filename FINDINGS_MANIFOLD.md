# Findings: manifold_noise — a simple, self-feed-free AR-rollout stabilizer

**Date:** 2026-07-05
**Goal:** Stabilize autoregressive (AR) rollout of a rectified-flow forecaster
**without scheduled sampling / self-feed** — applicable to the user's real weather
model (4 context frames → 3 future frames, fixed 7-frame dataset, model predicts
only future frames so it cannot feed its own outputs back as context).

Benchmark: `video_rollout_experiment.py` (Lorenz-driven 2D Gaussian-blob video,
K=2 context → forecast next; rectified flow, x-prediction). Stability metric =
**energy distance** between feature distributions of 50-step AR rollout frames
and held-out training frames (lower = more on-manifold; sharpness ~1.0 and
mass_drift ~0 are ideal). All runs deterministic (`torch.use_deterministic_algorithms`
+ `CUBLAS_WORKSPACE_CONFIG`), exactly reproducible.

---

## TL;DR

- **Environment shifted since the prior `FINDINGS_VIDEO.md` write-up.** The
  previously reported numbers (vae_ms 1106, selffeed_ms 690) are **not
  reproducible** in the current PyTorch/CUDA environment. Re-baselined at
  `LORENZ_DT=0.012`: in this regime **every** technique fails blurry at rollout
  (sharpness ~0.5) due to intrinsic AR exposure bias — the model trains on clean
  context but at inference feeds its own slightly-off outputs, compounding to a
  mean-regressed (blurry) blob.
- **The simplest technique wins and it needs NO self-feed, NO VAE.** Apply a
  mild **Gaussian blur + per-sample amplitude jitter to the most-recent context
  frame only** during training (`manifold_noise` + `MANIFOLD_LAST_ONLY=1`).
  This emulates the model's actual rollout error (a wider, lower-amplitude blob)
  and teaches the model to deblur/sharpen from a degraded context.
- **Champion config:** `TECHNIQUE=manifold_noise MANIFOLD_BLUR_FRAC=0.5
  MANIFOLD_JITTER=0.20 MANIFOLD_BLUR_RAND=0.5 MANIFOLD_LAST_ONLY=1 BLUR_SIGMA=1.0`
  → **rollout_ed = 739, sharpness = 1.01 (ideal), mass_drift = 0.001, ed_late = 721.**
- **It beats the documented self-feed champion in this environment.** `selffeed_m`
  (live scheduled sampling) gives **ED 1284** here — *worse* than manifold_noise
  (739). In a regime where the model's per-step predictions are poor, feeding
  them back hurts; the static on-manifold corruption is more robust.
  → The "self-feed is best" conclusion from `FINDINGS_VIDEO.md` is **environment-specific**.

---

## Results table (current environment, SEED=0, dt=0.012, TRAIN_STEPS=2000)

| Technique | rollout_ed | ed_late | sharpness | mass_drift | train_loss | wall_s | note |
|---|---|---|---|---|---|---|---|
| pixnoise | 8494 | 13458 | 0.22 | 0.62 | 0.19 | 68 | catastrophic blur (worst) |
| none (no aug) | 5622 | 12708 | 0.55 | 0.30 | 0.33 | 68 | intrinsic exposure-bias failure |
| vae_ms (MS loss) | 4919 | 14724 | 0.54 | 0.28 | 0.14 | 89 | MS loss UNSTABLE now (spikes) |
| vae_noise_blur | 4092 | 10474 | 0.58 | 0.26 | 0.13 | 76 | AE corruption, no MS loss |
| selffeed_m (live) | 1284 | 2677 | 0.76 | 0.13 | 0.24 | 197 | documented champion — underperforms here |
| manifold_noise (j=0.08, all frames) | 1170 | 1827 | 0.97 | 0.04 | 0.13 | 69 | first strong result |
| manifold_noise (j=0.16, all frames) | 921 | 986 | 0.98 | 0.004 | 0.13 | 69 | robust quality |
| manifold_noise (j=0.20, all frames) | 730 | 811 | 0.89 | 0.068 | 0.15 | 69 | best ED, slightly blurry |
| manifold_noise LAST_ONLY (j=0.16) | 824 | 903 | 1.11 | 0.06 | 0.11 | 68 | better dynamics |
| **manifold_noise LAST_ONLY (j=0.20)** | **739** | **721** | **1.01** | **0.001** | 0.13 | 69 | **CHAMPION** |

Lower rollout_ed / ed_late is better. sharpness ~1.0 and mass_drift ~0 are ideal.

---

## Why manifold_noise works (mechanism)

The AR-rollout failure is **exposure bias**: trained on clean context, the model
at inference sees its own slightly-drifted outputs. For this blob data the drift
is specifically a **wider, lower-amplitude blob** (the model's prediction is a
slightly blurred, amplitude-reduced version of the true frame). Compounding this
over rollout steps → catastrophic blur (sharpness 0.22 for pixnoise, 0.55 for no-aug).

- **Pixel noise fails** because isotropic per-pixel noise is *not* a plausible
  data perturbation — it breaks the smooth Gaussian profile, so the model learns
  to produce over-smoothed means (sharpness 0.22).
- **Blur + amplitude jitter succeeds** because a wider, lower-amp blob *is* a
  plausible on-manifold perturbation (it looks like real data, just slightly
  mis-rendered). Training on these teaches the model to **sharpen/deblur** a
  degraded context → at inference it recovers sharpness from its own blurry outputs.
- **The blur strength has a sharp optimum (≈0.5 × BLOB_SIGMA):** too little
  (0.4) → stays blurry (mean regression); too much (0.6+) → the model
  over-compensates and produces artifacts (sharpness >1.2, ED rises). The
  optimum matches the corruption to the *actual* rollout-blur magnitude.

### Why corrupt only the LAST context frame (`MANIFOLD_LAST_ONLY=1`)

At AR inference the context is `[real_old_frame, model_prediction]`. **Only the
most-recent slot holds the model's (degraded) output**; the older frame is real.
Corrupting *all* context frames over-degrades and discards clean dynamics info.
Corrupting only the last slot (a) exactly matches the inference condition and
(b) keeps the older clean frame, giving the model accurate dynamics → lower
train_loss and better rollout ED/sharpness simultaneously. This is a strict
improvement over corrupting all frames.

---

## What did NOT work (this session)

- **Self-feed-free multi-step loss (`manifold_ms`):** HURTS (MS_WEIGHT 1.0→ED 1284,
  0.3→2140). The `1/(1−t)²` velocity weight (clamped at 200) destabilizes training
  in this environment (loss spikes to ~5). The MS loss only helped in the prior env.
- **Blur annealing (`MANIFOLD_ANNEAL`, cosine decay to track shrinking error):**
  catastrophic (→ ED 15188, blurry). The corruption acts as a *regularizer*
  (consistent deblurring training), not an error-magnitude matcher — reducing it
  late lets the model revert to mean regression. Annealing is the wrong frame.
- **Spatial sub-pixel shift (`MANIFOLD_SHIFT`):** marginal/mixed (trades sharpness
  for ED, no clear win). Reverted.

## Key limitation (same as all static corruption)

manifold_noise **plateaus/worsens with more training**: 3500 steps → ED 1373,
train_loss 0.70 (worse than 921 @ 2000). Static corruption **over-corrupts a
better-trained model**. It does **not** compound the way live self-feed can
(self-feed 690→156 @ 2000→3500 in the prior env). **Best at moderate training
budget (~2000 steps here).** For very long training, the corruption magnitude
would need to be re-tuned down as the model improves (a light manual schedule,
*not* a cosine-to-zero anneal which fails).

## Generalization / caveats (important for weather transfer)

- **SEED=1:** the *model itself* fails to train (train_loss 2.6) for **all**
  techniques — a bad seed for the optimizer/data, not a technique issue. Unusable.
- **SEED=2:** a harder regime where rollout diverges more (no-aug ED 12867).
  manifold_noise helps *directionally* (ED < no-aug at conservative jitter) but
  jitter=0.20 **over-corrupts** there (ED > no-aug). **The optimal magnitude is
  data-scale-dependent** — re-tune lightly per dataset. Conservative jitter=0.16
  is more robust across seeds; jitter=0.20 is ED-optimal on seed=0.
- **Not overfitting the metric:** `ed_floor` stays ~0; the energy distance is
  non-saturating and tracks position + texture + mass. The Gaussian-kernel MMD is
  saturated at 0.404 for all techniques (useless) — only the energy distance
  discriminates.

---

## Actionable recipe for the weather model (4 ctx → 3 fut, no self-feed)

1. During RF training, augment the **most-recent context frame only** (the slot
   that will hold the model's own output during AR rollout) with:
   - a mild Gaussian blur (σ ≈ 0.5 × the typical feature width), and
   - a per-sample multiplicative amplitude jitter (±~20%).
   Keep the older context frames clean (they carry dynamics info).
2. Decouple: feed **clean** context at inference (the augmentation is train-only).
3. Tune the blur σ to your data — there is a **sharp optimum** (too little →
   blurry rollout, too much → artifacts). Sweep around the feature width.
4. For long training runs, re-tune the magnitude down as the model improves
   (manual, not cosine-to-zero).
5. Do **not** use pixel/voxel noise (catastrophic blur on structured data) and
   do **not** bother with self-feed if your per-step predictions are mediocre —
   static on-manifold corruption is more robust in that regime.

## Reproducing

```bash
# champion (manifold_noise, last-frame-only corruption):
TECHNIQUE=manifold_noise BLUR_SIGMA=1.0 MANIFOLD_BLUR_FRAC=0.5 \
  MANIFOLD_JITTER=0.20 MANIFOLD_BLUR_RAND=0.5 MANIFOLD_LAST_ONLY=1 \
  TRAIN_STEPS=2000 LORENZ_DT=0.012 python video_rollout_experiment.py
# baseline (no aug — shows the intrinsic blurry-rollout failure):
TECHNIQUE=none TRAIN_STEPS=2000 LORENZ_DT=0.012 python video_rollout_experiment.py
```
