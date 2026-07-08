# Autoresearch: Rollout Stability for Rectified-Flow 3D Field Forecasting

## Objective
Find training/inference techniques that keep an autoregressive (AR) rectified-flow
forecaster **on-distribution over long rollouts** — frames that "look like training
data" instead of going blurry / developing artifacts. This is the weather-model
rollout-instability problem (see ../FINDINGS.md), reproduced here on a fast,
structured 3D volumetric benchmark so we can iterate quickly.

The known result we want to reproduce-and-beat: on images/structured data,
**Gaussian pixel/voxel noise on the context does NOT stabilize rollout well;
Gaussian blur does.** The loop must beat the best stabilization we can find.

## Workload / Data
- 3D scalar density field (16x16x16) — a Gaussian "cloud" blob whose center
  follows a Lorenz attractor (chaotic, low-dim manifold), amplitude/width slowly
  modulated. Generated in `vol_rollout_experiment.py:VolumeSequenceData`.
- K=2 context frames → forecast next frame. Rectified flow, endpoint/x-prediction,
  inverse-conditional-variance weighted velocity loss (matches weather model).
- Train ~2600 steps; AR rollout 24 trajectories × 50 steps for eval.

## Metrics
- **Primary**: `rollout_mmd` (unitless, **lower is better**) — squared MMD with
  Gaussian kernel between feature distributions of AR-rollout frames and a
  held-out training-frame reference. Features = avg-pool(6^3) + grad-energy +
  Laplacian-energy + total-mass + max. Lower = rollout frames look like training.
- Secondary: `sharpness_ratio` (1.0 ideal; <1=blurry, >1=artifacts),
  `mass_drift` (relative total-mass error), `train_loss`, `mmd_floor` (noise
  floor: fresh training sample vs reference — sanity, should be ~0 and stable),
  `wall_s`.

## How to Run
`.auto/measure.sh` — runs `vol_rollout_experiment.py`. Technique + hyperparams via
env vars: `TECHNIQUE`, `SIGMA`, `BLUR_SIGMA`, `SELFFEED_PROB`, `SPECTRAL_W`,
`DIFFFORCE_P`, plus sizing knobs (`TRAIN_STEPS`, `BATCH`, `ODE_STEPS`, etc.).
Outputs `METRIC name=value` lines parsed automatically.

## The Lever (what to edit)
**`augment_context()`** in `vol_rollout_experiment.py` — context augmentation at
training time (and optionally inference). This is the AR-stability mechanism.
**`extra_loss()`** — auxiliary regularizers (e.g. spectral/TV HF penalty).
Add new techniques as new branches; switch via `TECHNIQUE` env var.

Techniques implemented: none, pixnoise, blur, blur_noise, manifold_noise,
selffeed (scheduled sampling), diff_forcing, spectral, inference_blur.

## Files in Scope
- `vol_rollout_experiment.py` — the whole benchmark (data/model/train/eval). Edit freely.
- `.auto/measure.sh` — benchmark runner.

## Off Limits
- Do NOT edit the other `*_experiment.py` / `*.md` files (prior experiments).
- Do NOT weaken the metric to "cheat" (e.g. don't make the reference set include
  rollout-like frames, don't train on holdout trajectories, don't reduce
  ROLLOUT_LEN/N_ROLLOUT to make MMD trivially low). The point is real stability.

## Constraints
- Must run on one L4 GPU, < ~4 min per iteration.
- Keep `train_loss` reasonable (model must actually learn dynamics).
- `mmd_floor` should stay low & stable (~1e-3 or below) — if it jumps, the metric
  is broken, fix before trusting improvements.

## What's Been Tried
(update as experiments accumulate)

Full results + mechanism in `FINDINGS_VIDEO.md`. Summary:

| Technique | rollout_ed | sharp | mass_drift | verdict |
|---|---|---|---|---|
| pixnoise (baseline) | 9770 | 0.18 | 0.65 | FAILS — blurry |
| blur σ=0.8 | 5214 | 1.65 | 0.28 | works but artifacts+precision loss |
| manifold_noise | 1635 | 0.98 | 0.05 | best per-frame quality; cheap |
| **selffeed** p=0.3+blur0.4 | **725** | 0.88 | 0.08 | best stability (7x blur, 13x pixnoise) |
| selffeed no-blur | 1200 | 0.65 | 0.24 | blur is needed |
| selffeed p=0.2 / p=0.5 | 1410/5869 | — | — | optimum at 0.3 |
| selffeed2 (2-step) | 988 | 0.77 | 0.18 | deeper compounding doesn't help |
| **selffeed_m** (champion) | **747** | **0.90** | **0.07** | stable + sharp |

**Winner: scheduled sampling (selffeed_m) + multi-step rollout loss (selffeed_ms).**
Generalizes to C=2 / different seed (selffeed_m: 3x pixnoise, sharpness 0.88).

**DETERMINISTIC REGIME (2026-07-04):** switched to `torch.use_deterministic_algorithms`
+ `CUBLAS_WORKSPACE_CONFIG` + `cudnn.deterministic`. Prior runs had ~±50% run-to-run
variance from GPU non-determinism × chaotic-rollout amplification (a false "285" ED
outlier appeared and did NOT reproduce). Now single runs reproduce exactly.

Reproducible-regime numbers (lower=better):
| Technique | rollout_ed@2000 | rollout_ed@3500 | sharp@3500 | note |
|---|---|---|---|---|
| pixnoise | 6746 | 2887 | 0.67 | fails |
| selffeed_m | 912 | — | — | scheduled sampling + amplitude jitter |
| **selffeed_ms (champion)** | **690** | **156** | 0.92 | + 2-step rollout loss (MS_PROB=0.3) |

selffeed_ms is **9.8× better than pixnoise at 2000 steps and 18.5× at 3500**
(the gap WIDENS with training — the win is the technique, not compute; both
improve with budget but selffeed_ms far faster). TRAIN_STEPS=5000 is impractical
(timeout). Standard iteration regime = 2000; max-quality = 3500 (ED 156).
MS_PROB sweep: 0.15->725, 0.3->690 (optimum), 0.5->1023. MS_DEPTH: 2 optimal
(3->870). BPTT (gradient through pred): worse, mode collapse -> DETACH.

Key insight: rollout instability = exposure bias. Selffeed augments the INPUT
context (feeds model's own predictions); the multi-step rollout loss adds a LOSS
on the model's 2-step compounded output, directly closing the compounding gap.
Both are needed. Pixel noise fails (breaks image structure).

## Key Insight from Prior Work (FINDINGS.md)
Stability comes from the model **seeing degraded conditions during training** so it
doesn't extrapolate catastrophically on drifted AR inputs — NOT from informing it
about the noise level (the "decoupled-uncond" ablation matched the informed version).
So at inference we feed clean context (decoupled). The open question for structured
data: WHAT KIND of degradation best emulates real AR drift? Pixel noise breaks local
structure (bad); blur is smoother (better). Candidates: blur, blur+noise, manifold-
aligned perturbation, scheduled sampling (feed model's own predictions), spectral
regularizer, diffusion-forcing-style per-frame noise.
