# Autoresearch: Joint RF-Context-Generation + Bridge-Future-Prediction for AR Rollout Stability

## Objective (user idea)
Stabilize the autoregressive (AR) rollout of a Brownian-bridge flow-matching video
forecaster by adding a **joint training objective**: the model simultaneously
**(a) DENOISES the context frames via a LINEAR rectified-flow interpolant** (learns
to *generate* valid data from scratch — a strong generative prior) and
**(b) PREDICTS the future frame via the Brownian-bridge interpolant** (learns
dynamics / prediction from other frames).

Mechanism / hypothesis:
- Input: `[noisy-context-frames..., noisy-future-frame]` -> output: clean frames for BOTH.
- Context uses **linear RF**: `z_ctx = (1-s)·eps + s·ctx`.
- Future uses **Brownian bridge**: `z_t = (1-t)·eps + t·tgt + c_t·eta`.
- The context RF time `s` is sampled **closer to data than** the future time `t`
  (`s ≥ t`), because at inference the context frames are the model's own
  already-refined outputs (near-clean) — "generate the data first, then predict".
- Total loss = `bridge_loss(future) + JOINT_CTX_WEIGHT · rf_loss(context)`.
- At **INFERENCE** the rollout is UNCHANGED: predict the future from CLEAN context.
  The context-generation head is a training-only auxiliary that (we hypothesize)
  teaches an on-manifold prior + robustifies against AR context drift.

The number to beat is the standard **Brownian-bridge forecaster with no joint head**
(JOINT_GEN=0), measured under the same arch/data/seed.

## Metrics
- **Primary**: `rollout_ed` (energy distance, **LOWER is better**) — distributional
  distance between long-AR-rollout frames and a held-out training-frame reference
  (energy distance on pooled/grad/Laplacian/mass/max features; non-saturating).
- **Secondary**: `ed_late` (energy distance on the LATE half of the rollout, where
  drift compounds — the key stability signal), `sharpness_ratio` (1.0 ideal; <1
  blurry, >1 artifacts), `mass_drift` (relative total-mass error), `train_loss`,
  `mmd` (saturating kernel MMD, sanity), `ed_floor` (noise floor), `wall_s`.
- **Rollout_ed (early/overall) can HIDE instability** — always check ed_late +
  sharpness + mass too (lesson from prior sessions: RF looked "tied" at early ED but
  ed_late/sharpness showed it was already diverging).

## How to Run
`.auto/measure.sh` — sources `.auto/run.env` (rewritten each iteration) and runs
`video_rollout_experiment.py`. All knobs are env vars. Outputs `METRIC name=value`.

## The Lever (what to edit / sweep)
The joint objective is implemented in `train()` (`if JOINT_GEN and ARCH=="jit3d":`)
plus helpers `_bridge_future_loss()` and `_rf_context_loss()` and
`VideoRFJiT3D.forward_joint()`. Sweep the JOINT_* env knobs:
- `JOINT_GEN` (1=on, 0=baseline-off), `JOINT_CTX_WEIGHT` (context loss weight),
- `JOINT_COUPLE` (1: `s=min(1,t+JOINT_LEAD)`; 0: `s~U(JOINT_CTX_TMIN,1)`),
- `JOINT_LEAD` (how far the context time leads toward data), `JOINT_CTX_TMIN`,
- `JOINT_DECOUPLE` (1: pass BOTH [s,t] to the net, context_dim=2; 0: pass only t),
- `JOINT_LAST_ONLY` (1: RF-noise only the most-recent context slot).

## Files in Scope
- `video_rollout_experiment.py` — the whole benchmark. Edit freely.
- `.auto/run.env` — the per-iteration hyperparams (rewritten by the loop).
- `.auto/measure.sh` — benchmark runner.

## Off Limits
- Do NOT edit other `*_experiment.py` / `*.md` files (prior experiments).
- Do NOT cheat the metric: don't include rollout-like frames in the reference, don't
  train on holdout trajectories, don't shrink ROLLOUT_LEN/N_ROLLOUT to trivialize ED.
- The point is REAL rollout stability improvement.

## Constraints
- One GPU (L4). Keep iterations < ~4 min (TRAIN_STEPS=4500, jit3d-128/4/4, PHW8 ~ OK).
- Determinism ON (`DETERMINISTIC=1` + `CUBLAS_WORKSPACE_CONFIG`); verify reproducibility
  before trusting a surprising win (chaotic rollout × non-determinism = ±50% noise).
- Bridge future MUST use `BRIDGE_LOSS=vloss` (`uniform` collapses to context-pred on
  easy data; the `(1-t c'/c)^2` one-sided data-end upweight prevents collapse).
- Use `LORENZ_DT=0.024` (moderate-fast regime where the bridge >> RF is established;
  RF develops artifacts/mass-drift here — the regime we want to stabilize).
- Keep sharpness ~1.0 and mass_drift ~0 (don't game ED via high-freq artifacts).

## What's Been Tried
(filled in as experiments accumulate)
- BASELINE (JOINT_GEN=0, bridge+none): see log.jsonl entry 1.
