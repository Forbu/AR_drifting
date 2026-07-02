# Findings: Brownian-Bridge Flow vs Rectified Flow for Autoregressive Stability

**Date:** 2026-07-02
**Goal:** Decide whether to replace rectified flow (RF) with a Brownian-bridge
probability path in the weather forecasting model, motivated by AR-rollout
stability (each model call's output becomes the next call's condition).

Benchmark: random walk on a hypersphere `S^{D-1}`. AR stability = how well
`||x|| ≈ 1` is maintained over many autoregressive steps, plus radial-energy
(off-manifold drift) and angular displacement. Lower `|err| = | ||x|| - 1 |`
is better.

---

## TL;DR

- **The bridge path loses to RF on this benchmark.** Best bridge config
  (`bridge+coupled`) reaches `|err| ≈ 0.41`; RF + decoupled condition-noise
  augmentation reaches `|err| ≈ 0.03` — over an order of magnitude better.
- **The AR-stability lever is *condition-noise augmentation*, not the flow
  path.** Ablation (`decoupled-uncond`, trained on noisy conditions but not
  told the noise level) matches the informed version, proving stability comes
  from training diversity.
- **The bridge under-disperses** (collapses `||x||` inward toward the mean).
  For weather this is the "blur out convective cells / lightning" failure
  mode. RF does not.
- **Recommendation for the weather model:** keep RF (linear interpolation) +
  the existing context blur/noise augmentation. Do **not** train the weather
  model with the bridge path. The branch is shelved as "explored, didn't pan
  out for this cadence."

---

## Headline result (D=64, speed=0.3, 200 AR steps)

From `hypersphere_flow_experiment.py` (RF) and `hypersphere_bridge_experiment.py` (bridge):

| Config | `|err|` | Note |
|---|---|---|
| baseline (clean RF) | **NaN** | explodes — no augmentation |
| coupled/noisy | 0.244 | RF + condition noise tied to `t` |
| coupled/clean | 0.276 | RF + coupled aug, clean inference |
| **decoupled/s=0.0** | **0.033** | RF + decoupled aug, clean inference |
| **decoupled-uncond** | **0.031** | same but no `s` input — the ablation |
| bridge+cou/s=0.5 | 0.41 | bridge + coupled aug |
| bridge+dec/s=0.5 | 0.98 | bridge + decoupled aug — severe inward collapse |
| bridge (clean) | inf/diverges | no aug |

Lower `|err|` is better. Decoupled RF wins by ~13× over the best bridge config.

---

## What actually drives AR stability

The `decoupled-uncond` ablation is decisive. It is trained with random
condition-noise `s ~ lognormal` but **the noise level is not passed to the
model**. At inference it uses a clean condition. It reaches `|err| ≈ 0.03` —
the same as the `s`-informed decoupled model. Therefore:

> Stability comes from the model seeing degraded conditions during training
> (so it doesn't extrapolate catastrophically on drifted AR inputs), **not**
> from the flow path, and **not** from informing the model about the noise
> level at inference.

This is the hypersphere analog of the context blur/noise augmentation already
present in `xpred_blur_v2` — and it is the load-bearing ingredient there too.

---

## Why the bridge underperformed (hypothesis)

The bridge path (`z_t = (1-t)x0 + t·x1 + c_t·η`, variance minimal at both
endpoints, maximal mid-path) was motivated by Lim et al. 2024
("Elucidating the Design Choice of Probability Paths in Flow Matching for
Forecasting", arXiv:2410.03229), whose win condition is **highly correlated
source/target pairs** (the path is shortened when `x0 ≈ x1`). In this
benchmark `speed=0.3` gives moderate correlation, and:

1. The bridge's endpoint-variance property does **not** directly reduce AR
   drift (a deterministic ODE sampler has no stochasticity to inject at the
   endpoint regardless).
2. The inverse-variance training weight `1/c_t²` heavily emphasizes the data
   end, which on this data biases outputs toward the conditional mean →
   inward norm collapse (`||x|| < 1`). The decoupled bridge variant collapsed
   to `||x|| ≈ 0.02`.

The bridge may still be worth revisiting for **near-straight-path regimes**
(sub-minute cadence, or forecasting in a latent space where consecutive
states are very close). 10-minute weather cadence is not that regime.

---

## Bugs found and fixed along the way

Investigating the bridge required fixing three real bugs. None change the
conclusion above, but the fixes stand on their own and some apply to the
weather model:

1. **Trivial-loss collapse.** Plain unweighted x-prediction MSE collapses to
   "predict the condition" (loss `≈ speed²/D ≈ 0.0013` at D=64, reached in 1
   epoch; model learns no dynamics). Fix: inverse-conditional-variance weight
   — `1/(1-t)²` for RF, `1/c_t²` for the bridge (Gagneux & Martin 2026,
   "Training Flow Matching: The Role of Weighting and Parameterization",
   arXiv:2603.06454). *Already correct in the weather `linear` branch
   (`1/t²`); the bridge branch in flashnet now matches (`1/c_t²`).*

2. **Stiff-Euler integrator divergence.** The bridge velocity contains
   `c'/c = σ²(1-2t)/(2·c_t²)`, which blows up to `~1/σ_min²` at the endpoints.
   Naive Euler amplified per-step error and diverged to `||x|| ~ 1e14` at
   D=64. Fix: exact closed-form step
   `x(t+dt) = μ_{t+dt} + (x(t) - μ_t)·c(t+dt)/c(t)`, which absorbs the stiff
   coefficient into an O(1) ratio. *Ported to the flashnet inference engine
   and `blur_v2.py`; relevant to any future high-D bridge use.*

3. **Unclamped training weight → NaN.** `1/c_t²` reaches `~1e6` at the
   endpoints; over a full run this NaN'd the weights. Fix: `clamp(1, 200)`
   on the bridge weight (RF branch already clamped `(1-t)` at `1e-2`).
   *Applied to flashnet's bridge branch too.*

---

## Actionable takeaways for the weather model

1. **Keep RF.** Do not adopt the bridge for 10-minute-cadence weather.
2. **Keep and trust the context blur/noise augmentation** (`xpred_blv2`'s
   `sigma` blur path) — it is the AR-stability mechanism, validated here.
3. **Mirror the augmentation at inference.** During AR rollout the model's
   own slightly-drifted outputs are *never* clean conditions, so feeding them
   in raw creates a train/inference mismatch. The decoupled results suggest
   adding a small amount of blur/noise to the context frames at each AR step
   (matching the training-noise distribution) is worthwhile — analogous to
   `decoupled/s≈0.1` here. This is the single most weather-relevant knob.
4. **The bridge branch is preserved** (exact integrator + inverse-variance
   loss, in both `AR_drifting` and flashnet) for future near-straight-path
   regimes, but is not the production path.

---

## Reproducing

```bash
# RF baseline + augmentation (the comparison that won):
python hypersphere_flow_experiment.py --dims 64 --speeds 0.3

# Bridge variants (the comparison that lost):
python hypersphere_bridge_experiment.py --dims 64 --speeds 0.3 --sigmas 0.5 1.0
```

Commits implementing the bridge + fixes (in `AR_drifting`):
`42ab32b` bridge setup → `95e9ca6` aug → `a806138` exact integrator →
`fe9e3fc`/`3247e57` inverse-variance loss + clamp.

Commits in `flashnet` (branch `brownian-bridge-flow-path`):
`bccad59` schedule_power → `a4b7c76` bridge path → `593dbd0` engine wiring →
`4308a80` exact integrator → `89c8f53` inverse-variance trainer weight.
