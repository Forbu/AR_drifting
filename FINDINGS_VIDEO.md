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
- **Error-GATED self-feed ("selffeed_msgate") is the regime-robust champion**: gate
  the self-feed by surrogate accuracy — keep only the low-error half (gate=0.5,
  measured vs the real frame). ED 690 → **654** on sparse data, AND it fixes the
  dense-data regime (multi-blob 2691 → 1822, near no-aug 1563). Adaptively
  self-feeds when predictions are good (sparse), skips when poor (dense). No
  sparsity detector needed. Gate sweep: 0.25→668, **0.5→654 (optimum)**, 0.75→699.
  **Budget caveat:** gating helps at low budget (2000: 654 vs ungated 690) but
  *hurts* at high budget (3500: gated 330 vs ungated **156**) — once well-trained
  all surrogates are decent, so the gate discards useful exposure. Use gated when
  budget-limited, ungated when training to convergence. Absolute (err/var) gating
  was tried and is worse (keeps a biased subset).
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

**Design principle (empirically validated): DETACH the model's own predictions**
in both the input-context self-feed and the multi-step rollout loss. Letting
gradient flow (BPTT, `selffeed_ms_grad`) made things worse — ED 690→950,
sharpness 0.91→0.75 — because it rewards "easy-to-predict-from" (degenerate /
blurry) outputs (partial mode collapse). The detached formulation decouples the
objective: the model is trained to *tolerate* imperfect context, not to *game*
the next-step loss.

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

## No-self-feed substitute: VAE/AE-latent context corruption

Scheduled sampling needs the model to *generate* context-like frames. For setups
where that's impossible (e.g. the user's weather model: **4 ctx -> 3 future,
predicted jointly, fixed 7-frame dataset** — no extra frames to roll out into, and
the model only predicts *future* frames, not context frames), we tested a learned
corruption model: train a VAE/AE on the data, corrupt context via **encode -> add
latent noise -> decode** (no model self-outputs, no extra data).

Results (deterministic, TRAIN_STEPS=2000):

| Technique | rollout_ed | sharpness | note |
|---|---|---|---|
| pixnoise | 6746 | 0.22 | fails (blur) |
| **VAE (β=1e-3) latent noise** | **12712** | **0.22** | **FAILS — worse than pixnoise!** |
| manifold_noise (blur+jitter) | 2142 | 1.26 | artifacty |
| AE (β≈1e-5) latent noise | 2113 | 0.77 | sharp AE works |
| AE latent-interp (random frame) + blur | 1268 | 1.18 | two-blob blends, artifacty |
| AE latent-interp (TEMPORAL neighbor) + blur | 1305 | **1.03** | clean position shift; best per-frame quality |
| **AE latent-noise + mild blur (σ_n=0.2)** | 1186 | 0.93 | + input aug only |
| **AE aug + self-feed-free multi-step loss (vae_ms)** | **1106** | 0.92 | **best no-self-feed (6.1× pixnoise)** — multi-step loss over AE-corrupted real frame |
| selffeed_ms (needs self-outputs) | 690 | 0.91 | champion |

vae_ms tuning: MS_PROB 0.15→1186, **0.3→1106 (optimum)**, 0.5→1798 (same 0.3
optimum as self-feed); VAE_NOISE 0.2; β=1e-5; blur 0.4.

VAE_NOISE sweep for the winner: 0.1→1600, **0.2→1186 (optimum)**, 0.3→2112
(sharp valley). Adding amplitude jitter on top *hurt* (over-corruption, 2077).

**Two strong no-self-feed options, different strengths:** `vae_noise_blur` wins on
rollout-ED (1186); `vae_interptemp_blur` wins on per-frame quality (sharpness 1.03,
mass_drift 0.006 — best of any technique). For weather (blurry = failure), the
temporal variant's sharpness may be preferable despite slightly higher ED.

**Why the VAE approach plateaus (~1186) and selffeed keeps improving:** selffeed is
a **closed loop** — the corruption *is* the model's own predictions, so as the model
improves, the corruption tracks its actual (shrinking) error distribution →
compounding gains with training (selffeed_ms: 690@2000 → 156@3500). The VAE
corruption is **static** (a fixed AE trained once) → no compounding; in fact
`vae_noise_blur` got *worse* with more training (1186@2000 → 1419@3500,
train_loss 0.14→0.64). This is the structural reason selffeed is superior and why
any proxy caps out. **Implication for the weather model:** if you can enable even a
small auxiliary AR head to generate context-frame proxies, scheduled sampling
unlocks the compounding benefit; otherwise the sharp-AE proxy is the best static
option (~1186).

**Critical principle: the corruption MUST stay sharp.** A normal VAE (KL>0)
blurs its reconstruction AND its decoder regresses perturbed latents to the data
mean → corrupted contexts are blurry/mean-ish → the model learns mean regression
(ED 12712, identical failure to pixel noise, sharpness 0.22). Fix: use a **sharp
autoencoder** (β≈0, recon MSE ~5e-6, near-perfect recon) + small latent noise +
mild blur. Then corruptions are *sharp but slightly off-manifold* — the same
regime as the model's own outputs that makes self-feed work. Adding amplitude
jitter on top *hurt* (over-corruption).

**Recommendation when self-feeding is impossible:** train a sharp AE on your
frames; during RF training corrupt context frames (w.p. ~0.3, decoupled) via
encode -> +small latent noise -> decode -> +mild blur. This is the best
self-feed-free option found (5.7× better than pixel noise), though ~1.7× worse
than true scheduled sampling (which remains preferable wherever feasible).

## Regime dependence: data STRUCTURE matters (important for weather)

The self-feed recommendation is **regime-specific**. Tested on multi-feature data
(N_BLOBS=2: 3 independently-moving blobs, simulating dense/complex fields):

| Technique | sparse (single blob) | dense (3 blobs) |
|---|---|---|
| no augmentation | 5940 | 1563 |
| pixel noise | 6746 | 1751 |
| selffeed_ms (ungated) | 690 | 2691 ❌ |
| selffeed_msgate (error-gated, gate=0.5) | 654 | 1822 |
| **vae_ms (AE-corruption + self-feed-free multi-step loss)** | 1106 | **821** ✅ |

**`vae_ms` is regime-robust** — it works on *both* regimes and is the only technique
that excels on dense data (821, beating even no-aug 1563 by 1.9×). Self-feed is
sparse-only: it wins on sparse (654) but catastrophically fails on dense (2691).
Reason: `vae_ms` corruption is **static + on-manifold** (no mean-regression like
pixel noise, no dependence on model prediction quality like self-feed), so it
regularizes consistently regardless of task difficulty.

**This corrects the earlier "augmentation hurts on dense" claim** — only *bad*
augmentation (pixel noise → mean regression; self-feed → bad surrogate) hurts on
dense. On-manifold AE corruption *helps* on dense.

**Recommendation for mixed sparse/dense real weather data:** `vae_ms` is the
robust default (works everywhere). Use self-feed *only* if your data is reliably
sparse AND you can generate self-feed signals; otherwise prefer `vae_ms`.

On **sparse-feature** data self-feed helps ~9×; on **dense-feature** data
augmentation of any kind *hurts* and no-aug is best. Two reasons:
1. Pixel-noise's catastrophic-blur failure only happens when the data **mean is
   degenerate** (sparse features → mean is blank). Dense data's mean looks like
   valid data, so pixel noise is harmless there.
2. Self-feed's "train on your own errors" needs decent per-step predictions. On
   hard/dense tasks the surrogate is poor → it feeds the model bad contexts →
   worse (train_loss 0.08 sparse → 0.32 dense).

**Implication for weather:** match the technique to the regime.
- **Sparse / isolated features** (convective cells, lightning, clear-air) → the
  user's stated "blurry cells" failure mode → **self-feed wins** (this is the
  regime the recommendation targets).
- **Dense / widespread** (stratiform precipitation, overcast) → augmentation is
  unnecessary and may hurt; a clean-condition model is already stable.
Mixing regimes (real radar has both) likely benefits from self-feed applied
*conditionally* (only when the context is sparse) — untested.

## Caveats / not overfitting

- Validated on a *single-blob Lorenz* dataset. The mechanism (exposure-bias fix)
  is data-agnostic, but the exact probability / blur σ should be re-tuned on real
  weather frames.
- Confirmed generalization to **multichannel (C=2)** and a different seed,
  including the NEW champion `selffeed_ms` (deterministic, SEED=1, C=2):
  ED 12269 vs pixnoise 35265 (**2.9× better**), sharpness **0.91**, mass_drift
  **0.05**. The added 2-step rollout loss did NOT overfit to the single-channel
  benchmark. (Earlier `selffeed_m` C=2: 3× better, sharpness 0.88.)
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

## Brownian-Bridge Flow Matching (FLOW=bridge) — AR-stability investigation

**Question (user):** Replace the linear rectified-flow path with a Brownian-bridge
probability path (cf. `hypersphere_bridge_experiment.py`, Lim et al. 2024
arXiv:2410.03229) and combine with the jit3d backbone + manifold context aug, to
improve AR rollout stability. Bridge path: `z_t = (1-t)eps + t y + c_t eta`,
`c_t^2 = sigma^2 t(1-t) + sigma_min^2` — variance minimal at BOTH endpoints
(clean data landing at t=1 -> minimal off-manifold jitter fed forward into the
next AR step), maximal mid-path.

### Implementation
- x-prediction model (predicts clean target `y`); three loss modes via `BRIDGE_LOSS`:
  - `vloss` (default): velocity-matching `||v_theta-u_t||^2 = (x_pred-y)^2 (1-t c'/c)^2`
    clamped. A **one-sided data-end upweight** (like RF's `1/(1-t)^2`) — prevents the
    context-prediction collapse. The literal bridge v-loss.
  - `ivar`: inverse-conditional-variance `1/c_t^2` (upweights BOTH endpoints).
  - `uniform`: plain x-pred MSE (the user's production reference; collapses here).
- Exact closed-form ODE sampler: `z(t+dt) = mu_next + (z-mu_t) c(t+dt)/c(t)` —
  absorbs the stiff `c'/c` mean-reversion into the O(1) ratio; lands exactly on
  `x_pred` at t=1. (Naive Euler with the bridge velocity diverges via c'/c.)

### Results (jit3d 800k champion arch, SEED=0, dt=0.012)

**1. Loss type matters — uniform COLLAPSES:**
| BRIDGE_LOSS | rollout_ed | sharp | mass | train_loss | note |
|---|---|---|---|---|---|
| vloss (wclamp10) | 1124 | 1.09 | 0.036 | 3.8e-5 | **best** — no collapse |
| ivar (wclamp200) | 1103 | 0.95 | 0.040 | 0.0011 | tied ED, slightly worse quality |
| uniform | 1363 | 1.18 | 0.072 | 3.8e-5 | **COLLAPSE** to context-pred |

Uniform x-pred collapses to "predict from context, ignore z_t" on this easy
single-blob data (train_loss anomalously low, sharpness 1.18 artifacts). The
`1/(1-t)^2` / `1/c_t^2` weighting exists precisely to prevent this by forcing z_t
usage at the data end. The user's uniform "v-loss" works on their production data
(multi-channel sat+lightning — context alone can't determine the target -> no
collapse) but NOT on easy data. **The vloss `(1-t c'/c)^2` weight is the correct
well-behaved bridge loss** (one-sided data-end upweight, no collapse). Note:
train_loss magnitude is NOT comparable across loss types (weight scale differs);
model QUALITY (sharp/mass) is the signal.

**2. sigma sweep (U-shape, vloss, wclamp10):** 0.5->ed_late 1630, **0.3->1222
(min)**, 0.2->1736. sigma=0.3 optimal — too much mid-path noise blurs, too little
loses the bridge regularization (path->RF, weight->collapse regime).

**3. wclamp:** 10 optimal. wclamp=6 (RF's optimal) -> sharp 1.16 artifacts, mass
0.062 (ED drop is metric-gaming via high-freq artifacts). RF's clamp doesn't
transfer to the bridge weight profile.

**4. sigma_min:** 1e-3 optimal. 1e-4 (sharper landing) -> sharp 1.17 artifacts
(too-sharp endpoint amplifies x_pred error). The small residual softens the landing.

**Best bridge config:** `FLOW=bridge BRIDGE_LOSS=vloss BRIDGE_SIGMA=0.3
BRIDGE_WCLAMP=10 BRIDGE_SIGMA_MIN=1e-3` -> ED 1174, ed_late 1222, sharp 1.06,
mass 0.025 (vloss sigma sweep at sigma=0.3; ivar sigma=0.5 baseline was ED 1103).

### vs RF champion (dt=0.012): bridge ~1.4x RF ED (1174 vs 803), competitive sharpness
On EASY data (single moderate-motion blob) the well-tuned RF is already great, so
the bridge does NOT improve ED here. Sharpness competitive (1.06 vs 1.03).

### KEY FINDING — bridge >> RF in the UNSTABLE regime (dt=0.024 fast motion)
At dt=0.024 the RF champion is severely unstable (run 158, same env). The bridge
(vloss/sigma=0.3) dramatically tames it (clean comparison, same arch/regime):
| metric | RF (run158) | **Bridge** | ratio |
|---|---|---|---|
| rollout_ed | 4326 | **1541** | **2.8x better** |
| sharpness | 2.87 (severe artifacts) | **1.13** (mild) | tamed |
| mass_drift | 0.60 | **0.086** | **7x better** |
| ed_late | 39817 | **2662** | **15x better** |

(sigma=0.3 above; tuned further below.)

This is the bridge's **stated purpose demonstrated**: the clean endpoint landing
(variance->sigma_min^2~0) means each AR step injects minimal off-manifold jitter
into the next -> drift doesn't compound -> stable long rollout. RF's straight-line
path (variance 0 everywhere) overshoots in fast motion; the bridge's
mean-reverting path is self-correcting. The bridge's value appears **WHERE RF
STRUGGLES** (hard/unstable dynamics), matching the hypersphere result (bridge
helped there too — harder data).

### Optimal sigma is REGIME-DEPENDENT (key for production tuning)
Sweeping BRIDGE_SIGMA at dt=0.024 (both regimes are U-shaped, but the optimum
shifts HIGHER for harder dynamics):
| regime | sigma 0.3 | sigma 0.5 | sigma 0.7 | optimum |
|---|---|---|---|---|
| dt=0.012 (easy) | **1174** | (0.5 blurs) | — | **0.3** |
| dt=0.024 (unstable) | 1541 | **1312** | 1949 (blur) | **0.5** |

Tuned bridge (sigma=0.5) at dt=0.024 beats the RF champion even harder, and the
rollout is now self-correcting (ed_late 1153 < rollout_ed 1312 — frames converge
toward the manifold over time, drift decays instead of compounding):
| metric (dt=0.024) | RF | bridge s=0.3 | **bridge s=0.5** | vs RF |
|---|---|---|---|---|
| rollout_ed | 4326 | 1541 | **1312** | **3.3x** |
| ed_late | 39817 | 2662 | **1153** | **35x** |
| sharpness | 2.87 | 1.13 | **1.07** | tamed |
| mass_drift | 0.60 | 0.086 | **0.031** | **19x** |

**Transferable rule:** tune BRIDGE_SIGMA per dynamics regime — faster/harder /
more-unstable dynamics want a higher sigma (more mid-path Brownian regularization
to tame the instability that cripples RF). Too little -> RF-like drift; too much
-> blur. Both extrema fail (U-shape).

### Bridge-vs-RF crossover is at MODERATE motion (dt~0.018) — key for production
Characterized bridge vs RF across motion speed (all same arch/env; rollout_ed is
early/overall, ed_late is the late-half where drift shows):
| motion | RF rollout_ed | RF ed_late | RF sharp | bridge rollout_ed | bridge ed_late | bridge sharp | winner |
|---|---|---|---|---|---|---|---|
| dt=0.012 (slow) | 803 | 802 | 1.03 | 1174 | 1222 | 1.06 | **RF** |
| dt=0.018 (moderate) | 1030 | **5141** | **1.38** | 1065 | **1112** | 1.06 | **bridge** |
| dt=0.024 (fast) | 4326 | 39817 | 2.87 | 1312 | 1153 | 1.07 | **bridge** |

The crossover is at **dt~0.015-0.018 (moderate-fast)**, not extreme motion. At
dt=0.018 the RF's early rollout_ed (1030) looks ~tied with the bridge (1065), but
RF is ALREADY unstable: ed_late 5141 (4.6x worse), sharpness 1.38 (artifacts),
mass 0.156 (11x worse) — it diverges over the long rollout. The bridge stays
rock-solid (ed_late~ED, sharp~1.06, mass~0.014).

**Methodology lesson:** rollout_ed (early/overall) HIDES RF's instability onset —
the divergence shows in ed_late / sharpness / mass. Always check ed_late +
sharpness + mass, not only rollout_ed, when judging rollout stability.

**Production recommendation (strengthened):** for any moderate-fast dynamics
(dt>=~0.018, i.e. realistic weather motion), the well-tuned RF champion develops
artifacts + mass drift in long rollouts while the bridge stays stable. Use the
bridge for production forecasting; tune sigma per regime (~0.3 slow, ~0.5 fast).
RF is only preferable for very slow/trivial dynamics.

### manifold_noise + bridge: HURTS
manifold_noise (blur+jitter context aug) over-corrupts the bridge (ED 1732 vs
bridge+none 1103). The bridge ALREADY regularizes the target via mid-path Brownian
noise + clean endpoint; adding context aug is redundant/conflicting. Use ONE, not
both. (For the bridge, no context aug = TECHNIQUE=none is best.)

### Recommendation
The bridge is **NOT a universal win over RF**. It's competitive on quality where RF
is already good, and dramatically better where RF is unstable (fast/hard dynamics).
For production: use the bridge if your dynamics are fast/unstable or you observe RF
rollout artifacts / mass drift; otherwise a well-tuned RF is fine. The `vloss`
loss is required (uniform collapses on easy data; the `(1-t c'/c)^2` one-sided
data-end upweight prevents collapse while being a genuine velocity loss).
