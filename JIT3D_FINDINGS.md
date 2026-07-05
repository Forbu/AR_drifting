# JiT-3D (Production Architecture) — Latent-Noise-Corruption Study + Backbone Comparison

**Question (user):** Use the production JiT-3D ViT with different latent-noise
corruption settings to improve AR rollout performance; compare with the old conv
backbone and with input-space jitter/blur augmentation.

## TL;DR
1. The ViT needs **two training fixes** before anything works: **fp32** (bf16
   diverges under the RF `1/(1-t)²` loss) and a **lower loss-weight clamp
   (`RF_WCLAMP=50`, default 200)** that tames the t≈0 instability. With these,
   the ViT trains properly (train_loss **0.068**, beating conv's 0.13).
2. **Latent-noise corruption WORKS on the well-trained ViT** — it improves both
   rollout ED (26104→23933) and sharpness (0.21→0.23) with no collapse. The
   earlier "it collapses" finding was an artifact of the undertrained model.
3. **Input-space jitter/blur (manifold_noise) is worse than latent corruption on
   the well-trained ViT** (ED 31611, sharp 0.18). There is a regime flip:
   input-aug wins when the ViT is *undertrained*, latent corruption wins when
   *well-trained*.
4. **The conv backbone still crushes the ViT on this tiny benchmark** (ED 451 vs
   23933; sharp 1.11 vs 0.23) — the ViT is data-hungry and the conv's
   locality/translation inductive bias is perfect for blob forecasting. The ViT's
   advantage appears at production data scale, not here.

## Setup
- `ARCH=jit3d` wraps the real production model
  (`../flashnet/meteolibre_model/models/jit3d.py:JiT3D_Modern`) behind the same
  RF `(z_t, ctx, t) -> x_pred` interface. K=2 context + 1 target stacked along
  time; the model predicts all frames, we supervise the target slice.
- The production `LatentContextCorruptor` (per-token-L2-normalized Gaussian
  noise on context tokens, at embed + after-block-0 stages) is exposed via
  `JIT_CORRUPT_{PROB,EMBED,BLOCK0}`.
- Config: embed_dim=128, depth=4, heads=4, patch=(1,4,4) → ~800k params.

## Critical training fixes (prerequisite — both required for jit3d)
| Fix | Why | Effect |
|---|---|---|
| `AMP=0` (fp32) | bf16 has insufficient range for the 200× loss-weight spikes at t≈0 → cascade to NaN | prevents divergence |
| `RF_WCLAMP=50` (was 200) | caps the `1/(1-t)²` weight; max loss spike 2.75→0.72 | train_loss 0.22→0.068, sharpness 0.10→0.21 |
| LR `1e-4` | 2e-4 knife-edge, 4e-4 diverge | stable |
| `grad_clip=1.0` (not 0.5) | 0.5 causes Adam second-moment amplification | stable |
| no warmup | warmup→cosine holds peak LR longer → more divergence | stable |
| ≤3000 steps | degrades past ~3000 (loss 0.42→1.28 @4000) | ceiling |

## Result table (jit3d = fp32, WCLAMP=50, 3000 steps, LR 1e-4 unless noted)

| Backbone | Aug | rollout_ed | sharp | mass_drift | train_loss |
|---|---|---|---|---|---|
| conv2d | manifold_noise + AVG4 (champion) | **451** | **1.11** | **0.054** | 0.13 |
| jit3d | none (well-trained baseline) | 26104 | 0.21 | 0.53 | **0.068** |
| jit3d | **latent corrupt PROB=.3 EMBED=.10 BLOCK0=.05** | **23933** | **0.23** | 0.57 | 0.37 |
| jit3d | manifold_noise (input jitter+blur) | 31611 | 0.18 | 0.22 | 0.15 |
| jit3d | latent corrupt PROB=.15 EMBED=.02 (WCLAMP=200, undertrained) | 19709* | 0.08* | 0.74* | 0.16 |

\* undertrained regime (WCLAMP=200) — the ED "win" is a blur-collapse artifact
(see below); do not trust.

## Key findings

1. **Latent corruption's effect depends entirely on training maturity.**
   - Undertrained ViT (WCLAMP=200, loss 0.22): corruption → context-ignoring →
     blur collapse (sharp 0.10→0.075; ED appears to drop but it's metric-gaming:
     blurry frames regress toward the reference mean on avg-pool/mass features).
   - Well-trained ViT (WCLAMP=50, loss 0.068): corruption → genuine robustness →
     sharper, lower-ED rollout (sharp 0.21→0.23, ED −8%), no collapse.
   - **Transferable rule:** the LatentContextCorruptor only helps once the model
     is trained well enough that it *can't afford to ignore context*. Validate
     corruption magnitude against a well-trained checkpoint, not a fresh one.

2. **Input-space jitter/blur (manifold_noise) vs latent corruption — regime flip.**
   Input aug is the safer choice when training is immature (it can't cause the
   context-ignoring collapse because it perturbs inputs, not the context
   representation). Once the model is well-trained, latent corruption wins.

3. **The ViT is far behind conv here, and that's expected.** ~53× worse ED, ~5×
   worse sharpness. The conv's locality/translation-equivariance inductive bias
   is ideal for the Lorenz-blob task; the ViT must learn it from 220 trajectories.
   The ViT's production advantage (scaling, long-range dependencies) needs
   production-scale data to appear.

4. **Metric caveat (important).** At these large ViT ED magnitudes (vs conv's
   451), rollout ED is dominated by positional/avg-pool features and can be
   gamed by blur-collapse. **Always cross-check sharpness_ratio and mass_drift**
   before trusting an ED drop on the ViT.

## Recommendation for the user's production model
- Keep the LatentContextCorruptor, but **validate its magnitude on a fully-trained
  checkpoint**. The failure mode to monitor is blur collapse via
  context-downweighting — watch rollout **sharpness + mass/energy drift**, not
  only distributional ED.
- Use **fp32** (or loss-scaled mixed precision) for this ViT + RF-velocity loss;
  bf16 diverges.
- Consider lowering the RF loss-weight clamp if you see optimizer instability
  spikes (it trades a small amount of high-t emphasis for big stability gains).
- Input-space context augmentation (blur + amplitude jitter) is a robust
  complement — safer early in training, and a good fallback if the latent
  corruptor shows blur-collapse signs.
