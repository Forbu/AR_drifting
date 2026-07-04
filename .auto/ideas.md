# Ideas backlog (deferred / future)

## High-value, not yet tried on this benchmark
- **Curriculum on selffeed probability**: ramp p from 0→0.3 over training to avoid
  poisoning early training with bad predictions. Likely improves stability further
  and allows higher final p.
- **Selffeed with gradient** (truncated BPTT through the surrogate step) instead of
  detach: lets the model learn to *correct* its errors, not just tolerate them.
  More expensive; may help sharpness.
- **EMA teacher for the surrogate prediction**: use a moving-average model to
  generate cleaner self-feed contexts (reduces noise in the degraded condition).
  Analogous to consistency-model / DPT teacher.
- **Consistency-loss / self-distillation**: add a loss term that pulls the
  one-step sample toward a multi-step (ODE-solved) sample — enforces rollout
  consistency directly.

## Other stabilization techniques to compare
- **Classifier-free guidance** on the context (drop context w.p. during training,
  guide at inference) — different mechanism, may combine with selffeed.
- **Restart / renoise during ODE sampling** at each AR step (inject structured
  noise mid-denoise) — stochasticity may prevent drift compounding.
- **Diffusion-Forcing-style per-frame noise schedule** (technique `diff_forcing`
  is implemented but not yet benchmarked) — each context frame noised along an
  independent schedule.
- **Spectral / TV regularizer** (`spectral` technique implemented, untested) —
  penalize excess high-frequency energy in predictions to suppress artifacts.
  May pair well with selffeed_m to push sharpness exactly to 1.0.

## Generalization checks (do before trusting for production)
- Harder dynamics: faster Lorenz (larger LORENZ_DT), or 2 independent blobs that
  cross — tests that selffeed isn't exploiting the single-blob structure.
- Larger images (IMG=64) and more channels (C=4) — confirm model capacity and
  that the blur σ scales with resolution.
- Longer rollout (ROLLOUT_LEN=120) — find where selffeed_m eventually diverges.
- K_CTX=3 / K_CTX=4 (more context) — does selffeed still help, or does extra
  context already stabilize?

## Metric improvements
- The energy distance is dominated by blob-position (pooled features). Add a
  texture-only ED (gradient/Laplacian features alone) as a secondary to separate
  "wrong position" from "wrong texture" failure modes.
