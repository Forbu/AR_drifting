"""
Hypersphere Random Walk — Brownian-Bridge Flow vs Rectified Flow (AR Stability)

Companion to hypersphere_flow_experiment.py. Same data (random walk on S^{D-1})
and same AR-stability metrics, but the probability path is changed from a
straight rectified-flow line to a Brownian-bridge interpolant:

    Rectified flow:  z_t = (1-t) eps + t y            (variance 0 everywhere)
    Bridge flow:     z_t = (1-t) eps + t y + c_t eta   with
                     c_t^2 = sigma^2 * t(1-t) + sigma_min^2

Convention (same as the rest of the hypersphere XPs):
    t = 0 -> noise (eps),  t = 1 -> data (y),  forward Euler  z += v dt.

The bridge variance is *minimal at both endpoints* (sigma_min^2) and *maximal
in the middle* (sigma^2/4 at t=0.5). Motivation for autoregressive forecasting:

  In AR rollout, each model call's output becomes the next call's condition.
  Any residual sampling jitter at the data endpoint (t=1) is fed forward and
  accumulates as off-manifold drift. The bridge drives the endpoint variance
  toward sigma_min^2 ~ 0, so each step lands cleanly on the manifold with
  nothing to inject into the next step. Mid-path, where structure is formed,
  the bridge still allows fluctuations (unlike RF which is a deterministic
  line). Ref: Lim et al. 2024, "Elucidating the Design Choice of Probability
  Paths in Flow Matching for Forecasting" (arXiv:2410.03229).

The model is x-prediction (predicts y from (z_t, cond, t)), identical to the
existing baseline. Only the input distribution z_t and the inference velocity
change — so weights could in principle be warm-started (not done here; we train
from scratch for a clean comparison).

Vector field (deterministic flow that generates the bridge path):
    v_t = (y - eps) + (c'_t / c_t) (z_t - mu_t),   mu_t = (1-t) eps + t y
At inference eps is the fixed source sample and y is replaced by the model's
x-prediction, so the velocity is fully determined:
    v_t = (x_pred - eps) + (c'/c) (z - mu_t),       mu_t = (1-t) eps + t x_pred

Usage:
    python hypersphere_bridge_experiment.py --quick
    python hypersphere_bridge_experiment.py --dims 16 32 64 --sigmas 0.3 0.5 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import time


# ============================================================================
# 1. BRIDGE COEFFICIENTS
# ============================================================================

def bridge_coeffs(t, sigma, sigma_min):
    """
    Brownian-bridge path coefficients (convention: t=0 noise, t=1 data).

        c_t^2    = sigma^2 * t(1-t) + sigma_min^2
        c'_t/c_t = sigma^2 (1-2t) / (2 c_t^2)

    Returns (c_t, cp_over_c) as float tensors. Works for scalar or tensor t.
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    var = sigma ** 2 * t * (1.0 - t) + sigma_min ** 2
    c = torch.sqrt(var)
    cp_over_c = sigma ** 2 * (1.0 - 2.0 * t) / (2.0 * var + 1e-12)
    return c, cp_over_c


# ============================================================================
# 2. DATA — Hypersphere Random Walk (same as hypersphere_flow_experiment.py)
# ============================================================================

class HypersphereWalkDataset:
    """
    Single-step transitions on S^{D-1}.
    (x, y): current point and next point after a random tangent step.
    """

    def __init__(self, D=16, speed=0.2, n_samples=50_000, seed=42, device='cuda'):
        self.D = D
        self.speed = speed
        self.n_samples = n_samples
        self.device = device

        rng = np.random.RandomState(seed)
        x = rng.randn(n_samples, D).astype(np.float32)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)

        g = rng.randn(n_samples, D).astype(np.float32)
        dot = np.sum(g * x, axis=1, keepdims=True)
        v = g - dot * x
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)
        mask = v_norm < 1e-8
        v[mask.squeeze()] = 0.0
        v_norm[mask] = 1.0
        v = v / v_norm

        y = x + speed * v
        y = y / np.linalg.norm(y, axis=1, keepdims=True)

        self.x = torch.tensor(x, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)

    def get_batches(self, batch_size, shuffle=True):
        if shuffle:
            idx = torch.randperm(self.n_samples, device=self.device)
        else:
            idx = torch.arange(self.n_samples, device=self.device)
        for i in range(0, self.n_samples, batch_size):
            bi = idx[i:i + batch_size]
            yield self.x[bi], self.y[bi]


# ============================================================================
# 3. MODEL — x-prediction MLP (predicts clean target y)
# ============================================================================

class FlowMLP(nn.Module):
    """
    Conditional flow model, x-prediction.
    Input: [z_t, x_cond, t (+ optional s)] -> predicts clean data y_hat in R^D.

    `with_s` is retained for completeness but unused now (decoupled mode was
    removed); all configs use with_s=False.
    """

    def __init__(self, D, hidden_dim=256, n_layers=5, with_s=False):
        super().__init__()
        self.D = D
        self.with_s = with_s
        in_dim = 2 * D + 1 + (1 if with_s else 0)
        layers = []
        for i in range(n_layers):
            out_dim = D if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, x_cond, t, s=None):
        if self.with_s:
            return self.net(torch.cat([z_t, x_cond, t, s], dim=-1))
        return self.net(torch.cat([z_t, x_cond, t], dim=-1))


# ============================================================================
# 4. TRAINING
# ============================================================================

def train_rectified(dataset, hidden_dim=256, n_layers=5, n_epochs=300,
                    batch_size=512, lr=1e-3, device='cuda'):
    """Standard rectified flow: z_t = (1-t) eps + t y.

    x-prediction with the inverse-conditional-variance weight w(t) = 1/(1-t)^2
    (the optimal FM / SNR weight; Gagneux & Martin 2026, "Training Flow Matching").
    Plain unweighted x-prediction collapses to the trivial 'predict the
    condition' solution: on this dataset that achieves loss ~ speed^2/D ~ 0.0013
    at D=64 (reached in 1 epoch) and the model never learns the dynamics. The
    1/(1-t)^2 weight upweights the low-noise regime (t->1) where the model must
    actually pinpoint y, preventing the collapse and matching the original
    experiment's loss regime (~0.06). Inference is unchanged (uses x_pred).
    """
    D = dataset.D
    model = FlowMLP(D, hidden_dim, n_layers).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    losses = []
    for epoch in range(n_epochs):
        el, nb = 0.0, 0
        for x_cond, y in dataset.get_batches(batch_size):
            B = x_cond.shape[0]
            t = torch.rand(B, 1, device=device)
            eps = torch.randn_like(y)
            z_t = (1 - t) * eps + t * y
            y_hat = model(z_t, x_cond, t)
            # inverse-variance weight: 1/((1-t)^2) for the linear path.
            # Clamp t away from 1 to keep the weight finite.
            w = 1.0 / ((1 - t).clamp(min=1e-2) ** 2)
            loss = (w * (y_hat - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            el += loss.item(); nb += 1
        sched.step()
        avg = el / nb
        losses.append(avg)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    [rectified] epoch {epoch+1:4d}/{n_epochs}  loss={avg:.6f}")
    return model, losses


def train_bridge(dataset, sigma=0.5, sigma_min=1e-3, hidden_dim=256, n_layers=5,
                 n_epochs=300, batch_size=512, lr=1e-3, device='cuda'):
    """
    Brownian-bridge flow: z_t = (1-t) eps + t y + c_t eta,  c_t^2 = sigma^2 t(1-t) + sigma_min^2.

    x-prediction with the INVERSE-CONDITIONAL-VARIANCE weight w(t) = 1/c_t^2.
    This is the bridge analog of the RF 1/(1-t)^2 weight (Gagneux & Martin
    2026, derived from maximum-likelihood / inverse-variance regression): both
    upweight the low-noise data end where the model must pinpoint y, and neither
    explodes because the weight is applied to the x-prediction residual (which
    stays O(1)), NOT to a velocity target whose c'/c term diverges near the
    sigma_min endpoints. (Naive bridge velocity loss explodes to ~1e3 because
    the target itself contains the singular coefficient c'/c -> 1/sigma_min^2.)
    Inference is unchanged (uses x_pred + exact step).
    """
    D = dataset.D
    model = FlowMLP(D, hidden_dim, n_layers).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    losses = []
    for epoch in range(n_epochs):
        el, nb = 0.0, 0
        for x_cond, y in dataset.get_batches(batch_size):
            B = x_cond.shape[0]
            t = torch.rand(B, 1, device=device)
            eps = torch.randn_like(y)
            eta = torch.randn_like(y)
            c, _ = bridge_coeffs(t, sigma, sigma_min)            # (B,1)
            c = c.view(B, 1)
            mu_t = (1 - t) * eps + t * y
            z_t = mu_t + c * eta
            y_hat = model(z_t, x_cond, t)
            w = 1.0 / (c ** 2)   # inverse conditional variance; (B,1)
            w = w.clamp(1.0, 200.0)   # cap the 1/sigma_min^2 endpoint blow-up
            loss = (w * (y_hat - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            el += loss.item(); nb += 1
        sched.step()
        avg = el / nb
        losses.append(avg)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    [bridge s={sigma}] epoch {epoch+1:4d}/{n_epochs}  loss={avg:.6f}")
    return model, losses


# ---------------------------------------------------------------------------
# Bridge + condition-noise augmentation (the hypersphere analog of the
# blur/noise context augmentation in xpred_blur_v2). The condition x_cond is
# corrupted as  c_t = (1-t) x_cond + t eps_cond  with condition noise TIED to
# the flow time (coupled: s=t). Combined with the bridge path on the target,
# this is the closest analog to the weather training setup. (An independent
# decoupled-noise variant was explored and dropped — it under-dispersed.)
# ---------------------------------------------------------------------------


def train_bridge_coupled(dataset, sigma=0.5, sigma_min=1e-3,
                         hidden_dim=256, n_layers=5, n_epochs=300,
                         batch_size=512, lr=1e-3, device='cuda'):
    """
    Bridge path on the target + condition noise TIED to the flow time (s=t).
    The model does NOT receive s (same arch as clean bridge); it must infer
    condition quality from c_s itself. At inference the condition is cleaned
    along the path, mirroring the V5-style coupled sampler.
    """
    D = dataset.D
    model = FlowMLP(D, hidden_dim, n_layers).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    losses = []
    for epoch in range(n_epochs):
        el, nb = 0.0, 0
        for x_cond, y in dataset.get_batches(batch_size):
            B = x_cond.shape[0]
            t = torch.rand(B, 1, device=device)
            eps = torch.randn_like(y)
            eta = torch.randn_like(y)
            c, _ = bridge_coeffs(t, sigma, sigma_min)
            c = c.view(B, 1)
            mu_t = (1 - t) * eps + t * y
            z_t = mu_t + c * eta

            eps_cond = torch.randn_like(x_cond)
            c_t = (1 - t) * eps_cond + t * x_cond   # s = t

            y_hat = model(z_t, c_t, t)
            w = 1.0 / (c ** 2)
            w = w.clamp(1.0, 200.0)
            loss = (w * (y_hat - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            el += loss.item(); nb += 1
        sched.step()
        avg = el / nb
        losses.append(avg)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    [bridge+coupled s={sigma}] "
                  f"epoch {epoch+1:4d}/{n_epochs}  loss={avg:.6f}")
    return model, losses


# ============================================================================
# 5. SAMPLING & AUTOREGRESSIVE ROLLOUT
# ============================================================================

@torch.no_grad()
def sample_rectified(model, x_cond, n_ode_steps=50):
    """Forward Euler 0->1 with v = (x_pred - z_t)/(1-t)."""
    B, D = x_cond.shape
    device = x_cond.device
    z = torch.randn(B, D, device=device)
    dt = 1.0 / n_ode_steps
    for i in range(n_ode_steps):
        t_val = i * dt
        t = torch.full((B, 1), t_val, device=device)
        x_pred = model(z, x_cond, t)
        v = (x_pred - z) / max(1 - t_val, 1e-2)
        z = z + v * dt
    return z


@torch.no_grad()
def sample_bridge(model, x_cond, sigma, sigma_min, n_ode_steps=50,
                  integrator='exact'):
    """
    Bridge solver. Convention: t=0 -> noise (eps), t=1 -> data.

    Two integrators:
      - 'exact': closed-form step (recommended). Treating the model prediction
        y as frozen over [t, t+dt], the bridge ODE integrates to
            z(t+dt) = mu_t(t+dt) + (z(t) - mu_t(t)) * c(t+dt)/c(t)
        because dw/ds = (c'/c) w with w = z - mu_t. This absorbs the stiff
        mean-reversion coefficient (c'/c blows up ~1/sigma_min^2 near the
        endpoints) into the O(1) ratio c(t+dt)/c(t), so it is stable for any
        sigma_min and lands exactly on y at t=1. Correct even at high D where
        naive Euler diverges.
      - 'euler': the original forward-Euler step
            v = (y-eps) + (c'/c)(z - mu_t),  z += v dt
        kept for comparison/diagnosis. Diverges at high D / small sigma_min
        because the (c'/c)(z-mu_t) term amplifies per-step error by ~1/sigma_min^2.

    eps is the FIXED noise source sampled once and reused across steps.
    """
    B, D = x_cond.shape
    device = x_cond.device
    eps = torch.randn(B, D, device=device)   # fixed source for this solve
    z = eps.clone()
    dt = 1.0 / n_ode_steps
    for i in range(n_ode_steps):
        t_val = i * dt
        t_next = t_val + dt
        t = torch.full((B, 1), t_val, device=device)
        x_pred = model(z, x_cond, t)
        z = _bridge_step(z, eps, x_pred, t_val, t_next, sigma, sigma_min, integrator)
    return z


def _bridge_step(z, eps, x_pred, t_val, t_next, sigma, sigma_min, integrator):
    """One bridge ODE step over [t_val, t_next] with x_pred frozen.

    'exact'  : closed-form  z(t_next) = mu_next + (z-mu_t) * c_next/c_t
    'euler'  : forward Euler with the stiff (c'/c)(z-mu_t) term.
    Shared by sample_bridge / sample_bridge_coupled.
    """
    mu_t = (1 - t_val) * eps + t_val * x_pred
    if integrator == 'exact':
        c_t, _ = bridge_coeffs(t_val, sigma, sigma_min)
        c_next, _ = bridge_coeffs(t_next, sigma, sigma_min)
        ratio = (c_next / c_t).item()
        mu_next = (1 - t_next) * eps + t_next * x_pred
        return mu_next + (z - mu_t) * ratio
    else:  # euler
        _, cp_over_c = bridge_coeffs(t_val, sigma, sigma_min)
        cp_over_c = cp_over_c.item()
        v = (x_pred - eps) + cp_over_c * (z - mu_t)
        return z + v * (t_next - t_val)


@torch.no_grad()
def sample_bridge_coupled(model, x_cond, sigma, sigma_min, n_ode_steps=50,
                          integrator='exact'):
    """
    Bridge solver with a coupled (s=t) model: condition is cleaned along the
    path, c_t = (1-t) eps_cond + t x_cond, mirroring the V5-style sampler.
    eps_cond is fixed across steps.
    """
    B, D = x_cond.shape
    device = x_cond.device
    eps = torch.randn(B, D, device=device)
    eps_cond_fixed = torch.randn_like(x_cond)
    z = eps.clone()
    dt = 1.0 / n_ode_steps
    for i in range(n_ode_steps):
        t_val = i * dt
        t_next = t_val + dt
        t = torch.full((B, 1), t_val, device=device)
        c_t = (1 - t_val) * eps_cond_fixed + t_val * x_cond
        x_pred = model(z, c_t, t)
        z = _bridge_step(z, eps, x_pred, t_val, t_next, sigma, sigma_min, integrator)
    return z


@torch.no_grad()
def autoregressive_rollout(model, start, n_ar_steps=200, n_ode_steps=50,
                           mode='rectified', sigma=0.5, sigma_min=1e-3,
                           reproject_to_sphere=False):
    traj = [start.cpu()]
    current = start
    for _ in range(n_ar_steps):
        if mode == 'rectified':
            current = sample_rectified(model, current, n_ode_steps)
        elif mode == 'bridge':
            current = sample_bridge(model, current, sigma, sigma_min, n_ode_steps)
        elif mode == 'bridge_coupled':
            current = sample_bridge_coupled(model, current, sigma, sigma_min, n_ode_steps)
        else:
            raise ValueError(mode)
        if reproject_to_sphere:
            current = current / current.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        traj.append(current.cpu())
    return torch.stack(traj, dim=0)


# ============================================================================
# 6. EVALUATION — same metrics as hypersphere_flow_experiment.py
# ============================================================================

def evaluate_rollout(trajectory):
    T, B, D = trajectory.shape
    norms = torch.norm(trajectory, dim=-1)  # (T, B)
    start = trajectory[0]
    angular_disp = []
    for t in range(T):
        cos_sim = torch.nn.functional.cosine_similarity(
            trajectory[t], start, dim=-1).clamp(-1, 1)
        angular_disp.append(torch.acos(cos_sim).mean().item())
    radial_energy = []
    for t in range(T - 1):
        delta = trajectory[t + 1] - trajectory[t]
        x = trajectory[t]
        x_hat = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        radial_comp = (delta * x_hat).sum(dim=-1)
        radial_energy.append((radial_comp ** 2).mean().item())
    return dict(
        norm_mean=norms.mean(dim=-1).tolist(),
        norm_std=norms.std(dim=-1).tolist(),
        norm_error=(norms - 1).abs().mean(dim=-1).tolist(),
        norm_min=norms.min(dim=-1).values.tolist(),
        norm_max=norms.max(dim=-1).values.tolist(),
        angular_displacement=angular_disp,
        radial_energy=radial_energy,
    )


# ============================================================================
# 7. PLOTTING
# ============================================================================

def plot_path_illustration(sigmas, save_path):
    """Show variance c_t^2 along the path for RF vs bridge."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t = np.linspace(0, 1, 200)
    ax.plot(t, np.zeros_like(t), 'k--', lw=2, label='rectified flow (var=0)')
    for s in sigmas:
        var = s ** 2 * t * (1 - t) + 1e-3
        ax.plot(t, var, lw=2, label=f'bridge sigma={s}')
    ax.axvline(0, color='gray', ls=':', alpha=0.5)
    ax.axvline(1, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('t  (0 = noise, 1 = data)')
    ax.set_ylabel("path variance  $c_t^2$")
    ax.set_title('Brownian-bridge variance is minimal at both endpoints\n'
                 '(clean landing at data end -> less AR drift injection)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_results(all_results, cfg, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Hypersphere Random Walk — Bridge vs Rectified Flow (AR Stability)',
                 fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    for key, res in all_results.items():
        steps = np.arange(len(res['norm_mean']))
        ax.plot(steps, res['norm_mean'], label=key, linewidth=1.2)
    ax.axhline(1.0, color='k', ls='--', alpha=0.5, label='target ||x||=1')
    ax.set_xlabel('AR Step'); ax.set_ylabel('Mean ||x||')
    ax.set_title('Norm Evolution (should stay = 1)')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for key, res in all_results.items():
        steps = np.arange(len(res['norm_mean']))
        ax.fill_between(steps, res['norm_min'], res['norm_max'], alpha=0.15)
        ax.plot(steps, res['norm_mean'], linewidth=1.2, label=key)
    ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('AR Step'); ax.set_ylabel('||x||')
    ax.set_title('Norm Envelope (min/max)')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for key, res in all_results.items():
        ax.plot(res['norm_error'], label=key, linewidth=1.2)
    ax.set_xlabel('AR Step'); ax.set_ylabel('Mean | ||x|| - 1 |')
    ax.set_title('Norm Error (off-manifold drift)')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for key, res in all_results.items():
        ax.plot(res['angular_displacement'], label=key, linewidth=1.2)
    ax.set_xlabel('AR Step'); ax.set_ylabel('Mean Angular Distance from Start (rad)')
    ax.set_title('Angular Displacement')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for key, res in all_results.items():
        ax.plot(res['radial_energy'], label=key, linewidth=1.2)
    ax.set_xlabel('AR Step'); ax.set_ylabel('Radial Step Energy')
    ax.set_title('Radial Component (off-manifold drift per step)')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    for key, res in all_results.items():
        if res.get('losses'):
            ax.plot(res['losses'], label=key, linewidth=0.8, alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_drift_vs_sigma(all_results, save_path):
    """Final norm error / radial energy vs bridge sigma."""
    sigmas, errs, radials, angles = [], [], [], []
    rf_err = rf_radial = rf_angle = None
    for key, res in all_results.items():
        if res.get('mode') == 'rectified':
            rf_err = res['norm_error'][-1]
            rf_radial = res['radial_energy'][-1]
            rf_angle = res['angular_displacement'][-1]
        elif res.get('mode') == 'bridge':
            sigmas.append(res['sigma'])
            errs.append(res['norm_error'][-1])
            radials.append(res['radial_energy'][-1])
            angles.append(res['angular_displacement'][-1])

    if not sigmas:
        return
    order = np.argsort(sigmas)
    sigmas = np.array(sigmas)[order]
    errs = np.array(errs)[order]
    radials = np.array(radials)[order]
    angles = np.array(angles)[order]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle('AR stability vs bridge sigma (rectified flow shown as flat line)',
                 fontsize=12, fontweight='bold')
    for ax, vals, rf_val, ylabel in [
        (axes[0], errs, rf_err, 'Final | ||x|| - 1 |'),
        (axes[1], radials, rf_radial, 'Final radial energy'),
        (axes[2], angles, rf_angle, 'Final angular disp (rad)'),
    ]:
        ax.plot(sigmas, vals, 'o-', lw=1.5, ms=6, label='bridge')
        if rf_val is not None:
            ax.axhline(rf_val, color='r', ls='--', lw=1.5, label='rectified flow')
        ax.set_xlabel('bridge sigma'); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


# ============================================================================
# 8. MAIN
# ============================================================================

def run_experiment(D, speed, sigmas, cfg, device, out):
    print(f"\n{'='*70}\n  D={D} (S^{D-1})  speed={speed}\n{'='*70}")

    dataset = HypersphereWalkDataset(D=D, speed=speed, n_samples=cfg['n_samples'],
                                     seed=42, device=device)
    print(f"  Dataset: ||x||={dataset.x.norm(dim=-1).mean():.6f}  "
          f"||y||={dataset.y.norm(dim=-1).mean():.6f}")

    # --- Train rectified baseline ---
    print("\n  Training [rectified]...")
    t0 = time.time()
    m_rf, l_rf = train_rectified(dataset, cfg['hidden_dim'], cfg['n_layers'],
                                 cfg['n_epochs'], cfg['batch_size'], 1e-3, device)
    m_rf.eval()
    print(f"    done {time.time()-t0:.1f}s")

    # --- Train one bridge model per sigma ---
    bridge_models = {}
    for s in sigmas:
        print(f"\n  Training [bridge sigma={s}]...")
        t0 = time.time()
        m, l = train_bridge(dataset, sigma=s, sigma_min=1e-3,
                            hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
                            n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
                            lr=1e-3, device=device)
        m.eval()
        bridge_models[s] = (m, l)
        print(f"    done {time.time()-t0:.1f}s")

    # --- Bridge + condition-noise augmentation (coupled only) ---
    # Pick the middle sigma for the aug comparison to limit compute.
    aug_sigma = sigmas[len(sigmas) // 2] if sigmas else 0.5
    print(f"\n  Training [bridge+coupled sigma={aug_sigma}]...")
    t0 = time.time()
    m_bcou, l_bcou = train_bridge_coupled(
        dataset, sigma=aug_sigma, sigma_min=1e-3,
        hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'], lr=1e-3, device=device)
    m_bcou.eval()
    print(f"    done {time.time()-t0:.1f}s")

    # --- Starting points ---
    rng = np.random.RandomState(1042)
    start_np = rng.randn(cfg['n_eval'], D).astype(np.float32)
    start_np = start_np / np.linalg.norm(start_np, axis=1, keepdims=True)
    start = torch.tensor(start_np, dtype=torch.float32, device=device)

    results, trajs = {}, {}

    def do_rollout(key, model, mode, losses=None, sigma=None, reproject=False):
        tag = " [reproj]" if reproject else ""
        print(f"  AR rollout [{key}{tag}] ({cfg['n_ar_steps']} steps)...")
        t0 = time.time()
        traj = autoregressive_rollout(
            model, start, n_ar_steps=cfg['n_ar_steps'],
            n_ode_steps=cfg['n_ode_steps'], mode=mode,
            sigma=sigma if sigma is not None else 0.5,
            sigma_min=1e-3,
            reproject_to_sphere=reproject,
        )
        print(f"    {time.time()-t0:.1f}s")
        res = evaluate_rollout(traj)
        res['losses'] = losses
        res['mode'] = mode
        if sigma is not None:
            res['sigma'] = sigma
        print(f"    final ||x||={res['norm_mean'][-1]:.4f}  "
              f"|err|={res['norm_error'][-1]:.4f}  "
              f"angle={res['angular_displacement'][-1]:.3f}")
        results[key] = res
        trajs[key] = traj

    # 1) Rectified baseline
    do_rollout(f'D={D} spd={speed} rectified', m_rf, 'rectified', losses=l_rf)
    # 2) Rectified + reproject oracle
    do_rollout(f'D={D} spd={speed} rectified+reproj', m_rf, 'rectified',
               losses=l_rf, reproject=True)
    # 3) Bridge at each sigma
    for s in sigmas:
        m_b, l_b = bridge_models[s]
        do_rollout(f'D={D} spd={speed} bridge/s={s}', m_b, 'bridge',
                   losses=l_b, sigma=s)
    # 4) Bridge + reproject (best-case manifold adherence)
    if sigmas:
        s0 = sigmas[len(sigmas) // 2]
        m_b, _ = bridge_models[s0]
        do_rollout(f'D={D} spd={speed} bridge/s={s0}+reproj', m_b, 'bridge',
                   sigma=s0, reproject=True)
    # 5) Bridge + coupled condition-noise augmentation (condition cleaned
    #    along the path, s=t). This is the promising config at low speed.
    do_rollout(f'D={D} spd={speed} bridge+cou/s={aug_sigma}',
               m_bcou, 'bridge_coupled', losses=l_bcou, sigma=aug_sigma)

    return results, trajs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--dims', type=int, nargs='+', default=None)
    parser.add_argument('--speeds', type=float, nargs='+', default=None)
    parser.add_argument('--sigmas', type=float, nargs='+', default=None,
                        help='bridge sigma values to sweep')
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--n_ar_steps', type=int, default=None)
    parser.add_argument('--n_ode_steps', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--outdir', type=str, default='./results_hypersphere_bridge')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    if args.quick:
        cfg = dict(n_samples=10_000, n_epochs=80, batch_size=512,
                   n_ar_steps=50, n_ode_steps=20, n_eval=128,
                   hidden_dim=args.hidden_dim, n_layers=args.n_layers)
        dims = args.dims or [16]
        speeds = args.speeds or [0.2]
        sigmas = args.sigmas or [0.3, 0.5]
    else:
        cfg = dict(n_samples=300_000, n_epochs=args.n_epochs or 400, batch_size=2048,
                   n_ar_steps=args.n_ar_steps or 200,
                   n_ode_steps=args.n_ode_steps or 50, n_eval=256,
                   hidden_dim=args.hidden_dim, n_layers=args.n_layers)
        dims = args.dims or [16, 32, 64]
        speeds = args.speeds or [0.1, 0.3]
        sigmas = args.sigmas or [0.3, 0.5, 1.0]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    plot_path_illustration(sigmas, str(out / 'path_variance.png'))

    all_results, all_traj = {}, {}
    t0 = time.time()
    for D in dims:
        for speed in speeds:
            res, trajs = run_experiment(D, speed, sigmas, cfg, device, out)
            all_results.update(res)
            all_traj.update(trajs)
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")

    plot_results(all_results, cfg, str(out / 'drift_analysis.png'))
    plot_drift_vs_sigma(all_results, str(out / 'drift_vs_sigma.png'))

    # Summary table
    print(f"\n{'='*110}")
    print(f"SUMMARY — AR Stability after {cfg['n_ar_steps']} steps")
    print(f"{'='*110}")
    print(f"{'Config':<48} {'||x||':>8} {'|err|':>8} {'Angle':>8} {'RadE':>10} {'Norm std':>10}")
    print('-' * 110)
    for label, res in all_results.items():
        print(f"{label:<48} {res['norm_mean'][-1]:>8.4f} "
              f"{res['norm_error'][-1]:>8.4f} "
              f"{res['angular_displacement'][-1]:>8.3f} "
              f"{res['radial_energy'][-1]:>10.6f} "
              f"{res['norm_std'][-1]:>10.4f}")

    summary = {
        'config': cfg, 'dims': dims, 'speeds': speeds, 'sigmas': sigmas,
        'results': {
            label: dict(mode=res['mode'],
                        sigma=res.get('sigma'),
                        final_norm=res['norm_mean'][-1],
                        final_norm_error=res['norm_error'][-1],
                        final_angular_disp=res['angular_displacement'][-1],
                        final_radial_energy=res['radial_energy'][-1],
                        norm_trajectory=res['norm_mean'],
                        radial_trajectory=res['radial_energy'])
            for label, res in all_results.items()
        }
    }
    with open(str(out / 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll outputs in {out.resolve()}")


if __name__ == '__main__':
    main()
