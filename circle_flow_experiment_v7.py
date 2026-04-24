"""
Rectified Flow on Circle Transitions — Decoupled Condition/Output Noise Schedules (V7)

Core idea: Instead of tying the condition noise to the output denoising time (V5),
use TWO independent time parameters:

  t  — output noise schedule (standard RF: z_t = (1-t)·ε + t·y)
  s  — condition noise schedule (c_s = (1-s)·x_cond + s·ε_cond)

  s=0 → clean condition, s=1 → pure noise condition.

Training:   t ~ U(0,1) and s ~ U(0,1) are sampled independently.
            Model learns f(z_t, c_s, t, s) → velocity.
            This gives the model exposure to ALL combinations of output noise
            and condition quality, making it maximally robust to condition drift.

Inference:  We can freely choose s at inference time:
  s=0       — clean condition (like baseline, but model is robust from training)
  s=ε       — slightly noisy condition (may match AR drift level)
  s linearly increasing with AR step — adapt to compounding drift
  s=const   — fixed noise level throughout rollout

Modes compared:
  baseline         — standard RF (s always 0 at train & inference)
  decoupled/s=0    — trained with decoupled (t,s), infer with clean condition
  decoupled/s=0.05 — trained with decoupled (t,s), infer with small condition noise
  decoupled/s=0.1  — trained with decoupled (t,s), infer with moderate condition noise
  decoupled/s=ramp — trained with decoupled (t,s), s increases linearly over AR steps
  coupled          — V5-style: s=t during training, s=t during inference (for comparison)

Usage:
    python circle_flow_experiment_v7.py                   # full run
    python circle_flow_experiment_v7.py --quick           # fast sanity check
    python circle_flow_experiment_v7.py --infer_s 0 0.05 0.1 0.2
    python circle_flow_experiment_v7.py --dims 128 --delta_rs 0.1 0.4
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
# 1. DATA
# ============================================================================

class CircleTransitionDataset:
    """
    Points on an annulus in R², projected to D dimensions.
    Radius ~ r + N(0, delta_r²), transitions are pure rotations.
    """

    def __init__(self, r=1.0, delta_r=0.1, theta_mean=0.3, delta_theta=0.2,
                 D=2, n_samples=50_000, seed=42, device='cuda'):
        self.r = r
        self.delta_r = delta_r
        self.theta_mean = theta_mean
        self.delta_theta = delta_theta
        self.D = D
        self.d = 2
        self.device = device

        rng = np.random.RandomState(seed)

        if D > 2:
            A = rng.randn(D, 2)
            Q, _ = np.linalg.qr(A)
            self.P = torch.tensor(Q, dtype=torch.float32, device=device)
        else:
            self.P = torch.eye(2, dtype=torch.float32, device=device)

        angles = rng.uniform(0, 2 * np.pi, n_samples).astype(np.float32)
        radii = (r + rng.randn(n_samples) * delta_r).astype(np.float32)

        x_2d = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

        d_angles = (theta_mean + rng.uniform(-delta_theta / 2, delta_theta / 2, n_samples)).astype(np.float32)
        new_angles = angles + d_angles
        y_2d = np.stack([radii * np.cos(new_angles), radii * np.sin(new_angles)], axis=1)

        x_2d = torch.tensor(x_2d, dtype=torch.float32, device=device)
        y_2d = torch.tensor(y_2d, dtype=torch.float32, device=device)

        self.x = x_2d @ self.P.T
        self.y = y_2d @ self.P.T
        self.n_samples = n_samples

    def get_batches(self, batch_size, shuffle=True):
        idx = torch.randperm(self.n_samples, device=self.device) if shuffle else torch.arange(self.n_samples, device=self.device)
        for i in range(0, self.n_samples, batch_size):
            bi = idx[i:i + batch_size]
            yield self.x[bi], self.y[bi]

    def project_to_2d(self, points_D):
        return points_D @ self.P

    def compute_radius(self, points_D):
        return torch.norm(self.project_to_2d(points_D), dim=-1)


# ============================================================================
# 2. MODEL
# ============================================================================

class FlowMLPDecoupled(nn.Module):
    """
    Conditional rectified flow with decoupled time parameters:
    [z_t, c_s, t, s] → velocity prediction.

    Input: 2*D + 2 (z_t, c_s, scalar t, scalar s)
    Output: D-dim velocity
    """

    def __init__(self, D, hidden_dim=256, n_layers=5):
        super().__init__()
        self.D = D
        layers = []
        in_dim = 2 * D + 2  # z_t, c_s, t, s
        for i in range(n_layers):
            out_dim = D if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, c_s, t, s):
        return self.net(torch.cat([z_t, c_s, t, s], dim=-1))

    def get_velocity(self, z_t, c_s, t, s):
        x_pred = self.forward(z_t, c_s, t, s)
        return (x_pred - z_t) / (1 - t).clamp(min=0.01)


class FlowMLPBaseline(nn.Module):
    """
    Standard conditional rectified flow (no s parameter):
    [z_t, x_cond, t] → velocity prediction.
    Same architecture as V5/V6 baseline for fair comparison.
    """

    def __init__(self, D, hidden_dim=256, n_layers=5):
        super().__init__()
        self.D = D
        layers = []
        in_dim = 2 * D + 1
        for i in range(n_layers):
            out_dim = D if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, x_cond, t):
        return self.net(torch.cat([z_t, x_cond, t], dim=-1))

    def get_velocity(self, z_t, x_cond, t):
        x_pred = self.forward(z_t, x_cond, t)
        return (x_pred - z_t) / (1 - t).clamp(min=0.01)


# ============================================================================
# 3. TRAINING
# ============================================================================

def train_decoupled(dataset, s_max=1.0, hidden_dim=256, n_layers=5,
                    n_epochs=300, batch_size=512, lr=1e-3, device='cuda'):
    """
    Train with decoupled (t, s) schedules.
    t ~ U(0,1) for output, s ~ U(0, s_max) for condition noise.
    s_max < 1.0 limits the max condition noise (softer training).
    """
    D = dataset.D
    model = FlowMLPDecoupled(D, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    label = f'decoupled(s_max={s_max})'
    losses = []
    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for x_cond, y_target in dataset.get_batches(batch_size):
            B = x_cond.shape[0]
            t = torch.rand(B, 1, device=device)
            s = torch.rand(B, 1, device=device) * s_max

            # Output interpolation
            eps = torch.randn_like(y_target)
            z_t = (1 - t) * eps + t * y_target
            v_target = y_target - eps

            # Condition interpolation (independent s)
            eps_cond = torch.randn_like(x_cond)
            c_s = (1 - s) * x_cond + s * eps_cond

            v_pred = model.get_velocity(z_t, c_s, t, s)
            loss = ((v_pred - v_target) ** 2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / n_batches
        losses.append(avg)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    [{label}] epoch {epoch+1:4d}/{n_epochs}  loss={avg:.6f}")

    return model, losses


def train_coupled(dataset, hidden_dim=256, n_layers=5,
                  n_epochs=300, batch_size=512, lr=1e-3, device='cuda'):
    """
    Train V5-style: condition noise tied to output time (s = t).
    Uses the baseline architecture (no s input).
    """
    D = dataset.D
    model = FlowMLPBaseline(D, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    label = 'coupled(V5-style)'
    losses = []
    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for x_cond, y_target in dataset.get_batches(batch_size):
            B = x_cond.shape[0]
            t = torch.rand(B, 1, device=device)

            # Output interpolation
            eps = torch.randn_like(y_target)
            z_t = (1 - t) * eps + t * y_target
            v_target = y_target - eps

            # Condition noise tied to t (V5 style)
            eps_cond = torch.randn_like(x_cond)
            c_t = (1 - t) * eps_cond + t * x_cond

            v_pred = model.get_velocity(z_t, c_t, t)
            loss = ((v_pred - v_target) ** 2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / n_batches
        losses.append(avg)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    [{label}] epoch {epoch+1:4d}/{n_epochs}  loss={avg:.6f}")

    return model, losses


def train_baseline(dataset, hidden_dim=256, n_layers=5,
                   n_epochs=300, batch_size=512, lr=1e-3, device='cuda'):
    """
    Standard baseline: clean condition, no s parameter.
    """
    D = dataset.D
    model = FlowMLPBaseline(D, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    label = 'baseline'
    losses = []
    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for x_cond, y_target in dataset.get_batches(batch_size):
            B = x_cond.shape[0]
            t = torch.rand(B, 1, device=device)
            eps = torch.randn_like(y_target)
            z_t = (1 - t) * eps + t * y_target
            v_target = y_target - eps

            v_pred = model.get_velocity(z_t, x_cond, t)
            loss = ((v_pred - v_target) ** 2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / n_batches
        losses.append(avg)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    [{label}] epoch {epoch+1:4d}/{n_epochs}  loss={avg:.6f}")

    return model, losses


# ============================================================================
# 4. SAMPLING & AUTOREGRESSIVE ROLLOUT
# ============================================================================

@torch.no_grad()
def sample_next_decoupled(model, x_cond, n_ode_steps=50,
                          infer_s=0.0, eps_cond_fixed=None):
    """
    Sample with the decoupled model. Condition noise level s is set externally.

    infer_s:      condition noise level for this step (scalar)
    eps_cond_fixed: if provided, reuse this noise vector (consistent within trajectory)
    """
    B, D = x_cond.shape
    device = x_cond.device
    z = torch.randn(B, D, device=device)
    dt = 1.0 / n_ode_steps

    # Fix eps_cond once per sample so condition is consistent across ODE steps
    if eps_cond_fixed is None:
        eps_cond_fixed = torch.randn_like(x_cond)

    s_tensor = torch.full((B, 1), infer_s, device=device)
    c_s = (1 - s_tensor) * x_cond + s_tensor * eps_cond_fixed

    for i in range(n_ode_steps):
        t_val = i * dt
        t = torch.full((B, 1), t_val, device=device)

        v = model.get_velocity(z, c_s, t, s_tensor)
        z = z + v * dt

    return z


@torch.no_grad()
def sample_next_coupled(model, x_cond, n_ode_steps=50,
                        infer_noisy=True, eps_cond_fixed=None):
    """
    Sample with the coupled (V5-style) model.
    infer_noisy=True:  condition noise follows t (noisy/noisy)
    infer_noisy=False: clean condition (noisy/clean)
    """
    B, D = x_cond.shape
    device = x_cond.device
    z = torch.randn(B, D, device=device)
    dt = 1.0 / n_ode_steps

    if eps_cond_fixed is None:
        eps_cond_fixed = torch.randn_like(x_cond)

    for i in range(n_ode_steps):
        t_val = i * dt
        t = torch.full((B, 1), t_val, device=device)

        if infer_noisy:
            c_t = (1 - t) * eps_cond_fixed + t * x_cond
        else:
            c_t = x_cond

        v = model.get_velocity(z, c_t, t)
        z = z + v * dt

    return z


@torch.no_grad()
def sample_next_baseline(model, x_cond, n_ode_steps=50):
    """Standard baseline sampling with clean condition."""
    B, D = x_cond.shape
    device = x_cond.device
    z = torch.randn(B, D, device=device)
    dt = 1.0 / n_ode_steps

    for i in range(n_ode_steps):
        t_val = i * dt
        t = torch.full((B, 1), t_val, device=device)
        v = model.get_velocity(z, x_cond, t)
        z = z + v * dt

    return z


@torch.no_grad()
def autoregressive_rollout(model, start, n_ar_steps=200, n_ode_steps=50,
                           mode='baseline', infer_s=0.0,
                           s_ramp_max=0.0, coupled_noisy=True):
    """
    Run an autoregressive rollout.

    mode:         'baseline', 'decoupled', 'coupled'
    infer_s:      fixed s value for decoupled inference
    s_ramp_max:   if >0, linearly ramp s from 0 to s_ramp_max over AR steps
    coupled_noisy: for coupled mode, whether to use noisy inference
    """
    traj = [start.cpu()]
    current = start
    eps_cond = None  # Will be set per step

    for step in range(n_ar_steps):
        if mode == 'baseline':
            current = sample_next_baseline(model, current, n_ode_steps)
        elif mode == 'decoupled':
            # Determine s for this AR step
            if s_ramp_max > 0:
                s = s_ramp_max * (step / max(n_ar_steps - 1, 1))
            else:
                s = infer_s
            # Fix eps_cond per ODE solve, not across AR steps
            current = sample_next_decoupled(model, current, n_ode_steps,
                                            infer_s=s)
        elif mode == 'coupled':
            current = sample_next_coupled(model, current, n_ode_steps,
                                          infer_noisy=coupled_noisy)
        traj.append(current.cpu())

    return torch.stack(traj, dim=0)


# ============================================================================
# 5. EVALUATION
# ============================================================================

def evaluate_rollout(trajectory, dataset, target_r=1.0):
    T, B, D = trajectory.shape
    P_cpu = dataset.P.cpu()

    radius_mean, radius_std, radius_err = [], [], []
    radius_median, radius_max, radius_min = [], [], []
    off_manifold = []

    for t in range(T):
        pts = trajectory[t]
        pts_2d = pts @ P_cpu
        r = torch.norm(pts_2d, dim=-1)
        radius_mean.append(r.mean().item())
        radius_std.append(r.std().item())
        radius_err.append((r - target_r).abs().mean().item())
        radius_median.append(r.median().item())
        radius_max.append(r.max().item())
        radius_min.append(r.min().item())

        if D > 2:
            recon = pts_2d @ P_cpu.T
            off_manifold.append(((pts - recon) ** 2).sum(-1).mean().item())
        else:
            off_manifold.append(0.0)

    return dict(
        radius_mean=radius_mean, radius_std=radius_std, radius_error=radius_err,
        radius_median=radius_median, radius_max=radius_max, radius_min=radius_min,
        off_manifold_energy=off_manifold,
    )


# ============================================================================
# 6. PLOTTING
# ============================================================================

def plot_schedule_illustration(save_path):
    """Illustrate the decoupled vs coupled noise schedules."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Decoupled: 2D heatmap of (t, s) ---
    ax = axes[0]
    ts = np.linspace(0, 1, 100)
    ss = np.linspace(0, 1, 100)
    T_grid, S_grid = np.meshgrid(ts, ss)
    # Color = output noise level (1-t) for visual reference
    ax.pcolormesh(T_grid, S_grid, 1 - T_grid, cmap='Blues', alpha=0.5, shading='auto')
    ax.set_xlabel('t (output time)')
    ax.set_ylabel('s (condition noise)')
    ax.set_title('Decoupled: (t, s) sampled independently\n(training covers entire square)')
    # Diagonal = coupled
    ax.plot(ts, ts, 'r--', linewidth=2, label='coupled: s=t (V5)')
    ax.plot(ts, np.zeros_like(ts), 'g-', linewidth=2, label='baseline: s=0')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    # --- Condition quality vs s ---
    ax = axes[1]
    s_vals = np.linspace(0, 1, 200)
    ax.plot(s_vals, 1 - s_vals, 'b-', linewidth=2, label='clean weight: (1-s)')
    ax.fill_between(s_vals, 0, 1 - s_vals, alpha=0.2, color='blue', label='x_cond contribution')
    ax.fill_between(s_vals, 1 - s_vals, 1, alpha=0.2, color='red', label='noise contribution')
    ax.set_xlabel('s (condition noise level)')
    ax.set_ylabel('Weight')
    ax.set_title('Condition: c_s = (1-s)·x_cond + s·ε')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Inference s strategies ---
    ax = axes[2]
    steps = np.arange(200)
    for s_val, color, ls in [(0, 'green', '-'), (0.05, 'blue', '-'),
                              (0.1, 'orange', '-'), (0.2, 'red', '-')]:
        ax.axhline(s_val, color=color, ls=ls, alpha=0.6, label=f's={s_val}')
    # Ramp strategies
    for ramp_max, color in [(0.1, 'purple'), (0.3, 'brown')]:
        ax.plot(steps, ramp_max * steps / steps[-1], color=color, ls='--',
                linewidth=2, label=f'ramp 0→{ramp_max}')
    ax.set_xlabel('AR Step')
    ax.set_ylabel('s (condition noise level)')
    ax.set_title('Inference strategies for s')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_results(all_results, cfg, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Rectified Flow — Decoupled (t,s) vs Coupled vs Baseline',
                 fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    for key, res in all_results.items():
        s = np.arange(len(res['radius_mean']))
        m = np.array(res['radius_mean'])
        lo = np.array(res['radius_min'])
        hi = np.array(res['radius_max'])
        ax.plot(s, m, label=key, linewidth=1.5)
        ax.fill_between(s, lo, hi, alpha=0.06)
    ax.axhline(1.0, color='k', ls='--', alpha=0.4, label='target r=1')
    ax.set_xlabel('AR Step'); ax.set_ylabel('Radius')
    ax.set_title('Radius Evolution (mean, min/max envelope)')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for key, res in all_results.items():
        ax.plot(res['radius_std'], label=key, linewidth=1.5)
    ax.set_xlabel('AR Step'); ax.set_ylabel('Std(radius)')
    ax.set_title('Radius Spread Over Time')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for key, res in all_results.items():
        drift = np.array(res['radius_mean']) - 1.0
        ax.plot(drift, label=key, linewidth=1.5)
    ax.axhline(0, color='k', ls='--', alpha=0.4)
    ax.set_xlabel('AR Step'); ax.set_ylabel('mean(r) - 1')
    ax.set_title('Signed Radius Drift (>0 = spiraling out)')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for key, res in all_results.items():
        if 'losses' in res:
            ax.plot(res['losses'], label=key, linewidth=0.8, alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_trajectories(all_traj, all_ds, save_path):
    n = len(all_traj)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)

    for idx, (key, traj) in enumerate(all_traj.items()):
        ax = axes[idx // cols][idx % cols]
        ds = all_ds[key]
        T, B, D = traj.shape
        P_cpu = ds.P.cpu()

        theta = np.linspace(0, 2 * np.pi, 200)
        for rr, ls, alpha_v in [
            (ds.r, '-', 0.4),
            (ds.r - ds.delta_r, '--', 0.2), (ds.r + ds.delta_r, '--', 0.2),
            (ds.r - 2*ds.delta_r, ':', 0.1), (ds.r + 2*ds.delta_r, ':', 0.1),
        ]:
            if rr > 0:
                ax.plot(rr * np.cos(theta), rr * np.sin(theta), 'k',
                        ls=ls, alpha=alpha_v, linewidth=0.8)

        pts_2d = (traj.reshape(-1, D) @ P_cpu).reshape(T, B, 2)
        colors = plt.cm.viridis(np.linspace(0, 1, T))

        for i in range(min(10, B)):
            for t in range(T - 1):
                ax.plot(
                    [pts_2d[t, i, 0], pts_2d[t + 1, i, 0]],
                    [pts_2d[t, i, 1], pts_2d[t + 1, i, 1]],
                    color=colors[t], alpha=0.4, linewidth=0.4,
                )

        ax.scatter(pts_2d[0, :, 0], pts_2d[0, :, 1], c='green', s=3, alpha=0.4, label='start', zorder=5)
        ax.scatter(pts_2d[-1, :, 0], pts_2d[-1, :, 1], c='red', s=3, alpha=0.4, label=f'step {T-1}', zorder=5)

        ax.set_title(key, fontsize=8)
        ax.set_aspect('equal')
        lim = max(2.0, ds.r + 3 * ds.delta_r + 0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.legend(fontsize=6, loc='upper right'); ax.grid(True, alpha=0.2)

    for idx in range(len(all_traj), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle('AR Trajectories (2D projection) — green=start, red=end',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_radius_histograms(all_traj, all_ds, save_path):
    n = len(all_traj)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (key, traj) in enumerate(all_traj.items()):
        ax = axes[idx // cols][idx % cols]
        ds = all_ds[key]
        T, B, D = traj.shape
        P_cpu = ds.P.cpu()

        steps_to_show = [0, T // 4, T // 2, 3 * T // 4, T - 1]
        colors_h = plt.cm.plasma(np.linspace(0.1, 0.9, len(steps_to_show)))

        all_r = []
        for step in steps_to_show:
            pts_2d = traj[step] @ P_cpu
            all_r.append(torch.norm(pts_2d, dim=-1).numpy())
        r_all = np.concatenate(all_r)
        r_min, r_max = r_all.min(), r_all.max()
        if r_max - r_min < 1e-6:
            r_min, r_max = r_min - 0.5, r_max + 0.5
        bins = np.linspace(r_min, r_max, 31)

        for si, (step, r) in enumerate(zip(steps_to_show, all_r)):
            ax.hist(r, bins=bins, alpha=0.4, color=colors_h[si],
                    label=f'step {step}', density=True)

        ax.axvline(ds.r, color='k', ls='--', alpha=0.5, label='r')
        if ds.delta_r > 0:
            ax.axvline(ds.r - ds.delta_r, color='gray', ls=':', alpha=0.3, label='±1σ')
            ax.axvline(ds.r + ds.delta_r, color='gray', ls=':', alpha=0.3)
            ax.axvline(ds.r - 2*ds.delta_r, color='gray', ls=':', alpha=0.15, label='±2σ')
            ax.axvline(ds.r + 2*ds.delta_r, color='gray', ls=':', alpha=0.15)
        ax.set_title(key, fontsize=8)
        ax.set_xlabel('Radius'); ax.set_ylabel('Density')
        ax.legend(fontsize=5); ax.grid(True, alpha=0.2)

    for idx in range(len(all_traj), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle('Radius Distribution at Various AR Steps', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_drift_vs_s(all_results, n_ar_steps, save_path):
    """Final drift vs inference s for the decoupled model."""
    by_mode = {}
    for key, res in all_results.items():
        mode = res.get('mode', '?')
        s_val = res.get('infer_s', None)
        drift = res['radius_mean'][-1] - 1.0
        error = res['radius_error'][-1]
        if mode not in by_mode:
            by_mode[mode] = {'s_vals': [], 'drifts': [], 'errors': []}
        if s_val is not None:
            by_mode[mode]['s_vals'].append(s_val)
            by_mode[mode]['drifts'].append(drift)
            by_mode[mode]['errors'].append(error)

    if not by_mode:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f'Drift vs Inference Condition Noise (s) after {n_ar_steps} AR steps',
                 fontsize=12, fontweight='bold')

    for ax, metric, ylabel in [
        (axes[0], 'drifts', 'Signed drift  mean(r) - 1'),
        (axes[1], 'errors', '|mean(r) - 1|'),
    ]:
        for mode, data in by_mode.items():
            if not data['s_vals']:
                continue
            s_arr = np.array(data['s_vals'])
            val_arr = np.array(data[metric])
            order = np.argsort(s_arr)
            ax.plot(s_arr[order], val_arr[order], 'o-', label=mode,
                    linewidth=1.5, markersize=5)
        ax.axhline(0, color='k', ls=':', alpha=0.3)
        ax.set_xlabel('Inference s (condition noise level)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


# ============================================================================
# 7. MAIN
# ============================================================================

def run_experiment(D, delta_r, infer_s_values, s_ramp_values, cfg, device, out: Path):
    print(f"\n{'='*70}")
    print(f"  D={D}  delta_r={delta_r}")
    print(f"{'='*70}")

    dataset = CircleTransitionDataset(
        r=1.0, delta_r=delta_r,
        theta_mean=0.3, delta_theta=0.2,
        D=D, n_samples=cfg['n_samples'], seed=42, device=device,
    )

    # --- Train models ---
    # 1) Baseline (clean condition, no s)
    print(f"\n  Training [baseline] model...")
    t0 = time.time()
    model_baseline, losses_baseline = train_baseline(
        dataset, hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
        lr=1e-3, device=device,
    )
    model_baseline.eval()
    print(f"    Training: {time.time()-t0:.1f}s")

    # 2) Coupled (V5-style: s=t)
    print(f"\n  Training [coupled/V5-style] model...")
    t0 = time.time()
    model_coupled, losses_coupled = train_coupled(
        dataset, hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
        lr=1e-3, device=device,
    )
    model_coupled.eval()
    print(f"    Training: {time.time()-t0:.1f}s")

    # 3) Decoupled (independent t and s)
    print(f"\n  Training [decoupled] model...")
    t0 = time.time()
    model_decoupled, losses_decoupled = train_decoupled(
        dataset, s_max=1.0,
        hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
        lr=1e-3, device=device,
    )
    model_decoupled.eval()
    print(f"    Training: {time.time()-t0:.1f}s")

    # --- Starting points ---
    rng = np.random.RandomState(1042)
    angles = rng.uniform(0, 2 * np.pi, cfg['n_eval']).astype(np.float32)
    start_2d = torch.tensor(
        np.stack([np.cos(angles), np.sin(angles)], axis=1),
        dtype=torch.float32, device=device,
    )
    start_D = start_2d @ dataset.P.T

    results, trajs, datasets = {}, {}, {}

    def do_rollout(key, model, mode, infer_s=0.0, s_ramp_max=0.0,
                   coupled_noisy=True, losses=None):
        print(f"  AR rollout [{key}] ({cfg['n_ar_steps']} steps)...")
        t0r = time.time()
        traj = autoregressive_rollout(
            model, start_D,
            n_ar_steps=cfg['n_ar_steps'],
            n_ode_steps=cfg['n_ode_steps'],
            mode=mode, infer_s=infer_s,
            s_ramp_max=s_ramp_max,
            coupled_noisy=coupled_noisy,
        )
        print(f"    Rollout: {time.time()-t0r:.1f}s")

        res = evaluate_rollout(traj, dataset)
        res['losses'] = losses
        res['mode'] = mode
        res['infer_s'] = infer_s if mode == 'decoupled' else None
        print(f"    Final: r_mean={res['radius_mean'][-1]:.4f}  "
              f"r_std={res['radius_std'][-1]:.4f}  "
              f"drift={res['radius_mean'][-1]-1.0:+.4f}")

        results[key] = res
        trajs[key] = traj
        datasets[key] = dataset

    # --- Baseline ---
    do_rollout(f'D={D} dr={delta_r:.2f} baseline',
               model_baseline, 'baseline', losses=losses_baseline)

    # --- Coupled V5-style: noisy/noisy ---
    do_rollout(f'D={D} dr={delta_r:.2f} coupled/noisy',
               model_coupled, 'coupled', coupled_noisy=True, losses=losses_coupled)

    # --- Coupled V5-style: noisy/clean ---
    do_rollout(f'D={D} dr={delta_r:.2f} coupled/clean',
               model_coupled, 'coupled', coupled_noisy=False, losses=losses_coupled)

    # --- Decoupled with various fixed s values ---
    for s_val in infer_s_values:
        do_rollout(f'D={D} dr={delta_r:.2f} decoupled/s={s_val}',
                   model_decoupled, 'decoupled', infer_s=s_val,
                   losses=losses_decoupled)

    # --- Decoupled with s ramp ---
    for ramp_max in s_ramp_values:
        do_rollout(f'D={D} dr={delta_r:.2f} decoupled/ramp→{ramp_max}',
                   model_decoupled, 'decoupled', s_ramp_max=ramp_max,
                   losses=losses_decoupled)

    return results, trajs, datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--dims', type=int, nargs='+', default=None)
    parser.add_argument('--delta_rs', type=float, nargs='+', default=None)
    parser.add_argument('--infer_s', type=float, nargs='+', default=None,
                        help='Fixed s values for decoupled inference (default: 0 0.05 0.1 0.2)')
    parser.add_argument('--s_ramps', type=float, nargs='+', default=None,
                        help='Ramp s from 0 to this value over AR steps (default: 0.1 0.3)')
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--n_ar_steps', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--outdir', type=str, default='./results_v7')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    if args.quick:
        cfg = dict(
            n_samples=10_000, n_epochs=80, batch_size=512,
            n_ar_steps=50, n_ode_steps=20, n_eval=128,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        )
        dims = args.dims or [128]
        delta_rs = args.delta_rs or [0.1]
        infer_s_values = args.infer_s or [0.0, 0.1]
        s_ramp_values = args.s_ramps or [0.1]
    else:
        cfg = dict(
            n_samples=50_000, n_epochs=args.n_epochs or 300, batch_size=512,
            n_ar_steps=args.n_ar_steps or 200, n_ode_steps=50, n_eval=256,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        )
        dims = args.dims or [128]
        delta_rs = args.delta_rs or [0.1, 0.4]
        infer_s_values = args.infer_s or [0.0, 0.05, 0.1, 0.2]
        s_ramp_values = args.s_ramps or [0.1, 0.3]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    plot_schedule_illustration(str(out / 'schedule_illustration.png'))

    all_results, all_traj, all_ds = {}, {}, {}
    total_t0 = time.time()

    for D in dims:
        for dr in delta_rs:
            res, trajs, datasets = run_experiment(
                D, dr, infer_s_values, s_ramp_values, cfg, device, out,
            )
            all_results.update(res)
            all_traj.update(trajs)
            all_ds.update(datasets)

    elapsed = time.time() - total_t0
    print(f"\nTotal wall time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ---- Save plots ----
    plot_results(all_results, cfg, str(out / 'drift_analysis.png'))
    plot_trajectories(all_traj, all_ds, str(out / 'trajectories_2d.png'))
    plot_radius_histograms(all_traj, all_ds, str(out / 'radius_histograms.png'))
    plot_drift_vs_s(all_results, cfg['n_ar_steps'], str(out / 'drift_vs_s.png'))

    # ---- Summary table ----
    print(f"\n{'='*120}")
    print(f"SUMMARY — AR Drift after {cfg['n_ar_steps']} steps (decoupled t,s)")
    print(f"{'='*120}")
    print(f"{'Config':<55} {'Final r':>9} {'r_std':>9} {'Drift':>9} {'|r-1|':>9} {'Off-mfld':>11}")
    print('-' * 120)
    for label, res in all_results.items():
        drift = res['radius_mean'][-1] - 1.0
        print(f"{label:<55} {res['radius_mean'][-1]:>9.4f} {res['radius_std'][-1]:>9.4f} "
              f"{drift:>+9.4f} {res['radius_error'][-1]:>9.4f} {res['off_manifold_energy'][-1]:>11.6f}")

    summary = {
        'config': cfg,
        'infer_s_values': infer_s_values,
        's_ramp_values': s_ramp_values,
        'results': {
            label: dict(
                mode=res['mode'],
                infer_s=res.get('infer_s'),
                final_radius=res['radius_mean'][-1],
                final_std=res['radius_std'][-1],
                signed_drift=res['radius_mean'][-1] - 1.0,
                final_error=res['radius_error'][-1],
                final_off_manifold=res['off_manifold_energy'][-1],
                radius_trajectory=res['radius_mean'],
                std_trajectory=res['radius_std'],
                error_trajectory=res['radius_error'],
            )
            for label, res in all_results.items()
        }
    }
    with open(str(out / 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs in {out.resolve()}")


if __name__ == '__main__':
    main()
