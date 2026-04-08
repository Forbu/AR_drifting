"""
Rectified Flow on Circle Transitions — Radial Noise Augmentation

Core idea: AR drift is a systematic radial error (the model consistently predicts
outputs at the wrong radius). Train the model with conditions that already simulate
this by scaling x_cond radially:

  x_cond_aug = x_cond * (1 + noise_scale * eps)    eps ~ N(0, 1) per sample

This is geometrically faithful: scaling a point on the circle changes only its
radius, not its angle. For D>2 projections it still works because x_cond_D = x_2d @ P.T
and P is linear, so radial scaling is preserved.

Variants tested:
  noise_scale=0.0       — baseline (no augmentation)
  noise_scale=σ         — symmetric radial noise (radius drifts ±σ)
  asymmetric_out        — condition always slightly outside: * (1 + σ|ε|)
  asymmetric_in         — condition always slightly inside:  * (1 - σ|ε|)

The asymmetric modes test whether matching the drift direction of the baseline
(which we expect to spiral outward) is important.

All modes use clean inference — augmentation is a training-only regularizer.

Usage:
    python circle_flow_experiment_v6.py                        # full run
    python circle_flow_experiment_v6.py --quick                # fast sanity check
    python circle_flow_experiment_v6.py --noise_scales 0 0.1 0.3
    python circle_flow_experiment_v6.py --no_asymmetric        # skip asymmetric modes
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

        x_2d_t = torch.tensor(x_2d, dtype=torch.float32, device=device)
        y_2d_t = torch.tensor(y_2d, dtype=torch.float32, device=device)

        self.x = x_2d_t @ self.P.T
        self.y = y_2d_t @ self.P.T
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

class FlowMLP(nn.Module):
    """Conditional rectified flow: [z_t, x_cond, t] -> x_pred (endpoint)."""

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
# 3. CONDITION AUGMENTATION
# ============================================================================

def augment_cond(x_cond, noise_scale: float, mode: str = 'symmetric'):
    """
    Radial augmentation of the condition.

    'symmetric':      x_cond * (1 + noise_scale * N(0,1))   — drift in either direction
    'asymmetric_out': x_cond * (1 + noise_scale * |N(0,1)|) — condition always outside
    'asymmetric_in':  x_cond * (1 - noise_scale * |N(0,1)|) — condition always inside
    'clean':          x_cond unchanged

    Scaling x_cond by a scalar changes only the radius, not the angle, which is
    exactly the geometry of AR drift on the circle.
    """
    if mode == 'clean' or noise_scale < 1e-8:
        return x_cond
    B = x_cond.shape[0]
    eps = torch.randn(B, 1, device=x_cond.device)
    if mode == 'symmetric':
        scale = 1.0 + noise_scale * eps
    elif mode == 'asymmetric_out':
        scale = 1.0 + noise_scale * eps.abs()
    elif mode == 'asymmetric_in':
        scale = 1.0 - noise_scale * eps.abs()
    else:
        raise ValueError(f"Unknown augmentation mode: {mode}")
    return x_cond * scale


# ============================================================================
# 4. TRAINING
# ============================================================================

def train(dataset, aug_mode: str, noise_scale: float,
          hidden_dim=256, n_layers=5, n_epochs=300,
          batch_size=512, lr=1e-3, device='cuda'):
    """
    Train a FlowMLP with radial condition augmentation.

    aug_mode:    'clean' | 'symmetric' | 'asymmetric_out' | 'asymmetric_in'
    noise_scale: standard deviation of the radial scale factor
    """
    D = dataset.D
    model = FlowMLP(D, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    label = 'clean' if aug_mode == 'clean' or noise_scale < 1e-8 else f'{aug_mode}(σ={noise_scale})'
    losses = []
    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for x_cond, y_target in dataset.get_batches(batch_size):
            B = x_cond.shape[0]
            t = torch.rand(B, 1, device=device)
            eps = torch.randn_like(y_target)
            z_t = t * y_target + (1 - t) * eps
            v_target = y_target - eps

            x_cond_in = augment_cond(x_cond, noise_scale, aug_mode)

            x_pred = model(z_t, x_cond_in, t)
            v_pred = (x_pred - z_t) / (1 - t).clamp(min=0.01)
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
# 5. SAMPLING & AUTOREGRESSIVE ROLLOUT
# ============================================================================

@torch.no_grad()
def sample_next(model, x_cond, n_ode_steps=50):
    """Standard Euler ODE integration — always clean condition at inference."""
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
def autoregressive_rollout(model, start, n_ar_steps=200, n_ode_steps=50):
    traj = [start.cpu()]
    current = start
    for _ in range(n_ar_steps):
        current = sample_next(model, current, n_ode_steps)
        traj.append(current.cpu())
    return torch.stack(traj, dim=0)


# ============================================================================
# 6. EVALUATION
# ============================================================================

def evaluate_rollout(trajectory, dataset):
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
        radius_err.append((r - dataset.r).abs().mean().item())
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
# 7. PLOTTING
# ============================================================================

def plot_augmentation_illustration(noise_scales, save_path):
    """Show the radial scale distribution for each noise_scale."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    x = np.linspace(-0.5, 2.5, 500)
    for σ in noise_scales:
        if σ < 1e-8:
            ax.axvline(1.0, color='black', ls='--', linewidth=1.5, label='clean (σ=0)')
            continue
        pdf = np.exp(-0.5 * ((x - 1) / σ) ** 2) / (σ * np.sqrt(2 * np.pi))
        ax.plot(x, pdf, label=f'symmetric σ={σ}', linewidth=1.5)
    ax.set_xlabel('Radial scale factor (1 + σ·ε)')
    ax.set_ylabel('Density')
    ax.set_title('Symmetric radial noise: x_cond → x_cond · (1 + σ·ε)')
    ax.axvline(1.0, color='k', ls=':', alpha=0.3)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    σ_range = np.linspace(0, 0.5, 200)
    for mode, label, color in [
        ('symmetric',     'symmetric: E[scale] = 1.0',      'blue'),
        ('asymmetric_out','asymmetric_out: E[scale] = 1+σ√(2/π)', 'red'),
        ('asymmetric_in', 'asymmetric_in: E[scale] = 1-σ√(2/π)', 'green'),
    ]:
        if mode == 'symmetric':
            e_scale = np.ones_like(σ_range)
        elif mode == 'asymmetric_out':
            e_scale = 1.0 + σ_range * np.sqrt(2 / np.pi)
        else:
            e_scale = 1.0 - σ_range * np.sqrt(2 / np.pi)
        ax.plot(σ_range, e_scale, label=label, color=color, linewidth=2)
    ax.axhline(1.0, color='k', ls=':', alpha=0.3, label='target r=1')
    ax.set_xlabel('noise_scale σ')
    ax.set_ylabel('E[radial scale]')
    ax.set_title('Expected condition radius vs noise_scale')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_results(all_results, n_ar_steps, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Rectified Flow — Radial Noise Augmentation vs Baseline',
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
    ax.set_title('Radius Evolution (mean ± min/max envelope)')
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
    ax.set_title('Signed Radius Drift (>0 = spiraling out, <0 = inward)')
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
        for rr, ls, alpha in [
            (ds.r, '-', 0.4),
            (ds.r - ds.delta_r, '--', 0.2), (ds.r + ds.delta_r, '--', 0.2),
            (ds.r - 2*ds.delta_r, ':', 0.1), (ds.r + 2*ds.delta_r, ':', 0.1),
        ]:
            if rr > 0:
                ax.plot(rr * np.cos(theta), rr * np.sin(theta), 'k', ls=ls, alpha=alpha, linewidth=0.8)

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

    plt.suptitle('AR Trajectories (2D projection) — green=start, red=end', fontsize=12, fontweight='bold')
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

        steps = [0, T // 4, T // 2, 3 * T // 4, T - 1]
        colors_h = plt.cm.plasma(np.linspace(0.1, 0.9, len(steps)))

        all_r = []
        for step in steps:
            pts_2d = traj[step] @ P_cpu
            all_r.append(torch.norm(pts_2d, dim=-1).numpy())

        r_all = np.concatenate(all_r)
        r_min = max(0, r_all.min() - 0.05)
        r_max = r_all.max() + 0.05
        bins = np.linspace(r_min, r_max, 31)

        for si, (step, r) in enumerate(zip(steps, all_r)):
            ax.hist(r, bins=bins, alpha=0.4, color=colors_h[si], label=f'step {step}', density=True)

        ax.axvline(ds.r, color='k', ls='--', alpha=0.5, label='r')
        ax.axvline(ds.r - ds.delta_r, color='gray', ls=':', alpha=0.3)
        ax.axvline(ds.r + ds.delta_r, color='gray', ls=':', alpha=0.3, label='±1σ')
        ax.axvline(ds.r - 2*ds.delta_r, color='gray', ls=':', alpha=0.15)
        ax.axvline(ds.r + 2*ds.delta_r, color='gray', ls=':', alpha=0.15, label='±2σ')
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


def plot_drift_vs_sigma(all_results, n_ar_steps, save_path):
    """Summary scatter: final drift vs noise_scale, grouped by aug_mode."""
    # Parse results: key format is "D=X dr=Y mode(σ=Z)" or "D=X dr=Y clean"
    by_mode = {}
    for key, res in all_results.items():
        # Extract the config dict attached to result
        mode = res.get('aug_mode', '?')
        σ = res.get('noise_scale', 0.0)
        drift = res['radius_mean'][-1] - 1.0
        error = res['radius_error'][-1]
        if mode not in by_mode:
            by_mode[mode] = {'sigmas': [], 'drifts': [], 'errors': []}
        by_mode[mode]['sigmas'].append(σ)
        by_mode[mode]['drifts'].append(drift)
        by_mode[mode]['errors'].append(error)

    if not any(len(v['sigmas']) > 1 for v in by_mode.values()):
        return  # Not enough data for a meaningful scatter

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Drift vs Noise Scale after {n_ar_steps} AR steps', fontsize=12, fontweight='bold')

    for ax, metric, ylabel in [
        (axes[0], 'drifts', 'Signed drift  mean(r) - 1'),
        (axes[1], 'errors', '|mean(r) - 1|'),
    ]:
        for mode, data in by_mode.items():
            σs = np.array(data['sigmas'])
            vals = np.array(data[metric])
            order = np.argsort(σs)
            ax.plot(σs[order], vals[order], 'o-', label=mode, linewidth=1.5, markersize=5)
        ax.axhline(0, color='k', ls=':', alpha=0.3)
        ax.set_xlabel('noise_scale σ')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


# ============================================================================
# 8. MAIN
# ============================================================================

def make_run_label(D, delta_r, aug_mode, noise_scale):
    if aug_mode == 'clean' or noise_scale < 1e-8:
        return f'D={D} dr={delta_r:.2f} clean'
    return f'D={D} dr={delta_r:.2f} {aug_mode} σ={noise_scale}'


def run_experiment(D, delta_r, aug_configs, cfg, device, out: Path):
    """
    aug_configs: list of (aug_mode, noise_scale) pairs to train and evaluate.
    """
    print(f"\n{'='*70}")
    print(f"  D={D}  delta_r={delta_r}")
    print(f"{'='*70}")

    dataset = CircleTransitionDataset(
        r=1.0, delta_r=delta_r,
        theta_mean=0.3, delta_theta=0.2,
        D=D, n_samples=cfg['n_samples'], seed=42, device=device,
    )

    rng = np.random.RandomState(1042)
    angles = rng.uniform(0, 2 * np.pi, cfg['n_eval']).astype(np.float32)
    start_2d = torch.tensor(
        np.stack([np.cos(angles), np.sin(angles)], axis=1),
        dtype=torch.float32, device=device,
    )
    start_D = start_2d @ dataset.P.T

    results, trajs, datasets = {}, {}, {}

    for aug_mode, noise_scale in aug_configs:
        label = make_run_label(D, delta_r, aug_mode, noise_scale)
        print(f"\n  Training [{label}]...")
        t0 = time.time()
        model, losses = train(
            dataset,
            aug_mode=aug_mode,
            noise_scale=noise_scale,
            hidden_dim=cfg['hidden_dim'],
            n_layers=cfg['n_layers'],
            n_epochs=cfg['n_epochs'],
            batch_size=cfg['batch_size'],
            lr=1e-3, device=device,
        )
        model.eval()
        print(f"    Training: {time.time()-t0:.1f}s")

        print(f"  AR rollout [{label}] ({cfg['n_ar_steps']} steps)...")
        t0 = time.time()
        traj = autoregressive_rollout(
            model, start_D,
            n_ar_steps=cfg['n_ar_steps'],
            n_ode_steps=cfg['n_ode_steps'],
        )
        print(f"    Rollout: {time.time()-t0:.1f}s")

        res = evaluate_rollout(traj, dataset)
        res['losses'] = losses
        res['aug_mode'] = aug_mode
        res['noise_scale'] = noise_scale
        print(f"    Final: r_mean={res['radius_mean'][-1]:.4f}  "
              f"r_std={res['radius_std'][-1]:.4f}  "
              f"drift={res['radius_mean'][-1]-1.0:+.4f}")

        results[label] = res
        trajs[label] = traj
        datasets[label] = dataset

    return results, trajs, datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--dims', type=int, nargs='+', default=None)
    parser.add_argument('--delta_rs', type=float, nargs='+', default=None)
    parser.add_argument('--noise_scales', type=float, nargs='+', default=None,
                        help='Radial noise scales to test (default: 0 0.05 0.1 0.2 0.4)')
    parser.add_argument('--no_asymmetric', action='store_true',
                        help='Skip asymmetric_out and asymmetric_in modes')
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--n_ar_steps', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--outdir', type=str, default='./results_v6')
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
        noise_scales = args.noise_scales or [0.0, 0.1, 0.3]
    else:
        cfg = dict(
            n_samples=50_000, n_epochs=args.n_epochs or 300, batch_size=512,
            n_ar_steps=args.n_ar_steps or 200, n_ode_steps=50, n_eval=256,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        )
        dims = args.dims or [128]
        delta_rs = args.delta_rs or [0.1, 0.4]
        noise_scales = args.noise_scales or [0.0, 0.05, 0.1, 0.2, 0.4]

    # Build aug_configs: always include baseline, then symmetric for all σ>0,
    # then asymmetric modes at a representative σ (median of non-zero scales)
    non_zero = [σ for σ in noise_scales if σ > 1e-8]
    aug_configs = [('clean', 0.0)]
    aug_configs += [('symmetric', σ) for σ in non_zero]
    if not args.no_asymmetric and non_zero:
        mid_σ = sorted(non_zero)[len(non_zero) // 2]
        aug_configs += [('asymmetric_out', mid_σ), ('asymmetric_in', mid_σ)]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    plot_augmentation_illustration(noise_scales, str(out / 'augmentation_illustration.png'))

    all_results, all_traj, all_ds = {}, {}, {}
    total_t0 = time.time()

    for D in dims:
        for dr in delta_rs:
            res, trajs, datasets = run_experiment(D, dr, aug_configs, cfg, device, out)
            all_results.update(res)
            all_traj.update(trajs)
            all_ds.update(datasets)

    elapsed = time.time() - total_t0
    print(f"\nTotal wall time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ---- Plots ----
    plot_results(all_results, cfg['n_ar_steps'], str(out / 'drift_analysis.png'))
    plot_trajectories(all_traj, all_ds, str(out / 'trajectories_2d.png'))
    plot_radius_histograms(all_traj, all_ds, str(out / 'radius_histograms.png'))
    plot_drift_vs_sigma(all_results, cfg['n_ar_steps'], str(out / 'drift_vs_sigma.png'))

    # ---- Summary ----
    print(f"\n{'='*110}")
    print(f"SUMMARY — AR Drift after {cfg['n_ar_steps']} steps")
    print(f"{'='*110}")
    print(f"{'Config':<50} {'Final r':>9} {'r_std':>9} {'Drift':>9} {'|r-1|':>9} {'Off-mfld':>11}")
    print('-' * 110)
    for label, res in all_results.items():
        drift = res['radius_mean'][-1] - 1.0
        print(f"{label:<50} {res['radius_mean'][-1]:>9.4f} {res['radius_std'][-1]:>9.4f} "
              f"{drift:>+9.4f} {res['radius_error'][-1]:>9.4f} {res['off_manifold_energy'][-1]:>11.6f}")

    summary = {
        'config': cfg,
        'aug_configs': aug_configs,
        'results': {
            label: dict(
                aug_mode=res['aug_mode'],
                noise_scale=res['noise_scale'],
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
