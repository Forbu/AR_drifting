"""
Rectified Flow on Circle Transitions — Unconditional Guidance for Manifold Anchoring

Train two models:
  - Conditional:   f_cond(z_t, x_current, t) -> next_position
  - Unconditional: f_uncond(z_t, t)           -> position on the annulus (marginal)

During ODE sampling:
  v = (1 - w) * v_conditional + w * v_unconditional

The unconditional velocity acts as a "manifold prior" pulling samples back
toward the data distribution, reducing autoregressive drift.

Usage:
    python circle_flow_experiment.py                          # full
    python circle_flow_experiment.py --quick                   # sanity check
    python circle_flow_experiment.py --guidance_weights 0.0 0.1 0.2 0.5
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

    Radius ~ r + N(0, delta_r²)   (gaussian centered on r, std = delta_r)
    Transition: rotation by Uniform(theta - delta_theta/2, theta + delta_theta/2)
    The next point keeps the SAME radius as the current point (rotation only).
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
# 2. MODELS
# ============================================================================

class FlowMLP(nn.Module):
    """
    Conditional rectified flow MLP with endpoint (x) parameterization.
    Input:  [z_t, x_cond, t] -> (2D+1)-dim
    Output: predicted clean target y (D-dim)
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


class UnconditionalFlowMLP(nn.Module):
    """
    Unconditional rectified flow MLP with endpoint (x) parameterization.
    Learns the marginal distribution of positions on the annulus.
    Input:  [z_t, t] -> (D+1)-dim
    Output: predicted clean target y (D-dim)
    """

    def __init__(self, D, hidden_dim=256, n_layers=5):
        super().__init__()
        self.D = D
        layers = []
        in_dim = D + 1
        for i in range(n_layers):
            out_dim = D if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, t):
        return self.net(torch.cat([z_t, t], dim=-1))

    def get_velocity(self, z_t, t):
        x_pred = self.forward(z_t, t)
        return (x_pred - z_t) / (1 - t).clamp(min=0.01)


# ============================================================================
# 3. TRAINING
# ============================================================================

def train_conditional(dataset, hidden_dim=256, n_layers=5, n_epochs=300,
                      batch_size=512, lr=1e-3, device='cuda'):
    D = dataset.D
    model = FlowMLP(D, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for x_cond, y_target in dataset.get_batches(batch_size):
            B = x_cond.shape[0]
            t = torch.rand(B, 1, device=device)
            eps = torch.randn_like(y_target)
            z_t = t * y_target + (1 - t) * eps
            v_target = y_target - eps

            x_pred = model(z_t, x_cond, t)
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
            print(f"    [cond]   epoch {epoch+1:4d}/{n_epochs}  loss={avg:.6f}")

    return model, losses


def train_unconditional(dataset, hidden_dim=256, n_layers=5, n_epochs=300,
                        batch_size=512, lr=1e-3, device='cuda'):
    """Train on the marginal distribution of y (next positions)."""
    D = dataset.D
    model = UnconditionalFlowMLP(D, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for _, y_target in dataset.get_batches(batch_size):
            B = y_target.shape[0]
            t = torch.rand(B, 1, device=device)
            eps = torch.randn_like(y_target)
            z_t = t * y_target + (1 - t) * eps
            v_target = y_target - eps

            x_pred = model(z_t, t)
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
            print(f"    [uncond] epoch {epoch+1:4d}/{n_epochs}  loss={avg:.6f}")

    return model, losses


# ============================================================================
# 4. SAMPLING & AUTOREGRESSIVE ROLLOUT
# ============================================================================

@torch.no_grad()
def sample_next(cond_model, x_cond, n_ode_steps=50,
                uncond_model=None, guidance_w=0.0):
    """
    Sample next position via ODE integration.

    v = (1 - w) * v_cond + w * v_uncond

    w=0.0: pure conditional (baseline)
    w>0.0: blend in unconditional manifold prior
    """
    B, D = x_cond.shape
    device = x_cond.device
    z = torch.randn(B, D, device=device)
    dt = 1.0 / n_ode_steps

    for i in range(n_ode_steps):
        t = torch.full((B, 1), i * dt, device=device)
        v_cond = cond_model.get_velocity(z, x_cond, t)

        if uncond_model is not None and guidance_w > 0:
            v_uncond = uncond_model.get_velocity(z, t)
            v = (1 - guidance_w) * v_cond + guidance_w * v_uncond
        else:
            v = v_cond

        z = z + v * dt

    return z


@torch.no_grad()
def autoregressive_rollout(cond_model, start, n_ar_steps=200, n_ode_steps=50,
                           uncond_model=None, guidance_w=0.0):
    traj = [start.cpu()]
    current = start
    for _ in range(n_ar_steps):
        current = sample_next(cond_model, current, n_ode_steps,
                              uncond_model, guidance_w)
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

def plot_results(all_results, cfg, save_path='drift_analysis.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Rectified Flow on Annulus — AR Drift with Unconditional Guidance (x-pred)',
                 fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    for key, res in all_results.items():
        s = np.arange(len(res['radius_mean']))
        m = np.array(res['radius_mean'])
        lo = np.array(res['radius_min'])
        hi = np.array(res['radius_max'])
        ax.plot(s, m, label=key, linewidth=1.5)
        ax.fill_between(s, lo, hi, alpha=0.08)
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
    for key, info in all_results.items():
        if 'losses_cond' in info:
            ax.plot(info['losses_cond'], label=f'{key} (cond)', linewidth=0.8, alpha=0.8)
        if 'losses_uncond' in info:
            ax.plot(info['losses_uncond'], label=f'{key} (uncond)', linewidth=0.8, alpha=0.6, ls='--')
        if 'losses' in info:
            ax.plot(info['losses'], label=key, linewidth=0.8, alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_trajectories(all_traj, all_ds, save_path='trajectories_2d.png'):
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

        ax.set_title(key, fontsize=9)
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


def plot_radius_histograms(all_traj, all_ds, cfg, save_path='radius_histograms.png'):
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
            ax.hist(r, bins=bins, alpha=0.4, color=colors_h[si], label=f'step {step}', density=True)

        ax.axvline(ds.r, color='k', ls='--', alpha=0.5, label='r')
        if ds.delta_r > 0:
            ax.axvline(ds.r - ds.delta_r, color='gray', ls=':', alpha=0.3, label='±1σ')
            ax.axvline(ds.r + ds.delta_r, color='gray', ls=':', alpha=0.3)
            ax.axvline(ds.r - 2*ds.delta_r, color='gray', ls=':', alpha=0.15, label='±2σ')
            ax.axvline(ds.r + 2*ds.delta_r, color='gray', ls=':', alpha=0.15)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel('Radius'); ax.set_ylabel('Density')
        ax.legend(fontsize=5); ax.grid(True, alpha=0.2)

    for idx in range(len(all_traj), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle('Radius Distribution at Various AR Steps', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


# ============================================================================
# 7. MAIN
# ============================================================================

def run_one(D, delta_r, guidance_w, label, cfg, device,
            cond_model=None, uncond_model=None, dataset=None,
            losses_cond=None, losses_uncond=None):
    """
    Run a single config. If models/dataset are passed, reuse them
    (so we only train once per (D, delta_r) and test multiple guidance_w).
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # --- Dataset ---
    if dataset is None:
        dataset = CircleTransitionDataset(
            r=1.0, delta_r=delta_r,
            theta_mean=0.3, delta_theta=0.2,
            D=D, n_samples=cfg['n_samples'], seed=42, device=device,
        )

    # --- Train conditional ---
    if cond_model is None:
        print("  Training conditional model...")
        t0 = time.time()
        cond_model, losses_cond = train_conditional(
            dataset, hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
            n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
            lr=1e-3, device=device,
        )
        print(f"    Cond training: {time.time()-t0:.1f}s")
        cond_model.eval()

    # --- Train unconditional (only if we need it) ---
    if uncond_model is None and guidance_w > 0:
        print("  Training unconditional model...")
        t0 = time.time()
        uncond_model, losses_uncond = train_unconditional(
            dataset, hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
            n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
            lr=1e-3, device=device,
        )
        print(f"    Uncond training: {time.time()-t0:.1f}s")
        uncond_model.eval()

    # --- Starting points ---
    rng = np.random.RandomState(1042)
    angles = rng.uniform(0, 2 * np.pi, cfg['n_eval']).astype(np.float32)
    start_2d = torch.tensor(
        np.stack([np.cos(angles), np.sin(angles)], axis=1),
        dtype=torch.float32, device=device,
    )
    start_D = start_2d @ dataset.P.T

    # --- AR rollout ---
    print(f"    AR rollout ({cfg['n_ar_steps']} steps, w={guidance_w})...")
    t0 = time.time()
    traj = autoregressive_rollout(
        cond_model, start_D, cfg['n_ar_steps'], cfg['n_ode_steps'],
        uncond_model=uncond_model if guidance_w > 0 else None,
        guidance_w=guidance_w,
    )
    print(f"    Rollout: {time.time()-t0:.1f}s")

    res = evaluate_rollout(traj, dataset)
    if losses_cond is not None:
        res['losses_cond'] = losses_cond
    if losses_uncond is not None:
        res['losses_uncond'] = losses_uncond
    print(f"    Final: r_mean={res['radius_mean'][-1]:.4f}  "
          f"r_std={res['radius_std'][-1]:.4f}  "
          f"drift={res['radius_mean'][-1]-1.0:+.4f}")

    return res, traj, dataset, cond_model, uncond_model, losses_cond, losses_uncond


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--dims', type=int, nargs='+', default=None,
                        help='Observed dimensions (default: 2 128)')
    parser.add_argument('--delta_rs', type=float, nargs='+', default=None,
                        help='Annulus std (default: 0.1 0.4)')
    parser.add_argument('--guidance_weights', type=float, nargs='+', default=None,
                        help='Uncond guidance weights (default: 0.0 0.05 0.1 0.2 0.5)')
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--n_ar_steps', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--outdir', type=str, default='./results')
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
        dims = args.dims or [2, 128]
        delta_rs = args.delta_rs or [0.1, 0.4]
        guidance_ws = args.guidance_weights or [0.0, 0.1, 0.3]
    else:
        cfg = dict(
            n_samples=50_000, n_epochs=args.n_epochs or 300, batch_size=512,
            n_ar_steps=args.n_ar_steps or 200, n_ode_steps=50, n_eval=256,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        )
        dims = args.dims or [2, 128]
        delta_rs = args.delta_rs or [0.1, 0.4]
        guidance_ws = args.guidance_weights or [0.0, 0.05, 0.1, 0.2, 0.5]

    all_results, all_traj, all_ds = {}, {}, {}

    total_t0 = time.time()

    for D in dims:
        for dr in delta_rs:
            # Train models once per (D, dr), reuse across guidance weights
            cond_model = None
            uncond_model = None
            dataset = None
            losses_cond = None
            losses_uncond = None

            for w in guidance_ws:
                label = f'D={D} s={dr:.2f} w={w:.2f}'

                res, traj, dataset, cond_model, uncond_model, losses_cond, losses_uncond = run_one(
                    D, dr, w, label, cfg, device,
                    cond_model=cond_model, uncond_model=uncond_model,
                    dataset=dataset, losses_cond=losses_cond, losses_uncond=losses_uncond,
                )

                all_results[label] = res
                all_traj[label] = traj
                all_ds[label] = dataset

    elapsed = time.time() - total_t0
    print(f"\nTotal wall time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ---- Save ----
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    plot_results(all_results, cfg, str(out / 'drift_analysis.png'))
    plot_trajectories(all_traj, all_ds, str(out / 'trajectories_2d.png'))
    plot_radius_histograms(all_traj, all_ds, cfg, str(out / 'radius_histograms.png'))

    # Summary
    print(f"\n{'='*95}")
    print(f"SUMMARY — AR Drift after {cfg['n_ar_steps']} steps (x-prediction)")
    print(f"{'='*95}")
    print(f"{'Config':<28} {'Final r':>9} {'r_std':>9} {'Drift':>9} {'|r-1|':>9} {'Off-mfld':>11}")
    print('-' * 95)
    for label, res in all_results.items():
        drift = res['radius_mean'][-1] - 1.0
        print(f"{label:<28} {res['radius_mean'][-1]:>9.4f} {res['radius_std'][-1]:>9.4f} "
              f"{drift:>+9.4f} {res['radius_error'][-1]:>9.4f} {res['off_manifold_energy'][-1]:>11.6f}")

    summary = {
        'config': cfg,
        'guidance_weights': guidance_ws,
        'results': {
            label: dict(
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