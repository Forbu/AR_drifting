"""
Rectified Flow on Circle Transitions — Autoregressive Distribution Drift Experiment

Tests whether a conditional rectified flow maintains the circle manifold
under autoregressive rollout, comparing x-prediction (endpoint) vs v-prediction
in native 2D and projected high-D spaces.

Inspired by "Back to Basics: Let Denoising Generative Models Denoise" (Li & He, 2025)
— their toy experiment (Fig.2) but extended to temporal/autoregressive settings.

Usage:
    python circle_flow_experiment.py                    # full experiment
    python circle_flow_experiment.py --quick             # quick sanity check
    python circle_flow_experiment.py --dims 2 16 512     # custom dimensions
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
    Points on a circle (or annulus) in R², with transitions = random rotations.
    Optionally projected to D dimensions via a fixed random orthogonal matrix P.

    At each step: angle_{t+1} = angle_t + Uniform(theta - delta/2, theta + delta/2)
    """

    def __init__(
        self,
        r: float = 1.0,
        delta_r: float = 0.0,
        theta_mean: float = 0.3,
        delta_theta: float = 0.2,
        D: int = 2,
        n_samples: int = 50_000,
        seed: int = 42,
        device: str = 'cuda',
    ):
        self.r = r
        self.delta_r = delta_r
        self.theta_mean = theta_mean
        self.delta_theta = delta_theta
        self.D = D
        self.d = 2
        self.device = device

        rng = np.random.RandomState(seed)

        # Orthogonal projection P: (D, 2), with P^T P = I_{2x2}
        if D > 2:
            A = rng.randn(D, 2)
            Q, _ = np.linalg.qr(A)
            self.P = torch.tensor(Q, dtype=torch.float32, device=device)
        else:
            self.P = torch.eye(2, dtype=torch.float32, device=device)

        # Current positions
        angles = rng.uniform(0, 2 * np.pi, n_samples).astype(np.float32)
        radii = (r + rng.uniform(0, delta_r, n_samples)).astype(np.float32)

        x_2d = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

        # Transitions
        d_angles = (theta_mean + rng.uniform(-delta_theta / 2, delta_theta / 2, n_samples)).astype(np.float32)
        new_angles = angles + d_angles
        y_2d = np.stack([radii * np.cos(new_angles), radii * np.sin(new_angles)], axis=1)

        x_2d = torch.tensor(x_2d, dtype=torch.float32, device=device)
        y_2d = torch.tensor(y_2d, dtype=torch.float32, device=device)

        # Project to D-dim
        self.x = x_2d @ self.P.T  # (N, D)
        self.y = y_2d @ self.P.T  # (N, D)
        self.n_samples = n_samples

    def get_batches(self, batch_size: int, shuffle: bool = True):
        if shuffle:
            idx = torch.randperm(self.n_samples, device=self.device)
        else:
            idx = torch.arange(self.n_samples, device=self.device)
        for i in range(0, self.n_samples, batch_size):
            bi = idx[i:i + batch_size]
            yield self.x[bi], self.y[bi]

    def project_to_2d(self, points_D: torch.Tensor) -> torch.Tensor:
        return points_D @ self.P  # P^T P = I so this is the pseudo-inverse

    def compute_radius(self, points_D: torch.Tensor) -> torch.Tensor:
        return torch.norm(self.project_to_2d(points_D), dim=-1)


# ============================================================================
# 2. MODEL
# ============================================================================

class FlowMLP(nn.Module):
    """
    Conditional rectified flow MLP.

    Input:  [z_t, x_cond, t]  →  (2D + 1)-dim
    Output: D-dim

    param='x' → endpoint parameterization: net predicts clean target y
    param='v' → velocity parameterization: net predicts v = y - eps
    """

    def __init__(self, D: int, hidden_dim: int = 256, n_layers: int = 5, param: str = 'x'):
        super().__init__()
        self.D = D
        self.param = param

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
        out = self.forward(z_t, x_cond, t)
        if self.param == 'x':
            return (out - z_t) / (1 - t).clamp(min=0.01)
        return out


# ============================================================================
# 3. TRAINING
# ============================================================================

def train_flow(
    dataset: CircleTransitionDataset,
    param: str = 'x',
    hidden_dim: int = 256,
    n_layers: int = 5,
    n_epochs: int = 300,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = 'cuda',
):
    D = dataset.D
    model = FlowMLP(D, hidden_dim=hidden_dim, n_layers=n_layers, param=param).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for x_cond, y_target in dataset.get_batches(batch_size):
            B = x_cond.shape[0]
            t = torch.rand(B, 1, device=device)
            eps = torch.randn_like(y_target)

            # Rectified flow interpolation
            z_t = t * y_target + (1 - t) * eps
            v_target = y_target - eps

            if param == 'x':
                x_pred = model(z_t, x_cond, t)
                v_pred = (x_pred - z_t) / (1 - t).clamp(min=0.01)
            else:
                v_pred = model(z_t, x_cond, t)

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
            print(f"    epoch {epoch+1:4d}/{n_epochs}  loss={avg:.6f}")

    return model, losses


# ============================================================================
# 4. SAMPLING & AUTOREGRESSIVE ROLLOUT
# ============================================================================

@torch.no_grad()
def sample_next(model, x_cond, n_ode_steps=50):
    B, D = x_cond.shape
    device = x_cond.device
    z = torch.randn(B, D, device=device)
    dt = 1.0 / n_ode_steps
    for i in range(n_ode_steps):
        t = torch.full((B, 1), i * dt, device=device)
        z = z + model.get_velocity(z, x_cond, t) * dt
    return z


@torch.no_grad()
def autoregressive_rollout(model, start, n_ar_steps=200, n_ode_steps=50):
    traj = [start.cpu()]
    current = start
    for _ in range(n_ar_steps):
        current = sample_next(model, current, n_ode_steps)
        traj.append(current.cpu())
    return torch.stack(traj, dim=0)  # (T+1, B, D)


# ============================================================================
# 5. EVALUATION
# ============================================================================

def evaluate_rollout(trajectory, dataset, target_r=1.0):
    T, B, D = trajectory.shape
    P_cpu = dataset.P.cpu()

    radius_mean, radius_std, radius_err, off_manifold = [], [], [], []

    for t in range(T):
        pts = trajectory[t]
        pts_2d = pts @ P_cpu
        r = torch.norm(pts_2d, dim=-1)
        radius_mean.append(r.mean().item())
        radius_std.append(r.std().item())
        radius_err.append((r - target_r).abs().mean().item())

        if D > 2:
            recon = pts_2d @ P_cpu.T
            off_manifold.append(((pts - recon) ** 2).sum(-1).mean().item())
        else:
            off_manifold.append(0.0)

    return dict(
        radius_mean=radius_mean,
        radius_std=radius_std,
        radius_error=radius_err,
        off_manifold_energy=off_manifold,
    )


# ============================================================================
# 6. PLOTTING
# ============================================================================

def plot_results(all_results, save_path='drift_analysis.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Rectified Flow on Circle — Autoregressive Drift Analysis', fontsize=14, fontweight='bold')

    # Radius over time
    ax = axes[0, 0]
    for key, res in all_results.items():
        s = np.arange(len(res['radius_mean']))
        m = np.array(res['radius_mean'])
        sd = np.array(res['radius_std'])
        ax.plot(s, m, label=key, linewidth=1.5)
        ax.fill_between(s, m - sd, m + sd, alpha=0.1)
    ax.axhline(1.0, color='k', ls='--', alpha=0.4, label='target r=1')
    ax.set_xlabel('AR Step'); ax.set_ylabel('Mean Radius')
    ax.set_title('Radius Evolution (mean ± std)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Radius error
    ax = axes[0, 1]
    for key, res in all_results.items():
        ax.plot(res['radius_error'], label=key, linewidth=1.5)
    ax.set_xlabel('AR Step'); ax.set_ylabel('Mean |r − 1|')
    ax.set_title('Radius Drift (absolute error)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Off-manifold
    ax = axes[1, 0]
    for key, res in all_results.items():
        if any(e > 0 for e in res['off_manifold_energy']):
            ax.plot(res['off_manifold_energy'], label=key, linewidth=1.5)
    ax.set_xlabel('AR Step'); ax.set_ylabel('Off-manifold Energy')
    ax.set_title('Energy Outside Circle Subspace (D > 2 only)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Training losses
    ax = axes[1, 1]
    for key, info in all_results.items():
        if 'losses' in info:
            ax.plot(info['losses'], label=key, linewidth=0.8, alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_trajectories(all_traj, all_ds, save_path='trajectories_2d.png'):
    n = len(all_traj)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (key, traj) in zip(axes, all_traj.items()):
        ds = all_ds[key]
        T, B, D = traj.shape
        P_cpu = ds.P.cpu()

        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)

        pts_2d = traj.reshape(-1, D) @ P_cpu
        pts_2d = pts_2d.reshape(T, B, 2)

        colors = plt.cm.viridis(np.linspace(0, 1, T))
        for i in range(min(8, B)):
            for t in range(T - 1):
                ax.plot(
                    [pts_2d[t, i, 0], pts_2d[t + 1, i, 0]],
                    [pts_2d[t, i, 1], pts_2d[t + 1, i, 1]],
                    color=colors[t], alpha=0.5, linewidth=0.5,
                )
            ax.plot(pts_2d[0, i, 0].item(), pts_2d[0, i, 1].item(), 'go', ms=4)
            ax.plot(pts_2d[-1, i, 0].item(), pts_2d[-1, i, 1].item(), 'rx', ms=4)

        ax.scatter(pts_2d[-1, :, 0], pts_2d[-1, :, 1], c='red', s=2, alpha=0.3, label=f'step {T-1}')
        ax.scatter(pts_2d[0, :, 0], pts_2d[0, :, 1], c='green', s=2, alpha=0.3, label='step 0')
        ax.set_title(key, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    plt.suptitle('Autoregressive Trajectories (2D projection)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


# ============================================================================
# 7. MAIN
# ============================================================================

def run_one(D, param, label, cfg, device):
    print(f"\n{'='*60}")
    print(f"  {label}  (D={D}, param={param})")
    print(f"{'='*60}")

    ds = CircleTransitionDataset(
        r=1.0, delta_r=0.0,
        theta_mean=0.3, delta_theta=0.2,
        D=D, n_samples=cfg['n_samples'], seed=42, device=device,
    )

    t0 = time.time()
    model, losses = train_flow(
        ds, param=param,
        hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
        lr=1e-3, device=device,
    )
    print(f"    Training took {time.time()-t0:.1f}s")
    model.eval()

    # Starting points on the circle
    rng = np.random.RandomState(1042)
    angles = rng.uniform(0, 2 * np.pi, cfg['n_eval']).astype(np.float32)
    start_2d = torch.tensor(
        np.stack([np.cos(angles), np.sin(angles)], axis=1),
        dtype=torch.float32, device=device,
    )
    start_D = start_2d @ ds.P.T

    print(f"    AR rollout ({cfg['n_ar_steps']} steps, {cfg['n_ode_steps']} ODE steps)...")
    t0 = time.time()
    traj = autoregressive_rollout(model, start_D, cfg['n_ar_steps'], cfg['n_ode_steps'])
    print(f"    Rollout took {time.time()-t0:.1f}s")

    res = evaluate_rollout(traj, ds)
    res['losses'] = losses
    print(f"    Final radius={res['radius_mean'][-1]:.4f}  error={res['radius_error'][-1]:.4f}")

    return res, traj, ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Fast sanity check')
    parser.add_argument('--dims', type=int, nargs='+', default=None,
                        help='Observed dimensions to test (default: 2 8 16 32 128 512)')
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
        dims = args.dims or [2, 16, 512]
    else:
        cfg = dict(
            n_samples=50_000, n_epochs=args.n_epochs or 300, batch_size=512,
            n_ar_steps=args.n_ar_steps or 200, n_ode_steps=50, n_eval=256,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        )
        dims = args.dims or [2, 8, 16, 32, 128, 512]

    configs = []
    for D in dims:
        configs.append((D, 'x', f'D={D}, x-pred'))
        configs.append((D, 'v', f'D={D}, v-pred'))

    all_results, all_traj, all_ds = {}, {}, {}

    total_t0 = time.time()
    for D, param, label in configs:
        res, traj, ds = run_one(D, param, label, cfg, device)
        all_results[label] = res
        all_traj[label] = traj
        all_ds[label] = ds

    elapsed = time.time() - total_t0
    print(f"\nTotal wall time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ---- Save ----
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    plot_results(all_results, str(out / 'drift_analysis.png'))

    # Trajectory plot for a subset
    show_keys = [k for k in all_traj if any(f'D={d}' in k for d in [2, min(dims[-1], 128)])]
    if show_keys:
        plot_trajectories(
            {k: all_traj[k] for k in show_keys},
            {k: all_ds[k] for k in show_keys},
            str(out / 'trajectories_2d.png'),
        )

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY — AR Drift after {cfg['n_ar_steps']} steps")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Final r':>10} {'|r-1|':>10} {'Off-mfld':>12} {'Train loss':>12}")
    print('-' * 80)
    for label, res in all_results.items():
        print(f"{label:<25} {res['radius_mean'][-1]:>10.4f} {res['radius_error'][-1]:>10.4f} "
              f"{res['off_manifold_energy'][-1]:>12.6f} {res['losses'][-1]:>12.6f}")

    summary = {
        label: dict(
            final_radius=res['radius_mean'][-1],
            final_error=res['radius_error'][-1],
            final_off_manifold=res['off_manifold_energy'][-1],
            final_train_loss=res['losses'][-1],
            radius_trajectory=res['radius_mean'],
            error_trajectory=res['radius_error'],
            off_manifold_trajectory=res['off_manifold_energy'],
        )
        for label, res in all_results.items()
    }
    with open(str(out / 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs in {out.resolve()}")


if __name__ == '__main__':
    main()
