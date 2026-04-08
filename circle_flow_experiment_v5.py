"""
Rectified Flow on Circle Transitions — Inpainting-style Noisy Conditioning

Instead of guidance (v4), noise the conditioning input along the same rectified-flow
interpolation schedule. The model learns f(z_t, x_t_cond, t) -> x_0 where:

  Training:   x_t_cond = t^alpha * x_cond + (1 - t^alpha) * eps_cond    (same t as z_t)
  Inference:  x_t_cond = t^alpha * x_cond + (1 - t^alpha) * eps_cond    (eps_cond fixed per trajectory)

The condition is progressively "revealed" as t -> 1, analogous to inpainting.
Always clean at t=1, always pure noise at t=0, regardless of alpha.

alpha=1.0 : standard RF interpolation (linear reveal)
alpha=2.0 : faster reveal — less noise at intermediate steps (softer)
alpha=0.5 : slower reveal — more noise at intermediate steps (harder)

The intuition: training with a noisy condition forces the model to learn more robust
conditional features that generalize better when AR outputs drift off-manifold.

Modes compared (train_mode / infer_mode):
  clean / clean     — standard rectified flow (baseline)
  noisy / noisy     — proposed inpainting conditioning (main experiment)
  noisy / clean     — ablation: robustness of noisy-trained model to clean inference
  clean / noisy     — ablation: distribution mismatch (expected to be worst)

Usage:
    python circle_flow_experiment_v5.py                   # full run
    python circle_flow_experiment_v5.py --quick           # fast sanity check
    python circle_flow_experiment_v5.py --modes clean/clean noisy/noisy
    python circle_flow_experiment_v5.py --alpha 2.0       # faster condition reveal (softer)
    python circle_flow_experiment_v5.py --alpha 0.5       # slower condition reveal (harder)
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

class FlowMLP(nn.Module):
    """
    Conditional rectified flow: [z_t, x_cond_input, t] -> x_pred (endpoint).
    x_cond_input can be either clean x_cond or noised x_t_cond depending on mode.
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

    def forward(self, z_t, x_cond_input, t):
        return self.net(torch.cat([z_t, x_cond_input, t], dim=-1))

    def get_velocity(self, z_t, x_cond_input, t):
        x_pred = self.forward(z_t, x_cond_input, t)
        return (x_pred - z_t) / (1 - t).clamp(min=0.01)


# ============================================================================
# 3. TRAINING
# ============================================================================

def make_cond_input(x_cond, t, noisy: bool, alpha: float = 1.0):
    """
    Build the condition input for the model.

    noisy=False (clean):  x_cond_input = x_cond
    noisy=True  (noisy):  x_cond_input = t^alpha * x_cond + (1 - t^alpha) * eps_cond

    alpha controls how fast the condition is revealed as t -> 1:
      alpha=1 : linear (standard RF interpolation)
      alpha>1 : faster reveal — less noise at intermediate t (softer)
      alpha<1 : slower reveal — more noise at intermediate t (harder)
    Always clean at t=1, always pure noise at t=0.
    """
    if not noisy:
        return x_cond
    eps_cond = torch.randn_like(x_cond)
    t_alpha = t ** alpha
    return t_alpha * x_cond + (1 - t_alpha) * eps_cond


def train(dataset, train_noisy_cond: bool, alpha: float = 1.0,
          hidden_dim=256, n_layers=5, n_epochs=300,
          batch_size=512, lr=1e-3, device='cuda'):
    """
    Train a FlowMLP.
    train_noisy_cond=False -> standard conditioning (clean x_cond)
    train_noisy_cond=True  -> inpainting conditioning (noised x_t_cond)
    alpha: power for the condition reveal schedule (1.0=linear, >1=faster reveal)
    """
    D = dataset.D
    model = FlowMLP(D, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    mode_str = f"noisy_cond(a={alpha})" if train_noisy_cond else "clean_cond"
    losses = []
    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for x_cond, y_target in dataset.get_batches(batch_size):
            B = x_cond.shape[0]
            t = torch.rand(B, 1, device=device)
            eps = torch.randn_like(y_target)
            z_t = t * y_target + (1 - t) * eps
            v_target = y_target - eps

            x_cond_input = make_cond_input(x_cond, t, train_noisy_cond, alpha)

            x_pred = model(z_t, x_cond_input, t)
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
            print(f"    [{mode_str}] epoch {epoch+1:4d}/{n_epochs}  loss={avg:.6f}")

    return model, losses


# ============================================================================
# 4. SAMPLING & AUTOREGRESSIVE ROLLOUT
# ============================================================================

@torch.no_grad()
def sample_next(model, x_cond, n_ode_steps=50,
                infer_noisy_cond: bool = False, alpha: float = 1.0):
    """
    Sample next position via Euler ODE integration.

    infer_noisy_cond=False: use x_cond directly at every step (standard).
    infer_noisy_cond=True:  fix eps_cond at step 0, then at ODE time t use
                              x_t_cond = t^alpha * x_cond + (1 - t^alpha) * eps_cond
                            The condition is fully revealed at t=1, pure noise at t=0.
    """
    B, D = x_cond.shape
    device = x_cond.device
    z = torch.randn(B, D, device=device)
    dt = 1.0 / n_ode_steps

    # Fix eps_cond once per trajectory (inpainting: fixed noisy observation)
    eps_cond = torch.randn_like(x_cond) if infer_noisy_cond else None

    for i in range(n_ode_steps):
        t_val = i * dt
        t = torch.full((B, 1), t_val, device=device)

        if infer_noisy_cond and eps_cond is not None:
            t_alpha = t_val ** alpha
            x_cond_input = t_alpha * x_cond + (1 - t_alpha) * eps_cond
        else:
            x_cond_input = x_cond

        v = model.get_velocity(z, x_cond_input, t)
        z = z + v * dt

    return z


@torch.no_grad()
def autoregressive_rollout(model, start, n_ar_steps=200, n_ode_steps=50,
                           infer_noisy_cond: bool = False, alpha: float = 1.0):
    traj = [start.cpu()]
    current = start
    for _ in range(n_ar_steps):
        current = sample_next(model, current, n_ode_steps, infer_noisy_cond, alpha)
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

def plot_noisy_cond_illustration(alpha, save_path='cond_noise_schedule.png'):
    """Show x_t_cond = t^alpha * x_cond + (1 - t^alpha) * eps for various alpha."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ts = np.linspace(0, 1, 200)
    ax = axes[0]
    for a in [0.25, 0.5, 1.0, 2.0, 4.0]:
        t_alpha = ts ** a
        ax.plot(ts, t_alpha, label=f'alpha={a}', linewidth=2)
    ax.set_xlabel('ODE time t (0=noise, 1=clean)')
    ax.set_ylabel('t^alpha  (weight on x_cond)')
    ax.set_title('Condition reveal curves: x_t_cond = t^alpha · x_cond + (1 - t^alpha) · eps')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.05)

    ax = axes[1]
    t_alpha = ts ** alpha
    ax.fill_between(ts, 0, t_alpha, alpha=0.4, label='x_cond weight', color='blue')
    ax.fill_between(ts, t_alpha, 1, alpha=0.4, label='eps_cond weight', color='orange')
    ax.set_xlabel('ODE time t')
    ax.set_ylabel('Weight')
    ax.set_title(f'Active config: alpha={alpha}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_results(all_results, cfg, save_path='drift_analysis.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Rectified Flow — Inpainting Conditioning vs Baseline',
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
    for key, info in all_results.items():
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


def plot_radius_histograms(all_traj, all_ds, save_path='radius_histograms.png'):
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


# ============================================================================
# 7. MAIN
# ============================================================================

# Map "train_mode/infer_mode" strings to (train_noisy, infer_noisy) bools
MODE_MAP = {
    'clean/clean':  (False, False),
    'noisy/noisy':  (True,  True),
    'noisy/clean':  (True,  False),
    'clean/noisy':  (False, True),
}


def run_experiment(D, delta_r, modes, alpha, cfg, device, out: Path):
    """
    For a given (D, delta_r) setting, train models once per unique train_mode,
    then run inference for all requested (train_mode, infer_mode) combinations.
    """
    print(f"\n{'='*70}")
    print(f"  D={D}  delta_r={delta_r}")
    print(f"{'='*70}")

    dataset = CircleTransitionDataset(
        r=1.0, delta_r=delta_r,
        theta_mean=0.3, delta_theta=0.2,
        D=D, n_samples=cfg['n_samples'], seed=42, device=device,
    )

    # Identify which training modes are needed
    needed_train_modes = set()
    for mode_str in modes:
        train_noisy, _ = MODE_MAP[mode_str]
        needed_train_modes.add(train_noisy)

    # Train one model per unique training mode
    trained_models = {}  # bool -> (model, losses)
    for train_noisy in sorted(needed_train_modes):
        mode_label = 'noisy_cond' if train_noisy else 'clean_cond'
        print(f"\n  Training [{mode_label}] model...")
        t0 = time.time()
        model, losses = train(
            dataset, train_noisy_cond=train_noisy, alpha=alpha,
            hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
            n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
            lr=1e-3, device=device,
        )
        model.eval()
        print(f"    Training: {time.time()-t0:.1f}s")
        trained_models[train_noisy] = (model, losses)

    # Starting points (on the unit circle, same for all modes)
    rng = np.random.RandomState(1042)
    angles = rng.uniform(0, 2 * np.pi, cfg['n_eval']).astype(np.float32)
    start_2d = torch.tensor(
        np.stack([np.cos(angles), np.sin(angles)], axis=1),
        dtype=torch.float32, device=device,
    )
    start_D = start_2d @ dataset.P.T

    results, trajs, datasets = {}, {}, {}

    for mode_str in modes:
        train_noisy, infer_noisy = MODE_MAP[mode_str]
        model, losses = trained_models[train_noisy]
        label = f'D={D} dr={delta_r:.2f} {mode_str}'

        print(f"\n  AR rollout [{mode_str}] ({cfg['n_ar_steps']} steps)...")
        t0 = time.time()
        traj = autoregressive_rollout(
            model, start_D,
            n_ar_steps=cfg['n_ar_steps'],
            n_ode_steps=cfg['n_ode_steps'],
            infer_noisy_cond=infer_noisy,
            alpha=alpha,
        )
        print(f"    Rollout: {time.time()-t0:.1f}s")

        res = evaluate_rollout(traj, dataset)
        res['losses'] = losses
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
    parser.add_argument('--modes', type=str, nargs='+', default=None,
                        choices=list(MODE_MAP.keys()),
                        help='train/infer mode pairs (default: all four)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Power for condition reveal schedule: t^alpha (1.0=linear, >1=faster/softer, <1=slower/harder)')
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--n_ar_steps', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--outdir', type=str, default='./results_v5')
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
        modes = args.modes or ['clean/clean', 'noisy/noisy', 'noisy/clean']
    else:
        cfg = dict(
            n_samples=50_000, n_epochs=args.n_epochs or 300, batch_size=512,
            n_ar_steps=args.n_ar_steps or 200, n_ode_steps=50, n_eval=256,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        )
        dims = args.dims or [128]
        delta_rs = args.delta_rs or [0.1, 0.4]
        modes = args.modes or list(MODE_MAP.keys())

    alpha = args.alpha

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    plot_noisy_cond_illustration(alpha, str(out / 'cond_noise_schedule.png'))

    all_results, all_traj, all_ds = {}, {}, {}
    total_t0 = time.time()

    for D in dims:
        for dr in delta_rs:
            res, trajs, datasets = run_experiment(D, dr, modes, alpha, cfg, device, out)
            all_results.update(res)
            all_traj.update(trajs)
            all_ds.update(datasets)

    elapsed = time.time() - total_t0
    print(f"\nTotal wall time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ---- Save plots ----
    plot_results(all_results, cfg, str(out / 'drift_analysis.png'))
    plot_trajectories(all_traj, all_ds, str(out / 'trajectories_2d.png'))
    plot_radius_histograms(all_traj, all_ds, str(out / 'radius_histograms.png'))

    # Summary table
    print(f"\n{'='*110}")
    print(f"SUMMARY — AR Drift after {cfg['n_ar_steps']} steps (alpha={alpha})")
    print(f"{'='*110}")
    print(f"{'Config':<44} {'Final r':>9} {'r_std':>9} {'Drift':>9} {'|r-1|':>9} {'Off-mfld':>11}")
    print('-' * 110)
    for label, res in all_results.items():
        drift = res['radius_mean'][-1] - 1.0
        print(f"{label:<44} {res['radius_mean'][-1]:>9.4f} {res['radius_std'][-1]:>9.4f} "
              f"{drift:>+9.4f} {res['radius_error'][-1]:>9.4f} {res['off_manifold_energy'][-1]:>11.6f}")

    summary = {
        'config': cfg,
        'alpha': alpha,
        'modes': modes,
        'results': {
            label: dict(
                mode=label.split(' ')[-1],
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
