"""
Rectified Flow on Bouncing Square — Scheduled Unconditional Guidance

Dataset: A unit square (rigid body) bouncing elastically inside a [-4, 4]² box.

State vector (7-dimensional):
  (cx, cy, vx, vy, cos_θ, sin_θ, ω)
   └─ center ─┘  └─ velocity ─┘  └─ orientation ─┘  └ angular vel

Physics:
  - Center moves with velocity (vx, vy) at each step (dt=0.1)
  - Elastic wall bouncing: component of velocity flips on contact
  - Square rotates at constant angular velocity ω (no friction)
  - Center bounded to [-WALL, WALL]² (WALL=3.5) so corners stay within [-4, 4]²

Conserved quantities tested during AR rollout:
  speed      = sqrt(vx² + vy²)    conserved by elastic bouncing
  |ω|        = |angular velocity|  conserved (no torque)
  in_bounds  = fraction with center in [-3.5, 3.5]²
  unit_err   = |cos²θ + sin²θ - 1|  orientation constraint

The "oracle" (true physics) trajectory is also computed and plotted as a reference.

Guidance schedules (identical to circle_flow_experiment_v4.py):
  none, constant, early, late, cosine, sin, mid
  v = (1 - w(t)) * v_cond + w(t) * v_uncond

Usage:
    python square_flow_experiment.py                        # full run
    python square_flow_experiment.py --quick                # fast sanity check
    python square_flow_experiment.py --schedules none constant early late sin
    python square_flow_experiment.py --w_max 0.15
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
import math


# ============================================================================
# CONSTANTS & STATE LAYOUT
# ============================================================================

HALF_SIZE = 0.5   # half side-length; corners at ±HALF_SIZE from center
WALL      = 3.5   # center bounded to [-WALL, WALL]²; max corner distance = HALF_SIZE*√2 ≈ 0.71
DT        = 0.1   # simulation timestep per AR step
D_STATE   = 7     # dimensionality of the state vector

# Named indices — avoids magic numbers throughout the code
I_CX, I_CY, I_VX, I_VY, I_CTH, I_STH, I_OM = range(7)


# ============================================================================
# 1. GUIDANCE SCHEDULES
# ============================================================================

def make_schedule(name, w_max):
    """
    Return w(t): float → float, where t ∈ [0,1] is ODE time (0=noise, 1=clean).
    Blending: v = (1 - w(t)) * v_cond + w(t) * v_uncond
    """
    if name == 'none':
        return lambda t: 0.0
    elif name == 'constant':
        return lambda t: w_max
    elif name == 'early':
        return lambda t: w_max * (1.0 - t)
    elif name == 'late':
        return lambda t: w_max * t
    elif name == 'cosine':
        return lambda t: w_max * math.cos(math.pi * t / 2) ** 2
    elif name == 'sin':
        return lambda t: w_max * math.sin(math.pi * t / 2) ** 2
    elif name == 'mid':
        return lambda t: w_max * 4.0 * t * (1.0 - t)
    else:
        raise ValueError(f"Unknown schedule: {name}")


# ============================================================================
# 2. PHYSICS
# ============================================================================

def step_numpy(states, dt=DT, wall=WALL):
    """
    One elastic-bounce step for N particles.
    states : np.ndarray [N, 7]  (float32)
    returns: np.ndarray [N, 7]  (float32)
    """
    cx  = states[:, I_CX].copy()
    cy  = states[:, I_CY].copy()
    vx  = states[:, I_VX].copy()
    vy  = states[:, I_VY].copy()
    cth = states[:, I_CTH].copy()
    sth = states[:, I_STH].copy()
    om  = states[:, I_OM].copy()

    # Translate
    cx_new = cx + vx * dt
    cy_new = cy + vy * dt
    vx_new = vx.copy()
    vy_new = vy.copy()

    # Elastic bounce — x axis
    hi_x = cx_new >  wall
    lo_x = cx_new < -wall
    cx_new = np.where(hi_x,  2 * wall - cx_new, cx_new)
    cx_new = np.where(lo_x, -2 * wall - cx_new, cx_new)
    vx_new = np.where(hi_x | lo_x, -vx_new, vx_new)

    # Elastic bounce — y axis
    hi_y = cy_new >  wall
    lo_y = cy_new < -wall
    cy_new = np.where(hi_y,  2 * wall - cy_new, cy_new)
    cy_new = np.where(lo_y, -2 * wall - cy_new, cy_new)
    vy_new = np.where(hi_y | lo_y, -vy_new, vy_new)

    # Rotate — exact integration for constant ω
    th_new  = np.arctan2(sth, cth) + om * dt
    cth_new = np.cos(th_new)
    sth_new = np.sin(th_new)

    return np.stack(
        [cx_new, cy_new, vx_new, vy_new, cth_new, sth_new, om],
        axis=1,
    ).astype(np.float32)


def run_physics_oracle(start_tensor, n_ar_steps, dt=DT, wall=WALL):
    """
    Ground-truth physics trajectory for comparison.
    start_tensor : [B, 7] (on any device)
    returns      : [T+1, B, 7] (cpu)
    """
    arr = start_tensor.cpu().numpy()
    traj = [arr]
    for _ in range(n_ar_steps):
        arr = step_numpy(arr, dt=dt, wall=wall)
        traj.append(arr)
    return torch.tensor(np.stack(traj, axis=0), dtype=torch.float32)


# ============================================================================
# 3. DATASET
# ============================================================================

class SquareBallDataset:
    """
    Pairs (x, y) = (current state, next state) for one elastic-bounce step.
    State: (cx, cy, vx, vy, cos_θ, sin_θ, ω)  — D_STATE = 7

    The transition is deterministic; variation comes from the broad initial
    state distribution (random position, speed, direction, orientation, ω).
    """

    def __init__(self,
                 wall=WALL, dt=DT,
                 speed_range=(0.5, 2.0),
                 omega_range=(-2.0, 2.0),
                 n_samples=50_000,
                 seed=42,
                 device='cuda'):
        self.D        = D_STATE
        self.wall     = wall
        self.dt       = dt
        self.device   = device
        self.n_samples = n_samples

        rng = np.random.RandomState(seed)

        # Position: safely inside walls
        cx = rng.uniform(-wall * 0.9, wall * 0.9, n_samples).astype(np.float32)
        cy = rng.uniform(-wall * 0.9, wall * 0.9, n_samples).astype(np.float32)

        # Velocity: random speed in [speed_range], random direction
        speed = rng.uniform(*speed_range, n_samples).astype(np.float32)
        phi   = rng.uniform(0.0, 2 * np.pi, n_samples).astype(np.float32)
        vx = (speed * np.cos(phi)).astype(np.float32)
        vy = (speed * np.sin(phi)).astype(np.float32)

        # Orientation: unit angle → (cos, sin)
        theta = rng.uniform(0.0, 2 * np.pi, n_samples).astype(np.float32)
        cth = np.cos(theta)
        sth = np.sin(theta)

        # Angular velocity
        om = rng.uniform(*omega_range, n_samples).astype(np.float32)

        x_np = np.stack([cx, cy, vx, vy, cth, sth, om], axis=1).astype(np.float32)
        y_np = step_numpy(x_np, dt=dt, wall=wall)

        self.x = torch.tensor(x_np, dtype=torch.float32, device=device)
        self.y = torch.tensor(y_np, dtype=torch.float32, device=device)

    def get_batches(self, batch_size, shuffle=True):
        idx = (torch.randperm(self.n_samples, device=self.device)
               if shuffle else torch.arange(self.n_samples, device=self.device))
        for i in range(0, self.n_samples, batch_size):
            bi = idx[i:i + batch_size]
            yield self.x[bi], self.y[bi]


# ============================================================================
# 4. MODELS
# ============================================================================

class FlowMLP(nn.Module):
    """Conditional rectified flow: [z_t, x_cond, t] → state_pred (endpoint)."""

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
        pred = self.forward(z_t, x_cond, t)
        return (pred - z_t) / (1 - t).clamp(min=0.01)


class UnconditionalFlowMLP(nn.Module):
    """Unconditional rectified flow: [z_t, t] → state_pred (endpoint)."""

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
        pred = self.forward(z_t, t)
        return (pred - z_t) / (1 - t).clamp(min=0.01)


# ============================================================================
# 5. TRAINING
# ============================================================================

def _train(model, dataset, n_epochs, batch_size, lr, device, label):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    is_cond = isinstance(model, FlowMLP)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for x_cond, y_target in dataset.get_batches(batch_size):
            B = y_target.shape[0]
            t   = torch.rand(B, 1, device=device)
            eps = torch.randn_like(y_target)
            z_t = t * y_target + (1 - t) * eps
            v_target = y_target - eps

            pred = model(z_t, x_cond, t) if is_cond else model(z_t, t)
            v_pred = (pred - z_t) / (1 - t).clamp(min=0.01)
            loss = ((v_pred - v_target) ** 2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg = epoch_loss / n_batches
        losses.append(avg)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    [{label}] epoch {epoch+1:4d}/{n_epochs}  loss={avg:.6f}")

    return losses


def train_conditional(dataset, hidden_dim=256, n_layers=5, n_epochs=300,
                      batch_size=512, lr=1e-3, device='cuda'):
    model = FlowMLP(D_STATE, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    losses = _train(model, dataset, n_epochs, batch_size, lr, device, 'cond')
    return model, losses


def train_unconditional(dataset, hidden_dim=256, n_layers=5, n_epochs=300,
                        batch_size=512, lr=1e-3, device='cuda'):
    model = UnconditionalFlowMLP(D_STATE, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    losses = _train(model, dataset, n_epochs, batch_size, lr, device, 'uncond')
    return model, losses


# ============================================================================
# 6. SAMPLING & AUTOREGRESSIVE ROLLOUT
# ============================================================================

@torch.no_grad()
def sample_next(cond_model, x_cond, n_ode_steps=50,
                uncond_model=None, w_schedule=None):
    """ODE integration with scheduled guidance blending."""
    B, D = x_cond.shape
    device = x_cond.device
    z  = torch.randn(B, D, device=device)
    dt = 1.0 / n_ode_steps

    for i in range(n_ode_steps):
        t_val = i * dt
        t = torch.full((B, 1), t_val, device=device)
        v_cond = cond_model.get_velocity(z, x_cond, t)

        if uncond_model is not None and w_schedule is not None:
            w = w_schedule(t_val)
            if w > 1e-8:
                v_uncond = uncond_model.get_velocity(z, t)
                v = (1 - w) * v_cond + w * v_uncond
            else:
                v = v_cond
        else:
            v = v_cond

        z = z + v * dt

    return z


@torch.no_grad()
def autoregressive_rollout(cond_model, start, n_ar_steps=200, n_ode_steps=50,
                           uncond_model=None, w_schedule=None):
    traj = [start.cpu()]
    current = start
    for _ in range(n_ar_steps):
        current = sample_next(cond_model, current, n_ode_steps,
                              uncond_model, w_schedule)
        traj.append(current.cpu())
    return torch.stack(traj, dim=0)   # [T, B, D]


# ============================================================================
# 7. EVALUATION
# ============================================================================

def decode_corners(state, half_size=HALF_SIZE):
    """
    Decode 4 corners of the square from state [B,7] → [B,4,2].
    Local corners (±1, ±1)*half_size are rotated by θ and translated by (cx,cy).
    """
    cx  = state[:, I_CX]
    cy  = state[:, I_CY]
    cth = state[:, I_CTH]
    sth = state[:, I_STH]

    loc = torch.tensor(
        [[-1, -1], [1, -1], [1, 1], [-1, 1]],
        dtype=state.dtype, device=state.device,
    ) * half_size   # [4, 2]

    # Rotation matrix R = [[cth, -sth], [sth, cth]] applied to each corner
    rx = cth[:, None] * loc[None, :, 0] - sth[:, None] * loc[None, :, 1]   # [B,4]
    ry = sth[:, None] * loc[None, :, 0] + cth[:, None] * loc[None, :, 1]   # [B,4]

    return torch.stack([cx[:, None] + rx, cy[:, None] + ry], dim=-1)  # [B,4,2]


def evaluate_rollout(trajectory, wall=WALL, half_size=HALF_SIZE):
    """
    trajectory : [T, B, 7]
    Returns dict with T-length lists for each metric:
      speed_mean / speed_std  : sqrt(vx²+vy²) statistics
      omega_mean / omega_std  : |ω| statistics
      in_bounds               : fraction with center in [-wall, wall]²
      corner_in_bounds        : fraction with ALL 4 corners in [-4, 4]²
      unit_err                : mean |cos²θ + sin²θ - 1|  (orientation drift)
    """
    T = trajectory.shape[0]
    out = dict(
        speed_mean=[], speed_std=[],
        omega_mean=[], omega_std=[],
        in_bounds=[], corner_in_bounds=[], unit_err=[],
    )

    for t in range(T):
        s = trajectory[t]   # [B, 7]

        speed = torch.sqrt(s[:, I_VX] ** 2 + s[:, I_VY] ** 2)
        out['speed_mean'].append(speed.mean().item())
        out['speed_std'].append(speed.std().item())

        om_abs = s[:, I_OM].abs()
        out['omega_mean'].append(om_abs.mean().item())
        out['omega_std'].append(om_abs.std().item())

        ib = ((s[:, I_CX].abs() <= wall) & (s[:, I_CY].abs() <= wall)).float()
        out['in_bounds'].append(ib.mean().item())

        # Corners must fit in the full [-4,4]² box
        corners = decode_corners(s, half_size)       # [B,4,2]
        cib = (corners.abs() <= (wall + half_size)).all(dim=-1).all(dim=-1).float()
        out['corner_in_bounds'].append(cib.mean().item())

        ue = (s[:, I_CTH] ** 2 + s[:, I_STH] ** 2 - 1.0).abs().mean().item()
        out['unit_err'].append(ue)

    return out


# ============================================================================
# 8. PLOTTING
# ============================================================================

def plot_schedules(schedules, w_max, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ts = np.linspace(0, 1, 200)
    for name in schedules:
        if name == 'none':
            continue
        fn = make_schedule(name, w_max)
        ax.plot(ts, [fn(t) for t in ts], label=name, linewidth=2)
    ax.set_xlabel('ODE time t  (0 = noise, 1 = clean)')
    ax.set_ylabel('w(t)')
    ax.set_title(f'Guidance Schedules  (w_max={w_max})')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, w_max * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_dynamics(all_results, cfg, w_max, save_path):
    """6-panel figure: speed, |omega|, in-bounds, unit-err + training loss."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Bouncing Square — Scheduled Unconditional Guidance  (w_max={w_max})',
        fontsize=13, fontweight='bold',
    )

    def _panel(ax, key, ylabel, title, ref=None, ylog=False):
        for label, res in all_results.items():
            style = dict(linewidth=2.0, linestyle='--') if label == 'oracle' else dict(linewidth=1.5)
            ax.plot(res[key], label=label, **style)
        if ref is not None:
            ax.axhline(ref, color='k', ls=':', alpha=0.4, linewidth=1)
        ax.set_xlabel('AR Step'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        if ylog:
            ax.set_yscale('log')

    first_model = next(k for k in all_results if k != 'oracle')
    speed_ref = all_results[first_model]['speed_mean'][0]
    omega_ref = all_results[first_model]['omega_mean'][0]

    _panel(axes[0, 0], 'speed_mean',        '|v|',           'Speed Mean  (conserved → flat)',  ref=speed_ref)
    _panel(axes[0, 1], 'speed_std',          'Std(|v|)',      'Speed Spread')
    _panel(axes[0, 2], 'omega_mean',         '|ω|',           '|Angular Velocity| Mean  (conserved → flat)', ref=omega_ref)
    _panel(axes[1, 0], 'in_bounds',          'Fraction',      'Center In-Bounds  [-3.5, 3.5]²')
    _panel(axes[1, 1], 'corner_in_bounds',   'Fraction',      'All Corners In-Bounds  [-4, 4]²')
    _panel(axes[1, 2], 'unit_err',           '|cos²+sin²-1|', 'Orientation Unit-Norm Error', ylog=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_training_loss(all_results, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, res in all_results.items():
        if 'losses_cond' in res:
            ax.plot(res['losses_cond'], label=f'{label} (cond)', linewidth=1.0, alpha=0.9)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss (conditional model)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_trajectories_2d(all_traj, save_path, wall=WALL, half_size=HALF_SIZE, n_show=8):
    """Top-down view: colored center paths + drawn square at start/end."""
    labels = list(all_traj.keys())
    n = len(labels)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    lim = wall + half_size + 0.5

    for idx, key in enumerate(labels):
        traj = all_traj[key]
        ax   = axes[idx // cols][idx % cols]
        T, B, D = traj.shape

        # Wall box
        bx = [-wall, wall, wall, -wall, -wall]
        by = [-wall, -wall, wall, wall, -wall]
        ax.plot(bx, by, 'k-', linewidth=1.5, alpha=0.4, zorder=2)
        # Full box at ±4
        ax.plot([-4,-4,4,4,-4], [-4,4,4,-4,-4], 'k:', linewidth=0.8, alpha=0.2, zorder=1)

        cx_all = traj[:, :, I_CX].numpy()   # [T, B]
        cy_all = traj[:, :, I_CY].numpy()   # [T, B]
        colors = plt.cm.viridis(np.linspace(0, 1, T))

        for i in range(min(n_show, B)):
            for t in range(T - 1):
                ax.plot([cx_all[t, i], cx_all[t+1, i]],
                        [cy_all[t, i], cy_all[t+1, i]],
                        color=colors[t], alpha=0.3, linewidth=0.4)

        # Draw the actual rotated square at step 0 (green) and T-1 (red) for particle 0
        for step, color, zord in [(0, 'green', 6), (T - 1, 'red', 6)]:
            s0 = traj[step, :1]
            corners = decode_corners(s0, half_size)[0].numpy()   # [4, 2]
            closed  = np.vstack([corners, corners[0]])
            ax.plot(closed[:, 0], closed[:, 1], color=color,
                    linewidth=2.0, alpha=0.85, zorder=zord)

        ax.scatter(cx_all[0],  cy_all[0],  c='green', s=6, alpha=0.4, zorder=5)
        ax.scatter(cx_all[-1], cy_all[-1], c='red',   s=6, alpha=0.4, zorder=5)

        ax.set_title(key, fontsize=9)
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.grid(True, alpha=0.2)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle('AR Trajectories (2D center) — green=start, red=end | solid box=center wall, dotted=±4',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_speed_histograms(all_traj, save_path):
    labels = list(all_traj.keys())
    n = len(labels)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, key in enumerate(labels):
        ax   = axes[idx // cols][idx % cols]
        traj = all_traj[key]
        T    = traj.shape[0]
        steps_to_show = [0, T // 4, T // 2, 3 * T // 4, T - 1]
        palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(steps_to_show)))

        all_speeds = []
        for step in steps_to_show:
            s = traj[step]
            all_speeds.append(torch.sqrt(s[:, I_VX]**2 + s[:, I_VY]**2).numpy())
        rng_min = min(v.min() for v in all_speeds)
        rng_max = max(v.max() for v in all_speeds)
        bins = np.linspace(max(0, rng_min - 0.1), rng_max + 0.1, 31)

        for si, (step, spd) in enumerate(zip(steps_to_show, all_speeds)):
            ax.hist(spd, bins=bins, alpha=0.4, color=palette[si],
                    label=f'step {step}', density=True)

        ax.set_title(key, fontsize=9)
        ax.set_xlabel('Speed |v|'); ax.set_ylabel('Density')
        ax.legend(fontsize=5); ax.grid(True, alpha=0.2)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle('Speed Distribution at Various AR Steps  (should stay constant)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_omega_histograms(all_traj, save_path):
    labels = list(all_traj.keys())
    n = len(labels)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, key in enumerate(labels):
        ax   = axes[idx // cols][idx % cols]
        traj = all_traj[key]
        T    = traj.shape[0]
        steps_to_show = [0, T // 4, T // 2, 3 * T // 4, T - 1]
        palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(steps_to_show)))

        all_oms = []
        for step in steps_to_show:
            s = traj[step]
            all_oms.append(s[:, I_OM].abs().numpy())
        rng_min = min(v.min() for v in all_oms)
        rng_max = max(v.max() for v in all_oms)
        bins = np.linspace(max(0, rng_min - 0.05), rng_max + 0.05, 31)

        for si, (step, om) in enumerate(zip(steps_to_show, all_oms)):
            ax.hist(om, bins=bins, alpha=0.4, color=palette[si],
                    label=f'step {step}', density=True)

        ax.set_title(key, fontsize=9)
        ax.set_xlabel('|ω|  (angular speed)'); ax.set_ylabel('Density')
        ax.legend(fontsize=5); ax.grid(True, alpha=0.2)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle('Angular Speed Distribution at Various AR Steps  (should stay constant)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


# ============================================================================
# 9. MAIN RUN FUNCTION
# ============================================================================

def run_one(sched_name, w_max, label, cfg, device,
            cond_model=None, uncond_model=None, dataset=None,
            losses_cond=None, losses_uncond=None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if dataset is None:
        dataset = SquareBallDataset(
            n_samples=cfg['n_samples'], seed=42, device=device,
        )

    if cond_model is None:
        print("  Training conditional model...")
        t0 = time.time()
        cond_model, losses_cond = train_conditional(
            dataset, hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
            n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
            lr=1e-3, device=device,
        )
        cond_model.eval()
        print(f"    Cond training: {time.time()-t0:.1f}s")

    need_uncond = sched_name != 'none'
    if uncond_model is None and need_uncond:
        print("  Training unconditional model...")
        t0 = time.time()
        uncond_model, losses_uncond = train_unconditional(
            dataset, hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
            n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
            lr=1e-3, device=device,
        )
        uncond_model.eval()
        print(f"    Uncond training: {time.time()-t0:.1f}s")

    w_schedule = make_schedule(sched_name, w_max) if need_uncond else None

    # Starting states: reproducible random sample from the data distribution
    rng = np.random.RandomState(1042)
    n_eval = cfg['n_eval']
    speed_s  = rng.uniform(0.5, 2.0, n_eval).astype(np.float32)
    phi_s    = rng.uniform(0.0, 2 * np.pi, n_eval).astype(np.float32)
    theta_s  = rng.uniform(0.0, 2 * np.pi, n_eval).astype(np.float32)
    start_np = np.stack([
        rng.uniform(-WALL * 0.8, WALL * 0.8, n_eval).astype(np.float32),  # cx
        rng.uniform(-WALL * 0.8, WALL * 0.8, n_eval).astype(np.float32),  # cy
        speed_s * np.cos(phi_s),                                            # vx
        speed_s * np.sin(phi_s),                                            # vy
        np.cos(theta_s),                                                    # cos_θ
        np.sin(theta_s),                                                    # sin_θ
        rng.uniform(-2.0, 2.0, n_eval).astype(np.float32),                 # ω
    ], axis=1).astype(np.float32)
    start = torch.tensor(start_np, dtype=torch.float32, device=device)

    print(f"    AR rollout ({cfg['n_ar_steps']} steps, schedule={sched_name}, w_max={w_max})...")
    t0 = time.time()
    traj = autoregressive_rollout(
        cond_model, start, cfg['n_ar_steps'], cfg['n_ode_steps'],
        uncond_model=uncond_model if need_uncond else None,
        w_schedule=w_schedule,
    )
    print(f"    Rollout: {time.time()-t0:.1f}s")

    res = evaluate_rollout(traj)
    if losses_cond is not None:
        res['losses_cond'] = losses_cond
    if losses_uncond is not None:
        res['losses_uncond'] = losses_uncond

    print(f"    Final: speed={res['speed_mean'][-1]:.4f} (init {res['speed_mean'][0]:.4f})  "
          f"|ω|={res['omega_mean'][-1]:.4f} (init {res['omega_mean'][0]:.4f})  "
          f"in_bounds={res['in_bounds'][-1]:.4f}  "
          f"unit_err={res['unit_err'][-1]:.6f}")

    return res, traj, dataset, cond_model, uncond_model, losses_cond, losses_uncond


# ============================================================================
# 10. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',      action='store_true',
                        help='Fast sanity-check run (fewer epochs / steps)')
    parser.add_argument('--schedules',  type=str, nargs='+', default=None,
                        help='Guidance schedules to test (default: all 7)')
    parser.add_argument('--w_max',      type=float, default=0.10,
                        help='Peak guidance weight (default: 0.10)')
    parser.add_argument('--n_epochs',   type=int,   default=None)
    parser.add_argument('--n_ar_steps', type=int,   default=None)
    parser.add_argument('--hidden_dim', type=int,   default=256)
    parser.add_argument('--n_layers',   type=int,   default=5)
    parser.add_argument('--outdir',     type=str,   default='./results_square')
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
        schedules = args.schedules or ['none', 'constant', 'early', 'late', 'sin']
    else:
        cfg = dict(
            n_samples=50_000, n_epochs=args.n_epochs or 300, batch_size=512,
            n_ar_steps=args.n_ar_steps or 200, n_ode_steps=50, n_eval=256,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        )
        schedules = args.schedules or ['none', 'constant', 'early', 'late', 'cosine', 'sin', 'mid']

    w_max = args.w_max
    out   = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    plot_schedules(schedules, w_max, str(out / 'schedules.png'))

    all_results, all_traj = {}, {}
    cond_model = uncond_model = dataset = losses_cond = losses_uncond = None
    total_t0 = time.time()

    # ---- Oracle (true physics) ----
    # We generate the oracle trajectory from the same starting points as the
    # first model run.  We reuse the same random seed, so the states match.
    oracle_start_np = None

    for sched in schedules:
        label = f'{sched}'
        res, traj, dataset, cond_model, uncond_model, losses_cond, losses_uncond = run_one(
            sched, w_max, label, cfg, device,
            cond_model=cond_model, uncond_model=uncond_model,
            dataset=dataset, losses_cond=losses_cond, losses_uncond=losses_uncond,
        )
        all_results[label] = res
        all_traj[label]    = traj

        # Capture starting state once (same for all schedules)
        if oracle_start_np is None:
            oracle_start_np = traj[0].clone()  # [B, 7] cpu tensor

    # ---- Oracle trajectory ----
    print("\n  Computing oracle (true physics)...")
    oracle_traj = run_physics_oracle(oracle_start_np, cfg['n_ar_steps'])
    oracle_res  = evaluate_rollout(oracle_traj)
    all_results['oracle'] = oracle_res
    all_traj['oracle']    = oracle_traj

    elapsed = time.time() - total_t0
    print(f"\nTotal wall time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ---- Outputs ----
    plot_dynamics(all_results, cfg, w_max,  str(out / 'dynamics_analysis.png'))
    plot_trajectories_2d(all_traj,          str(out / 'trajectories_2d.png'))
    plot_speed_histograms(all_traj,          str(out / 'speed_histograms.png'))
    plot_omega_histograms(all_traj,          str(out / 'omega_histograms.png'))
    plot_training_loss(all_results,          str(out / 'training_loss.png'))

    # ---- Summary table ----
    print(f"\n{'='*110}")
    print(f"SUMMARY — AR Rollout after {cfg['n_ar_steps']} steps  (w_max={w_max})")
    print(f"{'='*110}")
    print(f"{'Schedule':<18}  {'Speed₀':>8} {'Speedₜ':>8} {'ΔSpeed':>8}  "
          f"{'|ω|₀':>8} {'|ω|ₜ':>8} {'Δ|ω|':>8}  "
          f"{'InBnds':>7} {'CrnrBnd':>8} {'UnitErr':>9}")
    print('-' * 110)
    for label, res in all_results.items():
        s0, sf = res['speed_mean'][0], res['speed_mean'][-1]
        o0, of = res['omega_mean'][0], res['omega_mean'][-1]
        ib  = res['in_bounds'][-1]
        cib = res['corner_in_bounds'][-1]
        ue  = res['unit_err'][-1]
        print(f"{label:<18}  {s0:>8.4f} {sf:>8.4f} {sf-s0:>+8.4f}  "
              f"{o0:>8.4f} {of:>8.4f} {of-o0:>+8.4f}  "
              f"{ib:>7.4f} {cib:>8.4f} {ue:>9.6f}")

    # ---- JSON dump ----
    serialisable = {}
    for label, res in all_results.items():
        serialisable[label] = dict(
            final_speed         = res['speed_mean'][-1],
            initial_speed       = res['speed_mean'][0],
            speed_drift         = res['speed_mean'][-1] - res['speed_mean'][0],
            final_omega         = res['omega_mean'][-1],
            initial_omega       = res['omega_mean'][0],
            omega_drift         = res['omega_mean'][-1] - res['omega_mean'][0],
            final_in_bounds     = res['in_bounds'][-1],
            final_corner_bounds = res['corner_in_bounds'][-1],
            final_unit_err      = res['unit_err'][-1],
            speed_trajectory    = res['speed_mean'],
            omega_trajectory    = res['omega_mean'],
            in_bounds_trajectory= res['in_bounds'],
        )
    with open(str(out / 'results.json'), 'w') as f:
        json.dump({'config': cfg, 'w_max': w_max,
                   'schedules': schedules, 'results': serialisable}, f, indent=2)

    print(f"\nAll outputs saved to {out.resolve()}")


if __name__ == '__main__':
    main()
