"""
Hypersphere Random Walk — Rectified Flow with AR Stability Analysis

Core idea: Instead of a 1D circle manifold (v7), use a full (D-1)-dimensional
hypersphere S^{D-1} embedded in R^D. A point starts at a random position on
the sphere and takes steps in random tangent directions (Brownian motion on
the sphere / spherical random walk).

This gives MUCH richer intrinsic dimensionality:
  v7 (circle):   intrinsic dim = 1, regardless of embedding D
  this (sphere): intrinsic dim = D-1

Data generation:
  1. Start at x ∈ S^{D-1} (uniform random point on hypersphere)
  2. Sample a random tangent direction: v ⟂ x, ||v|| = 1
     (project a Gaussian onto the tangent plane, then normalize)
  3. Step: x' = x + speed * v, then re-normalize to the sphere
  4. Repeat for multi-step transitions

Training modes compared:
  - baseline:             clean condition, no condition noise parameter
  - coupled:              condition noise tied to output time s = t
  - decoupled-conditional: independent t and s, model receives s as input
  - decoupled-unconditional: independent t and s at training, but model
                             does NOT receive s — must handle all noise
                             levels blindly (ablation: isolates training
                             diversity from s-informed inference)

Evaluation:
  - Norm drift:  ||x|| should stay ≈ 1.0 across AR steps
  - Angular distance: how far the point has wandered from its start
  - Tangent energy: deviation from pure tangent direction (off-manifold drift)

Usage:
    python hypersphere_flow_experiment.py                   # full run
    python hypersphere_flow_experiment.py --quick           # fast sanity check
    python hypersphere_flow_experiment.py --dims 16 32 64
    python hypersphere_flow_experiment.py --speed 0.1 0.3
    python hypersphere_flow_experiment.py --infer_s 0 0.05 0.1 0.2
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
# 1. DATA — Hypersphere Random Walk
# ============================================================================

class HypersphereWalkDataset:
    """
    Dataset of single-step transitions on S^{D-1}.

    Each sample (x, y):
      x: current point on hypersphere (||x|| = 1)
      y: next point after a random tangent step (||y|| = 1)

    The transition: y = normalize(x + speed * v_tangent)
    where v_tangent is a random unit vector in the tangent plane at x.
    """

    def __init__(self, D=16, speed=0.2, n_samples=50_000, seed=42, device='cuda'):
        """
        Args:
            D:        Ambient dimension (sphere is S^{D-1})
            speed:    Step size (norm of tangent displacement before re-projection)
            n_samples: Number of (x, y) pairs
        """
        self.D = D
        self.speed = speed
        self.n_samples = n_samples
        self.device = device

        rng = np.random.RandomState(seed)

        # Uniform points on S^{D-1}: sample Gaussian, normalize
        x = rng.randn(n_samples, D).astype(np.float32)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)

        # Random tangent directions at each x
        # Method: sample g ~ N(0, I), project out x component: v = g - (g·x)x
        g = rng.randn(n_samples, D).astype(np.float32)
        dot = np.sum(g * x, axis=1, keepdims=True)  # (g · x)
        v = g - dot * x
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)
        # Handle degenerate case (very unlikely in high-D but safety first)
        mask = v_norm < 1e-8
        v[mask.squeeze()] = 0.0
        v_norm[mask] = 1.0
        v = v / v_norm  # unit tangent vector

        # Step and re-project to sphere
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


class MultiStepHypersphereWalkDataset:
    """
    Multi-step transitions on S^{D-1} for longer-range evaluation.
    Returns (x_start, x_end) after n_steps of random walking.
    """

    def __init__(self, D=16, speed=0.2, n_steps=5, n_samples=10_000,
                 seed=42, device='cuda'):
        self.D = D
        self.speed = speed
        self.n_samples = n_samples
        self.device = device

        rng = np.random.RandomState(seed)

        x = rng.randn(n_samples, D).astype(np.float32)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        start = x.copy()

        for _ in range(n_steps):
            g = rng.randn(n_samples, D).astype(np.float32)
            dot = np.sum(g * x, axis=1, keepdims=True)
            v = g - dot * x
            v_norm = np.linalg.norm(v, axis=1, keepdims=True)
            mask = v_norm.squeeze() < 1e-8
            v[mask] = 0.0
            v_norm[mask] = 1.0
            v = v / v_norm
            x = x + speed * v
            x = x / np.linalg.norm(x, axis=1, keepdims=True)

        self.x = torch.tensor(start, dtype=torch.float32, device=device)
        self.y = torch.tensor(x, dtype=torch.float32, device=device)


# ============================================================================
# 2. MODEL
# ============================================================================

class FlowMLPDecoupled(nn.Module):
    """
    Conditional rectified flow with decoupled time parameters.
    Input:  [z_t, c_s, t, s]  →  velocity in R^D
    """
    def __init__(self, D, hidden_dim=256, n_layers=5):
        super().__init__()
        self.D = D
        layers = []
        in_dim = 2 * D + 2
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


class FlowMLPDecoupledUncond(nn.Module):
    """
    Unconditional decoupled rectified flow.
    Same architecture as Baseline (no s input), but trained with random
    condition noise s ~ U(0,1) — the model must learn to denoise the
    output regardless of how noisy the condition is, WITHOUT being told
    the noise level.

    This isolates the effect of training diversity (seeing noisy conditions)
    from the effect of conditioning on s (informing the model about noise).

    Input:  [z_t, c_s, t]  →  velocity in R^D
    """
    def __init__(self, D, hidden_dim=256, n_layers=5):
        super().__init__()
        self.D = D
        layers = []
        in_dim = 2 * D + 1  # same as baseline: no s input
        for i in range(n_layers):
            out_dim = D if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, c_s, t):
        return self.net(torch.cat([z_t, c_s, t], dim=-1))

    def get_velocity(self, z_t, c_s, t):
        x_pred = self.forward(z_t, c_s, t)
        return (x_pred - z_t) / (1 - t).clamp(min=0.01)


class FlowMLPBaseline(nn.Module):
    """
    Standard conditional rectified flow.
    Input:  [z_t, x_cond, t]  →  velocity in R^D
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

            # LogNormal-skewed s (same as v7)
            eps_s = torch.randn(B, 1, device=device)
            s = (torch.sigmoid(1.4 + 2.0 * eps_s).clamp(1e-4, 1 - 1e-4) * s_max)

            eps = torch.randn_like(y_target)
            z_t = (1 - t) * eps + t * y_target
            v_target = y_target - eps

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


def train_decoupled_uncond(dataset, s_max=1.0, hidden_dim=256, n_layers=5,
                             n_epochs=300, batch_size=512, lr=1e-3, device='cuda'):
    """
    Train unconditional decoupled: condition noise s ~ U(0,s_max) is applied
    during training, but s is NOT passed to the model. The model must handle
    arbitrary condition noise without knowing its level.
    """
    D = dataset.D
    model = FlowMLPDecoupledUncond(D, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    label = f'decoupled-uncond(s_max={s_max})'
    losses = []
    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for x_cond, y_target in dataset.get_batches(batch_size):
            B = x_cond.shape[0]
            t = torch.rand(B, 1, device=device)

            # Same s distribution as conditional decoupled
            eps_s = torch.randn(B, 1, device=device)
            s = (torch.sigmoid(1.4 + 2.0 * eps_s).clamp(1e-4, 1 - 1e-4) * s_max)

            eps = torch.randn_like(y_target)
            z_t = (1 - t) * eps + t * y_target
            v_target = y_target - eps

            # Condition noise — same as conditional decoupled
            eps_cond = torch.randn_like(x_cond)
            c_s = (1 - s) * x_cond + s * eps_cond

            # But model does NOT receive s
            v_pred = model.get_velocity(z_t, c_s, t)
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

            eps = torch.randn_like(y_target)
            z_t = (1 - t) * eps + t * y_target
            v_target = y_target - eps

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
def sample_next_decoupled(model, x_cond, n_ode_steps=50, infer_s=0.0,
                         ode_s_start=None, ode_s_end=None):
    """
    Sample with the conditional decoupled model.

    infer_s:      fixed s across all ODE steps (used if ode_s_start is None)
    ode_s_start:  s value at t=0 (start of denoising) — overrides infer_s
    ode_s_end:    s value at t=1 (end of denoising)   — overrides infer_s

    When ode_s_start/ode_s_end are set, s is linearly interpolated across
    ODE steps. The condition c_s is recomputed each step (since s changes).
    The model sees the full (t, s(t)) trajectory.
    """
    B, D = x_cond.shape
    device = x_cond.device
    z = torch.randn(B, D, device=device)
    dt = 1.0 / n_ode_steps

    eps_cond_fixed = torch.randn_like(x_cond)

    # Fixed s mode (original behavior)
    if ode_s_start is None:
        s_tensor = torch.full((B, 1), infer_s, device=device)
        c_s = (1 - s_tensor) * x_cond + s_tensor * eps_cond_fixed
        for i in range(n_ode_steps):
            t_val = i * dt
            t = torch.full((B, 1), t_val, device=device)
            v = model.get_velocity(z, c_s, t, s_tensor)
            z = z + v * dt
    else:
        # Within-ODE s ramp: s(t) = s_start + (s_end - s_start) * t
        for i in range(n_ode_steps):
            t_val = i * dt
            t = torch.full((B, 1), t_val, device=device)
            # Linearly interpolate s from start to end as t goes 0→1
            s_val = ode_s_start + (ode_s_end - ode_s_start) * t_val
            s_tensor = torch.full((B, 1), s_val, device=device)
            # Recompute condition at each step since s changes
            c_s = (1 - s_tensor) * x_cond + s_tensor * eps_cond_fixed
            v = model.get_velocity(z, c_s, t, s_tensor)
            z = z + v * dt

    return z


@torch.no_grad()
def sample_next_decoupled_uncond(model, x_cond, n_ode_steps=50):
    """
    Sample with unconditional decoupled model.
    Always uses clean condition (s=0) since model was trained blind to s
    and must handle any condition quality. At inference we give it the best
    condition we have.
    """
    B, D = x_cond.shape
    device = x_cond.device
    z = torch.randn(B, D, device=device)
    dt = 1.0 / n_ode_steps

    for i in range(n_ode_steps):
        t_val = i * dt
        t = torch.full((B, 1), t_val, device=device)
        # Clean condition — model was trained to handle this
        v = model.get_velocity(z, x_cond, t)
        z = z + v * dt

    return z


@torch.no_grad()
def sample_next_coupled(model, x_cond, n_ode_steps=50, infer_noisy=True):
    B, D = x_cond.shape
    device = x_cond.device
    z = torch.randn(B, D, device=device)
    dt = 1.0 / n_ode_steps

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
                           s_ramp_max=0.0, coupled_noisy=True,
                           reproject_to_sphere=False,
                           ode_s_start=None, ode_s_end=None):
    """
    Autoregressive rollout on the hypersphere.

    reproject_to_sphere: if True, normalize output to ||x||=1 after each step.
                         This is the "cheating" baseline — shows what pure
                         manifold projection achieves.
    """
    traj = [start.cpu()]
    current = start

    for step in range(n_ar_steps):
        if mode == 'baseline':
            current = sample_next_baseline(model, current, n_ode_steps)
        elif mode == 'decoupled':
            if s_ramp_max > 0:
                s = s_ramp_max * (step / max(n_ar_steps - 1, 1))
            else:
                s = infer_s
            current = sample_next_decoupled(model, current, n_ode_steps, infer_s=s)
        elif mode == 'decoupled-ode-ramp':
            # s ramps WITHIN each ODE solve: s(t) = s_start + (s_end-s_start)*t
            # s_start and s_end come from ode_s_start/ode_s_end params
            current = sample_next_decoupled(
                model, current, n_ode_steps,
                ode_s_start=ode_s_start, ode_s_end=ode_s_end,
            )
        elif mode == 'decoupled-uncond':
            current = sample_next_decoupled_uncond(model, current, n_ode_steps)
        elif mode == 'coupled':
            current = sample_next_coupled(model, current, n_ode_steps,
                                          infer_noisy=coupled_noisy)

        if reproject_to_sphere:
            current = current / current.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        traj.append(current.cpu())

    return torch.stack(traj, dim=0)


# ============================================================================
# 5. EVALUATION — Hypersphere-specific metrics
# ============================================================================

def evaluate_rollout(trajectory):
    """
    Evaluate a rollout trajectory on S^{D-1}.

    Metrics:
      - norm_mean/std:       mean/std of ||x|| (should stay ≈ 1.0)
      - norm_error:          mean | ||x|| - 1 |
      - angular_displacement: cumulative angular distance from start
      - radial_component:     how much of the step is radial vs tangential
                              (radial = off-manifold drift)
    """
    T, B, D = trajectory.shape
    device = trajectory.device

    norms = torch.norm(trajectory, dim=-1)  # (T, B)

    # Angular displacement from start
    start = trajectory[0]  # (B, D)
    angular_disp = []
    for t in range(T):
        cos_sim = torch.nn.functional.cosine_similarity(
            trajectory[t], start, dim=-1
        ).clamp(-1, 1)
        angles = torch.acos(cos_sim)  # geodesic distance in radians
        angular_disp.append(angles.mean().item())

    # Radial vs tangential decomposition of steps
    radial_energy = []
    for t in range(T - 1):
        delta = trajectory[t + 1] - trajectory[t]  # (B, D)
        x = trajectory[t]
        x_hat = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        # Radial component: (delta · x_hat) * x_hat
        radial_comp = (delta * x_hat).sum(dim=-1)  # (B,)
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
# 6. PLOTTING
# ============================================================================

def plot_geometry_illustration(D, speed, save_path):
    """Illustrate the hypersphere random walk geometry."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: 2D sphere (circle) illustration ---
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, alpha=0.5)
    # Show a few random walk steps
    rng = np.random.RandomState(42)
    x = np.array([1.0, 0.0])
    path = [x.copy()]
    for _ in range(8):
        g = rng.randn(2)
        g -= g.dot(x) * x
        gn = np.linalg.norm(g)
        if gn > 1e-8:
            g /= gn
        x = x + 0.4 * g
        x /= np.linalg.norm(x)
        path.append(x.copy())
    path = np.array(path)
    colors = plt.cm.viridis(np.linspace(0, 1, len(path)))
    for i in range(len(path) - 1):
        ax.annotate('', xy=path[i+1], xytext=path[i],
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
    ax.scatter(path[:, 0], path[:, 1], c=range(len(path)), cmap='viridis',
              s=50, zorder=5, edgecolors='black', linewidths=0.5)
    ax.set_aspect('equal')
    ax.set_title(f'Random Walk on S¹ (illustration)\nspeed={0.4}')
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Tangent space visualization ---
    ax = axes[1]
    # Show how tangent projection works
    np.random.seed(42)
    x0 = np.array([0.6, 0.8])  # point on circle
    g = np.array([1.5, 1.0])    # random gaussian
    dot = g.dot(x0)
    v_tang = g - dot * x0
    v_tang_norm = v_tang / np.linalg.norm(v_tang)

    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, alpha=0.5)
    ax.scatter(*x0, c='blue', s=100, zorder=5, label='x (point on sphere)')
    ax.quiver(*x0, *g, angles='xy', scale_units='xy', scale=1,
             color='red', width=0.02, label='g ~ N(0,I)')
    ax.quiver(*x0, *v_tang, angles='xy', scale_units='xy', scale=1,
             color='green', width=0.02, label='v = g - (g·x̂)x̂ (tangent)')
    ax.quiver(*x0, *v_tang_norm * 0.3, angles='xy', scale_units='xy', scale=1,
             color='orange', width=0.025, label=f'step: x + speed·v/|v|')
    ax.set_aspect('equal')
    ax.set_title('Tangent Projection on S¹')
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2.5)
    ax.set_ylim(-1.5, 2)

    # --- Panel 3: Expected angular displacement vs steps ---
    ax = axes[2]
    for s_val, color in [(0.05, 'blue'), (0.1, 'green'),
                          (0.2, 'orange'), (0.3, 'red')]:
        steps = np.arange(0, 100)
        # Expected angular displacement ~ speed * sqrt(steps) for random walk on sphere
        # (approximation for small steps)
        disp = s_val * np.sqrt(steps)
        ax.plot(steps, disp, color=color, linewidth=2, label=f'speed={s_val}')
    ax.set_xlabel('Random Walk Steps')
    ax.set_ylabel('Expected Angular Displacement (rad)')
    ax.set_title(f'Random Walk: angular displacement ≈ speed·√n\n(for S^{{{D-1}}} in R^{D})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Hypersphere S^{{{D-1}}} Random Walk Geometry', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_results(all_results, cfg, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Hypersphere Random Walk — AR Stability Analysis',
                 fontsize=14, fontweight='bold')

    # --- Norm evolution ---
    ax = axes[0, 0]
    for key, res in all_results.items():
        steps = np.arange(len(res['norm_mean']))
        ax.plot(steps, res['norm_mean'], label=key, linewidth=1.2)
    ax.axhline(1.0, color='k', ls='--', alpha=0.5, label='target ||x||=1')
    ax.set_xlabel('AR Step')
    ax.set_ylabel('Mean ||x||')
    ax.set_title('Norm Evolution (should stay = 1)')
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

    # --- Norm spread (min/max envelope) ---
    ax = axes[0, 1]
    for key, res in all_results.items():
        steps = np.arange(len(res['norm_mean']))
        ax.fill_between(steps, res['norm_min'], res['norm_max'], alpha=0.15)
        ax.plot(steps, res['norm_mean'], linewidth=1.2, label=key)
    ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('AR Step')
    ax.set_ylabel('||x||')
    ax.set_title('Norm Envelope (min/max)')
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

    # --- Norm error ---
    ax = axes[0, 2]
    for key, res in all_results.items():
        ax.plot(res['norm_error'], label=key, linewidth=1.2)
    ax.set_xlabel('AR Step')
    ax.set_ylabel('Mean | ||x|| - 1 |')
    ax.set_title('Norm Error (off-manifold drift)')
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

    # --- Angular displacement ---
    ax = axes[1, 0]
    for key, res in all_results.items():
        ax.plot(res['angular_displacement'], label=key, linewidth=1.2)
    ax.set_xlabel('AR Step')
    ax.set_ylabel('Mean Angular Distance from Start (rad)')
    ax.set_title('Angular Displacement')
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

    # --- Radial energy ---
    ax = axes[1, 1]
    for key, res in all_results.items():
        ax.plot(res['radial_energy'], label=key, linewidth=1.2)
    ax.set_xlabel('AR Step')
    ax.set_ylabel('Radial Step Energy')
    ax.set_title('Radial Component (off-manifold drift per step)')
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

    # --- Training loss ---
    ax = axes[1, 2]
    for key, res in all_results.items():
        if 'losses' in res:
            ax.plot(res['losses'], label=key, linewidth=0.8, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_norm_histograms(all_traj, save_path):
    """Histogram of ||x|| at different AR steps for each method."""
    n = len(all_traj)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (key, traj) in enumerate(all_traj.items()):
        ax = axes[idx // cols][idx % cols]
        T, B, D = traj.shape

        steps_to_show = [0, T // 4, T // 2, 3 * T // 4, T - 1]
        colors_h = plt.cm.plasma(np.linspace(0.1, 0.9, len(steps_to_show)))

        all_norms = []
        for step in steps_to_show:
            norms = torch.norm(traj[step], dim=-1).numpy()
            all_norms.append(norms)
        concat = np.concatenate(all_norms)
        lo, hi = concat.min(), concat.max()
        if hi - lo < 1e-6:
            lo, hi = lo - 0.5, hi + 0.5
        bins = np.linspace(lo, hi, 31)

        for si, (step, norms) in enumerate(zip(steps_to_show, all_norms)):
            ax.hist(norms, bins=bins, alpha=0.4, color=colors_h[si],
                    label=f'step {step}', density=True)

        ax.axvline(1.0, color='k', ls='--', alpha=0.6, label='target=1')
        ax.set_title(key, fontsize=8)
        ax.set_xlabel('||x||')
        ax.set_ylabel('Density')
        ax.legend(fontsize=5)
        ax.grid(True, alpha=0.2)

    for idx in range(len(all_traj), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle('Norm Distribution at Various AR Steps', fontsize=12,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_pairwise_angles(all_traj, save_path):
    """
    For each method, compute pairwise cosine similarity between all samples
    at the final AR step. Shows how spread out the points are on the sphere.
    """
    n = len(all_traj)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (key, traj) in enumerate(all_traj.items()):
        ax = axes[idx // cols][idx % cols]
        T, B, D = traj.shape

        # Pairwise angles at start and end
        for step_idx, step_label in [(0, 'start'), (T // 2, 'mid'), (T - 1, 'end')]:
            pts = traj[step_idx]
            cos_sim = torch.nn.functional.cosine_similarity(
                pts.unsqueeze(1), pts.unsqueeze(0), dim=-1
            ).clamp(-1, 1)
            angles = torch.acos(cos_sim)
            # Only upper triangle
            mask = torch.triu_indices(B, B, offset=1)
            angle_vals = angles[mask[0], mask[1]].numpy()
            ax.hist(angle_vals, bins=50, alpha=0.4, density=True, label=step_label)

        ax.set_title(key, fontsize=8)
        ax.set_xlabel('Pairwise angle (rad)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)

    for idx in range(len(all_traj), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle('Pairwise Angular Distribution (how spread are points on sphere)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_drift_vs_s(all_results, n_ar_steps, save_path):
    """Final norm drift vs inference s for the decoupled model."""
    by_mode = {}
    for key, res in all_results.items():
        mode = res.get('mode', '?')
        s_val = res.get('infer_s', None)
        drift = res['norm_mean'][-1] - 1.0
        error = res['norm_error'][-1]
        if mode not in by_mode:
            by_mode[mode] = {'s_vals': [], 'drifts': [], 'errors': []}
        if s_val is not None:
            by_mode[mode]['s_vals'].append(s_val)
            by_mode[mode]['drifts'].append(drift)
            by_mode[mode]['errors'].append(error)

    if not by_mode:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f'Norm Drift vs Inference Condition Noise (s) — {n_ar_steps} AR steps',
                 fontsize=12, fontweight='bold')

    for ax, metric, ylabel in [
        (axes[0], 'drifts', 'Signed drift  mean(||x||) - 1'),
        (axes[1], 'errors', 'Mean | ||x|| - 1 |'),
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
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


# ============================================================================
# 7. MAIN
# ============================================================================

def run_experiment(D, speed, infer_s_values, s_ramp_values,
                  ode_s_end_values, ode_s_ramp_pairs,
                  cfg, device, out: Path):
    print(f"\n{'='*70}")
    print(f"  D={D} (sphere S^{D-1})  speed={speed}")
    print(f"{'='*70}")

    dataset = HypersphereWalkDataset(
        D=D, speed=speed, n_samples=cfg['n_samples'], seed=42, device=device,
    )

    # Verify data is on the sphere
    print(f"  Dataset check: ||x|| mean={dataset.x.norm(dim=-1).mean():.6f}  "
          f"||y|| mean={dataset.y.norm(dim=-1).mean():.6f}")

    # --- Train models ---
    print(f"\n  Training [baseline] model...")
    t0 = time.time()
    model_baseline, losses_baseline = train_baseline(
        dataset, hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
        lr=1e-3, device=device,
    )
    model_baseline.eval()
    print(f"    Done in {time.time()-t0:.1f}s")

    print(f"\n  Training [coupled/V5-style] model...")
    t0 = time.time()
    model_coupled, losses_coupled = train_coupled(
        dataset, hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
        lr=1e-3, device=device,
    )
    model_coupled.eval()
    print(f"    Done in {time.time()-t0:.1f}s")

    print(f"\n  Training [decoupled-conditional] model...")
    t0 = time.time()
    model_decoupled, losses_decoupled = train_decoupled(
        dataset, s_max=1.0,
        hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
        lr=1e-3, device=device,
    )
    model_decoupled.eval()
    print(f"    Done in {time.time()-t0:.1f}s")

    print(f"\n  Training [decoupled-unconditional] model...")
    t0 = time.time()
    model_decoupled_uncond, losses_decoupled_uncond = train_decoupled_uncond(
        dataset, s_max=1.0,
        hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
        n_epochs=cfg['n_epochs'], batch_size=cfg['batch_size'],
        lr=1e-3, device=device,
    )
    model_decoupled_uncond.eval()
    print(f"    Done in {time.time()-t0:.1f}s")

    # --- Starting points on the sphere ---
    rng = np.random.RandomState(1042)
    start_np = rng.randn(cfg['n_eval'], D).astype(np.float32)
    start_np = start_np / np.linalg.norm(start_np, axis=1, keepdims=True)
    start = torch.tensor(start_np, dtype=torch.float32, device=device)

    results, trajs = {}, {}

    def do_rollout(key, model, mode, infer_s=0.0, s_ramp_max=0.0,
                   coupled_noisy=True, losses=None, reproject=False,
                   ode_s_start=None, ode_s_end=None):
        tag = " [reproj]" if reproject else ""
        print(f"  AR rollout [{key}{tag}] ({cfg['n_ar_steps']} steps)...")
        t0r = time.time()
        traj = autoregressive_rollout(
            model, start,
            n_ar_steps=cfg['n_ar_steps'],
            n_ode_steps=cfg['n_ode_steps'],
            mode=mode, infer_s=infer_s,
            s_ramp_max=s_ramp_max,
            coupled_noisy=coupled_noisy,
            reproject_to_sphere=reproject,
            ode_s_start=ode_s_start, ode_s_end=ode_s_end,
        )
        print(f"    Rollout: {time.time()-t0r:.1f}s")

        res = evaluate_rollout(traj)
        res['losses'] = losses
        res['mode'] = mode
        res['infer_s'] = infer_s if mode in ('decoupled', 'decoupled-uncond', 'decoupled-ode-ramp') else None
        if mode == 'decoupled-ode-ramp':
            res['ode_s_start'] = ode_s_start
            res['ode_s_end'] = ode_s_end
        print(f"    Final: ||x||_mean={res['norm_mean'][-1]:.4f}  "
              f"norm_error={res['norm_error'][-1]:.4f}  "
              f"angular_disp={res['angular_displacement'][-1]:.3f} rad")

        results[key] = res
        trajs[key] = traj

    # --- Rollouts ---

    # 1) Baseline
    do_rollout(f'D={D} spd={speed} baseline',
               model_baseline, 'baseline', losses=losses_baseline)

    # 2) Baseline + re-projection (cheating oracle)
    do_rollout(f'D={D} spd={speed} baseline+reproj',
               model_baseline, 'baseline', losses=losses_baseline, reproject=True)

    # 3) Coupled noisy/noisy
    do_rollout(f'D={D} spd={speed} coupled/noisy',
               model_coupled, 'coupled', coupled_noisy=True, losses=losses_coupled)

    # 4) Coupled noisy/clean
    do_rollout(f'D={D} spd={speed} coupled/clean',
               model_coupled, 'coupled', coupled_noisy=False, losses=losses_coupled)

    # 5) Decoupled unconditional (blind to s, trained with noisy conditions)
    do_rollout(f'D={D} spd={speed} decoupled-uncond',
               model_decoupled_uncond, 'decoupled-uncond',
               losses=losses_decoupled_uncond)

    # 6) Decoupled conditional with fixed s values
    for s_val in infer_s_values:
        do_rollout(f'D={D} spd={speed} decoupled/s={s_val}',
                   model_decoupled, 'decoupled', infer_s=s_val,
                   losses=losses_decoupled)

    # 7) Decoupled conditional with s ramp (across AR steps)
    for ramp_max in s_ramp_values:
        do_rollout(f'D={D} spd={speed} decoupled/ramp→{ramp_max}',
                   model_decoupled, 'decoupled', s_ramp_max=ramp_max,
                   losses=losses_decoupled)

    # 8) Decoupled conditional with WITHIN-ODE s ramp
    #    Idea: s starts clean (0) at beginning of denoising (t=0),
    #    ramps up to target noise level by end (t=1).
    #    Hypothesis: early denoising doesn't need condition noise,
    #    but final placement benefits from matching the AR drift level.
    for s_end in ode_s_end_values:
        do_rollout(f'D={D} spd={speed} ode-ramp/0→{s_end}',
                   model_decoupled, 'decoupled-ode-ramp',
                   ode_s_start=0.0, ode_s_end=s_end,
                   losses=losses_decoupled)
    # Also try: start slightly noisy, ramp up more
    for s_start, s_end in ode_s_ramp_pairs:
        do_rollout(f'D={D} spd={speed} ode-ramp/{s_start}→{s_end}',
                   model_decoupled, 'decoupled-ode-ramp',
                   ode_s_start=s_start, ode_s_end=s_end,
                   losses=losses_decoupled)

    return results, trajs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--dims', type=int, nargs='+', default=None,
                        help='Ambient dimensions D (sphere is S^{D-1})')
    parser.add_argument('--speeds', type=float, nargs='+', default=None,
                        help='Step sizes for random walk on sphere')
    parser.add_argument('--infer_s', type=float, nargs='+', default=None)
    parser.add_argument('--s_ramps', type=float, nargs='+', default=None)
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--n_ar_steps', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--outdir', type=str, default='./results_hypersphere')
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
        dims = args.dims or [16]
        speeds = args.speeds or [0.2]
        infer_s_values = args.infer_s or [0.0, 0.1]
        s_ramp_values = args.s_ramps or [0.1]
        ode_s_end_values = [0.1, 0.2, 0.3]
        ode_s_ramp_pairs = [(0.05, 0.2), (0.1, 0.3)]
    else:
        cfg = dict(
            n_samples=50_000, n_epochs=args.n_epochs or 300, batch_size=512,
            n_ar_steps=args.n_ar_steps or 200, n_ode_steps=50, n_eval=256,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        )
        dims = args.dims or [16, 32, 64]
        speeds = args.speeds or [0.1, 0.3]
        infer_s_values = args.infer_s or [0.0, 0.05, 0.1, 0.2]
        s_ramp_values = args.s_ramps or [0.1, 0.3]
        ode_s_end_values = [0.1, 0.2, 0.3]
        ode_s_ramp_pairs = [(0.05, 0.2), (0.1, 0.3)]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Geometry illustration
    plot_geometry_illustration(dims[0], speeds[0], str(out / 'geometry_illustration.png'))

    all_results, all_traj = {}, {}
    total_t0 = time.time()

    for D in dims:
        for speed in speeds:
            res, trajs = run_experiment(
                D, speed, infer_s_values, s_ramp_values,
                ode_s_end_values, ode_s_ramp_pairs,
                cfg, device, out,
            )
            all_results.update(res)
            all_traj.update(trajs)

    elapsed = time.time() - total_t0
    print(f"\nTotal wall time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ---- Plots ----
    plot_results(all_results, cfg, str(out / 'drift_analysis.png'))
    plot_norm_histograms(all_traj, str(out / 'norm_histograms.png'))
    plot_pairwise_angles(all_traj, str(out / 'pairwise_angles.png'))
    plot_drift_vs_s(all_results, cfg['n_ar_steps'], str(out / 'drift_vs_s.png'))

    # ---- Summary table ----
    print(f"\n{'='*130}")
    print(f"SUMMARY — AR Stability on Hypersphere after {cfg['n_ar_steps']} steps")
    print(f"{'='*130}")
    print(f"{'Config':<58} {'||x||':>8} {'|err|':>8} {'Angle':>8} {'RadE':>10} {'Norm std':>10}")
    print('-' * 130)
    for label, res in all_results.items():
        print(f"{label:<58} {res['norm_mean'][-1]:>8.4f} "
              f"{res['norm_error'][-1]:>8.4f} "
              f"{res['angular_displacement'][-1]:>8.3f} "
              f"{res['radial_energy'][-1]:>10.6f} "
              f"{res['norm_std'][-1]:>10.4f}")

    # Save results
    summary = {
        'config': cfg,
        'dims': dims,
        'speeds': speeds,
        'infer_s_values': infer_s_values,
        's_ramp_values': s_ramp_values,
        'results': {
            label: dict(
                mode=res['mode'],
                infer_s=res.get('infer_s'),
                final_norm=res['norm_mean'][-1],
                final_norm_error=res['norm_error'][-1],
                final_angular_disp=res['angular_displacement'][-1],
                final_radial_energy=res['radial_energy'][-1],
                norm_trajectory=res['norm_mean'],
                norm_error_trajectory=res['norm_error'],
                angular_trajectory=res['angular_displacement'],
            )
            for label, res in all_results.items()
        }
    }
    with open(str(out / 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs in {out.resolve()}")


if __name__ == '__main__':
    main()
