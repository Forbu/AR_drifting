"""
2D Video Rollout-Stability Benchmark for Rectified-Flow Forecasters
====================================================================

Motivation
----------
Weather radar/satellite frame forecasting with a rectified-flow model goes
blurry / develops artifacts after a few autoregressive (AR) rollout steps.
Prior hypersphere work (see FINDINGS.md) established that *decoupled context
noise augmentation* is the load-bearing AR-stability lever, but on real images
plain Gaussian pixel noise on the context doesn't work well — only Gaussian
*blur* did. This benchmark isolates that question on structured 2D video data
that is fast to iterate on.

Data (VIDEO of 2D frames)
-------------------------
A grayscale (optionally multichannel) 2D image sequence. Each frame is a
Gaussian "cloud" blob whose center follows a Lorenz attractor trajectory
(chaotic, low-dim manifold) projected to the image plane, with amplitude /
width slowly modulated. Consecutive frames are highly correlated but move
along a low-dim chaotic manifold. This has the same structural property as
weather frames: pixel noise is *not* a plausible perturbation under the data
distribution (it breaks the smooth Gaussian profile), but blur *is* somewhat
plausible (a wider / misplaced blob). Multichannel (C>1) renders multiple
blobs driven by different projections of the latent — analogous to multi-
channel weather fields.

Task
----
Given K=2 context frames, forecast the next frame with a rectified-flow model
(endpoint / x-prediction, inverse-conditional-variance weighted velocity loss
— matches the weather model setup).

Metric (what we optimize)
-------------------------
**rollout_mmd** — squared MMD with a Gaussian kernel between feature
distributions of long AR-rollout frames and a held-out training-frame
reference set. Features = [avg-pool frame to 6x6, grad-magnitude-energy,
Laplacian-energy, total-mass, max-value] per channel. Lower MMD == rollout
frames look more like training frames (on-manifold, sharp, right mass).
Lower is better.

Secondary: sharpness_ratio (1.0 = ideal; <1 blurry, >1 artifacts),
mass_drift (relative total-mass error), train_loss, mmd_floor (noise floor:
MMD between a fresh training sample and the reference — sanity).

The stabilization TECHNIQUE is selected via env vars and dispatched in
`augment_context()` / `extra_loss()`. This is the lever the autoresearch loop
edits. Techniques implemented:
  none, pixnoise, blur, blur_noise, manifold_noise, selffeed, spectral,
  diff_forcing, inference_blur

Usage
-----
    TECHNIQUE=pixnoise SIGMA=0.4 python video_rollout_experiment.py
    TECHNIQUE=blur BLUR_SIGMA=1.2 python video_rollout_experiment.py
Defaults are picked up from env; sensible defaults exist if unset.
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Config (env-overridable so measure.sh / the loop can sweep without rewriting)
# --------------------------------------------------------------------------- #
def _env(name, default, cast=float):
    v = os.environ.get(name)
    return cast(v) if v is not None else default

SEED            = _env("SEED", 0, int)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
IMG             = _env("IMG", 32, int)           # image side (H=W)
C_CHAN          = _env("C_CHAN", 1, int)         # channels per frame
K_CTX           = _env("K_CTX", 2, int)          # number of context frames
N_TRAJ_TRAIN    = _env("N_TRAJ_TRAIN", 220, int)
N_TRAJ_HOLD     = _env("N_TRAJ_HOLD", 40, int)
TRAJ_LEN        = _env("TRAJ_LEN", 80, int)      # frames per trajectory
LORENZ_DT       = _env("LORENZ_DT", 0.012, float)
BLOB_SIGMA      = _env("BLOB_SIGMA", 2.2, float) # pixel width
# training
TRAIN_STEPS     = _env("TRAIN_STEPS", 2000, int)
BATCH           = _env("BATCH", 96, int)
LR              = _env("LR", 2e-3, float)
ODE_STEPS       = _env("ODE_STEPS", 16, int)     # Euler substeps for sampling
# rollout eval
N_ROLLOUT       = _env("N_ROLLOUT", 24, int)
ROLLOUT_LEN     = _env("ROLLOUT_LEN", 50, int)
# technique knobs
TECHNIQUE       = os.environ.get("TECHNIQUE", "pixnoise")
SIGMA           = _env("SIGMA", 0.40, float)     # pixel-noise std
BLUR_SIGMA      = _env("BLUR_SIGMA", 1.2, float) # 2D gaussian blur std
SELFFEED_PROB   = _env("SELFFEED_PROB", 0.25, float)
SPECTRAL_W      = _env("SPECTRAL_W", 1e-2, float)
DIFFFORCE_P     = _env("DIFFFORCE_P", 0.5, float)


# --------------------------------------------------------------------------- #
# 1. DATA — Lorenz-driven 2D Gaussian blob video
# --------------------------------------------------------------------------- #
def lorenz_trajectory(n_steps, dt, seed, s=10.0, r=28.0, b=8.0/3.0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1) + 0.0
    y = rng.uniform(-1, 1) + 0.0
    z = rng.uniform(20, 25) + 0.0
    xs = np.empty((n_steps, 3), dtype=np.float32)
    for i in range(n_steps):
        dx = s * (y - x)
        dy = x * (r - z) - y
        dz = x * y - b * z
        x += dx * dt; y += dy * dt; z += dz * dt
        xs[i] = [x, y, z]
    return xs


# fixed projection matrices (3 -> 2) per channel, so each channel sees a
# different 2D projection of the same 3D latent (multi-channel analog).
_PROJ = None
def _projections(C):
    global _PROJ
    if _PROJ is None or _PROJ.shape[0] != C:
        rng = np.random.RandomState(12345)
        mats = []
        for c in range(C):
            M = rng.randn(2, 3).astype(np.float32)
            M /= np.linalg.norm(M, axis=1, keepdims=True).mean()
            mats.append(M)
        _PROJ = np.stack(mats, 0)
    return _PROJ


class _BlobRenderer:
    """Caches the coordinate grid for fast 2D Gaussian blob rendering."""
    def __init__(self, side, device):
        coords = torch.arange(side, device=device).float() + 0.5
        self.grid = torch.stack(torch.meshgrid(coords, coords, indexing='ij'), -1)  # S,S,2
        self.side = side
        self.device = device
    def render(self, centers, amps, sigma):
        # centers: (C,2), amps: (C,)
        g = self.grid  # S,S,2
        d2 = ((g[None] - centers[:, None, None, :]) ** 2).sum(-1)  # C,S,S
        return amps[:, None, None] * torch.exp(-d2 / (2.0 * sigma * sigma))  # C,S,S


class VideoSequenceData:
    """
    Generates trajectories of 2D frames and stores them flat for sampling.
    """
    def __init__(self, n_traj, traj_len, seed, img, c_chan, blob_sigma, lorenz_dt):
        self.img = img
        self.c_chan = c_chan
        self.blob_sigma = blob_sigma
        renderer = _BlobRenderer(img, DEVICE)
        proj = _projections(c_chan)  # (C,2,3)
        all_frames = []
        self.traj_starts = []
        idx = 0
        lo = np.array([-20.0, -30.0, 0.0])
        hi = np.array([20.0, 30.0, 55.0])
        for t in range(n_traj):
            traj = lorenz_trajectory(traj_len, lorenz_dt, seed + t)
            normed = (traj - lo) / (hi - lo)  # ~[0,1]
            tt = np.arange(traj_len) * lorenz_dt
            amp_master = 0.78 + 0.18 * np.sin(0.6 * tt + t).astype(np.float32)
            for i in range(traj_len):
                latent = normed[i]  # (3,)
                centers2d = proj @ latent  # (C,2)
                # map [-something, something] -> [1.5, img-1.5]
                centers2d = centers2d * (img * 0.18) + img * 0.5
                centers2d = np.clip(centers2d, 1.5, img - 1.5)
                # per-channel amplitude variation
                amps = (amp_master[i] * (0.7 + 0.3 * np.arange(c_chan) / max(1, c_chan - 1))).astype(np.float32)
                ct = torch.tensor(centers2d, dtype=torch.float32, device=DEVICE)
                ap = torch.tensor(amps, dtype=torch.float32, device=DEVICE)
                all_frames.append(renderer.render(ct, ap, blob_sigma))
            self.traj_starts.append(idx)
            idx += traj_len
        self.n_traj = n_traj
        self.traj_len = traj_len
        self.frames = torch.stack(all_frames, 0)  # (N, C, H, W)
        self.N = self.frames.shape[0]

    def sample_windows(self, batch_size, rng):
        """Returns (ctx [B,K,C,H,W], target [B,C,H,W], extra [B,C,H,W])."""
        B = batch_size
        ti = rng.randint(0, self.n_traj, size=B)
        pi = rng.randint(K_CTX, self.traj_len - 1, size=B)
        base = np.array(self.traj_starts)[ti]
        idx_ctx = np.stack([base + pi - K_CTX + k for k in range(K_CTX)], axis=1)  # B,K
        idx_tgt = base + pi
        idx_extra = base + pi - K_CTX - 1
        ctx = self.frames[idx_ctx]            # B,K,C,H,W
        tgt = self.frames[idx_tgt]            # B,C,H,W
        extra = self.frames[idx_extra]        # B,C,H,W
        return ctx, tgt, extra


# --------------------------------------------------------------------------- #
# 2. MODEL — small 2D conv rectified flow, endpoint/x-prediction
# --------------------------------------------------------------------------- #
class SinTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        t = t.float().view(-1)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(1, half))
        args = t[:, None] * freqs[None, :] * 100.0
        return torch.cat([torch.sin(args), torch.cos(args)], -1)


class ConvBlock(nn.Module):
    def __init__(self, ch, tdim, groups=4):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.film = nn.Linear(tdim, 2 * ch)
        self.act = nn.SiLU()
    def forward(self, x, temb):
        h = self.norm1(x)
        scale, shift = self.film(temb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = h * (1 + scale) + shift
        x = x + self.conv1(self.act(h))
        h = self.norm2(x)
        x = x + self.conv2(self.act(h))
        return x


class VideoRF(nn.Module):
    """Rectified flow: predicts endpoint x_pred = y_hat given (z_t, context, t).
    context: (B, K, C, H, W) -> flattened to K*C channels."""
    def __init__(self, k_ctx, c_chan, ch=48, tdim=64, n_blocks=4):
        super().__init__()
        in_ch = k_ctx * c_chan + c_chan  # K context frames (each C channels) + z_t (C)
        self.stem = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.blocks = nn.ModuleList([ConvBlock(ch, tdim) for _ in range(n_blocks)])
        self.temb = SinTime(tdim)
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.out_norm = nn.GroupNorm(4, ch)
        self.out_conv = nn.Conv2d(ch, c_chan, 3, padding=1)
        self.out_act = nn.SiLU()

    def forward(self, z_t, ctx, t):
        # z_t: (B,C,H,W); ctx: (B,K,C,H,W); t: (B,)
        temb = self.tmlp(self.temb(t))
        ctx_flat = ctx.flatten(1, 2)  # (B, K*C, H, W)
        x = torch.cat([z_t, ctx_flat], dim=1)
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x, temb)
        x = self.out_act(self.out_norm(x))
        return self.out_conv(x)  # x_pred (B,C,H,W)

    def get_velocity(self, z_t, ctx, t):
        x_pred = self.forward(z_t, ctx, t)
        return (x_pred - z_t) / (1.0 - t).clamp(min=0.01).view(-1, 1, 1, 1)


# --------------------------------------------------------------------------- #
# 3. CONTEXT AUGMENTATION — THE LEVER
# --------------------------------------------------------------------------- #
def _gauss_kernel_2d(sigma, device):
    radius = max(1, int(3 * sigma))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    g = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    k = torch.outer(g, g)
    return k.view(1, 1, *k.shape)  # 1,1,r2,r2


def _blur2d(frames, sigma):
    """frames: (..., C, H, W). Separable-ish 2D Gaussian via grouped conv."""
    if sigma <= 0:
        return frames
    shape = frames.shape
    flat = frames.reshape(-1, shape[-3], shape[-2], shape[-1])  # N,C,H,W
    k = _gauss_kernel_2d(sigma, frames.device).expand(flat.shape[1], 1, -1, -1)
    out = F.conv2d(flat, k, padding=k.shape[-1] // 2, groups=flat.shape[1])
    return out.reshape(*shape)


def augment_context(ctx, model=None, extra=None, training=True):
    """
    ctx: (B, K, C, H, W) clean context frames
    extra: (B, C, H, W) frame before context (for selffeed) or None
    Returns augmented ctx (B, K, C, H, W). training=False -> clean (decoupled)
    unless the technique explicitly injects inference-time noise.
    """
    B = ctx.shape[0]
    if not training:
        if TECHNIQUE == "inference_blur":
            return _blur2d(ctx, BLUR_SIGMA * 0.25)
        return ctx

    if TECHNIQUE == "none":
        return ctx

    if TECHNIQUE == "pixnoise":
        s = torch.sigmoid(1.4 + 2.0 * torch.randn(B, 1, 1, 1, 1, device=ctx.device)).clamp(1e-3, 1 - 1e-3) * SIGMA
        return ctx + torch.randn_like(ctx) * s

    if TECHNIQUE == "blur":
        sb = torch.sigmoid(1.0 + 1.8 * torch.randn(B, 1, 1, 1, 1, device=ctx.device)).clamp(1e-3, 1 - 1e-3) * BLUR_SIGMA
        return _blur2d(ctx, float(sb.mean()))  # per-sample sigma approx by mean (cheap)

    if TECHNIQUE == "blur_noise":
        sb = torch.sigmoid(1.0 + 1.8 * torch.randn(B, 1, 1, 1, 1, device=ctx.device)).clamp(1e-3, 1 - 1e-3) * BLUR_SIGMA
        blurred = _blur2d(ctx, float(sb.mean()))
        return blurred + torch.randn_like(ctx) * (SIGMA * 0.5)

    if TECHNIQUE == "manifold_noise":
        # manifold-aligned-ish: mild blur (smooth, plausible) + amplitude jitter
        sb = BLUR_SIGMA * 0.6 * (0.5 + torch.rand(B, 1, 1, 1, 1, device=ctx.device))
        blurred = _blur2d(ctx, float(sb.mean()))
        scale = 1.0 + 0.08 * torch.randn(B, 1, 1, 1, 1, device=ctx.device)
        return blurred * scale

    if TECHNIQUE == "selffeed":
        # scheduled sampling: w.p. SELFFEED_PROB replace LAST context frame with
        # the model's own 1-step forecast from [extra, ctx[:,0]] (detached).
        out = ctx.clone()
        if model is not None and extra is not None:
            mask = torch.rand(B, device=ctx.device) < SELFFEED_PROB
            if mask.any():
                sur_ctx = torch.stack([extra[mask], ctx[mask, 0]], dim=1)  # Bm,K,C,H,W
                with torch.no_grad():
                    pred = sample_step(model, sur_ctx, ODE_STEPS)
                out[mask, 1] = pred
        return _blur2d(out, BLUR_SIGMA * 0.4)

    if TECHNIQUE == "diff_forcing":
        out = ctx.clone()
        for k in range(K_CTX):
            keep = torch.rand(B, device=ctx.device) > DIFFFORCE_P
            s = torch.sigmoid(1.2 + 1.8 * torch.randn(B, 1, 1, 1, 1, device=ctx.device)).clamp(1e-3, 1 - 1e-3) * SIGMA
            out[:, k] = torch.where(keep.view(B, 1, 1, 1, 1), ctx[:, k],
                                    ctx[:, k] + torch.randn_like(ctx[:, k]) * s)
        return out

    if TECHNIQUE in ("spectral",):
        s = torch.sigmoid(1.4 + 2.0 * torch.randn(B, 1, 1, 1, 1, device=ctx.device)).clamp(1e-3, 1 - 1e-3) * SIGMA
        return ctx + torch.randn_like(ctx) * s

    if TECHNIQUE == "inference_blur":
        sb = torch.sigmoid(1.0 + 1.8 * torch.randn(B, 1, 1, 1, 1, device=ctx.device)).clamp(1e-3, 1 - 1e-3) * BLUR_SIGMA
        return _blur2d(ctx, float(sb.mean()))

    raise ValueError(f"unknown TECHNIQUE {TECHNIQUE}")


def extra_loss(x_pred, y_target):
    if TECHNIQUE == "spectral" and SPECTRAL_W > 0:
        # Penalize excess high-frequency energy in prediction vs target.
        def lap2d(v):
            vp = F.pad(v, (1, 1, 1, 1), mode='replicate')
            l = (vp[..., 2:, 1:-1] + vp[..., :-2, 1:-1] + vp[..., 1:-1, 2:] + vp[..., 1:-1, :-2]
                 - 4 * vp[..., 1:-1, 1:-1])
            return l
        e_pred = (lap2d(x_pred) ** 2).mean(dim=(1, 2, 3))
        e_tgt = (lap2d(y_target) ** 2).mean(dim=(1, 2, 3))
        return SPECTRAL_W * F.relu(e_pred - e_tgt * 1.5).mean()
    return torch.zeros((), device=x_pred.device)


# --------------------------------------------------------------------------- #
# 4. TRAINING
# --------------------------------------------------------------------------- #
def train(data):
    torch.manual_seed(SEED)
    model = VideoRF(K_CTX, C_CHAN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.RandomState(SEED + 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_STEPS)
    last_loss = 0.0
    for step in range(TRAIN_STEPS):
        ctx, tgt, extra = data.sample_windows(BATCH, rng)
        t = torch.rand(BATCH, device=DEVICE)
        eps = torch.randn_like(tgt)
        z_t = (1 - t.view(-1, 1, 1, 1)) * eps + t.view(-1, 1, 1, 1) * tgt
        v_target = tgt - eps
        ctx_aug = augment_context(ctx, model=model, extra=extra, training=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            v_pred = model.get_velocity(z_t, ctx_aug, t)
            # inverse-conditional-variance weight 1/(1-t)^2 clamped (RF x-pred)
            w = (1.0 / (1.0 - t).clamp(min=0.05) ** 2).clamp(max=200.0)
            loss_vel = ((v_pred - v_target) ** 2).mean(dim=(1, 2, 3)) * w
            x_pred = model.forward(z_t, ctx_aug, t)
            loss = loss_vel.mean().float() + extra_loss(x_pred.float(), tgt.float()).float()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        last_loss = loss.item()
        if (step + 1) % 500 == 0:
            print(f"  step {step+1:4d}/{TRAIN_STEPS}  loss={last_loss:.5f}", flush=True)
    return model, last_loss


# --------------------------------------------------------------------------- #
# 5. SAMPLING & ROLLOUT
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_step(model, ctx, n_ode):
    """Sample one frame (B,C,H,W) from context ctx (B,K,C,H,W)."""
    B = ctx.shape[0]
    z = torch.randn(B, C_CHAN, IMG, IMG, device=DEVICE)
    dt = 1.0 / n_ode
    ctx_aug = augment_context(ctx, training=False)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for i in range(n_ode):
            t = torch.full((B,), i * dt, device=DEVICE)
            v = model.get_velocity(z, ctx_aug, t)
            z = z + v * dt
    return z.float()


@torch.no_grad()
def rollout(model, data, n_rollout, rollout_len):
    rng = np.random.RandomState(SEED + 999)
    ti = rng.randint(0, data.n_traj, size=n_rollout)
    pi = rng.randint(K_CTX, data.traj_len - rollout_len - 1, size=n_rollout)
    base = np.array(data.traj_starts)[ti]
    ctx_idx = np.stack([base + pi - K_CTX + k for k in range(K_CTX)], axis=1)
    ctx = data.frames[ctx_idx]  # B,K,C,H,W
    frames = []
    cur = ctx.clone()
    for _ in range(rollout_len):
        nxt = sample_step(model, cur, ODE_STEPS)  # B,C,H,W
        frames.append(nxt)
        cur = torch.cat([cur[:, 1:], nxt.unsqueeze(1)], dim=1)
    return torch.stack(frames, 0)  # T,B,C,H,W


# --------------------------------------------------------------------------- #
# 6. METRICS
# --------------------------------------------------------------------------- #
def features(vols):
    """vols: (N, C, H, W) -> features (N, D)."""
    pooled = F.adaptive_avg_pool2d(vols, (6, 6)).flatten(1)  # N, C*36
    gx, gy = torch.gradient(vols, dim=(2, 3))
    grad_e = (gx ** 2 + gy ** 2).mean(dim=(1, 2, 3)).unsqueeze(1)
    vp = F.pad(vols, (1, 1, 1, 1), mode='replicate')
    l = (vp[..., 2:, 1:-1] + vp[..., :-2, 1:-1] + vp[..., 1:-1, 2:] + vp[..., 1:-1, :-2]
         - 4 * vp[..., 1:-1, 1:-1])
    lap_e = (l ** 2).mean(dim=(1, 2, 3)).unsqueeze(1)
    mass = vols.mean(dim=(1, 2, 3)).unsqueeze(1)
    maxv = vols.amax(dim=(1, 2, 3)).unsqueeze(1)
    return torch.cat([pooled, grad_e, lap_e, mass, maxv], dim=1)


def mmd2_sq(a, b, bw):
    aa = torch.cdist(a, a) ** 2
    bb = torch.cdist(b, b) ** 2
    ab = torch.cdist(a, b) ** 2
    na, nb = a.shape[0], b.shape[0]
    maa = torch.zeros((), device=a.device) if na <= 1 else (
        lambda k: (k.fill_diagonal_(0.0), k.sum() / (na * (na - 1)))[-1])(torch.exp(-aa / bw))
    mbb = torch.zeros((), device=a.device) if nb <= 1 else (
        lambda k: (k.fill_diagonal_(0.0), k.sum() / (nb * (nb - 1)))[-1])(torch.exp(-bb / bw))
    mab = torch.exp(-ab / bw).mean()
    return float((maa + mbb - 2 * mab).clamp(min=0.0))


def raw_grad_mass(frames):
    gx, gy = torch.gradient(frames, dim=(2, 3))
    ge = float((gx ** 2 + gy ** 2).mean())
    ms = float(frames.mean())
    return ge, ms


def evaluate(model, data, ref_feats, ref_stats, bw, ref_grad, ref_mass):
    rollout_frames = rollout(model, data, N_ROLLOUT, ROLLOUT_LEN)
    rf = rollout_frames.flatten(0, 1)[:1024]  # (T*B, C, H, W)
    rf_feat = features(rf)
    mu, sd = ref_stats
    rf_feat = (rf_feat - mu) / sd
    mmd = mmd2_sq(rf_feat, ref_feats, bw)
    ge_roll, mass_roll = raw_grad_mass(rf)
    sharp = ge_roll / max(ref_grad, 1e-9)
    mass_drift = abs(mass_roll - ref_mass) / max(abs(ref_mass), 1e-9)
    return mmd, sharp, mass_drift


# --------------------------------------------------------------------------- #
# 7. MAIN
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"TECHNIQUE={TECHNIQUE} SIGMA={SIGMA} BLUR_SIGMA={BLUR_SIGMA} "
          f"IMG={IMG} C_CHAN={C_CHAN} K_CTX={K_CTX} device={DEVICE}", flush=True)

    print("[data] generating train + holdout video...", flush=True)
    train_data = VideoSequenceData(N_TRAJ_TRAIN, TRAJ_LEN, SEED + 100,
                                   IMG, C_CHAN, BLOB_SIGMA, LORENZ_DT)
    hold_data  = VideoSequenceData(N_TRAJ_HOLD,  TRAJ_LEN, SEED + 9000,
                                   IMG, C_CHAN, BLOB_SIGMA, LORENZ_DT)

    n_ref = min(1024, train_data.N)
    ridx = np.random.RandomState(SEED + 5).choice(train_data.N, n_ref, replace=False)
    ref_raw = train_data.frames[ridx]
    ref_f = features(ref_raw)
    mu, sd = ref_f.mean(0), ref_f.std(0).clamp(min=1e-6)
    ref_f = (ref_f - mu) / sd
    ref_grad, ref_mass = raw_grad_mass(ref_raw)
    with torch.no_grad():
        sub = ref_f[np.random.RandomState(SEED + 7).choice(n_ref, min(256, n_ref), replace=False)]
        d2 = torch.cdist(sub, sub) ** 2
        bw = float(d2[d2 > 0].median().item())
    fidx = np.random.RandomState(SEED + 11).choice(train_data.N, 512, replace=False)
    floor_feat = (features(train_data.frames[fidx]) - mu) / sd
    mmd_floor = mmd2_sq(floor_feat, ref_f, bw)
    print(f"[data] ref feats={ref_f.shape} bw={bw:.3f} mmd_floor={mmd_floor:.6f} "
          f"ref_grad={ref_grad:.5f} ref_mass={ref_mass:.5f}", flush=True)

    print(f"[train] {TRAIN_STEPS} steps batch={BATCH}", flush=True)
    model, train_loss = train(train_data)

    print("[eval] rollout...", flush=True)
    mmd, sharp, mass_drift = evaluate(model, hold_data, ref_f, (mu, sd), bw, ref_grad, ref_mass)
    wall = time.time() - t0

    print(f"[done] rollout_mmd={mmd:.6f} sharpness={sharp:.4f} mass_drift={mass_drift:.4f} "
          f"train_loss={train_loss:.5f} mmd_floor={mmd_floor:.6f} wall={wall:.1f}s", flush=True)
    print(f"METRIC rollout_mmd={mmd:.6f}", flush=True)
    print(f"METRIC sharpness_ratio={sharp:.6f}", flush=True)
    print(f"METRIC mass_drift={mass_drift:.6f}", flush=True)
    print(f"METRIC train_loss={train_loss:.6f}", flush=True)
    print(f"METRIC mmd_floor={mmd_floor:.6f}", flush=True)
    print(f"METRIC wall_s={wall:.2f}", flush=True)


if __name__ == "__main__":
    main()
