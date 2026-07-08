"""
Block-Bridge Flow Matching for Video Forecasting (3-frame -> 3-frame)
=====================================================================

A DIFFERENT experiment from video_rollout_experiment.py (1-frame AR flow).
Here the model forecasts a BLOCK of F=3 future frames at once, and the flow
interpolation is a Gaussian / Brownian BRIDGE whose two endpoints are:

    * source (t=0): the K=3 CONTEXT frames   (the "three frame setup")
    * target (t=1): the NEXT 3 frames         (the "three next frame")

i.e. literally: a gaussian bridge that goes from your three frames to the
three next frames.

  z_t = (1-t) * source + t * target + c_t * eta ,
  c_t^2 = sigma^2 * t(1-t) + sigma_min^2          (min at both endpoints,
                                                    max mid-path at sigma^2/4)

The model is an x-prediction (endpoint) network: given the noisy block z_t,
the clean context block, and t, predict the clean target block.

WHY THIS MIGHT BE MORE STABLE THAN 1-FRAME AR
---------------------------------------------
At inference we do a STRIDE-F (here stride-3) block rollout: from 3 frames
predict the next 3, slide the window, predict the next 3, ... To cover the
same horizon the model runs only ceil(ROLLOUT_LEN / F) autoregressive
iterations instead of ROLLOUT_LEN. Fewer AR steps => less exposure-bias
drift compounding (the dominant failure mode of the 1-frame forecaster).
The bridge additionally drives endpoint variance -> sigma_min^2 ~ 0, so each
block lands cleanly on the manifold and little off-manifold residual is fed
to the next block.

METRIC (primary, lower is better): rollout_ed
  Energy distance between feature distributions of long stride-F rollout
  frames and a held-out training-frame reference (same metric family as the
  1-frame benchmark so the two are directly comparable). Secondary:
  ed_late, mmd, sharpness_ratio, mass_drift, train_loss, num_blocks, wall_s.

LEVERS (env vars)
  Bridge:   BRIDGE_SIGMA, BRIDGE_SIGMA_MIN, BRIDGE_WCLAMP, BRIDGE_LOSS
  Source:   FLOW_SOURCE (ctx|noise|repeatlast), SRC_NOISE
  Model:    MODEL_CH, MODEL_BLOCKS, K_CTX, BLOCK_F
  Train:    TRAIN_STEPS, BATCH, LR, AMP, GRAD_CLIP
  Aug:      TECHNIQUE (none|ctx_bridge), CTX_NOISE, CTX_LAST_ONLY
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib as _contextlib

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _env(name, default, cast=float):
    v = os.environ.get(name)
    return cast(v) if v is not None else default

SEED            = _env("SEED", 0, int)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DETERMINISTIC   = _env("DETERMINISTIC", 1, int)
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = not DETERMINISTIC
    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

IMG             = _env("IMG", 32, int)
C_CHAN          = _env("C_CHAN", 1, int)
K_CTX           = _env("K_CTX", 3, int)          # context frames (the "three frame setup")
BLOCK_F         = _env("BLOCK_F", 3, int)        # future frames per block (the "three next frame")
N_TRAJ_TRAIN    = _env("N_TRAJ_TRAIN", 220, int)
N_TRAJ_HOLD     = _env("N_TRAJ_HOLD", 40, int)
TRAJ_LEN        = _env("TRAJ_LEN", 80, int)
LORENZ_DT       = _env("LORENZ_DT", 0.024, float)
BLOB_SIGMA      = _env("BLOB_SIGMA", 2.2, float)

TRAIN_STEPS     = _env("TRAIN_STEPS", 2000, int)
BATCH           = _env("BATCH", 96, int)
LR              = _env("LR", 2e-3, float)
GRAD_CLIP       = _env("GRAD_CLIP", 1.0, float)
AMP             = _env("AMP", 1, int)            # 1=bf16 autocast (fast); 0=fp32
WARMUP_STEPS    = _env("WARMUP_STEPS", 0, int)
ODE_STEPS       = _env("ODE_STEPS", 32, int)
SAMPLE_AVG      = _env("SAMPLE_AVG", 1, int)     # average N noise-sample blocks per rollout step

MODEL_CH        = _env("MODEL_CH", 40, int)
MODEL_BLOCKS    = _env("MODEL_BLOCKS", 3, int)
TDIM            = _env("TDIM", 64, int)

# Flow path
FLOW            = os.environ.get("FLOW", "bridge")     # "bridge" | "rf"
BRIDGE_SIGMA    = _env("BRIDGE_SIGMA", 0.3, float)
BRIDGE_SIGMA_MIN = _env("BRIDGE_SIGMA_MIN", 1e-3, float)
BRIDGE_WCLAMP   = _env("BRIDGE_WCLAMP", 10.0, float)
BRIDGE_LOSS     = os.environ.get("BRIDGE_LOSS", "vloss")  # vloss | ivar | uniform
# Source endpoint (t=0) of the interpolation
FLOW_SOURCE     = os.environ.get("FLOW_SOURCE", "ctx")   # ctx|noise|repeatlast
SRC_NOISE       = _env("SRC_NOISE", 0.1, float)          # perturb the source at train+infer (exposure-bias fix)

# Context augmentation (conditioning-context robustness)
TECHNIQUE       = os.environ.get("TECHNIQUE", "none")    # none | ctx_bridge
CTX_NOISE       = _env("CTX_NOISE", 0.06, float)         # ctx_bridge gaussian std on the context
CTX_LAST_ONLY   = _env("CTX_LAST_ONLY", 1, int)          # 1=corrupt only the most-recent context slot

N_ROLLOUT       = _env("N_ROLLOUT", 24, int)
ROLLOUT_LEN     = _env("ROLLOUT_LEN", 50, int)


def _amp():
    if AMP:
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return _contextlib.nullcontext()


# --------------------------------------------------------------------------- #
# 1. DATA — Lorenz-driven 2D Gaussian blob video (block windows)
# --------------------------------------------------------------------------- #
def lorenz_trajectory(n_steps, dt, seed, s=10.0, r=28.0, b=8.0/3.0):
    rng = np.random.RandomState(seed)
    x, y, z = rng.uniform(-1, 1) + 0.0, rng.uniform(-1, 1) + 0.0, rng.uniform(20, 25) + 0.0
    xs = np.empty((n_steps, 3), dtype=np.float32)
    for i in range(n_steps):
        dx = s * (y - x); dy = x * (r - z) - y; dz = x * y - b * z
        x += dx * dt; y += dy * dt; z += dz * dt
        xs[i] = [x, y, z]
    return xs


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
    def __init__(self, side, device):
        coords = torch.arange(side, device=device).float() + 0.5
        self.grid = torch.stack(torch.meshgrid(coords, coords, indexing='ij'), -1)
        self.side = side
    def render(self, centers, amps, sigma):
        g = self.grid
        d2 = ((g[None] - centers[:, None, None, :]) ** 2).sum(-1)
        return amps[:, None, None] * torch.exp(-d2 / (2.0 * sigma * sigma))


class VideoSequenceData:
    """Stores all frames flat; samples (context block, target block) windows."""
    def __init__(self, n_traj, traj_len, seed, img, c_chan, blob_sigma, lorenz_dt):
        self.img = img; self.c_chan = c_chan; self.blob_sigma = blob_sigma
        renderer = _BlobRenderer(img, DEVICE)
        proj = _projections(c_chan)
        all_frames = []
        self.traj_starts = []
        idx = 0
        lo = np.array([-20.0, -30.0, 0.0]); hi = np.array([20.0, 30.0, 55.0])
        for t in range(n_traj):
            traj = lorenz_trajectory(traj_len, lorenz_dt, seed + t)
            normed = (traj - lo) / (hi - lo)
            tt = np.arange(traj_len) * lorenz_dt
            amp_master = 0.78 + 0.18 * np.sin(0.6 * tt + t).astype(np.float32)
            for i in range(traj_len):
                latent = normed[i]
                centers2d = proj @ latent
                centers2d = centers2d * (img * 0.18) + img * 0.5
                centers2d = np.clip(centers2d, 1.5, img - 1.5)
                amps = (amp_master[i] * (0.7 + 0.3 * np.arange(c_chan) / max(1, c_chan - 1))).astype(np.float32)
                ct = torch.tensor(centers2d, dtype=torch.float32, device=DEVICE)
                ap = torch.tensor(amps, dtype=torch.float32, device=DEVICE)
                all_frames.append(renderer.render(ct, ap, blob_sigma))
            self.traj_starts.append(idx)
            idx += traj_len
        self.n_traj = n_traj; self.traj_len = traj_len
        self.frames = torch.stack(all_frames, 0)  # (N, C, H, W)
        self.N = self.frames.shape[0]

    def sample_windows(self, batch_size, rng):
        """Returns ctx (B,K,C,H,W) and target (B,F,C,H,W) where target is the F
        frames immediately following the K context frames."""
        B = batch_size
        ti = rng.randint(0, self.n_traj, size=B)
        # context needs pi-K_CTX >= 0; target needs pi+BLOCK_F-1 <= traj_len-1.
        pi = rng.randint(K_CTX, self.traj_len - BLOCK_F + 1, size=B)
        base = np.array(self.traj_starts)[ti]
        idx_ctx = np.stack([base + pi - K_CTX + k for k in range(K_CTX)], axis=1)   # B,K
        idx_tgt = np.stack([base + pi + f for f in range(BLOCK_F)], axis=1)          # B,F
        ctx = self.frames[idx_ctx]      # B,K,C,H,W
        target = self.frames[idx_tgt]   # B,F,C,H,W
        return ctx, target


# --------------------------------------------------------------------------- #
# 2. MODEL — 3D-conv block bridge (x-prediction / endpoint)
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


class Conv3dFiLM(nn.Module):
    def __init__(self, ch, tdim, groups=4):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.film = nn.Linear(tdim, 2 * ch)
        self.act = nn.SiLU()
    def forward(self, x, temb):
        h = self.norm1(x)
        scale, shift = self.film(temb).view(temb.shape[0], -1, 1, 1, 1).chunk(2, dim=1)
        h = h * (1 + scale) + shift
        x = x + self.conv1(self.act(h))
        h = self.norm2(x)
        x = x + self.conv2(self.act(h))
        return x


class BlockBridgeNet(nn.Module):
    """Predicts the clean F-frame target block from (z_t block, context block, t).
    Input volume = concat[context (K frames), z_t (F frames)] along time ->
    (B, C, K+F, H, W). Output the last F time slices -> (B, F, C, H, W)."""
    def __init__(self, k_ctx, block_f, c_chan, ch=40, tdim=64, n_blocks=3):
        super().__init__()
        self.k_ctx = k_ctx
        self.block_f = block_f
        self.stem = nn.Conv3d(c_chan, ch, 3, padding=1)
        self.blocks = nn.ModuleList([Conv3dFiLM(ch, tdim) for _ in range(n_blocks)])
        self.temb = SinTime(tdim)
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.out_norm = nn.GroupNorm(4, ch)
        self.out_conv = nn.Conv3d(ch, c_chan, 3, padding=1)
        self.out_act = nn.SiLU()

    def forward(self, z_t, ctx, t):
        # z_t: (B,F,C,H,W); ctx: (B,K,C,H,W); t: (B,)
        temb = self.tmlp(self.temb(t))
        vol = torch.cat([ctx, z_t], dim=1)                 # (B, K+F, C, H, W)
        vol = vol.permute(0, 2, 1, 3, 4).contiguous()       # (B, C, K+F, H, W)
        x = self.stem(vol)
        for blk in self.blocks:
            x = blk(x, temb)
        x = self.out_act(self.out_norm(x))
        x = self.out_conv(x)                                # (B, C, K+F, H, W)
        x = x[:, :, self.k_ctx:, :, :]                      # target block slices (B,C,F,H,W)
        return x.permute(0, 2, 1, 3, 4).contiguous()        # (B,F,C,H,W)

    def get_velocity(self, z_t, ctx, t):
        x_pred = self.forward(z_t, ctx, t)
        return (x_pred - z_t) / (1.0 - t).clamp(min=0.01).view(-1, 1, 1, 1, 1)


# --------------------------------------------------------------------------- #
# 3. FLOW — bridge / rf interpolation, source selection, loss
# --------------------------------------------------------------------------- #
def bridge_coeffs(t, sigma, sigma_min):
    """c_t^2 = sigma^2 t(1-t) + sigma_min^2 ; returns (c_t, c'_t/c_t)."""
    t = torch.as_tensor(t, dtype=torch.float32, device=DEVICE)
    var = sigma ** 2 * t * (1.0 - t) + sigma_min ** 2
    c = torch.sqrt(var)
    cp_over_c = sigma ** 2 * (1.0 - 2.0 * t) / (2.0 * var + 1e-12)
    return c, cp_over_c


def make_source(ctx, shape):
    """t=0 endpoint of the interpolation as a (B,F,C,H,W) block.
    ctx: (B,K,C,H,W). 'ctx' = last F context frames (the literal 3->3 bridge),
    'repeatlast' = last context frame tiled F times, 'noise' = pure Gaussian."""
    if FLOW_SOURCE == "noise":
        return torch.randn(*shape, device=ctx.device)
    if FLOW_SOURCE == "repeatlast":
        return ctx[:, -1:].expand(-1, BLOCK_F, -1, -1, -1).contiguous()
    # "ctx"
    return ctx[:, -BLOCK_F:].contiguous()


def make_zt(t, source, target):
    """t: (B,). source/target: (B,F,C,H,W)."""
    ts = t.view(-1, 1, 1, 1, 1)
    mu_t = (1.0 - ts) * source + ts * target
    if FLOW == "bridge":
        eta = torch.randn_like(target)
        c_t, _ = bridge_coeffs(t, BRIDGE_SIGMA, BRIDGE_SIGMA_MIN)
        return mu_t + c_t.view(-1, 1, 1, 1, 1) * eta
    return mu_t


def flow_loss(model, z_t, target, ctx, t):
    """x-prediction flow-matching loss. Returns (loss_scalar, x_pred)."""
    x_pred = model(z_t, ctx, t)
    if FLOW == "rf":
        v_pred = (x_pred - z_t) / (1.0 - t).clamp(min=0.05).view(-1, 1, 1, 1, 1)
        v_target = (target - z_t) / (1.0 - t).clamp(min=0.05).view(-1, 1, 1, 1, 1)
        w = torch.ones_like(t)
        loss = ((v_pred - v_target) ** 2).mean(dim=(1, 2, 3, 4)) * w
        return loss.mean().float(), x_pred
    # bridge
    if BRIDGE_LOSS == "ivar":
        c_t, _ = bridge_coeffs(t, BRIDGE_SIGMA, BRIDGE_SIGMA_MIN)
        w = (1.0 / (c_t ** 2)).clamp(max=BRIDGE_WCLAMP)
    elif BRIDGE_LOSS == "uniform":
        w = torch.ones_like(t)
    else:  # vloss: (1 - t c'/c)^2 clamped (one-sided data-end upweight, no collapse)
        c_t, cp_over_c = bridge_coeffs(t, BRIDGE_SIGMA, BRIDGE_SIGMA_MIN)
        wf = 1.0 - t * cp_over_c
        w = (wf ** 2).clamp(max=BRIDGE_WCLAMP)
    loss = ((x_pred - target) ** 2).mean(dim=(1, 2, 3, 4)) * w
    return loss.mean().float(), x_pred


# --------------------------------------------------------------------------- #
# 3a. CONTEXT AUGMENTATION (optional conditioning-context robustness)
# --------------------------------------------------------------------------- #
def augment_context(ctx, training):
    """ctx: (B,K,C,H,W). training=False -> clean (decoupled inference)."""
    if not training or TECHNIQUE == "none":
        return ctx
    if TECHNIQUE == "ctx_bridge":
        noise = torch.randn_like(ctx) * CTX_NOISE
        if CTX_LAST_ONLY and ctx.shape[1] > 1:
            out = ctx.clone()
            out[:, -1] = ctx[:, -1] + noise[:, -1]
            return out
        return ctx + noise
    return ctx


# --------------------------------------------------------------------------- #
# 4. TRAINING
# --------------------------------------------------------------------------- #
def train(data):
    torch.manual_seed(SEED)
    model = BlockBridgeNet(K_CTX, BLOCK_F, C_CHAN, ch=MODEL_CH, tdim=TDIM,
                           n_blocks=MODEL_BLOCKS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.RandomState(SEED + 1)
    if WARMUP_STEPS > 0:
        warmup = WARMUP_STEPS; eta0 = LR
        def _lr(step):
            if step < warmup:
                return eta0 * (step + 1) / warmup
            prog = (step - warmup) / max(1, TRAIN_STEPS - warmup)
            return 0.5 * eta0 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_STEPS)
    last_loss = 0.0
    for step in range(TRAIN_STEPS):
        ctx, target = data.sample_windows(BATCH, rng)
        t = torch.rand(BATCH, device=DEVICE)
        source = make_source(ctx, target.shape)
        if SRC_NOISE > 0:
            source = source + SRC_NOISE * torch.randn_like(source)
        z_t = make_zt(t, source, target)
        ctx_aug = augment_context(ctx, training=True)
        with _amp():
            loss, _ = flow_loss(model, z_t, target, ctx_aug, t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        last_loss = loss.item()
        if (step + 1) % 500 == 0:
            print(f"  step {step+1:4d}/{TRAIN_STEPS}  loss={last_loss:.5f}", flush=True)
    return model, last_loss


# --------------------------------------------------------------------------- #
# 5. SAMPLING & STRIDE-F ROLLOUT
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _ode_integrate(model, source, ctx, n_ode):
    """Integrate the flow 0->1 over a block. Returns endpoint (B,F,C,H,W).
    Bridge: closed-form step (absorbs stiff c'/c into the c(t+dt)/c(t) ratio;
    lands on x_pred at t=1). RF: forward-Euler velocity."""
    Bn = source.shape[0]
    z = source
    dt = 1.0 / n_ode
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        if FLOW == "bridge":
            for i in range(n_ode):
                t_val = i * dt; t_next = t_val + dt
                t = torch.full((Bn,), t_val, device=DEVICE)
                x_pred = model(z, ctx, t).float()
                mu_t = (1.0 - t_val) * source + t_val * x_pred
                mu_next = (1.0 - t_next) * source + t_next * x_pred
                c_t, _ = bridge_coeffs(t_val, BRIDGE_SIGMA, BRIDGE_SIGMA_MIN)
                c_next, _ = bridge_coeffs(t_next, BRIDGE_SIGMA, BRIDGE_SIGMA_MIN)
                ratio = c_next / c_t
                z = mu_next + (z - mu_t) * ratio
        else:  # rf Euler
            for i in range(n_ode):
                t = torch.full((Bn,), i * dt, device=DEVICE)
                v = model.get_velocity(z, ctx, t)
                z = z + v * dt
    return z


@torch.no_grad()
def sample_block(model, ctx, n_ode, n_samples=1):
    """ctx: (B,K,C,H,W) -> predicted next block (B,F,C,H,W)."""
    B = ctx.shape[0]
    g = 1.0
    if n_samples > 1:
        ctx_rep = ctx.repeat_interleave(n_samples, dim=0)
        src = make_source(ctx_rep, (B * n_samples, BLOCK_F, C_CHAN, IMG, IMG))
        if SRC_NOISE > 0:
            src = src + SRC_NOISE * torch.randn_like(src)
        z = _ode_integrate(model, src, ctx_rep, n_ode)
        return z.view(B, n_samples, BLOCK_F, C_CHAN, IMG, IMG).mean(dim=1).float()
    src = make_source(ctx, (B, BLOCK_F, C_CHAN, IMG, IMG))
    if SRC_NOISE > 0:
        src = src + SRC_NOISE * torch.randn_like(src)
    return _ode_integrate(model, src, ctx, n_ode).float()


@torch.no_grad()
def rollout(model, data, n_rollout, rollout_len):
    """Stride-F block rollout. Returns (frames (T,B,C,H,W), n_blocks)."""
    rng = np.random.RandomState(SEED + 999)
    ti = rng.randint(0, data.n_traj, size=n_rollout)
    pi = rng.randint(K_CTX, data.traj_len - rollout_len - BLOCK_F, size=n_rollout)
    base = np.array(data.traj_starts)[ti]
    ctx_idx = np.stack([base + pi - K_CTX + k for k in range(K_CTX)], axis=1)
    ctx = data.frames[ctx_idx]                  # (B,K,C,H,W)
    cur = augment_context(ctx, training=False)  # clean context
    blocks = []
    n_blocks = int(math.ceil(rollout_len / BLOCK_F))
    for _ in range(n_blocks):
        block = sample_block(model, cur, ODE_STEPS, n_samples=SAMPLE_AVG)   # (B,F,C,H,W)
        blocks.append(block)
        cat = torch.cat([cur, block], dim=1)    # (B, K+F, C,H,W)
        cur = cat[:, -K_CTX:]                    # slide window by F
    all_frames = torch.cat(blocks, dim=1)       # (B, n_blocks*F, C,H,W)
    all_frames = all_frames[:, :rollout_len]     # truncate to exactly rollout_len
    return all_frames.permute(1, 0, 2, 3, 4).contiguous(), n_blocks  # (T,B,C,H,W)


# --------------------------------------------------------------------------- #
# 6. METRICS
# --------------------------------------------------------------------------- #
def features(vols):
    pooled = F.adaptive_avg_pool2d(vols, (6, 6)).flatten(1)
    gx, gy = torch.gradient(vols, dim=(2, 3))
    grad_e = (gx ** 2 + gy ** 2).mean(dim=(1, 2, 3)).unsqueeze(1)
    vp = F.pad(vols, (1, 1, 1, 1), mode='replicate')
    l = (vp[..., 2:, 1:-1] + vp[..., :-2, 1:-1] + vp[..., 1:-1, 2:] + vp[..., 1:-1, :-2]
         - 4 * vp[..., 1:-1, 1:-1])
    lap_e = (l ** 2).mean(dim=(1, 2, 3)).unsqueeze(1)
    mass = vols.mean(dim=(1, 2, 3)).unsqueeze(1)
    maxv = vols.amax(dim=(1, 2, 3)).unsqueeze(1)
    return torch.cat([pooled, grad_e, lap_e, mass, maxv], dim=1)


def energy_distance(a, b):
    na, nb = a.shape[0], b.shape[0]
    D_ab = torch.cdist(a, b).mean()
    if na > 1:
        D_aa = torch.cdist(a, a)
        D_aa = D_aa[torch.triu_indices(na, na, 1).unbind()].mean()
    else:
        D_aa = torch.zeros((), device=a.device)
    if nb > 1:
        D_bb = torch.cdist(b, b)
        D_bb = D_bb[torch.triu_indices(nb, nb, 1).unbind()].mean()
    else:
        D_bb = torch.zeros((), device=a.device)
    return float((2 * D_ab - D_aa - D_bb).clamp(min=0.0))


def mmd2_sq(a, b, bw):
    aa = torch.cdist(a, a) ** 2; bb = torch.cdist(b, b) ** 2; ab = torch.cdist(a, b) ** 2
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
    rollout_frames, n_blocks = rollout(model, data, N_ROLLOUT, ROLLOUT_LEN)  # T,B,C,H,W
    flat = rollout_frames.flatten(0, 1)            # (T*B, C,H,W)
    rf = flat[:1024]
    mu, sd = ref_stats
    rf_feat = (features(rf) - mu) / sd
    ed = energy_distance(rf_feat, ref_feats)
    mmd = mmd2_sq(rf_feat, ref_feats, bw)
    half = rollout_frames.shape[0] // 2
    late = rollout_frames[half:].flatten(0, 1)[:1024]
    late_feat = (features(late) - mu) / sd
    ed_late = energy_distance(late_feat, ref_feats)
    ge_roll, mass_roll = raw_grad_mass(rf)
    sharp = ge_roll / max(ref_grad, 1e-9)
    mass_drift = abs(mass_roll - ref_mass) / max(abs(ref_mass), 1e-9)
    return ed, mmd, ed_late, sharp, mass_drift, n_blocks


# --------------------------------------------------------------------------- #
# 7. MAIN
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"BLOCK-BRIDGE  K_CTX={K_CTX} BLOCK_F={BLOCK_F} FLOW={FLOW} FLOW_SOURCE={FLOW_SOURCE} "
          f"SRC_NOISE={SRC_NOISE} BRIDGE_LOSS={BRIDGE_LOSS} BRIDGE_SIGMA={BRIDGE_SIGMA} "
          f"BRIDGE_WCLAMP={BRIDGE_WCLAMP} MODEL_CH={MODEL_CH} BLOCKS={MODEL_BLOCKS} "
          f"IMG={IMG} C_CHAN={C_CHAN} LORENZ_DT={LORENZ_DT} device={DEVICE}", flush=True)

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
    ed_floor = energy_distance(floor_feat, ref_f)
    print(f"[data] ref feats={ref_f.shape} bw={bw:.3f} ed_floor={ed_floor:.6f} "
          f"ref_grad={ref_grad:.5f} ref_mass={ref_mass:.5f}", flush=True)

    print(f"[train] {TRAIN_STEPS} steps batch={BATCH}", flush=True)
    model, train_loss = train(train_data)

    print("[eval] stride-F block rollout...", flush=True)
    ed, mmd, ed_late, sharp, mass_drift, n_blocks = evaluate(
        model, hold_data, ref_f, (mu, sd), bw, ref_grad, ref_mass)
    wall = time.time() - t0

    print(f"[done] rollout_ed={ed:.6f} ed_late={ed_late:.6f} mmd={mmd:.6f} "
          f"sharpness={sharp:.4f} mass_drift={mass_drift:.4f} "
          f"train_loss={train_loss:.5f} n_blocks={n_blocks} wall={wall:.1f}s", flush=True)
    print(f"METRIC rollout_ed={ed:.6f}", flush=True)
    print(f"METRIC ed_late={ed_late:.6f}", flush=True)
    print(f"METRIC mmd={mmd:.6f}", flush=True)
    print(f"METRIC sharpness_ratio={sharp:.6f}", flush=True)
    print(f"METRIC mass_drift={mass_drift:.6f}", flush=True)
    print(f"METRIC train_loss={train_loss:.6f}", flush=True)
    print(f"METRIC ed_floor={ed_floor:.6f}", flush=True)
    print(f"METRIC num_blocks={n_blocks}", flush=True)
    print(f"METRIC wall_s={wall:.2f}", flush=True)


if __name__ == "__main__":
    main()
