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
DETERMINISTIC   = _env("DETERMINISTIC", 1, int)  # 1: deterministic CUDA for reproducible runs
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = not DETERMINISTIC
    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
IMG             = _env("IMG", 32, int)           # image side (H=W)
C_CHAN          = _env("C_CHAN", 1, int)         # channels per frame
K_CTX           = _env("K_CTX", 2, int)          # number of context frames
N_TRAJ_TRAIN    = _env("N_TRAJ_TRAIN", 220, int)
N_TRAJ_HOLD     = _env("N_TRAJ_HOLD", 40, int)
TRAJ_LEN        = _env("TRAJ_LEN", 80, int)      # frames per trajectory
LORENZ_DT       = _env("LORENZ_DT", 0.012, float)
N_BLOBS         = _env("N_BLOBS", 0, int)       # extra INDEPENDENT blobs (own Lorenz traj) summed into ch0 - overfitting test
BLOB_SIGMA      = _env("BLOB_SIGMA", 2.2, float) # pixel width
# training
TRAIN_STEPS     = _env("TRAIN_STEPS", 2000, int)
BATCH           = _env("BATCH", 96, int)
LR              = _env("LR", 2e-3, float)
GRAD_CLIP       = _env("GRAD_CLIP", 1.0, float)    # max grad norm (ViT/jit3d needs tighter, e.g. 0.5, vs conv default 1.0)
AMP             = _env("AMP", 1, int)            # 1=bf16 autocast (fast); 0=fp32 (stable for jit3d, which diverges under bf16+200x loss weight)
RF_WCLAMP       = _env("RF_WCLAMP", 200.0, float)  # clamp on the 1/(1-t)^2 RF velocity loss weight; lower (e.g. 50) trades high-t emphasis for ViT optimization stability
WARMUP_STEPS    = _env("WARMUP_STEPS", 0, int)    # linear LR warmup (helps ViT/jit3d optimization stability); 0 = off (cosine from full LR)
ODE_STEPS       = _env("ODE_STEPS", 16, int)     # Euler substeps for sampling
ARCH            = os.environ.get("ARCH", "conv2d")   # conv2d (channel-concat) | conv3d (3D-conv context encoder) | jit3d (production JiT-3D ViT)
LATENT_BLUR     = _env("LATENT_BLUR", 0.0, float)    # ARCH=conv3d: sigma for blurring the CONTEXT LATENT in feature space (inside the net) during training
MODEL_CH        = _env("MODEL_CH", 48, int)       # conv channel width of the forecaster
MODEL_BLOCKS    = _env("MODEL_BLOCKS", 4, int)     # number of conv blocks
# --- JiT-3D (production architecture) knobs; only used when ARCH=jit3d ---
FLASHNET_PATH   = os.environ.get("FLASHNET_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flashnet"))
JIT_EMBED_DIM   = _env("JIT_EMBED_DIM", 128, int)   # ViT embedding dim (must be divisible by JIT_HEADS)
JIT_DEPTH       = _env("JIT_DEPTH", 4, int)         # number of transformer blocks
JIT_HEADS       = _env("JIT_HEADS", 4, int)          # attention heads
JIT_PATCH_T     = _env("JIT_PATCH_T", 1, int)        # temporal patch size
JIT_PATCH_HW    = _env("JIT_PATCH_HW", 4, int)       # spatial patch size (H=W)
JIT_MLP_RATIO   = _env("JIT_MLP_RATIO", 2.6, float)  # SwiGLU hidden ratio
# JiT-3D built-in latent-context corruptor (an ALTERNATIVE/COMPLEMENTARY aug to
# augment_context; the autoresearch lever remains augment_context). Default off.
JIT_CORRUPT_PROB   = _env("JIT_CORRUPT_PROB", 0.0, float)
JIT_CORRUPT_EMBED  = _env("JIT_CORRUPT_EMBED", 0.10, float)
JIT_CORRUPT_BLOCK0 = _env("JIT_CORRUPT_BLOCK0", 0.05, float)
NOISE_INJECT    = _env("NOISE_INJECT", 0.0, float)    # >0: stochastic RF sampler - re-noise z during ODE proportional to remaining noise level (regularizes deterministic drift)
SAMPLE_AVG      = _env("SAMPLE_AVG", 1, int)        # >1: average this many noise-sample predictions per AR rollout step (variance reduction -> less drift compounding)
# --- Flow path: linear rectified flow (default) or Brownian-bridge flow matching ---
# Bridge: z_t = (1-t) eps + t y + c_t eta,  c_t^2 = sigma^2 t(1-t) + sigma_min^2.
# Variance is minimal at BOTH endpoints (sigma_min^2) and maximal mid-path (sigma^2/4
# at t=0.5). Motivation for AR forecasting: residual sampling jitter at the data
# endpoint (t=1) is fed forward and accumulates as off-manifold drift; the bridge
# drives endpoint variance -> 0 so each AR step lands cleanly on the manifold.
# (Ref: Lim et al. 2024, arXiv:2410.03229; see hypersphere_bridge_experiment.py.)
FLOW            = os.environ.get("FLOW", "rf")     # "rf" (linear) | "bridge" (Brownian-bridge flow matching)
BRIDGE_SIGMA    = _env("BRIDGE_SIGMA", 0.5, float)        # bridge: Brownian perturbation scale (var at t=0.5 = sigma^2/4)
BRIDGE_SIGMA_MIN = _env("BRIDGE_SIGMA_MIN", 1e-3, float) # bridge: residual endpoint variance (smaller -> sharper landing)
BRIDGE_WCLAMP   = _env("BRIDGE_WCLAMP", 200.0, float)     # bridge: clamp on the loss weight (BRIDGE_LOSS=ivar -> 1/c_t^2; =vloss -> (1-t c'/c)^2)
BRIDGE_LOSS     = os.environ.get("BRIDGE_LOSS", "vloss")   # bridge loss: "vloss" (velocity-matching ||v-u_t||^2 via x-pred = (x_pred-y)^2 (1-t c'/c)^2 clamped; one-sided data-end upweight, prevents collapse) | "ivar" (inverse-cond-var 1/c_t^2, upweights BOTH endpoints) | "uniform" (plain x-pred MSE; collapses to context-pred on easy data)
# --- Context augmentation: Gaussian/Brownian-bridge-shaped perturbation on context ---
# ctx_bridge corrupts context frames with noise whose std follows the SAME bridge
# variance schedule used for the target: c_s^2 = sigma^2 s(1-s) + sigma_min^2.
# Rationale: train the model on context degraded by the bridge's own noise shape so
# it is robust to AR context drift WITHOUT the blur-collapse that killed manifold_noise
# on bridge (bridge already regularizes the target; this matches its noise structure).
# Inference is decoupled (clean context) per the established AR-stability principle.
CTX_BRIDGE_SIGMA     = _env("CTX_BRIDGE_SIGMA", BRIDGE_SIGMA, float)   # context bridge perturbation scale (var at s=0.5 = sigma^2/4)
CTX_BRIDGE_SIGMA_MIN = _env("CTX_BRIDGE_SIGMA_MIN", BRIDGE_SIGMA_MIN, float)  # context bridge residual endpoint std
CTX_BRIDGE_SCALE     = _env("CTX_BRIDGE_SCALE", 1.0, float)   # overall multiplier on the context bridge noise std (sweep magnitude)
CTX_BRIDGE_TIME      = os.environ.get("CTX_BRIDGE_TIME", "sample")  # "sample" (s~U(0,1), full bridge variance range) | "peak" (fixed s=0.5, max variance) | "uniform" (plain gaussian std=CTX_BRIDGE_SIGMA*SCALE, isolates the schedule effect) | "randsigma" (random NON-CONDITIONAL sigma~U(0,SCALE) per sample; noise std itself uniform, not bridge-schedule)
CTX_BRIDGE_LAST_ONLY = _env("CTX_BRIDGE_LAST_ONLY", 1, int)  # 1=corrupt only the most-recent context frame (the slot holding the model's own output at inference); 0=all frames
CTX_BRIDGE_ANNEAL   = _env("CTX_BRIDGE_ANNEAL", 0.0, float)  # >0: linearly decay the context noise magnitude to 0 over this fraction of training (e.g. 0.8 -> noise off by 80% of TRAIN_STEPS, clean refinement after). Rationale: ctx_bridge is an accelerator the model outgrows; annealing captures early-regularization benefit while avoiding the late tax. 0=off (static)
# --- Flow SOURCE (t=0 endpoint of the interpolation) ---
# "noise" (default): Gaussian source z_0~N(0,1) -> generate target from scratch.
# "lastctx": flow from the LAST CONTEXT FRAME -> target (delta/residual flow). The
#   model learns the MOTION (target - last_frame) instead of generating from noise; at
#   inference each AR step starts from the previous prediction, bounding drift accumulation.
#   REQUIRES SRC_NOISE>0 at training (exposure-bias fix: inference source is the model's
#   DRIFTED prediction, so the model must train on perturbed sources to not amplify drift).
FLOW_SOURCE     = os.environ.get("FLOW_SOURCE", "noise")
SRC_NOISE       = _env("SRC_NOISE", 0.0, float)   # lastctx: source perturbation std at TRAIN (eps=last+SRC_NOISE*randn) AND inference (per-sample diversity for SAMPLE_AVG). 0=pure deterministic (exposure-biased, unstable).
# --- JOINT context-generation(RF) + future-prediction(bridge) objective ---
# NEW (user idea): at TRAIN time the model jointly (a) DENOISES the context frames
# via a LINEAR rectified-flow interpolant (z_ctx=(1-s)eps+s*ctx -> learns to
# GENERATE valid data from scratch) and (b) PREDICTS the future frame via the
# Brownian-bridge interpolant (z_t=(1-t)eps+t*tgt+c_t eta -> learns dynamics).
# The model sees [noisy-ctx..., noisy-future] and outputs clean frames for BOTH.
# Hypothesis: the RF context-reconstruction head teaches a strong generative prior
# of on-manifold frames, and training on (lightly) RF-noised context robustifies
# the forecaster against AR context drift -> more stable long rollouts. At INFERENCE
# the rollout is UNCHANGED (predict future from clean context); the context-gen
# head is a training-only auxiliary. The context RF time s is sampled CLOSER to
# data than the future time t (s>=t, "generate the data first, then predict"),
# because at inference the context frames are the model's own already-refined
# outputs (near-clean). Decoupling optionally passes BOTH times to the net.
JOINT_GEN       = _env("JOINT_GEN", 0, int)               # 1=enable joint RF-ctx + bridge-fut objective (jit3d only; sets context_dim=2 if decoupled)
JOINT_CTX_WEIGHT= _env("JOINT_CTX_WEIGHT", 1.0, float)    # weight on the context RF reconstruction loss term (relative to the bridge future loss)
JOINT_COUPLE    = _env("JOINT_COUPLE", 1, int)            # 1: context time s = min(1, t + JOINT_LEAD) (context leads toward data); 0: s ~ U(JOINT_CTX_TMIN, 1) independent
JOINT_LEAD      = _env("JOINT_LEAD", 0.3, float)          # COUPLE=1: how far AHEAD of t the context time is (larger -> less context noise / closer to data)
JOINT_CTX_TMIN  = _env("JOINT_CTX_TMIN", 0.0, float)      # COUPLE=0: context RF time s ~ U(JOINT_CTX_TMIN, 1). Higher -> less context noise (closer to data)
JOINT_DECOUPLE  = _env("JOINT_DECOUPLE", 1, int)          # 1: pass BOTH [s, t] to the net (context_dim=2); 0: pass only t (context noise level hidden from the net)
JOINT_LAST_ONLY = _env("JOINT_LAST_ONLY", 0, int)         # 1: RF-noise only the most-recent context slot (holds model's own output at inference); 0=all K slots
# Context interpolant: "rf" (linear rectified flow, generate-from-scratch) | "bridge"
# (Brownian bridge, same clean-endpoint-landing family as the future). Bridge context
# adds mid-path Brownian noise c_s eta AND uses the one-sided vloss weight that forces
# sharp data-end reconstruction (the RF variant mean-pulls at high noise -> blur).
JOINT_CTX_FLOW        = os.environ.get("JOINT_CTX_FLOW", "rf")        # "rf" | "bridge" : context-slot interpolant
JOINT_CTX_BRIDGE_SIGMA= _env("JOINT_CTX_BRIDGE_SIGMA", 0.3, float)   # bridge-context: mid-path Brownian scale (var at s=0.5 = sigma^2/4)
JOINT_CTX_BRIDGE_SIGMA_MIN = _env("JOINT_CTX_BRIDGE_SIGMA_MIN", 1e-3, float) # bridge-context: residual endpoint variance
# rollout eval
N_ROLLOUT       = _env("N_ROLLOUT", 24, int)
ROLLOUT_LEN     = _env("ROLLOUT_LEN", 50, int)
# technique knobs
TECHNIQUE       = os.environ.get("TECHNIQUE", "pixnoise")
SIGMA           = _env("SIGMA", 0.40, float)     # pixel-noise std
BLUR_SIGMA      = _env("BLUR_SIGMA", 1.2, float) # 2D gaussian blur std
SELFFEED_PROB   = _env("SELFFEED_PROB", 0.25, float)
SELFFEED_ODE_STEPS = _env("SELFFEED_ODE_STEPS", 0, int)  # >0: cheap surrogate ODE steps for the self-generated prediction (emulates drift; only needs a ROUGH pred, e.g. 6-8 not the full ODE_STEPS=32 -> makes self-feed feasible on the ViT). 0=use full ODE_STEPS (accurate but ~13x slower).
SELFFEED_GATE   = _env("SELFFEED_GATE", 0.0, float)   # >0: error-gate self-feed (relative quantile kept, e.g. 0.5=keep low-error half)
SELFFEED_GATE_ABS = _env("SELFFEED_GATE_ABS", 0.0, float)  # >0: ABSOLUTE gate - keep if surrogate err < this * frame_variance (scale-invariant, regime-aware)
SPECTRAL_W      = _env("SPECTRAL_W", 1e-2, float)
MANIFOLD_BLUR_FRAC = _env("MANIFOLD_BLUR_FRAC", 0.6, float)  # manifold_noise: blur = BLUR_SIGMA * this
MANIFOLD_JITTER  = _env("MANIFOLD_JITTER", 0.08, float)      # manifold_noise: amplitude jitter std
MANIFOLD_BLUR_RAND = _env("MANIFOLD_BLUR_RAND", 0.5, float)  # manifold_noise: per-sample blur uniform jitter (0=fixed)
MANIFOLD_LAST_ONLY = _env("MANIFOLD_LAST_ONLY", 0, int)  # manifold_noise: 1=corrupt only the last (most-recent) context frame (the slot that holds the model's own output at inference); 0=all frames
MANIFOLD_ANNEAL   = _env("MANIFOLD_ANNEAL", 0.0, float)   # manifold_noise: cosine-anneal blur frac to this fraction of initial over training (0=off, e.g. 0.2 -> decay to 20%)
MS_WEIGHT       = _env("MS_WEIGHT", 1.0, float)     # weight on the multi-step rollout loss term
DIFFFORCE_P     = _env("DIFFFORCE_P", 0.5, float)
MS_PROB         = _env("MS_PROB", 0.3, float)    # prob of a 2-step rollout loss term (selffeed_ms)
# VAE/AE-latent context corruption (simulates rollout drift without model self-outputs)
VAE_LATENT      = _env("VAE_LATENT", 32, int)
VAE_EPOCHS      = _env("VAE_EPOCHS", 12, int)
VAE_NOISE       = _env("VAE_NOISE", 0.5, float)   # latent noise magnitude (in latent-std units)
VAE_BETA        = _env("VAE_BETA", 1e-3, float)   # KL weight (small -> sharp recon; ~0 = autoencoder)
# Classifier-free guidance on the context (overlay; applies to any technique)
UNCOND_PROB     = _env("UNCOND_PROB", 0.0, float)   # train: prob of dropping context (unconditional)
GUIDANCE        = _env("GUIDANCE", 1.0, float)     # infer: v = v_uncond + GUIDANCE*(v_cond - v_uncond)
_VAE = None   # global trained corruptor (vae, latent_std), set in main
_STEP = 0       # current training step (for annealing schedules)
import contextlib as _contextlib
def _amp():
    """autocast context: bf16 if AMP=1 (default, fast), else fp32 (jit3d-stable)."""
    if AMP:
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return _contextlib.nullcontext()
MS_PROB         = _env("MS_PROB", 0.3, float)    # prob of a 2-step rollout loss term (selffeed_ms)


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
            # extra INDEPENDENT blobs (each its own Lorenz trajectory) for the
            # multi-feature generalization test (N_BLOBS>0)
            extra = []
            for b in range(N_BLOBS):
                et = lorenz_trajectory(traj_len, lorenz_dt, seed + 100000*(b+1) + t)
                extra.append((et - lo) / (hi - lo))
            tt = np.arange(traj_len) * lorenz_dt
            amp_master = 0.78 + 0.18 * np.sin(0.6 * tt + t).astype(np.float32)
            for i in range(traj_len):
                latent = normed[i]  # (3,)
                centers2d = proj @ latent  # (C,2)
                centers2d = centers2d * (img * 0.18) + img * 0.5
                centers2d = np.clip(centers2d, 1.5, img - 1.5)
                amps = (amp_master[i] * (0.7 + 0.3 * np.arange(c_chan) / max(1, c_chan - 1))).astype(np.float32)
                ct = torch.tensor(centers2d, dtype=torch.float32, device=DEVICE)
                ap = torch.tensor(amps, dtype=torch.float32, device=DEVICE)
                frame = renderer.render(ct, ap, blob_sigma)  # (C,H,W)
                if extra:
                    # add independent blobs into channel 0 with random projections
                    for b, et in enumerate(extra):
                        c2 = (np.array([[0.9, 0.2, -0.1], [0.1, 0.8, 0.2]]) * (b + 1) @ et[i])
                        c2 = np.clip(c2 * (img * 0.18) + img * 0.5, 1.5, img - 1.5)
                        ct2 = torch.tensor(c2.reshape(1, 2), dtype=torch.float32, device=DEVICE)
                        ap2 = torch.tensor([0.7 - 0.1 * b], dtype=torch.float32, device=DEVICE)
                        frame[0] = frame[0] + renderer.render(ct2, ap2, blob_sigma)[0]
                    frame = frame.clamp(max=1.5)
                all_frames.append(frame)
            self.traj_starts.append(idx)
            idx += traj_len
        self.n_traj = n_traj
        self.traj_len = traj_len
        self.frames = torch.stack(all_frames, 0)  # (N, C, H, W)
        self.N = self.frames.shape[0]

    def sample_windows(self, batch_size, rng):
        """Returns (ctx [B,K,C,H,W], target [B,C,H,W], extra [B,C,H,W], tgt2 [B,C,H,W]).
        tgt2 = frame at pi+1 (for multi-step rollout loss)."""
        B = batch_size
        ti = rng.randint(0, self.n_traj, size=B)
        pi = rng.randint(K_CTX, self.traj_len - 2, size=B)  # leave room for tgt2=pi+1
        base = np.array(self.traj_starts)[ti]
        idx_ctx = np.stack([base + pi - K_CTX + k for k in range(K_CTX)], axis=1)  # B,K
        idx_tgt = base + pi
        idx_extra = base + pi - K_CTX - 1
        idx_tgt2 = base + pi + 1
        ctx = self.frames[idx_ctx]            # B,K,C,H,W
        tgt = self.frames[idx_tgt]            # B,C,H,W
        extra = self.frames[idx_extra]        # B,C,H,W
        tgt2 = self.frames[idx_tgt2]          # B,C,H,W
        return ctx, tgt, extra, tgt2


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
# 2a-alt. VideoRF3D — 3D-conv CONTEXT ENCODER + in-network CONTEXT-LATENT BLUR
#   Treats the K context frames as an explicit temporal volume (3D conv over
#   time x H x W) instead of flattening them into channels. This yields a
#   distinct CONTEXT LATENT volume that can be blurred IN FEATURE SPACE (inside
#   the network) to emulate rollout drift on high-level features rather than raw
#   pixels. Applicable to the user's weather model (4 ctx frames -> 3D conv).
# --------------------------------------------------------------------------- #
class Conv3dBlock(nn.Module):
    def __init__(self, ch, groups=4):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.act = nn.SiLU()
    def forward(self, x):
        x = x + self.conv1(self.act(self.norm1(x)))
        x = x + self.conv2(self.act(self.norm2(x)))
        return x


class VideoRF3D(nn.Module):
    """3D-conv context encoder + 2D FiLM decoder. The context latent volume can
    be blurred in feature space during training (latent_blur_sigma>0)."""
    def __init__(self, k_ctx, c_chan, ch=48, tdim=64, n_blocks=4, latent_blur_sigma=0.0):
        super().__init__()
        self.k_ctx = k_ctx
        self.c_chan = c_chan
        self.ch = ch
        self.latent_blur_sigma = latent_blur_sigma
        # context encoder: input (B, C, K, H, W) -> latent (B, ch, K, H, W)
        self.ctx_stem = nn.Conv3d(c_chan, ch, 3, padding=1)
        self.ctx_blocks = nn.ModuleList([Conv3dBlock(ch) for _ in range(2)])
        # collapse the temporal axis K -> 1 (let the net learn how to fuse frames)
        self.ctx_merge = nn.Conv3d(ch, ch, kernel_size=(k_ctx, 1, 1))
        # z path
        self.z_stem = nn.Conv2d(c_chan, ch, 3, padding=1)
        # fuse context + z features
        self.fuse = nn.Conv2d(ch * 2, ch, 3, padding=1)
        # 2D FiLM decoder (reuses ConvBlock)
        self.blocks = nn.ModuleList([ConvBlock(ch, tdim) for _ in range(n_blocks)])
        self.temb = SinTime(tdim)
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.out_norm = nn.GroupNorm(4, ch)
        self.out_conv = nn.Conv2d(ch, c_chan, 3, padding=1)
        self.out_act = nn.SiLU()

    def _blur_latent(self, feat):
        # feat: (B, ch, K, H, W) -> 2D Gaussian blur per (ch,K) slice, in feature space
        B, C, K, H, W = feat.shape
        flat = feat.reshape(B * C * K, 1, H, W)
        bl = _blur2d(flat, self.latent_blur_sigma)
        return bl.reshape(B, C, K, H, W)

    def forward(self, z_t, ctx, t):
        # z_t: (B,C,H,W); ctx: (B,K,C,H,W); t: (B,)
        temb = self.tmlp(self.temb(t))
        ctx5 = ctx.permute(0, 2, 1, 3, 4).contiguous()  # B,C,K,H,W
        h = self.ctx_stem(ctx5)
        for blk in self.ctx_blocks:
            h = blk(h)
        if self.training and self.latent_blur_sigma > 0:
            h = self._blur_latent(h)
        h = self.ctx_merge(h)            # B,ch,1,H,W
        ctx_feat = h.squeeze(2)          # B,ch,H,W
        z_feat = self.z_stem(z_t)        # B,ch,H,W
        x = self.fuse(torch.cat([ctx_feat, z_feat], dim=1))
        for blk in self.blocks:
            x = blk(x, temb)
        x = self.out_act(self.out_norm(x))
        return self.out_conv(x)

    def get_velocity(self, z_t, ctx, t):
        x_pred = self.forward(z_t, ctx, t)
        return (x_pred - z_t) / (1.0 - t).clamp(min=0.01).view(-1, 1, 1, 1)


# --------------------------------------------------------------------------- #
# 2a-alt2. VideoRFJiT3D — production JiT-3D Vision Transformer.
#   Wraps the actual production model
#   (../flashnet/meteolibre_model/models/jit3d.py:JiT3D_Modern) behind the same
#   (z_t, ctx, t) -> x_pred (B,C,H,W) interface as VideoRF/VideoRF3D, so the rest
#   of the benchmark (training, sampling, rollout, metrics, augment_context) is
#   unchanged. The K context frames + the noisy target z_t are stacked along the
#   time axis into a (B, C, K+1, H, W) volume; the model predicts all frames and
#   we keep only the last (target) frame. Carries the production model's built-in
#   LatentContextCorruptor (controllable via JIT_CORRUPT_* env vars; off by
#   default so augment_context remains the AR-stability lever).
# --------------------------------------------------------------------------- #
def _import_jit3d():
    import sys as _sys
    if _sys.path[0] != FLASHNET_PATH:
        _sys.path.insert(0, FLASHNET_PATH)
    from meteolibre_model.models.jit3d import JiT3D_Modern  # noqa: WPS433
    return JiT3D_Modern


class VideoRFJiT3D(nn.Module):
    """Adapter: rectified-flow endpoint predictor backed by the production
    JiT-3D ViT. Exposes forward(z_t, ctx, t) -> x_pred and get_velocity like the
    conv backbones. Optionally supports context_dim=2 (JOINT_DECOUPLE) so the net
    can be conditioned on BOTH a context time and a future time, plus forward_joint
    which returns clean-frame predictions for the context slots AND the future slot
    (used by the joint RF-ctx-generation + bridge-fut-prediction objective)."""
    def __init__(self, k_ctx, c_chan, img,
                 embed_dim=128, depth=4, num_heads=4,
                 patch_t=1, patch_hw=4, mlp_ratio=2.6,
                 corrupt_prob=0.0, corrupt_embed=0.10, corrupt_block0=0.05,
                 context_dim=1):
        super().__init__()
        JiT3D_Modern = _import_jit3d()
        self.k_ctx = k_ctx
        self.c_chan = c_chan
        self.context_dim = context_dim
        T = k_ctx + 1  # K context frames + 1 noisy target frame
        assert img % patch_hw == 0, f"IMG={img} must be divisible by JIT_PATCH_HW={patch_hw}"
        assert T % patch_t == 0, f"T={T} must be divisible by JIT_PATCH_T={patch_t}"
        assert embed_dim % num_heads == 0, "JIT_EMBED_DIM must be divisible by JIT_HEADS"
        self.jit = JiT3D_Modern(
            img_size=(T, img, img),
            patch_size=(patch_t, patch_hw, patch_hw),
            in_channels=c_chan,
            out_channels=c_chan,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            context_dim=context_dim,          # 1 (RF time only) | 2 (context-time + future-time for JOINT_DECOUPLE)
            time_emb_dim=64,
            n_context_frames=k_ctx,
            corruption_prob=corrupt_prob,
            embed_noise_scale=corrupt_embed,
            block0_noise_scale=corrupt_block0,
        )
        # patch the JiT block MLP ratio if a custom value was requested
        if mlp_ratio != 2.6:
            for blk in self.jit.blocks:
                hidden = int(embed_dim * mlp_ratio)
                blk.mlp = type(blk.mlp)(embed_dim, hidden, embed_dim)

    def _stack_volume(self, z_ctx, z_t):
        # z_ctx: (B,K,C,H,W) (may be noised or clean), z_t: (B,C,H,W) -> (B,C,K+1,H,W)
        vol_t = torch.cat([z_ctx, z_t.unsqueeze(1)], dim=1)        # (B, K+1, C, H, W)
        return vol_t.permute(0, 2, 1, 3, 4).contiguous()           # (B, C, K+1, H, W)

    def _cond(self, t_fut, t_ctx, B):
        # build the (B, context_dim) conditioning vector. If context_dim==2 the
        # FIRST slot is the context time, the LAST is the future time (JiT3D uses
        # t[:,-1] as the RF/bridge time and t[:,:-1] as extra context).
        if self.context_dim == 2:
            return torch.stack([t_ctx.view(B).float(), t_fut.view(B).float()], dim=1)
        return t_fut.view(B, 1).float()

    def forward(self, z_t, ctx, t):
        # z_t: (B,C,H,W); ctx: (B,K,C,H,W); t: (B,). Inference / single-time path:
        # context is assumed CLEAN -> context time = 1 (fully denoised) when decoupled.
        B = z_t.shape[0]
        vol = self._stack_volume(ctx, z_t)
        s = torch.ones(B, device=z_t.device, dtype=torch.float32)  # clean context
        t_in = self._cond(t, s, B)
        out = self.jit(vol, t_in)                        # (B, C, K+1, H, W)
        x_pred = out[:, :, self.k_ctx:, :, :]           # keep target slice
        return x_pred.flatten(1, 2)                     # (B, C, H, W)

    def forward_joint(self, z_ctx, z_t, t_fut, t_ctx):
        # TRAIN-time joint path: noised context (RF @ t_ctx) + noised future (bridge @ t_fut).
        # Returns (ctx_pred (B,K,C,H,W), fut_pred (B,C,H,W)) — clean-frame predictions.
        B = z_t.shape[0]
        vol = self._stack_volume(z_ctx, z_t)
        t_in = self._cond(t_fut, t_ctx, B)
        out = self.jit(vol, t_in)                        # (B, C, K+1, H, W)
        ctx_pred = out[:, :, :self.k_ctx, :, :]         # (B, C, K, H, W)
        fut_pred = out[:, :, self.k_ctx:, :, :].flatten(1, 2)  # (B, C, H, W)
        return ctx_pred.permute(0, 2, 1, 3, 4).contiguous(), fut_pred   # (B,K,C,H,W), (B,C,H,W)

    def get_velocity(self, z_t, ctx, t):
        x_pred = self.forward(z_t, ctx, t)
        return (x_pred - z_t) / (1.0 - t).clamp(min=0.01).view(-1, 1, 1, 1)


# --------------------------------------------------------------------------- #
# 2b. VAE/AE CORRUPTOR — learns the data manifold, then corrupts context frames
#     via encode -> add latent noise -> decode. Produces on-manifold "drifted"
#     frames WITHOUT needing the model's own outputs or extra data (the
#     scheduled-sampling substitute when self-feeding is impossible, e.g.
#     4->3 joint prediction with a fixed 7-frame dataset).
# --------------------------------------------------------------------------- #
class FrameVAE(nn.Module):
    def __init__(self, c_chan, base=32, latent=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(c_chan, base, 3, stride=2, padding=1), nn.SiLU(),     # 16
            nn.Conv2d(base, base*2, 3, stride=2, padding=1), nn.SiLU(),    # 8
            nn.Conv2d(base*2, base*4, 3, stride=2, padding=1), nn.SiLU(),  # 4
        )
        self.flat_dim = base*4*4*4
        self.fc_mu = nn.Linear(self.flat_dim, latent)
        self.fc_lv = nn.Linear(self.flat_dim, latent)
        self.fc_dec = nn.Linear(latent, self.flat_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base*4, base*2, 4, stride=2, padding=1), nn.SiLU(),  # 8
            nn.ConvTranspose2d(base*2, base, 4, stride=2, padding=1), nn.SiLU(),    # 16
            nn.ConvTranspose2d(base, c_chan, 4, stride=2, padding=1),               # 32
        )
        self.base = base

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.base*4, 4, 4)
        return self.dec(h)


def train_vae(frames, c_chan, epochs, latent, beta):
    """Train a VAE (beta~0 => sharp autoencoder) on all dataset frames."""
    vae = FrameVAE(c_chan, latent=latent).to(DEVICE)
    opt = torch.optim.AdamW(vae.parameters(), lr=3e-3)
    X = frames.detach()
    n = X.shape[0]
    bs = 256
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            xb = X[perm[i:i+bs]]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                mu, lv = vae.encode(xb)
                z = mu + torch.exp(0.5*lv) * torch.randn_like(mu)
                xr = vae.decode(z)
                recon = ((xr - xb)**2).mean()
                kl = (-0.5 * (1 + lv - mu**2 - torch.exp(lv)).sum(dim=1)).mean()
                loss = recon + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mu, lv = vae.encode(X)
        latent_std = float(torch.exp(0.5*lv).mean().item())
        recon_err = float(((vae.decode(mu) - X)**2).mean().item())
        latent_bank = mu.clone()  # (N, latent) for manifold-interpolation corruption
    return vae, max(latent_std, 1e-3), recon_err, latent_bank


def vae_perturb(vae, frames, noise_std, latent_std):
    """frames: (..., C,H,W) -> on-manifold corrupted frames (same shape)."""
    shape = frames.shape
    flat = frames.reshape(-1, *shape[-3:])
    with torch.no_grad():
        mu, _ = vae.encode(flat)
        z = mu + (noise_std * latent_std) * torch.randn_like(mu)
        out = vae.decode(z)
    return out.reshape(*shape).float()


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

    if TECHNIQUE == "ctx_bridge":
        # Brownian-bridge-shaped Gaussian perturbation on context frames.
        # Noise std follows the SAME variance schedule as the target flow path:
        #   c_s^2 = CTX_BRIDGE_SIGMA^2 * s(1-s) + CTX_BRIDGE_SIGMA_MIN^2
        # so the model sees context degraded by the bridge's own noise shape.
        # Optional annealing: linearly decay the noise magnitude to 0 over a fraction
        # of training (accelerator->clean refinement; the model outgrows static noise).
        decay = 1.0
        if CTX_BRIDGE_ANNEAL > 0 and TRAIN_STEPS > 0:
            decay = max(0.0, 1.0 - _STEP / (CTX_BRIDGE_ANNEAL * TRAIN_STEPS))
        if CTX_BRIDGE_TIME == "peak":
            s = torch.full((B,), 0.5, device=ctx.device)
            var = CTX_BRIDGE_SIGMA ** 2 * s * (1.0 - s) + CTX_BRIDGE_SIGMA_MIN ** 2
            std = (torch.sqrt(var) * CTX_BRIDGE_SCALE * decay).view(B, 1, 1, 1, 1)
        elif CTX_BRIDGE_TIME == "uniform":
            # ablation: plain gaussian (no bridge schedule), isolates the schedule effect
            std = CTX_BRIDGE_SIGMA * CTX_BRIDGE_SCALE * decay
        elif CTX_BRIDGE_TIME == "randsigma":
            # random NON-CONDITIONAL sigma: the noise std itself is sampled uniformly
            # per sample in [0, CTX_BRIDGE_SCALE] (NOT tied to the bridge schedule t).
            # Implements ctx += sigma*eps with sigma~U(0,SCALE), eps~N(0,1) — a flow/
            # diffusion forward process on the CONTEXT. Unlike 'sample' (bridge-shaped,
            # max ~sigma/2) this hits the FULL [0,SCALE] range incl. large corrupts.
            std = (torch.rand(B, device=ctx.device) * CTX_BRIDGE_SCALE * decay).view(B, 1, 1, 1, 1)
        else:  # "sample" — full bridge variance range per sample
            s = torch.rand(B, device=ctx.device)
            var = CTX_BRIDGE_SIGMA ** 2 * s * (1.0 - s) + CTX_BRIDGE_SIGMA_MIN ** 2
            std = (torch.sqrt(var) * CTX_BRIDGE_SCALE * decay).view(B, 1, 1, 1, 1)
        out = ctx + torch.randn_like(ctx) * std
        if CTX_BRIDGE_LAST_ONLY and ctx.shape[1] > 1:
            # only the most-recent context slot holds the model's own output at inference
            out = torch.stack([ctx[:, k] if k < ctx.shape[1] - 1 else out[:, k]
                                   for k in range(ctx.shape[1])], dim=1)
        return out

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

    if TECHNIQUE in ("vae_interp", "vae_interp_blur", "vae_interptemp", "vae_interptemp_blur"):
        # Latent-interpolation corruption. Random-frame blend (vae_interp*) gives
        # two-blob artifacts; TEMPORAL-neighbor blend (vae_interptemp*) shifts the
        # blob to a nearby on-manifold position (realistic rollout error: the model
        # predicting the blob slightly off-position), staying sharp & single-blob.
        if _VAE is None:
            return ctx
        vae, _, _, bank = _VAE
        shape = ctx.shape
        flat = ctx.reshape(-1, *shape[-3:])           # (B*K, C,H,W)
        with torch.no_grad():
            mu, _ = vae.encode(flat)
            if TECHNIQUE.startswith("vae_interptemp"):
                # temporal successor (bank is stored in trajectory order)
                idx = torch.randint(0, max(1, bank.shape[0] - 1), (mu.shape[0],), device=mu.device)
                mu_tgt = bank[idx + 1]
            else:
                idx = torch.randint(0, bank.shape[0], (mu.shape[0],), device=mu.device)
                mu_tgt = bank[idx]
            a = torch.sigmoid(1.0 + 1.8 * torch.randn(mu.shape[0], 1, device=mu.device)).clamp(1e-3, 1-1e-3) * VAE_NOISE
            z = (1 - a) * mu + a * mu_tgt
            out = vae.decode(z).reshape(*shape).float()
        if TECHNIQUE in ("vae_interp_blur", "vae_interptemp_blur"):
            out = _blur2d(out, BLUR_SIGMA * 0.4)
        return out

    if TECHNIQUE in ("vae_noise", "vae_noise_blur", "vae_noise_blurj", "vae_ms"):
        # VAE/AE-latent context corruption: encode -> add latent noise -> decode.
        # On-manifold "drifted" frames without model self-outputs (self-feed substitute).
        if _VAE is None:
            return ctx
        vae, lstd, _, _ = _VAE
        s = torch.sigmoid(1.0 + 1.8 * torch.randn(B, 1, 1, 1, 1, device=ctx.device)).clamp(1e-3, 1-1e-3) * VAE_NOISE
        corrupted = vae_perturb(vae, ctx, float(s.mean()), lstd)
        if TECHNIQUE in ("vae_noise_blur", "vae_noise_blurj", "vae_ms"):
            corrupted = _blur2d(corrupted, BLUR_SIGMA * 0.4)
        if TECHNIQUE == "vae_noise_blurj":
            # on-manifold amplitude jitter (keeps sharpness, like selffeed_m)
            scale = 1.0 + 0.08 * torch.randn(B, 1, 1, 1, 1, device=ctx.device)
            corrupted = corrupted * scale
        return corrupted

    if TECHNIQUE in ("manifold_noise", "manifold_ms"):
        # manifold-aligned-ish: mild blur (smooth, plausible) + amplitude jitter.
        # Emulates the model's actual rollout error (wider, lower-amp blob).
        base_sb = BLUR_SIGMA * MANIFOLD_BLUR_FRAC
        if MANIFOLD_ANNEAL > 0 and TRAIN_STEPS > 0:
            # cosine-anneal the blur down over training to track the model's
            # shrinking rollout error (semi-closed-loop: static corruption
            # over-corrupts late; annealing approximates self-feed's compounding).
            frac = 0.5 * (1 + math.cos(math.pi * min(1.0, _STEP / max(1, TRAIN_STEPS))))
            base_sb = base_sb * (MANIFOLD_ANNEAL + (1.0 - MANIFOLD_ANNEAL) * frac)
        if MANIFOLD_BLUR_RAND > 0:
            sb = base_sb * (1.0 - MANIFOLD_BLUR_RAND + 2.0 * MANIFOLD_BLUR_RAND * torch.rand(B, 1, 1, 1, 1, device=ctx.device))
        else:
            sb = base_sb
        blurred = _blur2d(ctx, float(sb.mean()))
        if MANIFOLD_LAST_ONLY and ctx.shape[1] > 1:
            # only the most-recent context slot is the model's (degraded) output at inference
            blurred = torch.stack([ctx[:, k] if k < ctx.shape[1] - 1 else blurred[:, k]
                                   for k in range(ctx.shape[1])], dim=1)
        scale = 1.0 + MANIFOLD_JITTER * torch.randn(B, 1, 1, 1, 1, device=ctx.device)
        return blurred * scale

    if TECHNIQUE in ("selffeed", "selffeed_m", "selffeed_ms", "selffeed_msgate"):
        # scheduled sampling: w.p. SELFFEED_PROB replace LAST context frame with
        # the model's own 1-step forecast from [extra, ctx[:,0]] (detached).
        out = ctx.clone()
        if model is not None and extra is not None:
            mask = torch.rand(B, device=ctx.device) < SELFFEED_PROB
            if mask.any():
                sur_ctx = torch.stack([extra[mask], ctx[mask, 0]], dim=1)  # Bm,K,C,H,W
                with torch.no_grad():
                    pred = sample_step(model, sur_ctx, SELFFEED_ODE_STEPS if SELFFEED_ODE_STEPS > 0 else ODE_STEPS, guidance=1.0)
                # ERROR GATING: only self-feed samples whose surrogate prediction
                # is accurate enough (low MSE vs the real frame). Adaptively
                # self-feeds on easy/sparse data (good preds) and skips on
                # hard/dense data (bad preds) -> works across regimes.
                if SELFFEED_GATE > 0 or SELFFEED_GATE_ABS > 0:
                    real = ctx[mask, 1]
                    err = ((pred - real) ** 2).mean(dim=(1, 2, 3))
                    if SELFFEED_GATE_ABS > 0:
                        # absolute, scale-invariant: err / frame_variance.
                        # sparse data + good pred -> tiny -> kept; dense/hard -> err~var -> dropped.
                        var = real.var(dim=(1, 2, 3)).clamp(min=1e-6)
                        keep = (err / var) <= SELFFEED_GATE_ABS
                    else:
                        keep = err <= torch.quantile(err, SELFFEED_GATE)
                    midx = mask.nonzero(as_tuple=True)[0][keep]
                    out[midx, 1] = pred[keep]
                else:
                    out[mask, 1] = pred
        blurred = _blur2d(out, BLUR_SIGMA * 0.4)
        if TECHNIQUE in ("selffeed_m", "selffeed_ms", "selffeed_msgate"):
            # manifold perturbation: + on-manifold amplitude jitter (keeps sharpness)
            scale = 1.0 + 0.08 * torch.randn(B, 1, 1, 1, 1, device=ctx.device)
            return blurred * scale
        return blurred

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
# 3a. FLOW PATH — linear rectified flow (RF) or Brownian-bridge flow matching
# --------------------------------------------------------------------------- #
def bridge_coeffs(t, sigma, sigma_min):
    """Brownian-bridge path coefficients (convention: t=0 noise, t=1 data).
        c_t^2    = sigma^2 * t(1-t) + sigma_min^2
        c'_t/c_t = sigma^2 (1-2t) / (2 c_t^2)
    Returns (c_t, cp_over_c). t may be a python float or a tensor."""
    t = torch.as_tensor(t, dtype=torch.float32, device=DEVICE)
    var = sigma ** 2 * t * (1.0 - t) + sigma_min ** 2
    c = torch.sqrt(var)
    cp_over_c = sigma ** 2 * (1.0 - 2.0 * t) / (2.0 * var + 1e-12)
    return c, cp_over_c


def _make_zt(t, tgt, eps):
    """Noised target z_t for the active flow. t: (B,).
    RF:     z_t = (1-t) eps + t tgt.
    Bridge: z_t = (1-t) eps + t tgt + c_t eta  (extra mid-path Brownian perturbation)."""
    ts = t.view(-1, 1, 1, 1)
    mu_t = (1.0 - ts) * eps + ts * tgt
    if FLOW == "bridge":
        eta = torch.randn_like(tgt)
        c_t, _ = bridge_coeffs(t, BRIDGE_SIGMA, BRIDGE_SIGMA_MIN)
        return mu_t + c_t.view(-1, 1, 1, 1) * eta
    return mu_t


def _flow_loss_term(model, z_t, tgt, ctx_aug, t, eps):
    """Flow-matching loss term for the active flow. Returns (loss_scalar, x_pred).
    RF:     velocity loss  (v_pred - (tgt-eps))^2 * 1/(1-t)^2  clamped (RF_WCLAMP).
    Bridge: x-prediction loss (model predicts clean target y); weight depends on
            BRIDGE_LOSS:
              vloss   = (x_pred-y)^2 * (1-t c'/c)^2  clamped  (velocity-matching;
                        one-sided data-end upweight like RF's 1/(1-t)^2 -> prevents
                        the context-prediction collapse; the literal bridge v-loss).
              ivar    = (x_pred-y)^2 * 1/c_t^2  clamped  (inverse-conditional-variance;
                        upweights BOTH endpoints incl. the irreducible noise-end floor).
              uniform = (x_pred-y)^2 * 1  (plain x-pred MSE; collapses on easy data).
            (The naive bridge velocity TARGET explodes via c'/c -> 1/sigma_min^2 near
            endpoints, so we match velocity through the x-prediction reparameterization.)"""
    if FLOW == "bridge":
        x_pred = model.forward(z_t, ctx_aug, t)
        if BRIDGE_LOSS == "ivar":
            c_t, _ = bridge_coeffs(t, BRIDGE_SIGMA, BRIDGE_SIGMA_MIN)
            w = (1.0 / (c_t ** 2)).clamp(max=BRIDGE_WCLAMP)
        elif BRIDGE_LOSS == "uniform":
            w = torch.ones_like(t)
        else:  # "vloss" — velocity-matching via x-prediction.
            # u_t = (y-eps) + (c'/c)(z_t-mu_t);  ||v_theta-u_t||^2 = (x_pred-y)^2 (1-t c'/c)^2.
            # (1 - t c'/c)^2 ~ 1 at low/mid t, rises toward the DATA end (t->1):
            # a one-sided data-end upweight that forces z_t usage (no collapse).
            c_t, cp_over_c = bridge_coeffs(t, BRIDGE_SIGMA, BRIDGE_SIGMA_MIN)
            wf = 1.0 - t * cp_over_c                 # (1 - t c'/c), shape (B,)
            w = (wf ** 2).clamp(max=BRIDGE_WCLAMP)
        loss_vel = ((x_pred - tgt) ** 2).mean(dim=(1, 2, 3)) * w
        return loss_vel.mean().float(), x_pred
    v_pred = model.get_velocity(z_t, ctx_aug, t)
    w = (1.0 / (1.0 - t).clamp(min=0.05) ** 2).clamp(max=RF_WCLAMP)
    v_target = tgt - eps
    loss_vel = ((v_pred - v_target) ** 2).mean(dim=(1, 2, 3)) * w
    x_pred = model.forward(z_t, ctx_aug, t)
    return loss_vel.mean().float(), x_pred


# --------------------------------------------------------------------------- #
# 3b. JOINT-OBJECTIVE loss components (RF context reconstruction + bridge future)
# --------------------------------------------------------------------------- #
def _bridge_future_loss(fut_pred, tgt, t):
    """Bridge x-prediction loss on the FUTURE slot (matches _flow_loss_term's
    bridge branch). fut_pred/tgt: (B,C,H,W); t: (B,). Returns (loss_scalar, fut_pred)."""
    if FLOW == "bridge":
        if BRIDGE_LOSS == "ivar":
            c_t, _ = bridge_coeffs(t, BRIDGE_SIGMA, BRIDGE_SIGMA_MIN)
            w = (1.0 / (c_t ** 2)).clamp(max=BRIDGE_WCLAMP)
        elif BRIDGE_LOSS == "uniform":
            w = torch.ones_like(t)
        else:  # "vloss"
            c_t, cp_over_c = bridge_coeffs(t, BRIDGE_SIGMA, BRIDGE_SIGMA_MIN)
            w = ((1.0 - t * cp_over_c) ** 2).clamp(max=BRIDGE_WCLAMP)
        loss = (((fut_pred - tgt) ** 2).mean(dim=(1, 2, 3)) * w).mean()
        return loss.float(), fut_pred
    # RF future (if FLOW=rf): velocity loss, fut_pred already predicts clean y.
    w = (1.0 / (1.0 - t).clamp(min=0.05) ** 2).clamp(max=RF_WCLAMP)
    loss = (((fut_pred - tgt) ** 2).mean(dim=(1, 2, 3)) * w).mean()
    return loss.float(), fut_pred


def _rf_context_loss(ctx_pred, ctx_clean, s):
    """Context-slot reconstruction loss for the joint objective.
    ctx_pred/ctx_clean: (B,K,C,H,W); s: (B,). Returns loss_scalar.
    JOINT_CTX_FLOW="rf":     linear RF velocity weight 1/(1-s)^2 clamped (endpoint reparam).
    JOINT_CTX_FLOW="bridge": Brownian-bridge vloss weight (1-s c'/c)^2 clamped
                             (one-sided data-end upweight -> forces SHARP recon, avoids
                             the mean-pull blur of the RF variant at high noise)."""
    if JOINT_CTX_FLOW == "bridge":
        c_s, cp_over_c = bridge_coeffs(s, JOINT_CTX_BRIDGE_SIGMA, JOINT_CTX_BRIDGE_SIGMA_MIN)
        w = ((1.0 - s * cp_over_c) ** 2).clamp(max=BRIDGE_WCLAMP)          # (B,)
    else:  # "rf"
        w = (1.0 / (1.0 - s).clamp(min=0.05) ** 2).clamp(max=RF_WCLAMP)    # (B,)
    loss = ((ctx_pred - ctx_clean) ** 2).mean(dim=(1, 2, 3, 4)) * w         # (B,)
    return loss.mean().float()


# --------------------------------------------------------------------------- #
# 4. TRAINING
# --------------------------------------------------------------------------- #
def train(data):
    torch.manual_seed(SEED)
    if ARCH == "conv3d":
        model = VideoRF3D(K_CTX, C_CHAN, ch=MODEL_CH, n_blocks=MODEL_BLOCKS,
                          latent_blur_sigma=LATENT_BLUR).to(DEVICE)
    elif ARCH == "jit3d":
        # JOINT_DECOPLE -> the net sees BOTH a context-time and a future-time
        # (context_dim=2); otherwise the RF/bridge time only (context_dim=1).
        _cdim = 2 if (JOINT_GEN and JOINT_DECOUPLE) else 1
        model = VideoRFJiT3D(
            K_CTX, C_CHAN, IMG,
            embed_dim=JIT_EMBED_DIM, depth=JIT_DEPTH, num_heads=JIT_HEADS,
            patch_t=JIT_PATCH_T, patch_hw=JIT_PATCH_HW, mlp_ratio=JIT_MLP_RATIO,
            corrupt_prob=JIT_CORRUPT_PROB, corrupt_embed=JIT_CORRUPT_EMBED,
            corrupt_block0=JIT_CORRUPT_BLOCK0, context_dim=_cdim,
        ).to(DEVICE)
    else:
        model = VideoRF(K_CTX, C_CHAN, ch=MODEL_CH, n_blocks=MODEL_BLOCKS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.RandomState(SEED + 1)
    if WARMUP_STEPS > 0:
        # linear warmup -> cosine decay. ViT/jit3d need warmup to avoid early loss spikes.
        warmup = WARMUP_STEPS
        eta0 = LR
        import math as _math
        def _lr(step):
            if step < warmup:
                return eta0 * (step + 1) / warmup
            prog = (step - warmup) / max(1, TRAIN_STEPS - warmup)
            return 0.5 * eta0 * (1.0 + _math.cos(_math.pi * min(1.0, prog)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_STEPS)
    last_loss = 0.0
    for step in range(TRAIN_STEPS):
        global _STEP
        _STEP = step
        ctx, tgt, extra, tgt2 = data.sample_windows(BATCH, rng)
        t = torch.rand(BATCH, device=DEVICE)
        eps = (ctx[:, -1] + SRC_NOISE * torch.randn_like(tgt)) if FLOW_SOURCE == "lastctx" else torch.randn_like(tgt)
        z_t = _make_zt(t, tgt, eps)
        # ------------------------------------------------------------------ #
        # JOINT RF-context-generation + bridge-future-prediction objective
        # (user idea): the net jointly DENOISES the context (linear RF, learns to
        # generate data from scratch) AND predicts the future (bridge). Loss =
        # bridge_loss(future) + JOINT_CTX_WEIGHT * rf_loss(context). The context
        # RF time s is biased closer to data than t ("generate data first, then
        # predict"). Inference rollout is unchanged (predict from clean context).
        # ------------------------------------------------------------------ #
        if JOINT_GEN and ARCH == "jit3d":
            # context RF time s: "closer to data" than the future time t.
            if JOINT_COUPLE:
                s = (t + JOINT_LEAD).clamp(max=1.0)          # context leads toward data by JOINT_LEAD
            else:
                s = torch.rand(BATCH, device=DEVICE) * (1.0 - JOINT_CTX_TMIN) + JOINT_CTX_TMIN  # U(TMIN,1)
            eps_ctx = torch.randn_like(ctx)                 # (B,K,C,H,W) per-slot source
            if JOINT_CTX_FLOW == "bridge":
                # bridge context: z_ctx = (1-s)eps + s*ctx + c_s eta  (clean-endpoint landing)
                c_s, _ = bridge_coeffs(s, JOINT_CTX_BRIDGE_SIGMA, JOINT_CTX_BRIDGE_SIGMA_MIN)
                z_ctx = (1.0 - s).view(BATCH, 1, 1, 1, 1) * eps_ctx + s.view(BATCH, 1, 1, 1, 1) * ctx \
                        + c_s.view(BATCH, 1, 1, 1, 1) * torch.randn_like(ctx)
            else:  # "rf"
                z_ctx = (1.0 - s).view(BATCH, 1, 1, 1, 1) * eps_ctx + s.view(BATCH, 1, 1, 1, 1) * ctx   # RF noised context
            if JOINT_LAST_ONLY and ctx.shape[1] > 1:
                # only RF-noise the most-recent slot (holds the model's own output at inference)
                keep = [k for k in range(ctx.shape[1] - 1)]
                z_ctx = torch.cat([ctx[:, keep], z_ctx[:, -1:]], dim=1)
            with _amp():
                ctx_pred, fut_pred = model.forward_joint(z_ctx, z_t, t, s)
                loss_fut, _ = _bridge_future_loss(fut_pred, tgt, t)
                loss_ctx = _rf_context_loss(ctx_pred, ctx, s)
                loss = loss_fut + JOINT_CTX_WEIGHT * loss_ctx
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            last_loss = loss.item()
            if (step + 1) % 500 == 0:
                print(f"  step {step+1:4d}/{TRAIN_STEPS}  loss={last_loss:.5f} (fut={loss_fut.item():.5f} ctx={loss_ctx.item():.5f})", flush=True)
            continue
        ctx_aug = augment_context(ctx, model=model, extra=extra, training=True)
        # classifier-free guidance: per-sample context dropout (train unconditional path)
        if UNCOND_PROB > 0:
            drop = torch.rand(ctx_aug.shape[0], device=ctx_aug.device) < UNCOND_PROB
            if drop.any():
                ctx_aug = ctx_aug.clone()
                ctx_aug[drop] = 0.0
        with _amp():
            loss_vel, x_pred = _flow_loss_term(model, z_t, tgt, ctx_aug, t, eps)
            loss = loss_vel + extra_loss(x_pred.float(), tgt.float()).float()
        # multi-step rollout loss: predict tgt2 from [ctx[:,1], model_pred(tgt)],
        # training the model to stay consistent when its own output is fed back.
        if TECHNIQUE in ("selffeed_ms", "selffeed_msgate", "vae_ms", "manifold_ms") and MS_PROB > 0 and rng.rand() < MS_PROB:
            if TECHNIQUE == "manifold_ms":
                # SELF-FEED-FREE multi-step loss: use a manifold-corrupted (blur+jitter)
                # version of the REAL frame t as the drifted context for predicting t+1.
                # No model self-outputs, no VAE. Emulates rollout drift as a LOSS.
                with torch.no_grad():
                    sb = BLUR_SIGMA * MANIFOLD_BLUR_FRAC
                    Bb = tgt.shape[0]
                    if MANIFOLD_BLUR_RAND > 0:
                        sb = sb * (1.0 - MANIFOLD_BLUR_RAND + 2.0 * MANIFOLD_BLUR_RAND * torch.rand(Bb, device=tgt.device))
                    pred1 = _blur2d(tgt, float(sb.mean()))
                    pred1 = pred1 * (1.0 + MANIFOLD_JITTER * torch.randn(Bb, 1, 1, 1, device=tgt.device))
            elif TECHNIQUE == "vae_ms" and _VAE is not None:
                # SELF-FEED-FREE multi-step loss: use an AE-corrupted version of the
                # REAL frame t (not a model prediction) as the drifted context for
                # predicting t+1. Emulates rollout drift as a LOSS, no self-outputs.
                vae, lstd, _, _ = _VAE
                s = float((torch.sigmoid(1.0 + 1.8 * torch.randn(1, device=DEVICE)).clamp(1e-3, 1 - 1e-3)) * VAE_NOISE)
                with torch.no_grad():
                    pred1 = vae_perturb(vae, tgt, s, lstd)   # AE-corrupted real frame t
                    pred1 = _blur2d(pred1, BLUR_SIGMA * 0.4)
            else:
                with torch.no_grad():
                    pred1 = sample_step(model, ctx, ODE_STEPS, guidance=1.0)  # model's pred of tgt from CLEAN ctx
            ctx2 = torch.stack([ctx[:, 1], pred1], dim=1)   # B,K,C,H,W
            ctx2_aug = augment_context(ctx2, model=model, extra=extra, training=True)
            t2 = torch.rand(BATCH, device=DEVICE)
            eps2 = (ctx2[:, -1] + SRC_NOISE * torch.randn_like(tgt2)) if FLOW_SOURCE == "lastctx" else torch.randn_like(tgt2)
            z_t2 = _make_zt(t2, tgt2, eps2)
            with _amp():
                loss_ms, _ = _flow_loss_term(model, z_t2, tgt2, ctx2_aug, t2, eps2)
            loss = loss + MS_WEIGHT * loss_ms
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
# 5. SAMPLING & ROLLOUT
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _ode_integrate(model, z0, ctx_aug, n_ode, g):
    """Integrate the flow ODE 0->1 from the noise source z0 (B',C,H,W).
    RF:     forward-Euler  v=(x_pred-z)/(1-t)  (+ optional CFG + stochastic re-noise).
    Bridge: closed-form step  z(t+dt) = mu_next + (z-mu_t)*c(t+dt)/c(t), with
            mu_t = (1-t)eps + t x_pred. Absorbs the stiff c'/c mean-reversion into
            the O(1) ratio c(t+dt)/c(t); lands exactly on x_pred at t=1 (endpoint
            variance -> sigma_min^2 ~ 0, so nothing off-manifold is fed forward).
            CFG (g!=1) is RF-only (bridge has no velocity to interpolate)."""
    Bn = z0.shape[0]
    z = z0
    dt = 1.0 / n_ode
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        if FLOW == "bridge":
            eps_fixed = z0
            for i in range(n_ode):
                t_val = i * dt
                t_next = t_val + dt
                t = torch.full((Bn,), t_val, device=DEVICE)
                x_pred = model.forward(z, ctx_aug, t).float()
                mu_t = (1.0 - t_val) * eps_fixed + t_val * x_pred
                mu_next = (1.0 - t_next) * eps_fixed + t_next * x_pred
                c_t, _ = bridge_coeffs(t_val, BRIDGE_SIGMA, BRIDGE_SIGMA_MIN)
                c_next, _ = bridge_coeffs(t_next, BRIDGE_SIGMA, BRIDGE_SIGMA_MIN)
                ratio = c_next / c_t                       # 0-dim tensor, broadcasts
                z = mu_next + (z - mu_t) * ratio
        else:
            for i in range(n_ode):
                t = torch.full((Bn,), i * dt, device=DEVICE)
                v = model.get_velocity(z, ctx_aug, t)
                if g != 1.0:
                    v_u = model.get_velocity(z, torch.zeros_like(ctx_aug), t)
                    v = v_u + g * (v - v_u)
                z = z + v * dt
                if NOISE_INJECT > 0:
                    remaining = max(0.0, 1.0 - (i + 1) * dt)
                    z = z + NOISE_INJECT * math.sqrt(dt) * remaining * torch.randn_like(z)
    return z


@torch.no_grad()
def sample_step(model, ctx, n_ode, guidance=None, n_samples=None):
    """Sample one frame (B,C,H,W) from context ctx (B,K,C,H,W).
    guidance=None -> use global GUIDANCE (inference); pass 1.0 to disable (training).
    n_samples>1: average that many noise-sample endpoints (variance reduction)."""
    n_samples = SAMPLE_AVG if n_samples is None else n_samples
    g = GUIDANCE if guidance is None else guidance
    B = ctx.shape[0]
    ctx_aug = augment_context(ctx, training=False)
    last = ctx[:, -1]  # B,C,H,W — the flow source when FLOW_SOURCE=lastctx
    if n_samples > 1:
        # replicate contexts across N noise samples, integrate together, average
        ctx_rep = ctx_aug.repeat_interleave(n_samples, dim=0)  # B*N,K,C,H,W
        if FLOW_SOURCE == "lastctx":
            src = last.repeat_interleave(n_samples, dim=0)
            z0 = src + (SRC_NOISE * torch.randn_like(src) if SRC_NOISE > 0 else 0.0)
        else:
            z0 = torch.randn(B * n_samples, C_CHAN, IMG, IMG, device=DEVICE)
        z = _ode_integrate(model, z0, ctx_rep, n_ode, g)
        return z.view(B, n_samples, C_CHAN, IMG, IMG).mean(dim=1).float()
    if FLOW_SOURCE == "lastctx":
        z0 = last + (SRC_NOISE * torch.randn_like(last) if SRC_NOISE > 0 else 0.0)
    else:
        z0 = torch.randn(B, C_CHAN, IMG, IMG, device=DEVICE)
    return _ode_integrate(model, z0, ctx_aug, n_ode, g).float()


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


def energy_distance(a, b):
    """Non-saturating energy distance (V-statistic) in standardized feature space.
    ED = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||  >= 0, =0 iff distributions match.
    Unlike Gaussian-kernel MMD it grows with distributional distance instead of
    saturating at ~1 when the rollout is catastrophically off-manifold."""
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


def raw_grad_mass(frames):
    gx, gy = torch.gradient(frames, dim=(2, 3))
    ge = float((gx ** 2 + gy ** 2).mean())
    ms = float(frames.mean())
    return ge, ms


def evaluate(model, data, ref_feats, ref_stats, bw, ref_grad, ref_mass):
    rollout_frames = rollout(model, data, N_ROLLOUT, ROLLOUT_LEN)  # T,B,C,H,W
    flat = rollout_frames.flatten(0, 1)            # (T*B, C,H,W)
    rf = flat[:1024]
    mu, sd = ref_stats
    rf_feat = (features(rf) - mu) / sd
    ed = energy_distance(rf_feat, ref_feats)
    mmd = mmd2_sq(rf_feat, ref_feats, bw)
    # late-half energy distance (where drift is worst) — diagnostic
    half = rollout_frames.shape[0] // 2
    late = rollout_frames[half:].flatten(0, 1)[:1024]
    late_feat = (features(late) - mu) / sd
    ed_late = energy_distance(late_feat, ref_feats)
    ge_roll, mass_roll = raw_grad_mass(rf)
    sharp = ge_roll / max(ref_grad, 1e-9)
    mass_drift = abs(mass_roll - ref_mass) / max(abs(ref_mass), 1e-9)
    return ed, mmd, ed_late, sharp, mass_drift


# --------------------------------------------------------------------------- #
# 7. MAIN
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"TECHNIQUE={TECHNIQUE} SIGMA={SIGMA} BLUR_SIGMA={BLUR_SIGMA} "
          f"IMG={IMG} C_CHAN={C_CHAN} K_CTX={K_CTX} ARCH={ARCH} FLOW={FLOW} "
          f"BRIDGE_LOSS={BRIDGE_LOSS} BRIDGE_SIGMA={BRIDGE_SIGMA} "
          f"BRIDGE_WCLAMP={BRIDGE_WCLAMP} device={DEVICE} "
          f"JOINT_GEN={JOINT_GEN} JOINT_CTX_W={JOINT_CTX_WEIGHT} "
          f"JOINT_COUPLE={JOINT_COUPLE} JOINT_LEAD={JOINT_LEAD} "
          f"JOINT_TMIN={JOINT_CTX_TMIN} JOINT_DECOUPLE={JOINT_DECOUPLE} "
          f"JOINT_LAST_ONLY={JOINT_LAST_ONLY} JOINT_CTX_FLOW={JOINT_CTX_FLOW}", flush=True)

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
    ed_floor = energy_distance(floor_feat, ref_f)
    print(f"[data] ref feats={ref_f.shape} bw={bw:.3f} ed_floor={ed_floor:.6f} "
          f"mmd_floor={mmd_floor:.6f} ref_grad={ref_grad:.5f} ref_mass={ref_mass:.5f}", flush=True)

    global _VAE
    if TECHNIQUE.startswith("vae_noise") or TECHNIQUE.startswith("vae_interp") or TECHNIQUE.startswith("vae_interptemp") or TECHNIQUE == "vae_ms":
        print(f"[vae] training corruptor latent={VAE_LATENT} epochs={VAE_EPOCHS} beta={VAE_BETA}", flush=True)
        vae, lstd, rerr, bank = train_vae(train_data.frames, C_CHAN, VAE_EPOCHS, VAE_LATENT, VAE_BETA)
        _VAE = (vae, lstd, rerr, bank)
        print(f"[vae] trained. latent_std={lstd:.4f} recon_mse={rerr:.6f}", flush=True)

    print(f"[train] {TRAIN_STEPS} steps batch={BATCH}", flush=True)
    model, train_loss = train(train_data)

    print("[eval] rollout...", flush=True)
    ed, mmd, ed_late, sharp, mass_drift = evaluate(model, hold_data, ref_f, (mu, sd), bw, ref_grad, ref_mass)
    wall = time.time() - t0

    print(f"[done] rollout_ed={ed:.6f} ed_late={ed_late:.6f} mmd={mmd:.6f} "
          f"sharpness={sharp:.4f} mass_drift={mass_drift:.4f} "
          f"train_loss={train_loss:.5f} ed_floor={ed_floor:.6f} wall={wall:.1f}s", flush=True)
    print(f"METRIC rollout_ed={ed:.6f}", flush=True)
    print(f"METRIC ed_late={ed_late:.6f}", flush=True)
    print(f"METRIC mmd={mmd:.6f}", flush=True)
    print(f"METRIC sharpness_ratio={sharp:.6f}", flush=True)
    print(f"METRIC mass_drift={mass_drift:.6f}", flush=True)
    print(f"METRIC train_loss={train_loss:.6f}", flush=True)
    print(f"METRIC ed_floor={ed_floor:.6f}", flush=True)
    print(f"METRIC wall_s={wall:.2f}", flush=True)


if __name__ == "__main__":
    main()
