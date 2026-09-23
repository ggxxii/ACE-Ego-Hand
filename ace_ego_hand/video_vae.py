"""Video decode and Wan-VAE encode — the pixel side of inference.

``decode_video`` reads frames with OpenCV, ``load_vae``/``encode`` run the
frozen causal VAE from the Wan2.2-Fun-5B-Control release (see the README,
"Download checkpoints"). ``encode`` streams 1-then-4-frame micro-chunks so a
long clip never occupies GPU as a single pixel tensor.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "ckpt" / "Wan2.2-Fun-5B-Control"
VIDEOX_CFG = ROOT / "third_party" / "config" / "wan2.2" / "wan_civitai_5b.yaml"


def decode_video(path: str, n_frames: int, resize_w: int = 0,
                 resize_hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Decode up to n_frames -> (F, 3, H, W) float32 in [-1, 1] (RGB).

    ``resize_hw`` rescales to an exact (W, H); ``resize_w`` rescales the width
    only. The caller is responsible for scaling K to match.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n_frames:
        ok, f = cap.read()
        if not ok:
            break
        if resize_hw and (f.shape[1] != resize_hw[0] or f.shape[0] != resize_hw[1]):
            f = cv2.resize(f, resize_hw, interpolation=cv2.INTER_AREA)
        elif resize_w and f.shape[1] != resize_w:
            f = cv2.resize(f, (resize_w, f.shape[0]), interpolation=cv2.INTER_AREA)
        frames.append(f[:, :, ::-1].copy())  # BGR -> RGB
    cap.release()
    if len(frames) < n_frames:
        raise RuntimeError(f"{path}: decoded {len(frames)} < expected {n_frames}")
    a = np.stack(frames).astype(np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()


def frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def load_vae(device: torch.device):
    from videox_fun.models import AutoencoderKLWan3_8
    config = OmegaConf.load(str(VIDEOX_CFG))
    vae = AutoencoderKLWan3_8.from_pretrained(
        os.path.join(str(MODEL_ROOT), config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(device, torch.bfloat16)
    vae.eval()
    vae.requires_grad_(False)
    return vae


@torch.no_grad()
def encode(vae, pixels: torch.Tensor, device) -> torch.Tensor:
    """(F,3,H,W) [-1,1] -> (C_lat, F_lat, H/16, W/16) f16 cpu.

    Replays ``AutoencoderKLWan2_2_.encode``'s causal 1-then-4-frame loop, but
    only the current micro-chunk lives on GPU. ``_enc_feat_map`` stays on GPU
    across chunks; ``clear_cache`` runs once at the start and once at the end,
    so this is not equivalent to ``for chunk in split(...): vae.encode(chunk)``.
    """
    from videox_fun.models.wan_vae3_8 import patchify

    inner = vae.model
    n = pixels.shape[0]
    iter_ = 1 + (n - 1) // 4
    scale = [s.to(device, torch.bfloat16) for s in vae.scale]
    mean = scale[0].view(1, inner.z_dim, 1, 1, 1)
    inv_std = scale[1].view(1, inner.z_dim, 1, 1, 1)

    inner.clear_cache()
    latents = []
    for i in range(iter_):
        sl = slice(0, 1) if i == 0 else slice(1 + 4 * (i - 1), 1 + 4 * i)
        x = pixels[sl].permute(1, 0, 2, 3).unsqueeze(0).to(device, torch.bfloat16)
        x = patchify(x, patch_size=2)
        inner._enc_conv_idx = [0]
        out = inner.encoder(
            x, feat_cache=inner._enc_feat_map, feat_idx=inner._enc_conv_idx)
        mu, _log_var = inner.conv1(out).chunk(2, dim=1)
        mu = (mu - mean) * inv_std
        latents.append(mu[0].to(torch.float16).cpu())
        del x, out, mu, _log_var
    inner.clear_cache()
    return torch.cat(latents, dim=1)


# RHD masters store the NATIVE RHD joint order: wrist=0, then per finger
# [tip -> palm] (README "1-4: thumb [tip to palm], 5-8: index, ...").  Our pred
# convention is OpenPose-21 (wrist, then per finger root -> tip), so reorder at
# cache-build time; the map is an involution (RHD->OP == OP->RHD).
_RHD_TO_OPENPOSE21 = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9,
                      16, 15, 14, 13, 20, 19, 18, 17]
