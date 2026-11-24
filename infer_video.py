#!/usr/bin/env python3
"""Run ACE-Ego-Hand on a single egocentric video.

The VAE and the backbone are both resident; the clip is decoded, VAE-encoded,
predicted and written in one pass, and the latent only ever exists in GPU
memory.

Camera: the K-given config needs a calibration (``--camera cam.json``, or the
inline shorthand ``--intrinsics fx,fy,cx,cy``); it is rescaled with the frames.
The K-free config needs none -- the model reads the camera off the image, and
its own fitted pinhole is written to the output as ``pred_intrinsics``.

One pickle is written for the whole clip; ``--split_segments`` reproduces the
benchmark 81-frame dump format instead.

Both hand slots are regressed on every frame, so whether a hand is there is a
decision this script makes: a hand is claimed in a frame iff its in-frame
presence ``exists_2d > 0.5``. The decision is written as ``claim`` and every
pose array is NaN where it is False; ``exists_2d`` / ``exists_3d`` are kept
ungated so the decision can be redone.

Only pinhole cameras are supported: undistort fisheye footage first.

  python infer_video.py --video clip.mp4 \
      --opt options/ace_ego_hand_kfree.yml --ckpt checkpoints/ace_ego_hand_kfree.pt \
      --out results/clip
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ace_ego_hand.archs.geodit_arch import _reject_non_pinhole  # noqa: E402
from ace_ego_hand.inference import DUMP_KEYS, predict_video  # noqa: E402
from ace_ego_hand.mano_utils import mano_convention          # noqa: E402
from ace_ego_hand.models.geodit_model import GeoDitModel   # noqa: E402
from ace_ego_hand.video_vae import decode_video, encode, frame_count, load_vae  # noqa: E402

# The VAE downsamples by 16 in space and is causal in time (4x + 1). Latent grids
# must therefore be even in both axes, so round the encode size to a multiple of 32.
_GRID = 32


def _encode_size(w: int, h: int, max_w: int | None) -> tuple[int, int]:
    """Frame size to encode at: keep the aspect, snap both axes to the VAE grid.

    Frames wider than ``max_w`` are downscaled to it; narrower ones keep their
    native size. Upscaling puts the hands on a canvas the model was not trained
    on and biases depth (measured: ARCTIC 672x480 upscaled to 832 wide, the K
    model's absolute MPJPE is 81 mm vs 23 mm at native size).
    """
    tw = min(w, max_w) if max_w else w
    s = tw / float(w)
    ew = max(_GRID, int(round(w * s / _GRID)) * _GRID)
    eh = max(_GRID, int(round(h * s / _GRID)) * _GRID)
    return ew, eh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="path to an egocentric RGB video")
    ap.add_argument("-opt", "--opt", required=True, help="option yml")
    ap.add_argument("--ckpt", required=True, help="released checkpoint (.pt)")
    ap.add_argument("--out", required=True, help="output directory for the pkl dumps")
    ap.add_argument("--camera", default="",
                    help='camera JSON for the K-given config: {"image_width", '
                         '"image_height", "frames": [{"intrinsics": '
                         '{"fx", "fy", "cx", "cy"}}, ...]}. Intrinsics are '
                         'taken as constant per video, so only frames[0] is read.')
    ap.add_argument("--intrinsics", default="",
                    help="shorthand for --camera: fx,fy,cx,cy in the video's own "
                         "pixels. Ignored by K-free.")
    ap.add_argument("--tap", type=int, default=15, help="DiT block to read features from")
    ap.add_argument("--mode", choices=["tiled", "full"], default="tiled",
                    help="tiled: 22-latent windows (benchmark setting). "
                         "full: one forward over the clip (RoPE extrapolation)")
    ap.add_argument("--tile_w", type=int, default=22)
    ap.add_argument("--encode_w", type=int, default=832,
                    help="downscale frames wider than this before the VAE; narrower "
                         "frames are never upscaled. 0 keeps native")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--split_segments", action="store_true",
                    help="write one pkl per 81-frame segment (the benchmark dump "
                         "format) instead of a single whole-clip pkl")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    opt = yaml.safe_load(open(args.opt))
    device = torch.device(args.device)

    n_src = frame_count(args.video)
    if n_src <= 0:
        raise SystemExit(f"{args.video}: could not read a frame count")
    n_frames = min(n_src, args.max_frames) if args.max_frames else n_src
    # the causal VAE consumes 4k+1 (>= 5) pixel frames: pad by repeating the last
    # frame and drop the padded predictions afterwards, so every real frame is kept
    n_enc = max(5, 4 * ((n_frames + 2) // 4) + 1)

    cap = cv2.VideoCapture(args.video)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    enc_w, enc_h = _encode_size(src_w, src_h, args.encode_w or None)
    print(f"[video] {args.video}  {src_w}x{src_h}  {n_src} frames "
          f"-> predicting {n_frames} at {enc_w}x{enc_h}"
          + (f" ({n_enc - n_frames} padded)" if n_enc > n_frames else ""))

    if args.camera and args.intrinsics:
        raise SystemExit("pass either --camera or --intrinsics, not both")

    if args.camera or args.intrinsics:
        if args.camera:
            with open(args.camera) as fh:
                cam = json.load(fh)
            try:
                k = cam["frames"][0]["intrinsics"]
                fx, fy = float(k["fx"]), float(k["fy"])
                cx, cy = float(k["cx"]), float(k["cy"])
            except (KeyError, IndexError, TypeError) as e:
                raise SystemExit(
                    f"{args.camera}: could not read intrinsics ({type(e).__name__}: {e}).\n"
                    'Expected {"image_width", "image_height", "frames": '
                    '[{"intrinsics": {"fx", "fy", "cx", "cy"}}, ...]}')
            _reject_non_pinhole(dict(k, **{r: cam[r] for r in cam
                                           if r in ("model", "fisheye")}))
            # the intrinsics are expressed in the calibration's own pixels, which
            # need not be the video's -- scale from there, not from the video
            cal_w = float(cam.get("image_width", src_w))
            cal_h = float(cam.get("image_height", src_h))
            if (int(cal_w), int(cal_h)) != (src_w, src_h):
                print(f"[camera] calibrated at {int(cal_w)}x{int(cal_h)}, "
                      f"video is {src_w}x{src_h}; scaling K to match")
        else:
            fx, fy, cx, cy = (float(v) for v in args.intrinsics.split(","))
            cal_w, cal_h = float(src_w), float(src_h)
        # rescale K to the encode grid: fx/cx follow width, fy/cy follow height
        sx, sy = enc_w / cal_w, enc_h / cal_h
        intr = {"fx": fx * sx, "fy": fy * sy, "cx": cx * sx, "cy": cy * sy,
                "image_width": enc_w, "image_height": enc_h}
        print(f"[camera] fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
              f"cx={intr['cx']:.1f} cy={intr['cy']:.1f} at {enc_w}x{enc_h}")
    else:
        if not opt.get("backbone", {}).get("self_ray_decode", False):
            raise SystemExit(
                f"the K-given config ({args.opt}) back-projects the wrist "
                "through K, so it needs a camera: pass --camera cam.json or "
                "--intrinsics fx,fy,cx,cy. To run with no camera at all, use "
                "options/ace_ego_hand_kfree.yml.")
        # K-free reads the camera off the image, so nothing real is known here.
        # Build the placeholder in ENCODED pixels so it is at least a coherent
        # square-pixel camera -- scaling an isotropic source camera by the two
        # axis ratios would produce an fx != fy camera that describes nothing.
        foc = 0.5 * enc_w / np.tan(np.deg2rad(30.0))
        intr = {"fx": foc, "fy": foc, "cx": enc_w / 2.0, "cy": enc_h / 2.0,
                "image_width": enc_w, "image_height": enc_h,
                "source": "nominal 60deg placeholder; K-free does not read it"}
        print(f"[camera] K-free: no intrinsics given, placeholder fx={foc:.1f} "
              "(the model's own fitted camera is written as pred_intrinsics)")

    model = GeoDitModel(opt, device)
    model.load_inference(str(Path(args.ckpt).resolve()))
    if args.tap not in model.net.tap_layers:
        raise SystemExit(f"tap {args.tap} not in {model.net.tap_layers}")

    vae = load_vae(device)
    pixels = decode_video(args.video, n_frames, resize_hw=(enc_w, enc_h))
    if n_enc > n_frames:
        pixels = torch.cat([pixels, pixels[-1:].repeat(n_enc - n_frames, 1, 1, 1)])
    ctrl = encode(vae, pixels, device).float()
    del vae, pixels
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pred = predict_video(model, args.tap, ctrl, intr, args.mode, device,
                           tile_w=args.tile_w)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    f_use = n_frames                    # predictions on padded frames are dropped
    if len(pred["global_orient"]) < f_use:
        raise RuntimeError(f"predicted {len(pred['global_orient'])} < {f_use} frames")

    # exists_2d is the in-frame head: is this hand visible in THIS frame. That is
    # the question presence answers, so that is what the gate reads. (exists_3d
    # also stays on for hands that are occluded or have left the frame.)
    claim = np.asarray(pred["exists_2d"][:f_use]) > 0.5           # (F, 2) bool
    n_on = claim.sum(0)
    print(f"[presence] left claimed on {n_on[0]}/{f_use} frames, "
          f"right on {n_on[1]}/{f_use}")

    def _pack(idx: np.ndarray) -> dict:
        rec = {k: pred[k][idx].astype(np.float32) for k in DUMP_KEYS}
        n = len(idx)
        on = claim[idx]                                            # (n, 2)

        def gate(a: np.ndarray) -> np.ndarray:
            """NaN the unclaimed rows, so ignoring `claim` fails loudly."""
            return np.where(on.reshape((n, 2) + (1,) * (a.ndim - 2)),
                            a, np.float32("nan"))

        return {
            "go": gate(rec["global_orient"].reshape(n, 2, 1, 3, 3)),
            "hp": gate(rec["hand_pose"]),
            "betas": gate(rec["betas"]),
            "cam_trans": gate(rec["cam_trans"]),
            # how go / hp / betas decode to a hand: "native" or "right_mirror"
            # (ace_ego_hand/mano_utils.py, MANO_CONVENTIONS)
            "mano_convention": mano_convention(opt),
            # no ground truth here: slots are fixed, slot 0 = left, slot 1 = right
            "is_right": np.tile(np.array([[0.0, 1.0]], np.float32), (n, 1)),
            # the presence decision; every pose array is NaN where it is False
            "claim": on,
            # the raw presence heads, ungated
            "exists_2d": rec["exists_2d"],
            "exists_3d": rec["exists_3d"],
            "joints_2d": gate(rec["direct_joints2d"]),
            "joints_cam_direct": gate(rec["direct_joints_cam"]),
            "wrist_cam_direct": gate(rec["direct_wrist_cam"]),
            "intrinsics": intr,
            # K-free: the effective pinhole fitted from the ray field the model
            # predicted off the image. This -- not `intrinsics` -- is the camera
            # to project these predictions through when none was supplied.
            "pred_intrinsics": pred.get("_pred_K"),
        }

    if args.split_segments:
        # benchmark dump format: one pkl per 81-frame segment. The 81 comes from
        # the training window (21 latent frames -> 4*(21-1)+1 px), and the
        # evaluator consumes segments -- it is a file convention, not a limit on
        # the forward pass, which already ran over the whole clip above. The
        # last segment holds the remaining frames and may be shorter than 81.
        n_seg = (f_use + 80) // 81
        for i in range(n_seg):
            idx = np.arange(81 * i, min(81 * i + 81, f_use))
            with open(out_dir / f"{stem}_seg_{i:03d}.pkl", "wb") as f:
                pickle.dump(_pack(idx), f)
        print(f"[done] {n_seg} segment pkl(s) -> {out_dir}")
    else:
        out_path = out_dir / f"{stem}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(_pack(np.arange(f_use)), f)
        print(f"[done] {f_use} frames -> {out_path}")


if __name__ == "__main__":
    main()
