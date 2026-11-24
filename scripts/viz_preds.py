#!/usr/bin/env python3
"""Render prediction pickles as overlay videos.

Reads the pkl(s) infer_video.py wrote for a clip (one whole-clip <stem>.pkl, or
<stem>_seg_<NNN>.pkl from --split_segments), runs the MANO forward on the
predicted parameters, projects the camera-space joints and draws the two hands
over the source video (left blue, right pink). Hands the dump does not claim
are not drawn; their parameters are NaN in the pkl.

  python scripts/viz_preds.py --pred_dir results/clip --video clip.mp4 \
      --out results/clip_viz --mesh

Rendering is CPU-only; no GPU mesh renderer is required. --mesh switches from
the skeleton overlay to a shaded MANO mesh (a small painter's-algorithm
soft-rasterizer on top of cv2). --use_joints2d draws the direct 2D-head output
instead of the projected MANO joints.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ace_ego_hand.mano_utils import init_mano_and_renderer, mano_forward_batch_full  # noqa: E402

# OpenPose-21 hand skeleton: wrist -> [root..tip] per finger.
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),        # thumb
          (0, 5), (5, 6), (6, 7), (7, 8),        # index
          (0, 9), (9, 10), (10, 11), (11, 12),   # middle
          (0, 13), (13, 14), (14, 15), (15, 16), # ring
          (0, 17), (17, 18), (18, 19), (19, 20)] # pinky
# BGR: slot 0 = left hand (blue), slot 1 = right hand (pink) — fixed slot order.
_COLORS = [(230, 180, 90), (180, 105, 255)]


def _abspath(p: str) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else ROOT / p)


def load_pred_video(pred_dir: Path, stem: str, seg_ids: list[int]) -> dict:
    """Concatenate `<stem>_seg_<NNN>.pkl` along time -> (81*n_seg, ...) arrays."""
    ids = sorted(seg_ids)
    if ids != list(range(len(ids))):
        raise ValueError(f"non-contiguous segment ids {ids} (broken/partial dump)")
    parts = []
    for i in ids:
        with open(pred_dir / f"{stem}_seg_{i:03d}.pkl", "rb") as f:
            parts.append(pickle.load(f))
    # per-frame arrays only (whole-video dumps also carry e.g. pred_intrinsics)
    return {k: np.concatenate([p[k] for p in parts], axis=0)
            for k in parts[0] if isinstance(parts[0][k], np.ndarray)}


def mano_cam_joints(pred: dict, mano_models: dict, device: str,
                    want_verts: bool = False):
    """Predicted MANO params -> camera-space 21 joints (F, 2, 21, 3).

    With ``want_verts`` additionally returns the 778 mesh vertices
    (F, 2, 778, 3); otherwise the second return value is ``None``.
    """
    F = pred["cam_trans"].shape[0]
    out = np.zeros((F, 2, 21, 3), np.float32)
    verts = np.zeros((F, 2, 778, 3), np.float32) if want_verts else None
    # unclaimed rows are NaN: run them through MANO as zeros (they are never
    # drawn) and put the NaN back on the output
    def arr(key: str, slot: int) -> np.ndarray:
        return np.nan_to_num(np.asarray(pred[key][:, slot], np.float32))

    with torch.no_grad():
        for slot, is_right in ((0, False), (1, True)):
            go = torch.from_numpy(arr("go", slot)).reshape(F, 3, 3).to(device)
            hp = torch.from_numpy(arr("hp", slot)).to(device)
            betas = torch.from_numpy(arr("betas", slot)).to(device)
            j21, v = mano_forward_batch_full(go, hp, betas, mano_models[is_right])
            bad = ~np.isfinite(pred["cam_trans"][:, slot]).all(-1)
            trans = np.where(bad[:, None], np.float32("nan"),
                             pred["cam_trans"][:, slot])[:, None, :]
            out[:, slot] = j21.cpu().numpy() + trans
            if want_verts:
                verts[:, slot] = v.cpu().numpy() + trans
    return out, verts




def project(j3d: np.ndarray, intr: dict, frame_w: int, frame_h: int):
    """(...,21,3) cam-space -> pixel (...,21,2) at the actual frame size + z."""
    sx = frame_w / float(intr["image_width"])
    sy = frame_h / float(intr["image_height"])
    if intr.get("fisheye") or str(intr.get("model", "")).lower() in ("mei", "kb", "fisheye"):
        # A fisheye rig would need its own forward model here; the pinhole
        # projection below is wrong at wide angles (on a MEI rig it puts the
        # mesh at roughly 7x the correct image offset). Refuse rather than draw
        # a confidently wrong overlay.
        raise NotImplementedError(
            "Fisheye intrinsics are not supported by this release. Undistort "
            "to a pinhole view first."
        )
    z = np.clip(j3d[..., 2], 1e-4, None)
    u = (float(intr["fx"]) * j3d[..., 0] / z + float(intr["cx"])) * sx
    v = (float(intr["fy"]) * j3d[..., 1] / z + float(intr["cy"])) * sy
    return np.stack([u, v], -1), j3d[..., 2]


def draw_hands(frame: np.ndarray, j2d: np.ndarray, show: np.ndarray) -> np.ndarray:
    """Draw the two-hand skeleton in place. j2d (2,21,2) px, show (2,) bool."""
    h, w = frame.shape[:2]
    for slot in (0, 1):
        if not show[slot]:
            continue
        pts = j2d[slot]
        finite = np.isfinite(pts).all(-1)
        # skip hands projecting entirely outside the frame
        inside = finite & (pts[:, 0] >= 0) & (pts[:, 0] < w) \
            & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        if not inside.any():
            continue
        color = _COLORS[slot]
        for a, b in _EDGES:
            if finite[a] and finite[b]:
                cv2.line(frame, tuple(np.round(pts[a]).astype(int)),
                         tuple(np.round(pts[b]).astype(int)), color, 2, cv2.LINE_AA)
        for j, p in enumerate(pts):
            if finite[j]:
                cv2.circle(frame, tuple(np.round(p).astype(int)), 3, color, -1,
                           cv2.LINE_AA)
    return frame


def draw_meshes(frame: np.ndarray, v2d: np.ndarray, v3d: np.ndarray,
                faces: dict, show: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Soft-rasterize the two MANO meshes over the frame (painter's algorithm).

    v2d (2,778,2) pixel coords, v3d (2,778,3) camera-space verts,
    faces {slot: (T,3) int}, show (2,) bool. Per-face Lambert-ish shading from
    the camera-space normal; faces of both hands are depth-sorted together so
    inter-hand occlusion is correct. Pure cv2 — no GPU renderer needed.
    """
    h, w = frame.shape[:2]
    tris = []                                  # (depth, (3,2) int pts, color)
    for slot in (0, 1):
        if not show[slot]:
            continue
        f = faces[slot]
        tz = v3d[slot][:, 2][f]                # (T, 3) per-vertex depth
        t2d = v2d[slot][f]                     # (T, 3, 2)
        # cull non-finite faces, faces behind the camera, and faces fully
        # outside the frame (per-face, so one bad vertex hides one face only)
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(t2d).all((-1, -2)) \
                & np.isfinite(v3d[slot][f]).all((-1, -2)) \
                & (tz > 1e-3).all(-1) \
                & (t2d[..., 0].max(-1) >= 0) & (t2d[..., 0].min(-1) < w) \
                & (t2d[..., 1].max(-1) >= 0) & (t2d[..., 1].min(-1) < h)
        if not ok.any():
            continue
        t3d = v3d[slot][f[ok]]                 # (T', 3, 3)
        nrm = np.cross(t3d[:, 1] - t3d[:, 0], t3d[:, 2] - t3d[:, 0])
        nz = np.abs(nrm[:, 2]) / np.clip(np.linalg.norm(nrm, axis=-1), 1e-9, None)
        shade = (0.35 + 0.65 * nz)[:, None]    # head-on faces brightest
        color = (shade * np.array(_COLORS[slot], np.float32)).astype(np.uint8)
        pts = np.round(np.clip(t2d[ok], [-2 * w, -2 * h], [2 * w, 2 * h])
                       ).astype(np.int32)
        depth = tz[ok].mean(-1)
        tris.extend(zip(depth, pts, color))
    if not tris:
        return frame
    layer = frame.copy()
    tris.sort(key=lambda t: -t[0])             # far -> near
    for _, pts, color in tris:
        cv2.fillConvexPoly(layer, pts, color.tolist(), cv2.LINE_AA)
    return cv2.addWeighted(frame, 1.0 - alpha, layer, alpha, 0.0, dst=frame)


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(frame, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def render_video(item: dict, pred: dict, gt_pack: dict, draw_gt: bool,
                 mano_models: dict, out_path: Path, device: str,
                 use_joints2d: bool, mesh: bool, max_w: int) -> int:
    cap = cv2.VideoCapture(item["control_file_path"])
    if not cap.isOpened():
        print(f"  SKIP {item['video_id']}: cannot open {item['control_file_path']}")
        return 0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(item.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30)
    intr = dict(gt_pack["intrinsics"])
    faces = {slot: np.asarray(mano_models[bool(slot)].faces, np.int64)
             for slot in (0, 1)}               # slot 0 = left, 1 = right

    F = min(n_vid, pred["cam_trans"].shape[0])
    pj3d, pverts = mano_cam_joints(pred, mano_models, device, want_verts=mesh)
    pj2d, pz = project(pj3d, intr, frame_w, frame_h)
    pv2d = project(pverts, intr, frame_w, frame_h)[0] if mesh else None
    # the dump's own presence decision; dumps without `claim` fall back to
    # exists_2d, the in-frame head it is derived from. Not exists_3d: that head
    # stays on for hands that are occluded or have left the frame.
    if "claim" in pred:
        p_show = np.asarray(pred["claim"], bool)
    else:
        p_show = pred["exists_2d"] > 0.5
    if use_joints2d:
        pj2d = pred["joints_2d"] * np.array([frame_w, frame_h], np.float32)
    else:
        # projected-MANO branch: also require a sane camera-space depth
        with np.errstate(invalid="ignore"):
            p_show = p_show & (pz.mean(-1) > 0.05)

    gj2d = g_show = gverts = gv2d = None
    scale = min(1.0, max_w / (frame_w * (2 if gj2d is not None else 1)))
    out_w = int(frame_w * scale) * (2 if gj2d is not None else 1)
    out_h = int(frame_h * scale)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (out_w, out_h))
    n = 0
    try:
        for f in range(F):
            ok, frame = cap.read()
            if not ok:
                break
            if mesh and pv2d is not None:
                panel = draw_meshes(frame.copy(), pv2d[f], pverts[f], faces,
                                    p_show[f])
            else:
                panel = draw_hands(frame.copy(), pj2d[f], p_show[f])
            panel = _label(panel, "pred")
            if gj2d is not None:
                if mesh and gv2d is not None:
                    gpanel = draw_meshes(frame, gv2d[f], gverts[f], faces,
                                         g_show[f])
                else:   # joints-only GT: skeleton fallback
                    gpanel = draw_hands(frame, gj2d[f], g_show[f])
                gpanel = _label(gpanel, "GT")
                panel = np.hstack([panel, gpanel])
            if scale < 1.0:
                panel = cv2.resize(panel, (out_w, out_h),
                                   interpolation=cv2.INTER_AREA)
            vw.write(panel)
            n += 1
    finally:
        vw.release()
        cap.release()
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred_dir", required=True,
                    help="dir with <stem>_seg_<NNN>.pkl prediction dumps")
    ap.add_argument("--video", required=True,
                    help="the source clip the predictions were computed from")
    ap.add_argument("--out", required=True, help="output dir for the mp4s")
    ap.add_argument("--mesh", action="store_true",
                    help="render the shaded MANO mesh instead of the skeleton "
                         "(CPU soft-rasterizer; slower than the skeleton mode)")
    ap.add_argument("--use_joints2d", action="store_true",
                    help="draw the direct 2D head instead of projected MANO joints")
    ap.add_argument("--limit", type=int, default=0, help="max videos (0 = all)")
    ap.add_argument("--max_w", type=int, default=1920,
                    help="cap output video width (downscales if larger)")
    ap.add_argument("--device", default="cpu", help="device for the MANO forward")
    args = ap.parse_args()
    if args.mesh and args.use_joints2d:
        ap.error("--mesh and --use_joints2d are mutually exclusive")

    pred_dir = Path(_abspath(args.pred_dir))
    # two dump shapes: infer_video.py writes one <stem>.pkl for the whole clip,
    # --split_segments writes <stem>_seg_<NNN>.pkl per 81 frames.
    segs = defaultdict(list)
    for p in pred_dir.glob("*_seg_*.pkl"):
        m = re.match(r"(.+)_seg_(\d+)\.pkl$", p.name)
        if m:
            segs[m.group(1)].append(int(m.group(2)))
    whole = {p.stem: p for p in pred_dir.glob("*.pkl")
             if not re.match(r".+_seg_\d+$", p.stem)}
    if not segs and not whole:
        print(f"no <stem>.pkl or <stem>_seg_<NNN>.pkl files in {pred_dir}")
        return 1

    # only the dump of --video itself: other clips' pkls in the same dir were
    # predicted on different footage and must not be drawn over this one
    stem0 = Path(args.video).stem
    if stem0 not in segs and stem0 not in whole:
        print(f"no pkl matching {stem0!r} in {pred_dir}")
        return 1
    items = {stem0: {"control_file_path": _abspath(args.video), "video_id": stem0}}
    mano_cache: dict = {}

    n_done = 0
    for stem in items:
        if args.limit and n_done >= args.limit:
            break
        item = items[stem]
        # infer_video.py records the camera it ran with in every pkl. Real
        # intrinsics win; under K-free the recorded one is a placeholder (it
        # carries a `source` marker) and the camera the model fitted from its
        # own ray field is what actually projects these joints.
        src = (pred_dir / f"{stem}_seg_{min(segs[stem]):03d}.pkl"
               if stem in segs else whole[stem])
        with open(src, "rb") as fh:
            first = pickle.load(fh)
        if "intrinsics" not in first:
            print(f"  SKIP {stem}: pkl carries no intrinsics")
            continue
        k = first["intrinsics"]
        if k.get("source") and first.get("pred_intrinsics"):
            k = first["pred_intrinsics"]
            print(f"  {stem}: no intrinsics were supplied — "
                  "projecting through the model-predicted camera")
        gt_pack = {"intrinsics": k}
        try:
            if stem in segs:
                pred = load_pred_video(pred_dir, stem, segs[stem])
            else:
                with open(whole[stem], "rb") as fh:
                    raw = pickle.load(fh)
                pred = {k: v for k, v in raw.items() if isinstance(v, np.ndarray)}
        except ValueError as e:
            print(f"  SKIP {stem}: {e}")
            continue
        # the convention the checkpoint decodes MANO with travels in the pkl
        convention = first.get("mano_convention", "native")
        if convention not in mano_cache:
            mano_cache[convention], _ = init_mano_and_renderer(
                device=args.device, convention=convention)
        mano_models = mano_cache[convention]
        out_path = Path(_abspath(args.out)) / f"{stem}.mp4"
        n = render_video(item, pred, gt_pack, False, mano_models, out_path,
                         args.device, args.use_joints2d, args.mesh, args.max_w)
        print(f"  {stem}: {n} frames -> {out_path}")
        n_done += 1
    print(f"DONE: {n_done} videos -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
