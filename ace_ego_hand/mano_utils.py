"""MANO utility functions: rotation conversions, forward passes, initialization.

Single canonical home for all MANO helpers + MANO conventions.

Conventions (single source of truth — used by the projector, models/ and viz):
  N_JOINTS = 16  — MANO native (matches smplx output.joints):
                    0=wrist, 1-3=index, 4-6=middle, 7-9=pinky,
                    10-12=ring, 13-15=thumb. NO fingertip vertices.
  N_SLOTS  = 2   — left + right hand. Slot 0 = left, slot 1 = right.
"""
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F


# MANO conventions (constants used across the codebase).
N_JOINTS: int = 16
N_SLOTS: int = 2

# MANO model files are external licensed DATA (not redistributed here); see
# README. Single source of truth for all `smplx.create(MANO_CHECKPOINT_DIR, ...)`
# calls. Override with the ACE_EGO_HAND_MANO_DIR environment variable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANO_CHECKPOINT_DIR = os.environ.get(
    "ACE_EGO_HAND_MANO_DIR", os.path.join(_REPO_ROOT, "data", "mano"))


# ── Rotation conversions ──────────────────────────────────────────────

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Args: x (B, 6) or any shape ending in 6
    Returns: (B, 3, 3) rotation matrices
    """
    orig_dtype = x.dtype
    x = x.float()
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1).to(orig_dtype)




def rotmat_to_axis_angle(R):
    """Differentiable (B, 3, 3) -> (B, 3) via Rodrigues inverse."""
    cos_angle = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cos_angle)
    axis = torch.stack([R[:, 2, 1] - R[:, 1, 2],
                        R[:, 0, 2] - R[:, 2, 0],
                        R[:, 1, 0] - R[:, 0, 1]], dim=-1)
    norm = torch.norm(axis, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis / norm
    return axis * angle.unsqueeze(-1)


# ── MANO forward ──────────────────────────────────────────────────────

# 5 finger-tip vertex indices in the 778-vertex MANO mesh (OpenPose order).
_MANO_TIP_IDS = [744, 320, 443, 554, 671]
# Remap smplx's 16-joint output + 5 tips → 21 OpenPose joints.
_MANO_TO_OP = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]




def mano_forward_batch_full(go_rotmat, hp_rotmat, betas, mano_model):
    """Batched differentiable MANO forward, returns ``(joints_21, vertices_778)``.

    Args:
        go_rotmat: (B, 3, 3) or (B, 1, 3, 3) global orient rotmats
        hp_rotmat: (B, 15, 3, 3) hand-pose rotmats
        betas:     (B, 10)
        mano_model: smplx MANO module (native convention) or a
            ``RightMirrorFullRotationMano`` (right-mirror convention), on the
            same device as the inputs.

    Returns:
        ``(joints_21, vertices_778)`` torch tensors on the model device.
    """
    B = go_rotmat.shape[0]
    if hasattr(mano_model, "forward_rotmats"):       # full-rotation convention
        out = mano_model.forward_rotmats(go_rotmat, hp_rotmat, betas)
    else:
        go_aa = rotmat_to_axis_angle(go_rotmat.reshape(B, 3, 3))             # (B, 3)
        hp_aa = rotmat_to_axis_angle(hp_rotmat.reshape(B * 15, 3, 3)).reshape(B, 45)
        out = mano_model(global_orient=go_aa, hand_pose=hp_aa, betas=betas)
    tips = out.vertices[:, _MANO_TIP_IDS]                                    # (B, 5, 3)
    joints_21 = torch.cat([out.joints, tips], dim=1)[:, _MANO_TO_OP]         # (B, 21, 3)
    return joints_21, out.vertices








# ── MANO conventions ──────────────────────────────────────────────────

# How a checkpoint's MANO parameters are decoded. It is fixed by the labels the
# checkpoint was trained on, so it is set in the model config
# (``geometry.mano_convention``) and written to every prediction file.
#   native       — the paper checkpoints. Left slot = MANO_LEFT, right slot =
#                  MANO_RIGHT; ``hand_pose`` is relative to the MANO mean pose
#                  (smplx ``flat_hand_mean=False``).
#   right_mirror — the EgoHandBench-retrained checkpoint. Both slots use
#                  MANO_RIGHT with full local rotations (mean pose included);
#                  the left hand is the right-hand output reflected in x. This
#                  is the EgoHandBench label format.
MANO_CONVENTIONS = ("native", "right_mirror")


def mano_convention(opt: dict) -> str:
    convention = (opt.get("geometry") or {}).get("mano_convention", "native")
    if convention not in MANO_CONVENTIONS:
        raise ValueError(f"unknown geometry.mano_convention {convention!r}; "
                         f"expected one of {MANO_CONVENTIONS}")
    return convention


class RightMirrorFullRotationMano(torch.nn.Module):
    """MANO_RIGHT driven by full rotations; the left slot reflects the output in x.

    ``forward_rotmats`` returns the 16 MANO joints and 778 vertices before any
    translation. Rotations go straight to LBS: no mean pose is added.
    """

    def __init__(self, is_right: bool):
        super().__init__()
        import smplx
        self.is_right = bool(is_right)
        self.layer = smplx.MANOLayer(os.path.join(MANO_CHECKPOINT_DIR, "mano"),
                                     is_rhand=True, num_pca_comps=45).eval()
        for parameter in self.layer.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("output_sign",
                             torch.tensor([1.0 if is_right else -1.0, 1.0, 1.0]))

    @property
    def faces(self):
        # the reflection flips the triangle winding
        faces = self.layer.faces.copy()
        return faces if self.is_right else faces[:, [0, 2, 1]]

    def forward_rotmats(self, go, hp, betas):
        n = betas.shape[0]
        with torch.autocast(device_type=self.output_sign.device.type, enabled=False):
            out = self.layer(global_orient=go.float().reshape(n, 1, 3, 3),
                             hand_pose=hp.float().reshape(n, 15, 3, 3),
                             betas=betas.float())
            return SimpleNamespace(vertices=out.vertices * self.output_sign,
                                   joints=out.joints[:, :16] * self.output_sign)


def _right_mirror_models(device) -> dict:
    return {is_right: RightMirrorFullRotationMano(is_right).to(device).eval()
            for is_right in (False, True)}


def init_mano_and_renderer(render_mode="solid", device="cpu", flat_hand_mean=False,
                           convention="native"):
    """Initialize MANO models.

    The second return value is a mesh-renderer slot kept for interface
    compatibility: training and evaluation never use it, and this release does
    not ship a visualization backend, so it is always ``None``.

    Args:
        render_mode: unused (kept for interface compatibility).
        device: torch device where the MANO models live. GPU makes batched
            forward passes substantially faster.
        flat_hand_mean: smplx convention. ``False`` (default) matches HaMeR /
            WiLoR / Hot3D / HoI4D / H2O / Arctic baseline pkls. ``True``
            matches manotorch / OakInk2 mocap pkls.
        convention: ``"native"`` or ``"right_mirror"`` (see ``MANO_CONVENTIONS``);
            use the ``mano_convention`` stored in the prediction file.

    Returns:
        ``(mano_models, mesh_renderer)`` where ``mano_models`` is
        ``{True: mano_right, False: mano_left}`` on ``device``; ``mesh_renderer``
        is always ``None``.
    """
    if convention == "right_mirror":
        return _right_mirror_models(device), None
    if convention != "native":
        raise ValueError(f"unknown MANO convention {convention!r}")
    import smplx
    mano_path = MANO_CHECKPOINT_DIR
    mano_r = smplx.create(mano_path, model_type='mano', is_rhand=True,
                          use_pca=False, flat_hand_mean=flat_hand_mean).to(device).eval()
    mano_l = smplx.create(mano_path, model_type='mano', is_rhand=False,
                          use_pca=False, flat_hand_mean=flat_hand_mean).to(device).eval()
    mano_models = {True: mano_r, False: mano_l}
    return mano_models, None


def setup_mano_models(opt: dict, device: torch.device):
    if mano_convention(opt) == "right_mirror":
        print(f"Loaded right-mirror full-rotation MANO on {device} for the "
              "projector's translation decode")
        return _right_mirror_models(device)
    import smplx  # noqa: WPS433

    models = {}
    for is_right in (False, True):
        mano = smplx.create(
            MANO_CHECKPOINT_DIR,
            model_type="mano",
            is_rhand=is_right,
            use_pca=False,
            flat_hand_mean=False,
        ).to(device).eval()
        for param in mano.parameters():
            param.requires_grad_(False)
        models[is_right] = mano
    print(f"Loaded MANO models on {device} for the projector's translation decode")
    return models
