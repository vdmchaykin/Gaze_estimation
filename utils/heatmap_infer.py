from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Any
import numpy as np
import cv2
import torch


from models.heatmap_net import HeatmapNet


@dataclass
class PupilPrediction:
    x: float
    y: float
    conf: float


def _strip_module_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        out[k.replace("module.", "")] = v
    return out


def load_heatmapnet(checkpoint_path: str, device: str) -> torch.nn.Module:
    """
    Loads HeatmapNet checkpoint robustly.
    Supports checkpoints that are either:
      - raw state_dict
      - dict with 'state_dict'
    """
    model = HeatmapNet()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    state = _strip_module_prefix(state)

    # strict=False to survive small mismatches (e.g., missing keys)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        print("[HeatmapNet] Unexpected keys:", unexpected)
    if len(missing) > 0:
        print("[HeatmapNet] Missing keys:", missing)

    model.to(device)
    model.eval()
    return model


def preprocess_eye(gray: np.ndarray, input_size: int) -> torch.Tensor:
    """
    gray: (H,W) uint8
    Returns tensor (1,3,input_size,input_size) float32 in [0,1]
    """
    img = cv2.resize(gray, (input_size, input_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img3 = np.repeat(img[None, ...], 3, axis=0)  # (3,H,W)
    x = torch.from_numpy(img3).unsqueeze(0)       # (1,3,H,W)
    return x


def extract_single_peak(heatmap_2d: np.ndarray, sigma: float) -> Tuple[int, int, float]:
    """
    heatmap_2d: (h,w) float
    returns (px, py, conf) in heatmap coordinates
    """
    hm = heatmap_2d.astype(np.float32)

    if sigma > 0:
        # kernel size ~ 6*sigma, make it odd and >=3
        k = int(max(3, int(round(sigma * 6)) | 1))
        hm = cv2.GaussianBlur(hm, (k, k), sigmaX=sigma, sigmaY=sigma)

    conf = float(hm.max())
    idx = int(hm.argmax())
    h, w = hm.shape
    py = idx // w
    px = idx % w
    return px, py, conf


@torch.no_grad()
def infer_pupil_from_heatmapnet(
    model: torch.nn.Module,
    eye_gray: np.ndarray,
    device: str,
    input_size: int,
    kp_index: int,
    sigma: float,
) -> PupilPrediction:
    """
    Run HeatmapNet on one eye image and return pupil center in ORIGINAL eye crop coordinates.
    """
    x = preprocess_eye(eye_gray, input_size).to(device)  # (1,3,S,S)
    out = model(x)

    # Support outputs like:
    # - Tensor: (1,K,h,w)
    # - dict with 'heatmaps'
    if isinstance(out, dict):
        if "heatmaps" in out:
            out = out["heatmaps"]
        else:
            # fallback: take first tensor-like value
            out = next(v for v in out.values() if torch.is_tensor(v))

    if not torch.is_tensor(out):
        raise TypeError(f"HeatmapNet output type not supported: {type(out)}")

    # shape: (1,K,h,w)
    hm = out[0, kp_index].detach().cpu().numpy()
    px, py, conf = extract_single_peak(hm, sigma=sigma)

    # Map peak from heatmap coords -> input_size -> original crop coords
    H0, W0 = eye_gray.shape[:2]
    h, w = hm.shape

    # heatmap -> input_size (continuous)
    x_in = (px / max(1, (w - 1))) * (input_size - 1)
    y_in = (py / max(1, (h - 1))) * (input_size - 1)

    # input_size -> original crop
    x0 = (x_in / max(1, (input_size - 1))) * (W0 - 1)
    y0 = (y_in / max(1, (input_size - 1))) * (H0 - 1)

    return PupilPrediction(x=float(x0), y=float(y0), conf=float(conf))


def draw_cross(gray: np.ndarray, x: float, y: float, size: int = 6) -> np.ndarray:
    """
    Returns BGR image with a green cross drawn at (x,y).
    """
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    xi = int(round(x))
    yi = int(round(y))
    cv2.line(img, (xi - size, yi), (xi + size, yi), (0, 255, 0), 1)
    cv2.line(img, (xi, yi - size), (xi, yi + size), (0, 255, 0), 1)
    return img
