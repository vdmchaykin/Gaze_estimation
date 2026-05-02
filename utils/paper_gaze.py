from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import cv2


# ----------------- IO -----------------

def read_world_timestamps_csv(path: str) -> pd.DataFrame:
    """
    Neon обычно даёт world_timestamps.csv со столбцом 'timestamp [ns]'
    и каким-то индексом кадра (или без него).
    Мы стараемся угадать столбец кадра. Если его нет — считаем, что row index = frame_idx.
    """
    df = pd.read_csv(path)
    if "timestamp [ns]" not in df.columns:
        raise KeyError(f"world_timestamps.csv missing 'timestamp [ns]': {path}")
    df["timestamp [ns]"] = df["timestamp [ns]"].astype("int64")

    # попробуем найти колонку кадра
    frame_col = None
    candidates = [c for c in df.columns if ("frame" in c.lower() and ("idx" in c.lower() or "index" in c.lower()))]
    if len(candidates) > 0:
        frame_col = candidates[0]
    else:
        # иногда просто "frame"
        candidates = [c for c in df.columns if c.lower() == "frame" or c.lower() == "frame_idx"]
        if len(candidates) > 0:
            frame_col = candidates[0]

    if frame_col is None:
        df["world_frame_idx"] = np.arange(len(df), dtype=np.int64)
    else:
        df["world_frame_idx"] = df[frame_col].astype("int64")

    return df[["timestamp [ns]", "world_frame_idx"]].sort_values("timestamp [ns]").reset_index(drop=True)


def load_paper_homographies(npz_path: str):
    z = np.load(npz_path)
    return {
        "frame_idx": z["frame_idx"].astype(np.int64),
        "H_world_to_paper": z["H_world_to_paper"].astype(np.float64),
        "H_valid": z["H_valid"].astype(bool),
        "H_age_s": z["H_age_s"].astype(np.float32),
        "fps": float(z["fps"]) if "fps" in z else None,
        "paper_w": int(z["paper_w"]) if "paper_w" in z else None,
        "paper_h": int(z["paper_h"]) if "paper_h" in z else None,
    }


# ----------------- matching -----------------

def attach_world_frame_idx_to_gaze(gaze: pd.DataFrame, world_ts: pd.DataFrame, tolerance_ms: float) -> pd.DataFrame:
    """
    Для каждого gaze timestamp найдём ближайший world frame timestamp.
    """
    g = gaze.copy()
    g["timestamp [ns]"] = g["timestamp [ns]"].astype("int64")
    w = world_ts.copy()
    w["timestamp [ns]"] = w["timestamp [ns]"].astype("int64")

    tol_ns = int(tolerance_ms * 1e6)
    out = pd.merge_asof(
        g.sort_values("timestamp [ns]"),
        w.sort_values("timestamp [ns]"),
        on="timestamp [ns]",
        direction="nearest",
        tolerance=tol_ns,
    )
    return out


def _nearest_index(sorted_arr: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    For each q, returns index of nearest element in sorted_arr.
    """
    pos = np.searchsorted(sorted_arr, q, side="left")
    pos = np.clip(pos, 0, len(sorted_arr) - 1)

    left = np.clip(pos - 1, 0, len(sorted_arr) - 1)
    right = pos

    dl = np.abs(sorted_arr[left] - q)
    dr = np.abs(sorted_arr[right] - q)

    use_left = dl <= dr
    return np.where(use_left, left, right)


def lookup_H_for_world_frames(world_frame_idx: np.ndarray, Hpack: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      H: (N,3,3)
      valid: (N,)
      age_s: (N,)
    Picks nearest stored frame_idx in the npz (works even if tracker used frame_step>1).
    """
    fi = Hpack["frame_idx"]
    Hs = Hpack["H_world_to_paper"]
    valid = Hpack["H_valid"]
    age = Hpack["H_age_s"]

    idx = _nearest_index(fi, world_frame_idx.astype(np.int64))
    return Hs[idx], valid[idx], age[idx]


# ----------------- geometry -----------------

def world_px_to_paper_uv(xw: np.ndarray, yw: np.ndarray, Hs: np.ndarray, paper_w: int, paper_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch transform world points -> paper UV using per-row homographies.
    """
    n = len(xw)
    u = np.full(n, np.nan, dtype=np.float32)
    v = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        H = Hs[i]
        p = np.array([[[float(xw[i]), float(yw[i])]]], dtype=np.float32)
        q = cv2.perspectiveTransform(p, H)[0, 0]
        u[i] = float(q[0]) / float(paper_w - 1)
        v[i] = float(q[1]) / float(paper_h - 1)

    return u, v


@dataclass
class PaperGazeConfig:
    tolerance_ms: float = 10.0     # match gaze -> world frame
    max_age_s: float = 2.0         # drop if H is too old
    uv_margin: float = 0.15        # keep u,v within [-m .. 1+m]


def build_paper_gaze_targets(
    gaze_df: pd.DataFrame,
    world_ts_df: pd.DataFrame,
    Hpack: dict,
    cfg: PaperGazeConfig,
) -> pd.DataFrame:
    """
    Input gaze_df must contain:
      timestamp [ns], gaze x [px], gaze y [px]
    Output adds:
      world_frame_idx, paper_u, paper_v, H_ok
    and filters out rows with invalid/too-old H or crazy uv.
    """
    g = attach_world_frame_idx_to_gaze(gaze_df, world_ts_df, cfg.tolerance_ms)
    g = g.dropna(subset=["world_frame_idx", "gaze x [px]", "gaze y [px]"]).reset_index(drop=True)

    wf = g["world_frame_idx"].astype(np.int64).to_numpy()
    Hs, Hvalid, Hage = lookup_H_for_world_frames(wf, Hpack)

    paper_w = int(Hpack["paper_w"])
    paper_h = int(Hpack["paper_h"])

    u, v = world_px_to_paper_uv(
        g["gaze x [px]"].astype(np.float32).to_numpy(),
        g["gaze y [px]"].astype(np.float32).to_numpy(),
        Hs,
        paper_w,
        paper_h,
    )

    g["paper_u"] = u
    g["paper_v"] = v
    g["H_ok"] = Hvalid & (Hage <= cfg.max_age_s)

    # drop invalid H
    g = g[g["H_ok"]].copy()

    # drop crazy uv
    m = float(cfg.uv_margin)
    g = g[(g["paper_u"] >= -m) & (g["paper_u"] <= 1.0 + m) & (g["paper_v"] >= -m) & (g["paper_v"] <= 1.0 + m)].copy()

    return g.reset_index(drop=True)
