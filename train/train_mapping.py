from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import os
import json
import numpy as np
import pandas as pd


@dataclass
class MappingConfig:
    alpha: float = 1e-2
    poly_degree: int = 2


# ---------- feature building ----------

def _poly2(x: float, y: float) -> np.ndarray:
    # degree=2: [x, y, x^2, x*y, y^2]
    return np.array([x, y, x*x, x*y, y*y], dtype=np.float32)

def _poly3(x: float, y: float) -> np.ndarray:
    # degree=3 adds cubic terms
    return np.array([x, y, x*x, x*y, y*y, x**3, (x**2)*y, x*(y**2), y**3], dtype=np.float32)

def _poly_feats(x: float, y: float, degree: int) -> np.ndarray:
    if degree <= 1:
        return np.array([x, y], dtype=np.float32)
    if degree == 2:
        return _poly2(x, y)
    if degree == 3:
        return _poly3(x, y)
    raise ValueError("Supported poly_degree: 1,2,3")

def _norm01_to_m11(v: float, vmax: float) -> float:
    # v in [0..vmax] -> [-1..1]
    if vmax <= 1:
        return 0.0
    return (v / (vmax - 1.0)) * 2.0 - 1.0

def build_features(df: pd.DataFrame, eye_w: int, eye_h: int, poly_degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input df columns: xL,yL,confL,xR,yR,confR, gaze x [px], gaze y [px]
    Output:
      X: (N, D)
      Y: (N, 2)
    """
    X = []
    Y = []

    for _, r in df.iterrows():
        xL = _norm01_to_m11(float(r["xL"]), eye_w)
        yL = _norm01_to_m11(float(r["yL"]), eye_h)
        xR = _norm01_to_m11(float(r["xR"]), eye_w)
        yR = _norm01_to_m11(float(r["yR"]), eye_h)

        fL = _poly_feats(xL, yL, poly_degree)
        fR = _poly_feats(xR, yR, poly_degree)

        confL = float(r["confL"])
        confR = float(r["confR"])

        feats = np.concatenate([fL, fR, np.array([confL, confR], dtype=np.float32)], axis=0)
        X.append(feats)

        Y.append([float(r["paper_u"]), float(r["paper_v"])])

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    return X, Y


# ---------- ridge regression ----------

def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Closed-form ridge:
      W = (Xb^T Xb + alpha I)^-1 Xb^T Y
    where Xb = [X, 1]
    Returns W: (D+1, 2)
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    ones = np.ones((X.shape[0], 1), dtype=np.float32)
    Xb = np.concatenate([X, ones], axis=1)  # (N, D+1)

    D = Xb.shape[1]
    A = Xb.T @ Xb + alpha * np.eye(D, dtype=np.float32)
    B = Xb.T @ Y
    W = np.linalg.solve(A, B)
    return W

def predict_ridge(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    ones = np.ones((X.shape[0], 1), dtype=np.float32)
    Xb = np.concatenate([X, ones], axis=1)
    return Xb @ W  # (N,2)


# ---------- metrics ----------

def pixel_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    d = pred - gt
    return np.sqrt((d * d).sum(axis=1))

def compute_metrics(err: np.ndarray) -> Dict[str, float]:
    err = np.asarray(err, dtype=np.float32)
    return {
        "n": int(err.shape[0]),
        "mean_px": float(err.mean()),
        "median_px": float(np.median(err)),
        "p90_px": float(np.quantile(err, 0.90)),
        "p95_px": float(np.quantile(err, 0.95)),
        "max_px": float(err.max()),
    }

def paper_uv_errors_px(pred_uv: np.ndarray, gt_uv: np.ndarray, paper_w: int, paper_h: int) -> np.ndarray:
    d = pred_uv - gt_uv
    dx = d[:, 0] * (paper_w - 1)
    dy = d[:, 1] * (paper_h - 1)
    return np.sqrt(dx*dx + dy*dy)


# ---------- main train/eval ----------

def train_and_validate(train_df, val_df, eye_w, eye_h, paper_w, paper_h, cfg, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    Xtr, Ytr = build_features(train_df, eye_w, eye_h, cfg.poly_degree)
    Xva, Yva = build_features(val_df, eye_w, eye_h, cfg.poly_degree)

    W = fit_ridge(Xtr, Ytr, alpha=cfg.alpha)
    pred = predict_ridge(W, Xva)

    err = paper_uv_errors_px(pred, Yva, paper_w=paper_w, paper_h=paper_h)
    metrics = compute_metrics(err)


    # save model
    np.savez(
        os.path.join(out_dir, "gaze_mapping_model.npz"),
        W=W,
        alpha=np.float32(cfg.alpha),
        poly_degree=np.int32(cfg.poly_degree),
        eye_w=np.int32(eye_w),
        eye_h=np.int32(eye_h),
    )

    # save val predictions (useful for demo)
    val_out = val_df.copy().reset_index(drop=True)
    val_out["pred_x"] = pred[:, 0]
    val_out["pred_y"] = pred[:, 1]
    val_out.to_csv(os.path.join(out_dir, "val_predictions.csv"), index=False)

    # save metrics
    with open(os.path.join(out_dir, "metrics_val.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
