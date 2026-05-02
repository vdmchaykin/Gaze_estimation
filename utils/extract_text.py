import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def extract_test_intervals(events_csv_path: str) -> List[Tuple[str, int, int]]:
    ev = pd.read_csv(events_csv_path).sort_values("timestamp [ns]").reset_index(drop=True)
    names = ev["name"].astype(str)
    begin = ev[names.str.match(r"^test\d+_begin$")].copy()
    end   = ev[names.str.match(r"^test\d+_end$")].copy()

    def key(name: str) -> Optional[str]:
        m = re.match(r"^(test\d+)_", str(name))
        return m.group(1) if m else None

    begin["k"] = begin["name"].map(key)
    end["k"]   = end["name"].map(key)

    intervals = []
    for k, b in begin.groupby("k"):
        b_ts = int(b["timestamp [ns]"].iloc[0])
        e_row = end[end["k"] == k]
        if len(e_row) == 0:
            continue
        e_ts = int(e_row["timestamp [ns]"].iloc[0])
        if e_ts > b_ts:
            intervals.append((k, b_ts, e_ts))
    return sorted(intervals, key=lambda x: x[1])


def mask_by_intervals(timestamps_ns: np.ndarray, intervals: List[Tuple[str, int, int]]) -> np.ndarray:
    t = timestamps_ns.astype(np.int64)
    m = np.zeros_like(t, dtype=bool)
    for _, a, b in intervals:
        m |= (t >= a) & (t <= b)
    return m


def extract_single_peak(heatmap: np.ndarray, sigma: float = 1.2):
    hm = heatmap.astype(np.float32)
    if sigma and sigma > 0:
        k = max(3, int(round(sigma * 6)) | 1)
        hm = cv2.GaussianBlur(hm, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)
    flat_idx = int(np.argmax(hm))
    y, x = divmod(flat_idx, hm.shape[1])
    conf = float(hm[y, x])
    return x, y, conf


def draw_cross(img: np.ndarray, x: int, y: int, size: int = 6, thickness: int = 2):
    h, w = img.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    cv2.line(img, (x - size, y), (x + size, y), (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def prep_gray_to_tensor(img_gray: np.ndarray, input_size: int, device: str) -> torch.Tensor:
    img = cv2.resize(img_gray, (input_size, input_size), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = np.stack([x, x, x], axis=0)  # (3,H,W)
    return torch.from_numpy(x).unsqueeze(0).to(device)  # (1,3,H,W)


def summarize(errors: np.ndarray):
    errors = np.asarray(errors, dtype=np.float32)
    return {
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "p90": float(np.quantile(errors, 0.90)),
        "p95": float(np.quantile(errors, 0.95)),
        "max": float(errors.max()),
        "n": int(errors.shape[0]),
    }


'''
intervals = extract_test_intervals("events.csv")

gaze = pd.read_csv("gaze.csv")
m = mask_by_intervals(gaze["timestamp [ns]"].values, intervals)
gaze_test = gaze[m].copy()

'''