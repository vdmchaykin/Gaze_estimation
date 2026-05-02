import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Any, List

import cv2
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# =========================
# Utilities (unchanged logic)
# =========================

def initial_guess_darkest_patch(gray, search_rect, patch=24, step=8):
    x0, y0, x1, y1 = search_rect
    h, w = gray.shape
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w, x1); y1 = min(h, y1)

    best_val = 1e18
    best = (int((x0 + x1) / 2), int((y0 + y1) / 2))

    for y in range(y0, y1 - patch, step):
        for x in range(x0, x1 - patch, step):
            roi = gray[y:y + patch, x:x + patch]
            val = float(np.mean(roi))
            if val < best_val:
                best_val = val
                best = (x + patch // 2, y + patch // 2)
    return best


def draw_dashed_rect(img, x0, y0, x1, y1, color=(255, 255, 255), thickness=1, dash=8, gap=6):
    def dashed_line(p1, p2):
        xA, yA = p1
        xB, yB = p2
        length = int(np.hypot(xB - xA, yB - yA))
        if length <= 0:
            return
        vx = (xB - xA) / length
        vy = (yB - yA) / length
        t = 0
        while t < length:
            t2 = min(t + dash, length)
            xa, ya = int(xA + vx * t), int(yA + vy * t)
            xb, yb = int(xA + vx * t2), int(yA + vy * t2)
            cv2.line(img, (xa, ya), (xb, yb), color, thickness)
            t += dash + gap

    dashed_line((x0, y0), (x1, y0))
    dashed_line((x1, y0), (x1, y1))
    dashed_line((x1, y1), (x0, y1))
    dashed_line((x0, y1), (x0, y0))


def draw_step1_debug(frame_bgr, roi_rect, guess_center, pupil_square_size=140):
    out = frame_bgr.copy()
    x0, y0, x1, y1 = roi_rect
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)

    cx, cy = guess_center
    hs = pupil_square_size // 2
    px0, py0 = cx - hs, cy - hs
    px1, py1 = cx + hs, cy + hs
    cv2.rectangle(out, (px0, py0), (px1, py1), (255, 255, 255), 2)

    inner = int(pupil_square_size * 0.6)
    hs2 = inner // 2
    qx0, qy0 = cx - hs2, cy - hs2
    qx1, qy1 = cx + hs2, cy + hs2
    draw_dashed_rect(out, qx0, qy0, qx1, qy1, color=(255, 255, 255), thickness=1)

    cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
    return out


def parse_roi(roi_str, w, h):
    # If ROI provided explicitly — keep as is
    if roi_str.strip():
        x0, y0, x1, y1 = map(int, roi_str.split(","))
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(1, min(w, x1))
        y1 = max(1, min(h, y1))
        if x1 <= x0 + 10 or y1 <= y0 + 10:
            raise ValueError("ROI too small or invalid.")
        return (x0, y0, x1, y1)

    # Default ROI: full screen
    return (0, 0, w, h)


def make_writer(output_path, fps, frame_size):
    w, h = frame_size
    fps = float(fps) if fps and fps > 0 else 30.0

    for codec in ("XVID", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, codec, fps

    raise RuntimeError("Could not open VideoWriter (XVID/MJPG).")


def draw_step2_canny(frame_bgr, roi_rect):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = roi_rect
    roi = gray[y0:y1, x0:x1]
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(roi_blur, 30, 90)

    out = frame_bgr.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
    ys, xs = np.where(edges > 0)
    out[y0 + ys, x0 + xs] = (0, 255, 0)
    return out


def compute_dark_region_mask(roi_gray, offset=18, smooth_hist=True):
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256]).flatten()
    if smooth_hist:
        hist = cv2.GaussianBlur(hist.reshape(-1, 1), (1, 9), 0).flatten()
    search_end = 80
    peak_idx = int(np.argmax(hist[:search_end]))
    thr = min(255, peak_idx + offset)
    dark_mask = (roi_gray <= thr).astype(np.uint8) * 255
    return dark_mask, peak_idx, thr


def compute_specular_mask(roi_gray, thr=220, dilate=3):
    spec = (roi_gray >= thr).astype(np.uint8) * 255
    if dilate and dilate > 0:
        k = 2 * dilate + 1
        spec = cv2.dilate(spec, np.ones((k, k), np.uint8), iterations=1)
    return spec


def _angle_deg(v1, v2):
    n1 = np.linalg.norm(v1) + 1e-9
    n2 = np.linalg.norm(v2) + 1e-9
    v1 = v1 / n1
    v2 = v2 / n2
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def split_contour_by_curvature(points, min_len=40, max_jump=6.0, max_corner_deg=75.0):
    if points is None or len(points) < 3:
        return []

    pts = points.astype(np.float32)
    segments = []
    seg = [pts[0]]

    for i in range(1, len(pts) - 1):
        p_prev = pts[i - 1]
        p = pts[i]
        p_next = pts[i + 1]

        if np.linalg.norm(p - p_prev) > max_jump:
            if len(seg) >= min_len:
                segments.append(np.array(seg, dtype=np.float32))
            seg = [p]
            continue

        v1 = p - p_prev
        v2 = p_next - p
        ang = _angle_deg(v1, v2)
        if ang > max_corner_deg:
            seg.append(p)
            if len(seg) >= min_len:
                segments.append(np.array(seg, dtype=np.float32))
            seg = [p]
            continue

        seg.append(p)

    seg.append(pts[-1])
    if len(seg) >= min_len:
        segments.append(np.array(seg, dtype=np.float32))

    return segments


def draw_polyline(img, pts, color, thickness=1):
    pts_i = pts.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [pts_i], isClosed=False, color=color, thickness=thickness)


def ellipse_area(ell):
    (_, _), (A, B), _ = ell
    return float(np.pi * (A / 2.0) * (B / 2.0))


def ellipse_aspect_ratio(ell):
    (_, _), (A, B), _ = ell
    if A <= 0 or B <= 0:
        return 1e9
    return float(max(A / B, B / A))


def ellipse_support_score(ell, edge_points_xy, dist_px=2.0):
    if ell is None or edge_points_xy is None or len(edge_points_xy) == 0:
        return 0.0, 0.0

    (cx, cy), (A, B), ang = ell
    a = (A / 2.0) + 1e-9
    b = (B / 2.0) + 1e-9
    theta = np.deg2rad(ang)

    pts = edge_points_xy.astype(np.float32)
    x = pts[:, 0] - cx
    y = pts[:, 1] - cy

    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    xr = x * cos_t - y * sin_t
    yr = x * sin_t + y * cos_t

    val = (xr / a) ** 2 + (yr / b) ** 2

    tol = dist_px / min(a, b)
    supported = np.abs(val - 1.0) <= tol

    support_frac = float(np.mean(supported))
    return support_frac, support_frac


# =========================
# Configuration / State
# =========================

@dataclass
class DetectorConfig:
    video: str
    step: int = 0
    roi: str = ""
    preview: bool = False
    out: str = ""

    spec_thr: int = 220
    spec_dilate: int = 3

    split_min_len: int = 18
    split_max_jump: float = 6.0
    split_corner_deg: float = 75.0

    min_ellipse_area: float = 250.0
    max_aspect: float = 3.5
    max_center_dist: float = 0.9

    draw_all_ellipses: bool = False
    topN: int = 10

    support_dist_px: float = 2.0
    support_min_frac: float = 0.20

    # Timestamps source
    states: str = ""

    # HeatmapNet-based ROI
    heatmap_ckpt: str = ""        # path to .pth; empty = disabled (falls back to full-frame ROI)
    heatmap_roi_size: int = 140   # side length of the square ROI in pixels
    heatmap_device: str = field(default_factory=lambda: "cuda" if _cuda_available() else "cpu")
    heatmap_input_size: int = 256
    heatmap_kp_index: int = 1     # 1 = pupil channel
    heatmap_sigma: float = 1.2



@dataclass
class FrameContext:
    frame_idx: int
    roi_rect: Tuple[int, int, int, int]
    # filled by _step8 each frame; NaN = no detection
    pupil_cx: float = float("nan")
    pupil_cy: float = float("nan")
    pupil_diameter: float = float("nan")
    pupil_confidence: float = float("nan")


@dataclass
class PreprocessResult:
    masked_frame: np.ndarray


# =========================
# Detector class
# =========================

class PupilDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.steps: Dict[int, Callable[[np.ndarray, PreprocessResult, FrameContext], np.ndarray]] = self._build_steps()

        self._heatmap_model = None
        if cfg.heatmap_ckpt:
            from utils.heatmap_infer import load_heatmapnet
            self._heatmap_model = load_heatmapnet(cfg.heatmap_ckpt, cfg.heatmap_device)
            print(f"[PupilDetector] HeatmapNet loaded from {cfg.heatmap_ckpt} "
                  f"(device={cfg.heatmap_device}, roi={cfg.heatmap_roi_size}px)")

    def _heatmap_roi(self, gray: np.ndarray) -> Tuple[int, int, int, int]:
        """Run HeatmapNet on the full eye frame and return an N×N ROI centered on the prediction."""
        from utils.heatmap_infer import infer_pupil_from_heatmapnet
        pred = infer_pupil_from_heatmapnet(
            self._heatmap_model, gray,
            self.cfg.heatmap_device,
            self.cfg.heatmap_input_size,
            self.cfg.heatmap_kp_index,
            self.cfg.heatmap_sigma,
        )
        h, w = gray.shape[:2]
        cx = int(round(pred.x))
        cy = int(round(pred.y))
        hs = self.cfg.heatmap_roi_size // 2
        x0 = max(0, cx - hs)
        y0 = max(0, cy - hs)
        x1 = min(w, cx + hs)
        y1 = min(h, cy + hs)
        return (x0, y0, x1, y1)

    def preprocess(self, frame: np.ndarray, ctx: FrameContext) -> PreprocessResult:
        if self._heatmap_model is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ctx.roi_rect = self._heatmap_roi(gray)

        return PreprocessResult(masked_frame=frame)

    def render(self, frame: np.ndarray, prep: PreprocessResult, ctx: FrameContext) -> np.ndarray:
        step_fn = self.steps.get(self.cfg.step, self.steps[-1])
        return step_fn(frame, prep, ctx)

    def _build_steps(self):
        return {
            1: self._step1_roi_guess,
            2: self._step2_canny,
            3: self._step3_dark_mask_overlay,
            4: self._step4_edges_in_dark,
            5: self._step5_edges_dark_not_spec,
            6: self._step6_segments,
            7: self._step7_ellipse_candidates,
            8: self._step8_best_ellipse_by_support,
            -1: self._step_not_implemented,
        }

    # ----- Steps -----

    def _step1_roi_guess(self, frame, prep, ctx):
        gray_m = cv2.cvtColor(prep.masked_frame, cv2.COLOR_BGR2GRAY)
        guess = initial_guess_darkest_patch(gray_m, ctx.roi_rect)
        return draw_step1_debug(prep.masked_frame, ctx.roi_rect, guess)

    def _step2_canny(self, frame, prep, ctx):
        return draw_step2_canny(prep.masked_frame, ctx.roi_rect)

    def _step3_dark_mask_overlay(self, frame, prep, ctx):
        gray_m = cv2.cvtColor(prep.masked_frame, cv2.COLOR_BGR2GRAY)
        x0, y0, x1, y1 = ctx.roi_rect
        roi = gray_m[y0:y1, x0:x1]

        dark_mask, peak, thr = compute_dark_region_mask(roi, offset=18)

        out = prep.masked_frame.copy()
        mask_col = np.zeros_like(out)
        mask_col[y0:y1, x0:x1, 0] = dark_mask
        out = cv2.addWeighted(out, 1.0, mask_col, 0.6, 0)

        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
        cv2.putText(out, f"dark peak={peak}, thr={thr}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _step4_edges_in_dark(self, frame, prep, ctx):
        gray_m = cv2.cvtColor(prep.masked_frame, cv2.COLOR_BGR2GRAY)
        x0, y0, x1, y1 = ctx.roi_rect
        roi = gray_m[y0:y1, x0:x1]

        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(roi_blur, 30, 90)

        dark_mask, peak, thr = compute_dark_region_mask(roi, offset=14)
        edges_dark = cv2.bitwise_and(edges, dark_mask)

        out = prep.masked_frame.copy()
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
        ys, xs = np.where(edges_dark > 0)
        out[y0 + ys, x0 + xs] = (0, 255, 0)

        cv2.putText(out, f"edges ∩ dark (peak={peak}, thr={thr})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _step5_edges_dark_not_spec(self, frame, prep, ctx):
        gray_m = cv2.cvtColor(prep.masked_frame, cv2.COLOR_BGR2GRAY)
        x0, y0, x1, y1 = ctx.roi_rect
        roi = gray_m[y0:y1, x0:x1]

        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(roi_blur, 30, 90)

        dark_mask, peak, thr = compute_dark_region_mask(roi, offset=18)
        spec_mask = compute_specular_mask(roi, thr=self.cfg.spec_thr, dilate=self.cfg.spec_dilate)

        edges_dark = cv2.bitwise_and(edges, dark_mask)
        edges_final = cv2.bitwise_and(edges_dark, cv2.bitwise_not(spec_mask))

        out = prep.masked_frame.copy()
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)

        ys, xs = np.where(edges_final > 0)
        out[y0 + ys, x0 + xs] = (0, 255, 0)

        red = np.zeros_like(out)
        red[y0:y1, x0:x1, 2] = spec_mask
        out = cv2.addWeighted(out, 1.0, red, 0.35, 0)

        cv2.putText(out, f"edges ∩ dark ∩ !spec (peak={peak}, thr={thr}, spec={self.cfg.spec_thr})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _step6_segments(self, frame, prep, ctx):
        gray_m = cv2.cvtColor(prep.masked_frame, cv2.COLOR_BGR2GRAY)
        x0, y0, x1, y1 = ctx.roi_rect
        roi = gray_m[y0:y1, x0:x1]

        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(roi_blur, 30, 90)

        dark_mask, peak, thr = compute_dark_region_mask(roi, offset=18)
        spec_mask = compute_specular_mask(roi, thr=self.cfg.spec_thr, dilate=self.cfg.spec_dilate)

        edges_dark = cv2.bitwise_and(edges, dark_mask)
        edges_final = cv2.bitwise_and(edges_dark, cv2.bitwise_not(spec_mask))

        contours, _ = cv2.findContours(edges_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        out = prep.masked_frame.copy()
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)

        palette = [
            (0, 255, 0), (0, 200, 255), (255, 200, 0),
            (255, 0, 255), (255, 255, 0), (0, 255, 255),
            (200, 200, 255),
        ]

        seg_count = 0
        for cnt in contours:
            if cnt is None or len(cnt) < 5:
                continue
            pts = cnt.reshape(-1, 2).astype(np.float32)
            segments = split_contour_by_curvature(
                pts,
                min_len=self.cfg.split_min_len,
                max_jump=self.cfg.split_max_jump,
                max_corner_deg=self.cfg.split_corner_deg
            )
            for seg in segments:
                c = palette[seg_count % len(palette)]
                seg_shift = seg.copy()
                seg_shift[:, 0] += x0
                seg_shift[:, 1] += y0
                draw_polyline(out, seg_shift, c, thickness=2)
                seg_count += 1

        cv2.putText(out, f"segments={seg_count} (peak={peak} thr={thr} spec={self.cfg.spec_thr})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _step7_ellipse_candidates(self, frame, prep, ctx):
        gray_m = cv2.cvtColor(prep.masked_frame, cv2.COLOR_BGR2GRAY)
        x0, y0, x1, y1 = ctx.roi_rect
        roi = gray_m[y0:y1, x0:x1]

        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(roi_blur, 30, 90)

        dark_mask, _, _ = compute_dark_region_mask(roi, offset=18)
        spec_mask = compute_specular_mask(roi, thr=self.cfg.spec_thr, dilate=self.cfg.spec_dilate)
        edges_dark = cv2.bitwise_and(edges, dark_mask)
        edges_final = cv2.bitwise_and(edges_dark, cv2.bitwise_not(spec_mask))

        contours, _ = cv2.findContours(edges_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        candidates = []
        roi_h, roi_w = roi.shape
        roi_cx, roi_cy = roi_w / 2.0, roi_h / 2.0
        roi_rad = 0.5 * min(roi_w, roi_h)

        for cnt in contours:
            if cnt is None or len(cnt) < 5:
                continue
            pts = cnt.reshape(-1, 2).astype(np.float32)
            segments = split_contour_by_curvature(
                pts,
                min_len=self.cfg.split_min_len,
                max_jump=self.cfg.split_max_jump,
                max_corner_deg=self.cfg.split_corner_deg
            )
            for seg in segments:
                if len(seg) < 5:
                    continue
                seg_cv = seg.reshape(-1, 1, 2).astype(np.float32)
                try:
                    ell = cv2.fitEllipse(seg_cv)
                except cv2.error:
                    continue

                area = ellipse_area(ell)
                if area < self.cfg.min_ellipse_area:
                    continue

                asp = ellipse_aspect_ratio(ell)
                if asp > self.cfg.max_aspect:
                    continue

                (ecx, ecy), (A, B), ang = ell
                dist = np.sqrt((ecx - roi_cx) ** 2 + (ecy - roi_cy) ** 2) / (roi_rad + 1e-9)
                if dist > self.cfg.max_center_dist:
                    continue

                candidates.append((area, dist, asp, ell))

        candidates.sort(key=lambda t: t[0], reverse=True)

        out = prep.masked_frame.copy()
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)

        ys, xs = np.where(edges_final > 0)
        out[y0 + ys, x0 + xs] = (0, 255, 0)

        to_draw = candidates if self.cfg.draw_all_ellipses else candidates[:self.cfg.topN]
        for (area, dist, asp, ell) in to_draw:
            (ecx, ecy), (A, B), ang = ell
            ell_full = ((ecx + x0, ecy + y0), (A, B), ang)
            cv2.ellipse(out, ell_full, (255, 0, 0), 2)

        cv2.putText(out, f"ellipses={len(candidates)} drawn={len(to_draw)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _step8_best_ellipse_by_support(self, frame, prep, ctx):
        ctx.pupil_cx = ctx.pupil_cy = ctx.pupil_diameter = ctx.pupil_confidence = float("nan")

        gray_m = cv2.cvtColor(prep.masked_frame, cv2.COLOR_BGR2GRAY)
        x0, y0, x1, y1 = ctx.roi_rect
        roi = gray_m[y0:y1, x0:x1]

        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(roi_blur, 30, 90)

        dark_mask, peak, thr = compute_dark_region_mask(roi, offset=18)
        spec_mask = compute_specular_mask(roi, thr=self.cfg.spec_thr, dilate=self.cfg.spec_dilate)

        edges_dark = cv2.bitwise_and(edges, dark_mask)
        edges_final = cv2.bitwise_and(edges_dark, cv2.bitwise_not(spec_mask))

        ys, xs = np.where(edges_final > 0)
        edge_pts = np.stack([xs, ys], axis=1).astype(np.float32) if len(xs) else np.zeros((0, 2), np.float32)

        contours, _ = cv2.findContours(edges_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        roi_h, roi_w = roi.shape
        roi_cx, roi_cy = roi_w / 2.0, roi_h / 2.0
        roi_rad = 0.5 * min(roi_w, roi_h)

        candidates = []
        for cnt in contours:
            if cnt is None or len(cnt) < 5:
                continue
            pts = cnt.reshape(-1, 2).astype(np.float32)
            segments = split_contour_by_curvature(
                pts,
                min_len=self.cfg.split_min_len,
                max_jump=self.cfg.split_max_jump,
                max_corner_deg=self.cfg.split_corner_deg
            )

            for seg in segments:
                if len(seg) < 5:
                    continue
                seg_cv = seg.reshape(-1, 1, 2).astype(np.float32)
                try:
                    ell = cv2.fitEllipse(seg_cv)
                except cv2.error:
                    continue

                area = ellipse_area(ell)
                if area < self.cfg.min_ellipse_area:
                    continue

                asp = ellipse_aspect_ratio(ell)
                if asp > self.cfg.max_aspect:
                    continue

                (ecx, ecy), (A, B), ang = ell
                dist = np.sqrt((ecx - roi_cx) ** 2 + (ecy - roi_cy) ** 2) / (roi_rad + 1e-9)
                if dist > self.cfg.max_center_dist:
                    continue

                support_frac, support_score = ellipse_support_score(ell, edge_pts, dist_px=self.cfg.support_dist_px)
                final_score = support_score - 0.25 * dist - 0.05 * max(0.0, asp - 1.0)
                candidates.append((final_score, support_frac, dist, asp, ell))

        out = prep.masked_frame.copy()
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
        out[y0 + ys, x0 + xs] = (0, 255, 0)

        if not candidates:
            cv2.putText(out, "no ellipse candidates",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
            return out

        candidates.sort(key=lambda t: t[0], reverse=True)
        best_score, best_support, best_dist, best_asp, best_ell = candidates[0]

        if best_support < self.cfg.support_min_frac:
            cv2.putText(out, f"best support too low ({best_support:.2f})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
            return out

        (cx, cy), (A, B), ang = best_ell
        cx_full = cx + x0
        cy_full = cy + y0

        ctx.pupil_cx = cx_full
        ctx.pupil_cy = cy_full
        ctx.pupil_diameter = (A + B) / 2.0
        ctx.pupil_confidence = best_support

        cv2.ellipse(out, ((cx_full, cy_full), (A, B), ang), (0, 0, 255), 2)
        cv2.circle(out, (int(cx_full), int(cy_full)), 3, (0, 0, 255), -1)

        cv2.putText(out, f"score={best_score:.3f} supp={best_support:.2f} dist={best_dist:.2f} asp={best_asp:.2f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _step_not_implemented(self, frame, prep, ctx):
        out = prep.masked_frame.copy()
        cv2.putText(out, f"Step {self.cfg.step} not implemented",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2, cv2.LINE_AA)
        return out


# =========================
# Video runner
# =========================

def _fmt(v: float) -> str:
    return "" if v != v else f"{v:.3f}"  # NaN → empty string


def _load_timestamps(states_path: str) -> list:
    if not states_path:
        return []
    with open(states_path, newline="") as f:
        return [int(row["timestamp [ns]"]) for row in csv.DictReader(f)]


def run_video(cfg: DetectorConfig):
    if not os.path.exists(cfg.video):
        raise FileNotFoundError(f"Input video not found: {cfg.video}")

    cap = cv2.VideoCapture(cfg.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {cfg.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Empty video or could not read first frame.")

    h, w = first.shape[:2]
    mid = w // 2

    timestamps = _load_timestamps(cfg.states)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if timestamps and len(timestamps) != nframes:
        raise RuntimeError(
            f"Timestamp count ({len(timestamps)}) != video frame count ({nframes}). "
            f"Check --states matches the recording."
        )

    output_path = cfg.out.strip() if cfg.out.strip() else f"debug_step_{cfg.step}.avi"
    # Output width = left_half + right_half (handles odd w by trimming right)
    out_w = mid + mid
    writer, used_codec, used_fps = make_writer(output_path, fps, (out_w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cfg.preview:
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    detector = PupilDetector(cfg)

    roi_l = parse_roi(cfg.roi, mid, h)
    roi_r = parse_roi(cfg.roi, mid, h)
    ctx_l = FrameContext(frame_idx=0, roi_rect=roi_l)
    ctx_r = FrameContext(frame_idx=0, roi_rect=roi_r)

    csv_path = os.path.splitext(output_path)[0] + "_pupils.csv"

    frame_idx = 0
    with open(csv_path, "w", newline="") as csv_file:
        cw = csv.writer(csv_file)
        cw.writerow([
            "timestamp [ns]",
            "xL", "yL", "diameter_L", "confidence_L",
            "xR", "yR", "diameter_R", "confidence_R",
        ])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            left  = frame[:, :mid].copy()
            right = frame[:, mid:mid*2].copy()

            ctx_l.frame_idx = frame_idx
            ctx_r.frame_idx = frame_idx

            prep_l = detector.preprocess(left,  ctx_l)
            prep_r = detector.preprocess(right, ctx_r)
            out_l  = detector.render(left,  prep_l, ctx_l)
            out_r  = detector.render(right, prep_r, ctx_r)

            out = np.concatenate([out_l, out_r], axis=1)
            writer.write(out)

            ts = timestamps[frame_idx] if timestamps else frame_idx
            cw.writerow([
                ts,
                _fmt(ctx_l.pupil_cx), _fmt(ctx_l.pupil_cy),
                _fmt(ctx_l.pupil_diameter), _fmt(ctx_l.pupil_confidence),
                _fmt(ctx_r.pupil_cx), _fmt(ctx_r.pupil_cy),
                _fmt(ctx_r.pupil_diameter), _fmt(ctx_r.pupil_confidence),
            ])

            if cfg.preview:
                cv2.imshow("preview", out)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    cv2.waitKey(0)

            frame_idx += 1

    cap.release()
    writer.release()
    if cfg.preview:
        cv2.destroyAllWindows()

    print(f"Saved video : {output_path}")
    print(f"Saved CSV   : {csv_path}")
    print(f"Codec: {used_codec}, FPS: {used_fps:.2f}, Frames: {frame_idx}")


# =========================
# CLI
# =========================

def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--roi", type=str, default="")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--out", type=str, default="")

    ap.add_argument("--spec_thr", type=int, default=220)
    ap.add_argument("--spec_dilate", type=int, default=3)

    ap.add_argument("--split_min_len", type=int, default=18)
    ap.add_argument("--split_max_jump", type=float, default=6.0)
    ap.add_argument("--split_corner_deg", type=float, default=75.0)

    ap.add_argument("--min_ellipse_area", type=float, default=250.0)
    ap.add_argument("--max_aspect", type=float, default=3.5)
    ap.add_argument("--max_center_dist", type=float, default=0.9)

    ap.add_argument("--draw_all_ellipses", action="store_true")
    ap.add_argument("--topN", type=int, default=10)

    ap.add_argument("--support_dist_px", type=float, default=2.0)
    ap.add_argument("--support_min_frac", type=float, default=0.20)

    ap.add_argument("--states", type=str, default="",
                    help="Path to 3d_eye_states.csv for timestamps (optional)")

    ap.add_argument("--heatmap_ckpt", type=str, default="",
                    help="HeatmapNet checkpoint (.pth). If set, uses heatmap to place ROI each frame.")
    ap.add_argument("--heatmap_roi_size", type=int, default=140,
                    help="Side length (px) of the square ROI centered on heatmap prediction")
    ap.add_argument("--heatmap_device", type=str, default="cuda" if _cuda_available() else "cpu")

    return ap


def cfg_from_args(args) -> DetectorConfig:
    return DetectorConfig(
        video=args.video,
        step=args.step,
        roi=args.roi,
        preview=args.preview,
        out=args.out,

        spec_thr=args.spec_thr,
        spec_dilate=args.spec_dilate,

        split_min_len=args.split_min_len,
        split_max_jump=args.split_max_jump,
        split_corner_deg=args.split_corner_deg,

        min_ellipse_area=args.min_ellipse_area,
        max_aspect=args.max_aspect,
        max_center_dist=args.max_center_dist,

        draw_all_ellipses=args.draw_all_ellipses,
        topN=args.topN,

        support_dist_px=args.support_dist_px,
        support_min_frac=args.support_min_frac,

        states=args.states,

        heatmap_ckpt=args.heatmap_ckpt,
        heatmap_roi_size=args.heatmap_roi_size,
        heatmap_device=args.heatmap_device,
    )


def main():
    args = build_arg_parser().parse_args()
    cfg = cfg_from_args(args)
    run_video(cfg)


if __name__ == "__main__":
    main()
