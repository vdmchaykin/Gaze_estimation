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


def draw_step2_canny(frame_bgr, roi_rect, canny_low=30, canny_high=90):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = roi_rect
    roi = gray[y0:y1, x0:x1]
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(roi_blur, canny_low, canny_high)

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


def bridge_edge_gaps(edges, max_gap=8, max_angle_deg=45.0):
    """Connect nearby arc endpoints separated by small gaps.

    For each pair of contour endpoints within max_gap pixels, draws a bridging
    line if both tangents point roughly toward the gap (direction check).
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    endpoints = []  # (point, outward_tangent, contour_idx)
    for i, cnt in enumerate(contours):
        pts = cnt.reshape(-1, 2).astype(float)
        if len(pts) < 3:
            continue
        k = min(3, len(pts) - 1)
        endpoints.append((pts[0],    pts[0] - pts[k],    i))
        endpoints.append((pts[-1],   pts[-1] - pts[-(k + 1)], i))

    result = edges.copy()
    n = len(endpoints)
    for i in range(n):
        pi, ti, ci = endpoints[i]
        for j in range(i + 1, n):
            pj, tj, cj = endpoints[j]
            if ci == cj:
                continue
            dist = float(np.linalg.norm(pi - pj))
            if dist < 1.0 or dist > max_gap:
                continue
            bridge = (pj - pi) / dist
            if _angle_deg(ti, bridge) > max_angle_deg or _angle_deg(tj, -bridge) > max_angle_deg:
                continue
            cv2.line(result, tuple(pi.astype(int)), tuple(pj.astype(int)), 255, 1)

    return result


def compute_eyelash_mask(roi_gray, dark_thr=70, kernel_len=13, n_angles=6, dilate=2):
    """Detect thin elongated dark structures (eyelashes) via morphological line opening."""
    dark = (roi_gray < dark_thr).astype(np.uint8) * 255
    eyelash = np.zeros_like(dark)
    half = kernel_len // 2
    angle_step = 180 // n_angles
    for a_deg in range(0, 180, angle_step):
        rad = np.deg2rad(a_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        kernel = np.zeros((kernel_len, kernel_len), np.uint8)
        for i in range(-half, half + 1):
            x = int(round(half + i * cos_a))
            y = int(round(half + i * sin_a))
            if 0 <= x < kernel_len and 0 <= y < kernel_len:
                kernel[y, x] = 1
        if int(kernel.sum()) < 3:
            continue
        opened = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
        eyelash = cv2.bitwise_or(eyelash, opened)
    if dilate > 0:
        dk = 2 * dilate + 1
        eyelash = cv2.dilate(eyelash, np.ones((dk, dk), np.uint8))
    return eyelash


def fit_ellipse_ransac(points, n_iter=100, inlier_dist=2.5):
    """Fit ellipse using RANSAC — robust to eyelash edge outliers."""
    pts = points.reshape(-1, 2).astype(np.float32)
    n = len(pts)
    if n < 5:
        return None
    best_ell, best_inliers = None, 0
    for _ in range(n_iter):
        idx = np.random.choice(n, 5, replace=False)
        sample = pts[idx].reshape(-1, 1, 2)
        try:
            ell = cv2.fitEllipse(sample)
        except cv2.error:
            continue
        frac, _ = ellipse_support_score(ell, pts, dist_px=inlier_dist)
        n_inliers = int(frac * n)
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_ell = ell
    return best_ell


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

    canny_low: int = 30
    canny_high: int = 90

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
    heatmap_prior_weight: float = 0.3  # penalty weight for distance from heatmap prediction

    # Edge gap closing — endpoint bridging
    gap_bridge_max_dist: int = 0        # max pixel gap to bridge; 0 = disabled
    gap_bridge_angle_deg: float = 45.0  # max angle between endpoint tangent and bridge direction

    # RANSAC ellipse fitting
    ransac_n_iter: int = 100      # 0 = use direct fitEllipse (original behaviour)

    # Segment linearity filter (eyelash rejection)
    # chord/arc_length ratio: 1.0 = perfectly straight, ~0.5 = strong curve
    # Segments above this threshold are rejected as eyelashes
    max_seg_straightness: float = 0.92

    # Preview eye selection: "both", "left", "right"
    preview_eye: str = "both"
    # Scale factor applied to the preview window (bilinear upscale, does not affect saved video)
    preview_scale: float = 2.0



@dataclass
class FrameContext:
    frame_idx: int
    roi_rect: Tuple[int, int, int, int]
    # filled by _step8 each frame; NaN = no detection
    pupil_cx: float = float("nan")
    pupil_cy: float = float("nan")
    pupil_diameter: float = float("nan")
    pupil_confidence: float = float("nan")
    # heatmap prediction in full-frame coords; NaN if heatmap disabled
    heatmap_cx: float = float("nan")
    heatmap_cy: float = float("nan")


@dataclass
class PipelineState:
    frame_bgr: np.ndarray
    gray: Optional[np.ndarray] = None
    roi_rect: Optional[Tuple[int, int, int, int]] = None
    roi_gray: Optional[np.ndarray] = None
    edges: Optional[np.ndarray] = None
    dark_mask: Optional[np.ndarray] = None
    dark_peak: int = 0
    dark_thr: int = 0
    edges_dark: Optional[np.ndarray] = None
    spec_mask: Optional[np.ndarray] = None
    edges_filtered: Optional[np.ndarray] = None
    bridge_mask: Optional[np.ndarray] = None
    segments: List[np.ndarray] = field(default_factory=list)
    candidates: List = field(default_factory=list)  # (area, dist, asp, ell, seg)
    best_ell: Any = None
    best_support: float = 0.0
    best_score: float = 0.0
    best_dist: float = 0.0
    best_asp: float = 0.0


# =========================
# Detector class
# =========================

class PupilDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self._heatmap_model = None
        if cfg.heatmap_ckpt:
            from utils.heatmap_infer import load_heatmapnet
            self._heatmap_model = load_heatmapnet(cfg.heatmap_ckpt, cfg.heatmap_device)
            print(f"[PupilDetector] HeatmapNet loaded from {cfg.heatmap_ckpt} "
                  f"(device={cfg.heatmap_device}, roi={cfg.heatmap_roi_size}px)")

    # ------------------------------------------------------------------ #
    # Compute steps — each builds on the previous PipelineState          #
    # ------------------------------------------------------------------ #

    def _compute1(self, state: PipelineState, ctx: FrameContext) -> PipelineState:
        state.gray = cv2.cvtColor(state.frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._heatmap_model is not None:
            state.roi_rect = self._heatmap_roi(state.gray, ctx)
        else:
            state.roi_rect = ctx.roi_rect
        ctx.roi_rect = state.roi_rect
        x0, y0, x1, y1 = state.roi_rect
        state.roi_gray = state.gray[y0:y1, x0:x1]
        return state

    def _compute2(self, state: PipelineState, ctx: FrameContext) -> PipelineState:
        roi_blur = cv2.GaussianBlur(state.roi_gray, (5, 5), 0)
        state.edges = cv2.Canny(roi_blur, self.cfg.canny_low, self.cfg.canny_high)
        return state

    def _compute3(self, state: PipelineState, ctx: FrameContext) -> PipelineState:
        state.dark_mask, state.dark_peak, state.dark_thr = compute_dark_region_mask(
            state.roi_gray, offset=10)
        return state

    def _compute4(self, state: PipelineState, ctx: FrameContext) -> PipelineState:
        dark_dilated = cv2.dilate(state.dark_mask, np.ones((3, 3), np.uint8), iterations=1)
        state.edges_dark = cv2.bitwise_and(state.edges, dark_dilated)
        return state

    def _compute5(self, state: PipelineState, ctx: FrameContext) -> PipelineState:
        state.spec_mask = compute_specular_mask(
            state.roi_gray, thr=self.cfg.spec_thr, dilate=self.cfg.spec_dilate)
        edges = cv2.bitwise_and(state.edges_dark, cv2.bitwise_not(state.spec_mask))
        if self.cfg.gap_bridge_max_dist > 0:
            bridged = bridge_edge_gaps(
                edges,
                max_gap=self.cfg.gap_bridge_max_dist,
                max_angle_deg=self.cfg.gap_bridge_angle_deg,
            )
            state.bridge_mask = cv2.bitwise_and(bridged, cv2.bitwise_not(edges))
            edges = bridged
        state.edges_filtered = edges
        return state

    def _compute6(self, state: PipelineState, ctx: FrameContext) -> PipelineState:
        contours, _ = cv2.findContours(state.edges_filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        segments = []
        for cnt in contours:
            if cnt is None or len(cnt) < 5:
                continue
            pts = cnt.reshape(-1, 2).astype(np.float32)
            for seg in split_contour_by_curvature(
                pts,
                min_len=self.cfg.split_min_len,
                max_jump=self.cfg.split_max_jump,
                max_corner_deg=self.cfg.split_corner_deg,
            ):
                chord = float(np.linalg.norm(seg[-1] - seg[0]))
                arc = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))) + 1e-9
                if chord / arc <= self.cfg.max_seg_straightness:
                    segments.append(seg)
        state.segments = segments
        return state

    def _compute7(self, state: PipelineState, ctx: FrameContext) -> PipelineState:
        roi_h, roi_w = state.roi_gray.shape
        roi_cx, roi_cy = roi_w / 2.0, roi_h / 2.0
        roi_rad = 0.5 * min(roi_w, roi_h)
        candidates = []
        for seg in state.segments:
            if len(seg) < 5:
                continue
            seg_cv = seg.reshape(-1, 1, 2).astype(np.float32)
            try:
                ell = cv2.fitEllipse(seg_cv)
            except cv2.error:
                continue
            if ellipse_area(ell) < self.cfg.min_ellipse_area:
                continue
            asp = ellipse_aspect_ratio(ell)
            if asp > self.cfg.max_aspect:
                continue
            (ecx, ecy), _, _ = ell
            dist = np.sqrt((ecx - roi_cx) ** 2 + (ecy - roi_cy) ** 2) / (roi_rad + 1e-9)
            if dist > self.cfg.max_center_dist:
                continue
            candidates.append((ellipse_area(ell), dist, asp, ell, seg))
        candidates.sort(key=lambda t: t[0], reverse=True)
        state.candidates = candidates
        return state

    def _compute8(self, state: PipelineState, ctx: FrameContext) -> PipelineState:
        ctx.pupil_cx = ctx.pupil_cy = ctx.pupil_diameter = ctx.pupil_confidence = float("nan")

        ys, xs = np.where(state.edges_filtered > 0)
        edge_pts = (np.stack([xs, ys], axis=1).astype(np.float32)
                    if len(xs) else np.zeros((0, 2), np.float32))

        roi_h, roi_w = state.roi_gray.shape
        roi_cx, roi_cy = roi_w / 2.0, roi_h / 2.0
        roi_rad = 0.5 * min(roi_w, roi_h)

        x0, y0 = state.roi_rect[0], state.roi_rect[1]
        hm_roi_cx = ctx.heatmap_cx - x0 if not np.isnan(ctx.heatmap_cx) else None
        hm_roi_cy = ctx.heatmap_cy - y0 if not np.isnan(ctx.heatmap_cy) else None

        scored = []
        for (area, dist, asp, ell_init, seg) in state.candidates:
            if self.cfg.ransac_n_iter > 0:
                ell = fit_ellipse_ransac(seg, self.cfg.ransac_n_iter, self.cfg.support_dist_px)
                if ell is None:
                    continue
                if ellipse_area(ell) < self.cfg.min_ellipse_area:
                    continue
                asp = ellipse_aspect_ratio(ell)
                if asp > self.cfg.max_aspect:
                    continue
                (ecx, ecy), _, _ = ell
                dist = np.sqrt((ecx - roi_cx) ** 2 + (ecy - roi_cy) ** 2) / (roi_rad + 1e-9)
                if dist > self.cfg.max_center_dist:
                    continue
            else:
                ell = ell_init
                (ecx, ecy), _, _ = ell

            support_frac, support_score = ellipse_support_score(
                ell, edge_pts, dist_px=self.cfg.support_dist_px)

            heatmap_penalty = 0.0
            if hm_roi_cx is not None:
                hm_dist = np.sqrt((ecx - hm_roi_cx) ** 2 + (ecy - hm_roi_cy) ** 2) / (roi_rad + 1e-9)
                heatmap_penalty = self.cfg.heatmap_prior_weight * hm_dist

            final_score = support_score - 0.25 * dist - 0.05 * max(0.0, asp - 1.0) - heatmap_penalty
            scored.append((final_score, support_frac, dist, asp, ell))

        if not scored:
            return state

        scored.sort(key=lambda t: t[0], reverse=True)
        best_score, best_support, best_dist, best_asp, best_ell = scored[0]

        if best_support < self.cfg.support_min_frac:
            return state

        state.best_ell = best_ell
        state.best_support = best_support
        state.best_score = best_score
        state.best_dist = best_dist
        state.best_asp = best_asp

        (cx, cy), (A, B), _ = best_ell
        ctx.pupil_cx = cx + x0
        ctx.pupil_cy = cy + y0
        ctx.pupil_diameter = (A + B) / 2.0
        ctx.pupil_confidence = best_support
        return state

    # ------------------------------------------------------------------ #
    # Visualization — one method per step                                 #
    # ------------------------------------------------------------------ #

    def _viz1(self, state: PipelineState) -> np.ndarray:
        out = state.frame_bgr.copy()
        x0, y0, x1, y1 = state.roi_rect
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)

        return out

    def _viz2(self, state: PipelineState) -> np.ndarray:
        out = state.frame_bgr.copy()
        x0, y0, x1, y1 = state.roi_rect
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
        ys, xs = np.where(state.edges > 0)
        out[y0 + ys, x0 + xs] = (0, 255, 0)

        return out

    def _viz3(self, state: PipelineState) -> np.ndarray:
        out = state.frame_bgr.copy()
        x0, y0, x1, y1 = state.roi_rect
        overlay = np.zeros_like(out)
        overlay[y0:y1, x0:x1, 0] = state.dark_mask
        out = cv2.addWeighted(out, 1.0, overlay, 0.6, 0)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)

        return out

    def _viz4(self, state: PipelineState) -> np.ndarray:
        out = state.frame_bgr.copy()
        x0, y0, x1, y1 = state.roi_rect
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
        ys, xs = np.where(state.edges_dark > 0)
        out[y0 + ys, x0 + xs] = (0, 255, 0)

        return out

    def _viz5(self, state: PipelineState) -> np.ndarray:
        out = state.frame_bgr.copy()
        x0, y0, x1, y1 = state.roi_rect
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
        ys, xs = np.where(state.edges_filtered > 0)
        out[y0 + ys, x0 + xs] = (0, 255, 0)
        if state.bridge_mask is not None:
            bys, bxs = np.where(state.bridge_mask > 0)
            out[y0 + bys, x0 + bxs] = (255, 0, 0)
        red = np.zeros_like(out)
        red[y0:y1, x0:x1, 2] = state.spec_mask
        out = cv2.addWeighted(out, 1.0, red, 0.35, 0)
        return out

    def _viz6(self, state: PipelineState) -> np.ndarray:
        out = state.frame_bgr.copy()
        x0, y0, x1, y1 = state.roi_rect
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
        palette = [(0, 255, 0), (0, 200, 255), (255, 200, 0),
                   (255, 0, 255), (255, 255, 0), (0, 255, 255), (200, 200, 255)]
        for i, seg in enumerate(state.segments):
            seg_shift = seg.copy()
            seg_shift[:, 0] += x0
            seg_shift[:, 1] += y0
            draw_polyline(out, seg_shift, palette[i % len(palette)], thickness=2)

        return out

    def _viz7(self, state: PipelineState) -> np.ndarray:
        out = state.frame_bgr.copy()
        x0, y0, x1, y1 = state.roi_rect
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
        ys, xs = np.where(state.edges_filtered > 0)
        out[y0 + ys, x0 + xs] = (0, 255, 0)
        to_draw = state.candidates if self.cfg.draw_all_ellipses else state.candidates[:self.cfg.topN]
        for (area, dist, asp, ell, _seg) in to_draw:
            (ecx, ecy), (A, B), ang = ell
            cv2.ellipse(out, ((ecx + x0, ecy + y0), (A, B), ang), (255, 0, 0), 2)

        return out

    def _viz8(self, state: PipelineState) -> np.ndarray:
        out = state.frame_bgr.copy()
        x0, y0, x1, y1 = state.roi_rect
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
        ys, xs = np.where(state.edges_filtered > 0)
        out[y0 + ys, x0 + xs] = (0, 255, 0)
        if state.best_ell is None:
            label = "no candidates" if not state.candidates else f"support too low ({state.best_support:.2f})"
            cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
            return out
        (cx, cy), (A, B), ang = state.best_ell
        cx_f, cy_f = cx + x0, cy + y0
        cv2.ellipse(out, ((cx_f, cy_f), (A, B), ang), (0, 0, 255), 2)
        cv2.circle(out, (int(cx_f), int(cy_f)), 3, (0, 0, 255), -1)

        return out

    # ------------------------------------------------------------------ #
    # Pipeline runner                                                     #
    # ------------------------------------------------------------------ #

    _PIPELINE = [
        (1, '_compute1', '_viz1'),
        (2, '_compute2', '_viz2'),
        (3, '_compute3', '_viz3'),
        (4, '_compute4', '_viz4'),
        (5, '_compute5', '_viz5'),
        (6, '_compute6', '_viz6'),
        (7, '_compute7', '_viz7'),
        (8, '_compute8', '_viz8'),
    ]

    def process(self, frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
        state = PipelineState(frame_bgr=frame)

        if self.cfg.step == 0:
            vizs = {}
            for step_num, compute_name, viz_name in self._PIPELINE:
                state = getattr(self, compute_name)(state, ctx)
                vizs[step_num] = getattr(self, viz_name)(state)
            return self._make_mosaic(frame, vizs)

        for step_num, compute_name, viz_name in self._PIPELINE:
            state = getattr(self, compute_name)(state, ctx)
            if step_num == self.cfg.step:
                return getattr(self, viz_name)(state)

        return frame

    def _make_mosaic(self, frame: np.ndarray, vizs: dict) -> np.ndarray:
        panels = []
        for i in range(1, 9):
            panel = vizs.get(i, frame.copy())
            cv2.putText(panel, f"S{i}", (3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        row1 = np.concatenate(panels[:4], axis=1)
        row2 = np.concatenate(panels[4:], axis=1)
        return np.concatenate([row1, row2], axis=0)

    def _heatmap_roi(self, gray: np.ndarray, ctx: FrameContext) -> Tuple[int, int, int, int]:
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
        ctx.heatmap_cx = float(cx)
        ctx.heatmap_cy = float(cy)
        hs = self.cfg.heatmap_roi_size // 2
        x0 = max(0, cx - hs)
        y0 = max(0, cy - hs)
        x1 = min(w, cx + hs)
        y1 = min(h, cy + hs)
        return (x0, y0, x1, y1)


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

            out_l = detector.process(left,  ctx_l)
            out_r = detector.process(right, ctx_r)

            if cfg.step == 0:
                # mosaic frames are 4*mid × 2*h — downscale to writer dims before saving
                write_l = cv2.resize(out_l, (mid, h), interpolation=cv2.INTER_AREA)
                write_r = cv2.resize(out_r, (mid, h), interpolation=cv2.INTER_AREA)
                out = np.concatenate([write_l, write_r], axis=1)
            else:
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
                eye = cfg.preview_eye.lower()
                if eye == "left":
                    preview_frame = out_l
                elif eye == "right":
                    preview_frame = out_r
                else:
                    # for step=0: out is already downscaled for the writer — use full-res mosaics
                    preview_frame = np.concatenate([out_l, out_r], axis=1) if cfg.step == 0 else out
                if cfg.preview_scale != 1.0:
                    ph, pw = preview_frame.shape[:2]
                    preview_frame = cv2.resize(
                        preview_frame,
                        (int(pw * cfg.preview_scale), int(ph * cfg.preview_scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                cv2.imshow("preview", preview_frame)
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
    ap.add_argument("--step", type=int, default=0,
                    help="Pipeline step to visualize (1-8). 0 = all steps as 4×2 mosaic.")
    ap.add_argument("--roi", type=str, default="")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--out", type=str, default="")

    ap.add_argument("--canny_low", type=int, default=50,
                    help="Canny lower threshold (hysteresis)")
    ap.add_argument("--canny_high", type=int, default=100,
                    help="Canny upper threshold (hysteresis)")

    ap.add_argument("--spec_thr", type=int, default=220)
    ap.add_argument("--spec_dilate", type=int, default=3)

    ap.add_argument("--split_min_len", type=int, default=20)
    ap.add_argument("--split_max_jump", type=float, default=10.0)
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
    ap.add_argument("--heatmap_roi_size", type=int, default=30,
                    help="Side length (px) of the square ROI centered on heatmap prediction")
    ap.add_argument("--heatmap_device", type=str, default="cuda" if _cuda_available() else "cpu")
    ap.add_argument("--heatmap_kp_index", type=int, default=1,
                    help="Keypoint channel index in heatmap output (0 for single-channel checkpoints)")
    ap.add_argument("--heatmap_prior_weight", type=float, default=0.3,
                    help="Scoring penalty weight for distance from heatmap prediction (0=disabled)")

    ap.add_argument("--gap_bridge_max_dist", type=int, default=0,
                    help="Max pixel gap between arc endpoints to bridge; 0 = disabled")
    ap.add_argument("--gap_bridge_angle_deg", type=float, default=45.0,
                    help="Max tangent-to-bridge angle for endpoint bridging")
    ap.add_argument("--ransac_n_iter", type=int, default=0,
                    help="RANSAC iterations for ellipse fitting; 0=use direct fitEllipse")
    ap.add_argument("--max_seg_straightness", type=float, default=0.92,
                    help="Segment chord/arc ratio above which it is rejected as eyelash (1.0=straight line)")
    ap.add_argument("--preview_eye", type=str, default="both",
                    choices=["both", "left", "right"],
                    help="Which eye to show in the preview window (saved video always has both)")
    ap.add_argument("--preview_scale", type=float, default=2.0,
                    help="Upscale factor for preview window (bilinear); does not affect saved video")

    return ap


def cfg_from_args(args) -> DetectorConfig:
    return DetectorConfig(
        video=args.video,
        step=args.step,
        roi=args.roi,
        preview=args.preview,
        out=args.out,

        canny_low=args.canny_low,
        canny_high=args.canny_high,

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
        heatmap_kp_index=args.heatmap_kp_index,
        heatmap_prior_weight=args.heatmap_prior_weight,

        gap_bridge_max_dist=args.gap_bridge_max_dist,
        gap_bridge_angle_deg=args.gap_bridge_angle_deg,
        ransac_n_iter=args.ransac_n_iter,
        max_seg_straightness=args.max_seg_straightness,
        preview_eye=args.preview_eye,
        preview_scale=args.preview_scale,
    )


def main():
    args = build_arg_parser().parse_args()
    cfg = cfg_from_args(args)
    run_video(cfg)


if __name__ == "__main__":
    main()
