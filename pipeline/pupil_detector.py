"""Combined pupil detector: floodfill (primary) + edge-based (fallback).

Ported from the Colab research pipeline. The HeatmapNet used to place the ROI
is reused from the project modules (``models.heatmap_net`` / ``utils.heatmap_infer``)
instead of being redefined here.

Public API used by the backend:
    build_combined_detector(floodfill_cfg, edge_cfg, heatmap_ckpt, device)
        -> (CombinedPupilDetector, device)
    CombinedPupilDetector.detect(frame, floodfill_ctx, edge_ctx)
        -> (CombinedFrameContext, floodfill_state, edge_state)

Standalone (CLI / notebook) helpers are also kept: run_combined_full_video,
mark_blinks, clean_and_smooth, render_clean_overlay_both_eyes.
"""
import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, List

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


# ================================================================
# HeatmapNet — reuse the project's model + inference
# ================================================================

def load_heatmap_model(checkpoint_path: str, device: str):
    """Load the project HeatmapNet checkpoint (models.heatmap_net via utils.heatmap_infer)."""
    from utils.heatmap_infer import load_heatmapnet
    return load_heatmapnet(checkpoint_path, device)


def _infer_center(model, gray, device, input_size, kp_index, sigma) -> Tuple[float, float]:
    """Pupil center (x, y) in ``gray`` pixel coords via the project HeatmapNet infer."""
    from utils.heatmap_infer import infer_pupil_from_heatmapnet
    pred = infer_pupil_from_heatmapnet(model, gray, device, input_size, kp_index, sigma)
    return float(pred.x), float(pred.y)


# ================================================================
# 1. Floodfill detector (primary method)
# ================================================================

@dataclass
class DetectorConfig:
    heatmap_roi_size: int = 35
    heatmap_input_size: int = 256
    heatmap_kp_index: int = 1
    heatmap_sigma: float = 1.2

    floodfill_lo_diff: int = 25
    floodfill_hi_diff: int = 15
    floodfill_blur_ksize: int = 3
    floodfill_min_area: float = 40.0
    floodfill_max_area_frac: float = 0.6
    floodfill_close_ksize: int = 9
    floodfill_open_ksize: int = 5
    floodfill_seed_search: int = 10      # radius (px) of dark-pixel search around the heatmap point
    floodfill_lash_open_ksize: int = 9   # 0 = off; grayscale-open kernel to erase bright eyelash stripes

    # "ellipse found but not acceptable" criteria
    floodfill_min_fill_frac: float = 0.55
    floodfill_max_aspect: float = 1.8


@dataclass
class FrameContext:
    frame_idx: int
    roi_rect: Tuple[int, int, int, int]
    pupil_cx: float = float("nan")
    pupil_cy: float = float("nan")
    pupil_diameter: float = float("nan")
    pupil_confidence: float = float("nan")
    pupil_A: float = float("nan")
    pupil_B: float = float("nan")
    pupil_angle: float = float("nan")
    heatmap_cx: float = float("nan")
    heatmap_cy: float = float("nan")
    debug_reason: str = ""
    seed_x: float = float("nan")
    seed_y: float = float("nan")


@dataclass
class PipelineState:
    frame_bgr: np.ndarray
    gray: Optional[np.ndarray] = None
    roi_rect: Optional[Tuple[int, int, int, int]] = None
    roi_gray: Optional[np.ndarray] = None
    dark_mask: Optional[np.ndarray] = None
    best_ell: Any = None
    best_support: float = 0.0


def ellipse_aspect_ratio(ell):
    (_, _), (A, B), _ = ell
    a, b = max(A, B), min(A, B)
    return a / b if b > 1e-6 else float("inf")


class PupilDetector:
    def __init__(self, cfg: DetectorConfig, heatmap_model=None, device="cpu"):
        self.cfg = cfg
        self._heatmap_model = heatmap_model
        self._device = device

    def _heatmap_roi(self, gray, ctx):
        cx, cy = _infer_center(
            self._heatmap_model, gray, self._device,
            self.cfg.heatmap_input_size, self.cfg.heatmap_kp_index, self.cfg.heatmap_sigma,
        )
        cx, cy = int(round(cx)), int(round(cy))
        ctx.heatmap_cx, ctx.heatmap_cy = float(cx), float(cy)
        hs = self.cfg.heatmap_roi_size // 2
        h, w = gray.shape[:2]
        return (max(0, cx - hs), max(0, cy - hs), min(w, cx + hs), min(h, cy + hs))

    def _compute1(self, state, ctx):
        state.gray = cv2.cvtColor(state.frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._heatmap_model is not None:
            state.roi_rect = self._heatmap_roi(state.gray, ctx)
        else:
            state.roi_rect = ctx.roi_rect
        ctx.roi_rect = state.roi_rect
        x0, y0, x1, y1 = state.roi_rect
        state.roi_gray = state.gray[y0:y1, x0:x1]
        return state

    def _compute_floodfill(self, state, ctx):
        ctx.pupil_cx = ctx.pupil_cy = ctx.pupil_diameter = ctx.pupil_confidence = float("nan")
        ctx.pupil_A = ctx.pupil_B = ctx.pupil_angle = float("nan")
        ctx.seed_x = ctx.seed_y = float("nan")
        ctx.debug_reason = ""

        roi = state.roi_gray
        h, w = roi.shape
        x0, y0 = state.roi_rect[0], state.roi_rect[1]

        # grayscale opening erases bright thin structures (eyelash stripes under IR light),
        # which cut the dark pupil into isolated pieces and stop the fill from growing;
        # the pupil is larger than the kernel, so it stays intact
        work = roi
        if self.cfg.floodfill_lash_open_ksize > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (self.cfg.floodfill_lash_open_ksize,) * 2)
            work = cv2.morphologyEx(roi, cv2.MORPH_OPEN, k)

        # blur BEFORE choosing the seed: on the blurred image the brightness minimum
        # is "area-like", so argmin falls into the pupil rather than onto a noisy pixel
        blurred = work
        if self.cfg.floodfill_blur_ksize > 0:
            blurred = cv2.GaussianBlur(work, (self.cfg.floodfill_blur_ksize, self.cfg.floodfill_blur_ksize), 0)

        if np.isnan(ctx.heatmap_cx):
            ctx.debug_reason = "no_heatmap"
            return state

        hx, hy = ctx.heatmap_cx - x0, ctx.heatmap_cy - y0
        r = self.cfg.floodfill_seed_search
        sx0 = int(max(0, round(hx) - r))
        sy0 = int(max(0, round(hy) - r))
        sx1 = int(min(w, round(hx) + r + 1))
        sy1 = int(min(h, round(hy) + r + 1))
        patch = blurred[sy0:sy1, sx0:sx1]
        if patch.size == 0:
            ctx.debug_reason = "empty_seed_patch"
            return state
        py, px = np.unravel_index(np.argmin(patch), patch.shape)
        seed = (sx0 + int(px), sy0 + int(py))
        seed = (min(max(seed[0], 0), w - 1), min(max(seed[1], 0), h - 1))
        ctx.seed_x, ctx.seed_y = float(seed[0] + x0), float(seed[1] + y0)

        mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        cv2.floodFill(blurred.copy(), mask, seed, 0,
                      loDiff=self.cfg.floodfill_lo_diff, upDiff=self.cfg.floodfill_hi_diff, flags=flags)
        pupil_mask = mask[1:-1, 1:-1]

        # close first — "jumps over" the eyelash bridge across the pupil and joins both
        # halves of the blob into one; open after — trims thin "whiskers" (eyelash tails
        # outside the round pupil) that also got into the mask by brightness
        pupil_mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_CLOSE,
                                      np.ones((self.cfg.floodfill_close_ksize,) * 2, np.uint8))
        pupil_mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_OPEN,
                                      np.ones((self.cfg.floodfill_open_ksize,) * 2, np.uint8))
        state.dark_mask = pupil_mask

        contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            ctx.debug_reason = "no_contours"
            return state
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 5:
            ctx.debug_reason = "contour_too_small"
            return state

        area = cv2.contourArea(cnt)
        if area < self.cfg.floodfill_min_area:
            ctx.debug_reason = f"area_low:{area:.0f}"
            return state
        if area > self.cfg.floodfill_max_area_frac * h * w:
            ctx.debug_reason = f"area_high:{area:.0f}"
            return state

        touches_border = (
            cnt[:, 0, 0].min() <= 1 or cnt[:, 0, 1].min() <= 1 or
            cnt[:, 0, 0].max() >= w - 2 or cnt[:, 0, 1].max() >= h - 2
        )
        if touches_border:
            ctx.debug_reason = "touches_border"
            return state

        try:
            ell = cv2.fitEllipse(cnt)
        except cv2.error:
            ctx.debug_reason = "fit_error"
            return state

        (cx, cy), (A, B), ang = ell
        fill_frac = float(np.clip(area / (np.pi * (A / 2.0) * (B / 2.0) + 1e-9), 0, 1))
        aspect = ellipse_aspect_ratio(ell)

        if fill_frac < self.cfg.floodfill_min_fill_frac:
            ctx.debug_reason = f"fill_frac_low:{fill_frac:.2f}"
            return state
        if aspect > self.cfg.floodfill_max_aspect:
            ctx.debug_reason = f"aspect_high:{aspect:.2f}"
            return state

        state.best_ell = ell
        state.best_support = fill_frac
        ctx.pupil_cx, ctx.pupil_cy = cx + x0, cy + y0
        ctx.pupil_diameter = (A + B) / 2.0
        ctx.pupil_confidence = fill_frac
        ctx.pupil_A, ctx.pupil_B, ctx.pupil_angle = A, B, ang
        ctx.debug_reason = "ok"
        return state

    def detect(self, frame, ctx):
        state = PipelineState(frame_bgr=frame)
        state = self._compute1(state, ctx)
        state = self._compute_floodfill(state, ctx)
        return state


# ================================================================
# 2. Edge detector (fallback when floodfill fails)
# ================================================================

def compute_dark_region_mask(roi_gray, offset=18, smooth_hist=True):
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256]).flatten()
    if smooth_hist:
        hist = cv2.GaussianBlur(hist.reshape(-1, 1), (1, 9), 0).flatten()
    peak_idx = int(np.argmax(hist[:80]))
    thr = min(255, peak_idx + offset)
    return (roi_gray <= thr).astype(np.uint8) * 255, peak_idx, thr


def compute_specular_mask(roi_gray, thr=220, dilate=3):
    spec = (roi_gray >= thr).astype(np.uint8) * 255
    if dilate and dilate > 0:
        k = 2 * dilate + 1
        spec = cv2.dilate(spec, np.ones((k, k), np.uint8), iterations=1)
    return spec


def _angle_deg(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    v2 = v2 / (np.linalg.norm(v2) + 1e-9)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def split_contour_by_curvature(points, min_len=40, max_jump=6.0, max_corner_deg=75.0):
    if points is None or len(points) < 3:
        return []
    pts = points.astype(np.float32)
    segments = []
    seg = [pts[0]]
    for i in range(1, len(pts) - 1):
        p_prev, p, p_next = pts[i - 1], pts[i], pts[i + 1]
        if np.linalg.norm(p - p_prev) > max_jump:
            if len(seg) >= min_len:
                segments.append(np.array(seg, dtype=np.float32))
            seg = [p]
            continue
        if _angle_deg(p - p_prev, p_next - p) > max_corner_deg:
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


def ellipse_area(ell):
    (_, _), (A, B), _ = ell
    return float(np.pi * (A / 2.0) * (B / 2.0))


def ellipse_aspect_ratio_edge(ell):
    (_, _), (A, B), _ = ell
    if A <= 0 or B <= 0:
        return 1e9
    return float(max(A / B, B / A))


def fit_circle_kasa(points):
    """Algebraic (Kasa) circle fit — cheap, no iterations."""
    pts = points.astype(np.float64)
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, bb, c = sol
    r_sq = c + a ** 2 + bb ** 2
    if r_sq <= 0:
        return None
    return float(a), float(bb), float(np.sqrt(r_sq))


def circle_fit_rms(points, cx, cy, r):
    d = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    return float(np.sqrt(np.mean((d - r) ** 2)))


def fit_ellipse_sanity(seg, ccx, ccy, r, max_axis_ratio=2.5, min_axis_ratio=0.4,
                       require_vertical_major=True):
    """fitEllipseDirect with a plausibility check against the already-validated
    circle of the segment; None = ellipse cannot be trusted, use the circle."""
    try:
        ell = cv2.fitEllipseDirect(seg.reshape(-1, 1, 2).astype(np.float32))
    except cv2.error:
        return None
    (ecx, ecy), (A, B), ang = ell
    semi_major, semi_minor = max(A, B) / 2.0, min(A, B) / 2.0
    if semi_major > max_axis_ratio * r or semi_minor < min_axis_ratio * r:
        return None
    if np.hypot(ecx - ccx, ecy - ccy) > r:
        return None
    if require_vertical_major:
        theta = np.radians(ang)
        a, b = A / 2.0, B / 2.0
        bbox_w = 2 * np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)
        bbox_h = 2 * np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
        if bbox_w >= bbox_h:
            return None
    return ell


def ellipse_support_score(ell, edge_points_xy, dist_px=2.0):
    """Fraction of ROI edge pixels lying in a +-dist_px band around the ellipse contour."""
    if ell is None or edge_points_xy is None or len(edge_points_xy) == 0:
        return 0.0
    (cx, cy), (A, B), ang = ell
    a, b = (A / 2.0) + 1e-9, (B / 2.0) + 1e-9
    theta = np.deg2rad(ang)
    pts = edge_points_xy.astype(np.float32)
    x, y = pts[:, 0] - cx, pts[:, 1] - cy
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    xr = x * cos_t - y * sin_t
    yr = x * sin_t + y * cos_t
    val = (xr / a) ** 2 + (yr / b) ** 2
    tol = dist_px / min(a, b)
    return float(np.mean(np.abs(val - 1.0) <= tol))


@dataclass
class EdgeDetectorConfig:
    canny_low: int = 30
    canny_high: int = 90

    spec_thr: int = 220
    spec_dilate: int = 3

    split_min_len: int = 10
    split_max_jump: float = 6.0
    split_corner_deg: float = 75.0
    max_seg_straightness: float = 0.92   # chord/arc; closer to 1 = nearly straight (eyelash)

    circle_fit_max_rms: float = 0.8               # px — max RMS residual of points from the circle
    circle_fit_min_radius: Optional[float] = 6.0  # px — allowed pupil radius range
    circle_fit_max_radius: Optional[float] = 15.0

    max_center_dist: float = 0.9      # max candidate distance from ROI center (fraction of ROI radius)
    support_dist_px: float = 2.0
    support_min_frac: float = 0.20
    heatmap_prior_weight: float = 0.3

    heatmap_roi_size: int = 120
    heatmap_input_size: int = 256
    heatmap_kp_index: int = 1
    heatmap_sigma: float = 1.2


@dataclass
class EdgeFrameContext:
    frame_idx: int
    roi_rect: Tuple[int, int, int, int]
    pupil_cx: float = float("nan")
    pupil_cy: float = float("nan")
    pupil_diameter: float = float("nan")
    pupil_confidence: float = float("nan")
    pupil_A: float = float("nan")
    pupil_B: float = float("nan")
    pupil_angle: float = float("nan")
    heatmap_cx: float = float("nan")
    heatmap_cy: float = float("nan")


@dataclass
class EdgePipelineState:
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
    segments: List[np.ndarray] = field(default_factory=list)
    segment_circles: List = field(default_factory=list)
    candidates: List = field(default_factory=list)
    best_ell: Any = None
    best_support: float = 0.0


class EdgePupilDetector:
    def __init__(self, cfg: EdgeDetectorConfig, heatmap_model=None, device="cpu"):
        self.cfg = cfg
        self._heatmap_model = heatmap_model
        self._device = device

    def _heatmap_roi(self, gray, ctx):
        # reuse the heatmap center already computed by the floodfill stage when present,
        # so the fallback does not run the network a second time on the same frame
        if not np.isnan(ctx.heatmap_cx):
            cx, cy = int(round(ctx.heatmap_cx)), int(round(ctx.heatmap_cy))
        else:
            fx, fy = _infer_center(
                self._heatmap_model, gray, self._device,
                self.cfg.heatmap_input_size, self.cfg.heatmap_kp_index, self.cfg.heatmap_sigma,
            )
            cx, cy = int(round(fx)), int(round(fy))
            ctx.heatmap_cx, ctx.heatmap_cy = float(cx), float(cy)
        hs = self.cfg.heatmap_roi_size // 2
        h, w = gray.shape[:2]
        return (max(0, cx - hs), max(0, cy - hs), min(w, cx + hs), min(h, cy + hs))

    def _compute1(self, state, ctx):
        state.gray = cv2.cvtColor(state.frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._heatmap_model is not None:
            state.roi_rect = self._heatmap_roi(state.gray, ctx)
        else:
            state.roi_rect = ctx.roi_rect
        ctx.roi_rect = state.roi_rect
        x0, y0, x1, y1 = state.roi_rect
        state.roi_gray = state.gray[y0:y1, x0:x1]
        return state

    def _compute2(self, state, ctx):
        roi_blur = cv2.GaussianBlur(state.roi_gray, (5, 5), 0)
        state.edges = cv2.Canny(roi_blur, self.cfg.canny_low, self.cfg.canny_high)
        return state

    def _compute3(self, state, ctx):
        state.dark_mask, state.dark_peak, state.dark_thr = compute_dark_region_mask(state.roi_gray, offset=10)
        return state

    def _compute4(self, state, ctx):
        dark_dilated = cv2.dilate(state.dark_mask, np.ones((3, 3), np.uint8), iterations=1)
        state.edges_dark = cv2.bitwise_and(state.edges, dark_dilated)
        return state

    def _compute5(self, state, ctx):
        state.spec_mask = compute_specular_mask(state.roi_gray, thr=self.cfg.spec_thr, dilate=self.cfg.spec_dilate)
        state.edges_filtered = cv2.bitwise_and(state.edges_dark, cv2.bitwise_not(state.spec_mask))
        return state

    def _compute6(self, state, ctx):
        contours, _ = cv2.findContours(state.edges_filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        segments, segment_circles = [], []
        for cnt in contours:
            if cnt is None or len(cnt) < 5:
                continue
            pts = cnt.reshape(-1, 2).astype(np.float32)
            for seg in split_contour_by_curvature(
                pts, min_len=self.cfg.split_min_len, max_jump=self.cfg.split_max_jump,
                max_corner_deg=self.cfg.split_corner_deg,
            ):
                chord = float(np.linalg.norm(seg[-1] - seg[0]))
                arc = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))) + 1e-9
                if chord / arc > self.cfg.max_seg_straightness:
                    continue  # nearly straight — likely an eyelash
                circle = fit_circle_kasa(seg)
                if circle is None:
                    continue
                ccx, ccy, r = circle
                if circle_fit_rms(seg, ccx, ccy, r) > self.cfg.circle_fit_max_rms:
                    continue
                if self.cfg.circle_fit_min_radius is not None and not (
                    self.cfg.circle_fit_min_radius <= r <= self.cfg.circle_fit_max_radius
                ):
                    continue
                segments.append(seg)
                segment_circles.append((ccx, ccy, r))
        state.segments = segments
        state.segment_circles = segment_circles
        return state

    def _compute7(self, state, ctx):
        roi_h, roi_w = state.roi_gray.shape
        roi_cx, roi_cy = roi_w / 2.0, roi_h / 2.0
        roi_rad = 0.5 * min(roi_w, roi_h)
        candidates = []
        for seg, (ccx, ccy, r) in zip(state.segments, state.segment_circles):
            ell = fit_ellipse_sanity(seg, ccx, ccy, r)
            if ell is None:
                ell = ((ccx, ccy), (2 * r, 2 * r), 0.0)  # circle as an honest fallback
            area = ellipse_area(ell)
            (ecx, ecy), _, _ = ell
            asp = ellipse_aspect_ratio_edge(ell)
            dist = np.sqrt((ecx - roi_cx) ** 2 + (ecy - roi_cy) ** 2) / (roi_rad + 1e-9)
            if dist > self.cfg.max_center_dist:
                continue
            candidates.append((area, dist, asp, ell, seg))
        candidates.sort(key=lambda t: t[0], reverse=True)
        state.candidates = candidates
        return state

    def _compute8(self, state, ctx):
        ctx.pupil_cx = ctx.pupil_cy = ctx.pupil_diameter = ctx.pupil_confidence = float("nan")
        ctx.pupil_A = ctx.pupil_B = ctx.pupil_angle = float("nan")

        ys, xs = np.where(state.edges_filtered > 0)
        edge_pts = (np.stack([xs, ys], axis=1).astype(np.float32)
                    if len(xs) else np.zeros((0, 2), np.float32))

        roi_h, roi_w = state.roi_gray.shape
        roi_rad = 0.5 * min(roi_w, roi_h)
        x0, y0 = state.roi_rect[0], state.roi_rect[1]
        hm_roi_cx = ctx.heatmap_cx - x0 if not np.isnan(ctx.heatmap_cx) else None
        hm_roi_cy = ctx.heatmap_cy - y0 if not np.isnan(ctx.heatmap_cy) else None

        scored = []
        for (area, dist, asp, ell, seg) in state.candidates:
            (ecx, ecy), _, _ = ell
            support = ellipse_support_score(ell, edge_pts, dist_px=self.cfg.support_dist_px)
            heatmap_penalty = 0.0
            if hm_roi_cx is not None:
                hm_dist = np.sqrt((ecx - hm_roi_cx) ** 2 + (ecy - hm_roi_cy) ** 2) / (roi_rad + 1e-9)
                heatmap_penalty = self.cfg.heatmap_prior_weight * hm_dist
            final_score = support - 0.25 * dist - 0.05 * max(0.0, asp - 1.0) - heatmap_penalty
            scored.append((final_score, support, ell))

        if not scored:
            return state
        scored.sort(key=lambda t: t[0], reverse=True)
        _, best_support, best_ell = scored[0]
        if best_support < self.cfg.support_min_frac:
            return state

        state.best_ell = best_ell
        state.best_support = best_support
        (cx, cy), (A, B), ang = best_ell
        ctx.pupil_cx, ctx.pupil_cy = cx + x0, cy + y0
        ctx.pupil_A, ctx.pupil_B, ctx.pupil_angle = A, B, ang
        ctx.pupil_diameter = (A + B) / 2.0
        ctx.pupil_confidence = best_support
        return state

    def detect(self, frame, ctx):
        state = EdgePipelineState(frame_bgr=frame)
        for step in (self._compute1, self._compute2, self._compute3, self._compute4,
                     self._compute5, self._compute6, self._compute7, self._compute8):
            state = step(state, ctx)
        return state


# ================================================================
# 3. Combined: floodfill primary, edge fallback
# ================================================================

@dataclass
class CombinedFrameContext:
    frame_idx: int
    pupil_cx: float = float("nan")
    pupil_cy: float = float("nan")
    pupil_diameter: float = float("nan")
    pupil_confidence: float = float("nan")
    pupil_A: float = float("nan")
    pupil_B: float = float("nan")
    pupil_angle: float = float("nan")
    source: str = ""          # "floodfill" | "edge" | "none"
    debug_reason: str = ""


class CombinedPupilDetector:
    def __init__(self, floodfill_detector, edge_detector):
        self.floodfill_detector = floodfill_detector
        self.edge_detector = edge_detector

    @staticmethod
    def _copy_result(combined, ctx):
        combined.pupil_cx = ctx.pupil_cx
        combined.pupil_cy = ctx.pupil_cy
        combined.pupil_diameter = ctx.pupil_diameter
        combined.pupil_confidence = ctx.pupil_confidence
        combined.pupil_A = ctx.pupil_A
        combined.pupil_B = ctx.pupil_B
        combined.pupil_angle = ctx.pupil_angle

    def detect(self, frame, floodfill_ctx, edge_ctx):
        floodfill_state = self.floodfill_detector.detect(frame, floodfill_ctx)
        combined = CombinedFrameContext(frame_idx=floodfill_ctx.frame_idx)

        if not np.isnan(floodfill_ctx.pupil_cx) and floodfill_ctx.debug_reason == "ok":
            self._copy_result(combined, floodfill_ctx)
            combined.source = "floodfill"
            combined.debug_reason = "ok"
            return combined, floodfill_state, None

        # reuse the heatmap center from the floodfill stage in the edge fallback
        edge_ctx.heatmap_cx = floodfill_ctx.heatmap_cx
        edge_ctx.heatmap_cy = floodfill_ctx.heatmap_cy

        edge_state = self.edge_detector.detect(frame, edge_ctx)
        if not np.isnan(edge_ctx.pupil_cx):
            self._copy_result(combined, edge_ctx)
            combined.source = "edge"
            combined.debug_reason = f"floodfill_failed({floodfill_ctx.debug_reason})|edge_ok"
        else:
            combined.source = "none"
            combined.debug_reason = f"floodfill_failed({floodfill_ctx.debug_reason})|edge_failed"
        return combined, floodfill_state, edge_state


def build_combined_detector(floodfill_cfg: DetectorConfig, edge_cfg: EdgeDetectorConfig,
                            heatmap_ckpt: str, device: Optional[str] = None):
    """Load the HeatmapNet once and wire the floodfill + edge detectors into a Combined one."""
    if device is None:
        device = "cuda" if _cuda_available() else "cpu"
    heatmap_model = load_heatmap_model(heatmap_ckpt, device)
    floodfill_detector = PupilDetector(floodfill_cfg, heatmap_model=heatmap_model, device=device)
    edge_detector = EdgePupilDetector(edge_cfg, heatmap_model=heatmap_model, device=device)
    return CombinedPupilDetector(floodfill_detector, edge_detector), device


# ================================================================
# 4. Standalone helpers: run over video -> DataFrame; clean + smooth; render
# ================================================================

def run_combined_full_video(floodfill_cfg, edge_cfg, video_path, heatmap_model, device,
                            split_stereo=True, eye="left", start_frame=0, max_frames=None):
    import pandas as pd
    floodfill_detector = PupilDetector(floodfill_cfg, heatmap_model=heatmap_model, device=device)
    edge_detector = EdgePupilDetector(edge_cfg, heatmap_model=heatmap_model, device=device)
    combined_detector = CombinedPupilDetector(floodfill_detector, edge_detector)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"start_frame={start_frame} out of range")
    h, w = first.shape[:2]
    mid = w // 2 if split_stereo else w

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    floodfill_ctx = FrameContext(frame_idx=start_frame, roi_rect=(0, 0, mid, h))
    edge_ctx = EdgeFrameContext(frame_idx=start_frame, roi_rect=(0, 0, mid, h))

    rows = []
    frame_idx = start_frame
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and frame_idx - start_frame >= max_frames):
            break
        if split_stereo:
            eye_frame = frame[:, :mid].copy() if eye == "left" else frame[:, mid:mid * 2].copy()
        else:
            eye_frame = frame

        floodfill_ctx.frame_idx = edge_ctx.frame_idx = frame_idx
        combined_ctx, _, _ = combined_detector.detect(eye_frame, floodfill_ctx, edge_ctx)
        rows.append({
            "frame_idx": frame_idx,
            "cx": combined_ctx.pupil_cx, "cy": combined_ctx.pupil_cy,
            "diameter": combined_ctx.pupil_diameter,
            "A": combined_ctx.pupil_A, "B": combined_ctx.pupil_B, "angle": combined_ctx.pupil_angle,
            "confidence": combined_ctx.pupil_confidence,
            "source": combined_ctx.source or "none",
        })
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"...{frame_idx} frames processed")

    cap.release()
    df = pd.DataFrame(rows)
    print("source counts:", df["source"].value_counts().to_dict())
    return df


def clean_and_smooth(df, max_gap_interp=30, smooth_window=5):
    from scipy.signal import medfilt
    df = df.copy()
    is_blink = df["source"] == "blink"
    # both "none" and "blink" lack a valid ellipse -> null them before interpolation
    is_missing = (df["source"] == "none") | is_blink
    for col in ["cx", "cy", "diameter", "A", "B", "angle"]:
        df.loc[is_missing, col] = np.nan
        df[col] = df[col].interpolate(method="linear", limit=max_gap_interp, limit_area="inside")
        # do NOT interpolate blinks back — the eye is physically closed, no data
        df.loc[is_blink, col] = np.nan

    k = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    for col in ["cx", "cy", "diameter", "A", "B", "angle"]:
        out_col = f"{col}_clean"
        df[out_col] = df[col]
        valid_mask = df[col].notna()
        if valid_mask.sum() > k:
            filtered_full = df[col].copy()
            filtered_full.loc[valid_mask] = medfilt(df.loc[valid_mask, col].values, kernel_size=k)
            edge_mask = (df["source"] == "edge") & valid_mask
            df.loc[edge_mask, out_col] = filtered_full[edge_mask]
        df.loc[is_blink, out_col] = np.nan
    return df


def mark_blinks(df_left, df_right, min_blink_frames=2, max_blink_frames=None):
    """Variant A: a blink = pupil lost in BOTH eyes simultaneously over a
    continuous run of frames of suitable length."""
    dl, dr = df_left.copy(), df_right.copy()
    l = dl.set_index("frame_idx")["source"]
    r = dr.set_index("frame_idx")["source"]
    frames = sorted(set(l.index) | set(r.index))
    both_none = [(l.get(f, "none") == "none") and (r.get(f, "none") == "none") for f in frames]

    blink_frames, n_events = set(), 0
    i, n = 0, len(frames)
    while i < n:
        if both_none[i]:
            j = i
            while j < n and both_none[j]:
                j += 1
            run = frames[i:j]
            if len(run) >= min_blink_frames and (max_blink_frames is None or len(run) <= max_blink_frames):
                blink_frames.update(run)
                n_events += 1
            i = j
        else:
            i += 1

    for d in (dl, dr):
        d.loc[d["frame_idx"].isin(blink_frames), "source"] = "blink"
    print(f"Blinks: {n_events} events, {len(blink_frames)} frames total")
    return dl, dr


def render_clean_overlay_both_eyes(df_clean_left, df_clean_right, video_path, output_path, split_stereo=True):
    lookup_l = df_clean_left.set_index("frame_idx").to_dict("index")
    lookup_r = df_clean_right.set_index("frame_idx").to_dict("index")
    frame_indices = sorted(set(lookup_l.keys()) | set(lookup_r.keys()))
    start_frame, end_frame = frame_indices[0], frame_indices[-1]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"start_frame={start_frame} out of range")
    h, w = first.shape[:2]
    mid = w // 2 if split_stereo else w
    out_w, out_h = (mid * 2, h) if split_stereo else (w, h)

    writer = None
    for codec in ("mp4v", "avc1", "XVID", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, float(fps), (out_w, out_h))
        if writer.isOpened():
            print(f"Opened with codec: {codec}")
            break
        writer.release()
        writer = None
    if writer is None:
        raise RuntimeError("VideoWriter failed with all codecs")

    def draw_eye(eye_frame, row):
        out = eye_frame.copy()
        if row is not None:
            source = row["source"]
            if source == "blink":
                cv2.putText(out, "BLINK", (10, out.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cx, cy = row["cx_clean"], row["cy_clean"]
                A, B, angle = row["A_clean"], row["B_clean"], row["angle_clean"]
                if not (np.isnan(cx) or np.isnan(cy) or np.isnan(A) or np.isnan(B)):
                    color = {"floodfill": (0, 255, 0), "edge": (0, 165, 255)}.get(source, (0, 0, 255))
                    cv2.ellipse(out, (int(cx), int(cy)), (max(1, int(A / 2)), max(1, int(B / 2))),
                                float(angle), 0, 360, color, 2)
                    cv2.circle(out, (int(cx), int(cy)), 2, color, -1)
            cv2.putText(out, str(source), (5, out.shape[0] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx, written = start_frame, 0
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        left = frame[:, :mid].copy() if split_stereo else frame
        right = frame[:, mid:mid * 2].copy() if split_stereo else frame
        out_l = draw_eye(left, lookup_l.get(frame_idx))
        out_r = draw_eye(right, lookup_r.get(frame_idx))
        out = np.concatenate([out_l, out_r], axis=1) if split_stereo else out_l
        writer.write(out)
        frame_idx += 1
        written += 1

    cap.release()
    writer.release()
    print(f"Saved {written} frames to {output_path}")
    return output_path


# ================================================================
# 5. CLI (standalone debugging / thesis runs)
# ================================================================

def main():
    ap = argparse.ArgumentParser(description="Combined floodfill+edge pupil detector over a stereo eye video.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--heatmap_ckpt", required=True)
    ap.add_argument("--out", default="combined_overlay.mp4")
    ap.add_argument("--no_stereo", action="store_true", help="video is a single eye (not left|right)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if _cuda_available() else "cpu")
    heatmap_model = load_heatmap_model(args.heatmap_ckpt, device)
    split_stereo = not args.no_stereo

    floodfill_cfg = DetectorConfig()
    edge_cfg = EdgeDetectorConfig(heatmap_roi_size=35, circle_fit_min_radius=6.0, circle_fit_max_radius=15.0)

    df_left = run_combined_full_video(floodfill_cfg, edge_cfg, args.video, heatmap_model, device,
                                      split_stereo=split_stereo, eye="left", max_frames=args.max_frames)
    if split_stereo:
        df_right = run_combined_full_video(floodfill_cfg, edge_cfg, args.video, heatmap_model, device,
                                           split_stereo=split_stereo, eye="right", max_frames=args.max_frames)
        df_left, df_right = mark_blinks(df_left, df_right, min_blink_frames=2)
    else:
        df_right = df_left.copy()

    df_clean_left = clean_and_smooth(df_left)
    df_clean_right = clean_and_smooth(df_right)
    render_clean_overlay_both_eyes(df_clean_left, df_clean_right, args.video, args.out, split_stereo=split_stereo)


if __name__ == "__main__":
    main()
