import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

@dataclass
class PupilEllipse:
    cx: float
    cy: float
    a: float        # semi-major axis (px)
    b: float        # semi-minor axis (px)
    angle: float    # degrees
    score: float

class PupilLabsLikeDetector:
    """
    Pipeline aligned with Figure-style description:
      1) grayscale (+ mild denoise), ROI assumed already cropped to one eye
      2) Canny edges
      3) define "dark" region from histogram (low percentile + offset)
      4) filter edges to exclude specular reflections and edges not near "dark" region
      5) contours from remaining edges
      6) candidate ellipses via fitEllipse
      7) select best by score (edge support + dark interior + darkness)
    """

    def __init__(
        self,
        canny_sigma: float = 0.33,
        dark_percentile: float = 7.0,
        dark_offset: int = 8,
        highlight_percentile: float = 99.2,
        dark_dilate_ksize: int = 9,
        min_contour_points: int = 30,
        min_radius: int = 6,
        max_radius: int = 300,
        max_aspect: float = 3.8,
        edge_support_min: float = 0.30,
        perimeter_samples: int = 260,
    ):
        self.canny_sigma = canny_sigma
        self.dark_percentile = dark_percentile
        self.dark_offset = dark_offset
        self.highlight_percentile = highlight_percentile
        self.dark_dilate_ksize = dark_dilate_ksize
        self.min_contour_points = min_contour_points
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.max_aspect = max_aspect
        self.edge_support_min = edge_support_min
        self.perimeter_samples = perimeter_samples

    @staticmethod
    def _auto_canny(gray: np.ndarray, sigma: float) -> np.ndarray:
        v = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(gray, lower, upper, L2gradient=True)

    @staticmethod
    def _ellipse_perimeter_points(cx: float, cy: float, a: float, b: float, angle_deg: float, n: int):
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        pts = np.stack([a * np.cos(t), b * np.sin(t)], axis=1)

        ang = np.deg2rad(angle_deg)
        R = np.array([[np.cos(ang), -np.sin(ang)],
                      [np.sin(ang),  np.cos(ang)]], dtype=np.float32)
        pts = pts @ R.T
        pts[:, 0] += cx
        pts[:, 1] += cy
        return pts

    def _score_ellipse(self, gray: np.ndarray, edges: np.ndarray, dark_mask: np.ndarray, ell) -> Optional[PupilEllipse]:
        (cx, cy), (MA, ma), angle = ell
        a = float(max(MA, ma) / 2.0)
        b = float(min(MA, ma) / 2.0)
        if b <= 1.0:
            return None

        # geometry constraints
        if a < self.min_radius or b < self.min_radius:
            return None
        if a > self.max_radius or b > self.max_radius:
            return None
        if (a / b) > self.max_aspect:
            return None

        H, W = edges.shape[:2]

        # edge support along ellipse perimeter (with 3x3 tolerance)
        pts = self._ellipse_perimeter_points(cx, cy, a, b, angle, self.perimeter_samples)
        p = np.round(pts).astype(np.int32)
        ok = (p[:, 0] >= 1) & (p[:, 0] < W - 1) & (p[:, 1] >= 1) & (p[:, 1] < H - 1)
        p = p[ok]
        if len(p) < 60:
            return None

        support = 0
        for x, y in p:
            if np.any(edges[y-1:y+2, x-1:x+2] > 0):
                support += 1
        edge_support = support / len(p)
        if edge_support < self.edge_support_min:
            return None

        # dark interior preference
        fill = np.zeros_like(gray, dtype=np.uint8)
        cv2.ellipse(fill, ell, 255, -1)
        inside = fill > 0
        if inside.sum() < 120:
            return None

        dark_frac = float((dark_mask[inside] > 0).mean())
        mean_inside = float(gray[inside].mean())
        mean_norm = 1.0 - (mean_inside / 255.0)

        # score weights (stable + explainable)
        score = 0.55 * edge_support + 0.30 * dark_frac + 0.15 * mean_norm

        return PupilEllipse(
            cx=float(cx), cy=float(cy), a=a, b=b, angle=float(angle), score=float(score)
        )

    def detect_one_eye(self, eye_bgr: np.ndarray, return_debug: bool = False):
        """
        Input: BGR ROI of ONE EYE.
        Output: PupilEllipse or None. If return_debug=True, returns (result, debug_dict).
        """
        if eye_bgr is None or eye_bgr.size == 0:
            if return_debug:
                return None, {"error": "empty_input"}
            return None

        # 1) grayscale + mild denoise + equalize (helps histogram-based thresholding)
        gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.equalizeHist(gray)

        # 2) Canny
        edges = self._auto_canny(gray, self.canny_sigma)

        # 3) define "dark" region from histogram low percentile + offset
        p = np.percentile(gray, self.dark_percentile)
        dark_thr = int(np.clip(p + self.dark_offset, 0, 255))
        dark_mask = (gray <= dark_thr).astype(np.uint8) * 255

        # 4) specular highlights (very bright) -> remove those edge pixels
        hi_thr = int(np.clip(np.percentile(gray, self.highlight_percentile), 0, 255))
        highlight = (gray >= hi_thr).astype(np.uint8) * 255

        edges_clean = edges.copy()
        edges_clean[highlight > 0] = 0

        # keep edges only near dark region (dilate dark mask)
        k = max(3, int(self.dark_dilate_ksize) | 1)  # odd >=3
        dark_dil = cv2.dilate(
            dark_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )
        edges_clean[dark_dil == 0] = 0

        # 5) contours
        cnts, _ = cv2.findContours(edges_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        best = None
        best_ell = None

        # 6) fit ellipses (candidates)
        for c in cnts:
            if len(c) < self.min_contour_points or len(c) < 5:
                continue
            try:
                ell = cv2.fitEllipse(c)
            except cv2.error:
                continue

            cand = self._score_ellipse(gray, edges_clean, dark_mask, ell)
            if cand is None:
                continue
            if best is None or cand.score > best.score:
                best, best_ell = cand, ell

        if not return_debug:
            return best

        dbg: Dict[str, Any] = {
            "gray": gray,
            "edges": edges,
            "edges_clean": edges_clean,
            "dark_mask": dark_mask,
            "highlight": highlight,
            "dark_thr": dark_thr,
            "hi_thr": hi_thr,
        }
        if best_ell is not None:
            overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            overlay[edges_clean > 0] = (0, 255, 0)
            overlay[highlight > 0] = (0, 255, 255)
            cv2.ellipse(overlay, best_ell, (0, 0, 255), 2)
            cv2.circle(overlay, (int(best.cx), int(best.cy)), 2, (0, 0, 255), -1)
            dbg["overlay"] = overlay

        return best, dbg
