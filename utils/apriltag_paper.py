from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import cv2

from pupil_apriltags import Detector  # pip install pupil-apriltags :contentReference[oaicite:2]{index=2}


@dataclass(frozen=True)
class PaperSpec:
    """
    Define how AprilTags correspond to paper corners (in paper coordinates).
    You MUST set this according to your printed layout.
    """
    # paper size in pixels of the rectified view
    paper_w: int = 1000
    paper_h: int = 700

    # Mapping: tag_id -> which corner of the paper it represents
    # corners: "TL", "TR", "BR", "BL"
    tag_to_corner: Dict[int, str] = None

    def __post_init__(self):
        if self.tag_to_corner is None:
            object.__setattr__(self, "tag_to_corner", {})


_CORNER_ORDER = ["TL", "TR", "BR", "BL"]

def _paper_corners(spec: PaperSpec) -> np.ndarray:
    # (4,2) in paper pixel coords
    return np.array([
        [0, 0],                       # TL
        [spec.paper_w - 1, 0],        # TR
        [spec.paper_w - 1, spec.paper_h - 1],  # BR
        [0, spec.paper_h - 1],        # BL
    ], dtype=np.float32)


def create_detector() -> Detector:
    # tag36h11 is the common one; adjust if your tags are different
    return Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )


def detect_tags(gray: np.ndarray, detector: Detector):
    """
    gray must be uint8 grayscale. :contentReference[oaicite:3]{index=3}
    Returns list of detections.
    """
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    if gray.ndim != 2:
        raise ValueError("detect_tags expects grayscale image (H,W)")
    return detector.detect(gray)


def estimate_paper_homography(
    frame_bgr: np.ndarray,
    spec: PaperSpec,
    detector: Detector,
) -> Tuple[Optional[np.ndarray], Dict[int, np.ndarray]]:
    """
    Returns:
      H_world_to_paper (3x3) or None if not enough tags
      tag_corners: dict tag_id -> (4,2) corners in world pixels
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    dets = detect_tags(gray, detector)

    tag_corners = {}
    for d in dets:
        tid = int(d.tag_id)
        # d.corners is (4,2): order is detector-defined; treat as polygon corners
        tag_corners[tid] = np.array(d.corners, dtype=np.float32)

    # We need 4 correspondences: paper corners in world pixels
    world_pts = []
    paper_pts = []

    paper_c = _paper_corners(spec)
    corner_name_to_idx = {name: i for i, name in enumerate(_CORNER_ORDER)}

    for tid, corner_name in spec.tag_to_corner.items():
        if tid not in tag_corners:
            continue

        # Use tag center as anchor OR a specific tag corner.
        # Safer: use tag CENTER (more stable under partial occlusion).
        # But we need one point per paper corner => use CENTER of each corner tag.
        poly = tag_corners[tid]
        center = poly.mean(axis=0)  # (2,)
        world_pts.append(center)

        pi = corner_name_to_idx[corner_name]
        paper_pts.append(paper_c[pi])

    if len(world_pts) < 4:
        return None, tag_corners

    world_pts = np.array(world_pts, dtype=np.float32)
    paper_pts = np.array(paper_pts, dtype=np.float32)

    H, inliers = cv2.findHomography(world_pts, paper_pts, method=0)
    if H is None:
        return None, tag_corners
    return H.astype(np.float64), tag_corners


def warp_world_to_paper(frame_bgr: np.ndarray, H_world_to_paper: np.ndarray, spec: PaperSpec) -> np.ndarray:
    return cv2.warpPerspective(frame_bgr, H_world_to_paper, (spec.paper_w, spec.paper_h))


def world_to_paper_uv(xw: float, yw: float, H_world_to_paper: np.ndarray, spec: PaperSpec) -> Tuple[float, float]:
    p = np.array([[[xw, yw]]], dtype=np.float32)  # (1,1,2)
    q = cv2.perspectiveTransform(p, H_world_to_paper)[0, 0]  # (2,)
    u = float(q[0]) / (spec.paper_w - 1)
    v = float(q[1]) / (spec.paper_h - 1)
    return u, v


def paper_uv_to_world(u: float, v: float, H_world_to_paper: np.ndarray, spec: PaperSpec) -> Tuple[float, float]:
    Hp = np.linalg.inv(H_world_to_paper)
    xp = u * (spec.paper_w - 1)
    yp = v * (spec.paper_h - 1)
    p = np.array([[[xp, yp]]], dtype=np.float32)
    q = cv2.perspectiveTransform(p, Hp)[0, 0]
    return float(q[0]), float(q[1])


def draw_tag_debug(frame_bgr: np.ndarray, tag_corners: Dict[int, np.ndarray]) -> np.ndarray:
    img = frame_bgr.copy()
    for tid, poly in tag_corners.items():
        poly_i = poly.astype(np.int32)
        cv2.polylines(img, [poly_i], isClosed=True, color=(0, 255, 0), thickness=2)
        c = poly.mean(axis=0).astype(int)
        cv2.putText(img, str(tid), (c[0], c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img
