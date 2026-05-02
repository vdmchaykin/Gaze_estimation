from typing import Tuple, Optional
import cv2
import numpy as np


def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, fps, nframes, w, h


def read_frame(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    """Random-access frame read (slow but ok for debugging / sparse sampling)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def bgr_to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def split_eyes(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Neon eye video is typically two eyes side-by-side in one frame.
    Returns: (left_eye_gray, right_eye_gray)
    """
    gray = bgr_to_gray(frame)
    h, w = gray.shape[:2]
    mid = w // 2
    left = gray[:, :mid]
    right = gray[:, mid:]
    return left, right


def save_debug_eye_frames(
    eye_video_path: str,
    out_path_left: str,
    out_path_right: str,
    out_path_full: str,
    frame_idx: int = 0,
):
    """
    Saves:
    - full eye frame
    - left crop
    - right crop
    """
    cap, fps, nframes, w, h = open_video(eye_video_path)
    frame = read_frame(cap, frame_idx)
    cap.release()

    if frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {eye_video_path}")

    left, right = split_eyes(frame)
    full_gray = bgr_to_gray(frame)

    cv2.imwrite(out_path_full, full_gray)
    cv2.imwrite(out_path_left, left)
    cv2.imwrite(out_path_right, right)

    return dict(fps=fps, nframes=nframes, w=w, h=h)
