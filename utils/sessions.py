from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import os

@dataclass(frozen=True)
class Session:
    """One recording/session."""
    session_id: str
    gendata_dir: Path           # data/GenData/<session_id>
    gaze_csv: Path
    events_csv: Path
    eye_states_csv: Path
    world_video: Path
    eye_video: Path

def _must_exist(p: Path, what: str):
    if not p.exists():
        raise FileNotFoundError(f"Missing {what}: {p}")

def discover_sessions(
    gendata_root: str,
    world_video_name: str,
    eye_video_name: str,
):
    gendata_root = Path(gendata_root).expanduser().resolve()
    _must_exist(gendata_root, "gendata_root directory")

    sessions = []

    for d in sorted([p for p in gendata_root.iterdir() if p.is_dir()]):
        session_id = d.name

        gaze_csv = d / "gaze.csv"
        events_csv = d / "events.csv"
        eye_states_csv = d / "3d_eye_states.csv"

        world_video = d / world_video_name
        eye_video = d / eye_video_name

        _must_exist(gaze_csv, f"gaze.csv for session {session_id}")
        _must_exist(events_csv, f"events.csv for session {session_id}")
        _must_exist(eye_states_csv, f"3d_eye_states.csv for session {session_id}")
        _must_exist(world_video, f"world video for session {session_id}")
        _must_exist(eye_video, f"eye video for session {session_id}")

        sessions.append(Session(
            session_id=session_id,
            gendata_dir=d,
            gaze_csv=gaze_csv,
            events_csv=events_csv,
            eye_states_csv=eye_states_csv,
            world_video=world_video,
            eye_video=eye_video,
        ))

    if not sessions:
        raise RuntimeError(f"No session folders found in {gendata_root}")

    return sessions


def split_train_val(sessions: List[Session], val_session_id: str) -> Tuple[List[Session], Session]:
    """Use 1 session as validation, remaining as train."""
    val = None
    train: List[Session] = []
    for s in sessions:
        if s.session_id == val_session_id:
            val = s
        else:
            train.append(s)

    if val is None:
        known = [s.session_id for s in sessions]
        raise ValueError(
            f"--val_session='{val_session_id}' not found.\n"
            f"Available sessions:\n  " + "\n  ".join(known)
        )
    if len(train) == 0:
        raise ValueError("Validation session consumed all sessions. Need at least 1 train session.")

    return train, val
