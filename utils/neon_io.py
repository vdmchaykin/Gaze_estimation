import re
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------- Reading ----------

def read_gaze_csv(path: str) -> pd.DataFrame:
    gaze = pd.read_csv(path)
    # normalize timestamp dtype
    gaze["timestamp [ns]"] = gaze["timestamp [ns]"].astype("int64")
    return gaze


def read_events_csv(path: str) -> pd.DataFrame:
    ev = pd.read_csv(path)
    ev["timestamp [ns]"] = ev["timestamp [ns]"].astype("int64")
    return ev.sort_values("timestamp [ns]").reset_index(drop=True)


def read_eye_states_csv(path: str) -> pd.DataFrame:
    st = pd.read_csv(path)
    st["timestamp [ns]"] = st["timestamp [ns]"].astype("int64")
    return st


# ---------- Test intervals from events.csv ----------

Interval = Tuple[str, int, int]  # (test_id, start_ns, end_ns)

def extract_test_intervals(events: pd.DataFrame) -> List[Interval]:
    """
    Extract intervals like:
      test01_begin ... test01_end
      test02_begin ... test02_end
    Returns list of (testXX, start_ns, end_ns), sorted by start_ns.
    """
    ev = events.sort_values("timestamp [ns]").reset_index(drop=True)

    begin = ev[ev["name"].astype(str).str.match(r"^test\d+_begin$")].copy()
    end   = ev[ev["name"].astype(str).str.match(r"^test\d+_end$")].copy()

    def key(name: str):
        m = re.match(r"^(test\d+)_", str(name))
        return m.group(1) if m else None

    begin["k"] = begin["name"].map(key)
    end["k"]   = end["name"].map(key)

    intervals: List[Interval] = []
    for k, b in begin.groupby("k"):
        start_ns = int(b["timestamp [ns]"].iloc[0])
        e_row = end[end["k"] == k]
        if len(e_row) == 0:
            continue
        end_ns = int(e_row["timestamp [ns]"].iloc[0])
        if end_ns > start_ns:
            intervals.append((k, start_ns, end_ns))

    intervals.sort(key=lambda x: x[1])
    if len(intervals) == 0:
        raise RuntimeError("No testXX_begin/testXX_end intervals found in events.csv")
    return intervals


# ---------- Filtering by intervals ----------

def mask_by_intervals(timestamps_ns: np.ndarray, intervals: List[Interval]) -> np.ndarray:
    t = timestamps_ns.astype(np.int64)
    m = np.zeros_like(t, dtype=bool)
    for _, a, b in intervals:
        m |= (t >= a) & (t <= b)
    return m


def filter_df_by_intervals(df: pd.DataFrame, ts_col: str, intervals: List[Interval]) -> pd.DataFrame:
    t = df[ts_col].astype("int64").values
    m = mask_by_intervals(t, intervals)
    return df[m].copy()


# ---------- Common gaze filters ----------

def filter_gaze_quality(gaze: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only gaze rows that are meaningful for supervised mapping.
    - worn == 1 (if present)
    - remove blinks (if present)
    - require gaze x/y columns to exist & be non-null
    """
    g = gaze.copy()

    if "worn" in g.columns:
        g = g[g["worn"] == 1.0]

    if "blink id" in g.columns:
        g = g[g["blink id"].isna()]

    # enforce presence
    required = ["gaze x [px]", "gaze y [px]"]
    for c in required:
        if c not in g.columns:
            raise KeyError(f"gaze.csv missing required column: '{c}'")
    g = g.dropna(subset=required)

    return g.reset_index(drop=True)
