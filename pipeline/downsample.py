import argparse
import numpy as np
import pandas as pd

PUPILS_CSV  = "outputs/pupils_pupils.csv"
OUTPUT_CSV  = "30_fps_data.csv"
TARGET_FPS  = 30

COLS_L = ["xL", "yL", "diameter_L", "confidence_L"]
COLS_R = ["xR", "yR", "diameter_R", "confidence_R"]


def weighted_avg(chunk: pd.DataFrame, cols: list) -> dict:
    conf_col = cols[3]
    valid = chunk[cols].copy()
    has_conf = valid[conf_col].notna()

    if has_conf.any():
        valid = valid[has_conf]
        w = valid[conf_col].to_numpy()
        w = w / w.sum()
        result = {col: float((valid[col] * w).sum()) for col in cols[:3]}
        result[conf_col] = float(valid[conf_col].mean())
    elif len(valid) > 0:
        result = {col: float(valid[col].mean()) for col in cols[:3]}
        result[conf_col] = np.nan
    else:
        result = {col: np.nan for col in cols}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/Vadim")
    args = parser.parse_args()

    pupils = pd.read_csv(PUPILS_CSV)
    ts_eye = pd.read_csv(f"{args.data_dir}/csv/timestamps.csv")

    t0 = ts_eye["timestamp_ns"].iloc[0]
    t1 = ts_eye["timestamp_ns"].iloc[-1]
    n_scene  = round((t1 - t0) / 1e9 * TARGET_FPS)
    half_win = int(1e9 / TARGET_FPS / 2)

    pupil_ts        = pupils["timestamp [ns]"].to_numpy(np.int64)
    scene_timestamps = np.linspace(t0, t1, n_scene, dtype=np.int64)

    print(f"[OK] pupils: {len(pupils)} rows, scene frames: {n_scene}, window: ±{half_win/1e6:.1f} ms")

    rows = []
    for t in scene_timestamps:
        mask  = np.abs(pupil_ts - t) <= half_win
        chunk = pupils[mask]
        row   = {"timestamp [ns]": t}
        row.update(weighted_avg(chunk, COLS_L))
        row.update(weighted_avg(chunk, COLS_R))
        rows.append(row)

    df = pd.DataFrame(rows, columns=["timestamp [ns]"] + COLS_L + COLS_R)

    df["xL"] = 192 - df["xL"]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] saved: {OUTPUT_CSV}")
    print(f"[OK] frames without data: {df['xL'].isna().sum()}")


if __name__ == "__main__":
    main()
