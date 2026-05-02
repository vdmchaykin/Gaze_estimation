import cv2
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

FEATURES = ["L_cx_px", "L_cy_px", "R_cx_px", "R_cy_px"]
EXTRA = ["xm", "ym", "dx", "dy"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["xm"] = (df["L_cx_px"] + df["R_cx_px"]) / 2.0
    df["ym"] = (df["L_cy_px"] + df["R_cy_px"]) / 2.0
    df["dx"] = df["R_cx_px"] - df["L_cx_px"]
    df["dy"] = df["R_cy_px"] - df["L_cy_px"]
    return df

def load_world_timestamps(path="world_timestamps.csv") -> pd.DataFrame:
    world = pd.read_csv(path)

    # На практике встречаются разные варианты названия
    if "timestamp [ns]" in world.columns:
        world = world.rename(columns={"timestamp [ns]": "timestamp"})
    elif "timestamp_ns" in world.columns:
        world = world.rename(columns={"timestamp_ns": "timestamp"})
    elif "timestamp" not in world.columns:
        raise ValueError(f"{path}: timestamp column not found. Columns: {list(world.columns)}")

    world["timestamp"] = pd.to_numeric(world["timestamp"], errors="coerce").round().astype("Int64")
    world = world.dropna(subset=["timestamp"]).copy()
    world["timestamp"] = world["timestamp"].astype(np.int64)

    world = world.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # ВАЖНО: делаем уникальное имя, чтобы не было конфликтов
    world["frame_idx_world"] = np.arange(len(world), dtype=np.int64)
    return world[["timestamp", "frame_idx_world"]]

def main():
    # ===== 1) TRAIN on Test B =====
    train = pd.read_csv("2024-11-20_15-32-05-32205695/test_A.csv")
    train = train.dropna(subset=FEATURES + ["gaze_x", "gaze_y", "timestamp"]).copy()
    train["timestamp"] = pd.to_numeric(train["timestamp"], errors="coerce").round().astype("Int64")
    train = train.dropna(subset=["timestamp"]).copy()
    train["timestamp"] = train["timestamp"].astype(np.int64)
    train = train.dropna(subset=["timestamp"])

    train = add_features(train)

    X_train = train[FEATURES + EXTRA].to_numpy(np.float64)
    Y_train = train[["gaze_x", "gaze_y"]].to_numpy(np.float64)

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=1.0, random_state=0)),
    ])
    model.fit(X_train, Y_train)

    # ===== 2) PREDICT on Test A =====
    test = pd.read_csv("2024-11-20_15-32-05-32205695/test_B.csv")
    test = test.dropna(subset=FEATURES + ["timestamp"]).copy()
    test["timestamp"] = pd.to_numeric(test["timestamp"], errors="coerce").round().astype("Int64")
    test = test.dropna(subset=["timestamp"]).copy()
    test["timestamp"] = test["timestamp"].astype(np.int64)
    test = test.dropna(subset=["timestamp"])

    test = add_features(test)

    X_test = test[FEATURES + EXTRA].to_numpy(np.float64)
    pred = model.predict(X_test)
    test["pred_gaze_x"] = pred[:, 0]
    test["pred_gaze_y"] = pred[:, 1]

    # ===== 3) MATCH test timestamps → world frames =====
    world = load_world_timestamps("world_timestamps.csv")

    matched = pd.merge_asof(
        test.sort_values("timestamp"),
        world.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=5_000_000  # 5 ms в ns, INT
    )


    # теперь колонка гарантированно frame_idx_world, и она точно есть
    matched = matched.dropna(subset=["frame_idx_world"]).copy()
    matched["frame_idx_world"] = matched["frame_idx_world"].astype(int)

    print("[OK] matched rows:", len(matched))
    print("[OK] matched columns:", list(matched.columns))

    # ===== 4) OVERLAY =====
    cap = cv2.VideoCapture("scene_video.mp4")
    if not cap.isOpened():
        raise RuntimeError("Cannot open scene_video.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "scene_with_gaze_TEST_ONLY.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    # mapping: frame_idx -> (x,y)
    draw = {
        int(r.frame_idx_world): (int(round(r.pred_gaze_x)), int(round(r.pred_gaze_y)))
        for r in matched.itertuples(index=False)
    }

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if i in draw:
            x, y = draw[i]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

        out.write(frame)
        i += 1

    cap.release()
    out.release()
    print("✔ overlay saved: scene_with_gaze_TEST_ONLY.mp4")

if __name__ == "__main__":
    main()
