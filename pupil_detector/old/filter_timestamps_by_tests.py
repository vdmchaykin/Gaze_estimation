import pandas as pd

EVENTS_CSV = "events.csv"
PUPIL_CSV = "pupil_by_timestamp.csv"
OUT_CSV = "pupil_by_timestamp_tests_only.csv"

def main():
    events = pd.read_csv(EVENTS_CSV)

    # columns in events.csv (as on your screenshot)
    # recording id, timestamp [ns], name, type
    if "timestamp [ns]" not in events.columns or "name" not in events.columns:
        raise RuntimeError(f"events.csv must contain 'timestamp [ns]' and 'name'. Got: {list(events.columns)}")

    # find test intervals
    tests = {}
    for _, row in events.iterrows():
        name = str(row["name"])
        ts = int(row["timestamp [ns]"])

        if name.endswith("_begin"):
            test = name[:-6]  # remove "_begin"
            tests.setdefault(test, {})["begin"] = ts
        elif name.endswith("_end"):
            test = name[:-4]  # remove "_end"
            tests.setdefault(test, {})["end"] = ts

    # keep only tests (by your note: exactly 2 tests)
    # but script works for any number
    intervals = []
    for test, d in tests.items():
        if "begin" in d and "end" in d:
            b, e = d["begin"], d["end"]
            if b < e:
                intervals.append((test, b, e))

    if not intervals:
        raise RuntimeError("No valid test intervals found (need *_begin and *_end).")

    intervals.sort(key=lambda x: x[1])
    print("Test intervals:")
    for t, b, e in intervals:
        print(f"  {t}: {b} -> {e}")

    pupil = pd.read_csv(PUPIL_CSV)

    # pupil CSV column name can be "timestamp_ns" (from our generator)
    if "timestamp_ns" not in pupil.columns:
        raise RuntimeError(f"{PUPIL_CSV} must contain 'timestamp_ns'. Got: {list(pupil.columns)}")

    # filter
    mask = False
    for _, b, e in intervals:
        mask |= (pupil["timestamp_ns"] >= b) & (pupil["timestamp_ns"] <= e)

    out = pupil[mask].copy()

    # optional: add test_name column (so you know which test each row belongs to)
    def which_test(ts):
        for t, b, e in intervals:
            if b <= ts <= e:
                return t
        return None

    out["test_name"] = out["timestamp_ns"].apply(which_test)

    out.to_csv(OUT_CSV, index=False)

    print(f"[OK] saved {OUT_CSV}")
    print(f"rows before: {len(pupil)}")
    print(f"rows after : {len(out)}")

if __name__ == "__main__":
    main()
