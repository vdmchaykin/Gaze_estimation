"""
Converts Neon eye tracker binary recordings to CSV files.

Supported files:
  gaze ps1.raw + .time       -> gaze.csv
  gaze_right ps1.raw + .time -> gaze_right.csv
  fixations ps1.raw + .time  -> fixations.csv
  imu ps1.raw                -> imu.csv          (timestamp embedded in data)
  extimu ps1.raw + .time     -> extimu.csv       (2-byte-length-prefixed protobuf ImuPacket)
  worn ps1.raw + gaze .time  -> worn.csv
  event.txt + event.time     -> event.csv

Usage:
  python convert_to_csv.py <recording_dir>
  python convert_to_csv.py  (uses the directory this script is in)
"""

import csv
import os
import struct
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def read_timestamps(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype="<i8")


def read_raw(path: str, dtype: np.dtype) -> np.ndarray:
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=dtype)


# ---------------------------------------------------------------------------
# Protobuf minimal decoder for ImuPacket
# ---------------------------------------------------------------------------

def _read_varint(data: bytes, pos: int):
    result, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _read_float(data: bytes, pos: int):
    return struct.unpack_from("<f", data, pos)[0], pos + 4


def _parse_sub_message(data: bytes, pos: int, length: int):
    """Parse a nested message and return all float32 values in field order."""
    end = pos + length
    fields = {}
    while pos < end:
        tag, pos = _read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 7
        if wire_type == 0:          # varint (reserved int32)
            val, pos = _read_varint(data, pos)
        elif wire_type == 5:        # 32-bit fixed (float)
            val, pos = _read_float(data, pos)
            fields[field_num] = val
        else:
            raise ValueError(f"Unexpected wire type {wire_type} in sub-message")
    return fields, pos


def parse_extimu_packet(msg_bytes: bytes):
    """
    Parse one ImuPacket protobuf message (without length prefix).
    Returns dict with: timestamp_ns, accel_x/y/z, gyro_x/y/z, quat_w/x/y/z
    """
    result = {}
    pos = 0
    while pos < len(msg_bytes):
        tag, pos = _read_varint(msg_bytes, pos)
        field_num = tag >> 3
        wire_type = tag & 7

        if wire_type == 0:          # varint
            val, pos = _read_varint(msg_bytes, pos)
            if field_num == 1:
                result["timestamp_ns"] = val
        elif wire_type == 2:        # length-delimited (sub-message)
            length, pos = _read_varint(msg_bytes, pos)
            sub, pos = _parse_sub_message(msg_bytes, pos, length)
            if field_num == 2:      # accelData
                result["accel_x"] = sub.get(1, float("nan"))
                result["accel_y"] = sub.get(2, float("nan"))
                result["accel_z"] = sub.get(3, float("nan"))
            elif field_num == 3:    # gyroData
                result["gyro_x"] = sub.get(1, float("nan"))
                result["gyro_y"] = sub.get(2, float("nan"))
                result["gyro_z"] = sub.get(3, float("nan"))
            elif field_num == 4:    # quaternionData
                result["quat_w"] = sub.get(1, float("nan"))
                result["quat_x"] = sub.get(2, float("nan"))
                result["quat_y"] = sub.get(3, float("nan"))
                result["quat_z"] = sub.get(4, float("nan"))
        else:
            raise ValueError(f"Unexpected wire type {wire_type} in ImuPacket")
    return result


# ---------------------------------------------------------------------------
# Per-stream converters
# ---------------------------------------------------------------------------

def convert_gaze(raw_path: str, time_path: str, out_path: str, stream_name: str = "gaze"):
    dtype = np.dtype([("x", "<f4"), ("y", "<f4")])
    data = read_raw(raw_path, dtype)
    timestamps = read_timestamps(time_path)
    assert len(data) == len(timestamps), (
        f"{stream_name}: record count mismatch {len(data)} vs {len(timestamps)}"
    )
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "x", "y"])
        for ts, row in zip(timestamps, data):
            writer.writerow([ts, row["x"], row["y"]])
    print(f"  {stream_name}: {len(data)} rows -> {out_path}")


def convert_fixations(raw_path: str, time_path: str, out_path: str):
    dtype = np.dtype([
        ("event_type",         "<i4"),
        ("start_timestamp_ns", "<i8"),
        ("end_timestamp_ns",   "<i8"),
        ("start_gaze_x",       "<f4"),
        ("start_gaze_y",       "<f4"),
        ("end_gaze_x",         "<f4"),
        ("end_gaze_y",         "<f4"),
        ("mean_gaze_x",        "<f4"),
        ("mean_gaze_y",        "<f4"),
        ("amplitude_pixels",   "<f4"),
        ("amplitude_angle_deg","<f4"),
        ("mean_velocity",      "<f4"),
        ("max_velocity",       "<f4"),
    ])
    data = read_raw(raw_path, dtype)
    timestamps = read_timestamps(time_path)
    assert len(data) == len(timestamps), (
        f"fixations: record count mismatch {len(data)} vs {len(timestamps)}"
    )
    cols = ["timestamp_ns"] + list(dtype.names)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for ts, row in zip(timestamps, data):
            writer.writerow([ts] + [row[n] for n in dtype.names])
    print(f"  fixations: {len(data)} rows -> {out_path}")


def convert_imu(raw_path: str, out_path: str):
    # imu.dtype embeds timestamp_ns as the first field
    dtype = np.dtype([
        ("timestamp_ns", "<i8"),
        ("gyro_x",       "<f4"),
        ("gyro_y",       "<f4"),
        ("gyro_z",       "<f4"),
        ("accel_x",      "<f4"),
        ("accel_y",      "<f4"),
        ("accel_z",      "<f4"),
        ("quat_w",       "<f4"),
        ("quat_x",       "<f4"),
        ("quat_y",       "<f4"),
        ("quat_z",       "<f4"),
    ])
    data = read_raw(raw_path, dtype)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(dtype.names)
        for row in data:
            writer.writerow([row[n] for n in dtype.names])
    print(f"  imu: {len(data)} rows -> {out_path}")


def convert_extimu(raw_path: str, out_path: str):
    with open(raw_path, "rb") as f:
        raw = f.read()

    cols = ["timestamp_ns", "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z", "quat_w", "quat_x", "quat_y", "quat_z"]
    rows = []
    pos = 0
    while pos < len(raw) - 2:
        length = struct.unpack_from("<H", raw, pos)[0]
        pos += 2
        if length == 0 or pos + length > len(raw):
            break
        packet = parse_extimu_packet(raw[pos:pos + length])
        rows.append([packet.get(c, float("nan")) for c in cols])
        pos += length

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)
    print(f"  extimu: {len(rows)} rows -> {out_path}")


def convert_timestamps(time_path: str, out_path: str):
    timestamps = read_timestamps(time_path)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns"])
        for ts in timestamps:
            writer.writerow([ts])
    print(f"  timestamps: {len(timestamps)} rows -> {out_path}")


def convert_worn(raw_path: str, time_path: str, out_path: str):
    data = read_raw(raw_path, np.dtype("u1"))
    timestamps = read_timestamps(time_path)
    assert len(data) == len(timestamps), (
        f"worn: record count mismatch {len(data)} vs {len(timestamps)}"
    )
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "worn"])
        for ts, val in zip(timestamps, data):
            writer.writerow([ts, int(val)])
    print(f"  worn: {len(data)} rows -> {out_path}")


def convert_events(txt_path: str, time_path: str, out_path: str):
    with open(txt_path, "r") as f:
        events = [line.rstrip("\n") for line in f]
    timestamps = read_timestamps(time_path)
    assert len(events) == len(timestamps), (
        f"events: count mismatch {len(events)} vs {len(timestamps)}"
    )
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "event"])
        for ts, ev in zip(timestamps, events):
            writer.writerow([ts, ev])
    print(f"  events: {len(events)} rows -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert_recording(directory: str):
    d = directory
    out_dir = os.path.join(d, "csv")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Converting recording in: {d}")
    print(f"Output directory:        {out_dir}")

    def path(*parts):
        return os.path.join(d, *parts)

    errors = []

    def try_convert(name, fn, *args):
        # last arg is the output path (doesn't exist yet), check only inputs
        missing = [a for a in args[:-1] if isinstance(a, str) and not os.path.exists(a)]
        if missing:
            print(f"  {name}: SKIPPED (missing: {', '.join(os.path.basename(m) for m in missing)})")
            return
        try:
            fn(*args)
        except Exception as e:
            print(f"  {name}: ERROR — {e}")
            errors.append((name, e))

    try_convert("gaze",
        convert_gaze,
        path("gaze ps1.raw"), path("gaze ps1.time"),
        os.path.join(out_dir, "gaze.csv"))

    try_convert("gaze_right",
        lambda *a: convert_gaze(*a, stream_name="gaze_right"),
        path("gaze_right ps1.raw"), path("gaze_right ps1.time"),
        os.path.join(out_dir, "gaze_right.csv"))

    try_convert("fixations",
        convert_fixations,
        path("fixations ps1.raw"), path("fixations ps1.time"),
        os.path.join(out_dir, "fixations.csv"))

    try_convert("imu",
        convert_imu,
        path("imu ps1.raw"),
        os.path.join(out_dir, "imu.csv"))

    try_convert("extimu",
        convert_extimu,
        path("extimu ps1.raw"),
        os.path.join(out_dir, "extimu.csv"))

    # worn has no .time file — uses gaze timestamps (same frame rate)
    try_convert("timestamps",
        convert_timestamps,
        path("gaze ps1.time"),
        os.path.join(out_dir, "timestamps.csv"))

    try_convert("worn",
        convert_worn,
        path("worn ps1.raw"), path("gaze ps1.time"),
        os.path.join(out_dir, "worn.csv"))

    try_convert("events",
        convert_events,
        path("event.txt"), path("event.time"),
        os.path.join(out_dir, "event.csv"))

    if errors:
        print(f"\n{len(errors)} error(s) occurred.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    recording_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    # If the script is run from the data_new root, look for subdirectories
    if not any(
        os.path.exists(os.path.join(recording_dir, f))
        for f in ["gaze ps1.raw", "imu ps1.raw", "event.txt"]
    ):
        # Try to find recording subdirectories automatically
        subdirs = [
            os.path.join(recording_dir, name)
            for name in os.listdir(recording_dir)
            if os.path.isdir(os.path.join(recording_dir, name))
        ]
        recording_dirs = [
            sd for sd in subdirs
            if any(os.path.exists(os.path.join(sd, f)) for f in ["gaze ps1.raw", "imu ps1.raw"])
        ]
        if not recording_dirs:
            print(f"No Neon recording found in {recording_dir}")
            sys.exit(1)
        for rd in recording_dirs:
            convert_recording(rd)
    else:
        convert_recording(recording_dir)
