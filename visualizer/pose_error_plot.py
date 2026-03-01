#!/usr/bin/env python3
"""
Generate x/y/yaw error plots against fixed_global_plan.

Error definition (for each trajectory sample):
- Find nearest point on fixed_global_plan (discrete nearest waypoint).
- x_err = x_traj - x_ref_nearest
- y_err = y_traj - y_ref_nearest
- yaw_err = wrap_to_pi(yaw_traj - yaw_ref_tangent)
  where yaw_ref_tangent is heading of local path segment.
"""

import argparse
import json
import math
import os
import sys
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Default data directory (same pattern as existing plot scripts)
try:
    from ament_index_python.packages import get_package_share_directory
    _share = get_package_share_directory("visualizer")
    _install_prefix = os.path.dirname(os.path.dirname(_share))
    _ws_root = os.path.dirname(os.path.dirname(_install_prefix))
    DEFAULT_DATA_DIR = os.path.join(_ws_root, "src", "visualizer", "data")
except Exception:
    DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def resolve_path(path_like: str, data_dir: str) -> str:
    if os.path.isfile(path_like):
        return path_like
    if not os.path.dirname(path_like):
        candidate = os.path.join(data_dir, path_like)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(path_like)


def load_record(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def extract_traj(data: dict, source: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (t, x, y, yaw[rad]).
    """
    amcl = data.get("amcl") or []
    odom = data.get("odom") or []

    if source == "amcl":
        traj = amcl
        src_name = "amcl"
    elif source == "odom":
        traj = odom
        src_name = "odom"
    else:
        traj = amcl if amcl else odom
        src_name = "amcl" if amcl else "odom"

    if not traj:
        raise ValueError("No trajectory data found (amcl/odom empty).")

    t = np.array([p["t"] for p in traj], dtype=float)
    x = np.array([p["x"] for p in traj], dtype=float)
    y = np.array([p["y"] for p in traj], dtype=float)

    # Record data usually already has yaw, but keep quaternion fallback.
    if "yaw" in traj[0]:
        yaw = np.array([p.get("yaw", np.nan) for p in traj], dtype=float)
    else:
        yaw = np.array([
            quat_to_yaw(
                p.get("qx", 0.0),
                p.get("qy", 0.0),
                p.get("qz", 0.0),
                p.get("qw", 1.0),
            )
            for p in traj
        ], dtype=float)

    return t, x, y, yaw


def extract_plan(data: dict) -> Tuple[np.ndarray, np.ndarray]:
    plan = data.get("fixed_global_plan")
    if not plan:
        raise ValueError("fixed_global_plan is missing in record.")
    poses = plan.get("poses") or []
    if len(poses) < 2:
        raise ValueError("fixed_global_plan poses are missing or too short (<2).")
    px = np.array([p["x"] for p in poses], dtype=float)
    py = np.array([p["y"] for p in poses], dtype=float)
    return px, py


def path_tangent_yaw(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """
    Discrete tangent yaw at each path waypoint.
    """
    n = len(px)
    yaw = np.zeros(n, dtype=float)
    for i in range(n):
        if i == 0:
            dx = px[1] - px[0]
            dy = py[1] - py[0]
        elif i == n - 1:
            dx = px[n - 1] - px[n - 2]
            dy = py[n - 1] - py[n - 2]
        else:
            dx = px[i + 1] - px[i - 1]
            dy = py[i + 1] - py[i - 1]
        yaw[i] = math.atan2(dy, dx)
    return yaw


def nearest_path_index(x: np.ndarray, y: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """
    Returns nearest path index for each trajectory sample.
    """
    idx = np.zeros(len(x), dtype=int)
    for i, (xi, yi) in enumerate(zip(x, y)):
        d = (px - xi) ** 2 + (py - yi) ** 2
        idx[i] = int(np.argmin(d))
    return idx


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot x/y/yaw error against fixed_global_plan.")
    parser.add_argument("file", help="Record JSON file path.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Data directory for relative filename.")
    parser.add_argument(
        "--source",
        choices=["auto", "amcl", "odom"],
        default="auto",
        help="Trajectory source to use for error computation.",
    )
    parser.add_argument("--output", default="", help="Output PNG path.")
    args = parser.parse_args()

    try:
        filepath = resolve_path(args.file, args.data_dir)
        data = load_record(filepath)
        t, x, y, yaw = extract_traj(data, args.source)
        px, py = extract_plan(data)
    except Exception as exc:
        print(f"Load error: {exc}", file=sys.stderr)
        return 1

    pyaw = path_tangent_yaw(px, py)
    nn_idx = nearest_path_index(x, y, px, py)

    x_ref = px[nn_idx]
    y_ref = py[nn_idx]
    yaw_ref = pyaw[nn_idx]

    x_err = x - x_ref
    y_err = y - y_ref
    yaw_err = wrap_to_pi(yaw - yaw_ref)

    t0 = t[0] if t.size else 0.0
    ts = t - t0

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(ts, x_err, "b-", linewidth=1.5)
    axes[0].axhline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("x error [m]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ts, y_err, "g-", linewidth=1.5)
    axes[1].axhline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_ylabel("y error [m]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ts, np.degrees(yaw_err), "r-", linewidth=1.5)
    axes[2].axhline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[2].set_ylabel("yaw error [deg]")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True, alpha=0.3)

    rmse_x = float(np.sqrt(np.mean(x_err ** 2))) if x_err.size else float("nan")
    rmse_y = float(np.sqrt(np.mean(y_err ** 2))) if y_err.size else float("nan")
    rmse_yaw = float(np.sqrt(np.mean(np.degrees(yaw_err) ** 2))) if yaw_err.size else float("nan")
    fig.suptitle(
        f"x/y/yaw error vs plan (RMSE: x={rmse_x:.3f} m, y={rmse_y:.3f} m, yaw={rmse_yaw:.3f} deg)",
        fontsize=11,
    )
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = args.output or os.path.join(args.data_dir, f"{base}_pose_error.png")
    try:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Save error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
