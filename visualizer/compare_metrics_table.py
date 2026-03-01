#!/usr/bin/env python3
"""
Compare trajectory/velocity/caster RMSE metrics from 9 record JSON files.

Input order (exact):
  diff_M, diff_S, diff_C,
  conc_M, conc_S, conc_C,
  prop_M, prop_S, prop_C
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Tuple

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


def rmse(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(valid ** 2)))


def time_align(ref_t: np.ndarray, ref_v: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    if ref_t.size == 0 or query_t.size == 0:
        return np.full_like(query_t, np.nan)
    return np.interp(query_t, ref_t, ref_v)


def maybe_rad_to_deg(values: np.ndarray) -> np.ndarray:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return values.copy()
    if np.nanpercentile(np.abs(valid), 95) <= 6.5:
        return np.degrees(values)
    return values.copy()


def wrap_deg(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def angular_error_deg(ref_deg: np.ndarray, meas_deg: np.ndarray) -> np.ndarray:
    return wrap_deg(ref_deg - meas_deg)


def extract_traj_xy(data: dict) -> Tuple[np.ndarray, np.ndarray]:
    # For ref comparison, AMCL is preferred (map frame), fallback to odom.
    traj = data.get("amcl") or data.get("odom") or []
    if not traj:
        return np.array([]), np.array([])
    return (
        np.array([p["x"] for p in traj], dtype=float),
        np.array([p["y"] for p in traj], dtype=float),
    )


def extract_ref_xy(data: dict) -> Tuple[np.ndarray, np.ndarray]:
    plan = data.get("fixed_global_plan")
    if not plan:
        return np.array([]), np.array([])
    poses = plan.get("poses") or []
    if not poses:
        return np.array([]), np.array([])
    return (
        np.array([p["x"] for p in poses], dtype=float),
        np.array([p["y"] for p in poses], dtype=float),
    )


def path_ref_rmse(data: dict) -> float:
    x, y = extract_traj_xy(data)
    px, py = extract_ref_xy(data)
    if x.size == 0 or px.size == 0:
        return float("nan")
    d_min = []
    for xi, yi in zip(x, y):
        d = np.sqrt((px - xi) ** 2 + (py - yi) ** 2)
        d_min.append(float(np.min(d)))
    return rmse(np.array(d_min, dtype=float))


def vel_rmse(data: dict) -> Tuple[float, float]:
    odom = data.get("odom") or []
    cmd = data.get("cmd_vel") or []
    joy = data.get("joy_target_velocity") or []
    target = cmd if cmd else joy
    if not odom or not target:
        return float("nan"), float("nan")

    to = np.array([p["t"] for p in odom], dtype=float)
    vo = np.array([p["linear_x"] for p in odom], dtype=float)
    wo = np.array([p["angular_z"] for p in odom], dtype=float)

    tt = np.array([p["t"] for p in target], dtype=float)
    vt = np.array([p["linear_x"] for p in target], dtype=float)
    wt = np.array([p["angular_z"] for p in target], dtype=float)

    vt_i = time_align(tt, vt, to)
    wt_i = time_align(tt, wt, to)

    return rmse(vt_i - vo), rmse(wt_i - wo)


def caster_rmse(data: dict, method: str) -> float:
    gt = data.get("caster_angle_encoder") or []
    if not gt:
        return float("nan")

    # Compare left-side estimate with left GT.
    gt_t = np.array([p["t"] for p in gt], dtype=float)
    gt_left = np.array([p.get("left_deg", p.get("value", np.nan)) for p in gt], dtype=float)
    gt_left = wrap_deg(gt_left)

    if method == "Proposed":
        est = data.get("caster_angle_network") or []
        if not est:
            return float("nan")
        est_t = np.array([p["t"] for p in est], dtype=float)
        est_v = np.array([p.get("left", p.get("value", np.nan)) for p in est], dtype=float)
    else:
        est = data.get("caster_angle_kinematic") or []
        if not est:
            return float("nan")
        est_t = np.array([p["t"] for p in est], dtype=float)
        est_v = np.array([p.get("value", np.nan) for p in est], dtype=float)

    est_v = wrap_deg(maybe_rad_to_deg(est_v))
    est_i = time_align(est_t, est_v, gt_t)
    err = angular_error_deg(gt_left, est_i)
    return rmse(err)


def torque_mean_abs_sum(data: dict) -> float:
    wheel = data.get("wheelmotor") or []
    if not wheel:
        return float("nan")
    tl = np.array([p.get("torque_left", p.get("current1", np.nan)) for p in wheel], dtype=float)
    tr = np.array([p.get("torque_right", p.get("current2", np.nan)) for p in wheel], dtype=float)
    s = np.abs(tl) + np.abs(tr)
    valid = s[np.isfinite(s)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def fmt(v: float) -> str:
    if not np.isfinite(v):
        return "N/A"
    return f"{v:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare RMSE metrics from 9 visualizer JSON files.")
    parser.add_argument(
        "files",
        nargs=9,
        help=(
            "9 json files in order: "
            "diff_M,S,C conc_M,S,C prop_M,S,C"
        ),
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Data directory for relative filenames.")
    parser.add_argument("--output-png", default="", help="Output table png path.")
    parser.add_argument("--output-csv", default="", help="Output csv path.")
    args = parser.parse_args()

    try:
        paths = [resolve_path(p, args.data_dir) for p in args.files]
    except Exception as exc:
        print(f"File resolve error: {exc}", file=sys.stderr)
        return 1

    method_names = ["Diff", "Concurrent", "Proposed"]
    scenario_names = ["M", "S", "C"]

    # Rows in scenario-major order: M(3 methods), S(3), C(3)
    rows: List[Dict[str, str]] = []
    for scenario_idx, scenario in enumerate(scenario_names):
        for method_idx, method in enumerate(method_names):
            file_idx = method_idx * 3 + scenario_idx
            data = load_record(paths[file_idx])

            path_rmse = path_ref_rmse(data)
            v_rmse, w_rmse = vel_rmse(data)
            c_rmse = caster_rmse(data, method)
            torque_mean = torque_mean_abs_sum(data)

            rows.append(
                {
                    "Scenario": scenario,
                    "Method": method,
                    "PathRMSE_m": fmt(path_rmse),
                    "VRMSE_mps": fmt(v_rmse),
                    "WRMSE_radps": fmt(w_rmse),
                    "CasterRMSE_deg": fmt(c_rmse),
                    "TorqueMeanAbsSum_Nm": fmt(torque_mean),
                }
            )

    # Print concise text table to stdout
    print("Scenario,Method,PathRMSE_m,VRMSE_mps,WRMSE_radps,CasterRMSE_deg,TorqueMeanAbsSum_Nm")
    for r in rows:
        print(
            f"{r['Scenario']},{r['Method']},{r['PathRMSE_m']},"
            f"{r['VRMSE_mps']},{r['WRMSE_radps']},{r['CasterRMSE_deg']},{r['TorqueMeanAbsSum_Nm']}"
        )

    base0 = os.path.splitext(os.path.basename(paths[0]))[0]
    out_png = args.output_png or os.path.join(args.data_dir, f"{base0}_rmse_table.png")
    out_csv = args.output_csv or os.path.join(args.data_dir, f"{base0}_rmse_table.csv")

    # Save CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Scenario",
                "Method",
                "PathRMSE_m",
                "VRMSE_mps",
                "WRMSE_radps",
                "CasterRMSE_deg",
                "TorqueMeanAbsSum_Nm",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Save PNG table
    cell_text = [
        [
            r["Scenario"],
            r["Method"],
            r["PathRMSE_m"],
            r["VRMSE_mps"],
            r["WRMSE_radps"],
            r["CasterRMSE_deg"],
            r["TorqueMeanAbsSum_Nm"],
        ]
        for r in rows
    ]
    col_labels = [
        "Scenario",
        "Method",
        "Path RMSE [m]",
        "v RMSE [m/s]",
        "w RMSE [rad/s]",
        "Caster RMSE [deg]",
        "Torque mean [|TL|+|TR|] [Nm]",
    ]

    fig = plt.figure(figsize=(15, 4.8))
    ax = fig.add_subplot(111)
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.09, 0.12, 0.13, 0.12, 0.13, 0.14, 0.27],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved PNG: {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
