#!/usr/bin/env python3
"""
Plot trajectories side-by-side from recorded JSON files.

Current mode:
- Takes exactly 9 JSON files.
- Input order:
  diff-M,S,C, Concurrent-M,S,C, Proposed-M,S,C
- Builds 3 panels (M, S, C), each overlaying 3 trajectories.
- Uses only trajectory fields (amcl preferred, odom fallback by default).
- Rotates coordinates by N degrees to align scene orientation.
- Single shared y-axis on the left, separate x-axis per subplot.
"""

import argparse
import json
import math
import os
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Default data directory (same pattern as existing plot_node)
try:
    from ament_index_python.packages import get_package_share_directory
    _share = get_package_share_directory("visualizer")
    _install_prefix = os.path.dirname(os.path.dirname(_share))
    _ws_root = os.path.dirname(os.path.dirname(_install_prefix))
    DEFAULT_DATA_DIR = os.path.join(_ws_root, "src", "visualizer", "data")
except Exception:
    DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _resolve_path(path_like: str, data_dir: str) -> str:
    if os.path.isfile(path_like):
        return path_like
    if not os.path.dirname(path_like):
        candidate = os.path.join(data_dir, path_like)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(path_like)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_xy(data: dict, source: str) -> Tuple[np.ndarray, np.ndarray, str]:
    amcl = data.get("amcl") or []
    odom = data.get("odom") or []

    selected = None
    selected_name = ""
    if source == "amcl":
        selected = amcl
        selected_name = "amcl"
    elif source == "odom":
        selected = odom
        selected_name = "odom"
    else:
        if amcl:
            selected = amcl
            selected_name = "amcl"
        else:
            selected = odom
            selected_name = "odom"

    if not selected:
        raise ValueError("Trajectory is empty (amcl/odom both empty or selected source empty).")

    x = np.array([p["x"] for p in selected], dtype=float)
    y = np.array([p["y"] for p in selected], dtype=float)
    return x, y, selected_name


def _extract_ref_xy(data: dict) -> Tuple[np.ndarray, np.ndarray]:
    plan = data.get("fixed_global_plan")
    if not plan:
        return np.array([], dtype=float), np.array([], dtype=float)
    poses = plan.get("poses") or []
    if not poses:
        return np.array([], dtype=float), np.array([], dtype=float)
    px = np.array([p["x"] for p in poses], dtype=float)
    py = np.array([p["y"] for p in poses], dtype=float)
    return px, py


def _rotate_xy(x: np.ndarray, y: np.ndarray, deg: float) -> Tuple[np.ndarray, np.ndarray]:
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    xr = c * x - s * y
    yr = s * x + c * y
    return xr, yr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Side-by-side trajectory plot for 3 JSON files (with rotation)."
    )
    parser.add_argument(
        "files",
        nargs=9,
        help=(
            "Exactly 9 JSON files in this order: "
            "diff-M,S,C concurrent-M,S,C proposed-M,S,C. "
            "If relative names are given, data dir is searched."
        ),
    )
    parser.add_argument(
        "--rotate-deg",
        type=float,
        default=0.0,
        help="Global rotation angle in degrees (CCW positive).",
    )
    parser.add_argument(
        "--rotate-deg-m",
        type=float,
        default=None,
        help="Panel M rotation angle override (degrees).",
    )
    parser.add_argument(
        "--rotate-deg-s",
        type=float,
        default=None,
        help="Panel S rotation angle override (degrees).",
    )
    parser.add_argument(
        "--rotate-deg-c",
        type=float,
        default=None,
        help="Panel C rotation angle override (degrees).",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "amcl", "odom"],
        default="auto",
        help="Trajectory source in JSON.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Data directory for relative filenames.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output PNG path. Default: <file1>_triptych.png in data-dir.",
    )
    args = parser.parse_args()

    try:
        paths = [_resolve_path(p, args.data_dir) for p in args.files]
    except Exception as exc:
        print(f"File resolve error: {exc}", file=sys.stderr)
        return 1

    panel_names = ["M", "S", "C"]
    method_names = ["Diff", "Concurrent", "Proposed"]
    method_colors = {
        "Diff": "#7f7f7f",
        "Concurrent": "#ff7f0e",
        "Proposed": "#1f77b4",
    }
    legend_order = ["Ref", "Diff", "Concurrent", "Proposed"]
    panel_rotate_deg = [
        args.rotate_deg if args.rotate_deg_m is None else float(args.rotate_deg_m),
        args.rotate_deg if args.rotate_deg_s is None else float(args.rotate_deg_s),
        args.rotate_deg if args.rotate_deg_c is None else float(args.rotate_deg_c),
    ]

    # panel_data[panel_idx] = list of (x, y, ref_x, ref_y, method_label, color)
    panel_data: List[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]]] = [
        [], [], []
    ]
    try:
        for file_idx, p in enumerate(paths):
            data = _load_json(p)
            x, y, src_name = _extract_xy(data, args.source)
            px, py = _extract_ref_xy(data)
            method_idx = file_idx // 3
            panel_idx = file_idx % 3
            rotate_deg = panel_rotate_deg[panel_idx]
            xr, yr = _rotate_xy(x, y, rotate_deg)
            pxr, pyr = _rotate_xy(px, py, rotate_deg)
            method = method_names[method_idx]
            panel = panel_names[panel_idx]
            panel_data[panel_idx].append((xr, yr, pxr, pyr, method, method_colors[method]))
    except Exception as exc:
        print(f"Load/parse error: {exc}", file=sys.stderr)
        return 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    legend_handles = {}
    legend_handles["Ref"] = Line2D([0], [0], color="k", linestyle="--", linewidth=1.5)

    for i, (ax, panel_curves) in enumerate(zip(axes, panel_data)):
        # Draw one ref per panel (from first curve that has ref)
        for (x, y, px, py, legend_label, color) in panel_curves:
            if px.size > 0 and py.size > 0:
                ax.plot(px, py, "k--", linewidth=1.5, alpha=0.9, label="_nolegend_")
                break

        for (x, y, px, py, method_label, color) in panel_curves:
            h, = ax.plot(x, y, linewidth=2.0, color=color, label=method_label)
            legend_handles[method_label] = h

        ax.grid(True, alpha=0.3)
        # Keep y-axis truly shared across panels.
        ax.set_aspect("auto")
        ax.set_xlim(-1.5, 2.0)
        ax.set_ylim(-1.0, 5.0)
        # Keep only axis lines (no full rectangular box around each subplot).
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.spines["left"].set_visible(True)
        if i == 2:
            ordered = [lbl for lbl in legend_order if lbl in legend_handles]
            handles = [legend_handles[lbl] for lbl in ordered]
            ax.legend(handles, ordered, loc="upper right", fontsize=8, frameon=False)

        if i == 0:
            ax.set_ylabel("y (m)")
        else:
            ax.spines["left"].set_visible(False)
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        # Keep x-axis label only at S panel (middle).
        if i == 1:
            ax.set_xlabel("x (m)", labelpad=4)
        else:
            ax.set_xlabel("")
        ax.tick_params(axis="x", pad=5)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.12, wspace=0.08)

    if args.output:
        out_path = args.output
    else:
        base0 = os.path.splitext(os.path.basename(paths[0]))[0]
        out_path = os.path.join(args.data_dir, f"{base0}_triptych.png")

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
