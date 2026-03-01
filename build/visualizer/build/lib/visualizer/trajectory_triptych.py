#!/usr/bin/env python3
"""
Plot three trajectories side-by-side from recorded JSON files.

- Takes exactly 3 JSON files.
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
        nargs=3,
        help="Exactly 3 JSON files. If relative name is given, data dir is searched.",
    )
    parser.add_argument(
        "--rotate-deg",
        type=float,
        default=0.0,
        help="Rotation angle in degrees (CCW positive).",
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

    trajs: List[Tuple[np.ndarray, np.ndarray, str, str]] = []
    try:
        for p in paths:
            data = _load_json(p)
            x, y, src_name = _extract_xy(data, args.source)
            xr, yr = _rotate_xy(x, y, args.rotate_deg)
            label = os.path.splitext(os.path.basename(p))[0]
            trajs.append((xr, yr, src_name, label))
    except Exception as exc:
        print(f"Load/parse error: {exc}", file=sys.stderr)
        return 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    y_min = min(float(np.nanmin(t[1])) for t in trajs)
    y_max = max(float(np.nanmax(t[1])) for t in trajs)
    y_pad = max(0.1, 0.05 * (y_max - y_min if y_max > y_min else 1.0))

    for i, (ax, (x, y, src_name, label)) in enumerate(zip(axes, trajs)):
        ax.plot(x, y, linewidth=2.0, color="#1f77b4")
        ax.set_title(f"{label} ({src_name})", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        if i == 0:
            ax.set_ylabel("y (m)")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    fig.suptitle(f"Trajectories (rotation: {args.rotate_deg:.1f} deg)", fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.12, wspace=0.03)

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
