#!/usr/bin/env python3
"""
Plot node: Load recorded JSON and plot path, velocity, torque, caster angles and summary table.
- Only existing data is plotted; empty series are skipped.
"""

import argparse
import json
import math
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data directory: src/visualizer/data in workspace (same as record)
try:
    from ament_index_python.packages import get_package_share_directory
    _share = get_package_share_directory('visualizer')
    _install_prefix = os.path.dirname(os.path.dirname(_share))
    _ws_root = os.path.dirname(os.path.dirname(_install_prefix))
    DEFAULT_DATA_DIR = os.path.join(_ws_root, 'src', 'visualizer', 'data')
except Exception:
    DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def load_record(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def time_align(ref_t: np.ndarray, ref_v: np.ndarray, query_t: np.ndarray):
    """Linear interpolation of ref at query_t; ref_t, ref_v are 1d."""
    if ref_t.size == 0 or query_t.size == 0:
        return np.full_like(query_t, np.nan)
    return np.interp(query_t, ref_t, ref_v)


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered moving-average with edge padding."""
    if window <= 1 or arr.size == 0:
        return arr.copy()
    w = min(window, arr.size)
    if w <= 1:
        return arr.copy()
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode='edge')
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(padded, kernel, mode='valid')


def maybe_rad_to_deg(values: np.ndarray) -> np.ndarray:
    """If magnitude looks like radians, convert to degrees."""
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return values.copy()
    # Heuristic: if most values are within roughly +/- 2*pi, treat as radians.
    if np.nanpercentile(np.abs(valid), 95) <= 6.5:
        return np.degrees(values)
    return values.copy()


def wrap_deg(values: np.ndarray) -> np.ndarray:
    """Wrap degree angles into [-180, 180)."""
    return (values + 180.0) % 360.0 - 180.0


def angular_error_deg(ref_deg: np.ndarray, meas_deg: np.ndarray) -> np.ndarray:
    """Smallest signed angular error (ref - meas) in degrees."""
    return wrap_deg(ref_deg - meas_deg)


def odom_to_map_transform(odom, amcl):
    if not odom or not amcl:
        return None
    to = np.array([p['t'] for p in odom])
    xo = np.array([p['x'] for p in odom])
    yo = np.array([p['y'] for p in odom])
    yaw_o = np.array([p['yaw'] for p in odom])
    ta = np.array([p['t'] for p in amcl])
    t0 = float(to[0])
    if t0 < ta.min() or t0 > ta.max():
        return None
    x_m0 = np.interp(t0, ta, np.array([p['x'] for p in amcl]))
    y_m0 = np.interp(t0, ta, np.array([p['y'] for p in amcl]))
    yaw_m0 = np.interp(t0, ta, np.array([p['yaw'] for p in amcl]))
    x_o0, y_o0, yaw_o0 = float(xo[0]), float(yo[0]), float(yaw_o[0])
    dtheta = yaw_m0 - yaw_o0
    c, s = math.cos(dtheta), math.sin(dtheta)
    tx = x_m0 - (c * x_o0 - s * y_o0)
    ty = y_m0 - (s * x_o0 + c * y_o0)
    return (c, s, tx, ty)


def apply_odom_to_map(odom, transform):
    if not transform or not odom:
        return None, None
    c, s, tx, ty = transform
    xo = np.array([p['x'] for p in odom])
    yo = np.array([p['y'] for p in odom])
    x_map = c * xo - s * yo + tx
    y_map = s * xo + c * yo + ty
    return x_map, y_map


def path_error_along_trajectory(x, y, path_x, path_y):
    if len(path_x) < 2 or len(x) == 0:
        return np.nan, np.nan
    errors = []
    for xi, yi in zip(x, y):
        d = np.sqrt((path_x - xi) ** 2 + (path_y - yi) ** 2)
        errors.append(np.min(d))
    errors = np.array(errors)
    xarr = np.array(x)
    yarr = np.array(y)
    dts = np.sqrt(np.diff(xarr)**2 + np.diff(yarr)**2) if len(x) > 1 else np.array([0.0])
    dt_approx = float(np.mean(dts)) if dts.size else 0.0
    cum = float(np.nansum(errors)) * (dt_approx if dt_approx > 0 else 1.0)
    return cum, float(np.nanmax(errors))


def main():
    parser = argparse.ArgumentParser(description='Plot recorded visualizer data')
    parser.add_argument('file', nargs='?', help='Record JSON file path (e.g. data/record_xxx.json)')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help='Data directory when listing')
    parser.add_argument('--list', action='store_true', help='List available records and exit')
    parser.add_argument(
        '--torque-filter-window',
        type=int,
        default=15,
        help='Moving-average window for torque smoothing (samples, >=1)',
    )
    args = parser.parse_args()

    if args.list:
        if not os.path.isdir(args.data_dir):
            print('Data dir not found:', args.data_dir)
            return 1
        for f in sorted(os.listdir(args.data_dir)):
            if f.endswith('.json'):
                print(os.path.join(args.data_dir, f))
        return 0

    filepath = args.file
    if not filepath:
        print('Usage: plot_node <path to record.json>', file=sys.stderr)
        if os.path.isdir(args.data_dir):
            print('Available:', [f for f in os.listdir(args.data_dir) if f.endswith('.json')], file=sys.stderr)
        return 1
    if not os.path.dirname(filepath) and not os.path.isfile(filepath):
        filepath = os.path.join(args.data_dir, filepath)
    if not os.path.isfile(filepath):
        print('Usage: plot_node <path to record.json>', file=sys.stderr)
        if os.path.isdir(args.data_dir):
            print('Available:', [f for f in os.listdir(args.data_dir) if f.endswith('.json')], file=sys.stderr)
        return 1

    data = load_record(filepath)
    t_start = data.get('t_start', 0.0)
    duration = data.get('duration', 0.0)

    odom = data.get('odom') or []
    amcl = data.get('amcl') or []
    cmd_vel = data.get('cmd_vel') or []
    joy_vel = data.get('joy_target_velocity') or []
    plan = data.get('fixed_global_plan')
    imu = data.get('imu') or []
    wheel = data.get('wheelmotor') or []
    torque_filter_window = max(1, int(args.torque_filter_window))

    caster_k = data.get('caster_angle_kinematic') or []
    caster_k_ss = data.get('caster_angle_kinematic_ss') or []
    caster_e = data.get('caster_angle_encoder') or []
    caster_n = data.get('caster_angle_network') or []

    def to_t(tt):
        return tt - t_start if t_start else tt

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # ---- 1. Path & reference ----
    ax1 = axes[0, 0]
    map_T_odom = odom_to_map_transform(odom, amcl) if odom and amcl else None
    # if odom:
    #     if map_T_odom is not None:
    #         xo_map, yo_map = apply_odom_to_map(odom, map_T_odom)
    #         ax1.plot(-yo_map, xo_map, 'b-', label='odom (map frame)', alpha=0.8)
    #     else:
    #         xo = np.array([p['x'] for p in odom])
    #         yo = np.array([p['y'] for p in odom])
    #         ax1.plot(-yo, xo, 'b-', label='odom (odom frame)', alpha=0.8)
    if amcl:
        xa = np.array([p['x'] for p in amcl])
        ya = np.array([p['y'] for p in amcl])
        ax1.plot(-ya, xa, 'g-', label='amcl', alpha=0.8)
    if plan and plan.get('poses'):
        xp = np.array([p['x'] for p in plan['poses']])
        yp = np.array([p['y'] for p in plan['poses']])
        ax1.plot(-yp, xp, 'k--', label='global_plan', alpha=0.9)
    ax1.set_xlabel('y (m) -> right positive')
    ax1.set_ylabel('x (m)')
    ax1.set_title('Path & Reference')
    ax1.legend(loc='best')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    # ---- 2a. Velocity tracking error: linear_x ----
    ax2a = axes[0, 1]
    if odom:
        to = np.array([to_t(p['t']) for p in odom])
        vo = np.array([p['linear_x'] for p in odom])
        if cmd_vel:
            tc = np.array([to_t(p['t']) for p in cmd_vel])
            vc = np.array([p['linear_x'] for p in cmd_vel])
            v_err_cmd = time_align(tc, vc, to) - vo
            ax2a.plot(to, v_err_cmd, 'r-', label='cmd_vel - odom', alpha=0.9)
        if joy_vel:
            tj = np.array([to_t(p['t']) for p in joy_vel])
            vj = np.array([p['linear_x'] for p in joy_vel])
            v_err_joy = time_align(tj, vj, to) - vo
            ax2a.plot(to, v_err_joy, 'm-', label='joy_target_velocity - odom', alpha=0.9)
    ax2a.axhline(0.0, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
    ax2a.set_xlabel('time (s)')
    ax2a.set_ylabel('linear_x error (m/s)')
    ax2a.set_title('Velocity Tracking Error: linear_x')
    ax2a.legend(loc='best', fontsize=8)
    ax2a.grid(True, alpha=0.3)

    # ---- 2b. Velocity tracking error: angular_z ----
    ax2b = axes[1, 0]
    if odom:
        to = np.array([to_t(p['t']) for p in odom])
        wo = np.array([p['angular_z'] for p in odom])
        if cmd_vel:
            tc = np.array([to_t(p['t']) for p in cmd_vel])
            wc = np.array([p['angular_z'] for p in cmd_vel])
            w_err_cmd = time_align(tc, wc, to) - wo
            ax2b.plot(to, w_err_cmd, 'r-', label='cmd_vel - odom', alpha=0.9)
        if joy_vel:
            tj = np.array([to_t(p['t']) for p in joy_vel])
            wj = np.array([p['angular_z'] for p in joy_vel])
            w_err_joy = time_align(tj, wj, to) - wo
            ax2b.plot(to, w_err_joy, 'm-', label='joy_target_velocity - odom', alpha=0.9)
    ax2b.axhline(0.0, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
    ax2b.set_xlabel('time (s)')
    ax2b.set_ylabel('angular_z error (rad/s)')
    ax2b.set_title('Velocity Tracking Error: angular_z')
    ax2b.legend(loc='best', fontsize=8)
    ax2b.grid(True, alpha=0.3)

    # ---- 3. Torque (Wheel Motors) ----
    ax3 = axes[1, 1]
    if wheel:
        tw = np.array([to_t(p['t']) for p in wheel])
        # Use torque_left/right if available, fallback to current1/2
        t_l = np.array([p.get('torque_left', p.get('current1', 0)) for p in wheel])
        t_r = np.array([p.get('torque_right', p.get('current2', 0)) for p in wheel])
        t_l_f = moving_average(t_l, torque_filter_window)
        t_r_f = moving_average(t_r, torque_filter_window)
        ax3.plot(tw, t_l_f, 'b-', linewidth=2.0, label=f'Torque Left (MA{torque_filter_window})')
        ax3.plot(tw, t_r_f, 'r-', linewidth=2.0, label=f'Torque Right (MA{torque_filter_window})')
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('Torque (Nm)')
    ax3.set_title('Wheel Motor Torque')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # ---- 4 & 5. Caster Angle: Left and Right ----
    ax4 = axes[2, 0] # Left
    ax5 = axes[2, 1] # Right (기존에 가려져 있던 플롯 활성화)

    # GT 각도 그리기
    if caster_e:
        te = np.array([to_t(p['t']) for p in caster_e])
        ve_l = wrap_deg(np.array([p.get('left_deg', p.get('value', np.nan)) for p in caster_e], dtype=float))
        ve_r = wrap_deg(np.array([p.get('right_deg', np.nan) for p in caster_e], dtype=float))
        ax4.plot(te, ve_l, 'b-', label='GT (Left)', alpha=0.9)
        if not np.isnan(ve_r).all():
            ax5.plot(te, ve_r, 'b-', label='GT (Right)', alpha=0.9)

    # Kinematic 그리기 (보통 한쪽 기준으로만 나옴)
    if caster_k:
        tk = np.array([to_t(p['t']) for p in caster_k])
        vk = np.array([p['value'] for p in caster_k], dtype=float)
        vk = wrap_deg(maybe_rad_to_deg(vk))
        ax4.plot(tk, vk, 'g-', label='Kinematic Est.', alpha=0.9)
    if caster_k_ss:
        tk_ss = np.array([to_t(p['t']) for p in caster_k_ss])
        vk_ss = np.array([p['value'] for p in caster_k_ss], dtype=float)
        vk_ss = wrap_deg(maybe_rad_to_deg(vk_ss))
        ax4.plot(tk_ss, vk_ss, 'g--', label='Kinematic SS', alpha=0.6)

    # Network(학습 모델) 각도 그리기
    if caster_n:
        tn = np.array([to_t(p['t']) for p in caster_n])
        vn_l = np.array([p.get('left', p.get('value', np.nan)) for p in caster_n], dtype=float)
        vn_r = np.array([p.get('right', np.nan) for p in caster_n], dtype=float)
        vn_l = wrap_deg(maybe_rad_to_deg(vn_l))
        vn_r = wrap_deg(maybe_rad_to_deg(vn_r))
        ax4.plot(tn, vn_l, 'r-', label='Network (Left)', alpha=0.8)
        if not np.isnan(vn_r).all():
            ax5.plot(tn, vn_r, 'r-', label='Network (Right)', alpha=0.8)

    # Left 축 설정
    ax4.set_xlabel('time (s)')
    ax4.set_ylabel('Angle (deg)')
    ax4.set_title('Caster Angle (Left Front)')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    # Right 축 설정
    ax5.set_xlabel('time (s)')
    ax5.set_ylabel('Angle (deg)')
    ax5.set_title('Caster Angle (Right Front)')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save to workspace data dir
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    out_plot = os.path.join(args.data_dir, base_name + '_plot.png')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        plt.savefig(out_plot, dpi=150)
    plt.close()
    print('Saved plot:', out_plot)

    # ---- Summary table ----
    results = []

    # 1. Path error
    if odom and plan and plan.get('poses'):
        xo = np.array([p['x'] for p in odom])
        yo = np.array([p['y'] for p in odom])
        px = np.array([p['x'] for p in plan['poses']])
        py = np.array([p['y'] for p in plan['poses']])
        cum_err, max_err = path_error_along_trajectory(xo, yo, px, py)
        results.append(('Path error (odom vs global_plan)', f'cum: {cum_err:.4f}', f'max: {max_err:.4f} m'))
    else:
        results.append(('Path error', 'no data', ''))

    # 2. cmd_vel vs odom velocity error
    if cmd_vel and odom:
        to = np.array([p['t'] for p in odom])
        vo = np.array([p['linear_x'] for p in odom])
        wo = np.array([p['angular_z'] for p in odom])
        tc = np.array([p['t'] for p in cmd_vel])
        vc = np.array([p['linear_x'] for p in cmd_vel])
        wc = np.array([p['angular_z'] for p in cmd_vel])
        vc_interp = time_align(tc, vc, to)
        wc_interp = time_align(tc, wc, to)
        err_lin = np.abs(vc_interp - vo)
        err_ang = np.abs(wc_interp - wo)
        dt = np.diff(to)
        dt = np.concatenate((dt, [dt[-1]])) if len(dt) else np.array([0.0])
        cum_lin = float(np.nansum(err_lin * dt))
        cum_ang = float(np.nansum(err_ang * dt))
        results.append(('cmd_vel vs odom (linear_x)', f'cum err: {cum_lin:.4f}', f'max err: {np.nanmax(err_lin):.4f}'))
        results.append(('cmd_vel vs odom (angular_z)', f'cum err: {cum_ang:.4f}', f'max err: {np.nanmax(err_ang):.4f}'))
    else:
        results.append(('cmd_vel vs odom', 'no data', ''))

    # 3. Torque integral (|TL|+|TR|)*dt
    if wheel:
        tw = np.array([p['t'] for p in wheel])
        t_l = np.array([p.get('torque_left', p.get('current1', 0)) for p in wheel])
        t_r = np.array([p.get('torque_right', p.get('current2', 0)) for p in wheel])
        dt = np.diff(tw)
        dt = np.concatenate((dt, [dt[-1]])) if len(dt) else np.array([0.0])
        cum_torque = float(np.nansum((np.abs(t_l) + np.abs(t_r)) * dt))
        results.append(('Torque integral (|TL|+|TR|)*dt', f'{cum_torque:.4f}', ''))
    else:
        results.append(('Torque integral', 'no data', ''))

    # 4. Caster Errors (Left & Right Network vs GT)
    if caster_e and caster_n:
        te = np.array([p['t'] for p in caster_e])
        ve_l = wrap_deg(np.array([p.get('left_deg', p.get('value', np.nan)) for p in caster_e], dtype=float))
        ve_r = wrap_deg(np.array([p.get('right_deg', np.nan) for p in caster_e], dtype=float))
        
        tn = np.array([p['t'] for p in caster_n])
        vn_l = np.array([p.get('left', p.get('value', np.nan)) for p in caster_n], dtype=float)
        vn_r = np.array([p.get('right', np.nan) for p in caster_n], dtype=float)
        vn_l = wrap_deg(maybe_rad_to_deg(vn_l))
        vn_r = wrap_deg(maybe_rad_to_deg(vn_r))

        dt = np.diff(te)
        dt = np.concatenate((dt, [dt[-1]])) if len(dt) else np.array([0.0])

        # Left Error
        vn_l_interp = time_align(tn, vn_l, te)
        err_l = np.abs(angular_error_deg(ve_l, vn_l_interp))
        results.append(('Caster Left Error (Network vs GT)', f'cum err: {float(np.nansum(err_l * dt)):.4f}', f'max err: {float(np.nanmax(err_l)):.4f} deg'))

        # Right Error
        if not np.isnan(ve_r).all() and not np.isnan(vn_r).all():
            vn_r_interp = time_align(tn, vn_r, te)
            err_r = np.abs(angular_error_deg(ve_r, vn_r_interp))
            results.append(('Caster Right Error (Network vs GT)', f'cum err: {float(np.nansum(err_r * dt)):.4f}', f'max err: {float(np.nanmax(err_r)):.4f} deg'))
    else:
        results.append(('Caster angle comparison', 'no data', ''))

    # 5. Total duration
    results.append(('Total duration (s)', f'{duration:.2f}', ''))

    # ---- Summary table figure ----
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    out_summary = os.path.join(args.data_dir, base_name + '_summary.png')

    fig_table = plt.figure(figsize=(10, max(4, len(results) * 0.35)))
    ax_table = fig_table.add_subplot(111)
    ax_table.axis('off')

    table_data = [['Item', 'Cumulative/Value', 'Max/Note']]
    summary_title = 'Summary'
    for row in results:
        table_data.append([str(row[0]), str(row[1]), str(row[2])])

    table = ax_table.table(
        cellText=table_data,
        loc='center',
        cellLoc='left',
        colWidths=[0.45, 0.28, 0.27],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    ax_table.set_title(summary_title, fontsize=14, pad=20)
    plt.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fig_table.savefig(out_summary, dpi=120, bbox_inches='tight')
    plt.close(fig_table)
    print('Saved summary:', out_summary)

    # Open summary image with default viewer (Linux)
    try:
        import subprocess
        import shutil
        if shutil.which('xdg-open'):
            subprocess.Popen(['xdg-open', out_summary], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif shutil.which('see'):
            subprocess.Popen(['see', out_summary], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())