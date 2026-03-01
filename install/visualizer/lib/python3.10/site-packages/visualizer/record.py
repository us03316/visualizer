#!/usr/bin/env python3
"""
Record node: Start/stop recording via service, save JSON to data dir.
- StartRecord.srv (filename) -> start recording
- Trigger (stop_recording) -> stop and save
"""

import json
import math
import os
import tempfile
from typing import Optional, List, Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import Imu, JointState  # JointState 추가
from std_msgs.msg import Float64, Float32, Float64MultiArray  # Float64MultiArray 추가
from std_srvs.srv import Trigger

from amr_msgs.msg import WheelMotor
from amr_msgs.srv import StartRecord


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class RecordNode(Node):
    def __init__(self):
        super().__init__('record_node')

        # Data dir: workspace src/visualizer/data (not install)
        try:
            share = get_package_share_directory('visualizer')
            # share = .../ros2_ws/install/visualizer/share/visualizer
            # install_prefix = .../ros2_ws/install/visualizer
            install_prefix = os.path.dirname(os.path.dirname(share))
            # ws_root = .../ros2_ws (parent of install)
            ws_root = os.path.dirname(os.path.dirname(install_prefix))
            self._data_dir = os.path.join(ws_root, 'src', 'visualizer', 'data')
        except Exception:
            # fallback: __file__ when run from source
            self._data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        os.makedirs(self._data_dir, exist_ok=True)

        self._recording = False
        self._filename: Optional[str] = None
        self._t_start: Optional[float] = None
        self._t_end: Optional[float] = None

        # buffers
        self._odom: List[dict] = []
        self._amcl: List[dict] = []
        self._cmd_vel: List[dict] = []
        self._joy_target_velocity: List[dict] = []
        self._global_plan: Optional[dict] = None
        self._last_global_plan: Optional[dict] = None
        self._imu: List[dict] = []
        self._wheelmotor: List[dict] = []
        self._caster_kinematic: List[dict] = []
        self._caster_kinematic_ss: List[dict] = []
        self._caster_encoder: List[dict] = []
        self._caster_network: List[dict] = []

        # Subscribers
        self._sub_odom = self.create_subscription(
            Odometry, '/odometry/filtered', self._cb_odom, 10)
        self._sub_amcl = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._cb_amcl, 10)
        self._sub_cmd_vel = self.create_subscription(
            Twist, '/target_velocity', self._cb_cmd_vel, 10)
        self._sub_joy_vel = self.create_subscription(
            Twist, '/joy_target_velocity', self._cb_joy_target_velocity, 10)
        # Fixed path: subscribe with TRANSIENT_LOCAL (receive even if publisher is latched)
        qos_plan_latched = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._sub_plan = self.create_subscription(
            Path, '/fixed_global_plan', self._cb_plan, qos_plan_latched)
        # Default QoS in case publisher is VOLATILE
        self._sub_plan_default = self.create_subscription(
            Path, '/fixed_global_plan', self._cb_plan, 10)
        self._sub_imu = self.create_subscription(
            Imu, '/imu/data', self._cb_imu, 10)
        self._sub_joint_states = self.create_subscription(
            JointState, '/joint_states', self._cb_joint_states, 10)  ## 토크 입력
        self._sub_caster_network = self.create_subscription(
            Float64MultiArray, '/dwl_output', self._cb_caster_network, 10)   ## 학습에서 나오는 캐스터 양쪽
        # Kinematic/steady-state from caster_kinematic_estimator
        self._sub_caster_kinematic = self.create_subscription(
            Float64, 'visualizer/debug/caster_kinematic_est', self._cb_caster_kinematic, 10)
        self._sub_caster_kinematic_ss = self.create_subscription(
            Float64, 'visualizer/debug/caster_kinematic_ss', self._cb_caster_kinematic_ss, 10)

        self._srv_start = self.create_service(
            StartRecord, 'visualizer/start_recording', self._srv_start_cb)
        self._srv_stop = self.create_service(
            Trigger, 'visualizer/stop_recording', self._srv_stop_cb)

        self.get_logger().info(
            f'Record node ready. data_dir={self._data_dir}. '
            'Call visualizer/start_recording (filename) then visualizer/stop_recording.')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _cb_odom(self, msg: Odometry):
        if not self._recording:
            return
        t = self._now()
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        tw = msg.twist.twist
        self._odom.append({
            't': t,
            'x': p.x, 'y': p.y, 'z': p.z,
            'qx': o.x, 'qy': o.y, 'qz': o.z, 'qw': o.w,
            'yaw': quat_to_yaw(o.x, o.y, o.z, o.w),
            'linear_x': tw.linear.x, 'linear_y': tw.linear.y, 'linear_z': tw.linear.z,
            'angular_x': tw.angular.x, 'angular_y': tw.angular.y, 'angular_z': tw.angular.z,
        })

    def _cb_amcl(self, msg: PoseWithCovarianceStamped):
        if not self._recording:
            return
        t = self._now()
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._amcl.append({
            't': t,
            'x': p.x, 'y': p.y, 'z': p.z,
            'qx': o.x, 'qy': o.y, 'qz': o.z, 'qw': o.w,
            'yaw': quat_to_yaw(o.x, o.y, o.z, o.w),
        })

    def _cb_cmd_vel(self, msg: Twist):
        if not self._recording:
            return
        t = self._now()
        self._cmd_vel.append({
            't': t,
            'linear_x': msg.linear.x,
            'angular_z': msg.angular.z,
        })

    def _cb_joy_target_velocity(self, msg: Twist):
        if not self._recording:
            return
        t = self._now()
        self._joy_target_velocity.append({
            't': t,
            'linear_x': msg.linear.x,
            'angular_z': msg.angular.z,
        })

    def _cb_plan(self, msg: Path):
        # Keep last received fixed global path
        poses = []
        for pose_stamped in msg.poses:
            p = pose_stamped.pose.position
            poses.append({'x': p.x, 'y': p.y})
        self._last_global_plan = {'t': self._now(), 'poses': poses}
        if self._recording:
            self._global_plan = self._last_global_plan

    def _cb_imu(self, msg: Imu):
        if not self._recording:
            return
        t = self._now()
        # Imu: linear_acceleration, angular_velocity (no linear velocity in standard Imu)
        self._imu.append({
            't': t,
            'linear_accel_x': msg.linear_acceleration.x,
            'linear_accel_y': msg.linear_acceleration.y,
            'angular_vel_z': msg.angular_velocity.z,
        })
    # ==========================================================
    # [NEW] 토크 및 GT 각도 추출 (JointState)
    # ==========================================================
    def _cb_joint_states(self, msg: JointState):
        if not self._recording:
            return
        t = float(self._now())
        
        # 1. 토크 (effort 3, 4번째 값 = 인덱스 2, 3)
        if len(msg.effort) >= 4:
            self._wheelmotor.append({
                't': t,
                'torque_left': float(msg.effort[2]),
                'torque_right': float(msg.effort[3]),
                # plot_node.py 호환성을 위해 current1, current2 키 유지
                'current1': float(msg.effort[2]), 
                'current2': float(msg.effort[3]),
            })
            
        # 2. GT 조인트 각도 (position 1, 2번째 값 = 인덱스 0, 1) -> Radian to Degree
        if len(msg.position) >= 2:
            left_deg = float(-math.degrees(msg.position[0]))
            right_deg = float(-math.degrees(msg.position[1]))
            self._caster_encoder.append({
                't': t,
                'left_deg': left_deg,
                'right_deg': right_deg,
                'value': left_deg,
            })

    # ==========================================================
    # [NEW] 학습된 네트워크 캐스터 각도 (Float64MultiArray)
    # ==========================================================
    def _cb_caster_network(self, msg: Float64MultiArray):
        if not self._recording:
            return
        t = float(self._now())
        if len(msg.data) >= 2:
            self._caster_network.append({
                't': t,
                'left': float(-msg.data[0]),
                'right': float(-msg.data[1]),
                'value': float(-msg.data[0]),
            })

    def _cb_caster_encoder(self, msg):
        if not self._recording:
            return
        self._caster_encoder.append({'t': self._now(), 'value': float(msg.data)})

    def _cb_caster_kinematic(self, msg: Float64):
        if not self._recording:
            return
        self._caster_kinematic.append({'t': float(self._now()), 'value': float(msg.data)})

    def _cb_caster_kinematic_ss(self, msg: Float64):
        if not self._recording:
            return
        self._caster_kinematic_ss.append({'t': float(self._now()), 'value': float(msg.data)})

    def _srv_start_cb(self, req: StartRecord.Request, res: StartRecord.Response):
        if self._recording:
            res.success = False
            res.message = 'Already recording. Call stop_recording first.'
            return res
        self._filename = req.filename.strip() or f'record_{int(self._now() * 1000)}'
        self._t_start = self._now()
        self._t_end = None
        self._odom = []
        self._amcl = []
        self._cmd_vel = []
        self._joy_target_velocity = []
        self._global_plan = None
        self._imu = []
        self._wheelmotor = []
        self._caster_kinematic = []
        self._caster_kinematic_ss = []
        self._caster_encoder = []
        self._caster_network = []
        self._recording = True
        # Use last fixed plan if already published before start
        if self._last_global_plan is not None:
            self._global_plan = self._last_global_plan
        res.success = True
        res.message = f'Recording started: {self._filename}'
        self.get_logger().info(res.message)
        return res

    def _srv_stop_cb(self, req: Trigger.Request, res: Trigger.Response):
        if not self._recording:
            res.success = False
            res.message = 'Not recording.'
            return res
        self._recording = False
        self._t_end = self._now()
        out_path = os.path.join(self._data_dir, self._filename + '.json')
        tmp_fd = None
        tmp_path = None
        try:
            data = {
                't_start': self._t_start,
                't_end': self._t_end,
                'duration': self._t_end - self._t_start,
                'odom': self._odom,
                'amcl': self._amcl,
                'cmd_vel': self._cmd_vel,
                'joy_target_velocity': self._joy_target_velocity,
                'fixed_global_plan': self._global_plan,
                'imu': self._imu,
                'wheelmotor': self._wheelmotor,
                'caster_angle_kinematic': self._caster_kinematic,
                'caster_angle_kinematic_ss': self._caster_kinematic_ss,
                'caster_angle_encoder': self._caster_encoder,
                'caster_angle_network': self._caster_network,
            }
            tmp_fd, tmp_path = tempfile.mkstemp(prefix='.record_', suffix='.json', dir=self._data_dir)
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                tmp_fd = None
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, out_path)
            res.success = True
            res.message = f'Saved to {out_path}'
        except Exception as e:
            res.success = False
            res.message = str(e)
            try:
                if tmp_fd is not None:
                    os.close(tmp_fd)
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
        self.get_logger().info(res.message)
        return res


def main(args=None):
    rclpy.init(args=args)
    node = RecordNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
