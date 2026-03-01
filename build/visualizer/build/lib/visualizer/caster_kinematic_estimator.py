#!/usr/bin/env python3
"""
Caster wheel kinematic angle estimation node.
Subscribe to /odometry/filtered (v, w), compute left-front caster steady-state and MPC state 6 (phiL)
integration estimate, publish each. Record node subscribes to these topics.
"""

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

# Same geometry/dynamics as mpc_caster solver3_caster (iLQRsolver_caster.hpp)
DELTA_X_CW = 0.925
DELTA_Y_CW = 0.245
L_TR = 0.05  # l_tr


def wrap_to_pi(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    if not math.isfinite(angle):
        return 0.0
    return math.remainder(angle, 2.0 * math.pi)


def kinematic_caster_left_front_ss(v: float, w: float) -> float:
    """Left-front caster kinematic steady-state angle [rad], [-pi, pi], from v, w (rad/s)."""
    if not (math.isfinite(v) and math.isfinite(w)):
        return 0.0
    eps = 1e-6
    if abs(v) < eps and abs(w) < eps:
        return 0.0
    phi_ss_L = math.atan2(w * DELTA_X_CW, v - w * DELTA_Y_CW)
    return wrap_to_pi(phi_ss_L)


def phiL_dot(phiL: float, v: float, w: float) -> float:
    """MPC state 6 (phiL) continuous dynamics: d(phiL)/dt (iLQRsolver_caster dynamics_continuous)."""
    return -(1.0 / L_TR) * (
        (v - w * DELTA_Y_CW) * math.sin(phiL) - (w * DELTA_X_CW) * math.cos(phiL)
    )


class CasterKinematicEstimatorNode(Node):
    def __init__(self):
        super().__init__('caster_kinematic_estimator')
        self._sub_odom = self.create_subscription(
            Odometry, '/odometry/filtered', self._cb_odom, 10)
        self._pub_kinematic = self.create_publisher(
            Float64, 'visualizer/debug/caster_kinematic_est', 10)
        self._pub_kinematic_ss = self.create_publisher(
            Float64, 'visualizer/debug/caster_kinematic_ss', 10)

        self._phiL_est: Optional[float] = None
        self._t_odom_prev: float = 0.0

        self.get_logger().info(
            'Caster kinematic estimator ready. Publishing caster_kinematic_est, caster_kinematic_ss.')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _cb_odom(self, msg: Odometry):
        t = self._now()
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z

        ss_val = wrap_to_pi(kinematic_caster_left_front_ss(v, w))

        if self._phiL_est is None:
            self._phiL_est = ss_val
        else:
            dt = t - self._t_odom_prev
            if dt > 0.0:
                self._phiL_est = self._phiL_est + dt * phiL_dot(self._phiL_est, v, w)
                self._phiL_est = wrap_to_pi(self._phiL_est)
        self._t_odom_prev = t

        # Publish in degrees; sign flipped so left is positive
        msg_k = Float64()
        msg_k.data = math.degrees(self._phiL_est)
        self._pub_kinematic.publish(msg_k)
        msg_ss = Float64()
        msg_ss.data = math.degrees(ss_val)
        self._pub_kinematic_ss.publish(msg_ss)


def main(args=None):
    rclpy.init(args=args)
    node = CasterKinematicEstimatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
