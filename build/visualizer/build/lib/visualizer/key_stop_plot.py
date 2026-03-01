#!/usr/bin/env python3
"""
Press a key to stop recording and run plot_node.
Default key: 'p'
"""

import atexit
import select
import subprocess
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class KeyStopPlotNode(Node):
    def __init__(self):
        super().__init__("key_stop_plot_node")

        self.declare_parameter("record_filename", "a")
        self.declare_parameter("plot_file", "")
        self.declare_parameter("stop_key", "p")

        self._record_filename = self.get_parameter("record_filename").value
        self._plot_file = self.get_parameter("plot_file").value
        if not self._plot_file:
            self._plot_file = f"{self._record_filename}.json"
        self._stop_key = str(self.get_parameter("stop_key").value or "p")[:1].lower()

        self._stop_called = False
        self._terminal_ready = False
        self._stdin_fd = None
        self._stdin_old_state = None
        self._warned_no_tty = False

        self._stop_client = self.create_client(Trigger, "/visualizer/stop_recording")
        self._timer = self.create_timer(0.05, self._poll_keyboard)

        self._setup_terminal()
        self.get_logger().info(
            f"Keyboard control ready. Press '{self._stop_key}' to stop recording and run plot: {self._plot_file}"
        )

    def _setup_terminal(self):
        if not sys.stdin.isatty():
            return
        try:
            self._stdin_fd = sys.stdin.fileno()
            self._stdin_old_state = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
            self._terminal_ready = True
            atexit.register(self._restore_terminal)
        except Exception as exc:
            self.get_logger().warn(f"Failed to enable keyboard input mode: {exc}")

    def _restore_terminal(self):
        if self._terminal_ready and self._stdin_fd is not None and self._stdin_old_state is not None:
            try:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_state)
            except Exception:
                pass
            self._terminal_ready = False

    def _poll_keyboard(self):
        if self._stop_called:
            return

        if not self._terminal_ready:
            if not self._warned_no_tty:
                self.get_logger().warn("No interactive TTY detected. Keyboard trigger disabled.")
                self._warned_no_tty = True
            return

        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not readable:
                return
            ch = sys.stdin.read(1).lower()
        except Exception as exc:
            self.get_logger().warn(f"Keyboard read failed: {exc}")
            return

        if ch != self._stop_key:
            return

        self._stop_called = True
        self.get_logger().info("Stop key pressed. Calling /visualizer/stop_recording ...")
        self._call_stop_recording()

    def _call_stop_recording(self):
        if not self._stop_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Service /visualizer/stop_recording not available.")
            return
        future = self._stop_client.call_async(Trigger.Request())
        future.add_done_callback(self._on_stop_done)

    def _on_stop_done(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Stop recording success: {response.message}")
            else:
                self.get_logger().warn(f"Stop recording failed: {response.message}")
                return
        except Exception as exc:
            self.get_logger().error(f"Stop service call failed: {exc}")
            return

        cmd = ["ros2", "run", "visualizer", "plot_node", self._plot_file]
        self.get_logger().info(f"Running plot command: {' '.join(cmd)}")
        try:
            subprocess.Popen(cmd)
        except Exception as exc:
            self.get_logger().error(f"Failed to run plot_node: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = KeyStopPlotNode()
    try:
        rclpy.spin(node)
    finally:
        node._restore_terminal()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
