#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _start_record_action(context, *args, **kwargs):
    filename = LaunchConfiguration("filename").perform(context).strip() or "a"
    auto_start = _as_bool(LaunchConfiguration("auto_start_recording").perform(context))
    start_delay = float(LaunchConfiguration("start_delay").perform(context))
    if not auto_start:
        return []

    start_cmd = (
        "ros2 service call /visualizer/start_recording "
        f"amr_msgs/srv/StartRecord \"{{filename: '{filename}'}}\""
    )
    return [
        TimerAction(
            period=max(0.0, start_delay),
            actions=[ExecuteProcess(cmd=["bash", "-lc", start_cmd], output="screen")],
        )
    ]


def generate_launch_description():
    filename = LaunchConfiguration("filename")
    plot_file = LaunchConfiguration("plot_file")
    stop_key = LaunchConfiguration("stop_key")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "filename",
                default_value="a",
                description="Record filename without extension",
            ),
            DeclareLaunchArgument(
                "plot_file",
                default_value="",
                description="Plot target json file path. Empty -> <filename>.json",
            ),
            DeclareLaunchArgument(
                "stop_key",
                default_value="p",
                description="Keyboard key to trigger stop_recording and plot",
            ),
            DeclareLaunchArgument(
                "auto_start_recording",
                default_value="true",
                description="Automatically call /visualizer/start_recording",
            ),
            DeclareLaunchArgument(
                "start_delay",
                default_value="2.0",
                description="Seconds to wait before start_recording service call",
            ),
            Node(
                package="visualizer",
                executable="record_node",
                name="record_node",
                output="screen",
            ),
            Node(
                package="visualizer",
                executable="caster_kinematic_estimator_node",
                name="caster_kinematic_estimator_node",
                output="screen",
            ),
            Node(
                package="visualizer",
                executable="key_stop_plot_node",
                name="key_stop_plot_node",
                output="screen",
                emulate_tty=True,
                # [여기가 추가된 핵심 코드입니다!] 이 노드만 새 터미널 창을 띄워서 실행합니다.
                prefix=['gnome-terminal -- '],
                parameters=[
                    {
                        "record_filename": filename,
                        "plot_file": plot_file,
                        "stop_key": stop_key,
                    }
                ],
            ),
            OpaqueFunction(function=_start_record_action),
        ]
    )