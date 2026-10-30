import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("urc_rock_detection")
    default_cfg = os.path.join(pkg, "config", "rock_detection.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_cfg),
        DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(
            package="urc_rock_detection",
            executable="rock_detection_node",
            name="rock_detection_node",
            output="screen",
            parameters=[
                LaunchConfiguration("params_file"),
                {"image_topic": LaunchConfiguration("image_topic"),
                 "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                 "use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        ),
    ])
