"""Launch file for xwr_ros radar stream node."""
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    """Generate launch description."""
    cfg_arg = DeclareLaunchArgument(
        "config",
        default_value=TextSubstitution(text="config"),
        description="configuration file name under xwr_ros/config without .yaml",
    )

    radar_stream = Node(
        package="xwr_ros",
        executable="stream",
        name="stream",
        output="screen",
        parameters=[{"config": LaunchConfiguration("config")}],
    )

    return LaunchDescription(
        [
            cfg_arg,
            radar_stream,
        ]
    )
