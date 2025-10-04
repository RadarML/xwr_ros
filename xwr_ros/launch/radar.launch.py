from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, TextSubstitution


def generate_launch_description():
    cfg_arg = DeclareLaunchArgument(
        "config",
        default_value=TextSubstitution(text="config"),
        description="configuration file name without .yaml",
    )

    # Node 1
    radar_stream = Node(
        package="radar_stream",
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
