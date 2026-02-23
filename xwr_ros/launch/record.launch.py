"""Launch xwr_ros radar stream node with composable recorder."""

import sys

from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    TextSubstitution,
)
from launch_ros.actions import ComposableNodeContainer
from launch_ros.substitutions import FindPackageShare

sys.path.insert(0, get_package_share_directory("ros_rec") + "/launch")
from composable import make_recorder_nodes  # type: ignore

from launch import LaunchDescription


def launch_setup(context):
    """Create a ComposableNodeContainer with recorder nodes for xwr sensor."""
    rec_nodes = make_recorder_nodes(keys=["xwr"])

    container = ComposableNodeContainer(
        name="xwr_recorder_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=rec_nodes,
        output="screen",
    )
    return [container]


def generate_launch_description():
    """Launch xwr_ros stream node and composable recorder for xwr sensor."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=TextSubstitution(text="grt-i1"),
                description="Config file name under xwr_ros/config without .yaml",
            ),
            DeclareLaunchArgument(
                "bag_cfg",
                default_value=TextSubstitution(text="ttstar.yaml"),
                description="Bag config file name under ros_rec/bag_cfg/",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("xwr_ros"),
                            "launch",
                            "radar.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "config": LaunchConfiguration("config")
                }.items(),
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
