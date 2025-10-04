from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Include another launch file
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [FindPackageShare("zed_wrapper"), "launch", "zed_camera.launch.py"]
                )
            ]
        ),
        launch_arguments={
            "camera_model": "zed",
        }.items(),
    )

    imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [FindPackageShare("razor"), "launch", "razor_pub.launch.py"]
                )
            ]
        ),
    )

    # Node 1
    radar_stream = Node(
        package="radar_stream", executable="stream", name="stream", output="screen"
    )

    # Node 2
    # radar_process = Node(
    #     package="radar_stream", executable="process", name="process", output="screen"
    # )

    return LaunchDescription(
        [
            zed_launch,
            imu_launch,
            radar_stream,
            # radar_process,
        ]
    )
