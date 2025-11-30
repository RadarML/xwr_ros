"""Launch file for xwr_ros radar signal processing node."""

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    """Generate launch description."""
    cfg_arg = DeclareLaunchArgument(
        "config",
        default_value=TextSubstitution(text="config"),
        description="config file name under xwr_ros/config without .yaml",
    )

    dsp_arg = DeclareLaunchArgument(
        "dsp",
        default_value=TextSubstitution(text="AWR1843AOP"),
        description="DSP processing type (e.g., AWR1843AOP)",
    )

    gain_arg = DeclareLaunchArgument(
        "gain",
        default_value=TextSubstitution(text="2e-6"),
        description="Gain value for signal processing",
    )

    azimuth_fft_size_arg = DeclareLaunchArgument(
        "azimuth_fft_size",
        default_value=TextSubstitution(text="64"),
        description="FFT size for azimuth angle processing",
    )

    elevation_fft_size_arg = DeclareLaunchArgument(
        "elevation_fft_size",
        default_value=TextSubstitution(text="64"),
        description="FFT size for elevation angle processing",
    )

    azimuth_fov_arg = DeclareLaunchArgument(
        "azimuth_fov",
        default_value=TextSubstitution(text="60.0"),
        description="Azimuth field of view in degrees",
    )

    elevation_fov_arg = DeclareLaunchArgument(
        "elevation_fov",
        default_value=TextSubstitution(text="60.0"),
        description="Elevation field of view in degrees",
    )

    radar_process = Node(
        package="xwr_ros",
        executable="process",
        name="sig_process",
        output="screen",
        parameters=[
            {
                "config": LaunchConfiguration("config"),
                "dsp": LaunchConfiguration("dsp"),
                "gain": LaunchConfiguration("gain"),
                "azimuth_fft_size": LaunchConfiguration("azimuth_fft_size"),
                "elevation_fft_size": LaunchConfiguration("elevation_fft_size"),
                "azimuth_fov": LaunchConfiguration("azimuth_fov"),
                "elevation_fov": LaunchConfiguration("elevation_fov"),
            }
        ],
    )

    return LaunchDescription(
        [
            cfg_arg,
            dsp_arg,
            gain_arg,
            azimuth_fft_size_arg,
            elevation_fft_size_arg,
            azimuth_fov_arg,
            elevation_fov_arg,
            radar_process,
        ]
    )
