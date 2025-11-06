# xwr_ros
A ROS2 wrapper for [XWR](https://github.com/RadarML/xwr)

## Depenendices
Depends on uv and [colcon-uv](https://github.com/nzlz/colcon-uv)
```sh
pip install uv colcon-uv
```

## Build with ROS2
Will create a uv venv and install necessary dependencies when building with colcon
```sh
UV_VENV_CLEAR=1 UV_LINK_MODE=symlink colcon build
```

## Setup
Modify the [config file](xwr_ros/config/config.yaml) with your radar USB port. Or add your own config file.
```yaml
radar:
    device: AWR1843
    port: /dev/ttyACM0
    frequency: 77.0
    idle_time: 6.0
    adc_start_time: 5.7
    ramp_end_time: 34.00
    tx_start_time: 1.0
    freq_slope: 67.012
    adc_samples: 256
    sample_rate: 10000
    frame_length: 64
    frame_period: 50.0
capture:
    sys_ip: 192.168.33.30
    fpga_ip: 192.168.33.180
    socket_buffer: 6291456
```

Set the ip address of the network interface for the DCA capture board as:
```
192.168.33.30
```

## Launch xwr_ros

Launch xwr streaming node with the config file.
```sh
ros2 launch xwr_ros radar.launch.py config:=config
```

This will publish radar signal topic
```
/xwr/iq
```

To run signal processing visualization. 
```sh
ros2 run xwr_ros process
```

This will publish range_doppler and range_azimuth visualization images.
```
/xwr/range_doppler /xwr/range_azimuth
```