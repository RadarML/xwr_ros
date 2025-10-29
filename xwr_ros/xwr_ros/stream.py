import logging
import os

import numpy as np
import rclpy
import xwr
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.time import Time
from rich.logging import RichHandler
from std_msgs.msg import MultiArrayDimension
from xwr_msgs.msg import IQ


class RadarPublisher(Node):
    def __init__(self):
        super().__init__("xwr_ros")
        self.declare_parameter("config", "config")
        cfg = self.get_parameter("config").get_parameter_value().string_value
        cfg_path = os.path.join(
            get_package_share_directory("xwr_ros"), "config", f"{cfg}.yaml"
        )

        with open(cfg_path, "r") as file:
            self.cfg = yaml.safe_load(file)

        logging.basicConfig(
            level=logging.INFO,
            format="%(name)-12s  %(message)s",
            datefmt="[%H:%M:%S]",
            handlers=[RichHandler()],
        )

        self.awr = xwr.XWRSystem(**self.cfg)

        self.msg = IQ()
        self.msg.iq.layout.dim = []
        self.msg.header.frame_id = "xwr"
        self.dim_label = ["chirp", "tx", "rx", "sample"]
        for i, dim_size in enumerate(self.awr.config.raw_shape):
            dim = MultiArrayDimension()
            dim.label = self.dim_label[i]
            dim.size = dim_size
            dim.stride = int(np.prod(self.awr.config.raw_shape[i:]))
            self.msg.iq.layout.dim.append(dim)
        self.msg.iq.layout.data_offset = 0

        self.pub_radar = self.create_publisher(IQ, "xwr/iq", 10)

    def stream(self):
        for frame in self.awr.dstream(numpy=False):
            seconds = int(frame.timestamp)
            nanoseconds = int((frame.timestamp - seconds) * 1_000_000_000)
            self.msg.header.stamp = Time(
                seconds=seconds, nanoseconds=nanoseconds
            ).to_msg()
            data = np.frombuffer(frame.data, dtype=np.int16)
            self.msg.iq.data = data.tolist()
            self._logger.info(f"{frame.timestamp}")
            self.pub_radar.publish(self.msg)


def main():
    rclpy.init()
    node = RadarPublisher()
    try:
        node.stream()
    except KeyboardInterrupt:
        node._logger.warning("stream interrupted by user.")
        node.awr.stop()
