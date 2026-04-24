"""Radar stream node."""

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
from std_msgs.msg import Int16MultiArray, MultiArrayDimension, MultiArrayLayout
from xwr_msgs.msg import IQ, ChirpInfo


class RadarPublisher(Node):
    """Radar Stream Node."""

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
        self.msg_iq = IQ()
        self.layout = MultiArrayLayout()
        self.msg_info = ChirpInfo()

        self.msg_iq.header.frame_id = "xwr"
        self.layout.dim = []
        self.dim_label = ["chirp", "tx", "rx", "sample"]
        for i, dim_size in enumerate(self.awr.config.raw_shape):
            dim = MultiArrayDimension()
            dim.label = self.dim_label[i]
            dim.size = dim_size
            dim.stride = int(np.prod(self.awr.config.raw_shape[i:]))
            self.layout.dim.append(dim)
        self.layout.data_offset = 0

        for attr in dir(self.awr.config):
            if not attr.startswith("__") and hasattr(self.msg_info, attr):
                val = getattr(self.awr.config, attr)
                field = getattr(self.msg_info, attr)
                if hasattr(field, 'data') and isinstance(val, str):
                    field.data = val
                else:
                    setattr(self.msg_info, attr, val)

        self.pub_iq = self.create_publisher(IQ, "xwr/iq", 10)
        self.pub_info = self.create_publisher(ChirpInfo, "xwr/info", 10)

    def stream(self):
        que = self.awr.qstream(numpy=False)
        while True:
            frame = que.get(block=True, timeout=None)
            if frame is None:
                self._logger.info("Stream ended.")
                break

            # publish IQ data
            sec = int(frame.timestamp)
            nano_sec = int((frame.timestamp - sec) * 1_000_000_000)
            self.msg_iq.header.stamp = Time(
                seconds=sec, nanoseconds=nano_sec
            ).to_msg()
            self.msg_iq.iq = Int16MultiArray(
                layout=self.layout, data=frame.data
            )
            self.msg_iq.complete = frame.complete
            self.pub_iq.publish(self.msg_iq)

            # publish chirp config info
            if self.pub_info.get_subscription_count() > 0:
                self.msg_info.header = self.msg_iq.header
                self.pub_info.publish(self.msg_info)


def main():
    """Node access point."""
    rclpy.init()
    node = RadarPublisher()
    try:
        node.stream()
    except KeyboardInterrupt:
        node._logger.warning("stream interrupted by user.")
        node.awr.stop()
