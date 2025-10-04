import logging
import os
from pprint import pprint

import numpy as np
import rclpy
import xwr
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rich.logging import RichHandler
from std_msgs.msg import Int16MultiArray, MultiArrayDimension


class RadarPublisher(Node):
    def __init__(self):
        super().__init__("radar_stream")
        self.declare_parameter("config", "config")
        cfg = self.get_parameter("config").get_parameter_value().string_value
        cfg_path = os.path.join(
            get_package_share_directory("radar_stream"), "config", f"{cfg}.yaml"
        )

        with open(cfg_path, "r") as file:
            self.cfg = yaml.safe_load(file)

        pprint(self.cfg)

        logging.basicConfig(
            level=logging.INFO,
            format="%(name)-12s  %(message)s",
            datefmt="[%H:%M:%S]",
            handlers=[RichHandler()],
        )
        self.log = logging.getLogger("XWRDemo")

        self.awr = xwr.XWRSystem(**self.cfg)

        self.msg = None
        self.dim_label = ["chirp", "tx", "rx", "sample"]
        self.pub_radar = self.create_publisher(Int16MultiArray, "radar_raw", 10)

    def stream(self):
        for frame in self.awr.dstream(numpy=True):
            # batch doppler elevation azimuth range
            if self.msg is None:
                self.msg = Int16MultiArray()

                # Set up the layout dimensions
                self.msg.layout.dim = []

                # Add dimension information
                for i, dim_size in enumerate(frame.shape):
                    dim = MultiArrayDimension()
                    dim.label = self.dim_label[i]
                    dim.size = dim_size
                    dim.stride = int(
                        np.prod(frame.shape[i:])
                    )  # stride for this dimension
                    self.msg.layout.dim.append(dim)

                self.msg.layout.data_offset = 0

            # Flatten the array and convert to list
            self.msg.data = frame.flatten().tolist()
            self.pub_radar.publish(self.msg)


def main():
    rclpy.init()
    node = RadarPublisher()
    try:
        node.stream()
    except KeyboardInterrupt:
        node.log.warning("stream interrupted by user.")
        node.awr.stop()


if __name__ == "__main__":
    main()
