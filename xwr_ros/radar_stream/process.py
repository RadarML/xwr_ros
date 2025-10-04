import os
import yaml

import numpy as np
import matplotlib.pyplot as plt

from ament_index_python.packages import get_package_share_directory

from xwr.rsp import numpy as xwr_rsp

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class RadarProcess(Node):
    def __init__(self, config_file_path, rsp="AWR1843AOP", gain=5e-6):
        super().__init__("radar_process")

        with open(config_file_path, "r") as file:
            self.cfg = yaml.safe_load(file)

        self.layout = None
        self.rsp_inst = getattr(xwr_rsp, rsp)(window=False, size={"azimuth": 128})
        self.cmap = plt.get_cmap("hot")  # type: ignore
        self.gain = gain
        self.bridge = CvBridge()

        self.pub_rd = self.create_publisher(Image, "radar_rd", 10)
        self.pub_ra = self.create_publisher(Image, "radar_ra", 10)

        # Create subscriber
        self.subscription = self.create_subscription(
            Int16MultiArray, "radar_raw", self.radar_cb, 10
        )

    def get_msg(self, rimg):
        img = (self.cmap(np.clip(rimg * self.gain, 0, 1))[:, :, :3] * 255).astype(
            np.uint8
        )
        r_msg = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
        r_msg.header.stamp = self.get_clock().now().to_msg()
        r_msg.header.frame_id = "radar"
        return r_msg

    def radar_cb(self, msg):
        data = np.array(msg.data, dtype=np.int16)

        if self.layout is None:
            self.layout = []
            for dim in msg.layout.dim:
                self.layout.append(dim.size)

        data = data.reshape(self.layout)
        # self.get_logger().info(f'Received: "{data.shape}"')

        dear = np.abs(self.rsp_inst(data[None, ...]))
        rd = np.swapaxes(np.mean(dear, axis=(0, 2, 3)), 0, 1)
        ra = np.swapaxes(np.mean(dear, axis=(0, 1, 2)), 0, 1)
        # self.get_logger().info(f'rd:"{rd.shape}", ra"{ra.shape}"')

        self.pub_rd.publish(self.get_msg(rd))
        self.pub_ra.publish(self.get_msg(ra))


def main():
    rclpy.init()

    package_share_directory = get_package_share_directory("radar_stream")
    config_file_path = os.path.join(package_share_directory, "config", "config.yaml")

    node = RadarProcess(config_file_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
