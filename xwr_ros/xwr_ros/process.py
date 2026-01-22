"""Radar signal processing node."""

import os

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from jax import numpy as jnp
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import MultiArrayDimension
from xwr import XWRConfig
from xwr_msgs.msg import IQ

from .dsp import RadarDSP


class RadarProcess(Node):
    """Radar signal process node."""

    def __init__(self):
        super().__init__("sig_process")

        self.declare_parameter("config", "config")
        self.declare_parameter("dsp", "AWR1843AOP")
        self.declare_parameter("gain", 2e-6)
        self.declare_parameter("azimuth_fft_size", 64)
        self.declare_parameter("elevation_fft_size", 64)
        self.declare_parameter("azimuth_fov", 60.0)
        self.declare_parameter("elevation_fov", 60.0)

        cfg = self.get_parameter("config").get_parameter_value().string_value
        rsp = self.get_parameter("dsp").get_parameter_value().string_value
        azi_ffts = (
            self.get_parameter("azimuth_fft_size")
            .get_parameter_value()
            .integer_value
        )
        ele_ffts = (
            self.get_parameter("elevation_fft_size")
            .get_parameter_value()
            .integer_value
        )
        azi_fov = (
            self.get_parameter("azimuth_fov").get_parameter_value().double_value
        )
        ele_fov = (
            self.get_parameter("elevation_fov")
            .get_parameter_value()
            .double_value
        )
        gain = self.get_parameter("gain").get_parameter_value().double_value

        cfg_path = os.path.join(
            get_package_share_directory("xwr_ros"), "config", f"{cfg}.yaml"
        )
        with open(cfg_path, "r") as file:
            cfg = yaml.safe_load(file)
        radar_config = XWRConfig(**cfg["radar"])

        self._logger.info(f"config: {cfg_path}")
        self._logger.info(f"range resolution: {radar_config.range_resolution}")
        self._logger.info(
            f"doppler resolution: {radar_config.doppler_resolution}"
        )
        self._logger.info(f"fov: (ele: {ele_fov}, azi {azi_fov})")
        self._logger.info(f"angle fft size: (ele: {ele_ffts}, azi; {azi_ffts})")
        self._logger.info(f"rsp: {rsp}")
        self._logger.info(f"gain: {gain}")

        # Initialize RadarDSP
        self.dsp = RadarDSP(
            config=radar_config,
            rsp=rsp,
            gain=gain,
            azimuth_fft_size=azi_ffts,
            elevation_fft_size=ele_ffts,
            azimuth_fov=azi_fov,
            elevation_fov=ele_fov,
            window=False,
        )

        self.layout = None

        fname = ["x", "y", "z", "doppler"]
        self.fields = [
            PointField(
                name=n, offset=0 + 4 * i, datatype=PointField.FLOAT32, count=1
            )
            for i, n in enumerate(fname)
        ]
        self.point_step = sum(
            [f.count * 4 for f in self.fields]
        )  # 4 bytes per float32

        self.pub_rd = self.create_publisher(Image, "xwr/range_doppler", 10)
        self.pub_ra = self.create_publisher(Image, "xwr/range_azimuth", 10)
        self.pub_pc = self.create_publisher(PointCloud2, "xwr/point_cloud", 10)

        self.subscription = self.create_subscription(
            IQ, "xwr/iq", self.radar_cb, 10
        )

    def radar_cb(self, msg: IQ):
        if self.layout is None:
            self.layout, name = [], []
            dim: MultiArrayDimension
            for dim in msg.iq.layout.dim:
                self.layout.append(dim.size)
                name.append(dim.label)
            self._logger.info(f"Radar IQ layout: {name}")
            self._logger.info(f"Radar IQ shape: {self.layout}")

        data = jnp.frombuffer(msg.iq.data, dtype=jnp.int16)
        data = data.reshape(self.layout)

        # Process using RadarDSP
        dear, rd_mask, pc, pc_mask = self.dsp.process_frame(data)

        if self.pub_rd.get_subscription_count() > 0:
            rd = self.dsp.get_range_doppler(dear)
            msg_rd = self.dsp.to_image_msg(rd, rd_mask)
            msg_rd.header = msg.header
            self.pub_rd.publish(msg_rd)

        if self.pub_ra.get_subscription_count() > 0:
            ra = self.dsp.get_range_azimuth(dear)
            msg_ra = self.dsp.to_image_msg(ra)
            msg_ra.header = msg.header
            self.pub_ra.publish(msg_ra)

        if self.pub_pc.get_subscription_count() > 0:
            points = self.dsp.get_point_cloud(pc, pc_mask)
            self._logger.info(f"pts: {points.shape}")

            pointcloud_msg = PointCloud2(
                header=msg.header,
                height=1,
                width=points.shape[0],
                is_dense=True,
                is_bigendian=False,
                fields=self.fields,
                point_step=self.point_step,
                row_step=self.point_step * points.shape[0],
                data=points.tobytes(),
            )
            self.pub_pc.publish(pointcloud_msg)


def main():
    """Node execution point."""
    rclpy.init()

    node = RadarProcess()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
