"""Radar signal processing node."""

import os
from functools import cached_property

import jax
import matplotlib.pyplot as plt
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from jax import numpy as jnp
from jaxtyping import Array, Float, Int16
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import MultiArrayDimension
from xwr import XWRConfig
from xwr.rsp import iq_from_iiqq
from xwr.rsp import jax as xwr_rsp
from xwr_msgs.msg import IQ

# warnings.filterwarnings("error")


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
        self.gain = (
            self.get_parameter("gain").get_parameter_value().double_value
        )

        cfg_path = os.path.join(
            get_package_share_directory("xwr_ros"), "config", f"{cfg}.yaml"
        )
        with open(cfg_path, "r") as file:
            cfg = yaml.safe_load(file)
        self.cfg = XWRConfig(**cfg["radar"])

        self._logger.info(f"config: {cfg_path}")
        self._logger.info(f"range resolution: {self.cfg.range_resolution}")
        self._logger.info(f"doppler resolution: {self.cfg.doppler_resolution}")
        self._logger.info(f"fov: (ele: {ele_fov}, azi {azi_fov})")
        self._logger.info(f"angle fft size: (ele: {ele_ffts}, azi; {azi_ffts})")
        self._logger.info(f"rsp: {rsp}")
        self._logger.info(f"gain: {self.gain}")

        self.layout = None
        self.rsp_inst: xwr_rsp.RSPJax = getattr(xwr_rsp, rsp)(
            window=False, size={"elevation": ele_ffts, "azimuth": azi_ffts}
        )
        self.cfar = xwr_rsp.CFARCASO()
        self.radar_pc = xwr_rsp.PointCloud(
            self.cfg.range_resolution,
            self.cfg.doppler_resolution,
            angle_fov=(ele_fov, azi_fov),
            angle_size=(ele_ffts, azi_ffts),
        )
        self.cmap = plt.get_cmap("hot")  # type: ignore

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
            IQ, "xwr/iq", self.radar_cb, 1
        )

    @cached_property
    def process(self):
        @jax.jit
        def _inner(iiqq: Int16[Array, "doppler tx rx _range"]):
            iq = iq_from_iiqq(iiqq[None, ...])  # batch
            d__r = self.rsp_inst.doppler_range(iq)
            dear = self.rsp_inst.elevation_azimuth(d__r)
            rd_mask, sig, snr = self.cfar(jnp.abs(d__r.squeeze(0)))
            pc_mask, pc = self.radar_pc(jnp.abs(dear.squeeze(0)), rd_mask)
            dear = jnp.abs(dear).squeeze(0)
            return dear, rd_mask, pc, pc_mask

        return _inner

    def get_msg(
        self,
        rimg: Float[np.ndarray | Array, "w h"],
        mask: Float[np.ndarray | Array, "w h"] | None = None,
        mask_color: list = [0, 0.99, 0],
    ):
        img = self.cmap(np.clip(rimg * self.gain, 0, 1))[:, :, :3]
        if mask is not None:
            img[mask] = mask_color
        img = np.ascontiguousarray((img * 255).astype(np.uint8))
        h, w, c = img.shape
        return Image(
            data=img.tobytes(),
            height=h,
            width=w,
            encoding="rgb8",
            is_bigendian=0,
            step=w * c,
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

        dear, rd_mask, pc, pc_mask = self.process(data)

        if self.pub_rd.get_subscription_count() > 0:
            rd = np.swapaxes(jnp.mean(dear, axis=(1, 2)), 0, 1)
            msg_rd = self.get_msg(rd, rd_mask)
            msg_rd.header = msg.header
            self.pub_rd.publish(msg_rd)

        if self.pub_ra.get_subscription_count() > 0:
            ra = np.swapaxes(jnp.mean(dear, axis=(0, 1)), 0, 1)
            msg_ra = self.get_msg(ra)
            msg_ra.header = msg.header
            self.pub_ra.publish(msg_ra)

        if self.pub_pc.get_subscription_count() > 0:
            points = np.asarray(pc)[pc_mask]
            points[:, 2] = -points[:, 2]
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
