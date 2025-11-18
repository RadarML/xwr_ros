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
from jaxtyping import Array, Complex64, Int16
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import MultiArrayDimension
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
        cfg = self.get_parameter("config").get_parameter_value().string_value
        rsp = self.get_parameter("dsp").get_parameter_value().string_value
        self.gain = (
            self.get_parameter("gain").get_parameter_value().double_value
        )

        cfg_path = os.path.join(
            get_package_share_directory("xwr_ros"), "config", f"{cfg}.yaml"
        )
        with open(cfg_path, "r") as file:
            self.cfg = yaml.safe_load(file)

        self._logger.info(f"config: {cfg_path}")
        self._logger.info(f"rsp: {rsp}")
        self._logger.info(f"gain: {self.gain}")

        self.layout = None
        self.rsp_inst: xwr_rsp.RSPJax = getattr(xwr_rsp, rsp)(
            window=False, size={"elevation": 64, "azimuth": 64}
        )
        self.cmap = plt.get_cmap("hot")

        self.pub_rd = self.create_publisher(Image, "xwr/range_doppler", 10)
        self.pub_ra = self.create_publisher(Image, "xwr/range_azimuth", 10)

        self.subscription = self.create_subscription(
            IQ, "xwr/iq", self.radar_cb, 1
        )

        self._logger.info("Radar RSP and visualization.")

    @cached_property
    def process(self):
        @jax.jit  # jit x30 faster
        def _inner(
            iiqq: Int16[Array, "doppler tx rx _range"],
        ) -> Complex64[Array, "doppler elevation azimuth range"]:
            iq = iq_from_iiqq(iiqq[None, ...])  # batch
            d__r = self.rsp_inst.doppler_range(iq)
            dear = self.rsp_inst.elevation_azimuth(d__r)
            return dear

        return _inner

    def get_msg(self, rimg):
        img = self.cmap(np.clip(rimg * self.gain, 0, 1))[:, :, :3] * 255
        img = np.ascontiguousarray(img.astype(np.uint8))
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

        dear = jnp.abs(self.process(data))

        if self.pub_rd.get_subscription_count() > 0:
            rd = jnp.swapaxes(jnp.mean(dear, axis=(0, 2, 3)), 0, 1)
            msg_rd = self.get_msg(rd)
            msg_rd.header = msg.header
            self.pub_rd.publish(msg_rd)

        if self.pub_ra.get_subscription_count() > 0:
            ra = jnp.swapaxes(jnp.mean(dear, axis=(0, 1, 2)), 0, 1)
            msg_ra = self.get_msg(ra)
            msg_ra.header = msg.header
            self.pub_ra.publish(msg_ra)


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
