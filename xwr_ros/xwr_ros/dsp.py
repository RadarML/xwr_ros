"""Radar signal processing module."""

from functools import cached_property

import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float, Int16
from sensor_msgs.msg import Image
from xwr import XWRConfig
from xwr.rsp import iq_from_iiqq
from xwr.rsp import jax as xwr_rsp

device_map = {
    "AWR1843AOP": "AWR1843",
    "AWR1843Boost": "AWR1843",
    "AWR1642Boost": "AWR1642",
}


class RadarDSP:
    """Radar Digital Signal Processing class.

    This class encapsulates all radar signal processing operations including:
    - Range-Doppler processing
    - Angle of Arrival (AoA) estimation
    - CFAR detection
    - Point cloud generation
    """

    def __init__(
        self,
        config: XWRConfig,
        rsp: str = "AWR1843AOP",
        gain: float = 2e-6,
        azimuth_fft_size: int = 64,
        elevation_fft_size: int = 64,
        azimuth_fov: float = 60.0,
        elevation_fov: float = 60.0,
        window: bool = False,
    ):
        """Initialize radar DSP.

        Args:
            config: XWR radar configuration
            rsp: DSP processor type (e.g., 'AWR1843AOP')
            gain: Gain factor for visualization
            azimuth_fft_size: FFT size for azimuth processing
            elevation_fft_size: FFT size for elevation processing
            azimuth_fov: Azimuth field of view in degrees
            elevation_fov: Elevation field of view in degrees
            window: Whether to apply windowing in FFT
        """
        self.cfg = config
        self.gain = gain
        self.azimuth_fft_size = azimuth_fft_size
        self.elevation_fft_size = elevation_fft_size
        self.azimuth_fov = azimuth_fov
        self.elevation_fov = elevation_fov

        # Initialize RSP instance
        self.rsp_inst: xwr_rsp.RSPJax = getattr(xwr_rsp, rsp)(
            window=window,
            size={"elevation": elevation_fft_size, "azimuth": azimuth_fft_size},
        )

        # Initialize CFAR detector
        self.cfar = xwr_rsp.CFARCASO()

        # Initialize point cloud generator
        self.radar_pc = xwr_rsp.PointCloud(
            self.cfg.range_resolution,
            self.cfg.doppler_resolution,
            angle_fov=(elevation_fov, azimuth_fov),
            angle_size=(elevation_fft_size, azimuth_fft_size),
        )

        # Colormap for visualization
        self.cmap = plt.get_cmap("hot")  # type: ignore

    @cached_property
    def process(self):
        """Get JIT-compiled processing function.

        Returns:
            Compiled function that processes radar IQ data
        """

        @jax.jit
        def _inner(iiqq: Int16[Array, "doppler tx rx _range"]):
            """Process radar IQ data.

            Args:
                iiqq: Interleaved IQ data with shape [doppler, tx, rx, range]

            Returns:
                Tuple of (dear, rd_mask, pc, pc_mask) where:
                - dear: Doppler-Elevation-Azimuth-Range tensor
                - rd_mask: Range-Doppler CFAR detection mask
                - pc: Point cloud coordinates
                - pc_mask: Point cloud validity mask
            """
            iq = iq_from_iiqq(iiqq[None, ...])  # Add batch dimension
            d__r = self.rsp_inst.doppler_range(iq)
            dear = self.rsp_inst.elevation_azimuth(d__r)
            rd_mask, sig, snr = self.cfar(jnp.abs(d__r.squeeze(0)))
            pc_mask, pc = self.radar_pc(jnp.abs(dear.squeeze(0)), rd_mask)
            dear = jnp.abs(dear).squeeze(0)
            return dear, rd_mask, pc, pc_mask

        return _inner

    def process_frame(self, data: np.ndarray | Array):
        """Process a single radar frame.

        Args:
            data: Radar IQ data with shape [doppler, tx, rx, range]

        Returns:
            Tuple of (dear, rd_mask, pc, pc_mask)
        """
        if isinstance(data, np.ndarray):
            data = jnp.array(data)
        return self.process(data)

    def get_range_doppler(self, dear: Array) -> np.ndarray:
        """Extract range-doppler map from DEAR tensor.

        Args:
            dear: Doppler-Elevation-Azimuth-Range tensor

        Returns:
            Range-Doppler map with shape [range, doppler]
        """
        return np.swapaxes(jnp.mean(dear, axis=(1, 2)), 0, 1)

    def get_range_azimuth(self, dear: Array) -> np.ndarray:
        """Extract range-azimuth map from DEAR tensor.

        Args:
            dear: Doppler-Elevation-Azimuth-Range tensor

        Returns:
            Range-Azimuth map with shape [range, azimuth]
        """
        return np.swapaxes(jnp.mean(dear, axis=(0, 1)), 0, 1)

    def get_range_elevation(self, dear: Array) -> np.ndarray:
        """Extract range-elevation map from DEAR tensor.

        Args:
            dear: Doppler-Elevation-Azimuth-Range tensor

        Returns:
            Range-Elevation map with shape [range, elevation]
        """
        return np.swapaxes(jnp.mean(dear, axis=(0, 2)), 0, 1)

    def get_azimuth_elevation(self, dear: Array) -> np.ndarray:
        """Extract azimuth-elevation map from DEAR tensor.

        Args:
            dear: Doppler-Elevation-Azimuth-Range tensor
        Returns:
            Azimuth-Elevation map with shape [elevation, azimuth]
        """
        return np.swapaxes(jnp.mean(dear, axis=(0, 3)), 0, 1)

    def get_point_cloud(self, pc: Array, pc_mask: Array) -> np.ndarray:
        """Get valid point cloud points.

        Args:
            pc: Point cloud coordinates
            pc_mask: Validity mask

        Returns:
            Array of valid points with shape [n_points, 4] (x, y, z, doppler)
        """
        return np.asarray(pc)[pc_mask]

    def to_image_msg(
        self,
        rimg: Float[np.ndarray | Array, "w h"],
        mask: Float[np.ndarray | Array, "w h"] | None = None,
        mask_color: list = [0, 0.99, 0],
    ) -> Image:
        """Convert numpy array to ROS Image message.

        Args:
            rimg: Input image array
            mask: Optional mask to overlay
            mask_color: Color for masked regions [R, G, B]

        Returns:
            ROS Image message
        """
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

    def to_image_array(
        self,
        rimg: Float[np.ndarray | Array, "w h"],
        mask: Float[np.ndarray | Array, "w h"] | None = None,
        mask_color: list = [0, 0.99, 0],
    ) -> np.ndarray:
        """Convert processed data to RGB image array.

        Args:
            rimg: Input image array
            mask: Optional mask to overlay
            mask_color: Color for masked regions [R, G, B]

        Returns:
            RGB image as numpy array with shape [h, w, 3]
        """
        img = self.cmap(np.clip(rimg * self.gain, 0, 1))[:, :, :3]
        if mask is not None:
            img[mask] = mask_color
        return (img * 255).astype(np.uint8)
