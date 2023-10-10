from typing import Tuple

import numpy as np
import torch

from utils.misc_utils import divide


class FlowUtils:
    """
    Handle different flow conversions: RGB, HSV, Unity, RAFT.
    Input shape: (H, W, C).

    :param bgr: wether to handle BGR or RGB frames.
    :param sensitivity: scale value for HSV conversion.
    :param epsilon: epsilon value for divisions.
    """

    def __init__(
        self, bgr: bool = False, sensitivity: int = 1, epsilon: float = 1e-5
    ):
        self._bgr = bgr
        self._sensitivity = sensitivity
        self._epsilon = epsilon
        self._colorwheel = self._make_colorwheel()  # shape [55x3]

    @staticmethod
    def _make_colorwheel() -> np.array:
        """
        Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow",
        ICCV, 2007
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

        Code follows the original C++ source code of Daniel Scharstein.
        Code follows the the Matlab source code of Deqing Sun.
        Code adapted from: https://github.com/princeton-vl/RAFT.

        :return: color wheel
        """
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = np.zeros((ncols, 3))
        col = 0

        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
        col = col + RY
        # YG
        colorwheel[col : col + YG, 0] = 255 - np.floor(
            255 * np.arange(0, YG) / YG
        )
        colorwheel[col : col + YG, 1] = 255
        col = col + YG
        # GC
        colorwheel[col : col + GC, 1] = 255
        colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
        col = col + GC
        # CB
        colorwheel[col : col + CB, 1] = 255 - np.floor(
            255 * np.arange(CB) / CB
        )
        colorwheel[col : col + CB, 2] = 255
        col = col + CB
        # BM
        colorwheel[col : col + BM, 2] = 255
        colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
        col = col + BM
        # MR
        colorwheel[col : col + MR, 2] = 255 - np.floor(
            255 * np.arange(MR) / MR
        )
        colorwheel[col : col + MR, 0] = 255

        return colorwheel

    def _flow_xy_to_colors(self, x: np.array, y: np.array) -> np.array:
        """
        Applies the flow color wheel to flow components x and y.

        According to the C++ source code of Daniel Scharstein
        According to the Matlab source code of Deqing Sun
        Code adapted from: https://github.com/princeton-vl/RAFT.

        :param x: input horizontal flow of shape [H,W].
        :param y: input vertical flow of shape [H,W].
        :return: flow visualization image of shape [H,W,3].
        """
        flow_image = np.zeros((x.shape[0], x.shape[1], 3), np.uint8)
        ncols = self._colorwheel.shape[0]

        mod = np.nan_to_num(np.sqrt(np.square(x) + np.square(y)))
        angle = np.nan_to_num(np.arctan2(-y, -x) / np.pi)

        fk = (angle + 1) / 2 * (ncols - 1)
        k0 = np.floor(fk).astype(np.int32)
        k1 = k0 + 1
        k1[k1 == ncols] = 0
        f = fk - k0

        for i in range(self._colorwheel.shape[1]):
            tmp = self._colorwheel[:, i]
            col0 = tmp[k0] / 255.0
            col1 = tmp[k1] / 255.0
            col = (1 - f) * col0 + f * col1
            idx = mod <= 1
            col[idx] = 1 - mod[idx] * (1 - col[idx])
            col[~idx] = col[~idx] * 0.75  # out of range
            # Note the 2-i => BGR instead of RGB
            ch_idx = 2 - i if self._bgr else i
            flow_image[:, :, ch_idx] = np.floor(255 * col)

        return flow_image

    def _raft_flow_to_frame(
        self, flow_xy: np.array, norm: bool = False
    ) -> np.array:
        """
        Convert xy-flow into colored frame.
        Expects a two dimensional flow image of shape.
        Code adapted from: https://github.com/princeton-vl/RAFT.

        :param flow_xy: xy-flow image of shape [H,W,2].
        :param norm: wether to min-max normalize flow modules.
        :return: flow visualization image of shape [H,W,3] and scaling value.
        """
        assert flow_xy.ndim == 3, "input flow must have three dimensions"
        assert flow_xy.shape[2] == 2, "input flow must have shape [H,W,2]"

        x = flow_xy[:, :, 0]
        y = flow_xy[:, :, 1]

        if norm:
            mod = np.sqrt(np.square(x) + np.square(y))
            mod_max = np.max(mod)
            x /= mod_max + self._epsilon
            y /= mod_max + self._epsilon

        frame = self._flow_xy_to_colors(x, y)

        return frame

    def _rgb_to_hsv(
        self, frame: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Convert a RGB frame to a HSV frame.
        Adapted from: https://github.com/opencv/opencv/blob/17234f82d025e3bbfb
        f611089637e5aa2038e7b8/3rdparty/openexr/Imath/ImathColorAlgo.cpp
        """
        r_index, g_index, b_index = (2, 1, 0) if self._bgr else (0, 1, 2)

        r_channel = frame[:, :, r_index]
        g_channel = frame[:, :, g_index]
        b_channel = frame[:, :, b_index]

        max_channel_arg = np.argmax(frame, axis=-1)
        max_channel = np.max(frame, axis=-1)
        min_channel = np.min(frame, axis=-1)
        range_channel = max_channel - min_channel
        zero_channel = np.zeros_like(r_channel, dtype=np.float64)
        sat = zero_channel
        hue = zero_channel

        val = max_channel
        sat = np.multiply(max_channel != 0, divide(range_channel, max_channel))

        sat_mask = sat != 0
        # Case 1: max channel is red
        h_r = np.multiply(
            sat_mask,
            np.multiply(
                max_channel_arg == r_index,
                divide(g_channel - b_channel, range_channel),
            ),
        )
        # Case 2: max channel is green
        h_g = np.multiply(
            sat_mask,
            np.multiply(
                max_channel_arg == g_index,
                2 + divide(b_channel - r_channel, range_channel),
            ),
        )
        # Case 3: max channel is blue
        h_b = np.multiply(
            sat_mask,
            np.multiply(
                max_channel_arg == b_index,
                4 + divide(r_channel - g_channel, range_channel),
            ),
        )
        hue = np.multiply(sat_mask, (h_r + h_b + h_g) / 6)
        hue += np.multiply(sat_mask, np.multiply(hue < 0, 1))

        return hue, sat, val

    def _hsv_to_rgb(
        self, frame: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Convert a HSV frame to a RGB frame.
        Adapted from: https://github.com/opencv/opencv/blob/17234f82d025e3bbfb
        f611089637e5aa2038e7b8/3rdparty/openexr/Imath/ImathColorAlgo.cpp
        """
        hue = frame[:, :, 0]
        sat = frame[:, :, 1]
        val = frame[:, :, 2]

        r_channel = np.zeros_like(hue, dtype=np.float64)
        g_channel = np.zeros_like(hue, dtype=np.float64)
        b_channel = np.zeros_like(hue, dtype=np.float64)
        zero_channel = np.zeros_like(hue, dtype=np.float64)

        hue = np.where(hue == 1, zero_channel, 6 * hue)

        i = np.floor(hue)
        f = hue - i
        p = val * (1 - sat)
        q = val * (1 - (sat * f))
        t = val * (1 - (sat * (1 - f)))

        r_channel = np.where(i == 0, val, r_channel)
        g_channel = np.where(i == 0, t, g_channel)
        b_channel = np.where(i == 0, p, b_channel)

        r_channel = np.where(i == 1, q, r_channel)
        g_channel = np.where(i == 1, val, g_channel)
        b_channel = np.where(i == 1, p, b_channel)

        r_channel = np.where(i == 2, p, r_channel)
        g_channel = np.where(i == 2, val, g_channel)
        b_channel = np.where(i == 2, t, b_channel)

        r_channel = np.where(i == 3, p, r_channel)
        g_channel = np.where(i == 3, q, g_channel)
        b_channel = np.where(i == 3, val, b_channel)

        r_channel = np.where(i == 4, t, r_channel)
        g_channel = np.where(i == 4, p, g_channel)
        b_channel = np.where(i == 4, val, b_channel)

        r_channel = np.where(i == 5, val, r_channel)
        g_channel = np.where(i == 5, p, g_channel)
        b_channel = np.where(i == 5, q, b_channel)

        return r_channel, g_channel, b_channel

    def _hsv_frame_to_flow(self, frame: np.array) -> np.array:
        """Convert a HSV (Unity) flow frame into flow field."""
        # Convert RGB frame to HSV
        hue, _, val = self._rgb_to_hsv(frame)

        # Get polar module and angle from hue and value encoding
        theta = ((2 * hue) - 1) * np.pi
        r = val / self._sensitivity

        # Convert polar coordinates to euclidean coordinates
        x = -r * np.cos(theta)
        y = r * np.sin(theta)

        flow = np.stack([x, y], axis=-1)

        return flow

    def _hsv_flow_to_frame(self, flow: np.array) -> np.array:
        """Convert flow to HSV frame."""
        x, y = flow[:, :, 0], flow[:, :, 1]

        theta = np.arctan2(y, -x)
        module = np.sqrt(x ** 2 + y ** 2)

        hue = (theta + np.pi) / (2 * np.pi)
        sat = np.ones(hue.shape)
        val = module * self._sensitivity
        hsv_frame = np.stack([hue, sat, val], axis=-1)

        r_channel, g_channel, b_channel = self._hsv_to_rgb(hsv_frame)
        frame = (
            np.stack([b_channel, g_channel, r_channel], axis=-1)
            if self._bgr
            else np.stack([r_channel, g_channel, b_channel], axis=-1)
        )

        return frame

    def flow_to_frame(self, flow: np.array, method: str = "raft") -> np.array:
        """Convert xy-flow to frame according the RAFT or HSV methods."""
        frame = (
            self._raft_flow_to_frame(flow)
            if method == "raft"
            else self._hsv_flow_to_frame(flow)
        )
        return frame

    def frame_to_flow(self, frame: np.array) -> np.array:
        """Convert an RGB/BGR frame to xy-flow."""
        flow = self._hsv_frame_to_flow(frame)
        return flow

    @staticmethod
    def xy_to_polar(xy_flow: torch.Tensor) -> torch.Tensor:
        """Convert euclidean coordinates to polar coordinates."""
        mod = torch.sqrt(xy_flow[:, :, 0] ** 2 + xy_flow[:, :, 1] ** 2)
        theta = torch.atan2(xy_flow[:, :, 1], xy_flow[:, :, 0])

        polar_flow = torch.zeros_like(xy_flow)
        polar_flow[:, :, 0] = mod
        polar_flow[:, :, 1] = theta

        return polar_flow

    @staticmethod
    def polar_to_xy(polar_flow: torch.Tensor) -> torch.Tensor:
        """Convert polar coordinates to euclidean coordinates."""
        mod = polar_flow[:, :, 0]
        theta = polar_flow[:, :, 1]

        xy_flow = torch.zeros_like(polar_flow)
        xy_flow[:, :, 0] = mod * torch.cos(theta)
        xy_flow[:, :, 1] = mod * torch.sin(theta)

        return xy_flow
