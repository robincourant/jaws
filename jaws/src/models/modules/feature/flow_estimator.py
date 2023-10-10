from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from jaws.src.models.modules.feature.raft import make_raft_estimator
from utils.flow_utils import FlowUtils


def g_sigmoid(x, q, b):
    return 1.0 / (1.0 + torch.exp(b * q) * torch.exp(-b * x))


def f_inverse(x):
    return 1 - (1 / (torch.max(x, torch.ones_like(x))))


class FlowEstimator(nn.Module):
    """Optical flow estimator.

    :param raft_pretrained_path: path to the pretrained raft model.
    """

    def __init__(self, raft_pretrained_path: str):
        super(FlowEstimator, self).__init__()
        self.flow_estimator = make_raft_estimator(freeze=True)
        self._flow_utils = FlowUtils()

    def _flow_polar(self, flow: torch.Tensor, step_module: float = 0) -> torch.Tensor:
        """Normalize flow by inversing polar modules (H, W, C)."""
        polar_flow = self._flow_utils.xy_to_polar(flow)

        scaled_polar_flow = torch.zeros_like(polar_flow)
        scaled_polar_flow[:, :, 0] = polar_flow[:, :, 0]
        scaled_polar_flow[:, :, 1] = polar_flow[:, :, 1]
        return scaled_polar_flow

    def _unit_normalize_flow(self, flow: torch.Tensor) -> torch.Tensor:
        """Normalize flow by unitarize their polar modules (H, W, C)."""
        polar_flow = self._flow_utils.xy_to_polar(flow)

        scaled_polar_flow = torch.zeros_like(polar_flow)
        scaled_polar_flow[:, :, 0] = torch.ones_like(polar_flow[:, :, 0])
        scaled_polar_flow[:, :, 1] = polar_flow[:, :, 1]

        scaled_flow = self._flow_utils.polar_to_xy(scaled_polar_flow)

        return scaled_flow

    def _step_normalize_flow(
        self, flow: torch.Tensor, step_module: float = 0
    ) -> torch.Tensor:
        """
        Normalize flow by unitarize their polar modules (H, W, C) greater than
        a threshold (`step_module`), otherwise, zero.
        """
        polar_flow = self._flow_utils.xy_to_polar(flow)

        scaled_polar_flow = torch.zeros_like(polar_flow)
        scaled_polar_flow[:, :, 0] = 1 * (polar_flow[:, :, 0] > step_module)
        scaled_polar_flow[:, :, 1] = polar_flow[:, :, 1]

        scaled_flow = self._flow_utils.polar_to_xy(scaled_polar_flow)

        return scaled_flow

    def _sigmoid_normalize_flow(
        self, flow: torch.Tensor, step_module: float = 0
    ) -> torch.Tensor:
        """
        Normalize flow by aplying a sigmoid on their polar modules (H, W, C).
        """
        polar_flow = self._flow_utils.xy_to_polar(flow)

        scaled_polar_flow = torch.zeros_like(polar_flow)
        scaled_polar_flow[:, :, 0] = g_sigmoid(
            polar_flow[:, :, 0], q=torch.tensor(20), b=torch.tensor(0.1)
        )
        scaled_polar_flow[:, :, 1] = polar_flow[:, :, 1]
        scaled_flow = self._flow_utils.polar_to_xy(scaled_polar_flow)

        return scaled_flow

    def _inverse_normalize_flow(
        self, flow: torch.Tensor, step_module: float = 0
    ) -> torch.Tensor:
        """Normalize flow by inversing polar modules (H, W, C)."""
        polar_flow = self._flow_utils.xy_to_polar(flow)

        scaled_polar_flow = torch.zeros_like(polar_flow)
        scaled_polar_flow[:, :, 0] = f_inverse(polar_flow[:, :, 0])
        scaled_polar_flow[:, :, 1] = polar_flow[:, :, 1]
        scaled_flow = self._flow_utils.polar_to_xy(scaled_polar_flow)

        return scaled_flow

    def _inverse_normalize_flow_polar(
        self, flow: torch.Tensor, step_module: float = 0
    ) -> torch.Tensor:
        """Normalize flow by inversing polar modules (H, W, C)."""
        polar_flow = self._flow_utils.xy_to_polar(flow)

        scaled_polar_flow = torch.zeros_like(polar_flow)
        scaled_polar_flow[:, :, 0] = f_inverse(polar_flow[:, :, 0])
        scaled_polar_flow[:, :, 1] = polar_flow[:, :, 1]

        return scaled_polar_flow

    def _estimate_flow(self, frames: torch.Tensor) -> torch.Tensor:
        """Estimate frows from RGB frames."""
        flows = self.flow_estimator([frames[:-1], frames[1:]]).permute([0, 2, 3, 1])
        return flows

    def compute_flow(
        self, frames: List[np.array], ftype: str = "EE"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract flow style features from a list of frames.
        WARNING: For inference only, please don't forget `.eval()` and
        `torch.no_grad()`.
        Types: including:

        EE: EndPoint flow -> XY
        NEE: normalised Endpoint -> Norm xy flow
        AN: Angular -> XY

        :param frames: list of raw RGB frames 0-255 range (T, C, H, W).
        :return: encoded flow style vectors (B, C_f, T_f, W_f, H_f).

        """
        # Estimate flows, output shape: (T, H, W, C)
        flows = self._estimate_flow(frames)

        # Normalize flow chunks, output shape: (T, H, W, C)
        if ftype == "NEE":
            normalized_flows = torch.stack(
                [self._inverse_normalize_flow(f) for f in flows]
            )
            return normalized_flows, flows.unsqueeze(0)

        if ftype == "EE" or ftype == "AN":
            return flows, flows.unsqueeze(0)  # Ck, chk_size, H, W, C
