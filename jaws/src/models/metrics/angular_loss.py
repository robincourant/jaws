"""Code adapted from: https://github.com/jadarve/optical-flow-filter."""

import torch
import torch.nn as nn


class AngularLoss(nn.Module):
    def forward(self, flow1: torch.Tensor, flow2: torch.Tensor):
        """Compute the angular error between two flow fields.

        :param flow1: first optical flow field.
        :param flow2: second optical flow field.
        :return: angular error field in degrees.
        """
        f1_x = flow1[..., 0]
        f1_y = flow1[..., 1]

        f2_x = flow2[..., 0]
        f2_y = flow2[..., 1]

        top = 1.0 + f1_x * f2_x + f1_y * f2_y
        bottom = torch.sqrt(1.0 + f1_x * f1_x + f1_y * f1_y) * torch.sqrt(
            1.0 + f2_x * f2_x + f2_y * f2_y
        )
        div = torch.clamp(top / bottom, min=-1, max=1)
        loss = torch.rad2deg(torch.arccos(div)).mean() / 180.0

        return loss
