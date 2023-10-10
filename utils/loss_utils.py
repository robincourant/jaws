import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from kornia.geometry.subpix import (
    spatial_soft_argmax2d,
    spatial_expectation2d,
    spatial_softmax2d,
)


class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in [1].
    Concretely, the spatial softmax of each feature
    map is used to compute a weighted mean of the pixel
    locations, effectively performing a soft arg-max
    over the feature dimension.
    References:
        [1]: End-to-End Training of Deep Visuomotor Policies,
        https://arxiv.org/abs/1504.00702
    """

    def __init__(self, normalize=False):
        """Constructor.
        Args:
            normalize (bool): Whether to use normalized
                image coordinates, i.e. coordinates in
                the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize

    def _coord_grid(self, h, w, device):
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w, device=device),
                    torch.linspace(-1, 1, h, device=device),
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, w, device=device),
                torch.arange(0, h, device=device),
            )
        )

    def forward(self, x):
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."
        # compute a spatial softmax over the input:
        # given an input of shape (B, C, H, W),
        # reshape it to (B*C, H*W) then apply
        # the softmax operator over the last dimension
        b, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # create a meshgrid of pixel coordinates
        # both in the x and y axes
        xc, yc = self._coord_grid(h, w, x.device)

        # element-wise multiply the x and y coordinates
        # with the softmax, then sum over the h*w dimension
        # this effectively computes the weighted mean of x
        # and y locations

        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # concatenate and reshape the result
        # to (B, C*2) where for every feature
        # we have the expected x and y pixel
        # locations

        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)


if __name__ == "__main__":
    b, c, h, w = 32, 64, 12, 12
    x = torch.zeros(b, c, h, w)
    true_max = torch.randint(0, 10, size=(b, c, 2))
    for i in range(b):
        for j in range(c):
            x[i, j, true_max[i, j, 0], true_max[i, j, 1]] = 1000
    soft_max = SpatialSoftArgmax()(x).reshape(b, c, 2)
    assert torch.allclose(true_max.float(), soft_max)


import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format="NCHW"):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.0

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self.height),
            np.linspace(-1.0, 1.0, self.width),
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == "NHWC":
            feature = (
                feature.transpose(1, 3)
                .tranpose(2, 3)
                .view(-1, self.height * self.width)
            )
        else:
            feature = feature.view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints


def softmax_2D(x):
    y = nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)
    return y


def softmax_2D_with_confidence(input, temperature):
    input_soft: torch.Tensor = spatial_softmax2d(input, temperature)
    confidence = input_soft.max() - input_soft.min()
    return input_soft, confidence


def argmax_Coord_2D(x):
    return (x == torch.max(x)).nonzero()


def spatial_soft_argmax2d_with_confidence(input, temperature, normalized_coordinates):
    input_soft: torch.Tensor = spatial_softmax2d(input, temperature)
    output: torch.Tensor = spatial_expectation2d(input_soft, normalized_coordinates)
    confidence = input_soft.max() - input_soft.min()
    return output, confidence


def spatial_soft_argmax2d_with_misc(
    input, temperature, normalized_coordinates, return_misc=True
):
    input_soft: torch.Tensor = spatial_softmax2d(input, temperature)
    output: torch.Tensor = spatial_expectation2d(input_soft, normalized_coordinates)

    # [B, C]
    confidence = input_soft.max() - input_soft.min()

    if return_misc:
        return output, input_soft, confidence

    return output


def minmaxnorm(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x


def DoG(
    x: torch.Tensor, blur_kernel: int = 3, blur_sigma: int = 5, interval_kernal: int = 2
):
    """Difference of Gaussian

    Args:
        x (torch.Tensor): C, H, W
        blur_kernel (int, optional):  Defaults to 3.
        blur_sigma (int, optional):  Defaults to 5.
        interval_kernal (int, optional):  Defaults to 5.
    """
    blur_0 = T.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)
    blur_1 = T.GaussianBlur(kernel_size=blur_kernel + interval_kernal, sigma=blur_sigma)

    return torch.abs(blur_0(x) - blur_1(x)).permute(1, 2, 0).squeeze(2)


if __name__ == "__main__":
    data = torch.zeros([1, 3, 3, 3])
    data[0, 0, 0, 1] = 10
    data[0, 1, 1, 1] = 10
    data[0, 2, 1, 2] = 10
    layer = SpatialSoftmax(3, 3, 3, temperature=1)
    print(layer(data))
