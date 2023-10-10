from typing import List

import torch
from torch.nn.functional import mse_loss
import torchvision
from torchvision.models.vgg import VGG16_Weights


class VGGLoss(torch.nn.Module):
    """
    VGG perceptual loss:
    Paper: https://arxiv.org/pdf/1603.08155.pdf
    Code: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    """

    def __init__(
        self,
        resize: bool = False,
        feature_blocks: List[int] = [0, 1, 2, 3],
        style_blocks: List[int] = [],
    ):
        super(VGGLoss, self).__init__()

        # Initialize VGG blocks
        weights = VGG16_Weights.DEFAULT
        blocks = [
            torchvision.models.vgg16(weights=weights).features[:4].eval(),
            torchvision.models.vgg16(weights=weights).features[4:9].eval(),
            torchvision.models.vgg16(weights=weights).features[9:16].eval(),
            torchvision.models.vgg16(weights=weights).features[16:23].eval(),
        ]
        self.feature_blocks = feature_blocks
        self.style_blocks = style_blocks

        # Freeze VGG's parameters
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        # Initialize transformation parameters
        self.transform = torch.nn.functional.interpolate if resize else None
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("std", std)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        # Order channels: [B, C, H, W]
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
            y = y.permute(0, 3, 1, 2)
        # Normalize in/outputs
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        # Resize in/outputs
        if self.transform:
            x = self.transform(x, mode="bilinear", size=(224, 224), align_corners=False)
            y = self.transform(y, mode="bilinear", size=(224, 224), align_corners=False)

        # Evaluate loss value
        loss = 0
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            # Compute feature loss
            if i in self.feature_blocks:
                loss += mse_loss(x, y)
            # Compute style loss
            if i in self.style_blocks:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1) / act_x.numel()
                gram_y = act_y @ act_y.permute(0, 2, 1) / act_x.numel()
                loss += torch.norm(gram_x - gram_y)

        return loss
