from typing import Tuple

from pytorch_lightning import LightningModule
import torch

from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import torchvision.transforms.functional as F


class RAFT_tv(LightningModule):
    def __init__(self):
        super(RAFT_tv, self).__init__()
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = x
        B, C, H, W = x1.shape
        x1 = 2 * (x1 / 255.0) - 1.0
        x2 = 2 * (x2 / 255.0) - 1.0
        x1, x2 = self.preprocess(x1, x2)
        flow_raw = self.model(x1, x2, num_flow_updates=12)[-1]
        flow_resized = torch.nn.functional.interpolate(flow_raw, size=[H, W])
        return flow_resized

    def preprocess(self, img1_batch, img2_batch):
        transforms = Raft_Small_Weights.DEFAULT.transforms()
        img1_batch = F.resize(img1_batch, size=[224, 224])
        img2_batch = F.resize(img2_batch, size=[224, 224])
        return transforms(img1_batch, img2_batch)

    def postprocess(self, flow, img_size):
        flow_resized = F.resize(flow, size=img_size)
        return flow_resized


def make_raft_estimator(freeze: bool):
    model = RAFT_tv().eval()

    if freeze:
        for p in model.model.parameters():
            p.requires_grad = False

    return model
