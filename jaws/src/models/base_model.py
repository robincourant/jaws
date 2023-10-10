import os.path as osp
import socket
import struct
from datetime import datetime
from typing import Any, Dict, Tuple, List, Callable
import cv2
import numpy as np
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from utils.file_utils import create_dir, save_pickle, load_pickle
from utils.nerf_utils import get_rays
from utils.image_utils import (
    save_torch_image,
    save_loss_marginal_image,
    save_heatmaps,
    save_heatmap,
    put_text_on_image,
)
from utils.camera_utils import CameraPoseGenerator as cam_gen, pose_distance

from kornia.color.hsv import rgb_to_hsv


class BaseModel(LightningModule):
    def __init__(
        self,
        result_dir: str,
        optimizer: nn.Module,
        lr_scheduler: nn.Module,
        criterion: nn.Module,
        run_type: str,
        num_steps: int,
        upsample_steps: int,
        max_ray_batch: int,
        saturation_loss: bool,
        error_map: bool,
        floater_ratio: float,
    ):
        super().__init__()

        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._result_dir = result_dir
        self.criterion = criterion
        self._val_dir = osp.join(result_dir, "validation")
        self.benchmark_dir = osp.join(result_dir, "benchmark")
        create_dir(self._val_dir)
        if run_type == "infer":
            timestamp = datetime.now().strftime("%m-%d_%H-%M")
            self._test_dir = osp.join(result_dir, "test", f"test_{timestamp}")
            create_dir(self._test_dir)

        self._num_steps = num_steps
        self._upsample_steps = upsample_steps
        self._max_ray_batch = max_ray_batch
        self._saturation_loss = saturation_loss
        self._error_map = error_map
        self._floater_ratio = floater_ratio

    def _save_step(self, rgb_pred: torch.Tensor, batch_idx: int):
        """Save predicted RGB images."""
        pred_path = osp.join(
            self._val_dir, f"{batch_idx:03}_{self.current_epoch+1:02}.png"
        )
        img = cv2.cvtColor(
            (rgb_pred[0].detach().cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        cv2.imwrite(
            pred_path,
            img,
        )
        # use directly logger to log image
        self.logger.log_image(key="val_img", images=[img[:, :, ::-1]])

        if self._error_map:
            emap = (
                self.trainer.datamodule.train_dataset.error_map[0]
                .view(128, 128)
                .cpu()
                .numpy()
            )
            emap = (emap - emap.min()) / (emap.max() - emap.min())
            emap_path = osp.join(
                self._val_dir,
                f"{batch_idx:03}_{self.current_epoch+1:02}_emap.png",
            )
            cv2.imwrite(
                emap_path,
                (emap * 255).astype(np.uint8),
            )

    def _test_save_step(
        self, rgb_pred: torch.Tensor, batch_idx: int, pose: torch.Tensor
    ):
        """Save predicted RGB images in prediction"""
        pred_path = osp.join(self._test_dir, f"test_{batch_idx:03}.png")
        img = cv2.cvtColor(
            (rgb_pred[0].detach().cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        cv2.imwrite(
            pred_path,
            img,
        )

        file_dir = osp.join(self._test_dir, "traj.txt")
        with open(file_dir, "a+") as output_file:
            output_file.write(
                " ".join(
                    [
                        str(elem)
                        for elem in pose.cpu().detach().numpy().flatten()[:-4]
                    ]
                )
                + "\n"
            )

    def _log_step(
        self,
        mode: str,
        loss: torch.Tensor,
    ):
        """Log metrics at each epoch and each step for the training."""
        on_step = True if mode == "train" else False
        self.log(
            f"{mode}/loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def _eval_step_wo_gt(self):
        raise NotImplementedError()

    def _eval_step_w_gt(self):
        raise NotImplementedError()

    def training_step(self):
        raise NotImplementedError()

    def validation_step(self):
        raise NotImplementedError()

    def test_step(self):
        raise NotImplementedError()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Define optimizers and LR schedulers."""
        if self._optimizer is None:
            optimizer = optim.Adam(
                self.model.parameters(), lr=0.001, weight_decay=5e-4
            )  # naive adam
        else:
            optimizer = self._optimizer(self.model)

        if self._lr_scheduler is None:
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1
            )  # fake scheduler
        else:
            lr_scheduler = self._lr_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train/loss",
        }
