from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn

from jaws.src.models.base_model import BaseModel
from kornia.color.hsv import rgb_to_hsv
from utils.nerf_utils import get_rays
from torch_efficient_distloss import eff_distloss  # , eff_distloss_native


class DNeRFModel(BaseModel):
    def __init__(
        self,
        result_dir: str,
        optimizer: nn.Module,
        lr_scheduler: nn.Module,
        criterion: nn.Module,
        run_type: str,
        bound: float,
        aabb: List,
        min_near: float,
        density_thresh: float,
        num_steps: int,
        upsample_steps: int,
        max_ray_batch: int,
        background_radius: int,
        time_encoding: str,
        warp_encoding: str,
        sigma_encoding: str,
        direction_encoding: str,
        background_encoding: str,
        n_warp_layers: int,
        n_sigma_layers: int,
        n_color_layers: int,
        n_background_layers: int,
        warp_hidden_dim: int,
        sigma_hidden_dim: int,
        color_hidden_dim: int,
        background_hidden_dim: int,
        geo_feat_dim: int,
        encoder_num_levels: int,
        background_perlin_noise: bool,
        saturation_loss: bool,
        error_map: bool,
        floater_ratio: float,
    ):
        super().__init__(
            result_dir=result_dir,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            run_type=run_type,
            num_steps=num_steps,
            upsample_steps=upsample_steps,
            max_ray_batch=max_ray_batch,
            saturation_loss=saturation_loss,
            error_map=error_map,
            floater_ratio=floater_ratio,
        )
        self.nerf_type = "dnerf"

        from jaws.src.models.modules.dnerf.network import DNeRFNetwork

        self.model = DNeRFNetwork(
            background_radius=background_radius,
            time_encoding=time_encoding,
            warp_encoding=warp_encoding,
            sigma_encoding=sigma_encoding,
            direction_encoding=direction_encoding,
            background_encoding=background_encoding,
            n_warp_layers=n_warp_layers,
            n_sigma_layers=n_sigma_layers,
            n_color_layers=n_color_layers,
            n_background_layers=n_background_layers,
            warp_hidden_dim=warp_hidden_dim,
            sigma_hidden_dim=sigma_hidden_dim,
            color_hidden_dim=color_hidden_dim,
            background_hidden_dim=background_hidden_dim,
            geo_feat_dim=geo_feat_dim,
            bound=bound,
            aabb=aabb,
            encoder_num_levels=encoder_num_levels,
            density_scale=1,
            min_near=min_near,
            density_thresh=density_thresh,
            background_perlin_noise=background_perlin_noise,
        )

    def _eval_step_wo_gt(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        time: torch.Tensor,
        background_color: int,
        staged: bool,
        perturb: bool,
    ) -> torch.Tensor:
        """Train step without ground-truth image."""
        rgb_pred, _ = self.model.render(
            rays_o,
            rays_d,
            time,
            staged=staged,
            background_color=background_color,
            perturb=perturb,
            num_steps=self._num_steps,
            upsample_steps=self._upsample_steps,
            max_ray_batch=self._max_ray_batch,
            eliminate_floater=True,  # TODO: change it -> eliminate floater
        )

        return rgb_pred

    def _eval_step_w_gt(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        time: torch.Tensor,
        images: torch.Tensor,
        background_color: int,
        staged: bool,
        perturb: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train step with ground-truth image."""
        C = images.shape[-1]
        # srgb color space
        if C == 4:
            if background_color is None:
                # Train with random background color if using alpha mixing
                background_color = torch.rand_like(images[..., :3])  # [N, 3]
            rgb_gt = images[..., :3] * images[..., 3:] + background_color * (
                1 - images[..., 3:]
            )
        else:
            rgb_gt = images

        rgb_pred, floater_pack = self.model.render(
            rays_o,
            rays_d,
            time,
            staged=staged,
            background_color=background_color,
            perturb=perturb,
            num_steps=self._num_steps,
            upsample_steps=self._upsample_steps,
            max_ray_batch=self._max_ray_batch,
            eliminate_floater=True,  # TODO: change it -> eliminate floater
        )

        return rgb_pred, rgb_gt, floater_pack

    def training_step(
        self, batch: Dict[str, Any], batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Training loop."""
        rays_o = batch["rays_o"]  # [B, N, 3]
        rays_d = batch["rays_d"]  # [B, N, 3]
        time = batch["time"]  # [B, 1]
        B, N = rays_o.shape[:2]
        H, W = batch["H"], batch["W"]

        # If there is no gt image the CLIP loss is used
        if "images" not in batch:
            rgb_pred = self._eval_step_wo_gt(
                rays_o,
                rays_d,
                time,
                background_color=None,
                staged=False,
                perturb=True,
            )
            rgb_pred = rgb_pred.reshape(B, H, W, 3).permute(0, 3, 1, 2)
            loss = self.clip_loss(rgb_pred.contiguous())
        else:
            images = batch["images"]  # [B, N, 3/4]
            rgb_pred, rgb_gt, floater_pack = self._eval_step_w_gt(
                rays_o,
                rays_d,
                time,
                images,
                background_color=None,
                staged=False,
                perturb=True,
            )
            loss = self.criterion(rgb_pred, rgb_gt)

            if self._saturation_loss:
                s_pred = (
                    rgb_to_hsv(rgb_pred.permute([2, 0, 1]).unsqueeze(0))
                    .squeeze(0)
                    .permute([1, 2, 0])
                )[:, :, 1]
                s_gt = (
                    rgb_to_hsv(rgb_gt.permute([2, 0, 1]).unsqueeze(0))
                    .squeeze(0)
                    .permute([1, 2, 0])
                )[:, :, 1]

                loss = self.criterion(rgb_pred, rgb_gt) + self.criterion(
                    s_pred, s_gt
                ).unsqueeze(2)
            else:
                loss = self.criterion(rgb_pred, rgb_gt)

            if floater_pack is not None and self._floater_ratio > 0:
                [weight, midpoints, interval] = floater_pack
                # decay_factor = self.current_epoch / self.trainer.max_epochs
                loss_floater = self._floater_ratio * eff_distloss(
                    weight, midpoints, interval
                )
            else:
                loss_floater = 0
        loss = loss.mean() + loss_floater
        self._log_step("train", loss)
        return {"loss": loss}

    def validation_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Validation loop."""
        rays_o = batch["rays_o"]  # [B, N, 3]
        rays_d = batch["rays_d"]  # [B, N, 3]
        time = batch["time"]  # [B, 1]
        B, N = rays_o.shape[:2]
        H, W = batch["H"], batch["W"]

        # If there is no gt image the CLIP loss is used
        if "images" not in batch:
            rgb_pred = self._eval_step_wo_gt(
                rays_o,
                rays_d,
                time,
                background_color=1,
                staged=True,
                perturb=False,
            )
            rgb_pred = rgb_pred.reshape(B, H, W, 3).permute(0, 3, 1, 2)
            loss = self.clip_loss(rgb_pred)
        else:
            images = batch["images"]  # [B, N, 3/4]
            rgb_pred, rgb_gt, _ = self._eval_step_w_gt(
                rays_o,
                rays_d,
                time,
                images,
                background_color=1,
                staged=True,
                perturb=False,
            )
            rgb_pred = rgb_pred.reshape(B, H, W, 3)
            rgb_gt = rgb_gt.reshape(B, H, W, 3)
            loss = self.criterion(rgb_pred, rgb_gt).mean(-1)

        loss = loss.mean()
        self._log_step("val", loss)
        self._save_step(rgb_pred, batch_idx)

        return {"loss": loss}

    def test_step(self, batch: torch.Tensor, batch_idx: torch.Tensor):
        """prediction_step loop."""
        rays_o = batch["rays_o"]  # [B, N, 3]
        rays_d = batch["rays_d"]  # [B, N, 3]
        time = batch["time"]  # [B, 1]
        B, N = rays_o.shape[:2]
        H, W = batch["H"], batch["W"]
        rgb_pred = self._eval_step_wo_gt(
            rays_o,
            rays_d,
            time,
            background_color=1,
            staged=True,
            perturb=False,
        )
        rgb_pred = rgb_pred.reshape(B, H, W, 3)
        self._test_save_step(rgb_pred, batch_idx, batch["poses"])

    @torch.no_grad()
    def render(
        self,
        pose: torch.Tensor,
        time: torch.Tensor,
        intrinsics: torch.Tensor,
        H: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """Render one image given view pose and parameter."""
        rays = get_rays(pose, intrinsics, H, W, -1)
        eval_kwargs = dict(
            rays_o=rays["rays_o"],
            rays_d=rays["rays_d"],
            time=time,
            background_color=None,
            staged=True,
            perturb=False,
        )
        rgb_pred = self._eval_step_wo_gt(**eval_kwargs)
        rgb_pred = rgb_pred.view(H, W, 3)

        return rgb_pred
