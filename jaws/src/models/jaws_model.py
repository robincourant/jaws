import os.path as osp
from typing import Any, Dict, List, Tuple

import cv2 as cv
import numpy as np
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import wandb

from jaws.src.models.callbacks.early_stopping import EarlyStopping
from jaws.src.models.callbacks.grad_norm import GradNorm
from jaws.src.models.metrics.angular_loss import AngularLoss
from jaws.src.models.metrics.pose_loss import LitePoseLoss, DEFAULT_ARGS
from jaws.src.models.metrics.vgg_loss import VGGLoss
from jaws.src.models.modules.camera_transform import MultiCameraTransform
from jaws.src.models.modules.feature.flow_estimator import FlowEstimator
from utils.camera_utils import pose_distance, get_sequence
from utils.file_utils import create_dir, save_pickle
from utils.image_utils import (
    save_image_gif,
    save_flow_gif,
    save_torch_image,
    save_torch_weighted_image,
    save_heatmaps_coords,
    convert_img_torch_to_cv,
    convert_img_cv_to_torch,
    InfoRecorder,
)
from utils.loss_utils import minmaxnorm
from utils.nerf_utils import get_rays, get_rays_mixed_grad


class Recorder:
    def __init__(self, content: Dict[str, Any]):
        for attr_name, attr_value in content.items():
            setattr(self, attr_name, attr_value)


class JAWSModel(LightningModule):
    def __init__(
        self,
        alpha_losses: float,
        dynamic: bool,
        encoder_pretrained_path: str,
        flow_loss: bool,
        flow_loss_type: str,
        grad_norm: bool,
        learning_rate: float,
        max_ray_batch: int,
        model_size: str,
        model: LightningModule,
        num_steps: int,
        pixel_loss_type: bool,
        pixel_loss: bool,
        pose_loss_type: str,
        pose_loss: bool,
        raft_pretrained_path: str,
        result_dir: str,
        upsample_steps: int,
    ):
        super().__init__()

        self._result_dir = result_dir
        self.jaws_dir = osp.join(result_dir, "jaws")
        create_dir(self.jaws_dir)

        # Pixel criterion
        self._pixel_loss = pixel_loss
        if pixel_loss_type == "mse":
            self.pixel_criterion = nn.MSELoss(reduction="mean")
        elif pixel_loss_type == "vgg":
            self.pixel_criterion = VGGLoss()
        else:
            raise NotImplementedError
        # Flow criterion
        self._flow_loss = flow_loss
        if flow_loss_type == "EE":  # EE: end-to-end
            self.flow_criterion = nn.MSELoss(reduction="mean")
        elif flow_loss_type == "AN" or flow_loss_type == "NAN":  # AN: angular
            self.flow_criterion = AngularLoss()
        else:
            raise NotImplementedError
        # Pose criterion
        self._pose_loss = pose_loss
        if pose_loss_type == "heatmap":
            self.pose_criterion = LitePoseLoss(DEFAULT_ARGS, heatmap_loss=True)
        elif pose_loss_type == "euclidean":
            self.pose_criterion = LitePoseLoss(
                DEFAULT_ARGS, euclidean_loss=True
            )
        else:
            raise NotImplementedError

        self.alpha = alpha_losses
        self.loss_mask = torch.tensor([pixel_loss, flow_loss, pose_loss])
        self.num_tasks = self.loss_mask.sum()
        self._grad_norm = grad_norm
        self._learning_rate = learning_rate

        self._num_steps = num_steps
        self._upsample_steps = upsample_steps
        self._max_ray_batch = max_ray_batch
        self._dynamic = dynamic
        self.model = model

        self.flow_estimator = FlowEstimator(
            raft_pretrained_path=raft_pretrained_path
        )
        for p in self.flow_estimator.parameters():
            p.requires_grad = False

    @staticmethod
    def _create_mesh(H: int, W: int) -> np.array:
        coords = np.asarray(
            np.stack(
                np.meshgrid(
                    np.linspace(0, W - 1, W),
                    np.linspace(0, H - 1, H),
                ),
                -1,
            ),
            dtype=int,
        )
        return coords.reshape(H * W, 2)

    def _jaws_fullgrad_step(
        self,
        rgb_gt: torch.Tensor,
        current_pose: torch.Tensor,
        current_time: torch.Tensor,
        intrinsics: torch.Tensor,
        num_sample_rays: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform an iNeRF step with or without sampling pixels."""
        H, W, C = rgb_gt.shape

        rays = get_rays(current_pose, intrinsics, H, W, -1)
        rays_o = rays["rays_o"]  # [B, N, 3]
        rays_d = rays["rays_d"]  # [B, N, 3]

        coords = self._create_mesh(H, W)  # Initialize a sampling mesh
        # Sample all the pixels if -1, else sample a sub part of the pixels
        if num_sample_rays == -1:
            batch = coords
        else:
            rand_inds = np.random.choice(
                coords.shape[0], size=num_sample_rays, replace=False
            )
            batch = coords[rand_inds]

        sampled_rays_o = rays_o.view(H, W, 3)[batch[:, 1], batch[:, 0]][None]
        sampled_rays_d = rays_d.view(H, W, 3)[batch[:, 1], batch[:, 0]][None]
        sampled_rgb_gt = rgb_gt[batch[:, 1], batch[:, 0]]

        rendering_params = dict(
            rays_o=sampled_rays_o,
            rays_d=sampled_rays_d,
            staged=True,
            background_color=None,
            perturb=False,
            num_steps=self._num_steps,
            upsample_steps=self._upsample_steps,
            max_ray_batch=self._max_ray_batch,
        )
        if current_time is not None:
            rendering_params["time"] = current_time

        sampled_rgb_pred = self.model.render(**rendering_params)

        return sampled_rgb_pred, sampled_rgb_gt

    def _jaws_mixedgrad_step(
        self,
        rgb_gt: torch.Tensor,
        current_pose: torch.Tensor,
        current_time: torch.Tensor,
        intrinsics: torch.Tensor,
        num_sample_rays: int,
        guidance_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform an iNeRF step with or without sampling pixels."""
        H, W, C = rgb_gt.shape
        coords = self._create_mesh(H, W)  # Initialize a sampling mesh
        if guidance_map is not None:
            # [B, N, 3]
            rays = get_rays_mixed_grad(
                current_pose, intrinsics, H, W, num_sample_rays, guidance_map
            )
            rays_o = rays["rays_o"]  # [B, H*W, 3]
            rays_d = rays["rays_d"]  # [B, H*W, 3]
            rand_inds = rays["randinds"].cpu().numpy().flatten()  # silly ...

        else:
            # all
            rays = get_rays(current_pose, intrinsics, H, W, -1)
            rays_o = rays["rays_o"]  # [B, H*W, 3]
            rays_d = rays["rays_d"]  # [B, H*W, 3]
            if num_sample_rays != -1:
                rand_inds = np.random.choice(
                    coords.shape[0], size=num_sample_rays, replace=False
                )  # when replace is False, no repeated elements is choosed
            else:
                # all pixels
                rand_inds = range(coords.shape[0])

        batch = coords[rand_inds]

        with torch.no_grad():
            all_inds = np.arange(coords.shape[0])
            ex_rand_inds = np.delete(all_inds, rand_inds)
            ex_batch = coords[ex_rand_inds]
            ex_sampled_rays_o = rays_o.view(H, W, 3)[
                ex_batch[:, 1], ex_batch[:, 0]
            ][None]
            ex_sampled_rays_d = rays_d.view(H, W, 3)[
                ex_batch[:, 1], ex_batch[:, 0]
            ][None]

            # (1, num_sample, 3)
            rendering_params = dict(
                rays_o=ex_sampled_rays_o,
                rays_d=ex_sampled_rays_d,
                staged=True,
                background_color=None,
                perturb=False,
                num_steps=self._num_steps,
                upsample_steps=self._upsample_steps,
                max_ray_batch=self._max_ray_batch,
            )
            if current_time is not None:
                rendering_params["time"] = current_time
            ex_sampled_rgb_pred = self.model.render(**rendering_params)

        sampled_rays_o = rays_o.view(H, W, 3)[batch[:, 1], batch[:, 0]][None]
        sampled_rays_d = rays_d.view(H, W, 3)[batch[:, 1], batch[:, 0]][None]

        rendering_params = dict(
            rays_o=sampled_rays_o,
            rays_d=sampled_rays_d,
            staged=True,
            background_color=None,
            perturb=False,
            num_steps=self._num_steps,
            upsample_steps=self._upsample_steps,
            max_ray_batch=self._max_ray_batch,
        )
        if current_time is not None:
            rendering_params["time"] = current_time
        sampled_rgb_pred = self.model.render(**rendering_params)

        # combine two images
        all_sampled_rgb_pred = torch.zeros(
            (1, ex_sampled_rgb_pred.shape[1] + sampled_rgb_pred.shape[1], 3),
            device=self.device,
        )

        all_sampled_rgb_pred[0][rand_inds] = sampled_rgb_pred[0]

        sample_map = all_sampled_rgb_pred.clone().detach()
        sample_map[torch.gt(sample_map, 0)] = 1

        all_sampled_rgb_pred[0][ex_rand_inds] = ex_sampled_rgb_pred[0]

        sampled_rgb_gt = rgb_gt
        return all_sampled_rgb_pred, sampled_rgb_gt, sample_map

    def _compute_keypoint_driven_guidance_map(
        self,
        cam_index: int,
        recorded_images: List[torch.Tensor],
        target_images: List[torch.Tensor],
        num_sample: int = 8000,
        ROI_size: int = 5,
    ):
        with torch.no_grad():
            half_num = int(num_sample / 2)
            # 1 torch to np image
            recorded_img = convert_img_torch_to_cv(
                recorded_images[cam_index][-1]
            )
            target_img = convert_img_torch_to_cv(
                target_images[cam_index].cpu()
            )
            H, W, C = target_img.shape
            # 2 opencv ORB image keypoint
            orb = cv.ORB_create(nfeatures=half_num)
            # kpoints
            kp_current = orb.detect(recorded_img, None)
            kp_target = orb.detect(target_img, None)

            kp_current_np = np.stack(
                [
                    np.array([int(kp.pt[0]), int(kp.pt[1])])
                    for kp in kp_current[:half_num]
                ]
            )
            kp_target_np = np.stack(
                [
                    np.array([int(kp.pt[0]), int(kp.pt[1])])
                    for kp in kp_target[:half_num]
                ]
            )

            # 3 keypoint to Region-of-Interest with 5x5
            kp_current_np = (
                kp_current_np
                + np.random.randint(ROI_size * 2, size=kp_current_np.shape)
                - ROI_size
            )
            kp_target_np = (
                kp_target_np
                + np.random.randint(ROI_size * 2, size=kp_target_np.shape)
                - ROI_size
            )

            kp_current_np = np.clip(kp_current_np, [0, 0], [H - 1, W - 1])
            kp_target_np = np.clip(kp_target_np, [0, 0], [H - 1, W - 1])

            # mesh_grid
            hs, ws = np.meshgrid(range(H), range(W))
            guidance_map = np.zeros((H, W))
            all_samples = np.vstack([kp_target_np, kp_current_np])
            guidance_map[all_samples[:, 0], all_samples[:, 1]] = 1
            pure_rand = int(num_sample - guidance_map.sum())

            while pure_rand != 0:
                rand_sample = np.random.randint(
                    [H - 1, W - 1], size=(pure_rand, 2)
                )
                guidance_map[rand_sample[:, 0], rand_sample[:, 1]] = 1
                pure_rand = int(num_sample - guidance_map.sum())

            guidance_map = convert_img_cv_to_torch(guidance_map * 255).to(
                device=self.device
            )

            # The guidance_map.sum() may lower than num
            assert guidance_map.sum() == num_sample
            return guidance_map

    def _compute_guidance_map(
        self,
        cam_index: int,
        recorded_images: List[torch.Tensor],
        target_images: List[torch.Tensor],
        gt_pred_flows: List[torch.Tensor],
        grad_norm: GradNorm = None,
        only_pose: bool = False,
    ):
        with torch.no_grad():
            guidance_map_pixel = None
            guidance_map_pose = None
            guidance_map_flow = None
            guidance_map = None
            if grad_norm is not None:
                loss_weights = (
                    grad_norm._get_loss_weights(self.loss_weights)
                    if self._grad_norm
                    else self.loss_weights
                )
                pixel_weight, pose_weight, flow_weight = loss_weights
            else:
                pixel_weight, pose_weight, flow_weight = 1, 1, 1

            if self._pixel_loss:
                # return self._compute_keypoint_driven_guidance_map(
                #     cam_index, recorded_images, target_images
                # )
                guidance_map_pixel = torch.abs(
                    recorded_images[cam_index][-1].to(device=self.device)
                    - target_images[cam_index]
                ).mean(axis=2)
                guidance_map_pixel = pixel_weight * minmaxnorm(
                    guidance_map_pixel
                )
                if guidance_map is None:
                    guidance_map = guidance_map_pixel

            if self._pose_loss:
                heatmapy = (
                    self.pose_criterion.heatmapy[cam_index]
                    .detach()
                    .mean(axis=0)
                )
                heatmapx = (
                    self.pose_criterion.heatmapx[cam_index]
                    .detach()
                    .mean(axis=0)
                )
                guidance_map_pose = torch.abs(
                    torch.abs(heatmapy) - torch.abs(heatmapx)
                ) + 0.5 * torch.abs(torch.abs(heatmapy) + torch.abs(heatmapx))
                guidance_map_pose = pose_weight * minmaxnorm(guidance_map_pose)

                if guidance_map is None:
                    guidance_map = guidance_map_pose
                else:
                    guidance_map += guidance_map_pose

            if (not self._flow_loss) or (gt_pred_flows is None) or only_pose:
                return guidance_map
            gt_flows, pred_flows = gt_pred_flows
            if pred_flows is None:
                return guidance_map

            guidance_map_flow_list = []
            guidance_map_flow_list.append(
                torch.abs(
                    gt_flows[0, 0].detach() - pred_flows[0, 0].detach()
                ).norm(dim=2)
            )
            guidance_map_flow = torch.stack(guidance_map_flow_list)
            guidance_map_flow = guidance_map_flow.mean(axis=0)
            guidance_map_flow = flow_weight * minmaxnorm(guidance_map_flow)
            guidance_map_flow = guidance_map_flow.to(self.device)

            if guidance_map is None:
                guidance_map = guidance_map_flow
            else:
                guidance_map += guidance_map_flow * guidance_map.mean()

            return guidance_map

    def _search_init_poses(self, recorder: Recorder) -> Recorder:
        with torch.no_grad():
            initial_time = torch.zeros((1, 1)) if self._dynamic else None
            old_temp_filter = self.pose_criterion.temp_filter
            self.pose_criterion.set_temporal_filter(False)

            H, W, C = recorder.frame_shape
            sv_dir = osp.join(self.jaws_dir, "init_search")
            create_dir(sv_dir)

            # For potential poses
            losses = [0 for _ in range(len(recorder.potential_poses))]
            bar = enumerate(tqdm(recorder.potential_poses))
            for potential_index, potential_pose in bar:
                for idx_camera in range(recorder.num_cameras):
                    with torch.cuda.amp.autocast(enabled=recorder.fp16):
                        pred_rgb, gt_rgb = self._jaws_fullgrad_step(
                            rgb_gt=recorder.target_images[idx_camera],
                            current_pose=potential_pose,
                            current_time=initial_time,
                            intrinsics=recorder.intrinsics,
                            num_sample_rays=-1,
                        )
                        pred_images = pred_rgb.view(H, W, C).unsqueeze(0)
                        gt_images = gt_rgb.view(H, W, C).unsqueeze(0)
                        poss_loss = self.pose_criterion(gt_images, pred_images)
                        losses[potential_index] += poss_loss

                save_heatmaps_coords(
                    self.pose_criterion.heatmapy,
                    self.pose_criterion.coords_y,
                    pred_images,
                    osp.join(sv_dir),
                    f"{potential_index:03}_l_"
                    + f"{losses[potential_index].item():.2f}_h",
                )
                save_heatmaps_coords(
                    self.pose_criterion.heatmapx,
                    self.pose_criterion.coords_x,
                    gt_images,
                    osp.join(sv_dir),
                    "x",
                )

            best_index = torch.argmin(torch.stack(losses))
            initial_pose = recorder.potential_poses[best_index]
            recorder.init_poses = [initial_pose] * recorder.num_cameras
            print(f"Found closest starting position: {best_index.item()}")

        self.pose_criterion.set_temporal_filter(old_temp_filter)

        return recorder

    def _regularize_loss(self, data, pose, range=0.6, weight=8):
        dist = torch.norm(data["pose_center"].to(self.device) - pose[:, :3, 3])
        loss = (
            torch.max(
                torch.tensor(0),
                (dist - range * data["pose_norm"].to(self.device)),
            )
            * weight
        )
        return loss**2

    def _regularize_loss_pure(self, pose):
        loss = torch.norm(pose[:, :3, 3])
        return loss

    def _initialize_optimizer(
        self, recorder: Recorder, params: Dict[str, torch.Tensor], lr: float
    ) -> Tuple[Recorder, Optimizer, nn.Module]:
        grad_norm = GradNorm(self.num_tasks, alpha=0.5).to(self.device)
        if self._grad_norm:
            params.append({"params": grad_norm.parameters()})
        recorder.retain_graph = True if self._grad_norm else False
        optimizer = torch.optim.Adam(
            params=params,
            lr=lr,
            betas=(0.9, 0.999),
        )
        return recorder, optimizer, grad_norm

    def _initialize_cameras(
        self, recorder: Recorder
    ) -> Tuple[Recorder, MultiCameraTransform]:
        # Initialize the cameras accordingly to the num of `target_images`
        num_cameras = len(recorder.target_images)
        num_frozen_cameras = len(recorder.frozen_camera_indices)

        camera_transforms = MultiCameraTransform(
            num_cameras=num_cameras - num_frozen_cameras,
            dynamic=(self._dynamic and recorder.diff_temporal),
            refocalize=recorder.diff_focal,
        ).to(self.device)

        # Initialize camera parameters
        H, W, C = recorder.target_images[0].shape
        scale_factor = H / recorder.dataloader["H"]
        intrinsics = recorder.dataloader["intrinsics"] * scale_factor

        camera_transforms.set_intrinsic(intrinsics, H, W)
        camera_transforms.set_focal_factor(recorder.focal_resize_factor)

        # SRGB color space
        if C == 4:
            recorder.target_images = [
                image[..., :3] * image[..., 3:] + (1 - image[..., 3:])
                for image in recorder.target_images
            ]

        # Update recorder
        recorder.num_cameras = num_cameras
        recorder.frame_shape = (H, W, C)
        recorder.intrinsics = intrinsics
        if self._dynamic:
            camera_transforms.set_init_time(recorder.initial_time)

        return recorder, camera_transforms

    def _optmize_multicamera(
        self,
        recorder: Recorder,
        epoch_index: int,
        optimizer: Optimizer,
        grad_norm: GradNorm,
        camera_transforms: MultiCameraTransform,
    ) -> Recorder:
        H, W, C = recorder.frame_shape
        optimizer.zero_grad()
        # Get current state of each camera  (i.e.: perform 1 "iNeRF step")
        pred_images, gt_images = [], []
        pred_poses, sample_maps, times, focals = [], [], [], []
        ignored_cam_num = 0
        reg_losses = []
        pred_flows = None
        for cam_index in range(recorder.num_cameras):
            initial_pose = recorder.init_poses[cam_index]
            if cam_index not in recorder.frozen_camera_indices:
                current_pose = camera_transforms(
                    cam_index - ignored_cam_num, initial_pose
                )
                current_time = (
                    camera_transforms.get_time(cam_index - ignored_cam_num)
                    if self._dynamic
                    else None
                )
                current_intrinsics = camera_transforms.get_intrinsic(
                    cam_index - ignored_cam_num
                )
                num_sample_grad = recorder.num_sample_grad
            else:
                current_pose = initial_pose
                current_time = (
                    torch.tensor([[recorder.initial_time]])
                    if self._dynamic
                    else None
                )
                current_intrinsics = (
                    camera_transforms.get_intrinsic_from_focal_factor(
                        recorder.focal_resize_factor
                    )
                )
                ignored_cam_num += 1
                num_sample_grad = recorder.num_sample_grad

            # apply guidance map during the sampling
            if (
                epoch_index > 0
                and recorder.use_guidance_map
                and recorder.guidance_type != "random"
            ):
                guidance_map = self._compute_guidance_map(
                    cam_index=cam_index,
                    recorded_images=recorder.info_recorder.get(
                        "recorded_images"
                    ),
                    target_images=recorder.target_images,
                    gt_pred_flows=(recorder.gt_flows, recorder.pred_flows)
                    if self._flow_loss
                    else None,
                    only_pose=recorder.only_pose_guidance,
                )
            else:
                guidance_map = None
            with torch.cuda.amp.autocast(enabled=recorder.fp16):
                pred_rgb, gt_rgb, sample_map = self._jaws_mixedgrad_step(
                    rgb_gt=recorder.target_images[cam_index],
                    current_pose=current_pose,
                    current_time=current_time,
                    intrinsics=current_intrinsics,
                    num_sample_rays=num_sample_grad,
                    guidance_map=guidance_map,
                )
                reg_losses.append(
                    self._regularize_loss(
                        recorder.dataloader,
                        current_pose,
                        recorder.regularizer_range,
                        recorder.regularizer_weight,
                    )
                )
                pred_images.append(pred_rgb.view(H, W, C))
                gt_images.append(gt_rgb.view(H, W, C))
                sample_maps.append(sample_map.view(H, W, C))
                pred_poses.append(current_pose.detach().cpu())
                if self._dynamic:
                    times.append(current_time.detach().cpu().item())
                else:
                    times.append(None)
                focals.append(
                    camera_transforms.get_focal_factor(
                        cam_index - ignored_cam_num
                    )
                    .detach()
                    .cpu()
                    .item()
                )

                recorder.info_recorder.append(
                    "recorded_images",
                    cam_index,
                    pred_images[-1].detach().cpu(),
                )
                recorder.info_recorder.append(
                    "recorded_poses", cam_index, pred_poses[-1]
                )
                recorder.info_recorder.append(
                    "recorded_times", cam_index, times[-1]
                )
                recorder.info_recorder.append(
                    "recorded_focals", cam_index, focals[-1]
                )

        reg_loss = torch.tensor(reg_losses).mean()
        pred_images = torch.stack(pred_images)
        gt_images = torch.stack(gt_images)
        tensor_sample_maps = torch.stack(sample_maps)

        # Add blur to images to smooth them (against low quality render)
        if recorder.blur_pred:
            pred_images = [
                recorder.blur(x.permute(2, 0, 1)) for x in pred_images
            ]
            pred_images = torch.stack(pred_images).permute(0, 2, 3, 1)

        # Compute loss and backprop + GradNorm
        _a, _b = self.alpha, (1 - self.alpha)
        with torch.cuda.amp.autocast(enabled=recorder.fp16):
            if (
                recorder.guidance_type == "inerf_original"
                and guidance_map is not None
            ):
                gt_nonzero = gt_images[
                    tensor_sample_maps.nonzero(as_tuple=True)
                ]
                pred_images_nonzero = pred_images[
                    tensor_sample_maps.nonzero(as_tuple=True)
                ]
                pixel_loss = _a * self.pixel_criterion(
                    gt_nonzero, pred_images_nonzero
                )

            else:
                # pixel loss
                pixel_loss = _a * self.pixel_criterion(gt_images, pred_images)

            # pose loss
            if self._pose_loss:
                pose_loss = _a * self.pose_criterion(gt_images, pred_images)
            else:
                pose_loss = 0 * pixel_loss

            # flow loss
            if self._flow_loss:
                sequence = get_sequence(pred_images) * 255
                pred_feat, pred_flows = self.flow_estimator.compute_flow(
                    sequence, recorder.flow_loss_type
                )
                flow_loss = _b * self.flow_criterion(
                    recorder.gt_feat, pred_feat
                )
            else:
                flow_loss = 0 * pixel_loss  # don't take memory then

        task_losses = torch.stack([pixel_loss, flow_loss, pose_loss])
        loss_weights = (
            grad_norm._get_loss_weights(self.loss_mask)
            if self._grad_norm
            else self.loss_mask
        )
        weighted_loss = torch.mul(loss_weights, task_losses)
        total_loss = weighted_loss.sum() + reg_loss

        total_loss.backward(retain_graph=recorder.retain_graph)

        for cam_index in range(recorder.num_cameras):
            recorder.info_recorder.append(
                "recorded_loss", cam_index, total_loss.item()
            )
            recorder.info_recorder.append(
                "recorded_pixelloss", cam_index, pixel_loss.item()
            )

        # Compute pose velocity
        d_r = np.zeros((recorder.num_cameras,))
        d_t = np.zeros((recorder.num_cameras,))
        if len(recorder.last_pred_poses) != 0:
            for cam_index in range(recorder.num_cameras):
                d_r[cam_index], d_t[cam_index] = pose_distance(
                    recorder.last_pred_poses[cam_index],
                    pred_poses[cam_index],
                )
        recorder.last_pred_poses = pred_poses

        if len(recorder.clip_scheduler_indices) > 0:
            camera_transforms.clip_parameters_schedule(
                recorder.clip_size,
                recorder.allow_backward_t,
                recorder.clip_scheduler_indices,
                (float(epoch_index) / recorder.num_epochs),
            )
        else:
            camera_transforms.clip_parameters(
                recorder.clip_size, recorder.allow_backward_t
            )

        if self._grad_norm:
            grad_norm.fit(
                task_losses[self.loss_mask],
                camera_transforms.parameters(),
            )
            optimizer.step()
            grad_norm.normalize_weights()
            loss_weights = grad_norm._get_loss_weights(self.loss_mask)
        else:
            optimizer.step()

        new_lrate = self._learning_rate * (
            0.8 ** ((epoch_index + 1) / recorder.num_epochs)
        )  # decay
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate
        # Store predicted results
        # Detach and move results to cpu
        recorder.pred_images = [x.detach().cpu() for x in pred_images]
        recorder.gt_images = [x.cpu() for x in gt_images]
        recorder.sample_maps = [x.cpu() for x in sample_maps]

        # pred_pose detached before
        recorder.pred_poses = pred_poses
        if pred_flows is not None:
            recorder.pred_flows = pred_flows.detach()
        else:
            recorder.pred_flows = pred_flows

        # Store losses
        recorder.total_loss = total_loss.item()
        recorder.pixel_loss = pixel_loss.item()
        recorder.flow_loss = flow_loss.item()
        recorder.pose_loss = pose_loss.item()
        recorder.reg_loss = reg_loss.item()
        recorder.loss_weights = loss_weights.detach()

        # Store camera parameters
        recorder.d_t = d_t
        recorder.d_r = d_r
        recorder.times = times
        recorder.focals = focals

        return recorder

    def _log_losses(
        self,
        recorder: Recorder,
        epoch_index: int,
        log_mode: str = "all",
    ):
        pixel_weight, flow_weight, pose_weight = recorder.loss_weights
        log_dict = {}
        if log_mode == "all" or log_mode == "main_branch":
            log_dict = {
                "jaws/batch_loss": recorder.total_loss,
                "jaws_loss/pixel_loss": recorder.pixel_loss,
                "jaws_loss/flow_loss": recorder.flow_loss,
                "jaws_loss/pose_loss": recorder.pose_loss,
                "jaws_loss/reg_loss": recorder.reg_loss,
                "jaws/epoch": epoch_index,
                "jaws_weights/pixel_weight": pixel_weight * 1,
                "jaws_weights/flow_weight": flow_weight * 1,
                "jaws_weights/pose_weight": pose_weight * 1,
                "jaws/early_stop": recorder.early_stop_count,
            }
            if self._dynamic:
                log_dict["jaws/init_time"] = recorder.initial_time
        for idx_cam in range(recorder.num_cameras):
            if log_mode == "all" or log_mode == "spatial":
                log_dict[f"cam_pose/delta_rotation{idx_cam}"] = (
                    np.rad2deg(recorder.d_r[idx_cam]) * 1
                )
                log_dict[f"cam_pose/delta_translation{idx_cam}"] = (
                    recorder.d_t[idx_cam] * 1
                )
                log_dict[f"cam_pose/focal{idx_cam}"] = recorder.focals[idx_cam]

            if self._dynamic and (log_mode == "all" or log_mode == "temporal"):
                log_dict[f"cam_pose/time{idx_cam}"] = recorder.times[idx_cam]
        wandb.log(log_dict)

    def _log_results(self, recorder: Recorder, epoch_index: int):
        # Create log directory
        log_dirname = f"{recorder.vid_idx}_{epoch_index:05}"
        log_dir = osp.join(self.jaws_dir, log_dirname)
        create_dir(log_dir)

        # Log every camera results
        for cam_index in range(recorder.num_cameras):
            # Get current camera state
            pred_rgb = recorder.pred_images[cam_index]
            gt_rgb = recorder.gt_images[cam_index]
            err_map = recorder.sample_maps[cam_index]

            # Save weighted pred/gt image
            weight_path = osp.join(log_dir, f"{cam_index:02}.png")
            err_path = osp.join(log_dir, f"err_{cam_index:02}.png")
            save_torch_weighted_image(pred_rgb, gt_rgb, weight_path)
            save_torch_image(err_map, err_path)

            if self._pose_loss:
                # Save heatmaps
                save_heatmaps_coords(
                    self.pose_criterion.heatmapy,
                    self.pose_criterion.coords_y,
                    recorder.pred_images,
                    osp.join(log_dir),
                    "_pred",
                )
                save_heatmaps_coords(
                    self.pose_criterion.heatmapx,
                    self.pose_criterion.coords_x,
                    recorder.gt_images,
                    osp.join(log_dir),
                    "_gt",
                )

        # Save camera current predicted poses
        pose_path = osp.join(log_dir, "poses.pk")
        save_pickle(recorder.pred_poses, pose_path)
        # Save images and flows as GIF
        save_image_gif(recorder.pred_images, recorder.gt_images, log_dir)

        flow_idx = 0
        if self._flow_loss:
            save_flow_gif(
                recorder.pred_flows[0, flow_idx].cpu().unsqueeze(0),
                recorder.gt_flows[0, flow_idx].detach().cpu().unsqueeze(0),
                log_dir,
            )

    def _run_jaws(
        self, recorder: Recorder
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Computes a batch of camera poses in NeRF coordinate system by
        optimizing them according to a batch of reference images.
        """
        # Initialize xamera parameters, optimizer and GradNorm
        recorder, camera_transforms = self._initialize_cameras(recorder)
        params = [{"params": camera_transforms.parameters()}]
        recorder, optimizer, grad_norm = self._initialize_optimizer(
            recorder, params, self._learning_rate
        )

        # Search for best initial pose from all reference poses
        recorder.init_poses = (
            self._search_init_poses(recorder)
            if (recorder.init_poses is None)
            and (recorder.potential_poses is not None)
            else recorder.init_poses
        )
        recorder.gt_flows = None

        if len(recorder.target_images) > 1:
            # Compute reference flow feature
            sequence = get_sequence(recorder.target_images) * 255
            (
                recorder.gt_feat,
                recorder.gt_flows,
            ) = self.flow_estimator.compute_flow(
                sequence, recorder.flow_loss_type
            )

        # Initialize empty sub-recorder
        recorder.info_recorder = InfoRecorder(
            [
                "recorded_images",
                "recorded_loss",
                "recorded_pixelloss",
                "recorded_poses",
                "recorded_times",
                "recorded_focals",
            ],
            recorder.num_cameras,
        )
        recorder.last_pred_poses = []
        recorder.pred_flows = None
        recorder.loss_weights = None
        recorder.old_flow_loss_button = None
        recorder.only_pose_guidance = False

        # Initialize blur
        recorder.blur = T.GaussianBlur(**recorder.blur_params)

        # Optimization loop
        pbar = tqdm(range(recorder.num_epochs))

        self.pose_criterion.only_update_y = False
        self.pose_criterion.reset_temporal_filter()
        early_stopper = EarlyStopping(
            recorder.early_stop_num, recorder.early_stop_delta
        )
        recorder.total_loss = 1e5
        for epoch_index in pbar:
            early_stopped = early_stopper.run_early_stopping_check(
                recorder.total_loss
            )
            recorder.early_stop_count = early_stopper.wait_count
            # Save flow for the last frame
            if (
                epoch_index == recorder.num_epochs - 1 or early_stopped
            ) and recorder.num_cameras > 1:
                recorder.old_flow_loss_button = self._flow_loss
                self._flow_loss = True
            # Optimization step
            recorder = self._optmize_multicamera(
                recorder, epoch_index, optimizer, grad_norm, camera_transforms
            )

            if not self.pose_criterion.only_update_y:
                self.pose_criterion.only_update_y = True

            # Log losses (every steps) and results (every `log_interval`)

            self._log_losses(recorder, epoch_index)
            pbar.set_postfix(loss=recorder.total_loss)
            if (
                not (
                    epoch_index % recorder.log_interval != 0
                    and epoch_index < recorder.num_epochs - 1
                )
                or early_stopped
            ):
                self._log_results(recorder, epoch_index)
            # torch.cuda.empty_cache()
            if early_stopped:
                print("early_stopped")
                break

        # Put back the flow configuration (important for consecutive work)
        if recorder.old_flow_loss_button is not None:
            self._flow_loss = recorder.old_flow_loss_button
        flow_idx = 0
        return (
            recorder.info_recorder,
            [x.cpu() for x in recorder.target_images],
            recorder.pred_flows[0, flow_idx].detach().cpu()
            if recorder.pred_flows is not None
            else None,
            recorder.gt_flows[0, flow_idx].detach().cpu()
            if recorder.gt_flows is not None
            else None,
        )

    def _run_jaws_two_strokes(
        self, recorder: Recorder
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Computes a batch of camera poses in NeRF coordinate system by
        optimizing first spatial and then temporal parameters according to a
        batch of reference images.
        """
        # Initialize xamera parameters, optimizer and GradNorm
        recorder, camera_transforms = self._initialize_cameras(recorder)
        spatial_params = [{"params": camera_transforms.spatial.parameters()}]
        (
            recorder,
            spatial_optimizer,
            spatial_grad_norm,
        ) = self._initialize_optimizer(
            recorder, spatial_params, self._learning_rate * 1.2
        )

        temporal_and_focal_params = []
        if recorder.diff_temporal and self._dynamic:
            temporal_and_focal_params.append(
                {"params": camera_transforms.temporal.parameters()}
            )
        if recorder.diff_focal:
            temporal_and_focal_params.append(
                {"params": camera_transforms.focal_factor_param.parameters()}
            )

        (
            recorder,
            temporal_optimizer,
            temporal_grad_norm,
        ) = self._initialize_optimizer(
            recorder, temporal_and_focal_params, self._learning_rate * 0.5
        )

        # Search for best initial pose from all reference poses
        recorder.init_poses = (
            self._search_init_poses(recorder)
            if (recorder.init_poses is None)
            and (recorder.potential_poses is not None)
            else recorder.init_poses
        )

        recorder.gt_flows = None
        if len(recorder.target_images) > 1:
            # Compute reference flow feature
            sequence = get_sequence(recorder.target_images) * 255
            (
                recorder.gt_feat,
                recorder.gt_flows,
            ) = self.flow_estimator.compute_flow(
                sequence, recorder.flow_loss_type
            )

        # Initialize empty sub-recorder
        recorder.info_recorder = InfoRecorder(
            [
                "recorded_images",
                "recorded_loss",
                "recorded_pixelloss",
                "recorded_poses",
                "recorded_times",
                "recorded_focals",
            ],
            recorder.num_cameras,
        )
        recorder.last_pred_poses = []
        recorder.pred_flows = None
        recorder.loss_weights = None
        recorder.old_flow_loss_button = None
        recorder.only_pose_guidance = False

        if recorder.old_flow_loss_button:
            self._flow_loss = recorder.old_flow_loss_button
        # Initialize blur
        recorder.blur = T.GaussianBlur(**recorder.blur_params)

        self.pose_criterion.only_update_y = False
        self.pose_criterion.reset_temporal_filter()
        recorder.total_loss = 1e5
        early_stopper = EarlyStopping(
            recorder.early_stop_num, recorder.early_stop_delta
        )

        # Optimization loop
        pbar = tqdm(range(recorder.num_epochs))
        info_recorder_merged = None
        for epoch_index in pbar:
            total_loss_merged = (
                info_recorder_merged.get("recorded_loss")[-1][-1]
                if info_recorder_merged is not None
                else 1e10
            )
            early_stopped = early_stopper.run_early_stopping_check(
                total_loss_merged
            )
            recorder.early_stop_count = early_stopper.wait_count
            # Save flow for the last frame
            if (
                epoch_index == recorder.num_epochs - 1 or early_stopped
            ) and recorder.num_cameras > 1:
                recorder.old_flow_loss_button = self._flow_loss
                self._flow_loss = True

            _tmp_flow_loss = self._flow_loss
            self._flow_loss = True
            recorder.only_pose_guidance = True
            # Spatial optimization step
            recorder = self._optmize_multicamera(
                recorder,
                epoch_index,
                spatial_optimizer,
                spatial_grad_norm,
                camera_transforms,
            )
            recorder.only_pose_guidance = False
            self._log_losses(recorder, epoch_index, "all")
            self._flow_loss = _tmp_flow_loss

            _tmp_pose_loss = self._pose_loss
            self._pose_loss = False
            # Temporal optimization step
            recorder = self._optmize_multicamera(
                recorder,
                epoch_index,
                temporal_optimizer,
                temporal_grad_norm,
                camera_transforms,
            )
            self._pose_loss = _tmp_pose_loss

            _info_recorder = recorder.info_recorder

            info_recorder_merged = InfoRecorder.merge_info_recorder(
                _info_recorder,
                ["recorded_loss"],
                recorder.alpha_two_strokes,
            )

            recorder.total_loss = info_recorder_merged.get("recorded_loss")[
                -1
            ][-1]

            self._log_losses(recorder, epoch_index, "all")
            pbar.set_postfix(loss=recorder.total_loss)
            if (
                not (
                    epoch_index % recorder.log_interval != 0
                    and epoch_index < recorder.num_epochs - 1
                )
                or early_stopped
            ):
                self._log_results(recorder, epoch_index)
            if early_stopped:
                print("early_stopped")
                break

        # Put back the flow configuration (important for consecutive work)
        if recorder.old_flow_loss_button is not None:
            self._flow_loss = recorder.old_flow_loss_button
        flow_idx = 0

        return (
            recorder.info_recorder,
            [x.cpu() for x in recorder.target_images],
            recorder.pred_flows[0, flow_idx].detach().cpu()
            if recorder.pred_flows is not None
            else None,
            recorder.gt_flows[0, flow_idx].detach().cpu()
            if recorder.gt_flows is not None
            else None,
        )

    def single_nograd_jaws(
        self,
        dataloader: DataLoader,
        target_image: torch.Tensor,
        initial_pose: torch.Tensor,
        num_sample_grad: int,
        num_epochs: int,
        log_interval: int,
        fp16: bool,
        focal_resize_factor: float = 1.0,
        regularizer_range: float = 2.0,
        regularizer_weight: float = 1.0,
        clip_size: float = 1e-5,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Computes the camera poses in NeRF coordinate system by optimizing them
        according to a reference image.

        :param dataloader: dataloader with intrinsic and dimension infos.
        :param target_image: reference image.
        :param initial_pose: (4, 4) camera transform matrix for initilization.
        :param num_sample_grad: render only some samples to run the inerf.
        :param num_epochs: number of iterations undertaken.
        :param log_interval: logging interval.
        :param fp16: mixed precision or not.
        :return: recorded images and poses obtained along the epochs.
        """
        recorder = Recorder(
            dict(
                clip_size=clip_size,
                fp16=fp16,
                blur_params=dict(kernel_size=1, sigma=0),
                log_interval=log_interval,
                focal_resize_factor=focal_resize_factor,
                regularizer_range=regularizer_range,
                regularizer_weight=regularizer_weight,
                num_epochs=num_epochs,
                use_guidance_map=False,
                num_sample_grad=num_sample_grad,
                dataloader=dataloader,
                target_images=[target_image],
                init_poses=[initial_pose],
                potential_poses=None,
            )
        )

        return self._run_jaws(recorder)

    def batch_fullgrad_jaws(
        self,
        dataloader: DataLoader,
        target_images: List[torch.Tensor],
        init_poses: List[torch.Tensor],
        blur_kernel: int,
        blur_sigma: int,
        num_epochs: int,
        log_interval: int,
        fp16: bool,
        vid_idx: int = 0,
        focal_resize_factor: float = 1.0,
        regularizer_range: float = 2.0,
        regularizer_weight: float = 1.0,
        clip_size: float = 1e-5,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """`num_sample_grad`=-1 + `use_guidance_map` is False -> all pixels."""
        recorder = Recorder(
            dict(
                clip_size=clip_size,
                fp16=fp16,
                blur_params=dict(kernel_size=blur_kernel, sigma=blur_sigma),
                log_interval=log_interval,
                focal_resize_factor=focal_resize_factor,
                regularizer_range=regularizer_range,
                regularizer_weight=regularizer_weight,
                num_epochs=num_epochs,
                use_guidance_map=True,
                num_sample_grad=-1,
                dataloader=dataloader,
                target_images=target_images,
                init_poses=init_poses,
                potential_poses=None,
                vid_idx=vid_idx,
            )
        )
        return self._run_jaws(recorder)

    def batch_mixedgrad_jaws(
        self,
        dataloader: DataLoader,
        target_images: List[torch.Tensor],
        init_poses: List[torch.Tensor],
        potential_poses: List[torch.Tensor],
        num_sample_grad: int,
        blur_kernel: int,
        blur_sigma: int,
        blur_pred: bool,
        num_epochs: int,
        log_interval: int,
        fp16: bool,
        flow_loss_type: str,
        guidance_type: str,
        clip_scheduler_indices: List[int],
        vid_idx: int = 0,
        alpha_two_strokes: float = 0.5,
        frozen_camera_indices: List[int] = [],
        focal_resize_factor: float = 1.0,
        regularizer_range: float = 2.0,
        regularizer_weight: float = 1.0,
        clip_size: float = 1e-3,
        use_guidance_map: bool = True,
        early_stop_num: int = 50,
        early_stop_delta: float = 5 * 1e-3,
        diff_temporal: bool = True,
        initial_time: float = 0.0,  # real time
        diff_focal: bool = False,
        allow_backward_t: bool = True,
        two_strokes: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """`num_sample_grad`=-1 + `use_guidance_map` is False -> all pixels."""
        recorder = Recorder(
            dict(
                clip_size=clip_size,
                fp16=fp16,
                blur_params=dict(kernel_size=blur_kernel, sigma=blur_sigma),
                blur_pred=blur_pred,
                log_interval=log_interval,
                focal_resize_factor=focal_resize_factor,
                regularizer_range=regularizer_range,
                regularizer_weight=regularizer_weight,
                num_epochs=num_epochs,
                use_guidance_map=use_guidance_map,
                num_sample_grad=num_sample_grad,
                dataloader=dataloader,
                target_images=target_images,
                potential_poses=potential_poses,
                vid_idx=vid_idx,
                frozen_camera_indices=frozen_camera_indices,
                diff_temporal=diff_temporal,
                diff_focal=diff_focal,
                allow_backward_t=allow_backward_t,
                initial_time=initial_time,
                init_poses=init_poses,
                early_stop_num=early_stop_num,
                early_stop_delta=early_stop_delta,
                flow_loss_type=flow_loss_type,
                clip_scheduler_indices=clip_scheduler_indices,
                alpha_two_strokes=alpha_two_strokes,
                guidance_type=guidance_type,
            )
        )

        if two_strokes:
            return self._run_jaws_two_strokes(recorder)
        else:
            return self._run_jaws(recorder)
