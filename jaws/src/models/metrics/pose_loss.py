from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from typing import Any, Dict, List

from kornia.geometry.transform import rotate
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms

import lib.LitePose._init_paths  # noqa
import models  # noqa
from arch_manager import ArchManager
from config import cfg
from config import check_config
from config import update_config_dict
from core.inference import aggregate_results
from core.inference import get_multi_stage_outputs
from fp16_utils.fp16util import network_to_half
from geomloss import SamplesLoss
from utils.loss_utils import spatial_soft_argmax2d_with_misc
from utils.transforms import get_base_scale_size
from utils.utils import create_logger
from utils.utils import get_model_summary

# #######################################################################################
litepose_dir = os.path.join(os.getcwd(), "lib", "LitePose")
DEFAULT_ARGS_S = {
    "cfg": f"{litepose_dir}/experiments/coco/mobilenet/mobile.yaml",
    "superconfig": f"{litepose_dir}/mobile_configs/search-S.json",
    "opts": [],
    "model_file": f"{litepose_dir}/ckpt/LitePose-Auto-S.pth.tar",
}

DEFAULT_ARGS_S_CROWD = {
    "cfg": f"{litepose_dir}/experiments/crowd_pose/mobilenet/mobile.yaml",
    "superconfig": f"{litepose_dir}/mobile_configs/search-S.json",
    "opts": [],
    "model_file": f"{litepose_dir}/ckpt/CrowdLitePose-Auto-S.tar",
}

DEFAULT_ARGS_XS = {
    "cfg": f"{litepose_dir}/experiments/crowd_pose/mobilenet/mobile.yaml",
    "superconfig": f"{litepose_dir}/mobile_configs/search-XS.json",
    "opts": [],
    "model_file": f"{litepose_dir}/ckpt/LitePose-Auto-XS.pth.tar",
}
DEFAULT_ARGS = DEFAULT_ARGS_S
# #######################################################################################


def earth_mover_distance(y_true, y_pred):
    return torch.mean(
        torch.square(
            (torch.cumsum(y_true, dim=-1)) - (torch.cumsum(y_pred, dim=-1))
        ),
        dim=-1,
    )


def EMD_2D(y_true, y_pred):
    return earth_mover_distance(y_true, y_pred).mean(
        axis=2
    ) + earth_mover_distance(
        y_true.permute([0, 1, 3, 2]),
        y_pred.permute([0, 1, 3, 2]),
    ).mean(
        axis=2
    )


def EMD_2D_r(y_true, y_pred, rot_num=8):
    loss = EMD_2D(y_true, y_pred)
    B, _, _, _ = y_true.shape
    for i in range(1, rot_num - 1):
        angle = torch.full((B,), i * (180.0 / rot_num), device=y_true.device)
        loss += earth_mover_distance(
            rotate(y_true, angle), rotate(y_pred, angle)
        ).mean(axis=2)
    return loss


def EMD_2D_sq(y_true, y_pred):
    return torch.sqrt(
        earth_mover_distance(y_true, y_pred) ** 2
        + earth_mover_distance(
            y_true.permute([0, 1, 3, 2]),
            y_pred.permute([0, 1, 3, 2]),
        )
        ** 2
    )


def EMD_2Dm(y_true, y_pred):
    return (
        earth_mover_distance(y_true, y_pred)
        + earth_mover_distance(
            y_true.permute([0, 1, 3, 2]),
            y_pred.permute([0, 1, 3, 2]),
        )
    ).mean()


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        torch.div(index, dim, rounding_mode="floor")
    return torch.vstack(out).permute(1, 0).type(torch.FloatTensor)


class LitePoseLoss(torch.nn.Module):
    def __init__(
        self,
        args_dict: Dict[str, Any],
        log_enable: bool = False,
        fp16: bool = False,
        heatmap_loss: bool = False,
        euclidean_loss: bool = False,
        only_update_y: bool = False,
        save_viz: bool = False,
        temp_filter: bool = True,
        temp_filter_length: int = 2,
        key_joints: List[int] = list(range(17)),
    ):
        super(LitePoseLoss, self).__init__()
        update_config_dict(cfg, args_dict)

        check_config(cfg)

        # change the resolution according to config
        fixed_arch = None
        if args_dict["superconfig"] is not None:
            with open(args_dict["superconfig"], "r") as f:
                fixed_arch = json.load(f)
            cfg.defrost()
            reso = fixed_arch["img_size"]
            cfg.DATASET.INPUT_SIZE = reso
            cfg.DATASET.OUTPUT_SIZE = [reso // 4, reso // 2]
            cfg.freeze()

        if log_enable:
            logger, _, _ = create_logger(cfg, args_dict["cfg"], "valid")
            logger.info(cfg)

        # cudnn related setting
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
        self.pool = torch.nn.MaxPool2d(
            cfg.TEST.NMS_KERNEL, 1, cfg.TEST.NMS_PADDING
        )
        if (
            cfg.MODEL.NAME == "pose_mobilenet"
            or cfg.MODEL.NAME == "pose_simplenet"
        ):
            arch_manager = ArchManager(cfg)
            cfg_arch = arch_manager.fixed_sample()
            if fixed_arch is not None:
                cfg_arch = fixed_arch

            self.model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(
                cfg, is_train=True, cfg_arch=cfg_arch
            )
        else:
            self.model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(
                cfg, is_train=True
            )

        # set super config
        if cfg.MODEL.NAME == "pose_supermobilenet":
            self.model.arch_manager.is_search = True
            if args_dict["superconfig"] is not None:
                with open(args_dict["superconfig"], "r") as f:
                    self.model.arch_manager.search_arch = json.load(f)
            else:
                self.model.arch_manager.search_arch = (
                    self.model.arch_manager.fixed_sample()
                )

        dump_input = torch.rand(
            (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
        )
        if log_enable:
            logger.info(
                get_model_summary(
                    cfg.DATASET.INPUT_SIZE, self.model, dump_input
                )
            )
            logger.info("=> loading model from {}".format(cfg.TEST.MODEL_FILE))
        if fp16:
            self.model = network_to_half(self.model)

        # loading model
        self.model.load_state_dict(
            torch.load(cfg.TEST.MODEL_FILE), strict=True
        )
        self.model = torch.nn.DataParallel(
            self.model, device_ids=cfg.GPUS
        ).cuda()

        if cfg.MODEL.NAME == "pose_hourglass":
            self.transforms = torchvision.transforms.Compose([])
        else:
            self.transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        self.model.eval()
        self.heatmapx = None
        self.heatmapy = None
        self.only_update_y = only_update_y
        self.save_viz = save_viz
        self.heatmap_loss = heatmap_loss
        self.euclidean_loss = euclidean_loss
        assert heatmap_loss ^ euclidean_loss
        self.temp_filter = temp_filter
        self.prev_heatmapx = []
        self.prev_heatmapy = []
        self.temp_filter_length = temp_filter_length
        self.key_joints = key_joints
        self.key_joints = []
        self.sample_loss = SamplesLoss("sinkhorn", p=1, blur=0.01)
        self.rot_num = 4

    def pre_rotate(self, heatx, heaty):
        self.heatx_rot = []
        self.heaty_rot = []
        B, _, _, _ = heatx.shape
        for i in range(0, self.rot_num):
            angle = torch.full(
                (B,),
                i * (180.0 / self.rot_num),
                device=heatx.device,
                dtype=heatx.dtype,
            )
            self.heatx_rot.append(rotate(heatx, angle))
            self.heaty_rot.append(rotate(heaty, angle))

    def sample_points(self, h, n=5000):
        v, i = torch.topk(h.flatten(), n)
        return unravel_index(i, h.shape)

    def EMD_sinkhorn(self, heatx, heaty):
        B, C, _, _ = heatx.shape

        losses = [[] for i in range(B)]
        for b in range(B):
            _losses = [
                self.sample_loss(self.sample_points(x), self.sample_points(y))
                for x, y in zip(heatx[b], heaty[b])
            ]
            _losses = torch.stack(_losses)
            losses[b] = _losses.mean()
        losses = torch.stack(losses)
        return losses

    def EMD_approx(self, heatx, heaty):
        return EMD_2D_r(heatx, heaty)

    def EMD_approx_inter(self, x, y):
        assert len(self.heaty_rot) > 0
        loss = 0
        for y_true, y_pred in zip(self.heatx_rot, self.heaty_rot):
            loss += earth_mover_distance(y_true, y_pred).mean(axis=2)
        return loss

    def EMD_approx_intra(self, heatmap, b, c0, c1):
        if heatmap == "x":
            h_rot = self.heatx_rot
        elif heatmap == "y":
            h_rot = self.heaty_rot
        loss = 0
        for h in h_rot:
            loss += earth_mover_distance(h[b][c0], h[b][c1]).mean(axis=0)
        return loss

    def EMD(self, heatx, heaty):
        return self.EMD_approx(heatx, heaty)

    def heatmap(self, img):
        # (B, C, H, W)
        if img.shape[1] != 3:
            img = img.permute(0, 3, 1, 2)
        B, C, H, W = img.shape

        # expected : [H, W, C]
        base_size, center, scale = get_base_scale_size(
            (W, H), cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
        )

        img = torch.nn.functional.interpolate(
            img, size=base_size, mode="bilinear"
        )
        final_heatmaps = None
        tags_list = []

        assert len(cfg.TEST.SCALE_FACTOR) == 1
        s = cfg.TEST.SCALE_FACTOR[0]

        outputs, heatmaps, tags = get_multi_stage_outputs(
            cfg,
            self.model,
            img,
            cfg.TEST.FLIP_TEST,
            cfg.TEST.PROJECT2IMAGE,
            base_size,
        )
        final_heatmaps, tags_list = aggregate_results(
            cfg, s, final_heatmaps, tags_list, heatmaps, tags
        )

        old_size = (H, W)
        final_heatmaps = torch.nn.functional.interpolate(
            final_heatmaps, size=old_size
        )
        if len(self.key_joints) != 0:
            final_heatmaps = final_heatmaps[:, self.key_joints, :, :]

        return final_heatmaps

    def set_temporal_filter(self, temp_filter):
        self.temp_filter = temp_filter
        self.reset_temporal_filter()

    def reset_temporal_filter(self):
        self.prev_heatmapy = []

    def temporal_confidence_y(self, lconfx):
        # No grad computing on this part
        with torch.no_grad():
            if self.temp_filter is False:
                return torch.ones_like(lconfx)
            if len(self.prev_heatmapy) < self.temp_filter_length:
                self.prev_heatmapy.append(self.heatmapy)
                return torch.ones_like(lconfx)

            # Add the new
            self.prev_heatmapy.append(self.heatmapy)
            self.prev_heatmapy.pop(0)
            assert len(self.prev_heatmapy) == self.temp_filter_length

            EMD_displacements = []

            # Compute EMD displacement
            for h_idx in range(len(self.prev_heatmapy) - 1):
                EMD_displacements.append(
                    self.EMD(
                        self.prev_heatmapy[h_idx],
                        self.prev_heatmapy[h_idx + 1],
                    )
                )

            # Compute average channel-wise disp
            # B, 14, mean except last one
            mean_displacements = torch.stack(EMD_displacements[:-1]).mean(
                axis=0
            )
            conf_temp = 1 / (
                1
                + torch.abs(mean_displacements - EMD_displacements[-1])
                / mean_displacements
            )

            return conf_temp

    def temporal_blending_y(self, blending_alpha=0.5):
        with torch.no_grad():
            if self.temp_filter is False:
                return
            if len(self.prev_heatmapy) < self.temp_filter_length:
                self.prev_heatmapy.append(self.heatmapy.detach())
                return

            self.prev_heatmapy.append(self.heatmapy.detach())
            self.prev_heatmapy.pop(0)
            assert len(self.prev_heatmapy) == self.temp_filter_length

            # Compute mean of prev_heatmap
            mean_hty = torch.stack(self.prev_heatmapy).mean(axis=0)

        self.heatmapy = (
            blending_alpha * mean_hty + (1.0 - blending_alpha) * self.heatmapy
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0
        # [17, W, H]
        if not self.only_update_y:
            self.heatmapx = self.pool(self.heatmap(x))

        self.heatmapy = self.pool(self.heatmap(y))
        if self.temp_filter:
            self.temporal_blending_y()

        B, C, H, W = self.heatmapx.shape
        self.coords_x = [[] for _ in range(B)]
        self.coords_y = [[] for _ in range(B)]
        temp = 200
        lconfx = [[] for _ in range(B)]
        lconfy = [[] for _ in range(B)]
        lheatmaps_x = [[] for _ in range(B)]
        lheatmaps_y = [[] for _ in range(B)]
        leuclidean = [[] for _ in range(B)]

        if self.heatmap_loss:
            innner_distance_matrix_x = torch.zeros(
                (B, C, C),
                dtype=self.heatmapx.dtype,
                device=self.heatmapx.device,
            )
            innner_distance_matrix_y = torch.zeros(
                (B, C, C),
                dtype=self.heatmapx.dtype,
                device=self.heatmapx.device,
            )
        if self.euclidean_loss:
            innner_distance_matrix_x = torch.zeros(
                (B, C, C, 2),
                dtype=self.heatmapx.dtype,
                device=self.heatmapx.device,
            )
            innner_distance_matrix_y = torch.zeros(
                (B, C, C, 2),
                dtype=self.heatmapx.dtype,
                device=self.heatmapx.device,
            )

        for b in range(B):
            for channel in range(C):  # 17
                hmap_1c_x = self.heatmapx[b, channel].unsqueeze(0).unsqueeze(0)

                (
                    coord_max_x,
                    soft_x,
                    conf_x,
                ) = spatial_soft_argmax2d_with_misc(
                    hmap_1c_x,
                    normalized_coordinates=True,
                    temperature=torch.tensor(temp),
                )

                hmap_1c_y = self.heatmapy[b, channel].unsqueeze(0).unsqueeze(0)
                (
                    coord_max_y,
                    soft_y,
                    conf_y,
                ) = spatial_soft_argmax2d_with_misc(
                    hmap_1c_y,
                    normalized_coordinates=True,
                    temperature=torch.tensor(temp),
                )

                lconfx[b].append(conf_x)
                lconfy[b].append(conf_y)
                # Save heatmap softmaxed
                lheatmaps_x[b].append(soft_x.squeeze(0).squeeze(0))
                lheatmaps_y[b].append(soft_y.squeeze(0).squeeze(0))
                self.coords_x[b].append(coord_max_x.squeeze(0))
                self.coords_y[b].append(coord_max_y.squeeze(0))
                leuclidean[b].append(
                    torch.nn.functional.mse_loss(coord_max_x, coord_max_y)
                )

        lconfx = torch.stack([torch.stack(confx) for confx in lconfx])
        lconfy = torch.stack([torch.stack(confy) for confy in lconfy])

        # Energy conservation -> Wassertein
        lconfx = (lconfx / torch.sum(lconfx)) * C
        lconfy = (lconfy / torch.sum(lconfy)) * C

        for _b in range(B):
            for _c in range(C):
                leuclidean[_b][_c] *= lconfx[_b, _c]  # list

        self.pre_rotate(self.heatmapx, self.heatmapy)

        heatmap_loss = (
            self.EMD_approx_inter(self.heatmapx, self.heatmapy) * lconfx
        ).mean()
        euclidean_loss = torch.stack(
            [torch.stack(leuc) for leuc in leuclidean]
        ).mean()
        for b in range(B):
            for c_i0 in range(C):
                for c_i1 in range(c_i0):
                    if self.heatmap_loss:
                        innner_distance_matrix_x[b, c_i0, c_i1] = (
                            self.EMD_approx_intra("x", b, c_i0, c_i1)
                            * lconfx[_b, _c]
                        ).mean()
                        innner_distance_matrix_y[b, c_i0, c_i1] = (
                            self.EMD_approx_intra("y", b, c_i0, c_i1)
                            * lconfx[_b, _c]
                        ).mean()
                    if self.euclidean_loss:
                        innner_distance_matrix_x[b, c_i0, c_i1] = (
                            self.coords_x[b][c_i0] - self.coords_x[b][c_i1]
                        ) * lconfx[_b, _c]
                        innner_distance_matrix_y[b, c_i0, c_i1] = (
                            self.coords_y[b][c_i0] - self.coords_y[b][c_i1]
                        ) * lconfx[_b, _c]

        inner_distance_loss = torch.nn.functional.mse_loss(
            innner_distance_matrix_x, innner_distance_matrix_y
        )
        _alpha = 0.7
        _beta = 1 - _alpha
        if self.heatmap_loss:
            loss = _beta * inner_distance_loss + _alpha * heatmap_loss
        if self.euclidean_loss:
            loss = 10 * inner_distance_loss + euclidean_loss
        scale = 1

        return loss * scale
