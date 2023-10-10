from omegaconf import DictConfig
import os
import os.path as osp
import sys

import torch
import numpy as np

from utils.file_utils import create_dir, load_pickle, save_pickle
from jaws.src.models.modules.nerf_factory import create_nerf_model
from utils.camera_utils import PoseInterpolator
from utils.image_utils import save_gif, save_torch_image, save_poses_kitti
from tqdm import tqdm


def render(config: DictConfig):
    sys.path.append(osp.join(".", "lib", "torch_ngp"))
    model = create_nerf_model(config)

    # Initialize trainer
    checkpoint_dir = osp.join(config.result_dir, "checkpoints")
    if not osp.exists(checkpoint_dir):
        create_dir(checkpoint_dir)
    if config.model.ckpt == "latest":
        checkpoint_list = sorted(os.listdir(checkpoint_dir))
        if len(checkpoint_list) > 0:
            checkpoint_path = osp.join(checkpoint_dir, checkpoint_list[-1])
        else:
            checkpoint_path = None
    else:
        checkpoint_path = config.model.ckpt

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    device = "cuda" if config.compnode.num_gpus > 0 else "cpu"
    model.to(device)

    # intrinsic parameters - constant during interpolating
    params_path = osp.join(config.render_target_dir, "params.pkl")
    params = load_pickle(params_path)

    # parameters can be interpolated
    focals_path = osp.join(config.render_target_dir, "focals.pkl")
    times_path = osp.join(config.render_target_dir, "times.pkl")
    poses_path = osp.join(config.render_target_dir, "poses.pkl")
    focals = load_pickle(focals_path)
    times = load_pickle(times_path)
    poses = load_pickle(poses_path)

    # interpolation here before rendering:
    (
        focals,
        times,
        poses,
    ) = PoseInterpolator.inpterpolate_render_sequence_from_keyframes_cubic(
        focals=focals,
        times=times,
        poses=poses,
        frm_num=config.render_frame_num,
    )

    res_factor = 720.0 / params[0]["H"]
    frames = []
    H = int(params[0]["H"] * res_factor)  # 224 -> control image
    W = int(params[0]["W"] * res_factor)
    intrinsics = params[0]["intrinsics"]
    scale_factor = H / (intrinsics[3] * 2)
    intrinsics = intrinsics * scale_factor
    intrinsics[2] = W / 2
    intrinsics[3] = H / 2

    # high resolution
    for camera_index in tqdm(range(len(poses))):
        focal = focals[camera_index]
        time = torch.tensor([[times[camera_index]]]).to(device)
        current_intrinsics = np.copy(intrinsics)
        current_intrinsics[:2] = intrinsics[:2] * focal
        pose = poses[camera_index].to(device)
        if config.dynamic:
            frames.append(
                (
                    model.render(pose, time, current_intrinsics, H, W).cpu()
                    * 255.0
                )
                .numpy()
                .astype(np.uint8)
            )
        else:
            frames.append(
                (model.render(pose, current_intrinsics, H, W).cpu() * 255.0)
                .numpy()
                .astype(np.uint8)
            )
    save_gif(
        frames, osp.join(config.render_target_dir, "interpolated.gif"), 25
    )
    save_pickle(
        poses, osp.join(config.render_target_dir, "interpolated_poses.pkl")
    )
    save_pickle(
        focals, osp.join(config.render_target_dir, "interpolated_focals.pkl")
    )
    save_poses_kitti(
        poses,
        config.render_target_dir,
        "interpolated_poses_kitti.csv",
    )

    # save_pickle(frames, config.render_target_dir, "frames.pkl")
    # save fig here too
