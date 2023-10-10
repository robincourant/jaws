from typing import Sequence, Dict

import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
import rich.tree
import rich.syntax
import cv2
import os.path as osp
import imageio
import trimesh
import torch

from utils.file_utils import save_pickle


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "compnode",
        "model",
        "datamodule",
        "jaws",
        "xp_name",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """
    Adapted from: https://github.com/ashleve/lightning-hydra-template.
    Prints content of DictConfig using Rich library and its tree structure.

    :param config: configuration composed by Hydra.
    :param fields: determines which main fields from config will be printed and
        in what order.
    :param resolve: whether to resolve reference fields of DictConfig.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


def divide(a: np.array, b: np.array) -> np.array:
    """Perform array element-wise division, 0 when dividing by 0."""
    res = np.divide(
        a,
        b,
        out=np.zeros_like(a, dtype=np.float64),
        where=(b != 0),
    )
    return res


def save_video(
    chunk, filename, over_write=False, is_resize=False, size=(224, 224)
):
    # for idx, chunk in enumerate(lchunks):
    if osp.exists(filename) and not over_write:
        return
    out = cv2.VideoWriter(
        filename + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5, (224, 224)
    )
    for frm in chunk:
        # print("np.shape frm: ", np.shape(frm))
        if np.shape(frm) != (size[0], size[1], 3):
            frm = cv2.resize(frm, size)
        out.write(frm)
    out.release()


def save_nerf_img(_frm, filename):
    # to8b
    frm = (_frm.detach().cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite(filename, frm)


def save_gif(imgs, filename, fps=5, is_resize=False, size=(224, 224)):
    final_imgs = []
    for _frm in imgs:
        # to8b
        frm = (_frm.detach().cpu().numpy() * 255).astype(np.uint8)
        if np.shape(frm) != (size[0], size[1], 3) and is_resize:
            final_imgs.append(cv2.resize(frm, size))
        else:
            final_imgs.append(frm)
    imageio.mimwrite(filename, final_imgs, fps=fps)


def save_traj(poses: torch.Tensor, filename: str, saving_format="pkl"):
    if saving_format == "pkl":
        dict_save = {}
        for idx, pose in enumerate(poses):
            dict_save[idx] = pose.cpu().numpy()
        save_pickle(dict_save, filename + "." + saving_format)


def draw_axis(img, R, t, K):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32(
        [[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]
    ).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(
        img,
        tuple(axisPoints[3].ravel()),
        tuple(axisPoints[0].ravel()),
        (255, 0, 0),
        3,
    )
    img = cv2.line(
        img,
        tuple(axisPoints[3].ravel()),
        tuple(axisPoints[1].ravel()),
        (0, 255, 0),
        3,
    )
    img = cv2.line(
        img,
        tuple(axisPoints[3].ravel()),
        tuple(axisPoints[2].ravel()),
        (0, 0, 255),
        3,
    )
    return img


def render_cam_pose(pose, intrinsics, render_pose, img):
    inv_render_pose = np.linalg.inv(render_pose)
    pose_2_render = pose @ inv_render_pose
    t = pose_2_render[:3, 3]
    R = pose_2_render[:3, :3]

    K = np.zeros((3, 3))
    K[0, 0] = intrinsics[0]  # fx
    K[1, 1] = intrinsics[1]  # fy
    K[0, 2] = intrinsics[2]  # cx
    K[1, 2] = intrinsics[3]  # cy
    K[2, 2] = 1

    draw_axis(img, R, t, K)

    # uvz = K@pose_2_render
    # x = uvz[0]/uvz[2]
    # y = uvz[1]/uvz[2]


def visualize_poses(poses, file_name, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    # sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array(
            [
                [pos, a],
                [pos, b],
                [pos, c],
                [pos, d],
                [a, b],
                [b, c],
                [c, d],
                [d, a],
            ]
        )
        segs = trimesh.load_path(segs)
        objects.append(segs)
        scene = trimesh.Scene(objects)
        png = scene.save_image(resolution=[800, 800], visible=True)
        with open(file_name, "wb") as f:
            f.write(png)
            f.close()


def cfg2dict(cfg: DictConfig) -> Dict:
    """
    Recursively convert OmegaConf to vanilla dict
    :param cfg:
    :return:
    """
    cfg_dict = {}
    for k, v in cfg.items():
        if type(v) == DictConfig:
            cfg_dict[k] = cfg2dict(v)
        else:
            cfg_dict[k] = v
    return cfg_dict
