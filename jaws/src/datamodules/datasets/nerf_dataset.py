import cv2
import glob
import json
import os

import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import torch
from torch.utils.data import DataLoader

from utils.data_utils import nerf_matrix_to_ngp, rand_poses
from utils.nerf_utils import get_rays


class NeRFDataset:
    def __init__(
        self,
        num_rays,
        path,
        mode,
        preload,
        scale,
        bound,
        rand_pose,
        error_map=False,
        type="train",
        downscale=1,
        n_test=5,
        ind_calibration=False,
        aabb=None,
        use_heatmap=False,
    ):
        super().__init__()
        self.type = type  # train, val, test
        self.downscale = downscale
        self.root_path = path
        self.mode = mode  # colmap, blender, llff
        self.preload = preload  # preload data into GPU
        # camera radius scale to make sure camera are inside the bounding box
        self.scale = scale
        # bounding box half length, also the radius to random sample poses
        self.bound = bound
        self.aabb = aabb
        self.error_map = error_map
        self.training = self.type in ["train", "all"]
        self.num_rays = num_rays if self.training else -1
        self.ind_calibration = ind_calibration

        self.rand_pose = rand_pose

        # load nerf-compatible format data.
        if self.mode == "colmap":
            with open(os.path.join(self.root_path, "transforms.json"), "r") as f:
                transform = json.load(f)
        elif self.mode == "blender":
            # load all splits (train/valid/test), this is what instant-ngp in
            # fact does...
            if type == "all":
                transform_paths = glob.glob(os.path.join(self.root_path, "*.json"))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, "r") as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform["frames"].extend(tmp_transform["frames"])
            # only load one specified split
            elif type != "infer":
                with open(
                    os.path.join(self.root_path, f"transforms_{type}.json"),
                    "r",
                ) as f:
                    transform = json.load(f)
            elif type == "infer":
                print("infer mode")
                with open(
                    os.path.join(self.root_path, "transforms_train.json"),
                    "r",
                ) as f:
                    transform = json.load(f)
        else:
            raise NotImplementedError(f"unknown dataset mode: {self.mode}")
        if type == "infer":
            self.H = int(transform["h"])
            factor = 224 / self.H  # TODO: 224
            downscale = self.downscale = int(1.0 / factor)
        # load image size
        if "h" in transform and "w" in transform:
            self.H = int(transform["h"]) // downscale
            self.W = int(transform["w"]) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None

        # read images
        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d["file_path"])
        if type == "val" and self.mode == "blender":
            frames = frames[:1]  # shorten val time

        # for colmap, manually interpolate a test set.
        if self.mode == "colmap" and type == "test":
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            f0 = frames[0]
            pose0 = nerf_matrix_to_ngp(
                np.array(f0["transform_matrix"], dtype=np.float32),
                scale=self.scale,
            )  # [4, 4]
            pose1 = nerf_matrix_to_ngp(
                np.array(f1["transform_matrix"], dtype=np.float32),
                scale=self.scale,
            )  # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        if type == "infer":
            # generate exmaplary clips for transfer
            # choose two random poses, and interpolate between.
            cams = np.random.choice(frames, 4, replace=False)
            gen_poses = []
            for cam in cams:
                gen_poses.append(
                    nerf_matrix_to_ngp(
                        np.array(cam["transform_matrix"], dtype=np.float32),
                        scale=self.scale,
                    )
                )  # [4, 4]

            print("choosed cams: ", [frames.index(_cam) for _cam in cams])
            self.poses = []
            self.images = None
            for idx in range(len(gen_poses) - 1):
                pose0, pose1 = gen_poses[idx], gen_poses[idx + 1]
                rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                slerp = Slerp([0, 1], rots)
                for i in range(n_test + 1):
                    ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = slerp(ratio).as_matrix()
                    pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                    self.poses.append(pose)
        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == "colmap":
                if type == "train":
                    frames = frames[1:]
                elif type == "val":
                    frames = frames[:1]
                # else 'all': use all frames

            if self.ind_calibration:
                self.calibs = []
            self.poses = []
            self.images = []
            for f in frames:
                f_path = os.path.join(self.root_path, f["file_path"])
                # print(f_path)
                if self.mode == "blender" and (
                    f_path[-4:] != ".png" and f_path[-4:] != ".jpg"
                ):
                    f_path += ".png"  # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue

                pose = np.array(f["transform_matrix"], dtype=np.float32)  # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale)

                image = cv2.imread(
                    f_path, cv2.IMREAD_UNCHANGED
                )  # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                image = cv2.resize(
                    image, (self.W, self.H), interpolation=cv2.INTER_AREA
                )
                image = image.astype(np.float32) / 255  # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)

                # independent calibration: only in colmap type
                if not self.ind_calibration:
                    # early return
                    continue

                # load intrinsics for each one
                if "fl_x" in f or "fl_y" in f:
                    fl_x = (f["fl_x"] if "fl_x" in f else f["fl_y"]) / downscale
                    fl_y = (f["fl_y"] if "fl_y" in f else f["fl_x"]) / downscale
                elif "camera_angle_x" in f or "camera_angle_y" in f:
                    # blender in radians. already downscaled since we use H/W
                    fl_x = (
                        self.W / (2 * np.tan(f["camera_angle_x"] / 2))
                        if "camera_angle_x" in f
                        else None
                    )
                    fl_y = (
                        self.H / (2 * np.tan(f["camera_angle_y"] / 2))
                        if "camera_angle_y" in f
                        else None
                    )
                    if fl_x is None:
                        fl_x = fl_y
                    if fl_y is None:
                        fl_y = fl_x
                else:
                    raise RuntimeError(
                        "Independent calibration : Failed to load focal"
                        + " length, please check the transforms.json"
                    )
                # still assuming same H and W
                cx = (f["cx"] / downscale) if "cx" in f else (self.H / 2)
                cy = (f["cy"] / downscale) if "cy" in f else (self.W / 2)

                # [TODO] distortion?

                self.calibs.append(np.array([fl_x, fl_y, cx, cy]))
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0))  # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(
                np.stack(self.images, axis=0)
            )  # [N, H, W, C]

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()

        # initialize error_map
        if type == "train" and self.error_map:
            # [B, 128 * 128], flattened for easy indexing, fixed resolution...
            self.error_map = torch.ones(
                [self.images.shape[0], 128 * 128], dtype=torch.float
            )
        else:
            self.error_map = None

        if self.preload:
            self.poses = self.poses
            if self.images is not None:
                self.images = self.images
            if self.error_map is not None:
                self.error_map = self.error_map

        # load intrinsics
        if "fl_x" in transform or "fl_y" in transform:
            fl_x = (
                transform["fl_x"] if "fl_x" in transform else transform["fl_y"]
            ) / downscale
            fl_y = (
                transform["fl_y"] if "fl_y" in transform else transform["fl_x"]
            ) / downscale
        elif "camera_angle_x" in transform or "camera_angle_y" in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = (
                self.W / (2 * np.tan(transform["camera_angle_x"] / 2))
                if "camera_angle_x" in transform
                else None
            )
            fl_y = (
                self.H / (2 * np.tan(transform["camera_angle_y"] / 2))
                if "camera_angle_y" in transform
                else None
            )
            if fl_x is None:
                fl_x = fl_y
            if fl_y is None:
                fl_y = fl_x
        else:
            raise RuntimeError(
                "Failed to load focal length, please check the transforms.json"
            )

        cx = (transform["cx"] / downscale) if "cx" in transform else (self.H / 2)
        cy = (transform["cy"] / downscale) if "cy" in transform else (self.W / 2)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        ts = self.poses[:, :3, 3]
        self.pose_center = ts.mean(axis=0)
        self.pose_norm = ts.norm(dim=1).mean()

    def collate(self, index):
        B = len(index)  # always 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):
            poses = rand_poses(B, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(
                self.H * self.W / self.num_rays
            )  # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                "H": rH,
                "W": rW,
                "rays_o": rays["rays_o"],
                "rays_d": rays["rays_d"],
            }

        poses = self.poses[index]  # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        rays = get_rays(
            poses, self.intrinsics, self.H, self.W, self.num_rays, error_map
        )
        results = {
            "H": self.H,
            "W": self.W,
            "poses": poses,
            "intrinsics": self.intrinsics,
            "rays_o": rays["rays_o"],
            "rays_d": rays["rays_d"],
            "pose_center": self.pose_center,
            "pose_norm": self.pose_norm,
        }

        if self.images is not None:
            images = self.images[index]  # [B, H, W, 3/4]
            C = images.shape[-1]
            if self.training:
                images = torch.gather(
                    images.view(B, -1, C),
                    1,
                    torch.stack(C * [rays["inds"]], -1),
                )  # [B, N, 3/4]
            else:
                images.reshape(B, -1, C)
            results["images"] = images

        # need inds to update error_map
        if error_map is not None:
            results["index"] = index
            results["inds_coarse"] = rays["inds_coarse"]

        return results

    def collate_ind_calibs(self, index):
        assert self.ind_calibration is True

        B = len(index)  # always 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):
            poses = rand_poses(B, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(
                self.H * self.W / self.num_rays
            )  # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.calibs[index[0]] / s, rH, rW, -1)

            return {
                "H": rH,
                "W": rW,
                "rays_o": rays["rays_o"],
                "rays_d": rays["rays_d"],
            }

        poses = self.poses[index]  # [B, 4, 4]
        error_map = None if self.error_map is None else self.error_map[index]

        rays = get_rays(
            poses,
            self.calibs[index[0]],
            self.H,
            self.W,
            self.num_rays,
            error_map,
        )
        results = {
            "H": self.H,
            "W": self.W,
            "poses": poses,
            "intrinsics": self.calibs[index[0]],
            "rays_o": rays["rays_o"],
            "rays_d": rays["rays_d"],
            "pose_center": self.pose_center,
            "pose_norm": self.pose_norm,
        }

        if self.images is not None:
            images = self.images[index]  # [B, H, W, 3/4]
            C = images.shape[-1]
            if self.training:
                images = torch.gather(
                    images.view(B, -1, C),
                    1,
                    torch.stack(C * [rays["inds"]], -1),
                )  # [B, N, 3/4]
            else:
                images.reshape(B, -1, C)
            results["images"] = images

        # need inds to update error_map
        if error_map is not None:
            results["index"] = index
            results["inds_coarse"] = rays["inds_coarse"]

        return results

    def dataloader(self):
        size = len(self.poses)
        if self.rand_pose > 0:
            # index >= size means we use random pose.
            size += size // self.rand_pose

        # independent calibration
        if self.ind_calibration:
            loader = DataLoader(
                list(range(size)),
                batch_size=1,
                collate_fn=self.collate_ind_calibs,
                shuffle=self.training,
                num_workers=5,
            )
        else:
            loader = DataLoader(
                list(range(size)),
                batch_size=1,
                collate_fn=self.collate,
                shuffle=self.training,
                num_workers=5,
            )
        # an ugly fix... we need to access error_map & poses in trainer.
        loader._data = self

        return loader
