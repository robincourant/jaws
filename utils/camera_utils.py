import torch
from typing import List, Tuple

from scipy.spatial.transform import Rotation as rot
import random
import numpy as np
import torch.nn as nn
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
import math


def vec2ss_matrix(vector):  # vector to skewsym. matrix
    ss_matrix = torch.zeros((3, 3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix


def pose_distance(pose1: torch.Tensor, pose2: torch.Tensor) -> Tuple[float]:
    # distance R
    r1 = rot.from_matrix(pose1[0].numpy()[:3, :3])
    r2 = rot.from_matrix(pose2[0].numpy()[:3, :3])
    delta_r_rad = (r1 * r2.inv()).magnitude()

    # distance T
    t1 = pose1[0].numpy()[:3, 3]
    t2 = pose2[0].numpy()[:3, 3]
    delta_t = np.linalg.norm(t1 - t2)

    return delta_r_rad, delta_t


class camera_transf(nn.Module):
    def __init__(self):
        super(camera_transf, self).__init__()
        self.w = nn.Parameter(torch.normal(0.0, 1e-6, size=(3,)))
        self.v = nn.Parameter(torch.normal(0.0, 1e-6, size=(3,)))
        self.theta = nn.Parameter(torch.normal(0.0, 1e-6, size=()))

    def forward(self, x):
        exp_i = torch.zeros((4, 4))
        w_skewsym = vec2ss_matrix(self.w)
        exp_i[:3, :3] = (
            torch.eye(3)
            + torch.sin(self.theta) * w_skewsym
            + (1 - torch.cos(self.theta)) * torch.matmul(w_skewsym, w_skewsym)
        )
        exp_i[:3, 3] = torch.matmul(
            torch.eye(3) * self.theta
            + (1 - torch.cos(self.theta)) * w_skewsym
            + (self.theta - torch.sin(self.theta)) * torch.matmul(w_skewsym, w_skewsym),
            self.v,
        )
        exp_i[3, 3] = 1.0
        T_i = torch.matmul(exp_i, x)
        return T_i


img2mse = lambda x, y: torch.mean((x - y) ** 2)


def get_sequence(images: List[torch.Tensor]) -> torch.Tensor:
    mock_sequence = torch.vstack([x.unsqueeze(0) for x in images]).permute(0, 3, 1, 2)
    return mock_sequence


class PoseInterpolator:
    @staticmethod
    def get_interpolated_sequence(
        poses: List[torch.Tensor],
        sequence_length: int = 17,
        type: str = "repeat",
    ) -> torch.Tensor:
        """
        Generate a 16-frames long mock sequence given less images.
        Example of sequence pattern given 4 images: 11111|22222|33333|44444
        """
        # Calculate the number of duplicates per image
        num_images = len(poses)
        num_duplicates = int(np.ceil(sequence_length / num_images))

        # Generate the mock sequence
        if type == "repeat":
            interpolated_poses = torch.vstack(
                [x[None].repeat((num_duplicates, 1, 1, 1)) for x in poses]
            )

        # apply grad
        interpolated_poses_lut = []
        list_intp_poses = []
        ctr = 0
        for idx, pose in enumerate(interpolated_poses):
            if idx % num_duplicates != 0:
                list_intp_poses.append(pose.detach())
            else:
                list_intp_poses.append(pose)
                ctr += 1
            interpolated_poses_lut.append(ctr - 1)
        # if type == "linear_t":
        return list_intp_poses, interpolated_poses_lut

    @staticmethod
    def get_interpolated_sequence_lerp(
        poses: List[torch.Tensor],
        ratios: torch.Tensor,
        sequence_length: int = 17,
        type: str = "lerp2ends",
    ) -> torch.Tensor:
        # Calculate the number of duplicates per image
        num_images = len(poses)
        num_duplicates = int(np.ceil(sequence_length / num_images))

        list_intp_poses = []
        intp_lut = []
        if type == "lerp2ends":
            for idx in range(sequence_length):
                # ratio = idx / float(sequence_length)
                pose = torch.tensor(poses[0])
                pose[0][:3, 3] = torch.lerp(
                    poses[0][0][:3, 3], poses[-1][0][:3, 3], ratios[idx]
                )
                list_intp_poses.append(pose)

                # detach grad
                # if idx != min(with_grad) and idx != max(with_grad):
                # list_intp_poses[-1] = list_intp_poses[-1].detach()

                intp_lut.append(int(idx / num_duplicates))

        return list_intp_poses, intp_lut

    @staticmethod
    def intpl_between(pose0, pose1, num_intpl):
        ret_poses = []
        for idx in range(num_intpl):
            ratio = idx / float(num_intpl)
            pose = pose0
            pose[0][:3, 3] = torch.lerp(pose0[0][:3, 3], pose1[0][:3, 3], ratio)
            ret_poses.append(pose)
            if idx != 0 and idx != num_intpl - 1:
                ret_poses[-1] = ret_poses[-1].detach()
        return ret_poses

    @staticmethod
    def get_interpolated_sequence_lerp_arb(
        poses: List[torch.Tensor],
        pose_idxes: List[int],
        sequence_length: int = 17,
        type: str = "lerp",
    ) -> torch.Tensor:
        if max(pose_idxes) != sequence_length:
            return None

        intp_poses = None
        intp_lut = []

        for idx_p in range(len(pose_idxes) - 1):
            pose_idx0 = pose_idxes[idx_p]
            pose_idx1 = pose_idxes[idx_p + 1]
            num2intpl = pose_idx1 - pose_idx0
            ret_poses = PoseInterpolator.intpl_between(
                poses[pose_idx0],
                poses[pose_idx1],
                num2intpl,
            )

            if intp_poses is None:
                intp_poses = ret_poses
            else:
                intp_poses = (
                    intp_poses + ret_poses[1:]
                )  # first element is the same of the last previous

        # make sure
        assert sequence_length == len(intp_poses)

        idx_p = 0
        for intp_idx in range(sequence_length):
            if intp_idx < pose_idxes[idx_p]:
                intp_lut.append(pose_idxes[idx_p])
                idx_p += 1

        return intp_poses, intp_lut

    @staticmethod
    def interpolate_between_2(
        pose0: torch.Tensor,
        pose1: torch.Tensor,
        ratio: float = None,
        check_rot_distance: bool = False,
    ):
        if check_rot_distance:
            delta_r_rad, delta_t = pose_distance(pose0, pose1)
            if np.rad2deg(delta_r_rad) > 120:
                return None
        rots = rot.from_matrix(
            np.stack(
                [
                    # neg rotation
                    # -torch.eye(3) @ pose0[0][:3, :3].numpy(),
                    # -torch.eye(3) @ pose1[0][:3, :3].numpy(),
                    pose0[0][:3, :3].numpy(),
                    pose1[0][:3, :3].numpy(),
                ]
            )
        )
        slerp = Slerp([0, 1], rots)
        if ratio is None:
            ratio = random.uniform(0, 1)

        pose = pose0.clone()
        pose[0][:3, :3] = torch.tensor(
            # -np.eye(3) @ slerp(ratio).as_matrix()
            slerp(ratio).as_matrix()
        ).unsqueeze(0)
        pose[0][:3, 3] = torch.lerp(pose0[0][:3, 3], pose1[0][:3, 3], ratio)
        return pose

    @staticmethod
    def inpterpolate_render_sequence_from_keyframes(
        focals,
        times,
        poses,
        n=5,
    ):
        assert len(focals) == len(times) == len(poses)

        interplated_times = []
        interplated_poses = []
        interplated_focals = []
        # loop till -1
        for idx in range(len(times) - 1):
            poses_ti = []
            focals_ti = []
            for interpolate_ratio in np.arange(0.0, 1.0, 1.0 / n):
                # pose slerp R + lerp t
                poses_ti.append(
                    PoseInterpolator.interpolate_between_2(
                        poses[idx], poses[idx + 1], interpolate_ratio
                    )
                )
                # f = f0 + r*(f1-f0) # lerp
                focals_ti.append(
                    float(
                        focals[idx]
                        + interpolate_ratio * (focals[idx + 1] - focals[idx])
                    )
                )
            interplated_focals = interplated_focals + focals_ti
            interplated_poses = interplated_poses + poses_ti
            interplated_times = np.linspace(
                min(times),
                max(times),
                len(interplated_poses),
                dtype=np.float32,
            )
        return interplated_focals, interplated_times, interplated_poses

    @staticmethod
    def inpterpolate_render_sequence_from_keyframes_cubic(
        focals,
        times,
        poses,
        frm_num=0,
    ):
        assert len(focals) == len(times) == len(poses)
        interplated_poses = []
        n = float(len(times) - 1) / (frm_num)

        interpolate_ratio_total = 0
        interpolate_ratio = 0
        idx = 0
        # loop till -1
        for _ in range(frm_num):
            poses_ti = []
            print(idx)
            # for interpolate_ratio in np.arange(0.0, 1.0, 1.0 / n):
            # pose slerp R + lerp t
            poses_ti.append(
                PoseInterpolator.interpolate_between_2(
                    poses[idx], poses[idx + 1], interpolate_ratio
                )
            )
            interpolate_ratio_total += n
            interpolate_ratio = interpolate_ratio_total - math.floor(
                interpolate_ratio_total
            )
            idx = math.floor(interpolate_ratio_total)
            interplated_poses = interplated_poses + poses_ti

        x = np.linspace(0, 1, len(times))
        f_focal = interp1d(x, focals, kind="cubic")
        t0 = np.array([pose[0, 0, 3].item() for pose in poses])
        t1 = np.array([pose[0, 1, 3].item() for pose in poses])
        t2 = np.array([pose[0, 2, 3].item() for pose in poses])
        f_t0 = interp1d(x, t0, kind="cubic")
        f_t1 = interp1d(x, t1, kind="cubic")
        f_t2 = interp1d(x, t2, kind="cubic")
        f_t = interp1d(x, times, kind="cubic")

        interplated_x = np.linspace(0, 1, len(interplated_poses)).tolist()
        interplated_focals = f_focal(interplated_x).tolist()
        interplated_times = f_t(interplated_x).tolist()

        for idx_t, time in enumerate(interplated_x):
            # print("time: ", time)
            interplated_poses[idx_t][0, 0, 3] = torch.tensor(f_t0(time).item())
            # print("f_t0:", f_t0(time).item())
            # print("f_t0_poses:", interplated_poses[idx_t][0, 0, 3])
            interplated_poses[idx_t][0, 1, 3] = torch.tensor(f_t1(time).item())
            interplated_poses[idx_t][0, 2, 3] = torch.tensor(f_t2(time).item())
        return interplated_focals, interplated_times, interplated_poses


class CameraPoseGenerator:
    @staticmethod
    def generate_random_pose_t(
        init_pose: torch.Tensor,
        sigma: float = 1e-2,
        rand_type: str = "Uniform",
        device: str = "cuda",
        dims=3,
    ):
        rand_pose = init_pose.clone()

        if rand_type == "Normal" or rand_type == "Gaussian":
            rand_pose[0][:dims, 3] = rand_pose[0][:dims, 3] + torch.normal(
                mean=0, std=sigma, size=(dims,)
            ).to(device=device)
        elif rand_type == "Uniform":
            rand_pose[0][:dims, 3] = rand_pose[0][:dims, 3] + sigma * (
                torch.rand((dims,)) - 0.5
            ).to(device=device)

        return rand_pose

    @staticmethod
    def generate_random_pose_r(
        init_pose: torch.Tensor,
        sigma: float = 1e-3,
        rand_type: str = "Uniform",
        device: str = "cuda",
    ):
        rand_pose = init_pose.clone()

        rotvec_rand_pose = rot.from_matrix(
            rand_pose[0][:3, :3].cpu().numpy()
        ).as_rotvec()
        if rand_type == "Normal" or rand_type == "Gaussian":
            rotvec_rand_pose = rotvec_rand_pose + np.random.normal(
                loc=0, scale=sigma, size=(1, 3)
            )
        elif rand_type == "Uniform":
            rotvec_rand_pose = rotvec_rand_pose + sigma * np.random.uniform(
                low=-0.5, high=0.5, size=(1, 3)
            )

        rand_pose[0][:3, :3] = torch.tensor(
            rot.from_rotvec(rotvec_rand_pose).as_matrix(), device=device
        )
        return rand_pose

    @staticmethod
    def generate_random_pose(
        init_pose: torch.Tensor,
        sigma_t: float = 1e-2,
        sigma_r: float = 1e-2,
        rand_type: str = "Uniform",
        device: str = "cuda",
    ):
        rand_pose_r = CameraPoseGenerator.generate_random_pose_r(
            init_pose, sigma_r, rand_type, device
        )
        rand_pose = CameraPoseGenerator.generate_random_pose_t(
            rand_pose_r, sigma_t, rand_type, device
        )
        return rand_pose

    @staticmethod
    def generate_random_pose_t_xy(
        init_pose: torch.Tensor,
        sigma_t: float = 1e-2,
        rand_type: str = "Uniform",
        device: str = "cuda",
    ):
        rand_pose = CameraPoseGenerator.generate_random_pose_t(
            init_pose, sigma_t, rand_type, device, dims=2
        )
        return rand_pose

    @staticmethod
    def generate_random_pose_t_xyz(
        init_pose: torch.Tensor,
        sigma_t: float = 1e-2,
        rand_type: str = "Uniform",
        device: str = "cuda",
    ):
        rand_pose = CameraPoseGenerator.generate_random_pose_t(
            init_pose, sigma_t, rand_type, device, dims=3
        )
        return rand_pose
