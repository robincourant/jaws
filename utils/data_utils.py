import numpy as np
import torch
import trimesh


def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array(
        [
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

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

    trimesh.Scene(objects).show()


def rand_poses(
    size,
    radius=1,
    theta_range=[np.pi / 3, 2 * np.pi / 3],
    phi_range=[0, 2 * np.pi],
):
    """generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    """

    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = (
        torch.rand(size) * (theta_range[1] - theta_range[0]) + theta_range[0]
    )
    phis = torch.rand(size) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack(
        [
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ],
        dim=-1,
    )  # [B, 3]

    # lookat
    forward_vector = -normalize(centers)
    up_vector = (
        torch.FloatTensor([0, -1, 0]).unsqueeze(0).repeat(size, 1)
    )  # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack(
        (right_vector, up_vector, forward_vector), dim=-1
    )
    poses[:, :3, 3] = centers

    return poses
