import os
import os.path as osp
from typing import List, Tuple
import copy

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.flow_utils import FlowUtils
from jaws.src.models.metrics.pose_loss import LitePoseLoss, DEFAULT_ARGS
from utils.loss_utils import (
    spatial_soft_argmax2d_with_misc,
)


def convert_img_torch_to_cv(image):
    cv_img = (image.numpy() * 255).astype(np.uint8)
    return cv_img


def convert_img_cv_to_torch(image):
    torch_img = torch.tensor(image.astype(np.float32) / 255.0)
    return torch_img


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def load_torch_image(
    input_filename: str, image_size: Tuple[int, int] = None
) -> torch.Tensor:
    """
    Load an image as a torch tensor (0-1 value), with an optional specified
    size.
    """
    numpy_image = cv2.imread(input_filename)[:, :, ::-1]
    H, W = image_size
    if image_size:
        numpy_image = image_resize(numpy_image, height=H)
    torch_image = torch.from_numpy(numpy_image) / 255
    return torch_image


def save_torch_image(torch_image: torch.Tensor, output_filename: str, type="clr"):
    """Save a torch tensor (0-1 value) as an image (to8b)."""
    numpy_image = (torch_image.numpy() * 255).astype(np.uint8)
    if type == "clr":
        cv2.imwrite(output_filename, numpy_image[:, :, ::-1])
    elif type == "gray":
        cv2.imwrite(output_filename, numpy_image[:, :])


def save_torch_weighted_image(
    torch_high_image: torch.Tensor,
    torch_low_image: torch.Tensor,
    output_filename: str,
):
    """Save a weighter sum of 2 torch tensor (0-1 value) as an image (to8b)."""
    numpy_high_image = (torch_high_image.numpy() * 255).astype(np.uint8)
    numpy_low_image = (torch_low_image.numpy() * 255).astype(np.uint8)
    weighted_image = cv2.addWeighted(numpy_high_image, 0.7, numpy_low_image, 0.3, 0)
    cv2.imwrite(output_filename, weighted_image[:, :, ::-1])


def save_gif(frames: List[torch.Tensor], output_filename: str, fps: int = 5):
    imageio.mimwrite(output_filename, frames, fps=fps)


def write_clip(frames: List[np.array], output_filename: str, fps: float = 24):
    """Write a clip in `mp4` format from a list of frames.

    :param frames: RGB frames to write.
    :param output_filename: file name of the saved output.
    :param fps: wanted frame per second rate.
    """
    frame_height, frame_width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Initialize the video writer
    clip = cv2.VideoWriter(
        output_filename,
        fourcc,
        fps,
        (frame_width, frame_height),
    )
    # Write each frame
    for frame in frames:
        clip.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Release the video writer
    clip.release()


def load_frames_fromdir(video_dir: str) -> List[np.array]:
    """Load BGR frames from a directory of frames."""
    frames = []
    for frame_filename in sorted(os.listdir(video_dir)):
        frame_path = osp.join(video_dir, frame_filename)
        frames.append(cv2.imread(frame_path))

    return frames


def load_frames(video_path: str) -> List[np.array]:
    """Load BGR frames from a video file."""
    video_clip = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(int(video_clip.get(7))):
        _, frame = video_clip.read()
        if frame is None:
            break
        frames.append(frame)

    return frames


def save_loss_marginal_image(
    ldt,
    ldr,
    loss,
    save_dir,
):
    min_dt = min(ldt)
    max_dt = max(ldt)
    min_dr = min(ldr)
    max_dr = max(ldr)

    t = np.linspace(min_dt, max_dt, 100)
    r = np.linspace(min_dr, max_dr, 100)
    loss_map = np.zeros((100, 100))
    num_map = np.zeros((100, 100))
    for idx, loss_elem in enumerate(loss):
        _t = ldt[idx]
        _r = ldr[idx]
        # t and r of current loss
        dist_loss_t = abs(t - _t)
        idxmin_t = np.argmin(dist_loss_t)

        dist_loss_r = abs(r - _r)
        idxmin_r = np.argmin(dist_loss_r)
        loss_map[idxmin_t][idxmin_r] += loss_elem
        num_map[idxmin_t][idxmin_r] += 1

    avg_loss_map = loss_map / num_map
    plt.imshow(avg_loss_map)
    plt.xticks(np.arange(len(t))[::20], np.round(t, 3)[::20])
    plt.xlabel("translational error")
    plt.yticks(np.arange(len(r))[::20], np.round(r, 3)[::20])
    plt.ylabel("rotational error")
    plt.colorbar()
    plt.savefig(save_dir, dpi=200)
    plt.close("all")

    sum_r = loss_map.sum(axis=1, where=num_map != 0)
    sum_num_r = num_map.sum(axis=1, where=num_map != 0)
    mean_r = sum_r / sum_num_r

    sum_t = loss_map.sum(axis=0, where=num_map != 0)
    sum_num_t = num_map.sum(axis=0, where=num_map != 0)
    mean_t = sum_t / sum_num_t

    plt.plot(t, mean_r)
    plt.xticks(np.arange(len(t))[::20], np.round(t, 3)[::20])
    plt.xlabel("translational error")
    plt.savefig(save_dir.replace(".png", "_t.png"), dpi=200)
    plt.close("all")

    plt.plot(r, mean_t)
    plt.xticks(np.arange(len(r))[::20], np.round(r, 3)[::20])
    plt.xlabel("rotational error")
    plt.savefig(save_dir.replace(".png", "_r.png"), dpi=200)
    plt.close("all")

    return avg_loss_map


def toUint8(np_array):
    return (np_array * 255).astype(np.uint8)


def save_heatmap(heatmap_17c, img, save_dir_fname):
    heatmap_17c = np.transpose(heatmap_17c, (1, 2, 0))
    heatmap_17c = toUint8(heatmap_17c)
    img = toUint8(img)

    _, _, C = heatmap_17c.shape
    H, W, _ = img.shape

    img = img[:, :, ::-1]

    for c in range(C):
        heatmap = cv2.resize(heatmap_17c[:, :, c], (W, H))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        img = cv2.addWeighted(heatmap, 1, img, 1, 0)
    cv2.imwrite(save_dir_fname, img)
    return img  # 8u


def save_heatmaps(heatmaps, imgs, save_dir, ext):
    B, C, _, _ = heatmaps.shape
    Bi = len(imgs)
    assert B == Bi
    imgs_8u = []
    for b in range(B):
        imgs_8u.append(
            save_heatmap(
                heatmaps[b].detach().cpu().numpy(),
                imgs[b].detach().cpu().numpy(),
                osp.join(save_dir, f"{b}_hmp_{ext}.png"),
            )
        )
    return imgs_8u


def save_heatmap_coords(
    heatmap_17c, coords_17, img, save_dir_fname, show_joint_num=False
):
    heatmap_17c = np.transpose(heatmap_17c, (1, 2, 0))
    heatmap_17c = toUint8(heatmap_17c)
    img = toUint8(img)

    _, _, C = heatmap_17c.shape
    H, W, _ = img.shape

    img = img[:, :, ::-1]
    for c in range(C):
        heatmap = cv2.resize(heatmap_17c[:, :, c], (W, H))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        img = cv2.addWeighted(heatmap, 0.8, img, 1, 0)
        center = (
            int((coords_17[c][0].detach().cpu().numpy()[0] + 1.0) * W / 2.0),
            int((coords_17[c][0].detach().cpu().numpy()[1] + 1.0) * H / 2.0),
        )
        if show_joint_num is False:
            continue
        img = cv2.circle(img, center, 5, (0, int(255 / C * c), 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255, 255, 255)
        thickness = 1
        lineType = 2
        img = cv2.putText(
            img,
            str(c),
            center,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )
        # cv2.imwrite(save_dir_fname.replace(".jpg", f"_{c}.jpg"), img_output)

    cv2.imwrite(save_dir_fname, img)


def put_text_on_image(img, text, pos=(100, 100)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0, 0, 255)
    thickness = 2
    lineType = 2
    img = cv2.putText(
        img,
        str(text),
        pos,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType,
    )
    return img


def save_heatmaps_coords(heatmaps, coords_17, imgs, save_dir, ext):
    B, C, _, _ = heatmaps.shape
    Bi = len(imgs)
    assert B == Bi

    for b in range(B):
        save_heatmap_coords(
            heatmaps[b].detach().cpu().numpy(),
            coords_17[b],
            imgs[b].detach().cpu().numpy(),
            osp.join(save_dir, f"{b}_hmp_{ext}.jpg"),
        )


def save_image_gif(
    pred_images: torch.Tensor,
    gt_images: torch.Tensor,
    log_dir: str,
    fname: str = "images.gif",
):
    """Save images as GIF."""
    stacked_images = [
        np.hstack(
            [
                (p_rgb.numpy() * 255).astype(np.uint8),
                (g_rgb.numpy() * 255).astype(np.uint8),
            ]
        )
        for p_rgb, g_rgb in zip(pred_images, gt_images)
    ]
    pred_image_path = osp.join(log_dir, fname)
    save_gif(stacked_images, pred_image_path)


def save_flow_gif(
    pred_flows: torch.Tensor,
    gt_flows: torch.Tensor,
    log_dir: str,
    filename: str = "flow.gif",
):
    """Save optical flows as GIF."""
    flow_utils = FlowUtils()
    stacked_rgb_flows = [
        np.hstack(
            [
                flow_utils.flow_to_frame(p_flow.numpy()),
                flow_utils.flow_to_frame(g_flow.numpy()),
            ]
        )
        for p_flow, g_flow in zip(pred_flows, gt_flows)
    ]
    pred_flow_path = osp.join(log_dir, filename)
    save_gif(stacked_rgb_flows, pred_flow_path)


def save_poses_kitti(poses: List[torch.Tensor], log_dir: str, fname: str = None):
    """_summary_ file saved in KITTI file version
                can be analysed by evo_traj:
                https://github.com/MichaelGrupp/evo/wiki/evo_traj
    Args:
        poses (List[torch.Tensor]): _description_
        log_dir (str): _description_
    """
    if fname is None:
        file_dir = osp.join(log_dir, "traj.txt")
    else:
        file_dir = osp.join(log_dir, fname)
    with open(file_dir, "w") as output_file:
        for row in poses:
            output_file.write(
                " ".join(
                    [str(elem) for elem in row.cpu().detach().numpy().flatten()[:-4]]
                )
                + "\n"
            )
    # print(f"pose saved in :\n{file_dir}")


def save_pixel_loss(pixloss: List[float], log_dir: str, fname: str = None):
    """_summary_ file saved in KITTI file version
                can be analysed by evo_traj:
                https://github.com/MichaelGrupp/evo/wiki/evo_traj
    Args:
        poses (List[torch.Tensor]): _description_
        log_dir (str): _description_
    """
    if fname is None:
        file_dir = osp.join(log_dir, "pixel_loss.csv")
    else:
        file_dir = osp.join(log_dir, fname)
    with open(file_dir, "w") as output_file:
        for row in pixloss:
            output_file.write(str(row) + "\n")
    # print(f"pose saved in :\n{file_dir}")


def compute_heatmaps_from_imgs(imgs, dir):
    with torch.no_grad():
        pose_loss = LitePoseLoss(DEFAULT_ARGS, fp16=False, heatmap_loss=True)
        heatmaps = []
        temp = 200

        for idx, img in enumerate(imgs):
            heatmap = pose_loss.pool(pose_loss.heatmap(img.unsqueeze(0)))
            B, C, H, W = heatmap.shape
            lheatmaps = [[] for _ in range(B)]
            coords = [[] for _ in range(B)]

            for b in range(B):
                for channel in range(C):  # 17
                    hmap_1c_x = heatmap[b, channel].unsqueeze(0).unsqueeze(0)
                    (
                        coord_max_x,
                        soft_x,
                        conf_x,
                    ) = spatial_soft_argmax2d_with_misc(
                        hmap_1c_x,
                        normalized_coordinates=True,
                        temperature=torch.tensor(temp),
                    )
                    lheatmaps[b].append(soft_x.squeeze(0).squeeze(0))
                    coords[b].append(coord_max_x.squeeze(0))
            # heatmap = torch.stack([torch.stack(heatmap_x)
            # for heatmap_x in lheatmaps])

            heatmap[heatmap < 0.5] = 0.0
            heatmap[heatmap > 0.5] = 1.0
            ht = torch.zeros_like(heatmap)
            ht_img = heatmap.permute(0, 2, 3, 1)
            save_heatmaps(ht, ht_img, dir, f"{idx}")
            heatmaps.append(heatmap.squeeze(0))
        return heatmaps


class InfoRecorder:
    def __init__(self, list_name: List, num_cameras: int):
        self.data_dict = {i: [[] for _ in range(num_cameras)] for i in list_name}

    def append(self, name, cam_idx, data):
        self.data_dict[name][cam_idx].append(data)

    def get(self, name):
        return self.data_dict[name]

    def argmin(a):
        return min(range(len(a)), key=lambda x: a[x])

    def argmax(a):
        return max(range(len(a)), key=lambda x: a[x])

    def get_item_best_loss(self, name_item, name_loss, cam_idx, is_min=True):
        best_idx = self.get_idx_best_loss(
            name_loss=name_loss, cam_idx=cam_idx, is_min=is_min
        )

        return self.get(name_item)[cam_idx][best_idx]

    def get_idx_best_loss(self, name_loss, cam_idx, is_min=True):
        if is_min:
            best_idx = InfoRecorder.argmin((self.get(name_loss)[cam_idx]))
        else:
            best_idx = InfoRecorder.argmax((self.get(name_loss)[cam_idx]))

        return best_idx

    def get_item_last(self, name_item, cam_idx):
        return self.get(name_item)[cam_idx][-1]

    @staticmethod
    def merge_info_recorder(info_rec_base, names, alpha):
        info_rec_out = copy.deepcopy(info_rec_base)

        for name in names:
            for cam_idx in range(len(info_rec_out.data_dict[name])):
                half_list = [
                    alpha * b + (1 - alpha) * m
                    for (b, m) in zip(
                        info_rec_base.get(name)[cam_idx][::2],
                        info_rec_base.get(name)[cam_idx][1::2],
                    )
                ]
                info_rec_out.data_dict[name][cam_idx] = list(
                    np.repeat(np.array(half_list), 2)
                )
                assert len(info_rec_out.data_dict[name][cam_idx]) == len(
                    info_rec_base.get(name)[cam_idx]
                )

        return info_rec_out
