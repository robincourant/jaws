from datetime import datetime
import glob
import os
import os.path as osp
from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig
import torch
from torch import Tensor
from torch.distributions.normal import Normal
import wandb

from jaws.src.models.jaws_model import JAWSModel, Recorder
from jaws.src.models.modules.nerf_factory import create_nerf_model
from jaws.src.datamodules.nerf_datamodule import NeRFDataModule
from utils.file_utils import create_dir, load_pickle, save_pickle
from utils.image_utils import (
    load_torch_image,
    save_flow_gif,
    save_poses_kitti,
    save_pixel_loss,
    save_image_gif,
)
from utils.misc_utils import cfg2dict


def load_targets(config: DictConfig) -> Tensor:
    target_paths = (
        glob.glob(osp.join(config.jaws.target_dir, "*.png"))
        + glob.glob(osp.join(config.jaws.target_dir, "*.jpg"))
        + glob.glob(osp.join(config.jaws.target_dir, "*.jpeg"))
    )
    assert len(target_paths) != 0
    target_images = [
        load_torch_image(path, config.jaws.image_size).to(config.device)
        for path in sorted(target_paths)
    ]
    return target_images


def init_search(
    config: DictConfig, data_module: NeRFDataModule, model: JAWSModel
) -> Tuple[List[Tensor], Dict[str, Any], Tensor, Tensor, float, float]:
    target_images = load_targets(config)

    for data in data_module.test_dataloader():
        break

    if config.datamodule.init_search != "saved":
        _potential_poses = [
            data["poses"].to(config.device)
            for data in data_module.train_dataloader()
        ]
        save_poses_kitti(
            _potential_poses, model._result_dir, "gt_train_poses.csv"
        )

    # Get initial parameters
    if config.datamodule.init_search == "ground_truth":
        # refactor distance for initial pose searching
        multiplicator = torch.ones_like(data["poses"].to(config.device))
        multiplicator[:, :3, 3] = config.datamodule.init_pose_dist_factor

        potential_poses = [multiplicator * pose for pose in _potential_poses]
        focal_factor = config.datamodule.focal_resize_factor
        init_time = config.datamodule.anim_start_time
    elif config.datamodule.init_search == "index":
        initial_cam_index = config.datamodule.init_cam_idx
        print("choosing index: ", initial_cam_index)
        init_pose = _potential_poses[initial_cam_index]
        potential_poses = None
        focal_factor = config.datamodule.focal_resize_factor
        init_time = config.datamodule.anim_start_time
    elif config.datamodule.init_search == "saved":
        # saved_pose_dir = osp.join(config.jaws.target_dir, "saved_init_data")
        saved_data = load_pickle(
            osp.join(config.jaws.target_dir, "saved_data.pk")
        )

        init_pose = saved_data["pose"]
        potential_poses = None
        focal_factor = saved_data["focal"]
        init_time = saved_data["time"].item()

    # Noise initial poses if not using ground truth to help with local minima
    if config.datamodule.init_search != "ground_truth":
        init_poses = []
        for _ in range(len(target_images)):
            noise = Normal(loc=0, scale=1e-2).sample((3,)).to(config.device)
            rand_pose = init_pose.clone().detach()
            rand_pose[0][:3, 3] += noise
            init_poses.append(rand_pose)
    else:
        init_poses = None

    return (
        target_images,
        data,
        init_poses,
        potential_poses,
        focal_factor,
        init_time,
    )


def get_recorded_data(
    recorder: Recorder,
    target_images: List[Tensor],
    imgs_final: List[Tensor],
    poses_final: List[Tensor],
    pixloss_final: List[Tensor],
    times_final: List[Tensor],
    focals_final: List[Tensor],
) -> Tuple[
    List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]
]:
    x, y = "recorded_", "recorded_loss"

    for cam_index in range(len(target_images)):
        imgs_final.append(
            recorder.get_item_best_loss(f"{x}images", y, cam_index)
        )
        poses_final.append(
            recorder.get_item_best_loss(f"{x}poses", y, cam_index)
        )
        pixloss_final.append(
            recorder.get_item_best_loss(f"{x}pixelloss", y, cam_index)
        )
        times_final.append(
            recorder.get_item_best_loss(f"{x}times", y, cam_index)
        )
        focals_final.append(
            recorder.get_item_best_loss(f"{x}focals", y, cam_index)
        )

    return imgs_final, poses_final, pixloss_final, times_final, focals_final


def jaws(config: DictConfig):
    # Initialize dataset
    data_module = NeRFDataModule(
        data_type="dynamic" if config.dynamic else "static",
        num_rays=config.num_rays,
        path=config.data_dir,
        mode=config.datamodule.mode,
        preload=config.datamodule.preload,
        scale=config.datamodule.scale,
        bound=config.datamodule.bound,
        aabb=config.aabb,
        rand_pose=config.datamodule.rand_pose,
    )

    # Initialize model
    nerf_model = create_nerf_model(config).model
    model = JAWSModel(
        alpha_losses=config.datamodule.alpha_losses,
        dynamic=config.dynamic,
        encoder_pretrained_path=config.model.encoder_checkpoint,
        flow_loss=config.jaws.flow_loss,
        flow_loss_type=config.jaws.flow_loss_type,
        grad_norm=config.jaws.grad_norm,
        learning_rate=config.jaws.learning_rate,
        max_ray_batch=config.max_ray_batch,
        model_size=config.model.model_size,
        model=nerf_model,
        num_steps=config.num_steps,
        pixel_loss_type=config.jaws.pixel_loss_type,
        pixel_loss=config.jaws.pixel_loss,
        pose_loss_type=config.jaws.pose_loss_type,
        pose_loss=config.jaws.pose_loss,
        raft_pretrained_path=config.model.raft_checkpoint,
        result_dir=config.result_dir,
        upsample_steps=config.upsample_steps,
    )
    model.model.training = False
    aabb = config.datamodule.aabb
    model.state_dict()["model.aabb_infer"].copy_(torch.tensor(aabb))
    model.state_dict()["model.aabb_train"].copy_(torch.tensor(aabb))

    # Initialize trainer
    checkpoint_dir = osp.join(config.result_dir, "checkpoints")
    checkpoint_filename = sorted(os.listdir(checkpoint_dir))[-1]
    checkpoint_path = osp.join(checkpoint_dir, checkpoint_filename)
    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    # Freeze model parameters
    for p in model.model.parameters():
        p.requires_grad = False
    model.to(config.device)

    # Initialize logger
    wandb.init(
        name=config.xp_name,
        project=config.project_name,
        group=config.group_name,
        config=cfg2dict(config),
    )
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    model.jaws_dir = osp.join(model.jaws_dir, timestamp)
    create_dir(model.jaws_dir)

    # Get initial parameters
    out = init_search(config, data_module, model)
    (
        target_images,
        init_data,
        init_poses,
        potential_poses,
        focal_factor,
        init_time,
    ) = out
    raw_intrisics = init_data["intrinsics"]

    # Start with 2 poses if only flow loss (flow loss enabled and pose loss disabled)
    if config.jaws.flow_loss and not config.jaws.pose_loss:
        _tmp_flow_loss = None
        _target_images = target_images[:2]
    # Start with 1 pose otherwise
    else:
        print("starting with 1 pose")
        _tmp_flow_loss = model._flow_loss
        _target_images = target_images[:1]
        model._flow_loss = False

    # Start JAWS's optimization
    out = model.batch_mixedgrad_jaws(
        alpha_two_strokes=config.datamodule.alpha_two_strokes,
        blur_kernel=list(config.jaws.blur_kernel),
        blur_pred=config.datamodule.blur_pred,
        blur_sigma=config.jaws.blur_sigma,
        clip_scheduler_indices=[],
        dataloader=init_data,
        diff_focal=config.jaws.diff_focal,
        diff_temporal=config.jaws.diff_temporal,
        early_stop_delta=config.jaws.early_stop_delta,
        early_stop_num=config.jaws.early_stop_num,
        flow_loss_type=config.jaws.flow_loss_type,
        focal_resize_factor=focal_factor,
        fp16=config.model.fp16,
        frozen_camera_indices=[],
        guidance_type=config.jaws.guidance_type,
        init_poses=init_poses,
        initial_time=init_time,
        log_interval=config.jaws.log_interval,
        num_epochs=int(config.jaws.num_epochs * 0.5),
        num_sample_grad=config.jaws.num_sample_grad,
        potential_poses=potential_poses,
        target_images=_target_images,
        two_strokes=False,
        use_guidance_map=config.jaws.guidance_map,
        vid_idx=0,
    )
    recorder, tg_imgs, pred_flow, gt_flow = out

    # Treat the first epoch as pose only if pose exist
    if _tmp_flow_loss:
        model._flow_loss = _tmp_flow_loss

    # Store step data
    (
        imgs_final,
        poses_final,
        pixloss_final,
        times_final,
        focals_final,
    ) = get_recorded_data(recorder, _target_images, [], [], [], [], [])
    gt_imgs_final, params_final = [], []
    for cam_index in range(len(_target_images)):
        gt_imgs_final.append(tg_imgs[cam_index])
        H, W, _ = imgs_final[cam_index].shape
        params_final.append(dict(H=H, W=W, intrinsics=raw_intrisics))
    flow_final, gt_flow_final = (
        ([pred_flow], [gt_flow])
        if pred_flow is not None and gt_flow is not None
        else ([], [])
    )
    # Each time only optimize the second image
    for index in range(len(_target_images) - 1, len(target_images) - 1):
        torch.cuda.empty_cache()
        init_time = (
            init_time + (1 - init_time) * (index / (len(target_images) - 2))
            if config.datamodule.auto_anim_time
            else times_final[-1]
        )
        diff_focal = (
            False
            if config.datamodule.only_init_focal_search
            else config.jaws.diff_focal
        )
        focal_factor = focals_final[-1]
        _target_images = target_images[index : index + 2]

        # Start from previous best camera pose
        _init_pose = [poses_final[-1], poses_final[-1]]
        out = model.batch_mixedgrad_jaws(
            dataloader=init_data,
            target_images=_target_images,
            init_poses=[x.to(config.device) for x in _init_pose],
            potential_poses=potential_poses,
            frozen_camera_indices=[0],  # freeze first camera
            clip_scheduler_indices=[],
            blur_kernel=list(config.jaws.blur_kernel),
            blur_sigma=config.jaws.blur_sigma,
            num_epochs=config.jaws.num_epochs,
            blur_pred=config.datamodule.blur_pred,
            fp16=config.model.fp16,
            log_interval=config.jaws.log_interval,
            num_sample_grad=config.jaws.num_sample_grad,
            vid_idx=index + 1,
            focal_resize_factor=focal_factor,
            use_guidance_map=config.jaws.guidance_map,
            early_stop_num=config.jaws.early_stop_num,
            early_stop_delta=config.jaws.early_stop_delta,
            diff_temporal=config.jaws.diff_temporal,
            diff_focal=diff_focal,
            allow_backward_t=config.jaws.allow_backward_t,
            initial_time=init_time,
            two_strokes=config.datamodule.two_strokes,
            flow_loss_type=config.jaws.flow_loss_type,
            alpha_two_strokes=config.datamodule.alpha_two_strokes,
            guidance_type=config.jaws.guidance_type,
        )
        recorder, tg_imgs, pred_flow, gt_flow = out

        imgs_final.pop()
        poses_final.pop()
        pixloss_final.pop()
        times_final.pop()
        focals_final.pop()

        # Store step data
        (
            imgs_final,
            poses_final,
            pixloss_final,
            times_final,
            focals_final,
        ) = get_recorded_data(
            recorder,
            _target_images,
            imgs_final,
            poses_final,
            pixloss_final,
            times_final,
            focals_final,
        )
        gt_imgs_final.append(tg_imgs[-1])
        flow_final.append(pred_flow)
        gt_flow_final.append(gt_flow)
        H, W, _ = gt_imgs_final[-1].shape
        params_final.append(dict(H=H, W=W, intrinsics=raw_intrisics))
        flow_final_tsr = torch.stack(flow_final)
        gt_flow_final_tsr = torch.stack(gt_flow_final)

        # Save step data
        log_dir = osp.join(model.jaws_dir, f"final_res_{index}")
        create_dir(log_dir)
        save_image_gif(imgs_final, gt_imgs_final, log_dir)
        save_flow_gif(flow_final_tsr, gt_flow_final_tsr, log_dir)
        save_poses_kitti(poses_final, log_dir)
        save_pickle(params_final, osp.join(log_dir, "params.pkl"))
        save_pickle(focals_final, osp.join(log_dir, "focals.pkl"))
        save_pickle(poses_final, osp.join(log_dir, "poses.pkl"))
        save_pickle(times_final, osp.join(log_dir, "times.pkl"))

    # Save the final pose same name of the folder
    save_poses_kitti(poses_final, model.jaws_dir, timestamp + ".csv")
    save_pixel_loss(
        pixloss_final, model.jaws_dir, timestamp + "_pixel_loss.txt"
    )
    dir_to_outputfile = osp.join(model.jaws_dir, timestamp + ".csv")
    target_traj_file = osp.join(config.jaws.target_dir, "traj.txt")
    output_res_dir_ape = osp.join(model.jaws_dir, "res_ape.zip")
    output_res_dir_rpe = osp.join(model.jaws_dir, "res_rpe.zip")

    # Save copy of GT
    os.system(f"cp {target_traj_file} {model.jaws_dir}")
    os.system(
        f"evo_ape kitti {target_traj_file} \
            {dir_to_outputfile} \
            --save_results {output_res_dir_ape}"
    )
    os.system(
        f"evo_rpe kitti {target_traj_file} \
            {dir_to_outputfile} \
            --save_results {output_res_dir_rpe}"
    )
