import subprocess
import os

import hydra
from omegaconf import DictConfig


def get_losses(loss_type):
    if loss_type == "pixel":
        return ["true", "false", "false"]
    if loss_type == "pose":
        return ["false", "true", "false"]
    if loss_type == "flow":
        return ["false", "false", "true"]
    if loss_type == "pose_flow":
        return ["false", "true", "true"]


def run_demo(
    data_dir: str,
    model: str,
    num_steps: int,
    seed: int,
    xpname: str,
    datamodule: str,
    alpha_losses: float,
    alpha_two_strokes: float,
    init_idx: int,
    init_focal_search: bool,
    two_strokes: float,
    flow_loss_type: str,
    guidance_type: str,
    learning_rate: float,
    num_epochs: int,
    num_mixed_grad: int,
    target_dir_name: str,
    loss_type: str,
):
    """
    Run dolly zoom demo with the given parameters.

    :param data_dir: path to the data directory.
    :param model: model name.
    :param num_steps: number of steps sampled per ray.
    :param seed: random seed.
    :param xpname: experiment name.
    :param datamodule: datamodule name.
    :param alpha_losses: weight of the losses.
    :param alpha_two_strokes: weight of the two strokes (spatial and temporal).
    :param init_idx: index of the initial camera.
    :param init_focal_search: whether to search for the initial focal length.
    :param two_strokes:  whether to uncouple spatial and temporal optimization.
    :param flow_loss_type: type of the flow metric (end-to-end, angular).
    :param guidance_type: type of the guidance (guidance map, inerf, random).
    :param learning_rate: learning rate.
    :param num_epochs: number of epochs.
    :param num_mixed_grad: number of guidance points.
    :param target_dir_name: name of the target directory.
    :param loss_type: type of the loss (pixel, pose, flow, pose_flow).
    """
    losses = get_losses(loss_type)
    command = (
        f"python {os.path.dirname(os.path.abspath(__file__))}/jaws/run.py \
            --config-name batch_jaws \
            run_type=jaws \
            data_dir={data_dir} \
            dynamic=true \
            group_name=demo \
            model={model} \
            num_steps={num_steps} \
            seed={seed} \
            xp_name={xpname} \
            datamodule={datamodule} \
            datamodule.alpha_losses={alpha_losses} \
            datamodule.alpha_two_strokes={alpha_two_strokes} \
            datamodule.blur_pred=false \
            datamodule.init_cam_idx={init_idx} \
            datamodule.only_init_focal_search={init_focal_search} \
            datamodule.two_strokes={two_strokes} \
            jaws.diff_focal=true \
            jaws.diff_temporal=false \
            jaws.flow_loss={losses[2]} \
            jaws.flow_loss_type={flow_loss_type} \
            jaws.grad_norm=true \
            jaws.guidance_type={guidance_type} \
            jaws.learning_rate={learning_rate} \
            jaws.num_epochs={num_epochs} \
            jaws.num_sample_grad={num_mixed_grad} \
            jaws.pixel_loss={losses[0]} \
            jaws.pose_loss={losses[1]} \
            jaws.target_dir={target_dir_name}",
    )
    subprocess.call(command, shell=True)


@hydra.main(
    config_path="./jaws/configs",
    config_name="demo_jaws.yaml",
    version_base="1.2",
)
def main(config: DictConfig):
    run_demo(
        data_dir=config.data_dir,
        xpname=config.xp_name,
        model=config.model,
        target_dir_name=config.target_dir_video,
        datamodule=config.datamodule,
        num_epochs=config.num_epochs,
        num_mixed_grad=config.num_mixed_grad,
        num_steps=config.num_steps,
        learning_rate=config.lr,
        seed=config.seed,
        loss_type=config.loss_type,
        init_idx=config.init_cam_idx_same,
        init_focal_search=config.init_focal_search,
        two_strokes=config.two_strokes,
        flow_loss_type=config.flow_loss_type,
        alpha_losses=config.alpha_losses,
        alpha_two_strokes=config.alpha_two_strokes,
        guidance_type=config.guidance_type,
    )


if __name__ == "__main__":
    main()
