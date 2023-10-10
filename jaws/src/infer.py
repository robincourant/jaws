from datetime import datetime
from omegaconf import DictConfig
import os
import os.path as osp
import sys

import torch
from torch import optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from jaws.src.models.nerf_model import NeRFModel
from jaws.src.datamodules.nerf_datamodule import NeRFDataModule
from utils.file_utils import create_dir
from jaws.src.models.modules.nerf_factory import create_nerf_model


def infer(config: DictConfig):
    sys.path.append(osp.join(".", "lib", "torch_ngp"))
    model = create_nerf_model(config)

    # Initialize dataset
    data_module = NeRFDataModule(
        data_type="dynamic" if config.dynamic else "static",
        num_rays=config.num_rays,
        path=config.data_dir,
        mode=config.datamodule.mode,
        preload=config.datamodule.preload,
        scale=config.datamodule.scale,
        bound=config.datamodule.bound,
        rand_pose=config.datamodule.rand_pose,
        ind_calib=config.datamodule.independent_calibration,
        error_map=config.error_map,
    )

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
    elif config.model.ckpt == "scratch":
        checkpoint_path = None
    else:
        checkpoint_path = config.model.ckpt
    checkpoint = ModelCheckpoint(
        monitor=config.checkpoint_metric,
        mode="min",
        save_top_k=config.num_checkpoints,
        dirpath=checkpoint_dir,
        filename="{epoch}",
        save_on_train_epoch_end=True,
    )
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    wandb_logger = WandbLogger(
        name="_".join([config.xp_name, "nerf", timestamp]),
        project=config.project_name,
        offline=config.log_offline,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [lr_monitor, checkpoint]
    trainer = Trainer(
        gpus=config.compnode.num_gpus,
        num_nodes=config.compnode.num_nodes,
        accelerator=config.compnode.accelerator,
        max_epochs=config.num_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        log_every_n_steps=5,
        precision=16 if config.model.fp16 else 32,
        num_sanity_val_steps=config.num_sanity_val_steps,
    )
    # Launch model training
    trainer.test(model, data_module, ckpt_path=checkpoint_path)
