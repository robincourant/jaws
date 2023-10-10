from pytorch_lightning import LightningModule
import torch
from torch import optim


def create_nerf_model(config) -> LightningModule:
    criterion = torch.nn.MSELoss(reduction="none")

    ff = config.datamodule.ff
    background_radius = 0 if ff else config.datamodule.background_radius
    background_encoding = None if ff else config.model.background_encoding
    n_background_layers = None if ff else config.model.n_background_layers
    background_hidden_dim = None if ff else config.model.background_hidden_dim
    background_perlin_noise = None if ff else config.datamodule.background_perlin_noise

    if config.dynamic:
        from jaws.src.models.dnerf_model import DNeRFModel

        # Initialize model
        optimizer = lambda model: torch.optim.Adam(
            model.get_params(config.model.lr, config.model.lr_net),
            betas=(0.9, 0.99),
            eps=1e-14,
        )
        lr_scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda iter: 0.1 ** min(iter / (config.num_epochs * 100), 1),
        )

        model = DNeRFModel(
            result_dir=config.result_dir,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            bound=config.datamodule.bound,
            aabb=config.datamodule.aabb,
            run_type=config.run_type,
            min_near=config.datamodule.min_near,
            density_thresh=config.datamodule.density_thresh,
            num_steps=config.num_steps,
            upsample_steps=config.upsample_steps,
            max_ray_batch=config.max_ray_batch,
            background_radius=background_radius,
            time_encoding=config.model.time_encoding,
            warp_encoding=config.model.warp_encoding,
            sigma_encoding=config.model.sigma_encoding,
            direction_encoding=config.model.direction_encoding,
            background_encoding=background_encoding,
            background_perlin_noise=background_perlin_noise,
            n_warp_layers=config.model.n_warp_layers,
            n_sigma_layers=config.model.n_sigma_layers,
            n_color_layers=config.model.n_color_layers,
            n_background_layers=n_background_layers,
            warp_hidden_dim=config.model.warp_hidden_dim,
            sigma_hidden_dim=config.model.sigma_hidden_dim,
            color_hidden_dim=config.model.color_hidden_dim,
            background_hidden_dim=background_hidden_dim,
            geo_feat_dim=config.model.geo_feat_dim,
            encoder_num_levels=config.model.encoder_num_levels,
            saturation_loss=config.saturation_loss,
            error_map=config.error_map,
            floater_ratio=config.floater_ratio,
        )
    else:
        from jaws.src.models.nerf_model import NeRFModel

        # Initialize model
        optimizer = lambda model: torch.optim.Adam(
            model.get_params(config.model.lr), betas=(0.9, 0.99), eps=1e-14
        )
        lr_scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda iter: 0.1 ** min(iter / (config.num_epochs * 100), 1),
        )

        model = NeRFModel(
            result_dir=config.result_dir,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            bound=config.datamodule.bound,
            aabb=config.datamodule.aabb,
            run_type=config.run_type,
            min_near=config.datamodule.min_near,
            density_thresh=config.datamodule.density_thresh,
            num_steps=config.num_steps,
            upsample_steps=config.upsample_steps,
            max_ray_batch=config.max_ray_batch,
            fully_fuse=ff,
            background_radius=background_radius,
            sigma_encoding=config.model.sigma_encoding,
            direction_encoding=config.model.direction_encoding,
            background_encoding=background_encoding,
            background_perlin_noise=background_perlin_noise,
            n_sigma_layers=config.model.n_sigma_layers,
            n_color_layers=config.model.n_color_layers,
            n_background_layers=n_background_layers,
            sigma_hidden_dim=config.model.sigma_hidden_dim,
            color_hidden_dim=config.model.color_hidden_dim,
            background_hidden_dim=background_hidden_dim,
            geo_feat_dim=config.model.geo_feat_dim,
            encoder_num_levels=config.model.encoder_num_levels,
            saturation_loss=config.saturation_loss,  # [TODO:]
            error_map=config.error_map,
            floater_ratio=config.floater_ratio,
        )
    if config.run_type != "train":
        model.training = False
    return model
