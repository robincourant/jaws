import os.path as osp
import sys

import hydra
from omegaconf import DictConfig

from utils.nerf_utils import seed_everything


@hydra.main(
    config_path="configs/", config_name="train_nerf.yaml", version_base="1.2"
)
def main(config: DictConfig):
    sys.path.append(osp.join(config.root, "lib", "torch_ngp"))
    seed_everything(config.seed)

    if config.run_type == "train":
        from jaws.src.train import train

        train(config)

    if config.run_type == "jaws":
        from jaws.src.jaws import jaws

        jaws(config)

    if config.run_type == "render":
        from jaws.src.render import render

        render(config)


if __name__ == "__main__":
    main()
