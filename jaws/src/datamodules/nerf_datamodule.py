from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import jaws.src.datamodules.datasets.dnerf_dataset as dnerf
import jaws.src.datamodules.datasets.nerf_dataset as nerf


class NeRFDataModule(LightningDataModule):
    """Initialize train, val and test base data loader."""

    def __init__(
        self,
        data_type: str,
        num_rays: int,
        path: str,
        mode: str,
        preload: bool,
        scale: float,
        bound: int,
        rand_pose: bool,
        ind_calib: bool = False,
        error_map: bool = False,
        aabb=None,
    ):
        super().__init__()
        self._num_rays = num_rays
        self._path = path
        self._mode = mode
        self._preload = preload
        self._scale = scale
        self._bound = bound
        self._aabb = aabb
        self._rand_pose = rand_pose
        self._ind_calib = ind_calib
        self._error_map = error_map

        if data_type == "dynamic":
            self.dataset = dnerf.DNeRFDataset
        else:
            self.dataset = nerf.NeRFDataset

    def train_dataloader(self) -> DataLoader:
        """Load train set loader."""
        self.train_dataset = self.dataset(
            num_rays=self._num_rays,
            path=self._path,
            mode=self._mode,
            preload=self._preload,
            scale=self._scale,
            bound=self._bound,
            aabb=self._aabb,
            rand_pose=self._rand_pose,
            type="train",
            ind_calibration=self._ind_calib,
            error_map=self._error_map,
        )
        return self.train_dataset.dataloader()

    def val_dataloader(self) -> DataLoader:
        """Load val set loader."""
        return self.dataset(
            num_rays=self._num_rays,
            path=self._path,
            mode=self._mode,
            preload=self._preload,
            scale=self._scale,
            bound=self._bound,
            aabb=self._aabb,
            rand_pose=self._rand_pose,
            type="val",
            ind_calibration=self._ind_calib,
            error_map=self._error_map,
        ).dataloader()

    def test_dataloader(self) -> DataLoader:
        """Load test set loader."""
        return self.dataset(
            num_rays=self._num_rays,
            path=self._path,
            mode=self._mode,
            preload=self._preload,
            scale=self._scale,
            bound=self._bound,
            aabb=self._aabb,
            rand_pose=self._rand_pose,
            type="test",
            ind_calibration=self._ind_calib,
            error_map=self._error_map,
        ).dataloader()
