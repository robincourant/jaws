from typing import Any, Dict, Tuple, List

import torch

from jaws.src.models.modules.nerf.renderer import NeRFRenderer
from lib.torch_ngp.encoding import get_encoder
from lib.torch_ngp.activation import trunc_exp
from lib.torch_ngp.ffmlp import FFMLP


class NeRFNetwork(NeRFRenderer):
    def __init__(
        self,
        sigma_encoding: str,
        direction_encoding: str,
        n_sigma_layers: int,
        n_color_layers: int,
        sigma_hidden_dim: int,
        color_hidden_dim: int,
        geo_feat_dim: int,
        bound: int,
        aabb: List,
        encoder_num_levels: int,
        **kwargs
    ):
        super().__init__(
            bound, aabb, background_radius=0, background_perlin_noise=False, **kwargs
        )

        # Density network
        self._n_sigma_layers = n_sigma_layers
        self._sigma_hidden_dim = sigma_hidden_dim
        self._geo_feat_dim = geo_feat_dim
        self.sigma_encoder, self._sigma_in_dim = get_encoder(
            sigma_encoding,
            desired_resolution=2048 * bound,
            num_levels=encoder_num_levels,
        )
        self.sigma_net = FFMLP(
            input_dim=self._sigma_in_dim,
            output_dim=1 + self._geo_feat_dim,
            hidden_dim=self._sigma_hidden_dim,
            num_layers=self._n_sigma_layers,
        )

        # Color network
        self._n_color_layers = n_color_layers
        self._color_hidden_dim = color_hidden_dim
        self.color_encoder, self._color_in_dim = get_encoder(
            direction_encoding,
            desired_resolution=2048,
            num_levels=encoder_num_levels,
        )
        self._color_in_dim += self._geo_feat_dim + 1
        self.color_net = FFMLP(
            input_dim=self._color_in_dim,
            output_dim=3,
            hidden_dim=self._color_hidden_dim,
            num_layers=self._n_color_layers,
        )

    def forward(
        self, x: torch.Tensor, d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: [N, 3], in [-bound, bound]
        :param d: [N, 3], nomalized in [-1, 1]
        """
        # Sigma
        x = self.sigma_encoder(x, bound=self.bound)
        h = self.sigma_net(x)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # Color
        d = self.color_encoder(d)
        p = torch.zeros_like(geo_feat[..., :1])  # manual input padding
        h = torch.cat([d, geo_feat, p], dim=-1)
        h = self.color_net(h)

        # Sigmoid activation for rgb
        rgb = torch.sigmoid(h)

        return sigma, rgb

    def density(self, _x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param x: [N, 3], in [-bound, bound]
        """
        x = self.sigma_encoder(_x, bound=self.bound)
        h = self.sigma_net(x)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        if torch.isnan(sigma).any():
            assert False
        return {
            "sigma": sigma,
            "geo_feat": geo_feat,
        }

    def color(
        self,
        x: torch.Tensor,
        d: torch.Tensor,
        mask: torch.Tensor = None,
        geo_feat: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Allow masked inference.

        :param x: [N, 3] in [-bound, bound]
        :param mask: [N,], bool, indicates where rgb is needed to be computed.
        """
        if mask is not None:
            # [N, 3]
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)
            # Empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.color_encoder(d)

        p = torch.zeros_like(geo_feat[..., :1])  # manual input padding
        h = torch.cat([d, geo_feat, p], dim=-1)
        h = self.color_net(h)

        # Sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)
        else:
            rgbs = h

        return rgbs

    def get_params(self, lr: float) -> Dict[str, Any]:
        params = [
            {"params": self.sigma_encoder.parameters(), "lr": lr},
            {"params": self.sigma_net.parameters(), "lr": lr},
            {"params": self.color_encoder.parameters(), "lr": lr},
            {"params": self.color_net.parameters(), "lr": lr},
        ]
        return params
