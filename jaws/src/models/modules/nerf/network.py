from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from jaws.src.models.modules.nerf.renderer import NeRFRenderer
from lib.torch_ngp.encoding import get_encoder
from lib.torch_ngp.activation import trunc_exp


class NeRFNetwork(NeRFRenderer):
    def __init__(
        self,
        sigma_encoding: str,
        direction_encoding: str,
        background_encoding: str,
        n_sigma_layers: int,
        n_color_layers: int,
        n_background_layers: int,
        sigma_hidden_dim: int,
        color_hidden_dim: int,
        background_hidden_dim: int,
        geo_feat_dim: int,
        bound: int,
        aabb: List,
        encoder_num_levels: int = 16,
        **kwargs
    ):
        super().__init__(bound, aabb, **kwargs)

        # Density network
        self._n_sigma_layers = n_sigma_layers
        self._sigma_hidden_dim = sigma_hidden_dim
        self._geo_feat_dim = geo_feat_dim
        self.sigma_encoder, self._sigma_in_dim = get_encoder(
            sigma_encoding,
            desired_resolution=2048 * bound,
            num_levels=encoder_num_levels,
        )
        self.sigma_net = self._build_net(
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
        self.color_net = self._build_net(
            input_dim=self._color_in_dim + self._geo_feat_dim,
            output_dim=3,
            hidden_dim=self._color_hidden_dim,
            num_layers=self._n_color_layers,
        )
        # Background network
        if self.background_radius > 0:
            self._n_background_layers = n_background_layers
            self._background_hidden_dim = background_hidden_dim
            self.background_encoder, self._background_in_dim = get_encoder(
                background_encoding,
                input_dim=2,
                num_levels=4,
                log2_hashmap_size=19,
                desired_resolution=2048,
            )  # much smaller hashgrid
            self.background_net = self._build_net(
                input_dim=self._background_in_dim + self._color_in_dim,
                output_dim=3,
                hidden_dim=self._background_hidden_dim,
                num_layers=self._n_background_layers,
            )
        else:
            self.background_net = None

    @staticmethod
    def _build_net(
        input_dim: int, output_dim: int, hidden_dim: int, num_layers: int
    ) -> nn.Module:
        net = []
        for layer_ind in range(num_layers):
            in_dim = input_dim if layer_ind == 0 else hidden_dim
            out_dim = output_dim if layer_ind == num_layers - 1 else hidden_dim
            net.append(nn.Linear(in_dim, out_dim, bias=False))
        net = nn.ModuleList(net)
        return net

    def forward(
        self, x: torch.Tensor, d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: [N, 3], in [-bound, bound]
        :param d: [N, 3], nomalized in [-1, 1]
        """
        # Sigma
        x_encoded = self.sigma_encoder(x, bound=self.bound)
        h = x_encoded
        for layer_ind in range(self._n_sigma_layers):
            h = self.sigma_net[layer_ind](h)
            if layer_ind != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # Color
        d_encoded = self.color_encoder(d)
        h = torch.cat([d_encoded, geo_feat], dim=-1)
        for layer_ind in range(self._n_color_layers):
            h = self.color_net[layer_ind](h)
            if layer_ind != self._n_color_layers - 1:
                h = F.relu(h, inplace=True)

        # Sigmoid activation for rgb
        rgb = torch.sigmoid(h)

        return sigma, rgb

    def density(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param x: [N, 3], in [-bound, bound]
        """
        x_encoded = self.sigma_encoder(x, bound=self.bound)
        h = x_encoded
        for layer_ind in range(self._n_sigma_layers):
            h = self.sigma_net[layer_ind](h)
            if layer_ind != self._n_sigma_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

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

        d_encoded = self.color_encoder(d)
        h = torch.cat([d_encoded, geo_feat], dim=-1)
        for layer_ind in range(self._n_color_layers):
            h = self.color_net[layer_ind](h)
            if layer_ind != self._n_color_layers - 1:
                h = F.relu(h, inplace=True)

        # Sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)
        else:
            rgbs = h

        return rgbs

    def background(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N, 2], in [-1, 1]
        """
        d_encoded = self.color_encoder(d)
        h_encoded = self.background_encoder(x)  # [N, C]

        h = torch.cat([d_encoded, h_encoded], dim=-1)
        for layer_ind in range(self._n_background_layers):
            h = self.background_net[layer_ind](h)
            if layer_ind != self._n_background_layers - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    def get_params(self, lr: float) -> Dict[str, Any]:
        params = [
            {"params": self.sigma_encoder.parameters(), "lr": lr},
            {"params": self.sigma_net.parameters(), "lr": lr},
            {"params": self.color_encoder.parameters(), "lr": lr},
            {"params": self.color_net.parameters(), "lr": lr},
        ]
        if self.background_radius > 0:
            params.append({"params": self.background_encoder.parameters(), "lr": lr})
            params.append({"params": self.background_net.parameters(), "lr": lr})

        return params
