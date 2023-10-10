from typing import Any, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from jaws.src.models.modules.dnerf.renderer import DNeRFRenderer
from lib.torch_ngp.encoding import get_encoder
from lib.torch_ngp.activation import trunc_exp


class DNeRFNetwork(DNeRFRenderer):
    def __init__(
        self,
        time_encoding: str,
        warp_encoding: str,
        sigma_encoding: str,
        direction_encoding: str,
        background_encoding: str,
        n_warp_layers: int,
        n_sigma_layers: int,
        n_color_layers: int,
        n_background_layers: int,
        warp_hidden_dim: int,
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

        # Deformation network
        self._n_warp_layers = n_warp_layers
        self._warp_hidden_dim = warp_hidden_dim
        self.warp_encoder, self._warp_in_dim = get_encoder(
            warp_encoding,
            multires=10,
            desired_resolution=2048,
            num_levels=encoder_num_levels,
        )
        self.time_encoder, self._time_in_dim = get_encoder(
            time_encoding,
            input_dim=1,
            desired_resolution=2048,
            num_levels=encoder_num_levels,
        )
        self.warp_net = self._build_net(
            input_dim=self._warp_in_dim + self._time_in_dim,
            output_dim=3,
            hidden_dim=self._warp_hidden_dim,
            num_layers=self._n_warp_layers,
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
        in_dim = self._sigma_in_dim + self._warp_in_dim + self._time_in_dim
        self.sigma_net = self._build_net(
            input_dim=in_dim,
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
        self, x: torch.Tensor, d: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: [N, 3], in [-bound, bound]
        :param d: [N, 3], nomalized in [-1, 1]
        :param t: [1, 1], in [0, 1]
        """
        # Deformation
        x0_encoded = self.warp_encoder(x, bound=self.bound)  # [N, C]
        t_encoded = self.time_encoder(t)  # [1, 1] --> [1, C']
        if t_encoded.shape[0] == 1:
            t_encoded = t_encoded.repeat(x.shape[0], 1)  # [1, C'] --> [N, C']
        warp = torch.cat([x0_encoded, t_encoded], dim=1)  # [N, C + C']
        for layer_ind in range(self._n_warp_layers):
            warp = self.warp_net[layer_ind](warp)
            if layer_ind != self._n_warp_layers - 1:
                warp = F.relu(warp, inplace=True)
        x += warp

        # Sigma
        x_encoded = self.sigma_encoder(x, bound=self.bound)
        h = torch.cat([x_encoded, x0_encoded, t_encoded], dim=1)
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

        return sigma, rgb, warp

    def density(self, x: torch.Tensor, t: torch.tensor) -> Dict[str, torch.Tensor]:
        """
        :param x: [N, 3], in [-bound, bound]
        :param t: [1, 1], in [0, 1]
        """
        # Deformation
        x0_encoded = self.warp_encoder(x, bound=self.bound)  # [N, C]
        t_encoded = self.time_encoder(t)  # [1, 1] --> [1, C']
        if t_encoded.shape[0] == 1:
            t_encoded = t_encoded.repeat(x.shape[0], 1)  # [1, C'] --> [N, C']

        warp = torch.cat([x0_encoded, t_encoded], dim=1)  # [N, C + C']
        for layer_ind in range(self._n_warp_layers):
            warp = self.warp_net[layer_ind](warp)
            if layer_ind != self._n_warp_layers - 1:
                warp = F.relu(warp, inplace=True)
        x = x + warp

        # Sigma
        x_encoded = self.sigma_encoder(x, bound=self.bound)
        h = torch.cat([x_encoded, x0_encoded, t_encoded], dim=1)
        for layer_ind in range(self._n_sigma_layers):
            h = self.sigma_net[layer_ind](h)
            if layer_ind != self._n_sigma_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            "sigma": sigma,
            "geo_feat": geo_feat,
            "warp": warp,
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
        :param t: [1, 1], in [0, 1]
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

    def get_params(self, lr: float, lr_net: float) -> Dict[str, Any]:
        params = [
            {"params": self.warp_encoder.parameters(), "lr": lr},
            {"params": self.time_encoder.parameters(), "lr": lr},
            {"params": self.warp_net.parameters(), "lr": lr_net},
            {"params": self.sigma_encoder.parameters(), "lr": lr},
            {"params": self.sigma_net.parameters(), "lr": lr},
            {"params": self.color_encoder.parameters(), "lr": lr},
            {"params": self.color_net.parameters(), "lr": lr},
        ]
        if self.background_radius > 0:
            params.append({"params": self.background_encoder.parameters(), "lr": lr})
            params.append({"params": self.background_net.parameters(), "lr": lr_net})

        return params
