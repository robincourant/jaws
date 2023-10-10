from itertools import product
import math
from typing import List

from perlin_numpy import generate_perlin_noise_3d
import torch
import torch.nn as nn

from lib.torch_ngp.nerf.utils import custom_meshgrid
import lib.torch_ngp.raymarching as raymarching
from utils.nerf_utils import sample_pdf


class DNeRFRenderer(nn.Module):
    """NGP NeRF renderer.

    :param density_scale: scale up deltas (or sigmas), to make the density
        grid sharper. Larger value than 1 usually improves performance.
    """

    def __init__(
        self,
        bound: int,
        aabb: List,
        density_scale: int,
        min_near: float,
        density_thresh: float,
        background_radius: int,
        background_perlin_noise: bool,
    ):
        super().__init__()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.time_size = 64
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.background_radius = background_radius
        self.background_perlin_noise = background_perlin_noise

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we
        # still rely on bound (cubic) to calculate density grid and hashing.
        if aabb is not None and len(aabb) > 0:
            aabb_train = torch.FloatTensor(aabb)
        else:
            aabb_train = torch.FloatTensor(
                [-bound, -bound, -bound, bound, bound, bound]
            )

        # aabb_train[0] = -1.3  # xmin
        # aabb_train[2] = -0.7  # ymax
        # aabb_train[3] = 1.3  # xmax
        # aabb_train[4] = -0.4  # ymax
        # aabb_train[5] = 0.7  # ymax

        aabb_infer = aabb_train.clone()

        self.register_buffer("aabb_train", aabb_train)
        self.register_buffer("aabb_infer", aabb_infer)

        # gen perlin noise
        if self.background_perlin_noise:
            # perlin noise [-1, 1]
            self.perlin_noise = torch.tensor(
                generate_perlin_noise_3d(
                    (300, 300, 3), (4, 4, 1), tileable=(False, False, True)
                ),
                dtype=torch.float32,
            )
            self.perlin_noise = self.perlin_noise / 2.0 + 0.5

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        raise NotImplementedError()

    def density(self, x: torch.Tensor):
        raise NotImplementedError()

    def background(self, x: torch.Tensor):
        raise NotImplementedError()

    def color(
        self,
        x: torch.Tensor,
        d: torch.Tensor,
        mask: torch.Tensor = None,
        geo_feat: torch.Tensor = None,
        **kwargs,
    ):
        raise NotImplementedError()

    def reset_extra_state(self):
        # Density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # Step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def run(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        time: torch.Tensor,
        num_steps: int,
        upsample_steps: int,
        background_color: torch.Tensor,
        perturb: bool,
        eliminate_floater: bool,
    ) -> torch.Tensor:
        """
        :param rays_o, rays_d: [B, N, 3], assumes B == 1
        :param background_color: [3] in range [0, 1]
        :param time: [B, 1]
        :param return: image: [B, N, 3], depth: [B, N]
        """
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        with torch.no_grad():
            nears, fars = raymarching.near_far_from_aabb(
                rays_o, rays_d, aabb, self.min_near
            )
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(
            0
        )  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = (
                z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            )

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(
            -1
        )  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3), time)

        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():
                deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]
                deltas = torch.cat(
                    [deltas, sample_dist * torch.ones_like(deltas[..., :1])],
                    dim=-1,
                )

                alphas = 1 - torch.exp(
                    -deltas * self.density_scale * density_outputs["sigma"].squeeze(-1)
                )  # [N, T]
                alphas_shifted = torch.cat(
                    [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15],
                    dim=-1,
                )  # [N, T+1]
                weights = (
                    alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
                )  # [N, T]

                # sample new z_vals
                z_vals_mid = z_vals[..., :-1] + 0.5 * deltas[..., :-1]  # [N, T-1]
                new_z_vals = sample_pdf(
                    z_vals_mid,
                    weights[:, 1:-1],
                    upsample_steps,
                    det=not self.training,
                ).detach()  # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(
                    -2
                ) * new_z_vals.unsqueeze(
                    -1
                )  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(
                    torch.max(new_xyzs, aabb[:3]), aabb[3:]
                )  # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3), time)
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1)  # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1)  # [N, T+t, 3]
            xyzs = torch.gather(
                xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs)
            )

            for k in density_outputs:
                tmp_output = torch.cat(
                    [density_outputs[k], new_density_outputs[k]], dim=1
                )
                density_outputs[k] = torch.gather(
                    tmp_output,
                    dim=1,
                    index=z_index.unsqueeze(-1).expand_as(tmp_output),
                )

        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat(
            [deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1
        )
        alphas = 1 - torch.exp(
            -deltas * self.density_scale * density_outputs["sigma"].squeeze(-1)
        )  # [N, T+t]
        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1
        )  # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        mask = weights > 1e-4  # hard coded-

        rgbs = self.color(
            xyzs.reshape(-1, 3),
            dirs.reshape(-1, 3),
            mask=mask.reshape(-1),
            **density_outputs,
        )
        rgbs = rgbs.view(N, -1, 3)  # [N, T+t, 3]

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)  # [N]

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)  # [N, 3], in [0, 1]
        # mix background color
        with torch.no_grad():
            if self.background_radius > 0 and not self.background_perlin_noise:
                # use the background model to calculate background_color
                sph = raymarching.sph_from_ray(
                    rays_o, rays_d, self.background_radius
                )  # [N, 2] in [-1, 1]
                background_color = self.background(sph, rays_d.reshape(-1, 3))  # [N, 3]
            elif self.background_radius > 0 and self.background_perlin_noise:
                # use the background model to calculate background_color
                sph = raymarching.sph_from_ray(
                    rays_o, rays_d, self.background_radius
                )  # [N, 2] in [-1, 1]
                if self.perlin_noise.device != device:
                    self.perlin_noise = self.perlin_noise.to(device=device)
                sph_idx = torch.ceil(
                    sph * 128 + 128
                ).long()  # perlin_noise is in [300, 300] no worry to overflow
                background_color = self.perlin_noise[sph_idx[:, 0], sph_idx[:, 1]]

            elif background_color is None:
                background_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * background_color
        image = image.view(*prefix, 3)
        if torch.isnan(image).any():
            print("nan occured")
        if eliminate_floater:
            return image, [weights, z_vals, sample_dist]
        else:
            return image, None

    def _cascading(
        self, cas: torch.Tensor, xyzs: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        bound = min(2**cas, self.bound)
        half_grid_size = bound / self.grid_size
        half_time_size = 0.5 / self.time_size
        # scale to current cascade's resolution
        cas_xyzs = xyzs * (bound - half_grid_size)
        # add noise in [-hgs, hgs]
        cas_xyzs += torch.rand_like(cas_xyzs) * 2 - 1
        cas_xyzs *= half_grid_size
        # add noise in time [-hts, hts]
        time_perturb = time + (torch.rand_like(time) * 2 - 1)
        time_perturb *= half_time_size
        # query density
        sigmas = self.density(cas_xyzs, time_perturb)["sigma"].reshape(-1).detach()
        sigmas *= self.density_scale

        return sigmas

    @torch.no_grad()
    def update_extra_state(self, decay: float = 0.95, S: int = 128):
        """Call before each epoch to update extra states."""
        # Update density grid
        tmp_grid = torch.ones_like(self.density_grid)

        # full update
        T = enumerate(self.times)
        if self.iter_density < 16:
            # if True:
            X = torch.arange(
                self.grid_size,
                dtype=torch.int32,
                device=self.density_bitfield.device,
            ).split(S)
            Y = torch.arange(
                self.grid_size,
                dtype=torch.int32,
                device=self.density_bitfield.device,
            ).split(S)
            Z = torch.arange(
                self.grid_size,
                dtype=torch.int32,
                device=self.density_bitfield.device,
            ).split(S)

            for (t, time), xs, ys, zs in product(T, X, Y, Z):
                # construct points
                xx, yy, zz = custom_meshgrid(xs, ys, zs)
                coords = torch.cat(
                    [
                        xx.reshape(-1, 1),
                        yy.reshape(-1, 1),
                        zz.reshape(-1, 1),
                    ],
                    dim=-1,
                )  # [N, 3], in [0, 128]
                indices = raymarching.morton3D(coords).long()  # [N]
                # [N, 3] in [-1, 1]
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1

                # cascading
                for cas in range(self.cascade):
                    sigmas = self._cascading(cas, xyzs, time)
                    # assign
                    tmp_grid[t, cas, indices] = sigmas

        # partial update (half the computation)
        elif self.iter_density < 100:
            N = self.grid_size**3 // 4  # T * C * H * H * H / 4
            for (t, time), cas in product(T, range(self.cascade)):
                # random sample some positions
                coords = torch.randint(
                    0,
                    self.grid_size,
                    (N, 3),
                    device=self.density_bitfield.device,
                )  # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long()  # [N]
                # random sample occupied positions  [Nz]
                occ_indices = torch.nonzero(self.density_grid[t, cas] > 0).squeeze(-1)
                rand_mask = torch.randint(
                    0,
                    occ_indices.shape[0],
                    [N],
                    dtype=torch.long,
                    device=self.density_bitfield.device,
                )
                # [Nz] --> [N], allow for duplication
                occ_indices = occ_indices[rand_mask]
                occ_coords = raymarching.morton3D_invert(occ_indices)  # [N, 3]
                # concat
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                # same below [N, 3] in [-1, 1]
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1
                sigmas = self._cascading(cas, xyzs, time)
                # assign
                tmp_gridt, [cas, indices] = sigmas

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(
            self.density_grid[valid_mask] * decay, tmp_grid[valid_mask]
        )
        # -1 non-training regions are viewed as 0 density.
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item()
        self.iter_density += 1

        # Convert to bitfield
        # density_thresh = self.mean_density
        density_thresh = min(self.mean_density, self.density_thresh)
        # density_thresh = self.density_thresh

        for t in range(self.time_size):
            self.density_bitfield = raymarching.packbits(
                self.density_grid[t], density_thresh, self.density_bitfield[t]
            )

        # Update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(
                self.step_counter[:total_step, 0].sum().item() / total_step
            )
        self.local_step = 0

    def render(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        time: torch.Tensor,
        staged: bool,
        background_color: int,
        perturb: bool,
        max_ray_batch: int,
        num_steps: int,
        upsample_steps: int,
        eliminate_floater: bool = False,  # only in train
    ) -> torch.Tensor:
        """
        :param rays_o, rays_d: [B, N, 3], assumes B == 1
        :return: pred_rgb: [B, N, 3]
        """
        B, N = rays_o.shape[:2]

        if staged:
            device = rays_o.device
            image = torch.empty((B, N, 3), device=device)
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    image_chunk, floater_pack = self.run(
                        rays_o=rays_o[b : b + 1, head:tail],
                        rays_d=rays_d[b : b + 1, head:tail],
                        time=time[b : b + 1],
                        background_color=background_color,
                        perturb=perturb,
                        num_steps=num_steps,
                        upsample_steps=upsample_steps,
                        eliminate_floater=eliminate_floater,
                    )
                    image[b : b + 1, head:tail] = image_chunk
                    head += max_ray_batch

        else:
            image, floater_pack = self.run(
                rays_o=rays_o,
                rays_d=rays_d,
                time=time,
                background_color=background_color,
                perturb=perturb,
                num_steps=num_steps,
                upsample_steps=upsample_steps,
                eliminate_floater=eliminate_floater,
            )

        if eliminate_floater:
            return image, floater_pack

        return image
