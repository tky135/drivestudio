import torch
from typing import Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pytorch3d.ops import knn_points
import nvdiffrast.torch as dr
from utils.geometry import rotation_6d_to_matrix

logger = logging.getLogger()

class XYZ_Encoder(nn.Module):
    encoder_type = "XYZ_Encoder"
    """Encode XYZ coordinates or directions to a vector."""

    def __init__(self, n_input_dims):
        super().__init__()
        self.n_input_dims = n_input_dims

    @property
    def n_output_dims(self) -> int:
        raise NotImplementedError

class SinusoidalEncoder(XYZ_Encoder):
    encoder_type = "SinusoidalEncoder"
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(
        self,
        n_input_dims: int = 3,
        min_deg: int = 0,
        max_deg: int = 10,
        enable_identity: bool = True,
    ):
        super().__init__(n_input_dims)
        self.n_input_dims = n_input_dims
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.enable_identity = enable_identity
        self.register_buffer(
            "scales", Tensor([2**i for i in range(min_deg, max_deg + 1)])
        )

    @property
    def n_output_dims(self) -> int:
        return (
            int(self.enable_identity) + (self.max_deg - self.min_deg + 1) * 2
        ) * self.n_input_dims

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., n_input_dims]
        Returns:
            encoded: [..., n_output_dims]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[..., None, :] * self.scales[:, None]),
            list(x.shape[:-1])
            + [(self.max_deg - self.min_deg + 1) * self.n_input_dims],
        )
        encoded = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
        if self.enable_identity:
            encoded = torch.cat([x] + [encoded], dim=-1)
        return encoded

class MLP(nn.Module):
    """A simple MLP with skip connections."""

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        num_layers: int = 3,
        hidden_dims: Optional[int] = 256,
        skip_connections: Optional[Tuple[int]] = [0],
    ) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.n_output_dims = out_dims
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(in_dims, out_dims))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    layers.append(nn.Linear(in_dims, hidden_dims))
                elif i in skip_connections:
                    layers.append(nn.Linear(in_dims + hidden_dims, hidden_dims))
                else:
                    layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.Linear(hidden_dims, out_dims))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        input = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                x = torch.cat([x, input], -1)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.functional.relu(x)
        return x
    
class SkyModel(nn.Module):
    def __init__(
        self,
        class_name: str,
        n: int, 
        head_mlp_layer_width: int = 64,
        enable_appearance_embedding: bool = True,
        appearance_embedding_dim: int = 16,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.direction_encoding = SinusoidalEncoder(
            n_input_dims=3, min_deg=0, max_deg=6
        )
        self.direction_encoding.requires_grad_(False)
        
        self.enable_appearance_embedding = enable_appearance_embedding
        if self.enable_appearance_embedding:
            self.appearance_embedding_dim = appearance_embedding_dim
            self.appearance_embedding = nn.Embedding(n, appearance_embedding_dim, dtype=torch.float32)
            
        in_dims = self.direction_encoding.n_output_dims + appearance_embedding_dim \
            if self.enable_appearance_embedding else self.direction_encoding.n_output_dims
        self.sky_head = MLP(
            in_dims=in_dims,
            out_dims=3,
            num_layers=3,
            hidden_dims=head_mlp_layer_width,
            skip_connections=[1],
        )
        self.in_test_set = False
    
    def forward(self, image_infos):
        directions = image_infos["viewdirs"]
        self.device = directions.device
        prefix = directions.shape[:-1]
        
        dd = self.direction_encoding(directions.reshape(-1, 3)).to(self.device)
        if self.enable_appearance_embedding:
            # optionally add appearance embedding
            if "img_idx" in image_infos and not self.in_test_set:
                appearance_embedding = self.appearance_embedding(image_infos["img_idx"]).reshape(-1, self.appearance_embedding_dim)
            else:
                # use mean appearance embedding
                appearance_embedding = torch.ones(
                    (*dd.shape[:-1], self.appearance_embedding_dim),
                    device=dd.device,
                ) * self.appearance_embedding.weight.mean(dim=0)
            dd = torch.cat([dd, appearance_embedding], dim=-1)
        rgb_sky = self.sky_head(dd).to(self.device)
        rgb_sky = F.sigmoid(rgb_sky)
        return rgb_sky.reshape(prefix + (3,))
    
    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }
        
class EnvLight(torch.nn.Module):

    def __init__(
        self,
        class_name: str,
        resolution=1024,
        device: torch.device = torch.device("cuda"),
        **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True),
        )
        
    def forward(self, image_infos):
        l = image_infos["viewdirs"]
        
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape)
        l = l.contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])

        light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        light = light.view(*prefix, -1)

        return light

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }
        
class AffineTransform(nn.Module):
    def __init__(
        self,
        class_name: str,
        n: int, 
        embedding_dim: int = 4,
        pixel_affine: bool = False,
        base_mlp_layer_width: int = 64,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.embedding_dim = embedding_dim
        self.pixel_affine = pixel_affine
        self.embedding = nn.Embedding(n, embedding_dim, dtype=torch.float32)
        
        input_dim = (embedding_dim + 2)if self.pixel_affine else embedding_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, base_mlp_layer_width),
            nn.ReLU(),
            nn.Linear(base_mlp_layer_width, 12),
        )
        self.in_test_set = False
        
        self.zero_init()
        
    def zero_init(self):
        torch.nn.init.zeros_(self.embedding.weight)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, image_infos):
        if "img_idx" in image_infos and not self.in_test_set:
            embedding = self.embedding(image_infos["img_idx"])
        else:
            # use mean appearance embedding
            embedding = torch.ones(
                (*image_infos["viewdirs"].shape[:-1], self.embedding_dim),
                device=image_infos["viewdirs"].device,
            ) * self.embedding.weight.mean(dim=0)
        if self.pixel_affine:
            embedding = torch.cat([embedding, image_infos["pixel_coords"]], dim=-1)
        affine = self.decoder(embedding)
        affine = affine.reshape(*embedding.shape[:-1], 3, 4)
        
        affine[..., :3, :3] = affine[..., :3, :3] + torch.eye(3, device=affine.device).reshape(1, 3, 3)
        return affine

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }
        
class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(
        self,
        class_name: str,
        n: int,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        
        self.zero_init() # important for initialization !!

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }

def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, x_multires=10, t_multires=10):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.x_multires = x_multires
        self.t_multires = t_multires
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(self.x_multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling
    
    
class ConditionalDeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, embed_dim=10,
                 x_multires=10, t_multires=10, 
                 deform_quat=True, deform_scale=True):
        super(ConditionalDeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.embed_dim = embed_dim
        self.deform_quat = deform_quat
        self.deform_scale = deform_scale
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(x_multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch + embed_dim

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 3)
        if self.deform_quat:
            self.gaussian_rotation = nn.Linear(W, 4)
        if self.deform_scale:
            self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t, condition):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb, condition], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, condition, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling, rotation = None, None
        if self.deform_scale: 
            scaling = self.gaussian_scaling(h)
        if self.deform_quat:
            rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling

class VoxelDeformer(nn.Module):
    def __init__(
        self,
        vtx,
        vtx_features,
        resolution_dhw=[8, 32, 32],
        short_dim_dhw=0,  # 0 is d, corresponding to z
        long_dim_dhw=1,
        is_resume=False
    ) -> None:
        super().__init__()
        # vtx B,N,3, vtx_features: B,N,J
        # d-z h-y w-x; human is facing z; dog is facing x, z is upward, should compress on y
        B = vtx.shape[0]
        assert vtx.shape[0] == vtx_features.shape[0], "Batch size mismatch"

        # * Prepare Grid
        self.resolution_dhw = resolution_dhw
        device = vtx.device
        d, h, w = self.resolution_dhw

        self.register_buffer(
            "ratio",
            torch.Tensor(
                [self.resolution_dhw[long_dim_dhw] / self.resolution_dhw[short_dim_dhw]]
            ).squeeze(),
        )
        self.ratio_dim = -1 - short_dim_dhw
        x_range = (
            (torch.linspace(-1, 1, steps=w, device=device))
            .view(1, 1, 1, w)
            .expand(1, d, h, w)
        )
        y_range = (
            (torch.linspace(-1, 1, steps=h, device=device))
            .view(1, 1, h, 1)
            .expand(1, d, h, w)
        )
        z_range = (
            (torch.linspace(-1, 1, steps=d, device=device))
            .view(1, d, 1, 1)
            .expand(1, d, h, w)
        )
        grid = (
            torch.cat((x_range, y_range, z_range), dim=0)
            .reshape(1, 3, -1)
            .permute(0, 2, 1)
        )
        grid = grid.expand(B, -1, -1)

        gt_bbox_min = (vtx.min(dim=1).values).to(device)
        gt_bbox_max = (vtx.max(dim=1).values).to(device)
        offset = (gt_bbox_min + gt_bbox_max) * 0.5
        self.register_buffer(
            "global_scale", torch.Tensor([1.2]).squeeze()
        )  # from Fast-SNARF
        scale = (
            (gt_bbox_max - gt_bbox_min).max(dim=-1).values / 2 * self.global_scale
        ).unsqueeze(-1)

        corner = torch.ones_like(offset) * scale
        corner[:, self.ratio_dim] /= self.ratio
        min_vert = (offset - corner).reshape(-1, 1, 3)
        max_vert = (offset + corner).reshape(-1, 1, 3)
        self.bbox = torch.cat([min_vert, max_vert], dim=1)

        self.register_buffer("scale", scale.unsqueeze(1)) # [B, 1, 1]
        self.register_buffer("offset", offset.unsqueeze(1)) # [B, 1, 3]

        grid_denorm = self.denormalize(
            grid
        )  # grid_denorm is in the same scale as the canonical body

        if not is_resume:
            weights = (
                self._query_weights_smpl(
                    grid_denorm,
                    smpl_verts=vtx.detach().clone(),
                    smpl_weights=vtx_features.detach().clone(),
                )
                .detach()
                .clone()
            )
        else:
            # random initialization
            weights = torch.randn(
                B, vtx_features.shape[-1], *resolution_dhw
            ).to(device)

        self.register_buffer("lbs_voxel_base", weights.detach())
        self.register_buffer("grid_denorm", grid_denorm)

        self.num_bones = vtx_features.shape[-1]

        # # debug
        # import numpy as np
        # np.savetxt("./debug/dbg.xyz", grid_denorm[0].detach().cpu())
        # np.savetxt("./debug/vtx.xyz", vtx[0].detach().cpu())
        return

    def enable_voxel_correction(self):
        voxel_w_correction = torch.zeros_like(self.lbs_voxel_base)
        self.voxel_w_correction = nn.Parameter(voxel_w_correction)

    def enable_additional_correction(self, additional_channels, std=1e-4):
        additional_correction = (
            torch.ones(
                self.lbs_voxel_base.shape[0],
                additional_channels,
                *self.lbs_voxel_base.shape[2:]
            )
            * std
        )
        self.additional_correction = nn.Parameter(additional_correction)

    @property
    def get_voxel_weight(self):
        w = self.lbs_voxel_base
        if hasattr(self, "voxel_w_correction"):
            w = w + self.voxel_w_correction
        if hasattr(self, "additional_correction"):
            w = torch.cat([w, self.additional_correction], dim=1)
        return w

    def get_tv(self, name="dc"):
        if name == "dc":
            if not hasattr(self, "voxel_w_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.voxel_w_correction
        elif name == "rest":
            if not hasattr(self, "additional_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.additional_correction
        tv_x = torch.abs(d[:, :, 1:, :, :] - d[:, :, :-1, :, :]).mean()
        tv_y = torch.abs(d[:, :, :, 1:, :] - d[:, :, :, :-1, :]).mean()
        tv_z = torch.abs(d[:, :, :, :, 1:] - d[:, :, :, :, :-1]).mean()
        return (tv_x + tv_y + tv_z) / 3.0
        # tv_x = torch.abs(d[:, :, 1:, :, :] - d[:, :, :-1, :, :]).sum()
        # tv_y = torch.abs(d[:, :, :, 1:, :] - d[:, :, :, :-1, :]).sum()
        # tv_z = torch.abs(d[:, :, :, :, 1:] - d[:, :, :, :, :-1]).sum()
        # return tv_x + tv_y + tv_z

    def get_mag(self, name="dc"):
        if name == "dc":
            if not hasattr(self, "voxel_w_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.voxel_w_correction
        elif name == "rest":
            if not hasattr(self, "additional_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.additional_correction
        return torch.norm(d, dim=1).mean()

    def forward(self, xc, mode="bilinear"):
        shape = xc.shape  # ..., 3
        # xc = xc.reshape(1, -1, 3)
        w = F.grid_sample(
            self.get_voxel_weight,
            self.normalize(xc)[:, :, None, None],
            align_corners=True,
            mode=mode,
            padding_mode="border",
        )
        w = w.squeeze(3, 4).permute(0, 2, 1)
        w = w.reshape(*shape[:-1], -1)
        # * the w may have more channels
        return w

    def normalize(self, x):
        x_normalized = x.clone()
        x_normalized -= self.offset
        x_normalized /= self.scale
        x_normalized[..., self.ratio_dim] *= self.ratio
        return x_normalized

    def denormalize(self, x):
        x_denormalized = x.clone()
        x_denormalized[..., self.ratio_dim] /= self.ratio
        x_denormalized *= self.scale
        x_denormalized += self.offset
        return x_denormalized

    def _query_weights_smpl(self, x, smpl_verts, smpl_weights):
        # adapted from https://github.com/jby1993/SelfReconCode/blob/main/model/Deformer.py
        dist, idx, _ = knn_points(x, smpl_verts.detach(), K=30) # [B, N, 30]
        dist = dist.sqrt().clamp_(0.0001, 1.0)
        expanded_smpl_weights = smpl_weights.unsqueeze(2).expand(-1, -1, idx.shape[2], -1) # [B, N, 30, J]
        weights = expanded_smpl_weights.gather(1, idx.unsqueeze(-1).expand(-1, -1, -1, expanded_smpl_weights.shape[-1])) # [B, N, 30, J]

        ws = 1.0 / dist
        ws = ws / ws.sum(-1, keepdim=True)
        weights = (ws[..., None] * weights).sum(-2)

        b = x.shape[0]
        c = smpl_weights.shape[-1]
        d, h, w = self.resolution_dhw
        weights = weights.permute(0, 2, 1).reshape(b, c, d, h, w)
        for _ in range(30):
            mean = (
                weights[:, :, 2:, 1:-1, 1:-1]
                + weights[:, :, :-2, 1:-1, 1:-1]
                + weights[:, :, 1:-1, 2:, 1:-1]
                + weights[:, :, 1:-1, :-2, 1:-1]
                + weights[:, :, 1:-1, 1:-1, 2:]
                + weights[:, :, 1:-1, 1:-1, :-2]
            ) / 6.0
            weights[:, :, 1:-1, 1:-1, 1:-1] = (
                weights[:, :, 1:-1, 1:-1, 1:-1] - mean
            ) * 0.7 + mean
            sums = weights.sum(1, keepdim=True)
            weights = weights / sums
        return weights.detach()

def nn_2d_k(A, B, k=1):
    """
    A: (N, 2)
    B: (M, 2) query points
    return: (M) indices of the nearest neighbors in A
    """
    A = A.unsqueeze(0)
    B = B.unsqueeze(1)
    distances = torch.norm(A - B, dim=-1)
    nn_dist, nn_idx = torch.topk(distances, k, dim=-1, largest=False) #[M, k], [M, k]
    return nn_dist, nn_idx


from .fields import SDFNetwork_2d_hash, SingleVarianceNetwork, RenderingNetwork, LabelNetwork
from .renderer import NeuSRenderer
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import os
import open3d as o3d
import cv2
idx2color = [[157, 234, 50], [211, 211, 202], [233, 74, 127], [85, 37, 136], [250, 220, 2], [157, 234, 50]]

class Ground(nn.Module):
    def __init__(self):
        super(Ground, self).__init__()
        # initialize everything model
        
        self.device = torch.device('cuda')
        
        
        # confs
        self.end_iter = 1000000
        self.max_iter = 50000
        self.save_freq = 10000
        self.report_freq = 1000
        self.val_freq = 1000
        self.val_mesh_freq = 10000
        self.batch_size = 1024
        self.validate_resolution_level = 4
        self.learning_rate = 5e-4
        self.learning_rate_alpha = 0.05
        self.use_white_bkgd = False
        self.warm_up_end = 1000
        self.anneal_end = 50000
        
        
        # models
        self.igr_weight = 5.0
        self.mask_weight = 0.1
        self.ssim_weight = 0.0
        self.sdf_network = SDFNetwork_2d_hash(**{'d_out': 257, 'd_in': 2, 'd_hidden': 256, 'n_layers': 8, 'skip_in': [4], 'multires': 6, 'bias': 0.5, 'scale': 10.0, 'geometric_init': True, 'weight_norm': True}, base_resolution=256, num_clusters=1).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**{'init_val': 0.3}).to(self.device)
        self.color_network = RenderingNetwork(**{'d_feature': 64, 'mode': 'xy_embed', 'd_in': 9, 'd_out': 3, 'd_hidden': 256, 'n_layers': 4, 'weight_norm': True, 'multires_view': 4, 'squeeze_out': True}, n_camera=6).to(self.device)
        self.label_network = LabelNetwork(**{'d_feature': 64, 'd_in': 2, 'd_out': 5, 'd_hidden': 256, 'n_layers': 4}).to(self.device)
        self.renderer = NeuSRenderer(None,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.label_network,
                                     **{'n_samples': 8, 'n_importance': 16, 'n_outside': 0, 'up_sample_steps': 2, 'perturb': 1.0})
        
        
        # optimizers
        params_to_train = []
        self.prior_optimizer = torch.optim.Adam(self.sdf_network.prior_network.parameters(), lr=0.01)
        params_to_train += list(self.sdf_network.tiny_mlp.parameters()) + list(self.sdf_network.hash_encoding.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        params_to_train += list(self.label_network.parameters())
        self.optimizer = torch.optim.Adam([{'params':params_to_train, 'name': 'all_other'}], lr=self.learning_rate)

        self.ego_points = None
        self.ego_normals = None
        self.scale_mat = np.asarray([[500, 0, 0, 0],
                                       [0, 500, 0, 0],
                                       [0, 0, 9, 0],
                                       [0, 0, 0, 1]], dtype=np.float32)
        self.omnire_w2neus_w = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 1.51], [0, 0, 0, 1]]).type(torch.float32).to(self.device)
        # self.pretrain_sdf()
        
        self.iter_step = 0
        self.base_exp_dir = '3_6cams_selfsup'
        if not os.path.exists(self.base_exp_dir):
            os.makedirs(self.base_exp_dir)
        
        
        
    def validate_image(self, image_infos, camera_infos, idx=-1):
        # assert idx >= 0

        # print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        rays_d = image_infos['viewdirs'] @ self.omnire_w2neus_w[:3, :3].T @ torch.linalg.inv(torch.from_numpy(self.scale_mat[:3, :3])).cuda().float().T
        camera_encod = camera_infos['cam_id']
        c2w = camera_infos['camera_to_world']
        gt_img = image_infos['pixels']
        c2w = self.omnire_w2neus_w @ c2w
        c2w = torch.linalg.inv(torch.from_numpy(self.scale_mat)).cuda().float() @ c2w
        H, W, _ = gt_img.shape
        # mask = torch.ones([H, W], device=self.device)
        # mask[image_infos['dynamic_masks'] == 1] = 0
        # mask[image_infos['sky_masks'] == 1] = 0
        # mask[:int(3 / 5 * H), :] = 0
        mask = image_infos['road_masks']
        mask = mask.cpu().numpy()
        
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays_o = c2w[:3, 3].unsqueeze(0).expand(rays_d.shape)
        
        # rays_d = torch.matmul(rays_d, extrinsic_modification[:3, :3].T)
        # rays_o = torch.matmul(rays_o, extrinsic_modification[:3, :3].T) + extrinsic_modification[:3, 3]
        
        camera_encod = camera_infos['cam_id'].reshape(-1, 1)[0]
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_rgb_orig = []
        out_normal_fine = []
        out_label_fine = []
        out_depth_fine = []
        out_beta = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near = torch.zeros_like(rays_o_batch[:, 0])
            far = torch.ones_like(rays_o_batch[:, 0]) * 0.5
            near = near.unsqueeze(-1)
            far = far.unsqueeze(-1)

            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb,
                                              camera_encod=camera_encod)

            out_rgb_orig.append(render_out['orig_color'].detach().cpu().numpy())

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            if feasible('label'):
                out_label_fine.append(render_out['label'].detach().cpu().numpy())
            out_depth_fine.append(render_out['depth'].detach().cpu().numpy())
            out_beta.append(render_out['beta'].detach().cpu().numpy())
            del render_out
            
        img_fine = None
        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 255).clip(0, 255).astype(np.uint8)
        if len(out_rgb_orig) > 0:
            img_orig = (np.concatenate(out_rgb_orig, axis=0).reshape([H, W, 3, -1]) * 255).clip(0, 255).astype(np.uint8)
        label_img = None
        if len(out_label_fine) > 0:
            label_img = np.concatenate(out_label_fine, axis=0)
            label_img = np.argmax(label_img, axis=-1)
            label_img_gt_raw = label_img
            label_img = np.array(idx2color)[label_img].reshape([H, W, 3, -1]).clip(0, 255)
            label_img_gt = np.array(idx2color)[label_img_gt_raw].reshape([H, W, 3]).clip(0, 255)
        gt_img = gt_img.cpu().numpy() * 255
        road_mask = label_img_gt_raw > 0
        gaussian_img = image_infos['gaussian_rgb'].cpu().numpy() * 255
        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                rgb_diff = np.abs(img_fine[..., i] - gt_img)
                rgb_diff[mask == 0] = np.array([255, 0, 0])
                image_cat = np.concatenate([img_fine[..., i],
                                            gt_img, rgb_diff])
                orig_cat = np.concatenate([img_orig[..., i],
                                           gaussian_img, 
                                           img_orig[..., i]])
                label_diff = np.abs(label_img[..., i] - label_img_gt)
                # label_diff[~road_mask] = 0
                label_cat = np.concatenate([label_img[..., i], label_img_gt, label_diff])
                
                output_img = np.concatenate([orig_cat, image_cat, label_cat], axis=1).astype(np.uint8)
                cv2.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                            cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                        )
            
        
    def points2cluster(self, points, k=1, is_world=False, kd_tree=True, return_idx=False):
        """
        points: np.ndarray/torch.tensor, shape (N, 3)
        return: torch.tensor, shape (N, k)
        """
        if type(points) == np.ndarray:
            points = torch.from_numpy(points).cpu()
        if kd_tree:
            scale_mat = self.scale_mat[:3, :3]
        else:
            scale_mat = torch.from_numpy(self.scale_mat[:3, :3]).cuda()
        if not is_world:
            points_world = points @ scale_mat
        else:
            points_world = points

        modified_ego_points = self.ego_points.copy()
        if not kd_tree:
            modified_ego_points = torch.tensor(modified_ego_points, device=points.device)

        modified_ego_points[:, 2] *= 3
        points_world[:, 2] *= 3
        
        if kd_tree:
            kd_tree = cKDTree(modified_ego_points)
            # filter based on closest ego point
            vertices_prior_world_cpu = points_world
            dist, idx = kd_tree.query(vertices_prior_world_cpu, k=k)
        else:
            dist, idx = nn_2d_k(modified_ego_points, points_world, k=k)
        # grid2cluster = ego2cluster[idx]
        grid2cluster = torch.zeros_like(idx)

        if return_idx:
            return grid2cluster, idx
        else:
            return grid2cluster
        
    def pretrain_sdf(self):
        
        if type(self.ego_points) == torch.Tensor:
            self.ego_points = self.ego_points.cpu().numpy()
        if type(self.ego_normals) == torch.Tensor:
            self.ego_normals = self.ego_normals.cpu().numpy()
        
        world_ego_points = self.ego_points
        bev_ego_points = world_ego_points @ np.linalg.inv(self.scale_mat[:3, :3]) - self.scale_mat[:3, 3]
        bev_ego_points = torch.from_numpy(bev_ego_points).to(self.device).float()
        world_ego_points = torch.from_numpy(world_ego_points).to(self.device).float()

        world_ego_normals = self.ego_normals
        
        world_ego_normals = torch.from_numpy(world_ego_normals).to(self.device)

        world_coff_A, world_coff_B = world_ego_normals[:, 0] / world_ego_normals[:, 2], world_ego_normals[:, 1] / world_ego_normals[:, 2]
        self.sdf_network.prior_network.world_coff_A = world_coff_A
        self.sdf_network.prior_network.world_coff_B = world_coff_B
        self.sdf_network.prior_network.world_ego_points = world_ego_points
        loop = tqdm(range(20000))
        loop.set_description("initializing sdf with ego points normal")
        for it in loop:
            # generate random xyz around ego points
            bev_rand_xyz = bev_ego_points + torch.randn(bev_ego_points.shape, device=self.device) / 100
            world_rand_xyz = bev_rand_xyz @ torch.from_numpy(self.scale_mat[:3, :3]).cuda().float()


            cluster_idx, nn_idx_more = self.points2cluster(world_rand_xyz, k=1, is_world=True, kd_tree=False, return_idx=True)
            # find closest ego point for each random xyz
            # nn_dist, nn_idx_more = nn_2d_k(world_ego_points, world_rand_xyz, k=8)
            
            # nn_idx = nn_2d(world_ego_points, world_rand_xyz)
            choice = torch.randint(0, 1, [bev_rand_xyz.shape[0]])
            nn_idx = nn_idx_more[torch.arange(bev_rand_xyz.shape[0]), choice]

            # get cluster for each random xyz
            cluster_idx = cluster_idx[:, 0]

            # get target z (each random xyz should be on the plane of the closest ego point)
            world_rand_z = -world_coff_A[nn_idx] * (world_rand_xyz[:, 0] - world_ego_points[nn_idx, 0]) - world_coff_B[nn_idx] * (world_rand_xyz[:, 1] - world_ego_points[nn_idx, 1]) + world_ego_points[nn_idx, 2]
            bev_rand_z = world_rand_z / self.scale_mat[2, 2]


            # iterate over clusters
            for i in range(1):
                self.prior_optimizer.zero_grad()
                mask = cluster_idx == i
                input_xy = bev_rand_xyz[mask, :2]
                target_z = bev_rand_z[mask]
                prior_sdf = self.sdf_network.prior_network[i](input_xy) # select i-th prior network
                prior_loss = F.l1_loss(prior_sdf.squeeze(), target_z.squeeze())
                prior_loss.backward()
                self.prior_optimizer.step()

                if it < 5000:
                    sdf = self.sdf_network(bev_rand_xyz[:, :2], delta=False, force_cluster=i)[:, 0]
                    loss = torch.abs(sdf).mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        
        
    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])
    def forward(self, image_infos, camera_infos) -> torch.Tensor:
        """
        image_infos: {
            'origins': torch.Tensor, [900 / d, 1600 / d, 3]. 都是同一个origin
            'viewdirs': torch.Tensor, [900 / d, 1600 / d, 3]. 
            'direction_norm': torch.Tensor, [900 / d, 1600 / d, 1]. ???
            'pixel_coords': torch.Tensor, [900 / d, 1600 / d, 2]. normalized pixel coordinates
            'normed_time': torch.Tensor, [900 / d, 1600 / d]. normalized time. 猜测是整个bag的时间戳在0-1之间的归一化
            'img_idx': torch.Tensor, [900 / d, 1600 / d]. 
            'frame_idx': torch.Tensor, [900 / d, 1600 / d].
            'pixels': torch.Tensor, [900 / d, 1600 / d, 3]. RGB
            'sky_masks': torch.Tensor, [900 / d, 1600 / d]. 估计1代表天空
            'dynamic_masks': torch.Tensor, [900 / d, 1600 / d]. 
            'human_masks': torch.Tensor, [900 / d, 1600 / d].
            'vehicle_masks': torch.Tensor, [900 / d, 1600 / d].
            'lidar_depth_map': torch.Tensor, [900 / d, 1600 / d].
        }
        
        camera_infos: {
            'cam_id': torch.Tensor, [900 / d, 1600 / d].
            'cam_name': str.
            'camera_to_world': torch.Tensor, [4, 4]. #TODO: nuscenes相机高度从哪里来
            'height': torch.Tensor, [1]. image height
            'width': torch.Tensor, [1]. image width
            'intrinsics': torch.Tensor, [3, 3].
        }
        
        return rgb
        """
        
        # if self.iter_step % 100 == 0:
        #     self.validate_image(image_infos, camera_infos)
        # if self.iter_step % 1000 == 0:
        #     self.validate_mesh()
        
        H, W, _ = image_infos['viewdirs'].shape

        mask = image_infos['road_masks']
        rays_d = image_infos['viewdirs'] @ self.omnire_w2neus_w[:3, :3].T @ torch.linalg.inv(torch.from_numpy(self.scale_mat[:3, :3])).cuda().float().T
        camera_encod = camera_infos['cam_id']
        rays_d = rays_d[mask == 1]
        camera_encod = camera_encod[mask == 1]
        gt_rgb = image_infos['pixels'][mask == 1]
        
        # rand_indices = torch.randint(0, rays_d.shape[0], [self.batch_size])
        # rays_d = rays_d[rand_indices]
        # camera_encod = camera_encod[rand_indices]
        # gt_rgb = gt_rgb[rand_indices]
        
        c2w = camera_infos['camera_to_world']
        c2w = self.omnire_w2neus_w @ c2w
        
        c2w = torch.linalg.inv(torch.from_numpy(self.scale_mat)).cuda().float() @ c2w
        
        # gt reshape
        
        near = torch.zeros_like(rays_d[:, 0], device=self.device)
        far = torch.ones_like(rays_d[:, 0], device=self.device) * 0.5
        near = near.unsqueeze(-1)
        far = far.unsqueeze(-1)
        
        modified_c2w = c2w
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays_o = modified_c2w[:3, 3].unsqueeze(0).expand(rays_d.shape)

        self.sdf_network.iter_step = self.iter_step
        self.renderer.iter_step = self.iter_step
        
        self.sdf_network.requires_grad_(True)

        render_out = self.renderer.render(rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), near.reshape(-1, 1), far.reshape(-1, 1),
                                        background_rgb=None,
                                        cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                        camera_encod=camera_encod.reshape(-1))
        render_out['gt_rgb'] = gt_rgb
        rgb_full = torch.zeros_like(image_infos['pixels'])
        rgb_full[mask == 1] = render_out['color_fine']
        render_out['rgb_full'] = rgb_full
        self.iter_step += 1
        return render_out
    def get_loss(self, render_out, image_infos, camera_infos):
        """
        image_infos: {
            'origins': torch.Tensor, [900 / d, 1600 / d, 3]. 都是同一个origin
            'viewdirs': torch.Tensor, [900 / d, 1600 / d, 3]. 
            'direction_norm': torch.Tensor, [900 / d, 1600 / d, 1]. ???
            'pixel_coords': torch.Tensor, [900 / d, 1600 / d, 2]. normalized pixel coordinates
            'normed_time': torch.Tensor, [900 / d, 1600 / d]. normalized time. 猜测是整个bag的时间戳在0-1之间的归一化
            'img_idx': torch.Tensor, [900 / d, 1600 / d]. 
            'frame_idx': torch.Tensor, [900 / d, 1600 / d].
            'pixels': torch.Tensor, [900 / d, 1600 / d, 3]. RGB
            'sky_masks': torch.Tensor, [900 / d, 1600 / d]. 估计1代表天空
            'dynamic_masks': torch.Tensor, [900 / d, 1600 / d]. 
            'human_masks': torch.Tensor, [900 / d, 1600 / d].
            'vehicle_masks': torch.Tensor, [900 / d, 1600 / d].
            'lidar_depth_map': torch.Tensor, [900 / d, 1600 / d].
        }
        
        camera_infos: {
            'cam_id': torch.Tensor, [900 / d, 1600 / d].
            'cam_name': str.
            'camera_to_world': torch.Tensor, [4, 4]. #TODO: nuscenes相机高度从哪里来
            'height': torch.Tensor, [1]. image height
            'width': torch.Tensor, [1]. image width
            'intrinsics': torch.Tensor, [3, 3].
        }
        
        return rgb
        """
        true_rgb = render_out['gt_rgb']
        B, _ = true_rgb.shape
        
        # mask = torch.ones([H, W], device=self.device)
        # mask[image_infos['dynamic_masks'] == 1] = 0
        # mask[image_infos['sky_masks'] == 1] = 0
        # mask[:int(3 / 5 * H), :] = 0
        # # gt_seg = data['seg'].to(self.device)
        # image_batch_size, batch_size = H, W
        

        color_fine = render_out['color_fine']
        gradient_error = render_out['gradient_error']
        # label = render_out['label']
        delta_loss = render_out['delta_loss']
        delta_color = render_out['delta_color']
        s_val = render_out['s_val']
        
        delta_color_loss = F.mse_loss(delta_color, torch.zeros_like(delta_color))
        
        
        color_error = (color_fine - true_rgb)
        color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='mean')

        bev_to_world_factor = 9 ** 2 / 500 ** 2
        eikonal_loss = gradient_error * bev_to_world_factor

        if self.iter_step == 1:
            print('eikonal_loss weight: {}'.format(bev_to_world_factor * self.igr_weight))
        loss = color_fine_loss +\
            eikonal_loss * self.igr_weight  +\
            delta_loss * 0

        loss += delta_color_loss * 0.05 * 10
        
        # s_val loss
        loss += s_val.mean()

        return loss
    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def validate_mesh(self, cluster_idx=0):
        
        xy = torch.meshgrid(torch.arange(-1, 1, 0.05 / self.scale_mat[0, 0], device=torch.device("cpu")), torch.arange(-1, 1, 0.05 / self.scale_mat[1, 1], device=torch.device("cpu")))
        xy_grid = torch.stack([xy[0], xy[1]], dim=-1)
        num_pixels = xy_grid.shape[0]
        canvas = np.zeros([num_pixels, num_pixels], dtype=np.uint8)
        scale_xy = self.scale_mat[0, 0]
        for x, y, z in self.ego_points:
            x = int((x / scale_xy + 1) / 2 * num_pixels)
            y = int((y / scale_xy + 1) / 2 * num_pixels)
            canvas[x, y] = 1
        
        # dilate 
        canvas_big = cv2.dilate(canvas, np.ones((1000, 1000), np.uint8), iterations=1)
        xy = xy_grid[canvas_big.astype(np.bool_)]
        xy = xy.cuda()


        # filter based on ego points
        torch.cuda.empty_cache()
        xy_world = xy @ torch.from_numpy(self.scale_mat[:2, :2]).cuda() + torch.from_numpy(self.scale_mat[:2, 3]).cuda()
        mask_xy = torch.zeros(xy_world.shape[0], dtype=torch.bool, device=xy_world.device)

        for i in tqdm(range(len(self.ego_points)), desc='Filtering'):
            x, y, _ = self.ego_points[i]
            dist = torch.norm(xy_world - torch.tensor([x, y], dtype=torch.float32, device=xy_world.device), dim=-1)
            mask = dist < 15
            mask_xy = mask_xy | mask
        xy = xy[mask_xy]
        
            
            
        print("# of points after cropping: " + str(xy.shape[0]))
        BATCH_SIZE = 4096
        torch.cuda.empty_cache()
        with torch.no_grad():
            z = torch.zeros(xy.shape[0], dtype=torch.float32, device=xy.device)
            z_prior = torch.zeros(xy.shape[0], dtype=torch.float32, device=xy.device)
            for i in range(0, xy.shape[0], BATCH_SIZE):
                xy_batch = xy[i:i + BATCH_SIZE]
                sdf_nn_output_batch = self.sdf_network(xy_batch, output_height=True, force_cluster=cluster_idx)
                z[i:i + BATCH_SIZE] = sdf_nn_output_batch[:, 0]
                
                z_prior[i:i + BATCH_SIZE] = self.sdf_network.prior_network[cluster_idx](xy_batch).squeeze(-1)
            z = z.unsqueeze(-1)
            z_prior = z_prior.unsqueeze(-1)
        vertices = torch.cat([xy, z], dim=-1)
        vertices_prior = torch.cat([xy, z_prior], dim=-1)
        
        vertices_prior_world = vertices_prior @ torch.from_numpy(self.scale_mat[:3, :3]).cuda() + torch.from_numpy(self.scale_mat[:3, 3]).cuda()
        
        # 获得ego pose to cluster #
        # grid2cluster = self.points2cluster(vertices_prior_world.cpu().numpy(), k=1, is_world=True)
        
        # mask = grid2cluster == cluster_idx
        
        # vertices = vertices[mask]
        # z = z[mask]
        # z_prior = z_prior[mask]
        # xy = xy[mask]

        delta_z = (z - z_prior) * self.scale_mat[2, 2]
        delta_z = delta_z.detach().cpu().numpy()

        vertices_prior = torch.cat([xy, z_prior], dim=-1)
        vertices_prior = vertices_prior @ torch.from_numpy(self.scale_mat[:3, :3]).cuda() + torch.from_numpy(self.scale_mat[:3, 3]).cuda()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices_prior.detach().cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(self.base_exp_dir, "prior_{}.pcd".format(cluster_idx)), pcd)

        colors, betas = self.renderer.extract_color(vertices, cluster_idx=cluster_idx)
        labels = self.renderer.extract_labels(vertices, cluster_idx=cluster_idx)
        vertices = vertices.detach().cpu().numpy()
        vertices = vertices @ self.scale_mat[:3, :3] + self.scale_mat[:3, 3]
        colors = torch.stack([colors[:, 2], colors[:, 1], colors[:, 0]], dim=-1)
        colors = colors.detach().cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(self.base_exp_dir, '2d_%d_%d.pcd' % (self.iter_step, cluster_idx)), pcd)
        


        labels = labels.argmax(dim=-1).detach().cpu().numpy()
        
        label_colors = np.array(idx2color)[labels].astype(np.float32) / 255.

        pcd.colors = o3d.utility.Vector3dVector(label_colors)
        o3d.io.write_point_cloud(os.path.join(self.base_exp_dir, '2d_label_%d_%d.pcd' % (self.iter_step, cluster_idx)), pcd)
        
        
        
        # save ego_points
        all_points = []
        for i in range(len(self.ego_points)):
            for j in np.arange(0, 1, 0.1):
                all_points.append(self.ego_points[i] + self.ego_normals[i] * j)
        all_points = np.array(all_points)

        # save ego points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        o3d.io.write_point_cloud(os.path.join(self.base_exp_dir, "ego_points.pcd"), pcd)
        
    def get_param_groups(self):
        return {
            "Ground#"+"all": self.parameters(),
        }
