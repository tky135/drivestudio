import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
import cv2
import open3d as o3d
def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u




def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 label_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.label_network = label_network
        self.counter = 0
        self.sample_points = []
        self.scale_matrix = None
        self.iter_step = -1

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        # 获得射线上的点
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        # 每个线段是否和单位球相交
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        #inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        # 用线段顶点的sdf和z_val（线段上的长度）计算线段中点的sdf和cos
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=rays_o.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0)# * inside_sphere # if not inside sphere, weight = 0

        # 计算alpha和weights
        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5  # 部分来自于sdf输出，部分来自于几何
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(    # T_i\alpha_i
            torch.cat([torch.ones([batch_size, 1], device=rays_o.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        # 重新采样 n_importance 个点
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]    # new points
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            # 正确排列sdf
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1) # 其实n_samples是之前（z_vals)的个数，n_importance是（new_z_vals）的个数
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0,
                    label_network=None,
                    camera_encod=None,
                    is_test=False):
        batch_size, n_samples = z_vals.shape
        camera_encod = camera_encod.unsqueeze(-1).expand(batch_size, n_samples).reshape(-1)
        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([sample_dist], device=dists.device).expand(dists[..., :1].shape)], -1)    # 补一个 sample_dist 作为最后一个线段的长度
        mid_z_vals = z_vals + dists * 0.5   # 每个线段的中点z_vals

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3，每个线段的中点xyz

        # 在这里把点存下来
        # new_pts = pts[:50, :, :].reshape(-1, 3)
        # self.sample_points.append(new_pts)
        # self.counter += 1
        # if self.counter == 20:
        #     sample_points = torch.cat(self.sample_points, dim=0).cpu()
        #     print(sample_points[:50])
        #     print(self.scale_matrix)
        #     world_sample_points = sample_points @ self.scale_matrix[:3, :3].T + self.scale_matrix[:3, 3]
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(world_sample_points.numpy())
        #     o3d.io.write_point_cloud('sample_points.pcd', pcd)
        #     raise Exception
        
        # 计算depth
        dirs = rays_d[:, None, :].expand(pts.shape)
        origs = rays_o[:, None, :].expand(pts.shape)
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        origs = origs.reshape(-1, 3)
        depth = torch.linalg.norm((pts - origs)[:, :2], ord=2, dim=-1).reshape(batch_size, n_samples)


        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]

        abs_sdf = torch.abs(sdf).reshape(batch_size, n_samples)
        min_abs_sdf_idx = torch.argmin(abs_sdf, dim=-1)
        depth = depth[torch.arange(batch_size), min_abs_sdf_idx]
        
        
        feature_vector = sdf_nn_output[:, 1:]

        # TODO: 不要算两次
        delta = sdf_network(pts, delta=False)[:, 0]
        delta_loss = torch.mean(delta ** 2)

        gradients = sdf_network.gradient(pts).squeeze()
        # sampled_color, sampled_delta_color, sampled_beta = color_network(pts, gradients, dirs, feature_vector, camera_encod, is_test=is_test)
        color_output_dict = color_network(feature_vector, pts, gradients, dirs, camera_encod)
        sampled_color, sampled_delta_color, sampled_beta, sampled_orig_color = color_output_dict["rgb"], color_output_dict["delta"], color_output_dict["beta"], color_output_dict["rgb_orig"]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        sampled_orig_color = sampled_orig_color.reshape(batch_size, n_samples, 3)
        sampled_label = label_network(pts[:, :2], feature_vector).reshape(batch_size, n_samples, 5)
        sampled_beta = sampled_beta.reshape(batch_size, n_samples, 1)
        

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1) # larger inv_s means less deviation

        true_cos = (dirs * gradients).sum(-1, keepdim=True)     # gradients: surface normal direction, dirs: ray direction

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        # pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = torch.ones((batch_size, n_samples), device=pts.device)
        relax_inside_sphere = inside_sphere

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        orig_color = (sampled_orig_color * weights[:, :, None]).sum(dim=1)
        seg_label = (sampled_label * weights[:, :, None]).sum(dim=1)
        beta = (sampled_beta * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'orig_color': orig_color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere, 
            'label': seg_label,
            'delta_loss': delta_loss, 
            'depth': depth, 
            'delta_color': sampled_delta_color, 
            'beta': beta
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0, camera_encod=None, is_test=False):
        batch_size = len(rays_o)
        sample_dist = 1.0 / self.n_samples   #TODO??? 2.0 -> 1.0
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=rays_o.device)
        z_vals = near + (far - near) * z_vals[None, :]  # 初始z_vals 是在near和far之间的均匀采样点

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1], device=rays_o.device) - 0.5)
            z_vals = z_vals + t_rand * sample_dist # sample_dist * [-0.5, 0.5] + z_vals

        background_alpha = None
        background_sampled_color = None

        # Up sample, 原始的均匀z_vals被干掉
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]    # [batch_size, n_samples, 3]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**0) # 越来越小的variance
                new_z_vals, index = torch.sort(new_z_vals, dim=-1)
                
                z_vals = new_z_vals
                pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_importance // self.up_sample_steps)

                for i in range(1, self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i) # 越来越小的variance
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_importance



        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio, 
                                    label_network=self.label_network,
                                    camera_encod=camera_encod,
                                    is_test=is_test)

        color_fine = ret_fine['color']
        labels = ret_fine['label']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            'orig_color': ret_fine['orig_color'],
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'], 
            'label': labels,
            'delta_loss': ret_fine['delta_loss'],
            'depth': ret_fine['depth'], 
            'delta_color': ret_fine['delta_color'],
            'beta': ret_fine['beta']
        }

    def extract_color(self, pts, cameras=None, **kwargs):
        """
        pts: [N, 3]
        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]
        """

        batch_size = 2000
        self.sdf_network.eval()

        sample_color = torch.zeros([pts.shape[0], 3], device=pts.device)
        sample_beta = torch.zeros([pts.shape[0]], device=pts.device)

        for i in tqdm(range(0, pts.shape[0], batch_size), desc='Extracting color'):
            pts_batch = pts[i:i + batch_size].clone()
            sdf_nn_output = self.sdf_network(pts_batch, force_cluster=kwargs['cluster_idx'])
            # gradients = self.sdf_network.gradient(pts_batch).squeeze()
            with torch.no_grad():
                sdf = sdf_nn_output[:, :1]
                feature_vector = sdf_nn_output[:, 1:]
                outputs = self.color_network(feature_vector)
                local_color = outputs['rgb_orig']
                local_beta = outputs['beta']                
                sample_color[i:i + batch_size] = local_color.detach()
                sample_beta[i:i + batch_size] = local_beta.detach()


        return sample_color, sample_beta

    def extract_labels(self, pts, **kwargs):
        """
        pts: [N, 3]
        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]
        """

        batch_size = 2000
        self.sdf_network.eval()

        sample_label = torch.zeros([pts.shape[0], 5], device=pts.device)

        for i in tqdm(range(0, pts.shape[0], batch_size), desc='Extracting labels'):
            pts_batch = pts[i:i + batch_size].clone()
            sdf_nn_output = self.sdf_network(pts_batch, force_cluster=kwargs['cluster_idx'])
            with torch.no_grad():
                feature_vector = sdf_nn_output[:, 1:]
                local_label = self.label_network(pts_batch[:, :2], feature_vector)
                sample_label[i:i + batch_size] = local_label.detach()

        return sample_label
    

    # TODO: extract_color using rendering to blend colors
    def extract_color_v2(self, pts):
        
        """
        pts: [N, 3] \in [-1, 1]^3

        """
        batch_size = 1000
        self.sdf_network.eval()
        self.color_network.eval()


        sample_color = torch.zeros([pts.shape[0], 3], device=pts.device)

        sample_color.requires_grad = False
        for i in tqdm(range(0, pts.shape[0], batch_size), desc='Extracting color v2'):
            pts_batch = pts[i:i + batch_size].clone()
            rays_o = pts_batch
            rays_d = torch.Tensor([0, 0, -1]).expand_as(rays_o)
            near = torch.ones([pts_batch.shape[0], 1]).to(pts.device) * (-0.025)
            far = torch.ones([pts_batch.shape[0], 1]).to(pts.device) * 0.025
            # print(rays_o.shape, rays_d.shape, near.shape, far.shape)
            ret = self.render(rays_o, rays_d, near, far, perturb_overwrite=0)
            sample_color[i:i + batch_size] = ret['color_fine'].detach().clone()
        
        return sample_color
            




    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
