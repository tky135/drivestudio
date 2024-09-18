import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder
import tinycudann as tcnn

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()
        inside_outside = True
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

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


class SDFNetwork_2d(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork_2d, self).__init__()

        assert d_in == 2
        inside_outside = True
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
        self.A, self.B, self.C = None, None, None

    def forward(self, inputs, output_height=False, delta=True, gradient=False):
        if delta:
            prior_z = inputs[:, 0] * self.A + inputs[:, 1] * self.B + self.C
            prior_z = prior_z.unsqueeze(-1)
        z_vals = inputs[:, 2:]
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs[:, :2])
        ###########################################
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        ############################################
        if not delta:
            raise Exception("Not updated")
            if output_height:
                return torch.cat([x[:, :1] / self.scale, x[:, 1:], z_vals], dim=-1)
            else:
                return torch.cat([z_vals - x[:, :1] / self.scale, x[:, 1:]], dim=-1)
        else:
            if gradient:
                return torch.cat([z_vals - x[:, :1] / self.scale, x[:, 1:]], dim=-1)
            if output_height:
                return torch.cat([x[:, :1] / self.scale + prior_z, x[:, 1:], z_vals], dim=-1)
            # elif output_delta:
            #     return torch.cat([x[:, :1] / self.scale, x[:, 1:], z_vals - x[:, :1] / self.scale - prior_z], dim=-1)
            else:
                # print(z_vals.shape, x[:, :1].shape, prior_z.shape)
                return torch.cat([z_vals - x[:, :1] / self.scale - prior_z, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x, gradient=True)[:, :1]
        # y = self.forward(x, gradient=False)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class LabelNetwork(nn.Module):
    def __init__(self, d_feature, d_in, d_out, d_hidden, n_layers):
        super().__init__()

        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        # no 
        # self.num_layers = len(dims)
        # for i in range(self.num_layers - 1):
        #     out_dim = dims[i + 1]
        #     lin = nn.Linear(dims[i], out_dim)
        #     lin = nn.utils.weight_norm(lin)
        #     setattr(self, "lin" + str(i), lin)
        # self.relu = nn.ReLU()
        self.mlp = tcnn.Network(n_input_dims=2 + d_feature, n_output_dims=5, network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2
        })
        
    def forward(self, xys, feature_vectors):
        x = torch.cat([xys, feature_vectors], dim=-1)
        # print(x.shape)
        # for i in range(self.num_layers - 1):
        #     lin = getattr(self, "lin" + str(i))
        #     x = lin(x)
        #     if i < self.num_layers - 2:
        #         x = self.relu(x)
        x = self.mlp(x)
        # print("out", x.shape)

        # raise Exception
        return x
    

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True, 
                 n_camera=6):
        super().__init__()

        self.mode = mode
        self.n_camera = n_camera
        self.enable_camera_encod = True
        if self.mode == "view_dir":
            view_embedview_fn, view_input_ch = get_embedder(4)    # 调整超参数
            self.view_embedview_fn = view_embedview_fn
            n_input_dims = 6 + view_input_ch + d_feature + n_camera if self.enable_camera_encod else 6 + view_input_ch + d_feature
        elif self.mode == "xy_embed":
            view_embedview_fn, view_input_ch = get_embedder(4)    # 调整超参数
            self.view_embedview_fn = view_embedview_fn

            xy_embed_fn, xy_input_ch = get_embedder(4, input_dims=2)
            self.xy_embed_fn = xy_embed_fn

            n_input_dims = 6 + view_input_ch + xy_input_ch + d_feature + n_camera

        else:
            raise Exception("Not implemented")

        self.mlp1 = tcnn.Network(n_input_dims=n_input_dims - n_camera, n_output_dims=64, network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2
        })

        self.mlp2 = tcnn.Network(n_input_dims=64 + n_camera, n_output_dims=4, network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 32,
            "n_hidden_layers": 2
        })
        
        
        
        self.orig_rgb = tcnn.Network(d_feature, 4, network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 32,
            "n_hidden_layers": 2
        }
        )

        self.mode = mode
        # self.squeeze_out = squeeze_out
        # dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        # self.embedview_fn = None

        # if self.mode == "no_view_dir":
        #     dims[0] = 6 + d_feature
        # elif self.mode == "idr":
        #     if multires_view > 0:
        #         embedview_fn, input_ch = get_embedder(multires_view)
        #         self.embedview_fn = embedview_fn
        #         dims[0] += (input_ch - 3)
        # elif self.mode == "no_normal_no_dir":
        #     dims[0] = 3 + d_feature
        # else:
        #     raise Exception("Not implemented")

        # self.num_layers = len(dims)

        # for l in range(0, self.num_layers - 1):
        #     out_dim = dims[l + 1]
        #     lin = nn.Linear(dims[l], out_dim)

        #     if weight_norm:
        #         lin = nn.utils.weight_norm(lin)

        #     setattr(self, "lin" + str(l), lin)

        # self.relu = nn.ReLU()

    def forward(self, feature_vectors, points=None, normals=None, view_dirs=None, camera_encod=None):
        if points == None: # color extraction mode
            output = self.orig_rgb(feature_vectors)
            return {"rgb_orig": torch.sigmoid(output[:, :3]), "beta": output[:, 3]}
        # training mode
        if self.view_embedview_fn is not None:
            view_dirs = self.view_embedview_fn(view_dirs)
        if self.mode == 'view_dir':
            encod = torch.zeros([points.shape[0], self.n_camera], device=points.device)
            encod[torch.arange(encod.shape[0]), camera_encod] = 1
            rendering_input = torch.cat([points, normals, view_dirs, feature_vectors, encod], dim=-1)
        elif self.mode == 'xy_embed':
            encod = torch.zeros([points.shape[0], self.n_camera], device=points.device)
            encod[torch.arange(encod.shape[0]), camera_encod] = 1
            xy_embed = self.xy_embed_fn(points[:, :2])
            rendering_input = torch.cat([points, normals, view_dirs, feature_vectors, xy_embed], dim=-1)
        else:
            raise Exception("Not implemented")
        x = self.mlp1(rendering_input) # penalize delta x here
        x = self.mlp2(torch.cat([x, encod], dim=-1))
        x, _ = x[:, :3], x[:, 3]
        output = self.orig_rgb(feature_vectors) # orig color depends on feature vectors only
        x_orig, beta = output[:, :3], output[:, 3]
        return {"rgb": torch.sigmoid(x + x_orig), "delta": x, "rgb_orig": torch.sigmoid(x_orig), "beta": beta}


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * torch.exp(self.variance * 10.0)

def nn_2d(A, B):
    """
    A: (N, 2)
    B: (M, 2) query points
    return: (M) indices of the nearest neighbors in A
    """
    A = A.unsqueeze(0)
    B = B.unsqueeze(1)
    distances = torch.norm(A - B, dim=-1)
    idx = torch.argmin(distances, dim=-1)
    return idx

class SDFNetwork_2d_hash(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 base_resolution,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False, 
                 num_clusters=1,
                 ):
        super(SDFNetwork_2d_hash, self).__init__()
        
        self.num_clusters = num_clusters

        self.hash_encoding = nn.ModuleList([tcnn.Encoding(n_input_dims=2, encoding_config={
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": base_resolution,  # 最低1000 / (46 * 1.5 ** 15) = 0.05的分辨率
            "per_level_scale": 1.5
        }) for _ in range(self.num_clusters)])

        self.tiny_mlp = nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(self.hash_encoding[i].n_output_dims, 65),
            # torch.nn.Identity()
            # torch.nn.Linear(self.hash_encoding.n_output_dims, 65)
            # torch.nn.Linear(self.hash_encoding.n_output_dims, 128), 
            # torch.nn.ReLU(),
            # torch.nn.Linear(128, 128),
            # torch.nn.ReLU(),
            # torch.nn.Linear(128, 128),
            # torch.nn.ReLU(),
            # torch.nn.Linear(128, 65)
        ) for i in range(self.num_clusters)])
    #     self.mlp = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=257, encoding_config={
	# 	"otype": "HashGrid",
	# 	"n_levels": 16,
	# 	"n_features_per_level": 2,
	# 	"log2_hashmap_size": 1,
	# 	"base_resolution": 16,
	# 	"per_level_scale": 1.5
	# }, network_config={
	# 	"otype": "FullyFusedMLP",
	# 	"activation": "ReLU",
	# 	"output_activation": "None",
	# 	"n_neurons": 64,
	# 	"n_hidden_layers": 2
	# })
 
        self.iter_step = 0

        assert d_in == 2
        inside_outside = True
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        # if multires > 0:
        #     embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
        #     self.embed_fn_fine = embed_fn
        #     dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        
        
        # filled by exp_runner
        self.dataset = None 
        self.exp_runner = None
        

        self.prior_network = nn.ModuleList([tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1, network_config={
            "otype": "FullyFusedMLP",
            "activation": "Sigmoid",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2
        }, encoding_config={
            "otype": "Frequency", 
            "n_frequencies": 3
        }) for _ in range(self.num_clusters)])
        # print(self.mlp)
        # raise Exception
    
    
    def forward(self, inputs, output_height=False, delta=True, gradient=False, force_cluster=None):

        if force_cluster is not None:
            return self.forward_core(inputs, force_cluster, output_height, delta, gradient)
        
        if self.num_clusters == 1:
            return self.forward_core(inputs, 0, output_height, delta, gradient)
        # get cluster indices for inputs
        raise Exception("Not implemented")
        world_input_xyz = inputs @ torch.from_numpy(self.dataset.scale_mat[:3, :3]).cuda().float()
        
        # # find closest ego point for each random xyz
        # nn_dist, nn_idx_more = nn_2d_k(torch.from_numpy(self.dataset.ego_points).cuda().float(), world_input_xyz, k=1)

        # choice = torch.randint(0, 1, [inputs.shape[0]])
        # nn_idx = nn_idx_more[torch.arange(len(inputs)), choice]

        # # get cluster for each random xyz
        # cluster_idx = torch.from_numpy(self.dataset.ego2cluster).cuda().float()[nn_idx]
        
        cluster_idx = self.exp_runner.points2cluster(world_input_xyz, k=1, is_world=True, kd_tree=False).squeeze()
        all_outputs = None
        for i in range(self.num_clusters):
            mask = cluster_idx == i
            if mask.sum() > 0:
                
                output = self.forward_core(inputs[mask], i, output_height, delta, gradient)
                if all_outputs is None:
                    all_outputs = torch.zeros([len(inputs), output.shape[-1]], device=output.device)
                all_outputs[mask] = output
        return all_outputs
                
                
    def forward_core(self, inputs, cluster_idx, output_height=False, delta=True, gradient=False):

        if delta:
            with torch.no_grad():
                                
                
                ### use ground truth prior
                # world_xy = inputs[:, :2] * 500
                # nn_idx = nn_2d(self.prior_network.world_ego_points[:, :2], world_xy)
                # prior_z = -self.prior_network.world_coff_A[nn_idx] * (world_xy[:, 0] - self.prior_network.world_ego_points[nn_idx, 0]) - self.prior_network.world_coff_B[nn_idx] * (world_xy[:, 1] - self.prior_network.world_ego_points[nn_idx, 1]) + self.prior_network.world_ego_points[nn_idx, 2]
                # prior_z = prior_z / 9
                # prior_z = prior_z.unsqueeze(-1).detach()
                
                prior_z = self.prior_network[cluster_idx](inputs[:, :2]).detach()
        z_vals = inputs[:, 2:]
        inputs = inputs * self.scale
        # if self.embed_fn_fine is not None:
        #     inputs = self.embed_fn_fine(inputs[:, :2])
        
        x = inputs[:, :2]
        # print(x.shape)
        # temp = (x / 30 + 0.5).flatten()
        # # temp = x.flatten()
        # print(temp.min(), temp.max())
        # raise Exception
        x = self.hash_encoding[cluster_idx](x / 30 + 0.5)   # input range should be in [0, 1] (original was [-1, 1])
        x = self.tiny_mlp[cluster_idx](x.float())
        if self.iter_step > -1 or self.iter_step == 0:
            if not delta:
                return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)
            else:
                if gradient:
                    return torch.cat([z_vals - x[:, :1] / self.scale, x[:, 1:]], dim=-1)
                if output_height:
                    return torch.cat([x[:, :1] / self.scale + prior_z, x[:, 1:], z_vals], dim=-1)
                # elif output_delta:
                #     return torch.cat([x[:, :1] / self.scale, x[:, 1:], z_vals - x[:, :1] / self.scale - prior_z], dim=-1)
                else:
                    # print(z_vals.shape, x[:, :1].shape, prior_z.shape)
                    return torch.cat([z_vals - x[:, :1] / self.scale - prior_z, x[:, 1:]], dim=-1)
        else:
            if not delta:
                return torch.cat([torch.zeros_like(x[:, :1], device=x.device) / self.scale, x[:, 1:]], dim=-1)
            else:
                if gradient:
                    return torch.cat([z_vals - torch.zeros_like(x[:, :1], device=x.device) / self.scale, x[:, 1:]], dim=-1)
                if output_height:
                    return torch.cat([torch.zeros_like(x[:, :1], device=x.device) / self.scale + prior_z, x[:, 1:], z_vals], dim=-1)
                # elif output_delta:
                #     return torch.cat([x[:, :1] / self.scale, x[:, 1:], z_vals - x[:, :1] / self.scale - prior_z], dim=-1)
                else:
                    # print(z_vals.shape, x[:, :1].shape, prior_z.shape)
                    return torch.cat([z_vals - torch.zeros_like(x[:, :1], device=x.device) / self.scale - prior_z, x[:, 1:]], dim=-1)
            
    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):

        x.requires_grad_(True)
        y = self.forward(x, gradient=True)[:, :1]
        # y = self.forward(x, gradient=False)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)
    
    
    
    
    
class SDFNetwork_2d_hash_fix(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 base_resolution,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False, 
                 num_clusters=1,
                 ):
        super(SDFNetwork_2d_hash_fix, self).__init__()
        
        self.num_clusters = num_clusters

        self.hash_encoding = nn.ModuleList([tcnn.Encoding(n_input_dims=2, encoding_config={
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": base_resolution,  # 最低1000 / (46 * 1.5 ** 15) = 0.05的分辨率
            "per_level_scale": 1.5
        }) for _ in range(self.num_clusters)])

        self.tiny_mlp = nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(self.hash_encoding[i].n_output_dims, 65),
            # torch.nn.Identity()
            # torch.nn.Linear(self.hash_encoding.n_output_dims, 65)
            # torch.nn.Linear(self.hash_encoding.n_output_dims, 128), 
            # torch.nn.ReLU(),
            # torch.nn.Linear(128, 128),
            # torch.nn.ReLU(),
            # torch.nn.Linear(128, 128),
            # torch.nn.ReLU(),
            # torch.nn.Linear(128, 65)
        ) for i in range(self.num_clusters)])
    #     self.mlp = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=257, encoding_config={
	# 	"otype": "HashGrid",
	# 	"n_levels": 16,
	# 	"n_features_per_level": 2,
	# 	"log2_hashmap_size": 1,
	# 	"base_resolution": 16,
	# 	"per_level_scale": 1.5
	# }, network_config={
	# 	"otype": "FullyFusedMLP",
	# 	"activation": "ReLU",
	# 	"output_activation": "None",
	# 	"n_neurons": 64,
	# 	"n_hidden_layers": 2
	# })
 
        self.iter_step = 0

        assert d_in == 2
        inside_outside = True
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        # if multires > 0:
        #     embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
        #     self.embed_fn_fine = embed_fn
        #     dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        
        
        # filled by exp_runner
        self.dataset = None 
        

        self.prior_network = nn.ModuleList([tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1, network_config={
            "otype": "FullyFusedMLP",
            "activation": "Sigmoid",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2
        }, encoding_config={
            "otype": "Frequency", 
            "n_frequencies": 3
        }) for _ in range(self.num_clusters)])
        # print(self.mlp)
        # raise Exception
    
    
    def forward(self, inputs, output_height=False, delta=True, gradient=False, force_cluster=None):

        if force_cluster is not None:
            return self.forward_core(inputs, force_cluster, output_height, delta, gradient)
        # get cluster indices for inputs
        world_input_xyz = inputs @ torch.from_numpy(self.dataset.scale_mat[:3, :3]).cuda().float()
        
        # find closest ego point for each random xyz
        nn_dist, nn_idx_more = nn_2d_k(torch.from_numpy(self.dataset.ego_points).cuda().float(), world_input_xyz, k=1)

        choice = torch.randint(0, 1, [inputs.shape[0]])
        nn_idx = nn_idx_more[torch.arange(len(inputs)), choice]

        # get cluster for each random xyz
        cluster_idx = torch.from_numpy(self.dataset.ego2cluster).cuda().float()[nn_idx]
        
        all_outputs = None
        for i in range(self.num_clusters):
            mask = cluster_idx == i
            if mask.sum() > 0:
                
                output = self.forward_core(inputs[mask], i, output_height, delta, gradient)
                if all_outputs is None:
                    all_outputs = torch.zeros([len(inputs), output.shape[-1]], device=output.device)
                all_outputs[mask] = output
        return all_outputs
                
                
    def forward_core(self, inputs, cluster_idx, output_height=False, delta=True, gradient=False):

        if delta:
            with torch.no_grad():
                                
                
                ### use ground truth prior
                # world_xy = inputs[:, :2] * 500
                # nn_idx = nn_2d(self.prior_network.world_ego_points[:, :2], world_xy)
                # prior_z = -self.prior_network.world_coff_A[nn_idx] * (world_xy[:, 0] - self.prior_network.world_ego_points[nn_idx, 0]) - self.prior_network.world_coff_B[nn_idx] * (world_xy[:, 1] - self.prior_network.world_ego_points[nn_idx, 1]) + self.prior_network.world_ego_points[nn_idx, 2]
                # prior_z = prior_z / 9
                # prior_z = prior_z.unsqueeze(-1).detach()
                
                prior_z = self.prior_network[cluster_idx](inputs[:, :2]).detach()
        z_vals = inputs[:, 2:]
        inputs = inputs * self.scale
        # if self.embed_fn_fine is not None:
        #     inputs = self.embed_fn_fine(inputs[:, :2])
        
        x = inputs[:, :2]
        # print(x.shape)
        # temp = (x / 30 + 0.5).flatten()
        # # temp = x.flatten()
        # print(temp.min(), temp.max())
        # raise Exception
        x = self.hash_encoding[cluster_idx](x / 30 + 0.5)   # input range should be in [0, 1] (original was [-1, 1])
        x = self.tiny_mlp[cluster_idx](x.float())
        if self.iter_step > -1 or self.iter_step == 0:
            if not delta:
                return torch.cat([torch.zeros_like(x[:, :1], device=x.device) / self.scale, x[:, 1:]], dim=-1)
            else:
                if gradient:
                    return torch.cat([z_vals - torch.zeros_like(x[:, :1], device=x.device) / self.scale, x[:, 1:]], dim=-1)
                if output_height:
                    return torch.cat([torch.zeros_like(x[:, :1], device=x.device) / self.scale + prior_z, x[:, 1:], z_vals], dim=-1)
                # elif output_delta:
                #     return torch.cat([torch.zeros_like(x[:, :1], device=x.device) / self.scale, x[:, 1:], z_vals - torch.zeros_like(x[:, :1], device=x.device) / self.scale - prior_z], dim=-1)
                else:
                    # print(z_vals.shape, torch.zeros_like(x[:, :1], device=x.device).shape, prior_z.shape)
                    return torch.cat([z_vals - torch.zeros_like(x[:, :1], device=x.device) / self.scale - prior_z, x[:, 1:]], dim=-1)
        else:
            if not delta:
                return torch.cat([torch.zeros_like(torch.zeros_like(x[:, :1], device=x.device), device=x.device) / self.scale, x[:, 1:]], dim=-1)
            else:
                if gradient:
                    return torch.cat([z_vals - torch.zeros_like(torch.zeros_like(x[:, :1], device=x.device), device=x.device) / self.scale, x[:, 1:]], dim=-1)
                if output_height:
                    return torch.cat([torch.zeros_like(torch.zeros_like(x[:, :1], device=x.device), device=x.device) / self.scale + prior_z, x[:, 1:], z_vals], dim=-1)
                # elif output_delta:
                #     return torch.cat([torch.zeros_like(x[:, :1], device=x.device) / self.scale, x[:, 1:], z_vals - torch.zeros_like(x[:, :1], device=x.device) / self.scale - prior_z], dim=-1)
                else:
                    # print(z_vals.shape, torch.zeros_like(x[:, :1], device=x.device).shape, prior_z.shape)
                    return torch.cat([z_vals - torch.zeros_like(torch.zeros_like(x[:, :1], device=x.device), device=x.device) / self.scale - prior_z, x[:, 1:]], dim=-1)
            
    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        
        
        
        x.requires_grad_(True)
        y = self.forward(x, gradient=True)[:, :1]
        # y = self.forward(x, gradient=False)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)