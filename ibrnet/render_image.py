# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from collections import OrderedDict
from ibrnet.render_ray import render_rays


def render_single_image(ray_sampler,
                        ray_batch,
                        model,
                        projector,
                        chunk_size,
                        N_samples,
                        inv_uniform=False,
                        N_importance=0,
                        det=False,
                        white_bkgd=False,
                        render_stride=1,
                        featmaps=None,
                        mask_img=None):
    '''
    :param ray_sampler: RaySamplingSingleImage for this view
    :param model:  {'net_coarse': , 'net_fine': , ...}
    :param chunk_size: number of rays in a chunk
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :return: {'outputs_coarse': {'rgb': numpy, 'depth': numpy, ...}, 'outputs_fine': {}}
    '''

    all_ret = OrderedDict([('outputs_coarse', OrderedDict()),
                           ('outputs_fine', OrderedDict())])

    pts_coarse = []
    rgb_coarse = []
    sigma_coarse = []
    z_vals_coarse = []
    pts_fine = []
    rgb_fine = []
    sigma_fine = []
    z_vals_fine = []

    visualize = False
    method = 1

    N_rays = ray_batch['ray_o'].shape[0]

    for i in range(0, N_rays, chunk_size):
        chunk = OrderedDict()
        for k in ray_batch:
            if k in ['camera', 'depth_range', 'src_rgbs', 'src_cameras']:
                chunk[k] = ray_batch[k]
            elif ray_batch[k] is not None:
                chunk[k] = ray_batch[k][i:i+chunk_size]
            else:
                chunk[k] = None

        ret = render_rays(chunk, model, featmaps,
                          projector=projector,
                          N_samples=N_samples,
                          inv_uniform=inv_uniform,
                          N_importance=N_importance,
                          det=det,
                          white_bkgd=white_bkgd)

        if visualize:

            pts_coarse.append(ret['pts_coarse'].cpu())
            rgb_coarse.append(ret['raw_coarse'][:, :, :3].reshape(-1, 3).cpu())
            sigma_coarse.append(ret['raw_coarse'][:, :, 3].view(-1).cpu())
            z_vals_coarse.append(ret['z_vals_coarse'].view(-1).cpu())

            pts_fine.append(ret['pts_fine'].cpu())
            rgb_fine.append(ret['raw_fine'][:, :, :3].reshape(-1, 3).cpu())
            sigma_fine.append(ret['raw_fine'][:, :, 3].view(-1).cpu())
            z_vals_fine.append(ret['z_vals_fine'].view(-1).cpu())

        # handle both coarse and fine outputs
        # cache chunk results on cpu
        if i == 0:
            for k in ret['outputs_coarse']:
                all_ret['outputs_coarse'][k] = []

            if ret['outputs_fine'] is None:
                all_ret['outputs_fine'] = None
            else:
                for k in ret['outputs_fine']:
                    all_ret['outputs_fine'][k] = []

        for k in ret['outputs_coarse']:
            all_ret['outputs_coarse'][k].append(ret['outputs_coarse'][k].cpu())

        if ret['outputs_fine'] is not None:
            for k in ret['outputs_fine']:
                all_ret['outputs_fine'][k].append(ret['outputs_fine'][k].cpu())

    if visualize:

        sigma2alpha = lambda sigma, dists: 1. - torch.exp(-sigma)

        pts_coarse = torch.cat(pts_coarse, dim=0).reshape(512, 512, -1, 3)
        rgb_coarse = torch.cat(rgb_coarse, dim=0).reshape(512, 512, -1, 3)
        sigma_coarse = torch.cat(sigma_coarse, dim=0).reshape(512, 512, -1, 1)
        z_vals_coarse = torch.cat(z_vals_coarse, dim=0).reshape(512, 512, -1, 1)

        pts_fine = torch.cat(pts_fine, dim=0).reshape(512, 512, -1, 3)
        rgb_fine = torch.cat(rgb_fine, dim=0).reshape(512, 512, -1, 3)
        sigma_fine = torch.cat(sigma_fine, dim=0).reshape(512, 512, -1, 1)
        z_vals_fine = torch.cat(z_vals_fine, dim=0).reshape(512, 512, -1, 1)

        if method == 1:  ## 方案一：根据 sigma > threshold 来筛选空间点
            threshold = 0.5
            end_idx_coarse = 0

            pts_coarse = pts_coarse[mask_img][:, :64-end_idx_coarse].reshape(-1, 3)
            rgb_coarse = rgb_coarse[mask_img][:, :64-end_idx_coarse].reshape(-1, 3)
            sigma_coarse = sigma_coarse[mask_img][:, :64-end_idx_coarse].contiguous().view(-1)

            mask_coarse = (sigma_coarse > threshold)
            pts_coarse = pts_coarse[mask_coarse].permute(1, 0)
            rgb_coarse = rgb_coarse[mask_coarse]
            sigma_coarse = sigma_coarse[mask_coarse]

            pts_fine = pts_fine[mask_img].reshape(-1, 3)
            rgb_fine = rgb_fine[mask_img].reshape(-1, 3)
            sigma_fine = sigma_fine[mask_img].view(-1)

            mask_fine = (sigma_fine > threshold)
            pts_fine = pts_fine[mask_fine].permute(1, 0)
            rgb_fine = rgb_fine[mask_fine]
            sigma_fine = sigma_fine[mask_fine]

        elif method == 2: ## 方案二：保留每条光线中 alpha 最大的那个点

            sigma_coarse = sigma_coarse[mask_img].reshape(-1, 64)
            z_vals_coarse = z_vals_coarse[mask_img].reshape(-1, 64)
            dists_coarse = z_vals_coarse[:, 1:] - z_vals_coarse[:, :-1]
            dists_coarse = torch.cat((dists_coarse, dists_coarse[:, -1:]), dim=-1)  # [N_rays, N_samples]
            alpha_coarse = sigma2alpha(sigma_coarse, dists_coarse)  # [N_rays, N_samples]
            T_coarse = torch.cumprod(1. - alpha_coarse + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
            T_coarse = torch.cat((torch.ones_like(T_coarse[:, 0:1]), T_coarse), dim=-1)  # [N_rays, N_samples]
            weights_coarse = alpha_coarse * T_coarse
            weights_coarse = weights_coarse.argmax(dim=1)

            sigma_fine = sigma_fine[mask_img].reshape(-1, 128)
            z_vals_fine = z_vals_fine[mask_img].reshape(-1, 128)
            dists_fine = z_vals_fine[:, 1:] - z_vals_fine[:, :-1]
            dists_fine = torch.cat((dists_fine, dists_fine[:, -1:]), dim=-1)  # [N_rays, N_samples]
            alpha_fine = sigma2alpha(sigma_fine, dists_fine)  # [N_rays, N_samples]
            T_fine = torch.cumprod(1. - alpha_fine + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
            T_fine = torch.cat((torch.ones_like(T_fine[:, 0:1]), T_fine), dim=-1)  # [N_rays, N_samples]
            weights_fine = alpha_fine * T_fine
            weights_fine = weights_fine.argmax(dim=1)

            tmp_idxes = range(mask_img.sum())
            pts_coarse = pts_coarse[mask_img][tmp_idxes, weights_coarse].reshape(-1, 3).permute(1, 0)
            rgb_coarse = rgb_coarse[mask_img][tmp_idxes, weights_coarse].reshape(-1, 3)
            sigma_coarse = sigma_coarse[tmp_idxes, weights_coarse].reshape(-1)

            pts_fine = pts_fine[mask_img][tmp_idxes, weights_fine].reshape(-1, 3).permute(1, 0)
            rgb_fine = rgb_fine[mask_img][tmp_idxes, weights_fine].reshape(-1, 3)
            sigma_fine = sigma_fine[tmp_idxes, weights_fine].reshape(-1)

        else: ## 方案三：alpha + sigma

            threshold_sigma = 1
            threshold_alpha = 0.8

            sigma_coarse = sigma_coarse[mask_img].reshape(-1, 64)
            z_vals_coarse = z_vals_coarse[mask_img].reshape(-1, 64)
            dists_coarse = z_vals_coarse[:, 1:] - z_vals_coarse[:, :-1]
            dists_coarse = torch.cat((dists_coarse, dists_coarse[:, -1:]), dim=-1)  # [N_rays, N_samples]
            alpha_coarse = sigma2alpha(sigma_coarse, dists_coarse)  # [N_rays, N_samples]
            T_coarse = torch.cumprod(1. - alpha_coarse + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
            T_coarse = torch.cat((torch.ones_like(T_coarse[:, 0:1]), T_coarse), dim=-1)  # [N_rays, N_samples]
            weights_coarse = alpha_coarse * T_coarse
            weights_max_coarse, _ = weights_coarse.max(dim=1)
            mask_coarse = torch.logical_and(sigma_coarse > threshold_sigma, weights_coarse > weights_max_coarse.unsqueeze(1).repeat(1, 64) * threshold_alpha)

            sigma_fine = sigma_fine[mask_img].reshape(-1, 128)
            z_vals_fine = z_vals_fine[mask_img].reshape(-1, 128)
            dists_fine = z_vals_fine[:, 1:] - z_vals_fine[:, :-1]
            dists_fine = torch.cat((dists_fine, dists_fine[:, -1:]), dim=-1)  # [N_rays, N_samples]
            alpha_fine = sigma2alpha(sigma_fine, dists_fine)  # [N_rays, N_samples]
            T_fine = torch.cumprod(1. - alpha_fine + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
            T_fine = torch.cat((torch.ones_like(T_fine[:, 0:1]), T_fine), dim=-1)  # [N_rays, N_samples]
            weights_fine = alpha_fine * T_fine
            weights_max_fine, _ = weights_fine.max(dim=1)
            mask_fine = torch.logical_and(sigma_fine > threshold_sigma, weights_fine > weights_max_fine.unsqueeze(1).repeat(1, 128) * threshold_alpha)

            pts_coarse = pts_coarse[mask_img][mask_coarse].reshape(-1, 3).permute(1, 0)
            rgb_coarse = rgb_coarse[mask_img][mask_coarse].reshape(-1, 3)
            sigma_coarse = sigma_coarse[mask_coarse].reshape(-1)

            pts_fine = pts_fine[mask_img][mask_fine].reshape(-1, 3).permute(1, 0)
            rgb_fine = rgb_fine[mask_img][mask_fine].reshape(-1, 3)
            sigma_fine = sigma_fine[mask_fine].reshape(-1)

        def get_color(sigma):
            sigma_max = sigma.max()
            alpha = 1.0 - (sigma / sigma_max).reshape(-1, 1)
            ones = torch.ones_like(alpha)
            colors = torch.cat([ones, alpha, alpha, alpha], 1)
            return colors

        color_coarse = get_color(sigma_coarse)
        color_fine = get_color(sigma_fine)

        import pickle
        res = {
            "pts_coarse": pts_coarse,
            "color_coarse": color_coarse,
            "rgb_coarse": rgb_coarse,
            "pts_fine": pts_fine,
            "color_fine": color_fine,
            "rgb_fine": rgb_fine
        }
        with open("results.pickle", "wb") as f:
            pickle.dump(res, f)

        print("[INFO] Rendering...")

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax.scatter(pts_coarse[0], pts_coarse[1], pts_coarse[2], c=color_coarse, alpha=alpha_coarse, s=0.1)
        ax.scatter(pts_fine[0], pts_fine[1], pts_fine[2], c=rgb_fine, s=0.1)
        plt.savefig("display_fine.png")

        exit()

    rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride, :]
    # merge chunk results and reshape
    for k in all_ret['outputs_coarse']:
        if k == 'random_sigma':
            continue
        tmp = torch.cat(all_ret['outputs_coarse'][k], dim=0).reshape((rgb_strided.shape[0],
                                                                      rgb_strided.shape[1], -1))
        all_ret['outputs_coarse'][k] = tmp.squeeze()

    all_ret['outputs_coarse']['rgb'][all_ret['outputs_coarse']['mask'] == 0] = 1.
    if all_ret['outputs_fine'] is not None:
        for k in all_ret['outputs_fine']:
            if k == 'random_sigma':
                continue
            tmp = torch.cat(all_ret['outputs_fine'][k], dim=0).reshape((rgb_strided.shape[0],
                                                                        rgb_strided.shape[1], -1))

            all_ret['outputs_fine'][k] = tmp.squeeze()

    return all_ret



