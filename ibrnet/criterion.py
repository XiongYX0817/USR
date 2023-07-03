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
import torch.nn as nn
from utils import img2mse


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, scalars_to_log):
        '''
        training criterion
        '''
        pred_rgb = outputs['rgb']
        pred_mask = outputs['mask'].float()
        gt_rgb = ray_batch['rgb']

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        return loss, scalars_to_log

def segmentation_loss(T, pred, gt):
    cross_entropy = nn.CrossEntropyLoss(reduce=False)
    gt = gt.unsqueeze(1).repeat((1, pred.shape[1])).reshape(-1)
    pred = pred.reshape(-1, pred.shape[-1])
    tmp = cross_entropy(pred, gt)
    return (T * tmp).sum()

def normal_loss(model, projector, ray_batch, surface_pts, featmaps, batch_size=128):
    normal_ref = []
    normal_neighbor = []
    for points in torch.split(surface_pts, batch_size, dim=0):
        normal_ref.append(torch.nn.functional.normalize(gradient(model, projector, ray_batch, points, featmaps), dim=1))
        neighbors = points + (torch.rand_like(points) - 0.5) * 0.005
        normal_neighbor.append(torch.nn.functional.normalize(gradient(model, projector, ray_batch, neighbors, featmaps), dim=1))
    normal_ref = torch.cat(normal_ref, dim=0)
    normal_neighbor = torch.cat(normal_neighbor, dim=0)
    diff_norm = torch.norm(normal_ref - normal_neighbor, dim=-1)
    return diff_norm.mean()


def gradient(model, projector, ray_batch, points, featmaps):
    points = points.unsqueeze(0)
    with torch.enable_grad():
        points.requires_grad_(True)
        rgb_feat, ray_diff, mask = projector.compute_grad(points, ray_batch['camera'],
                                                          ray_batch['src_rgbs'],
                                                          ray_batch['src_cameras'],
                                                          featmaps=featmaps[0])
        sigma = model.net_coarse(rgb_feat, ray_diff, mask)[..., 3]
        d_output = torch.ones_like(sigma, requires_grad=False, device=sigma.device)
        gradients = torch.autograd.grad(
            outputs=sigma,
            inputs=points,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True, allow_unused=True)[0]
        return gradients[0] + 1e-8
