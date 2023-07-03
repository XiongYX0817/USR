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


import os
import time
import numpy as np
from tqdm import tqdm
import shutil
import imageio
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader

from ibrnet.data_loaders import dataset_dict
from ibrnet.render_ray import render_rays
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.criterion import Criterion, normal_loss
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, cycle, img2psnr, save_current_code
import config
import torch.distributed as dist
from ibrnet.projection import Projector
from ibrnet.data_loaders.create_training_dataset import create_training_dataset


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def infer(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, 'out', args.expname)
    print('outputs will be saved to {}'.format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, 'config.txt')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    # create validation dataset
    args.testskip = 1
    test_dataset = dataset_dict[args.eval_dataset](args, 'test', scenes=args.eval_scenes)

    test_loader = DataLoader(test_dataset, batch_size=1)
    # test_loader_iterator = iter(cycle(test_loader))

    # Create IBRNet model
    model = IBRNetModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)
    model.switch_to_eval()

    # create projector
    projector = Projector(device=device)

    for idx, test_data in enumerate(tqdm(test_loader)):
        if idx % 140 >= 20:
            continue
        tmp_ray_test_sampler = RaySamplerSingleImage(test_data, device, render_stride=1)
        H, W = tmp_ray_test_sampler.H, tmp_ray_test_sampler.W
        gt_img = tmp_ray_test_sampler.rgb.reshape(H, W, 3)
        inference(args, model, idx, tmp_ray_test_sampler, projector, gt_img, out_folder, render_stride=1)


def inference(args, model, idx, ray_sampler, projector, gt_img, out_folder, render_stride=1):
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net is not None:
            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps = [None, None]
        ret = render_single_image(ray_sampler=ray_sampler,
                                  ray_batch=ray_batch,
                                  model=model,
                                  projector=projector,
                                  chunk_size=args.chunk_size,
                                  N_samples=args.N_samples,
                                  inv_uniform=args.inv_uniform,
                                  det=True,
                                  N_importance=args.N_importance,
                                  white_bkgd=args.white_bkgd,
                                  render_stride=render_stride,
                                  featmaps=featmaps)

    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    if not os.path.exists(out_folder + "/rendering/"):
        os.mkdir(out_folder + "/rendering/")

    rgb_coarse = ret['outputs_coarse']['rgb'].detach().cpu() * 255
    rgb_fine = ret['outputs_fine']['rgb'].detach().cpu() * 255

    conf_coarse = ret['outputs_coarse']['confidence'].detach().cpu()
    conf_fine = ret['outputs_fine']['confidence'].detach().cpu()

    conf_coarse = conf_coarse.reshape(conf_coarse.shape[0], conf_coarse.shape[1], -1, 6)
    conf_fine = conf_fine.reshape(conf_fine.shape[0], conf_fine.shape[1], -1, 6)

    T_coarse = ret['outputs_coarse']['T'].detach().cpu().argmax(dim=-1)
    T_fine = ret['outputs_fine']['T'].detach().cpu().argmax(dim=-1)

    # print(T_coarse.shape)
    # print(conf_coarse.shape)

    tmp_idx1 = torch.tensor(range(T_coarse.shape[0])).unsqueeze(1).repeat(1, T_coarse.shape[1])
    tmp_idx2 = torch.tensor(range(T_coarse.shape[1])).unsqueeze(0).repeat(T_coarse.shape[0], 1)

    seg_coarse = conf_coarse[tmp_idx1, tmp_idx2, T_coarse].argmax(dim=-1)
    seg_fine = conf_fine[tmp_idx1, tmp_idx2, T_fine].argmax(dim=-1)

    canvas_coarse = torch.zeros(seg_coarse.shape[0], seg_coarse.shape[1], 3)
    canvas_coarse[seg_coarse == 1] = torch.tensor([255.0, 0, 0])
    canvas_coarse[seg_coarse == 2] = torch.tensor([0, 255.0, 0])
    canvas_coarse[seg_coarse == 3] = torch.tensor([0, 0, 255.0])
    canvas_coarse[seg_coarse == 4] = torch.tensor([255.0, 255.0, 0])
    canvas_coarse[seg_coarse == 5] = torch.tensor([250.0, 235.0, 215.0])

    canvas_fine = torch.zeros(seg_fine.shape[0], seg_fine.shape[1], 3)
    canvas_fine[seg_fine == 1] = torch.tensor([255.0, 0, 0])
    canvas_fine[seg_fine == 2] = torch.tensor([0, 255.0, 0])
    canvas_fine[seg_fine == 3] = torch.tensor([0, 0, 255.0])
    canvas_fine[seg_fine == 4] = torch.tensor([255.0, 255.0, 0])
    canvas_fine[seg_fine == 5] = torch.tensor([250.0, 235.0, 215.0])

    imageio.imwrite(out_folder + f'/rendering/coarse_new_{idx}.png', rgb_coarse.numpy().astype(np.uint8))
    imageio.imwrite(out_folder + f'/rendering/fine_new_{idx}.png', rgb_fine.numpy().astype(np.uint8))
    imageio.imwrite(out_folder + f'/rendering/coarse_seg_{idx}.png', canvas_coarse.numpy().astype(np.uint8))
    imageio.imwrite(out_folder + f'/rendering/fine_seg_{idx}.png', canvas_fine.numpy().astype(np.uint8))


if __name__ == '__main__':
    parser = config.config_parser()
    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    infer(args)
