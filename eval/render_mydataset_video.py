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


from torch.utils.data import Dataset
import sys
sys.path.append('../')
from torch.utils.data import DataLoader
import imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from utils import *
from ibrnet.projection import Projector
from ibrnet.data_loaders import get_nearest_pose_ids
from ibrnet.data_loaders.mydataset import MyRenderDataset
import time


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

    # Create ibrnet model
    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
    print('saving results to {}...'.format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    projector = Projector(device='cuda:0')
    # projector = Projector(device='cpu')

    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, model.start_step), 'videos')
    print('saving results to {}'.format(out_scene_dir))

    os.makedirs(out_scene_dir, exist_ok=True)

    test_dataset = MyRenderDataset(args, scenes=args.eval_scenes)
    save_prefix = scene_name
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    out_frames = []
    crop_ratio = 0.075

    for i, data in enumerate(test_loader):

        if i < 4:
            continue

        start = time.time()
        src_rgbs = data['src_rgbs'][0].cpu().numpy()
        averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_average.png'.format(i)), averaged_img)

        ### Mine ###
        import numpy as np
        from PIL import Image

        def int2str_mask(a):
            if a < 10:
                return "00" + str(a)
            if a < 100:
                return "0" + str(a)
            return str(a)

        mask_img = torch.tensor(np.array(Image.open(f"../data/mydataset/test/cg_25/mask/{int2str_mask(i * 8)}.png", "r")))
        mask_img = mask_img[:, :, 0] > 0.5

        ### The End ###

        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
            # ray_sampler = RaySamplerSingleImage(data, device='cpu')
            ray_batch = ray_sampler.get_all()
            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))

            ret = render_single_image(ray_sampler=ray_sampler,
                                      ray_batch=ray_batch,
                                      model=model,
                                      projector=projector,
                                      chunk_size=args.chunk_size,
                                      det=True,
                                      N_samples=args.N_samples,
                                      inv_uniform=args.inv_uniform,
                                      N_importance=args.N_importance,
                                      white_bkgd=args.white_bkgd,
                                      featmaps=featmaps,
                                      mask_img=mask_img)
            torch.cuda.empty_cache()

        coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
        coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_coarse.png'.format(i)), coarse_pred_rgb)

        coarse_pred_depth = ret['outputs_coarse']['depth'].detach().cpu()
        imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_coarse.png'.format(i)),
                        (coarse_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
        coarse_pred_depth_colored = colorize_np(coarse_pred_depth,
                                                range=tuple(data['depth_range'].squeeze().numpy()))
        imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_coarse.png'.format(i)),
                        (255 * coarse_pred_depth_colored).astype(np.uint8))

        coarse_acc_map = torch.sum(ret['outputs_coarse']['weights'].detach().cpu(), dim=-1)
        coarse_acc_map_colored = (colorize_np(coarse_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_coarse.png'.format(i)),
                        coarse_acc_map_colored)

        if ret['outputs_fine'] is not None:
            fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
            fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_fine.png'.format(i)), fine_pred_rgb)
            fine_pred_depth = ret['outputs_fine']['depth'].detach().cpu()
            imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_fine.png'.format(i)),
                            (fine_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
            fine_pred_depth_colored = colorize_np(fine_pred_depth,
                                                  range=tuple(data['depth_range'].squeeze().cpu().numpy()))
            imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_fine.png'.format(i)),
                            (255 * fine_pred_depth_colored).astype(np.uint8))
            fine_acc_map = torch.sum(ret['outputs_fine']['weights'].detach().cpu(), dim=-1)
            fine_acc_map_colored = (colorize_np(fine_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_fine.png'.format(i)),
                            fine_acc_map_colored)
        else:
            fine_pred_rgb = None

        out_frame = fine_pred_rgb if fine_pred_rgb is not None else coarse_pred_rgb
        h, w = averaged_img.shape[:2]
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        # crop out image boundaries
        out_frame = out_frame[crop_h:h - crop_h, crop_w:w - crop_w, :]
        out_frames.append(out_frame)

        print('frame {} completed, {}'.format(i, time.time() - start))

    imageio.mimwrite(os.path.join(extra_out_dir, '{}.mp4'.format(scene_name)), out_frames, fps=30, quality=8)
