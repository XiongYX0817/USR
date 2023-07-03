import os
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

import config
from ibrnet.data_loaders import dataset_dict
from ibrnet.render_ray import render_rays_batch, sample_along_camera_ray
from ibrnet.model import IBRNetModel
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.projection import Projector
from utils import cycle

from skimage import measure
from tqdm import tqdm

logger_py = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")


def export_obj(vertices, triangles, diffuse, normals, filename):
    """
    Exports a mesh in the (.obj) format.
    """
    print('Writing to obj...')

    with open(filename, "w") as fh:

        for index, v in enumerate(vertices):
            fh.write("v {} {} {}".format(*v))
            if len(diffuse) > index:
                fh.write(" {} {} {}".format(*diffuse[index]))

            fh.write("\n")

        for n in normals:
            fh.write("vn {} {} {}\n".format(*n))

        for f in triangles:
            fh.write("f")
            for index in f:
                fh.write(" {}//{}".format(index + 1, index + 1))

            fh.write("\n")

    print(f"Finished writing to {filename} with {len(vertices)} vertices")

def extract_geometry(model, projector, data, res, limit, threshold):

    # Query points
    res = res + 1
    t = torch.linspace(-limit, limit, res)
    query_pts = torch.stack(torch.meshgrid(t, t, t), -1).reshape(-1, 3).float()

    tmp_ray_sampler = RaySamplerSingleImage(data, device, render_stride=args.render_stride)

    with torch.no_grad():

        ray_batch = tmp_ray_sampler.get_all()
        ray_batch['ray_o'] = query_pts[:res * res].cuda()
        ray_batch['ray_d'] = torch.tensor([[1.0, 0.0, 0.0]]).repeat(res * res, 1).cuda()
        ray_batch['depth_range'] = torch.tensor([[0.0, 2 * limit]])

        # print(ray_batch["src_cameras"][0, 0, -16:].reshape(4, 4).inverse())
        # exit()

        featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))

        density = []
        N_rays = ray_batch['ray_o'].shape[0]
        for i in tqdm(range(0, N_rays, args.chunk_size)):
            chunk = OrderedDict()
            for k in ray_batch:
                if k in ['camera', 'depth_range', 'src_rgbs', 'src_cameras']:
                    chunk[k] = ray_batch[k]
                elif ray_batch[k] is not None:
                    chunk[k] = ray_batch[k][i:i+args.chunk_size]
                else:
                    chunk[k] = None
            pts, z_vals = sample_along_camera_ray(ray_o=chunk['ray_o'],
                                                  ray_d=chunk['ray_d'],
                                                  depth_range=chunk['depth_range'],
                                                  N_samples=res,
                                                  det=True)
            rgb_feat, ray_diff, mask = projector.compute(pts, chunk['camera'],
                                                         chunk['src_rgbs'],
                                                         chunk['src_cameras'],
                                                         featmaps=featmaps[0])
            raw_coarse = model.net_coarse(rgb_feat, ray_diff, mask)
            density.append(raw_coarse[:, :, 3])
            # raw_fine = model.net_fine(rgb_feat, ray_diff, mask)
            # density.append(raw_fine[:, :, -1])
        density = torch.cat(density, dim=0).cpu().numpy().reshape((res, res, res))

    results = measure.marching_cubes(density, threshold)
    vertices, triangles, normals, _ = [torch.from_numpy(np.ascontiguousarray(result)) for result in results]

    vertices = torch.from_numpy(np.ascontiguousarray(vertices))
    triangles = torch.from_numpy(np.ascontiguousarray(triangles))
    normals = torch.from_numpy(np.ascontiguousarray(normals))

    vertices = limit * (vertices / ((res - 1.) / 2.) - 1.)

    vertices = ((vertices.T)[[2, 0, 1]]).T
    normals = ((normals.T)[[2, 0, 1]]).T

    return vertices, triangles, normals

color_dict = {
    0: torch.tensor([0, 0, 0]),
    1: torch.tensor([255, 0, 0]),
    2: torch.tensor([0, 255, 0]),
    3: torch.tensor([0, 0, 255]),
    4: torch.tensor([255, 255, 0]),
    5: torch.tensor([250, 235, 215])
}

def extract_appearance(model, projector, data, targets, method="coarse"):
    diffuse = []
    normals = []
    N_samples = 128
    batch_size = 16
    random_range = 0.05

    tmp_ray_sampler = RaySamplerSingleImage(data, device, render_stride=args.render_stride)
    ray_batch = tmp_ray_sampler.get_all()
    ray_batch["depth_range"] = torch.tensor([[0.0, 2 * random_range]])
    featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
    ray_batch_camera_ = ray_batch['camera']

    for mesh_points in tqdm(torch.split(targets, batch_size, dim=0)):
        points = mesh_points.to("cuda")
        ray_batch['camera'] = ray_batch_camera_
        normal = - torch.nn.functional.normalize(gradient(ray_batch, points, featmaps), dim=-1)
        normals.append(normal.detach())

        with torch.no_grad():
            ray_batch["ray_o"] = points + random_range * normal
            ray_batch["ray_d"] = - normal
            ray_batch["camera"] = ray_batch["camera"].repeat(points.shape[0], 1)
            ray_batch["camera"][:, -16:] = get_rotate_matrix(ray_batch["ray_o"]).reshape(points.shape[0], -1)
            ret = render_rays_batch(ray_batch, model, featmaps,
                          projector=projector,
                          N_samples=N_samples,
                          # det=True,
                          N_importance=64)

            if method.lower() == "coarse":
                coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
                idxes_0 = torch.tensor(range(ret['outputs_coarse']['confidence'].shape[0]))
                idxes_1 = ret['outputs_coarse']['T'].argmax(dim=-1)
                coarse_pred_seg = ret['outputs_coarse']['confidence'][idxes_0, idxes_1].argmax(dim=-1).cpu()
                coarse_seg_rgb = torch.zeros_like(coarse_pred_rgb)
                for i in range(ret['outputs_coarse']['confidence'].shape[0]):
                    coarse_seg_rgb[i] = color_dict[coarse_pred_seg[i].item()]
                # coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                # diffuse.append(torch.tensor(coarse_pred_rgb))
                coarse_seg_rgb = (255 * np.clip(coarse_seg_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                diffuse.append(torch.tensor(coarse_seg_rgb))
            elif method.lower() == "fine":
                fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
                idxes_0 = torch.tensor(range(ret['outputs_fine']['confidence'].shape[0]))
                idxes_1 = ret['outputs_fine']['T'].argmax(dim=-1)
                fine_pred_seg = ret['outputs_fine']['confidence'][idxes_0, idxes_1].argmax(dim=-1).cpu()
                fine_seg_rgb = torch.zeros_like(fine_pred_rgb)
                for i in range(ret['outputs_fine']['confidence'].shape[0]):
                    fine_seg_rgb[i] = color_dict[fine_pred_seg[i].item()]
                # fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                # diffuse.append(torch.tensor(fine_pred_rgb))
                fine_seg_rgb = (255 * np.clip(fine_seg_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                diffuse.append(torch.tensor(fine_seg_rgb))
            else:
                exit("Method error! Please input (coarse/fine)")

            torch.cuda.empty_cache()

    diffuse = torch.cat(diffuse, dim=0).cpu().numpy()
    normals = torch.cat(normals, dim=0).cpu().numpy()
    return diffuse, normals

def extract_appearance_2(model, projector, data, targets, normals):
    diffuse = []
    N_samples = 128
    batch_size = 16
    random_range = 0.05

    tmp_ray_sampler = RaySamplerSingleImage(data, device, render_stride=args.render_stride)
    ray_batch = tmp_ray_sampler.get_all()
    ray_batch["depth_range"] = torch.tensor([[0.0, 2 * random_range]])
    featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
    ray_batch_camera_ = ray_batch['camera']

    for mesh_points, normal in zip(tqdm(torch.split(targets, batch_size, dim=0)), torch.split(normals, batch_size, dim=0)):
        points = mesh_points.to("cuda")
        normal = normal.to("cuda")
        ray_batch['camera'] = ray_batch_camera_
        # normal = - torch.nn.functional.normalize(gradient(ray_batch, points, featmaps), dim=-1)
        # normals.append(normal.detach())

        with torch.no_grad():
            ray_batch["ray_o"] = points + random_range * normal
            ray_batch["ray_d"] = - normal
            ray_batch["camera"] = ray_batch["camera"].repeat(points.shape[0], 1)
            ray_batch["camera"][:, -16:] = get_rotate_matrix(ray_batch["ray_o"]).reshape(points.shape[0], -1)
            ret = render_rays_batch(ray_batch, model, featmaps,
                          projector=projector,
                          N_samples=N_samples,
                          # det=True,
                          N_importance=64)

            # coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
            # coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            # diffuse.append(torch.tensor(coarse_pred_rgb))
            fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
            fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            diffuse.append(torch.tensor(fine_pred_rgb))

            torch.cuda.empty_cache()

    diffuse = torch.cat(diffuse, dim=0).cpu().numpy()
    # normals = torch.cat(normals, dim=0).cpu().numpy()
    return diffuse, normals

def gradient(ray_batch, points, featmaps):
    points = points.unsqueeze(0)
    with torch.enable_grad():
        points.requires_grad_(True)
        rgb_feat, ray_diff, mask = projector.compute(points, ray_batch['camera'],
                                                     ray_batch['src_rgbs'],
                                                     ray_batch['src_cameras'],
                                                     featmaps=featmaps[0])
        raw_coarse = model.net_coarse(rgb_feat, ray_diff, mask, with_sigmoid=False)
        # raw_coarse = model.net_coarse(rgb_feat, ray_diff, mask)
        sigma = raw_coarse[..., 3]
        d_output = torch.ones_like(sigma, requires_grad=False, device=sigma.device)
        gradients = torch.autograd.grad(
            outputs=sigma,
            inputs=points,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True, allow_unused=True)[0]
    return gradients[0]

def get_rotate_matrix(origin):
    device = origin.device
    d = torch.norm(origin, dim=1).unsqueeze(0).T
    ones = torch.ones_like(d)
    zeros = torch.zeros_like(d)
    origin_ = torch.cat([zeros, zeros, d], dim=1).to(device)
    cos, sin = angle_around_x(origin_, -origin)
    x_rot_mat = torch.cat([torch.cat([ones, zeros, zeros], dim=1).unsqueeze(-1),
                           torch.cat([zeros, cos, -sin], dim=1).unsqueeze(-1),
                           torch.cat([zeros, sin, cos], dim=1).unsqueeze(-1)], dim=2).to(device)
    tmp = torch.bmm(x_rot_mat, origin_.unsqueeze(-1)).squeeze(-1)
    cos, sin = angle_around_z(tmp, -origin)
    z_rot_mat = torch.cat([torch.cat([cos, -sin, zeros], dim=1).unsqueeze(-1),
                           torch.cat([sin, cos, zeros], dim=1).unsqueeze(-1),
                           torch.cat([zeros, zeros, ones], dim=1).unsqueeze(-1)], dim=2).to(device)
    mmm = (torch.bmm(z_rot_mat, tmp.unsqueeze(-1)).squeeze(-1) + origin).max()
    if not mmm < 1e-3:
        print(mmm)
        exit("error")
    rot_mat = torch.cat([torch.cat([torch.bmm(z_rot_mat, x_rot_mat), origin.unsqueeze(2)], dim=2),
                         torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(origin.shape[0], 1, 1).to(device)], dim=1)
    # rot_mat_axis = torch.tensor([[[0.0, -1.0, 0.0, 0.0],
    #                              [1.0, 0.0, 0.0, 0.0],
    #                              [0.0, 0.0, 1.0, 0.0],
    #                              [0.0, 0.0, 0.0, 1.0]]]).repeat(rot_mat.shape[0], 1, 1).to(device)
    # return torch.bmm(rot_mat_axis, rot_mat)
    return rot_mat

def angle_around_x(v1, v2):
    v1 = (v1.T / (torch.norm(v1, dim=1) + 1e-8)).T
    v2 = (v2.T / (torch.norm(v2, dim=1) + 1e-8)).T
    cos = (v1 * v2).sum(dim=1)
    sin = (1 - cos ** 2) ** 0.5
    mask = torch.cross(v1, v2)[:, 0] < 0
    sin[mask] = - sin[mask]
    return cos.unsqueeze(0).T, sin.unsqueeze(0).T

def angle_around_z(v1, v2):
    v1 = torch.cat([(v1[:, :2].T / (torch.norm(v1[:, :2], dim=1).T + 1e-8)).T, torch.zeros((v1.shape[0], 1)).to(v1.device)], dim=1)
    v2 = torch.cat([(v2[:, :2].T / (torch.norm(v2[:, :2], dim=1).T + 1e-8)).T, torch.zeros((v2.shape[0], 1)).to(v2.device)], dim=1)
    cos = (v1 * v2).sum(dim=1)
    sin = (1 - cos ** 2) ** 0.5
    mask = torch.cross(v1, v2)[:, 2] > 0
    sin[mask] = - sin[mask]
    return cos.unsqueeze(0).T, sin.unsqueeze(0).T

def get_config(exp_name, target):
    with open(f"out/{exp_name}/config.txt", "r") as f:
        lines = f.read().split("\n")
        for line in lines:
            if line.startswith(target):
                return int(line.split("=")[-1].strip())

if __name__ == "__main__":

    # Arguments
    parser = config.config_parser()
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name.')
    parser.add_argument('--exp_idx', type=int, default=0, help='Experiment idx.')
    parser.add_argument('--cg', type=str, default='cg_25', help='Target CG model.')
    parser.add_argument('--it', type=int, default=-1, help='Iteration to be used.')
    parser.add_argument('--method', type=str, default="corase", help='Rendering method.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--res', type=int, default=128, help='Grid resolution')
    parser.add_argument('--limit', type=float, default=1.0, help='Grid range')
    parser.add_argument('--threshold', type=float, default=0.5, help='The threshold')
    parser.add_argument('--no-color', action='store_true', help='Get color or not.')

    args = parser.parse_args()
    args.N_samples = args.res + 1
    # config_path = f"configs/pretrain_baseline_20_getmesh.txt"
    # model_path = f"out/pretraining_20_unisurf_new/model_{'%06d' % int(args.it)}.pth"
    save_path = f"mesh/mesh_{args.cg}_{args.res}_{args.exp_name}_{args.it}_{args.method}.obj"
    if not args.exp_name:
        save_path = f"mesh/mesh_{args.cg}_{args.res}_{args.exp_idx}_{args.it}_{args.method}.obj"
    os.makedirs("mesh/", exist_ok=True)
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    # init dataset
    dataset = dataset_dict[args.eval_dataset](args, 'getmesh', scenes=args.cg)
    data_loader = DataLoader(dataset, batch_size=1)
    data_loader_iterator = iter(cycle(data_loader))
    data = next(data_loader_iterator)

    # init network
    if args.it >= 0:
        if args.exp_name:
            args.ckpt_path = "out/{}/model_{:0>6}.pth".format(args.exp_name, args.it)
            args.seg_fc_hidden_layers = get_config(args.exp_name, "seg_fc_hidden_layers")
        else:
            args.ckpt_path = "out/pretraining_20_unisurf_normal_{}/model_{:0>6}.pth".format(args.exp_idx, args.it)
            args.seg_fc_hidden_layers = 0
    model = IBRNetModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)
    model.start_step = model.load_from_ckpt("out/pretraining_20_unisurf_new/", load_opt=False, load_scheduler=False)
    model.switch_to_eval()
    projector = Projector(device=device)

    # Mesh Extraction
    vertices, triangles, normals_ = extract_geometry(model, projector, data, args.res, args.limit, args.threshold)

    # Extracting the mesh appearance
    if args.no_color:
        diffuse, normals = [], []
    else:
        diffuse, normals = extract_appearance(model, projector, data, vertices, args.method)
        # diffuse, normals = extract_appearance_2(model, projector, data, vertices, normals_)

    # print(torch.cat([torch.from_numpy(normals), normals_], dim=1))
    # import pickle
    # with open("ttt.pickle", "wb") as f:
    #     pickle.dump([normals, normals_.numpy()], f)
    # exit()

    # Export model
    export_obj(vertices, triangles, diffuse, normals, save_path)

    print("[INFO] Done!")
