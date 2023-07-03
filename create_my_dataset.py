import re
import os
import shutil
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils


def int2str(a):
    if a < 10:
        return "00000" + str(a)
    if a < 100:
        return "0000" + str(a)
    if a < 1000:
        return "000" + str(a)
    if a < 10000:
        return "00" + str(a)
    if a < 100000:
        return "0" + str(a)
    return str(a)

def create(model_name, img_num):

    name = "mydataset"
    source_path = f"../mydataset/data/{model_name}/scan/"
    target_path = f"data/{name}/{model_name}/"

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # depth folder
    ## no depth folder

    # pose folder
    pose_folder = os.path.join(target_path, "pose/")
    if not os.path.exists(pose_folder):
        os.makedirs(pose_folder)
    cameras = np.load(os.path.join(source_path, "cameras.npz"))
    for idx in range(img_num):
        rot_mat = np.linalg.inv(cameras[f"camera_mat_{idx}"] @ cameras[f"world_mat_{idx}"])
        with open(os.path.join(pose_folder, int2str(idx) + ".txt"), "w") as f:
            f.write(" ".join([str(a) for a in rot_mat.reshape(-1)]))

    # rbg folder
    rgb_folder = os.path.join(target_path, "rgb/")
    if not os.path.exists(rgb_folder):
        os.makedirs(rgb_folder)
    for idx in range(img_num):
        shutil.copy(os.path.join(source_path, "image", int2str(idx) + ".png"), os.path.join(target_path, "rgb", int2str(idx) + ".png"))

    # intrisics.txt
    with open(os.path.join(target_path, "intrinsics.txt"), "w") as f:
        f.write("711.111 256. 256. 0.\n0. 0. 0.\n0.\n6.\n512. 512.")

if __name__ == "__main__":
    img_num = 20

    # for idx in range(27):
    #     if idx in [13, 22]:
    #         continue
    #     model_name = f"cg_{idx + 1}"
    #     print(model_name)
    #     create(model_name, img_num)

    create("cg_25_white", img_num)
