import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_obj(objFilePath):
    with open(objFilePath) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append(np.array([float(strs[1]), float(strs[2]), float(strs[3])]))
            else:
                continue
    return np.array(points)


if __name__ == "__main__":

    # with open("tmp.pickle", "rb") as f:
    #     ttt, ppp = pickle.load(f)
    # ttt, ppp = ttt.T, ppp.T

    with open("tmp.pickle", "rb") as f:
        diffs, pts = pickle.load(f)
    diffs = diffs[0, 0, :, -1]
    sss = diffs.argsort()
    pts = pts[0].T

    colors = np.array([[255, 240, 240]] * 20)
    colors[sss[5:10]] = [255, 160, 160]
    colors[sss[10:15]] = [255, 80, 80]
    colors[sss[15:]] = [255, 0, 0]
    colors = colors / 255

    cameras = np.load("../mydataset/new_data/cg_25/cameras.npz")
    ccc = np.array([np.linalg.inv(cameras[f"rot_mat_{i}"])[:3, 3] for i in range(20)]).T

    yyy = read_obj("mesh/mesh_cg_25_128_new_.obj").T

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.scatter(-yyy[1], yyy[0], -yyy[2], c="g", s=0.01)
    ax.scatter(ccc[0], ccc[1], ccc[2], c=colors, s=2)
    ax.scatter(-pts[1], pts[0], -pts[2], c="b", s=2)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.set_zlabel('z', fontsize=16)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_zlim(-3.2, 3.2)
    plt.show()
