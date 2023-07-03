import os
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class UnionFindSet(object):
    def __init__(self, data_list):
        self.parent = {}
        self.rank = {}
        self.sets_count=len(data_list) # 判断并查集里共有几个集合, 初始化默认互相独立

        for d in data_list:
            self.parent[d] = d		   # 初始化节点的父节点为自身
            self.rank[d] = 1		   # 初始化rank为1

    def find(self, d):
        """使用递归的方式来查找根节点
        路径压缩：在查找父节点的时候，顺便把当前节点移动到父节点下面
        """
        father = self.parent[d]
        if(d != father):
            father = self.find(father)
        self.parent[d] = father
        return father

    def is_same_set(self, a,b):
        """查看两个节点是不是在一个集合里面"""
        return self.find(a) == self.find(b)

    def union(self, a, b):

        a_head = self.find(a)
        b_head = self.find(b)

        if a_head != b_head:
            a_rank = self.rank[a_head]
            b_rank = self.rank[b_head]
            if a_rank >= b_rank:
                self.parent[b_head] = a_head
                if a_rank==b_rank:
                    self.rank[a_rank]+=1
            else:
                self.parent[a_head] = b_head

            self.sets_count -= 1

def read_obj(objFilePath):
    with open(objFilePath) as file:
        edges = []
        adj_list = {}
        while True:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "f":
                A = int(strs[1].split("/")[0])-1
                B = int(strs[2].split("/")[0])-1
                C = int(strs[3].split("/")[0])-1
                edges.append((A, B))
                edges.append((A, C))
                edges.append((B, C))
                if A not in adj_list:
                    adj_list[A] = {B, C}
                else:
                    adj_list[A].add(B)
                    adj_list[A].add(C)
                if B not in adj_list:
                    adj_list[B] = {A, C}
                else:
                    adj_list[B].add(A)
                    adj_list[B].add(C)
                if C not in adj_list:
                    adj_list[C] = {A, B}
                else:
                    adj_list[C].add(A)
                    adj_list[C].add(B)
    return edges, adj_list

def delete_obj(objFilePath, ufs, target, save_path="new_mesh.obj"):

    f = open(save_path, "w")

    # Get vertices
    print("[INFO] Save vertices")
    idx = 0
    point_ids = []
    with open(objFilePath, "r") as file:
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                idx += 1
                if ufs.find(idx - 1) == target:
                    f.write(line)
                    point_ids.append(idx)

    # Get reverse index
    reverse_idx = [0] * (idx + 1)
    for i, j in enumerate(point_ids):
        reverse_idx[j] = i + 1

    # Get vertex normals
    print("[INFO] Save normals")
    idx = 0
    with open(objFilePath, "r") as file:
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "vn":
                idx += 1
                if idx in point_ids:
                    f.write(line)

    # Get faces
    print("[INFO] Save faces")
    with open(objFilePath, "r") as file:
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "f":
                id1 = int(strs[1].split("/")[0])
                id2 = int(strs[2].split("/")[0])
                id3 = int(strs[3].split("/")[0])
                if id1 in point_ids and id2 in point_ids and id3 in point_ids:
                    f.write(f"f {reverse_idx[id1]}//{reverse_idx[id1]} {reverse_idx[id2]}//{reverse_idx[id2]} {reverse_idx[id3]}//{reverse_idx[id3]}\n")

    f.close()


if __name__ == "__main__":

    # cg = "cg_1"

    # for cg in ["cg_1", "cg_2", "cg_3", "cg_5", "cg_6", "cg_7", "cg_9", "cg_10", "cg_11", "cg_12", "cg_13", "cg_15", "cg_16", "cg_17", "cg_18", "cg_19", "cg_20", "cg_21", "cg_22", "cg_24", "cg_25", "cg_26", "cg_27"]:
    # for cg in ["cg_19", "cg_20", "cg_21", "cg_22", "cg_24", "cg_25", "cg_26", "cg_27"]:
    for cg in ["cg_27"]:
        for res in [256]:

            print(cg, res)

            mesh_path = f"mesh/mesh_{cg}_{res}.obj"
            edges, adj_list = read_obj(mesh_path)

            ufs = UnionFindSet(range(len(adj_list.keys())))
            for u, v in edges:
                ufs.union(u, v)

            branches = {}
            for u in ufs.parent:
                root_u = ufs.find(u)
                if root_u not in branches:
                    branches[root_u] = [u]
                else:
                    branches[root_u].append(u)
            target = -1
            max_num = 0
            for k in branches:
                if len(branches[k]) > max_num:
                    target = k
                    max_num = len(branches[k])

            delete_obj(mesh_path, ufs, target, save_path=f"mesh/mesh_{cg}_{res}_new.obj")

    # epoch = 10000
    #
    # mesh_path = f"../out/DTU/result_cg_1/different_epochs/mesh_512_rad_{epoch}.obj"
    # edges, adj_list = read_obj(mesh_path)
    #
    # ufs = UnionFindSet(range(len(adj_list.keys())))
    # for u, v in edges:
    #     ufs.union(u, v)
    #
    # branches = {}
    # for u in ufs.parent:
    #     root_u = ufs.find(u)
    #     if root_u not in branches:
    #         branches[root_u] = [u]
    #     else:
    #         branches[root_u].append(u)
    # target = -1
    # max_num = 0
    # for k in branches:
    #     if len(branches[k]) > max_num:
    #         target = k
    #         max_num = len(branches[k])
    #
    # delete_obj(mesh_path, ufs, target, save_path=f"../out/DTU/result_cg_1/different_epochs/mesh_512_rad_{epoch}_new.obj")
