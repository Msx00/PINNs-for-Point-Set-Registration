import os
import numpy as np
import random
from datetime import datetime


def load_ori_mr_us(abs_dir):
    mr_ori_list = []
    us_ori_list = []

    files = os.listdir(abs_dir)

    mr_npy_files = [f for f in files if f.endswith('point_set_mri_original_list.npy')]
    mr_npy_files.sort(key=lambda x: int(x.split('_')[0]))
    for file in mr_npy_files:
        file_path = os.path.join(abs_dir, file)
        data = np.load(file_path)
        mr_ori_list.append(data[0])

    us_npy_files = [f for f in files if f.endswith('point_set_us_original_list.npy')]
    us_npy_files.sort(key=lambda x: int(x.split('_')[0]))
    for file in us_npy_files:
        file_path = os.path.join(abs_dir, file)
        data = np.load(file_path)
        us_ori_list.append(data[0])
    return np.array(mr_ori_list), np.array(us_ori_list)

def load_us_centroid_distance(abs_dir):
    mr_centroid_list = []
    us_centroid_list = []
    mr_furthest_distance_list = []
    us_furthest_distance_list = []

    files = os.listdir(abs_dir)

    centroid_npy_files = [f for f in files if f.endswith('centroid.npy')]
    centroid_npy_files.sort(key=lambda x: int(x.split('_')[0]))
    for file in centroid_npy_files:
        file_path = os.path.join(abs_dir, file)
        data = np.load(file_path)

        mr_centroid_list.append(data[0,:])
        us_centroid_list.append(data[1,:])
        print(f"Read file: {file_path}")
        print(f"Data shape: {data.shape}")
        print(f"Data content:\n{data}\n")

    fd_npy_files = [f for f in files if f.endswith('furthest_distance.npy')]
    fd_npy_files.sort(key=lambda x: int(x.split('_')[0]))
    for file in fd_npy_files:
        file_path = os.path.join(abs_dir, file)
        data = np.load(file_path)
        mr_furthest_distance_list.append(data[0])
        us_furthest_distance_list.append(data[1])

    print("\nDebugging Information:")
    print(
        f"mr_centroid_list: Number of elements = {len(mr_centroid_list)}, Shape of each element = {np.array(mr_centroid_list).shape}")
    print(
        f"us_centroid_list: Number of elements = {len(us_centroid_list)}, Shape of each element = {np.array(us_centroid_list).shape}")
    print(
        f"mr_furthest_distance_list: Number of elements = {len(mr_furthest_distance_list)}, Shape of each element = {np.array(mr_furthest_distance_list).shape}")
    print(
        f"us_furthest_distance_list: Number of elements = {len(us_furthest_distance_list)}, Shape of each element = {np.array(us_furthest_distance_list).shape}")

    return np.array(mr_centroid_list), np.array(us_centroid_list), np.array(mr_furthest_distance_list), np.array(us_furthest_distance_list)#, np.array(mr_gt)


def load_normalised_mr(abs_dir = "./data/sampled_points_surface_all_patients"):
    datalist = []
    files = [f for f in os.listdir(abs_dir) if f.endswith('.npy')]
    files.sort(key=lambda x: int(x[:-4]))
    for file in files:
        file_path = os.path.join(abs_dir, file)
        data = np.load(file_path)
        datalist.append(data)
        print("load_normalised_mr begin :")
        print(f"读取文件: {file}, 数据形状: {data.shape}")
    return datalist


def de_normalise_pointcloud(points, centroid, furthest_distance):
    points = points.astype(float)
    centroid = centroid.astype(float)
    furthest_distance = furthest_distance.astype(float)

    points *= furthest_distance
    points += centroid
    return points


def normalise_pointcloud(points):
    points   = points.astype(float)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    furthest_distance = np.array([np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))])

    points /= furthest_distance 

    return points, centroid, furthest_distance


def compute_rmse(pc1, pc2):
    if pc1.shape != pc2.shape:
        raise ValueError("Point clouds must have the same shape")
    distances = np.linalg.norm(pc1 - pc2, axis=1)
    rmse = np.sqrt(np.mean(distances**2))
    return rmse


def dataset():
    abs_dir = "./data/gt_centroid_distance"

    normalised_mrs = load_normalised_mr()
    mr_centroid_list, us_centroid_list, mr_furthest_distance_list,us_furthest_distance_list = load_us_centroid_distance(abs_dir)
    ori_mr_list,ori_us_list = load_ori_mr_us(abs_dir)

    pointsets_list = []
    mr_centroids_list = []
    us_centroids_list = []
    mr_fds_list = []
    us_fds_list = []

    rmse_s = 0
    for i in range(0, 1):
        individual_normalmr = normalised_mrs[i]
        # print(f"normalised_mrs[{i}] shape:", individual_normalmr.shape)

        denormalise_mr = de_normalise_pointcloud(individual_normalmr[0], mr_centroid_list[i], mr_furthest_distance_list[i])

        ori_mr = ori_mr_list[i]
        ori_us = ori_us_list[i]

        indices = []
        for row in denormalise_mr:
            index = np.where((ori_mr == row).all(axis=1))[0]
            if index.size > 0:
                indices.append(index[0])

        re_sample_us = ori_us[indices]
        rmse = compute_rmse(re_sample_us,denormalise_mr)

        normalised_mr, mr_centroid, mr_furthest_distance= normalise_pointcloud(denormalise_mr)
        normalised_us, us_centroid, us_furthest_distance= normalise_pointcloud(re_sample_us)

        pointsets = np.stack((normalised_mr, normalised_us))
        centroids = np.stack((mr_centroid, us_centroid))
        fds = np.stack((mr_furthest_distance, us_furthest_distance))

        pointsets_list.append(pointsets)
        mr_centroids_list.append(mr_centroid)
        us_centroids_list.append(us_centroid)
        mr_fds_list.append(mr_furthest_distance)
        us_fds_list.append(us_furthest_distance)

    return pointsets_list, mr_centroids_list, us_centroids_list, mr_fds_list, us_fds_list


# if __name__=="__main__":
#     abs_dir = "/home/data/msx/Project/MICCAI_PINNS/dataset/Simulation/dataset/gt_centroid_distance"

#     normalised_mrs = load_normalised_mr()
#     mr_centroid_list, us_centroid_list, mr_furthest_distance_list,us_furthest_distance_list = load_us_centroid_distance(abs_dir)
#     ori_mr_list,ori_us_list = load_ori_mr_us(abs_dir)

#     pointsets_list = []
#     centroids_list = []
#     fds_list = []

#     rmse_s = 0
#     for i in range(0, 22):
#         individual_normalmr = normalised_mrs[i]
#         # print(individual_normalmr.shape, individual_normalmr[0].shape)
#         denormalise_mr = de_normalise_pointcloud(individual_normalmr[0], mr_centroid_list[i], mr_furthest_distance_list[i])

#         ori_mr = ori_mr_list[i]
#         ori_us = ori_us_list[i]

#         indices = []
#         for row in denormalise_mr:
#             index = np.where((ori_mr == row).all(axis=1))[0]
#             if index.size > 0:
#                 indices.append(index[0])

#         # print(indices)

#         re_sample_us = ori_us[indices]
#         # print(re_sample_us)
#         rmse = compute_rmse(re_sample_us,denormalise_mr)

#         print("rmse", rmse)

#         normalised_mr, mr_centroid, mr_furthest_distance= normalise_pointcloud(denormalise_mr)
#         normalised_us, us_centroid, us_furthest_distance= normalise_pointcloud(re_sample_us)

#         # print(normalised_mr.shape, mr_centroid.shape, mr_furthest_distance.shape)

#         pointsets = np.stack((normalised_mr, normalised_us))
#         centroids = np.stack((mr_centroid, us_centroid))
#         fds = np.stack((mr_furthest_distance, us_furthest_distance))

#         pointsets_list.append(pointsets)
#         centroids_list.append(centroids)
#         fds_list.append(fds)

#         rmse = compute_rmse(de_normalise_pointcloud(normalised_mr, mr_centroid, mr_furthest_distance), 
#                             de_normalise_pointcloud(normalised_us, us_centroid, us_furthest_distance))

#         print(rmse)
#         rmse_s = rmse_s + rmse
#     print(rmse_s/22)
