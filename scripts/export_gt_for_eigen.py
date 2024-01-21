import os
import numba
import numpy as np
from tqdm import tqdm


@numba.jit(nopython=True)
def get_depth_map(points_in_img, depth_map):
    for x, y, depth in points_in_img:
        depth_map[int(y), int(x)] = min(depth_map[int(y), int(x)], depth)


def get_depth_by_velo_and_param(c2c_txt, v2c_txt, velo_bin, vel_depth=True):
    with open(c2c_txt, "r") as rf:
        c2c_lines = rf.readlines()
        for line in c2c_lines:
            if line.startswith("R_rect_00"):
                rect = np.eye(4)
                rect[:3, :3] = \
                    np.array([float(d) for d in line.split(":")[-1].split()]).reshape(3, 3).astype(np.float32)
            if line.startswith("P_rect_02"):
                project_2 = np.array([float(d) for d in line.split(":")[-1].split()]).reshape(3, 4).astype(np.float32)
            if line.startswith("S_rect_02"):
                img_shape = np.array([float(d) for d in line.split(":")[-1].split()]).astype(np.int32)
    with open(v2c_txt, 'r') as rf:
        v2c_lines = rf.readlines()
        for line in v2c_lines:
            if line.startswith("R"):
                r_v2c = np.array([float(d) for d in line.split(":")[-1].split()]).reshape(3, 3).astype(np.float32)
            if line.startswith("T"):
                t_v2c = np.array([float(d) for d in line.split(":")[-1].split()]).astype(np.float32)
        trans_v2c = np.eye(4, dtype=np.float32)
        trans_v2c[:3, :3] = r_v2c
        trans_v2c[:3, -1] = t_v2c
    transform = np.dot(np.dot(project_2, rect), trans_v2c)
    points = np.fromfile(velo_bin, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0
    points = points[points[:, 0] >= 0, :]
    points_in_cam = np.dot(transform, points.T).T
    points_in_cam[:, :2] = points_in_cam[:, :2] / points_in_cam[:, 2:3]
    if vel_depth:
        points_in_cam[:, 2] = points[:, 0]
    points_in_cam[:, :2] = np.round(points_in_cam[:, :2]) - 1
    gt_index = np.bitwise_and(points_in_cam[:, 0] >= 0, points_in_cam[:, 1] >= 0)
    lt_index = np.bitwise_and(points_in_cam[:, 0] < img_shape[0], points_in_cam[:, 1] < img_shape[1])
    valid_index = np.bitwise_and(gt_index, lt_index)
    points_in_cam = points_in_cam[valid_index, :]
    depth_map = np.ones(img_shape[::-1]) * 1000
    get_depth_map(points_in_cam, depth_map)
    depth_map[np.bitwise_or(depth_map == 1000, depth_map < 0)] = 0
    return depth_map


def get_depth_for_eigen():
    kitti_data_dir = "/home/lion/large_data/data/kitti/raw"
    split_txt = "../splits/eigen/test_files.txt"
    gt_depths = list()
    with open(split_txt, "r") as rf:
        split_lines = rf.readlines()
    for line in tqdm(split_lines):
        folder, frame_id, _ = line.split()
        dir_name = folder.split("/")[0]
        calib_c2c_path = os.path.join(kitti_data_dir, dir_name, "calib_cam_to_cam.txt")
        calib_v2c_path = os.path.join(kitti_data_dir, dir_name, "calib_velo_to_cam.txt")
        velo_bin_path = os.path.join(kitti_data_dir, folder, "velodyne_points", "data", "{:s}.bin".format(frame_id))
        assert os.path.exists(calib_c2c_path) and os.path.exists(calib_v2c_path) and os.path.exists(velo_bin_path)
        gt_depth = get_depth_by_velo_and_param(calib_c2c_path, calib_v2c_path, velo_bin_path)
        gt_depths.append(gt_depth.astype(np.float32))
    output_path = os.path.join("../splits", "gt_depths.npz")
    np.savez_compressed(output_path, data=np.array(gt_depths, dtype=object))


def read_from_file():
    data = np.load("../splits/gt_depths.npz", allow_pickle=True)["data"]
    print(len(data))


if __name__ == '__main__':
    read_from_file()
