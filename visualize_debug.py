import utils
import numpy as np
# poses = "/home/dji/workspace/tartanair_tools/data/TartanAirKitti/neighborhood_test/00/poses.txt"
# poses = utils.read_kitti_poses(poses)
# utils.visualize_poses(poses)
pointcloud = "/home/dji/workspace/tartanair_tools/data/TartanAirKitti/neighborhood_test/00/velodyne/000000.bin"
pointcloud_semantic =  np.fromfile(pointcloud, dtype=np.float32).reshape(-1, 4)
utils.visualize_point_cloud_with_semantics(pointcloud_semantic, utils.semantic_rgb_remap)


