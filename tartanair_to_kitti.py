import os
import numpy as np
import cv2
import utils
import shutil
import tqdm
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor

#configurations

camera_intrinsics = [[320.0, 0.0, 320.0],
                     [0.0, 320.0, 240.0], 
                     [0.0, 0.0, 1.0]]

def generate_kitti_sub_folder(tartanair_path, kitti_path, data_name="neighborhood" , verbose=False):
    left_image_path = os.path.join(tartanair_path, "image_left")
    right_image_path = os.path.join(tartanair_path, "image_right")
    left_seg_path = os.path.join(tartanair_path, "seg_left")
    left_depth_path = os.path.join(tartanair_path, "depth_left")

    kitti_left = os.path.join(kitti_path, "image_2")
    kitti_right = os.path.join(kitti_path, "image_3")
    kitti_pesudo_lidar = os.path.join(kitti_path, "velodyne")
    kitti_label = os.path.join(kitti_path, "labels") # point cloud semantic
    kitti_voxels = os.path.join(kitti_path, "voxels")
    kitti_depth = os.path.join(kitti_path, "depth")

    if not os.path.exists(kitti_left):
        os.makedirs(kitti_left)
    if not os.path.exists(kitti_right):
        os.makedirs(kitti_right)
    if not os.path.exists(kitti_pesudo_lidar):
        os.makedirs(kitti_pesudo_lidar)
    if not os.path.exists(kitti_label):
        os.makedirs(kitti_label)
    if not os.path.exists(kitti_voxels):
        os.makedirs(kitti_voxels)
    if not os.path.exists(kitti_depth):
        os.makedirs(kitti_depth)
    
    # copy poses.txt to kitti_path
    pose_file = os.path.join(tartanair_path, "pose_left.txt")
    kitti_pose_file = os.path.join(kitti_path, "poses.txt")
    utils.transform_tartanair_poses_to_kitti_poses(pose_file, kitti_pose_file)
    

    # copy calib.txt to kitti_path
    current_folder = os.path.dirname(os.path.abspath(__file__))
    calib_file = os.path.join(current_folder, "supply_files", "tartanair_calib.txt")
    kitti_calib_file = os.path.join(kitti_path, "calib.txt")
    shutil.copy(calib_file, kitti_calib_file)
    
    #generate fake times.txt with length of left_image_path
    times_file = os.path.join(kitti_path, "times.txt")
    with open(times_file, "w") as f:
        for i in range(len(os.listdir(left_image_path))):
            f.write(f"{i}\n")
    f.close()
    
    # copy left_image_path files to kitti_left_path
    left_image_files = os.listdir(left_image_path)
    left_image_files = sorted(left_image_files)
    for left_image_file in tqdm.tqdm(left_image_files, desc="Copying left images"):
        file_path = os.path.join(left_image_path, left_image_file)
        kitti_name = left_image_file.split("_")[0] + ".png"
        kitti_file_path = os.path.join(kitti_left, kitti_name)
        shutil.copy(file_path, kitti_file_path)
    
    # copy right_image_path files to kitti_right_path
    right_image_files = os.listdir(right_image_path)
    right_image_files = sorted(right_image_files)
    for right_image_file in tqdm.tqdm(right_image_files, desc="Copying right images"):
        file_path = os.path.join(right_image_path, right_image_file)
        kitti_name = right_image_file.split("_")[0] + ".png"
        kitti_file_path = os.path.join(kitti_right, kitti_name)
        shutil.copy(file_path, kitti_file_path)

    # copy left_seg and left_depth into kitti_label and kitti_pesudo_lidar and kitti_depth
    left_seg_files = os.listdir(left_seg_path)
    left_seg_files = sorted(left_seg_files)

    for left_seg_file in tqdm.tqdm(left_seg_files, desc="Processing semantic point cloud"):
        left_seg_file_path = os.path.join(left_seg_path, left_seg_file)
        cor_depth_name = left_seg_file.replace("seg", "depth")
        left_depth_file_path = os.path.join(left_depth_path, cor_depth_name)
        
        left_seg = np.load(left_seg_file_path)# tartan air label
        left_seg = utils.remap_tartanair_seg_to_kitti_label(left_seg, type=data_name)
        left_depth = np.load(left_depth_file_path)

        semantic_pointclouds = utils.generate_semantic_pointcloud(left_depth, left_seg, camera_intrinsics)
        semantic_pointclouds = semantic_pointclouds.reshape(-1, 4)

        # [x, y, z, label]

        # rotate point cloud
        R_frd_flu = np.array([  [0.0, 0.0, 1.0, 0.0],
                                [-1.0, 0.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])
        point_clouds = R_frd_flu @ semantic_pointclouds.T
        point_clouds = point_clouds.T
        point_clouds = point_clouds.astype(np.float32)

        # utils.visualize_point_cloud_with_semantics(point_clouds, utils.semantic_rgb_remap,camera_view)
        # save semantic point cloud
        kitti_pesudo_lidar_name = left_seg_file.split("_")[0] + ".bin"
        kitti_pesudo_lidar_file_path = os.path.join(kitti_pesudo_lidar, kitti_pesudo_lidar_name)
        point_clouds.tofile(kitti_pesudo_lidar_file_path)

        kitti_label_name = left_seg_file.split("_")[0] + ".label"
        kitti_label_file_path = os.path.join(kitti_label, kitti_label_name)
        valid_labels = point_clouds[:, 3].astype(np.int32)
        valid_labels.tofile(kitti_label_file_path)

        kitti_depth_name = left_seg_file.split("_")[0] + ".png"
        kitti_depth_file_path = os.path.join(kitti_depth, kitti_depth_name)
        left_depth = utils.convert_to_depth_image(left_depth)
        left_depth = left_depth.astype(np.uint16)
        cv2.imwrite(kitti_depth_file_path, left_depth)

        if verbose:
            camera_view = np.array([[0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 1.0]])
            utils.visualize_point_cloud_with_semantics(point_clouds, utils.semantic_rgb_remap, camera_view)

    #     # if 
    #     if verbose:
    #         camera_view = np.array([[0.0, -1.0, 0.0, 0.0],
    #                     [0.0, 0.0, -1.0, 0.0],
    #                     [1.0, 0.0, 0.0, 5.0],
    #                     [0.0, 0.0, 0.0, 1.0]])
            
    #         if not update_flag:
    #             vis.clear_geometries()

    #         point_clouds_3d = o3d.geometry.PointCloud()

    #         points = point_clouds[:, :3]
    #         semantics = point_clouds[:, 3].astype(np.int32)

    #         colors = np.zeros((points.shape[0], 3))
    #         for semantic_value, rgb in utils.semantic_rgb_remap.items():
    #             rgb_normalized = np.array(rgb) / 255.0
    #             colors[semantics == semantic_value] = rgb_normalized

    #         point_clouds_3d.points = o3d.utility.Vector3dVector(points)
    #         point_clouds_3d.colors = o3d.utility.Vector3dVector(colors)
    #         coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    #         vis.add_geometry(coordinate_frame)
    #         vis.add_geometry(point_clouds_3d)
    #         ctr = vis.get_view_control()
    #         ctr.set_front([0, 0, 0])
    #         camera_params = ctr.convert_to_pinhole_camera_parameters()
    #         camera_params.extrinsic = camera_view
    #         ctr.convert_from_pinhole_camera_parameters(camera_params)
    #         vis.poll_events()
    #         vis.update_renderer()
    #         update_flag = True
    
    # if verbose:
    #     vis.destroy_window()


def convert_tartanair_to_kitti(tartanair_secense_path, kitti_path):
    tartanair_paths = [os.path.join(tartanair_secense_path,"Easy"), os.path.join(tartanair_secense_path,"Hard")]
    reindex = 00
    sequnce_name = os.path.basename(tartanair_secense_path)
    for tartanair_path in tartanair_paths:
        sequences = os.listdir(tartanair_path)
        sequences = sorted(sequences)
        for sequence in sequences:
            reindex_str= format(reindex, '02d')
            sub_sequence_path = os.path.join(tartanair_path, sequence)
            output_kitti_path = os.path.join(kitti_path, reindex_str)
            if not os.path.exists(output_kitti_path):
                os.makedirs(output_kitti_path)
            generate_kitti_sub_folder(sub_sequence_path, output_kitti_path, data_name=sequnce_name, verbose=False)
            reindex += 1
            print(f"Saved {output_kitti_path}") 
    return


def process_dataset(dataset, tartanair_base_path, kitti_base_path):
    tartanair_path = os.path.join(tartanair_base_path, dataset)
    kitti_path = os.path.join(kitti_base_path, dataset)
    convert_tartanair_to_kitti(tartanair_path, kitti_path)
    # print(f"Saved {kitti_path}")

# Main function
def main():
    dataset_list = [
        "neighborhood",
        # "amusement",
        # "gascola",
        # "oldtown",
        # "seasonsforest",
    ]
    tartanair_base_path = "/home/dji/workspace/tartanair_tools/data/TartanAir"
    kitti_base_path = "/home/dji/workspace/tartanair_tools/data/TartanAirKitti"  # Assuming this is the base KITTI path

    # Use ThreadPoolExecutor for multi-threading
    with ThreadPoolExecutor() as executor:
        # Submit tasks for each dataset
        futures = [
            executor.submit(process_dataset, dataset, tartanair_base_path, kitti_base_path)
            for dataset in dataset_list
        ]

        # Optionally, wait for all tasks to complete
        for future in futures:
            future.result()  # This will raise an exception if any task fails

if __name__ == "__main__":
    # tartanair_path = "/home/dji/workspace/tartanair_tools/data/TartanAir/neighborhood"
    # kitti_path = "/home/dji/workspace/tartanair_tools/data/TartanAirKitti/neighborhood_tiny"
    # convert_tartanair_to_kitti(tartanair_path, kitti_path)
    
    
    
    # dataset_list = [
    #     "neighborhood",
    #     "amusement",
    #     "gascola",
    #     "oldtown",
    #     "seasonforest",
    # ]
    # tartanair_path = "/home/dji/workspace/tartanair_tools/data/TartanAir"

    # for dataset in dataset_list:
    #     tartanair_path = os.path.join(tartanair_path, dataset)
    #     kitti_path = os.path.join(kitti_path, dataset)
    #     convert_tartanair_to_kitti(tartanair_path, kitti_path)
    #     print(f"Saved {kitti_path}")
    main()