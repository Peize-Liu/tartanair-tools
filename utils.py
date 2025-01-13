import os
import numpy as np
import quaternion
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2



def extract_translation_and_rotation(poses):
    """
    Extract translation (x, y, z) and rotation (3x3 matrix) from 3x4 transformation matrices.
    """
    translations = []
    rotations = []
    for pose in poses:
        pose = pose.reshape(3, 4)
        translations.append(pose[:, 3])       # The last column is the translation vector
        rotations.append(pose[:, :3])        # The 3x3 rotation matrix
    return np.array(translations), np.array(rotations)

def plot_trajectory_with_poses_3d(translations, rotations, step=10):
    """
    Plot the 3D trajectory (x, y, z) and visualize poses every 'step' frames.
    """

    fig = plt.figure(figsize=(60, 60))
    canvas = fig.canvas
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(translations[:, 0], translations[:, 1], translations[:, 2], label="Trajectory", color="blue")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")
    ax.set_title("KITTI Trajectory with Poses (3D View)")

    # Visualize poses every 'step' frames
    for i in range(0, len(translations), step):
        x, y, z = translations[i]
        rot = rotations[i]
        origin = np.array([x, y, z])
        scale = 1.0
        x_axis = rot[:, 0] * scale
        y_axis = rot[:, 1] * scale
        z_axis = rot[:, 2] * scale

        # Draw arrows to represent the orientation (pose)
        ax.quiver(*origin, *x_axis, color='r', length=scale, normalize=True)
        # 绘制Y轴（绿色）
        ax.quiver(*origin, *y_axis, color='g', length=scale, normalize=True)
        # 绘制Z轴（蓝色）
        ax.quiver(*origin, *z_axis, color='b', length=scale, normalize=True)
    canvas.draw()
    plt.legend()
    plt.savefig("trajectory_with_poses_3d_test.png")
    plt.show()
    
def read_tartanair_pose(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found: {}".format(file_path))
    #
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, z, qx, qy, qz, qw = map(float, line.strip().split())
            poses.append([x, y, z, qx, qy, qz, qw])
    return np.array(poses)

def transform_tartanair_pose_in_kitti_format(tartan_poses):
    kitti_poses = []
    for pose in tartan_poses:
        kitti_pose = np.eye(4)
        kitti_pose[:3, 3] = pose[:3]
        qx,qy,qz,w = pose[3:]
        kitti_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion.quaternion(w, qx, qy, qz))
        kitti_pose_3x4 = kitti_pose[:3, :4]
        kitti_pose_12 = kitti_pose_3x4.reshape(-1)
        kitti_poses.append(kitti_pose_12)
        # f.write(" ".join(map(str, kitti_pose_3x4)) + "\n")
    return kitti_poses

def ned_poses_to_frd_poses(kitti_poses):
    T_ned_frd = np.array([[0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 1]])
    ned_poses = []
    for ned_pose in kitti_poses:
        #3x4 -> 4x4
        ned_pose = np.vstack([ned_pose.reshape(3,4), np.array([0,0,0,1])])
        frd_pose = T_ned_frd @ ned_pose @ np.linalg.inv(T_ned_frd)
        frd_pose = frd_pose[:3, :4]
        ned_poses.append(frd_pose.flatten())
    return ned_poses


def write_poses_to_file(file_path, poses):
    with open(file_path, 'w') as f:
        for pose in poses:
            np.savetxt(f, pose.reshape(1, 12), fmt='%.10e')
    f.close()

def read_tartanAir_calib(file_path):
    pass

def read_tartanair_depth(file_path):
    depth = np.load(file_path)
    return depth

def save_tartanair_depth_with_16bit_depth(depth, output_file):
    depth = depth.astype(np.float32)
    depth_16bit = (depth * 256).astype(np.uint16)
    cv2.imwrite(output_file, depth_16bit)
    return

def read_tartanair_semantic_seg(file_path):
    seg = np.load(file_path)
    return seg


def visualize_semantic_seg(seg):
    pass

def visualize_depth(depth):
    pass


# visualize poses in kitti format
def visualize_poses(poses):
    try:
        translations, rotations = extract_translation_and_rotation(poses)
    except Exception as e:
        print("[visualize_poses] Error input poses is not kitti 3x4 format")
        return
    plot_trajectory_with_poses_3d(translations, rotations, step=10)

def transform_tartanair_poses_to_kitti_poses(tartan_air_file,kitti_pose_file):
    tartan_air_poses = read_tartanair_pose(tartan_air_file)
    kitti_poses = transform_tartanair_pose_in_kitti_format(tartan_air_poses)
    write_poses_to_file(kitti_pose_file, kitti_poses)
    print(f"Transformed poses from {tartan_air_file} are saved to {kitti_pose_file}")
    return  kitti_poses
    

        
if __name__ == "__main__":
    # this 
    # pose_file = "/home/dji/uav_group_sharespace/SharedDatasets/TartanAir_tiny/abandonedfactory/Easy/P011/pose_left.txt"
    # tartan_air_poses = read_tartanair_pose(pose_file)
    # ned_kitti_poses = transform_tartanair_pose_in_kitti_format(tartan_air_poses)
    # frd_kitti_poses = ned_poses_to_frd_poses(ned_kitti_poses)
    # # write_poses_to_file("/home/dji/workspace/tartanair_tools/test_frd_kitti_poses.txt", frd_kitti_poses)
    # # kitti_poses = transform_tartanair_poses_to_kitti_poses("/home/dji/uav_group_sharespace/SharedDatasets/TartanAir_tiny/abandonedfactory/Easy/P011/pose_left.txt","/home/dji/workspace/tartanair_tools/test_kitti_poses.txt")
    # visualize_poses(ned_kitti_poses)
    # depth_file = "/home/dji/uav_group_sharespace/SharedDatasets/TartanAir_tiny/abandonedfactory/Easy/P000/depth_left/000000_left_depth.npy"
    # depth = read_tartanair_depth(depth_file)
    # depth_png = "/home/dji/workspace/tartanair_tools/test_depth.png"
    # save_tartanair_depth_with_16bit_depth(depth, depth_png)

    # depth = cv2.imread("/home/dji/workspace/tartanair_tools/test_depth.png", cv2.IMREAD_UNCHANGED)
    # depth = depth.astype(np.float32) / 256 #right
    # print(depth.shape)
    semantic_lable = "/home/dji/uav_group_sharespace/SharedDatasets/TartanAir_tiny/abandonedfactory/Easy/P000/seg_left/000000_left_seg.npy"
    read_tartanair_semantic_seg(semantic_lable)
    pass