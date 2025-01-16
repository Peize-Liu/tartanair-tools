import os
import numpy as np
import quaternion
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import open3d as o3d


# check voxelizer
semantic_rgb_remap = {
    50: [150, 150, 0],  # 建筑
    40: [128, 64, 128],  # 马路地面
    49: [107, 142, 35],  # 草坪地面
    70: [0, 102, 0],  # 植物
    80: [153, 153, 153],  # 电线杆/电线
    10: [70, 70, 70],  # 汽车
    1: [70, 130, 180],  # 天空
    81: [220, 20, 60],  # 路牌
    72: [220, 180, 50],  # 地形
    0: [50, 50, 50],  # 其他
}


neighborhood_remap = {
    # 建筑
    105: 50, 237: 50, 217: 50, 151: 50, 85: 50, 218: 50, 222: 50, 180: 50, 
    53: 50, 95: 50, 124: 50, 170: 50, 118: 50,
    # 马路地面
    251: 40, 168: 40, 104: 40,
    # 草坪地面
    104: 49,  # 注意：104重复，覆盖为草坪地面
    # 植物
    152: 70, 224: 70, 196: 70, 85: 70, 171: 70, 253: 70,
    # 电线杆/电线
    221: 80, 205: 80,
    # 汽车
    220: 10,
    # 水面
    122: 49, 200: 49,
    # 天空
    146: 1,
    # 其他 0
}

amusement_remap = {
    # 建筑/结构
    241: 50, 161: 50, 247: 50, 228: 50, 138: 50, 217: 50, 187: 50, 174: 50, 
    242: 50, 190: 50, 28: 50,
    # 草坪地面
    205: 49, 198: 49, 192: 49,
    # 植物
    152: 70, 109: 70,
    # 天空
    182: 1,
    # 其他分类没有对应 ID，未添加到字典
}

gascola_remap = {
    # 建筑/结构
    234: 50, 75: 50, 218: 50,
    # 草坪地面
    205: 49, 219: 49,
    # 植物
    152: 70, 151: 70, 240: 70,
    # 天空
    112: 1,
    # 其他分类没有对应 ID，未添加到字典
}

oldtown_remap = {
    # 建筑
    53: 50, 174: 50, 156: 50, 170: 50, 196: 50, 232: 50, 59: 50, 199: 50,
    # 马路地面
    222: 40, 91: 40, 56: 40, 205: 40, 244: 40,
    # 植物
    208: 70, 46: 70,
    # 电线杆/电线
    221: 80, 253: 80, 237: 80,
    # 天空
    223: 1,
    # 路牌
    209: 81, 128: 81
}

seasonsforest_remap = {
    # 建筑/结构
    246: 50, 
    239: 50, 
    73: 50, 
    175: 50, 
    86: 50, 
    247: 50, 
    98: 50, 
    102: 50,
    # 地形
    205: 72,
    # 植物
    152: 70,
    # 天空
    196: 1
}

camera_intrinsics = [[320.0, 0.0, 320.0],
                     [0.0, 320.0, 240.0], 
                     [0.0, 0.0, 1.0]]


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

def remap_tartanair_seg_to_kitti_label(seg, type= "neighborhood"):
    tartan_air_label = np.array(seg)
    kitti_matrix = np.zeros_like(seg,dtype = np.int32)
    if type == "neighborhood":
        remap_dict = neighborhood_remap
    elif type == "amusement":
        remap_dict = amusement_remap
    elif type == "gascola":
        remap_dict = gascola_remap
    elif type == "oldtown":
        remap_dict = oldtown_remap
    elif type == "seasonsforest":
        remap_dict = seasonsforest_remap
    else:
        raise ValueError("Invalid type")
    for k, v in remap_dict.items():
        kitti_matrix[tartan_air_label == k] = v
    return kitti_matrix

def generate_semantic_pointcloud(depth, seg, cam_k):
    cam_k_inv = np.linalg.inv(cam_k)
    height, width = depth.shape
    seg = seg.reshape(height, width)
    cam_k = np.array(cam_k).reshape(3, 3)
    points = []
    for v in range(height):
        for u in range(width):
            Z = depth[v, u]
            if Z > 1000:
                continue
            if Z == 0:
                continue
            X = (u - cam_k[0, 2]) * Z / cam_k[0, 0]
            Y = (v - cam_k[1, 2]) * Z / cam_k[1, 1]
            points.append([X, Y, Z, seg[v, u]])
    return np.array(points)

def visualize_point_cloud_with_semantics(point_cloud_data, semantic_to_rgb_map, camera_view = None):
    """
    可视化点云数据，并根据 semantic 映射 RGB 颜色。
    
    参数:
        point_cloud_data (numpy.ndarray): 点云数据，形状为 [N, 4]，格式为 [x, y, z, semantic]。
        semantic_to_rgb_map (dict): 语义到 RGB 的映射，例如 {0: (255, 0, 0), 1: (0, 255, 0)}。
        view angle np.array([4,4]): camera view angle
        [[1,0,0, 0],
        [0,1,0, 0],
        [0,0,1, 5],
        [0,0,0, 1]]
    """
    # 检查数据格式
    if point_cloud_data.shape[1] != 4:
        raise ValueError("点云数据应为 [x, y, z, semantic] 格式，形状为 [N, 4]。")
    
    # 提取点的坐标 (x, y, z)
    points = point_cloud_data[:, :3]
    
    # 获取语义标签
    semantics = point_cloud_data[:, 3].astype(int)

    # 为每个点分配颜色
    colors = np.zeros((points.shape[0], 3))  # 初始化颜色数组为全黑
    for semantic_value, rgb in semantic_to_rgb_map.items():
        # 将 RGB 值从 0-255 范围转换为 0-1 范围
        rgb_normalized = np.array(rgb) / 255.0
        colors[semantics == semantic_value] = rgb_normalized

    if camera_view is None:
        camera_view = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 5],
                                [0, 0, 0, 1]])
    # 创建 Open3D 的点云对象
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
    point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=[0, 0, 0])

    vis.add_geometry(coordinate_frame) 
    vis.add_geometry(point_cloud_o3d)
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 0])
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = camera_view
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    vis.run()
    vis.destroy_window()

    # # 可视化点云
    # o3d.visualization.draw_geometries([point_cloud_o3d],
    #                                   window_name="Point Cloud Visualization",
    #                                   width=800,
    #                                   height=600)

def visualize_pointcloud(pointcloud):
    # if without semantic label
    if pointcloud.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c='b', marker='.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    # if with semantic label
    elif pointcloud.shape[1] == 4:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(pointcloud.shape[0]):
            x, y, z, label = pointcloud[i]
            label = int(label)
            ax.scatter(x, y, z, c=[tuple(np.array(semantic_rgb_remap[label])/255)], marker='.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    else:
        raise ValueError("Invalid pointcloud shape")
    


def visualize_semantic_seg(seg):
    pass

def visualize_depth(depth):
    pass

def convert_to_depth_image(depth):
    UINT16_MAX = 65535
    # 创建一个与输入深度图相同形状的数组
    output_depth_map = np.full_like(depth, UINT16_MAX, dtype=np.uint16)
    # 找到深度值小于等于 1000 的区域
    valid_mask = depth <= 1000

    # 对有效区域的深度值进行转换
    output_depth_map[valid_mask] = (depth[valid_mask] * 256.0).astype(np.uint16)
    return output_depth_map

def check_depth_png(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 256
    print(depth.shape)

def read_kitti_poses(kitti_pose):
    poses = []
    with open(kitti_pose, 'r') as f:
        for line in f:
            pose = np.array(list(map(float, line.strip().split())))
            poses.append(pose)
    return np.array(poses)

def read_semantic_pointcloud(file_path):
    pointcloud = np.load(file_path)
    
    

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
    frd_kitti_poses = ned_poses_to_frd_poses(kitti_poses)
    write_poses_to_file(kitti_pose_file, frd_kitti_poses)
    print(f"Transformed poses from {tartan_air_file} are saved to {kitti_pose_file}")
    return  kitti_poses
    

        
if __name__ == "__main__":
    # this 
    pose_file = "/home/dji/uav_group_sharespace/SharedDatasets/TartanAir_tiny/abandonedfactory/Easy/P011/pose_left.txt"
    # tartan_air_poses = read_tartanair_pose(pose_file)
    # ned_kitti_poses = transform_tartanair_pose_in_kitti_format(tartan_air_poses)
    # frd_kitti_poses = ned_poses_to_frd_poses(ned_kitti_poses)
    # write_poses_to_file("/home/dji/workspace/tartanair_tools/test_frd_kitti_poses.txt", frd_kitti_poses)
    # kitti_poses = transform_tartanair_poses_to_kitti_poses("/home/dji/uav_group_sharespace/SharedDatasets/TartanAir_tiny/abandonedfactory/Easy/P011/pose_left.txt","/home/dji/workspace/tartanair_tools/test_kitti_poses.txt")
    # visualize_poses(ned_kitti_poses)
    # depth_file = "/home/dji/uav_group_sharespace/SharedDatasets/TartanAir_tiny/abandonedfactory/Easy/P000/depth_left/000000_left_depth.npy"
    # depth = read_tartanair_depth(depth_file)
    # depth_png = "/home/dji/workspace/tartanair_tools/test_depth.png"
    # save_tartanair_depth_with_16bit_depth(depth, depth_png)

    # depth = cv2.imread("/home/dji/workspace/tartanair_tools/test_depth.png", cv2.IMREAD_UNCHANGED)
    # depth = depth.astype(np.float32) / 256 #right
    # print(depth.shape)
    semantic_lable = "/home/dji/uav_group_sharespace/SharedDatasets/TartanAir/neighborhood/Easy/P000/seg_left/000000_left_seg.npy"
    depth = read_tartanair_depth("/home/dji/uav_group_sharespace/SharedDatasets/TartanAir/neighborhood/Easy/P000/depth_left/000000_left_depth.npy")
    cam_k = np.array(camera_intrinsics)
    seg = read_tartanair_semantic_seg(semantic_lable)
    seg = remap_tartanair_seg_to_kitti_label(seg, type="neighborhood")
    pointcloud = generate_semantic_pointcloud(depth, seg, cam_k)
    visualize_point_cloud_with_semantics(pointcloud, semantic_rgb_remap)

    # pass