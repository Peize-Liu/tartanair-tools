import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


# Display semmantic label with corresponding image

def depth2vis(depth, maxthresh = 50):
    depthvis = np.clip(depth,0,maxthresh)
    depthvis = depthvis/maxthresh*255
    depthvis = depthvis.astype(np.uint8)
    depthvis = np.tile(depthvis.reshape(depthvis.shape+(1,)), (1,1,3))

    return depthvis

def seg2vis(segnp):
    colors = np.loadtxt('/home/dji/workspace/tartanair_tools/seg_rgbs.txt')
    segvis = np.zeros(segnp.shape+(3,), dtype=np.uint8)

    for k in range(256):
        mask = segnp==k
        colorind = k % len(colors)
        if np.sum(mask)>0:
            segvis[mask,:] = colors[colorind]

    return segvis


def get_sorted_file_list(folder):
    """
    获取文件夹内所有文件的完整路径，并按照文件名排序。
    """
    if not os.path.exists(folder):
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return sorted(files)

def show_seg_and_image(sequence_path, mode="Easy"):
    base_path = os.path.join(sequence_path,mode)
    subfolders = os.listdir(base_path)

    # 遍历每个 P00X 文件夹
    all_image_pairs = []
    for subfolder in subfolders:
        seg_left_path = os.path.join(base_path, subfolder, "seg_left")
        image_left_path = os.path.join(base_path, subfolder, "image_left")

        # 获取 seg_left 和 image_left 文件夹下的图片
        seg_files = get_sorted_file_list(seg_left_path)
        image_files = get_sorted_file_list(image_left_path)

        # 将两组图片按照索引配对（假设 seg_left 和 image_left 中图片数量相等）
        for seg_file, image_file in zip(seg_files, image_files):
            all_image_pairs.append((seg_file, image_file))

    # 如果没有图片，直接退出
    if not all_image_pairs:
        print("未找到任何图片！")
        return

    # 遍历图片对并显示
    index = 0
    while index < len(all_image_pairs):
        seg_file, image_file = all_image_pairs[index]

        # 读取图片
        seg_image = np.load(seg_file)
        seg_image = seg2vis(seg_image)
        seg_image = cv2.cvtColor(seg_image,cv2.COLOR_BGR2RGB)
        image_left = cv2.imread(image_file)

        # 检查图片是否成功读取
        if seg_image is None or image_left is None:
            print(f"无法读取图片: {seg_file} 或 {image_file}")
            index += 1
            continue

        # 显示图片
        cv2.imshow("Seg Left", seg_image)
        cv2.imshow("Image Left", image_left)

        # 等待用户按键
        key = cv2.waitKey(0)

        # 按下右箭头键，显示下一组图片
        if key == 83:  # 右箭头键的 ASCII 码
            index += 1
        # 按下 q 键，退出程序
        elif key == ord('q'):
            break

    # 释放所有窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    data_root = "/home/dji/uav_group_sharespace/SharedDatasets/TartanAir/neighborhood"
    semantic_lable = "/home/dji/uav_group_sharespace/SharedDatasets/TartanAir/neighborhood/Easy/P000/seg_left/000000_left_seg.npy"
    semantic = np.load(semantic_lable)
    seg = seg2vis(semantic)
    cv2.imshow("seg",seg)
    cv2.waitKey(0)

    # depth = data_root + "/Easy/P000/depth_left/000020_left_depth.npy"
    # show_seg_and_image(data_root,"Easy")