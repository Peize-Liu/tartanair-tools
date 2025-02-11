import os
import threading
import subprocess

# 定义目标文件夹的路径
TARGET_FOLDER = "/home/dji/uav_nas_local/SharedDatasets/TartanAirKitti/neighborhood"  # 替换为实际文件夹路径

# 定义工具和配置文件
GEN_TOOL = "/home/dji/workspace/voxelizer/bin/gen_data"
CONFIG_FILE = "/home/dji/workspace/voxelizer/bin/settings.cfg"

def execute_command(subfolder):
    """
    执行生成命令的线程函数
    """
    voxels_path = os.path.join(subfolder, "voxels")
    command = [GEN_TOOL, CONFIG_FILE, subfolder, voxels_path]
    
    print(f"执行命令: {' '.join(command)}")
    
    try:
        # 执行命令
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"子文件夹 {subfolder} 执行完成，输出:\n{result.stdout}")
        if result.stderr:
            print(f"子文件夹 {subfolder} 错误信息:\n{result.stderr}")
    except Exception as e:
        print(f"子文件夹 {subfolder} 执行出错: {e}")

def main():
    # 获取文件夹中的所有子文件夹
    subfolders = [os.path.join(TARGET_FOLDER, f) for f in os.listdir(TARGET_FOLDER)
                  if os.path.isdir(os.path.join(TARGET_FOLDER, f)) and f.isdigit()]
    
    # 按自然顺序排序子文件夹（如 00, 01, 02）
    subfolders.sort(key=lambda x: int(os.path.basename(x)))
    
    # 创建线程列表
    threads = []
    
    for subfolder in subfolders:
        # 为每个子文件夹创建一个线程
        thread = threading.Thread(target=execute_command, args=(subfolder,))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()