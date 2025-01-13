import tartanair as ta
# Initialize TartanAir.
tartanair_data_root = '/home/dji/workspace/tartanair_tools/tartanair-v2'
ta.init(tartanair_data_root)

# Download a trajectory.
ta.download(env = "ArchVizTinyHouseDay",
            difficulty = ['easy'], # this can be 'easy', and/or 'hard'
            modality = ['image', 'depth', 'seg', 'imu'], # available modalities are: image', 'depth', 'seg', 'imu', 'lidar', 'flow', 'pose'
            camera_name = ['lcam_front', 'lcam_left', 'lcam_right', 'lcam_back', 'lcam_top', 'lcam_bottom'],
 ) # unzip files autonomously after download