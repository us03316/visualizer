import sys
if sys.prefix == '/home/lee/anaconda3/envs/ros2':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/lee/ros2_ws/src/visualizer/install/visualizer'
