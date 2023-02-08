from logging import INFO

from utils.confort import BaseConfig
from utils.input import RealSense
import pyrealsense2 as rs


class Logging(BaseConfig):
    level = INFO


class Network(BaseConfig):
    ip = 'localhost'
    port = 50000
    out_queues = ['source_human', 'source_grasping']
    # make the output queue blocking (can be used to put a breakpoint in the sink and debug the process output)
    blocking = False


class Input(BaseConfig):
    camera = RealSense

    class Params:
        rgb_res = (640, 480)
        depth_res = (640, 480)
        fps = 30
        # depth_format = rs.format.z16  # TODO MAY CAUSE PROBLEM
        # color_format = rs.format.rgb8  # TODO MAY CAUSE PROBLEM
        # from_file = 'assets/robot_arena_videos/tilting_camera.bag'
        skip_frames = True
