from logging import INFO

from utils.concurrency import SrcYarpNode
from utils.concurrency.generic_node import GenericNode
from utils.concurrency.yarp_queue import YarpQueue
from utils.confort import BaseConfig
from utils.input import RealSense
from utils.winrealsesnse import WinRealSense
import pyrealsense2 as rs


class Logging(BaseConfig):
    level = INFO


# class Network(BaseConfig):
#     node = SrcYarpNode
#
#     class Args:
#         out_queues = {'depthCamera': ['rgbImage', 'depthImage']}

class Network(BaseConfig):
    node = GenericNode

    class Args:
        out_queues = {
            # in_port_name, out_port_name, data_type, out_name
            'rgb': YarpQueue(local_port_name='/depthCamera/rgbImage:r',
                             data_type='rgb', write_format='rgb', blocking=False),
            'depth': YarpQueue(local_port_name='/depthCamera/depthImage:r',
                             data_type='depth', write_format='depth', blocking=False)
        }


class Input(BaseConfig):
    camera = RealSense

    class Params:
        rgb_res = (640, 480)
        depth_res = (640, 480)
        fps = 30
        # depth_format = rs.format.z16  # TODO MAY CAUSE PROBLEM
        # color_format = rs.format.rgb8  # TODO MAY CAUSE PROBLEM
        from_file = 'assets/robot_arena_videos/tilting_camera.bag'
        skip_frames = True

