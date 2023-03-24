from logging import INFO

from utils.concurrency.generic_node import GenericNode
from utils.concurrency.py_queue import PyQueue
from utils.concurrency.yarp_queue import YarpQueue
from utils.confort import BaseConfig


class Logging(BaseConfig):
    class Logger:
        class Params:
            level = INFO  # Minimum logging level or list of logging levels
            recurring = False

    debug = True
    # options: rgb depth mask 'fps center hands partial scene reconstruction transform


class Network(BaseConfig):
    node = GenericNode

    #  /realsense-holder-publisher/pose:o
    class Args:
        in_queues = {
            'from_seg': PyQueue(ip="localhost", port=50000, queue_name='seg_to_gc', blocking=True),
            'from_pose_streamer': YarpQueue(remote_port_name='/ergocub-rs-pose/pose:o',
                                            local_port_name='/VisualPerception/ShapeCompletion/camera_pose:i',
                                            data_type='list', read_format='camera_pose', blocking=True),
        }
