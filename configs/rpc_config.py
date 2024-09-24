from logging import INFO
from utils.concurrency.generic_node import GenericNode
from utils.concurrency.py_queue import PyQueue
from utils.confort import BaseConfig
from utils.concurrency.utils.signals import Signals
import numpy as np


class Logging(BaseConfig):
    level = INFO


class Network(BaseConfig):
    node = GenericNode

    class Args:
        in_queues = {
            'from_segmentation': PyQueue(ip="localhost", port=50000, queue_name='seg_to_rpc', blocking=False, read_format={"obj_distance": Signals.USE_LATEST,
                                                                                                                          "obj_center": Signals.USE_LATEST}),
            'from_grasp_detection': PyQueue(ip="localhost", port=50000, queue_name='gd_to_rpc', blocking=False, read_format={"hands_root_frame": Signals.USE_LATEST}),
            'focus_to_rpc': PyQueue(ip="localhost", port=50000, queue_name='focus_to_rpc', blocking=False, read_format={"focus": Signals.USE_LATEST,
                                                                                                                        "face_point": Signals.USE_LATEST}),
            'ar_to_rpc': PyQueue(ip="localhost", port=50000, queue_name='ar_to_rpc', blocking=False, read_format={"action": Signals.USE_LATEST}),
            'hpe_to_rpc': PyQueue(ip="localhost", port=50000, queue_name='hpe_to_rpc', blocking=False, read_format={"human_position": Signals.USE_LATEST, 
                                                                                                                    'human_occupancy': Signals.USE_LATEST, 'human_pixels':Signals.USE_LATEST,
                                                                                                                    'yarp_read_time': Signals.USE_LATEST}),
        }
        
class RPC:
    port_name = '/eCubPerception/rpc:i'
