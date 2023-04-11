from logging import INFO

import numpy as np

from grasping.segmentation.fcn.fcn_segmentator_trt import FcnSegmentatorTRT
from utils.concurrency.generic_node import GenericNode
from utils.concurrency.ipc_queue import IPCQueue
from utils.concurrency.py_queue import PyQueue
from utils.concurrency.utils.signals import Signals
from utils.concurrency.yarp_queue import YarpQueue
from utils.confort import BaseConfig


class Logging(BaseConfig):
    class Logger:
        class Params:
            level = INFO  # Minimum logging level or list of logging levels
            recurring = False

    debug = True


class Network(BaseConfig):
    node = GenericNode

    class Args:
        in_queues = {
            # in_port_name, out_port_name, data_type, out_name
            'rgb': YarpQueue(remote_port_name='/depthCamera/rgbImage:r', local_port_name='/Segmentation/rgbImage:i',
                             data_type='rgb', read_format='rgb', read_default=Signals.USE_LATEST, blocking=False),
            'depth': YarpQueue(remote_port_name='/depthCamera/depthImage:r',
                               local_port_name='/Segmentation/depthImage:i',
                               data_type='depth', read_format='depth', read_default=Signals.USE_LATEST, blocking=False),
            'segmentation_in': PyQueue(ip="localhost", port=50000, queue_name='segmentation_in', blocking=False)
        }

        out_queues = {
            'to_visualizer': PyQueue(ip="localhost", port=50000, queue_name='visualizer',
                                     write_format={'mask': Signals.NOT_OBSERVED}),
            'to_shape_completion': PyQueue(ip="localhost", port=50000, queue_name='seg_to_sc',
                                           write_format={k: Signals.NOT_OBSERVED for k in ['segmented_pc',
                                                                                           'obj_distance',
                                                                                           'point']}),  # TODO MAKE IT BETTER
            'to_3d_viz': PyQueue(ip="localhost", port=50000, queue_name='3d_visualizer',
                                 write_format={k: Signals.NOT_OBSERVED for k in
                                               ['point']}),
            # 'to_gaze_control': PyQueue(ip="localhost", port=50000, queue_name='seg_to_gc',
            #                            write_format={k: Signals.NOT_OBSERVED for k in
            #                                          ['point']}),
            # 'rpc': IPCQueue(ipc_key=1234, write_format={'obj_distance': -1})
        }

        auto_write = False


class Segmentator(BaseConfig):
    model = FcnSegmentatorTRT

    class Args:
        engine_path = './grasping/segmentation/fcn/trt/assets/seg_fp16_docker.engine'
