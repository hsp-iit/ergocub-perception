from logging import INFO

import numpy as np

from grasping.segmentation.fcn.fcn_segmentator_trt import FcnSegmentatorTRT
from utils.concurrency.generic_node import GenericNode
from utils.concurrency.py_queue import PyQueue
from utils.concurrency.yarp_queue import YarpQueue
from utils.concurrency.yarppy_node import YarpPyNode
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

    class Args:
        in_queues = {
            # in_port_name, out_port_name, data_type, out_name
            'rgb': YarpQueue(remote_port_name='/depthCamera/rgbImage:r', local_port_name='/Segmentation/rgbImage:i',
                             data_type='rgb', read_format='rgb'),
            'depth': YarpQueue(remote_port_name='/depthCamera/depthImage:r', local_port_name='/Segmentation/depthImage:i',
                             data_type='depth', read_format='depth')
        }

        out_queues = {
            'visualizer': PyQueue(ip="localhost", port=50000, queue_name='visualizer',write_format={'mask': None}),
            'shape_completion': PyQueue(ip="localhost", port=50000, queue_name='seg_to_sc',
                                        write_format={'mask': None, 'depth': None, 'rgb': None})
        }


class Segmentator(BaseConfig):
    model = FcnSegmentatorTRT

    class Args:
        engine_path = './grasping/segmentation/fcn/trt/assets/seg_fp16_docker.engine'
