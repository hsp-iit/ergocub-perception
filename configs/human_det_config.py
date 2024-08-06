import os
from logging import INFO
from action_rec.hd.hd import HumanDetector
from utils.concurrency.generic_node import GenericNode
from utils.concurrency.generic_node_fps import GenericNodeFPS
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
    # options: rgb depth mask 'fps center hands partial scene reconstruction transform
    keys = {'bbox': None, 'rgb': None}  # Debugging


class Network(BaseConfig):
    node = GenericNodeFPS

    class Args:
        in_queues = {
            'rgb': YarpQueue(remote_port_name='/depthCamera/rgbImage:r', local_port_name='/HumanDetection/rgbImage:i',
                             data_type='rgb', read_format='rgb', read_default=Signals.USE_LATEST, blocking=False),
            'rec_hd': PyQueue(ip="localhost", port=50000, queue_name='rec_hd', blocking=False)
        }

        out_queues = {
            'hd_to_hpe': PyQueue(ip="localhost", port=50000, queue_name='hd_to_hpe',
                                 write_format={'rgb': Signals.NOT_OBSERVED, 'bbox': Signals.NOT_OBSERVED}),
            'visualizer': PyQueue(ip="localhost", port=50000, queue_name='visualizer',
                                  write_format={'bbox': Signals.NOT_OBSERVED, 'fps_hd': Signals.NOT_OBSERVED}),
        }

        max_fps = 40


class HD(BaseConfig):
    model = HumanDetector

    class Args:
        yolo_thresh = 0.3
        nms_thresh = 0.7
        yolo_engine_path = os.path.join('action_rec', 'hd', 'weights', 'engines', 'docker', 'yolo.engine')
