from logging import INFO

from utils.concurrency.generic_node import GenericNode
from utils.concurrency.py_queue import PyQueue
from utils.concurrency.utils.signals import Signals
from utils.concurrency.yarp_queue import YarpQueue
from utils.confort import BaseConfig


class Logging(BaseConfig):
    level = INFO


class Network(BaseConfig):
    node = GenericNode

    class Args:
        in_queues = {
            # in_port_name, out_port_name, data_type, out_name
            'grasping': PyQueue(ip="localhost", port=50000, queue_name='3d_visualizer',
                                read_format={k: Signals.MISSING_VALUE for k in
                                             ['reconstruction', 'transform', 'scene',
                                              'hands', 'vertices']},
                                blocking=False),
            'rgb': YarpQueue(remote_port_name='/depthCamera/rgbImage:r', local_port_name='/Visualizer3D/rgbImage:i',
                             data_type='rgb', read_format='rgb'),
            'depth': YarpQueue(remote_port_name='/depthCamera/depthImage:r',
                               local_port_name='/Visualizer3D/depthImage:i',
                               data_type='depth', read_format='depth')
        }
