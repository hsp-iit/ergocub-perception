from logging import INFO
from utils.concurrency.generic_node import GenericNode
from utils.concurrency.py_queue import PyQueue
from utils.concurrency.yarp_queue import YarpQueue
from utils.confort import BaseConfig


class Logging(BaseConfig):
    level = INFO


class Network(BaseConfig):
    node = GenericNode

    class Args:
        in_queues = {
            # 'depth': YarpQueue(remote_port_name='/depthCamera/depthImage:r', local_port_name='/Visualizer/depthImage:i',
            #                  data_type='depth', read_format='depth'),
            'rgb': YarpQueue(remote_port_name='/depthCamera/rgbImage:r', local_port_name='/Visualizer/rgbImage:i',
                             data_type='rgb', read_format='rgb', blocking=True),
            'visualizer': PyQueue(ip="localhost", port=50000, queue_name='visualizer', blocking=False),
        }
