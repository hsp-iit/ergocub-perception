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
            'rgb': YarpQueue(remote_port_name='/depthCamera/rgbImage:r', local_port_name='/HumanConsole/rgbImage:i',
                             data_type='rgb', read_format='rgb', blocking=True),
            'rec_human_console': PyQueue(ip="localhost", port=50000, queue_name='rec_human_console', blocking=False),

            'human_console_visualizer': PyQueue(ip="localhost", port=50000, queue_name='visualizer', blocking=True),
        }

        out_queues = {
            'console_to_ar': PyQueue(ip="localhost", port=50000, queue_name='console_to_ar',
                                     write_format={"train": None, "remove": None, "debug": None, "load": None,
                                                   "save": None, "remove_action": None, "remove_example": None})}
        auto_write = False
