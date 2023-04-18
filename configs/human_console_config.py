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
            # 'rec_human_console': PyQueue(ip="localhost", port=50000, queue_name='rec_human_console', blocking=False),

            'human_console_visualizer': PyQueue(ip="localhost", port=50000, queue_name='human_console_visualizer', blocking=False),
        }

        out_queues = {
            'console_to_ar': PyQueue(ip="localhost", port=50000, queue_name='console_to_ar',
                                     write_format={"command": None})}
        auto_write = False
