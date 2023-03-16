from logging import INFO

from utils.concurrency.generic_node import GenericNode
from utils.concurrency.py_queue import PyQueue
from utils.concurrency.utils.signals import Signals
from utils.concurrency.yarp_queue import YarpQueue
from utils.confort import BaseConfig


class Logging(BaseConfig):
    class Logger:
        class Args:
            level = INFO  # Minimum logging level or list of logging levels
            recurring = False

    debug = True


class Network(BaseConfig):
    node = GenericNode

    class Args:
        in_queues = {
            'rgb_in': YarpQueue(remote_port_name='/depthCamera/rgbImage:r', local_port_name='/Replay/rgbImage:i',
                                data_type='rgb', read_format='rgb', blocking=True),
            'depth_in': YarpQueue(remote_port_name='/depthCamera/depthImage:r', local_port_name='/Replay/depthImage:i',
                                  data_type='depth', read_format='depth', blocking=True)
        }

        out_queues = {
            'rec_viz': PyQueue(ip="localhost", port=50000, queue_name='rec_viz',
                                    write_format={k: Signals.NOT_OBSERVED for k in ['rgb', 'depth']}),
            'segmentation_in': PyQueue(ip="localhost", port=50000, queue_name='segmentation_in',
                               write_format={k: Signals.NOT_OBSERVED for k in ['rgb', 'depth']}),
            'action_rec_in': PyQueue(ip="localhost", port=50000, queue_name='action_rec_in',
                                       write_format={k: Signals.NOT_OBSERVED for k in ['rgb', 'depth']}),
            'focus_in': PyQueue(ip="localhost", port=50000, queue_name='focus_in',
                                       write_format={k: Signals.NOT_OBSERVED for k in ['rgb', 'depth']})
        }

        auto_read = False
