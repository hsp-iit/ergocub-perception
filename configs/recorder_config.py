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
            'rgb_in': YarpQueue(remote_port_name='/depthCamera/rgbImage:r', local_port_name='/Recorder/rgbImage:i',
                                 data_type='rgb', read_format='rgb', blocking=False),
            'depth_in': YarpQueue(remote_port_name='/depthCamera/depthImage:r', local_port_name='/Recorder/depthImage:i',
                                  data_type='depth', read_format='depth', blocking=False)
        }

        out_queues = {
            'rec_viz': PyQueue(ip="localhost", port=50000, queue_name='rec_viz',
                               write_format={k: Signals.NOT_OBSERVED for k in ['rgb', 'depth']}),
            'segmentation_in': PyQueue(ip="localhost", port=50000, queue_name='segmentation_in',
                                       write_format={k: Signals.NOT_OBSERVED for k in ['rgb', 'depth']}),
            'action_rec_in': PyQueue(ip="localhost", port=50000, queue_name='action_rec_in',
                                     write_format={k: Signals.NOT_OBSERVED for k in ['rgb', 'depth']}),
            'focus_in': PyQueue(ip="localhost", port=50000, queue_name='focus_in',
                                write_format={k: Signals.NOT_OBSERVED for k in ['rgb', 'depth']}),
            'od3d_viz_in': PyQueue(ip="localhost", port=50000, queue_name='rec_od3dviz',
                                   write_format={k: Signals.NOT_OBSERVED for k in ['rgb', 'depth']}),
            'rec_focus': PyQueue(ip="localhost", port=50000, queue_name='rec_focus',
                                 write_format={k: Signals.NOT_OBSERVED for k in ['rgb', 'depth']}),
            'rec_hd': PyQueue(ip="localhost", port=50000, queue_name='rec_hd',
                              write_format={k: Signals.NOT_OBSERVED for k in ['rgb', 'depth']}),
            'rec_human_console': PyQueue(ip="localhost", port=50000, queue_name='rec_human_console',
                                         write_format={k: Signals.NOT_OBSERVED for k in ['rgb', 'depth']})

        }
        # out_queues = {
        #     'rgb_out': YarpQueue(local_port_name='/Recorder/rgbImage:r',
        #                          data_type='rgb', read_format='rgb', blocking=False),
        #     'depth_out': YarpQueue(local_port_name='/Recorder/depthImage:r',
        #                          data_type='depth', read_format='depth', blocking=False),
        # }

        auto_read = False
        auto_write = False
