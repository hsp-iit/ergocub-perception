import os
from logging import INFO

import numpy as np

from action_rec.ar.ar import ActionRecognizer
from utils.concurrency.generic_node import GenericNode
from utils.concurrency.generic_node_fps import GenericNodeFPS
from utils.concurrency.ipc_queue import IPCQueue
from utils.concurrency.py_queue import PyQueue
from utils.concurrency.utils.signals import Signals
from utils.confort import BaseConfig

input_type = "skeleton"  # rgb, skeleton or hybrid
seq_len = 8 if input_type != "skeleton" else 16


class Logging(BaseConfig):
    class Logger:
        class Params:
            level = INFO  # Minimum logging level or list of logging levels
            recurring = False

    debug = True
    # options: rgb depth mask 'fps center hands partial scene reconstruction transform
    keys = {'action': -1, 'human_distance': -1., 'focus': False,  # Used by rpc
            'bbox': None, 'face_bbox': None, 'pose': None, 'actions': None, 'edges': None, 'is_true': -1,  # Debugging
            'requires_focus': False, "requires_os": None}  # Debugging


class Network(BaseConfig):
    node = GenericNodeFPS

    class Args:
        in_queues = {
            'hpe_to_ar': PyQueue(ip="localhost", port=50000, queue_name='hpe_to_ar', blocking=True),
            'console_to_ar': PyQueue(ip="localhost", port=50000, queue_name='console_to_ar',
                                     read_format={"command": None}),
        }

        out_queues = {
            'human_console_visualizer': PyQueue(ip="localhost", port=50000, queue_name='human_console_visualizer',
                                                write_format={'fps_ar': Signals.NOT_OBSERVED, 'actions':
                                                    Signals.NOT_OBSERVED, 'is_true': Signals.NOT_OBSERVED,
                                                              'log': Signals.NOT_OBSERVED}),
            'visualizer': PyQueue(ip="localhost", port=50000, queue_name='visualizer',
                                  write_format={'fps_ar': Signals.NOT_OBSERVED, 'action': Signals.NOT_OBSERVED}),

            'ar_to_rpc': PyQueue(ip="localhost", port=50000, queue_name='ar_to_rpc', write_format={'action': -1}),
        }

        max_fps = 12


base_dir = os.path.join('action_rec', 'ar', 'weights', 'engines')


# TODO GO HERE TO CHANGE OPTION FOR TRAINING ACTION RECOGNITION (CHANGE IN FUTURE)
# from ISBFSAR.ar.utils.configuration import TRXTrainConfig


class AR(BaseConfig):
    model = ActionRecognizer

    class Args:
        input_type = input_type  # skeleton or rgb
        device = 'cuda'
        support_set_path = os.path.join("action_rec", "ar", "assets", "saved")

        if input_type == "rgb":
            final_ckpt_path = os.path.join(base_dir, "../action_rec/ar", "weights", "raws", "rgb", "5-w-5-s.pth")
        elif input_type == "skeleton":
            final_ckpt_path = os.path.join("action_rec", "ar", "weights", "raws", "skeleton", "5-w-5-s.pth")
        elif input_type == "hybrid":
            final_ckpt_path = os.path.join(base_dir, "../action_rec/ar", "weights", "raws", "hybrid",
                                           "1714_truncated_resnet.pth")

        seq_len = seq_len
        way = 5
        n_joints = 30
        shot = 5

    class Main:
        input_type = 'skeleton'
        window_size = 4
        acquisition_time = 3
        consistency_window_length = 4
        os_score_thr = 0.5
