import os
from logging import INFO
from action_rec.ar.ar import ActionRecognizer
from utils.concurrency.generic_node import GenericNode
from utils.concurrency.generic_node_fps import GenericNodeFPS
from utils.concurrency.ipc_queue import IPCQueue
from utils.concurrency.py_queue import PyQueue
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
            'pose': PyQueue(ip="localhost", port=50000, queue_name='pose', blocking=True,
                            # read_format={"pose": None}
                            ),
            'human_console_commands': PyQueue(ip="localhost", port=50000, queue_name='human_console_commands',
                                              read_format={"train": None, "remove": None, "debug": None})
        }

        out_queues = {
            'visualizer': PyQueue(ip="localhost", port=50000, queue_name='visualizer',
                                  write_format={'fps_ar': None, 'actions': None, 'is_true': None,
                                                'requires_focus': None, 'log': None,
                                                'requires_os': None, 'action': None}),

            'rpc': IPCQueue(ipc_key=5678, write_format={'action': -1})}


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
