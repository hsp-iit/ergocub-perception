import os
from logging import INFO
from action_rec.hpe.hpe import HumanPoseEstimator
from utils.concurrency.generic_node_fps import GenericNodeFPS
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


class MAIN(BaseConfig):
    class Args:
        input_type = input_type  # rgb or skeleton
        cam_width = 640
        cam_height = 480
        window_size = seq_len
        skeleton_scale = 2200.
        acquisition_time = 3  # Seconds
        fps = 100  # /2.5 # Fps car for action recognition
        consistency_window_length = 8  # 12
        os_score_thr = 0.5


class Network(BaseConfig):
    node = GenericNodeFPS

    class Args:
        in_queues = {
            'hd_to_hpe': PyQueue(ip="localhost", port=50000, queue_name='hd_to_hpe', blocking=True)
        }

        out_queues = {
            'visualizer': PyQueue(ip="localhost", port=50000, queue_name='visualizer',
                                  write_format={'fps_hpe': Signals.NOT_OBSERVED, 'human_distance': Signals.NOT_OBSERVED,
                                                'pose': Signals.NOT_OBSERVED, 'edges': Signals.NOT_OBSERVED}),
            'human_console_visualizer': PyQueue(ip="localhost", port=50000, queue_name='human_console_visualizer',
                                                write_format={'pose': Signals.NOT_OBSERVED}),  # TO ADD ACTION
            'hpe_to_ar': PyQueue(ip="localhost", port=50000, queue_name='hpe_to_ar', blocking=False,
                                 write_format={'pose': Signals.NOT_OBSERVED,
                                               'human_distance': Signals.NOT_OBSERVED})}

    max_fps = 18


base_dir = os.path.join('action_rec', 'hpe', 'weights', 'engines', 'docker')


class HPE(BaseConfig):
    model = HumanPoseEstimator

    class Args:
        image_transformation_engine_path = os.path.join(base_dir, 'image_transformation1.engine')
        bbone_engine_path = os.path.join(base_dir, 'bbone1.engine')
        heads_engine_path = os.path.join(base_dir, 'heads1.engine')
        expand_joints_path = os.path.join('action_rec', 'hpe', 'assets', '32_to_122.npy')
        skeleton_types_path = os.path.join('action_rec', 'hpe', 'assets', 'skeleton_types.pkl')
        skeleton = 'smpl+head_30'

        # D435i (got from andrea)
        fx = 612.7910766601562
        fy = 611.8779296875
        ppx = 321.7364196777344
        ppy = 245.0658416748047

        width = 640
        height = 480

        necessary_percentage_visible_joints = 0.3
