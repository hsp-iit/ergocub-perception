import os
from logging import INFO
from action_rec.focus.gaze_estimation.focus import FocusDetector
from utils.concurrency.generic_node_fps import GenericNodeFPS
from utils.concurrency.py_queue import PyQueue
from utils.concurrency.utils.signals import Signals
from utils.concurrency.yarp_queue import YarpQueue
from utils.confort import BaseConfig
import numpy as np


class Logging(BaseConfig):
    class Logger:
        class Params:
            level = INFO  # Minimum logging level or list of logging levels
            recurring = False

    debug = True
    # options: rgb depth mask 'fps center hands partial scene reconstruction transform


class Network(BaseConfig):
    node = GenericNodeFPS

    class Args:
        in_queues = {
            'rgb': YarpQueue(remote_port_name='/depthCamera/rgbImage:r', local_port_name='/Focus/rgbImage:i',
                             data_type='rgb', read_format='rgb', read_default=Signals.USE_LATEST, blocking=False),
            'rec_focus': PyQueue(ip="localhost", port=50000, queue_name='rec_focus', blocking=False),
            'from_pose_streamer': YarpQueue(remote_port_name='/realsense-holder-publisher/pose:o',
                                            local_port_name='/VisualPerception/FocusDetection/camera_pose:i',
                                            data_type='list', read_format='camera_pose',
                                            read_default=Signals.MISSING_VALUE, blocking=True),
        }

        out_queues = {
            'visualizer': PyQueue(ip="localhost", port=50000, queue_name='visualizer',
                                  write_format={'focus': Signals.NOT_OBSERVED, 'face_bbox': Signals.NOT_OBSERVED,
                                                'fps_focus': Signals.NOT_OBSERVED}),
            'focus_to_rpc': PyQueue(ip="localhost", port=50000, queue_name='focus_to_rpc', write_format={'focus': False, 'face_point': np.full(3, -1.)}),

        }

        max_fps = 30


class FOCUS(BaseConfig):
    model = FocusDetector

    class Args:
        area_thr = 0.03  # head bounding box must be over this value to be close
        close_thr = -0.95  # When close, z value over this thr is considered focus
        dist_thr = 0.3  # when distant, roll under this thr is considered focus
        foc_rot_thr = 0.7  # when close, roll above this thr is considered not focus
        patience = 3  # result is based on the majority of previous observations
        sample_params_path = os.path.join("action_rec", "focus", "gaze_estimation", "assets",
                                          "sample_params.yaml")
