import os
from logging import INFO
from action_rec.ar.ar import ActionRecognizer
from action_rec.hpe.hpe import HumanPoseEstimator
from utils.concurrency.generic_node import GenericNode
from utils.concurrency.ipc_queue import IPCQueue
from utils.concurrency.py_queue import PyQueue
from utils.concurrency.utils.signals import Signals
from utils.concurrency.yarp_queue import YarpQueue
from utils.confort import BaseConfig
import platform


input_type = "skeleton"  # rgb, skeleton or hybrid
docker = os.environ.get('AM_I_IN_A_DOCKER_CONTAINER', False)
seq_len = 8 if input_type != "skeleton" else 16
ubuntu = platform.system() == "Linux"
base_dir = "action_rec"
engine_dir = "engines" if not docker else os.path.join("engines", "docker")


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
    node = GenericNode

    class Args:
        in_queues = {
            # in_port_name, out_port_name, data_type, out_name
            'rgb': YarpQueue(remote_port_name='/depthCamera/rgbImage:r', local_port_name='/ActionRecognition/rgbImage:i',
                             data_type='rgb', read_format='rgb', read_default=Signals.USE_LATEST, blocking=False),
            'action_rec_in': PyQueue(ip="localhost", port=50000, queue_name='action_rec_in', blocking=False),
                             data_type='rgb', read_format='rgb'),
            'human_console_commands': PyQueue(ip="localhost", port=50000, queue_name='human_console_commands',
                                              read_format={"train": None, "remove": None, "debug": None})
        }

        out_queues = {
            'visualizer': PyQueue(ip="localhost", port=50000, queue_name='visualizer',
                                  write_format={'fps_ar': None, 'human_distance': None, 'pose': None,
                                                'bbox': None, 'actions': None, 'is_true': None,
                                                'requires_focus': None, 'edges': None, 'log': None,
                                                'requires_os': None, 'action': None}),

            'rpc': IPCQueue(ipc_key=1234, write_format={'action': -1, 'human_distance': -1})}


class HPE(BaseConfig):
    model = HumanPoseEstimator

    class Args:
        yolo_engine_path = os.path.join(base_dir, 'hpe', 'weights', engine_dir, 'yolo.engine')
        image_transformation_engine_path = os.path.join(base_dir, 'hpe', 'weights', engine_dir,
                                                        'image_transformation1.engine')
        bbone_engine_path = os.path.join(base_dir, 'hpe', 'weights', engine_dir, 'bbone1.engine')
        heads_engine_path = os.path.join(base_dir, 'hpe', 'weights', engine_dir, 'heads1.engine')
        expand_joints_path = os.path.join(base_dir, "hpe", 'assets', '32_to_122.npy')
        skeleton_types_path = os.path.join(base_dir, "hpe", "assets", "skeleton_types.pkl")
        skeleton = 'smpl+head_30'
        yolo_thresh = 0.3
        nms_thresh = 0.7
        num_aug = 0  # if zero, disables test time augmentation
        just_box = input_type == "rgb"

        # D435i (got from andrea)
        fx = 612.7910766601562
        fy = 611.8779296875
        ppx = 321.7364196777344
        ppy = 245.0658416748047

        width = 640
        height = 480

        necessary_percentage_visible_joints = 0.3


# TODO GO HERE TO CHANGE OPTION FOR TRAINING ACTION RECOGNITION (CHANGE IN FUTURE)
# from ISBFSAR.ar.utils.configuration import TRXTrainConfig


class AR(BaseConfig):
    model = ActionRecognizer

    class Args:
        input_type = input_type  # skeleton or rgb
        device = 'cuda'
        support_set_path = os.path.join(base_dir, "ar", "assets", "saved")

        if input_type == "rgb":
            final_ckpt_path = os.path.join(base_dir, "ar", "weights", "raws", "rgb", "5-w-5-s.pth")
        elif input_type == "skeleton":
            final_ckpt_path = os.path.join(base_dir, "ar", "weights", "raws", "skeleton", "5-w-5-s.pth")
        elif input_type == "hybrid":
            final_ckpt_path = os.path.join(base_dir, "ar", "weights", "raws", "hybrid",
                                           "1714_truncated_resnet.pth")

        seq_len = seq_len
        way = 5
        n_joints = 30
        shot = 5
