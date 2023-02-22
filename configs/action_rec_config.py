import os
from logging import INFO
from action_rec.ar.ar import ActionRecognizer
from action_rec.focus.gaze_estimation.focus import FocusDetector
from action_rec.hpe.hpe import HumanPoseEstimator
from utils.concurrency import YarpSysNode  # TODO THIS CAUSES TROUBLES
from utils.concurrency.yarppy_node import YarpPyNode
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
        fps = seq_len # Fps car for action recognition


class Network(BaseConfig):
    node = YarpPyNode

    class Args:
        ip = "localhost"
        port = 50000

        in_config = {'depthCamera': ['rgbImage', 'depthImage']}

        out_config = {'visualizer': {k: None for k in
                                     ['fps_ar', 'human_distance', 'focus', 'pose', 'bbox', 'face_bbox', 'actions',
                                      'is_true', 'requires_focus', 'edges', 'log', 'requires_os']},
                      'action_recognition_rpc': {'action': -1, 'human_distance': -1., 'focus': False,
                                                 "filtered_action": -1}}
        # make the output queue blocking (can be used to put a breakpoint in the sink and debug the process output)
        blocking = False


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

        fx = 384.025146484375
        fy = 384.025146484375
        ppx = 319.09661865234375
        ppy = 237.75723266601562
        width = 640
        height = 480

        necessary_percentage_visible_joints = 0.7


# TODO GO HERE TO CHANGE OPTIONS FOR FOCUS (CHANGE IN FUTURE)


class FOCUS(BaseConfig):
    model = FocusDetector

    class Args:
        area_thr = 0.03  # head bounding box must be over this value to be close
        close_thr = -0.95  # When close, z value over this thr is considered focus
        dist_thr = 0.3  # when distant, roll under this thr is considered focus
        foc_rot_thr = 0.7  # when close, roll above this thr is considered not focus
        patience = 3  # result is based on the majority of previous observations
        sample_params_path = os.path.join(base_dir, "focus", "gaze_estimation", "assets",
                                          "sample_params.yaml")


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
