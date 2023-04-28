import sys
from pathlib import Path
import numpy as np
from loguru import logger
import cv2
from grasping.utils.misc import pose_to_matrix
from utils.concurrency.utils.signals import Signals
sys.path.insert(0, Path(__file__).parent.parent.as_posix())
from utils.logging import setup_logger
import tensorrt as trt
# https://github.com/NVIDIA/TensorRT/issues/1945
import torch
import pycuda.autoinit

from configs.focus_config import FOCUS, Network, Logging

setup_logger(**Logging.Logger.Params.to_dict())


class Focus(Network.node):
    def __init__(self):
        super().__init__(**Network.Args.to_dict())
        self.focus_model = None
        # self.timer = Timer(window=10)

    def startup(self):
        self.focus_model = FOCUS.model(**FOCUS.Args.to_dict())

    def loop(self, data):

        # output = copy.deepcopy(data)
        output = {}
        # self.timer.start()
        logger.info("Read camera input", recurring=True)

        rgb = data['rgb']

        if rgb in Signals:
            return {}

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        ret = self.focus_model.estimate(rgb)

        if ret is not None:
            foc, face = ret
            output["focus"] = foc
            output["face_bbox"] = face.bbox.reshape(-1)
            output["fps_focus"] = self.fps()

            camera_pose = data['camera_pose']
            if camera_pose not in Signals:
                camera_pose = pose_to_matrix(camera_pose)
                face_position = np.array(face.head_position)[None]
                face_position = np.concatenate([face_position, np.array([[1]])], axis=1).T
                point = camera_pose @ face_position
                output["face_point"] = point.reshape(-1)[:3]
            else:
                output["face_point"] = np.array([0.3, 0.3, 0.3])

        logger.info("FOCUS detected", recurring=True)

        return output


if __name__ == '__main__':
    focus = Focus()
    focus.run()
