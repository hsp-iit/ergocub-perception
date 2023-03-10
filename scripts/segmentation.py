import copy
import sys
import time
from pathlib import Path

import cv2
from loguru import logger

sys.path.insert(0, Path(__file__).parent.parent.as_posix())

from grasping.utils.avg_timer import Timer
from utils.logging import setup_logger
import tensorrt as trt
# https://github.com/NVIDIA/TensorRT/issues/1945
import torch
import pycuda.autoinit

from configs.segmentation_config import Segmentator, Network, Logging

setup_logger(**Logging.Logger.Params.to_dict())


class Segmentation(Network.node):
    def __init__(self):
        super().__init__(**Network.Args.to_dict())
        self.seg_model = None
        self.timer = Timer(window=10)

    def startup(self):
        self.seg_model = Segmentator.model(**Segmentator.Args.to_dict())

    def loop(self, data):

        output = copy.deepcopy(data)

        self.timer.start()
        logger.info("Read camera input", recurring=True)

        rgb = data['rgb']

        # Segment the rgb and extract the object depth
        # start = time.perf_counter()
        mask = self.seg_model(rgb)
        # print(1 / (time.perf_counter() - start))
        mask = cv2.resize(mask, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)

        logger.info("RGB segmented", recurring=True)

        output['mask'] = mask

        return output


if __name__ == '__main__':
    grasping = Segmentation()
    grasping.run()
