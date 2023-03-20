from loguru import logger
from utils.logging import setup_logger
import tensorrt as trt
# https://github.com/NVIDIA/TensorRT/issues/1945
import torch
import pycuda.autoinit
from configs.human_det_config import HD, Network, Logging

setup_logger(**Logging.Logger.Params.to_dict())


class HumanDetection(Network.node):
    def __init__(self):
        super().__init__(**Network.Args.to_dict())
        self.hd_model = None

    def startup(self):
        self.hd_model = HD.model(**HD.Args.to_dict())

    def loop(self, data):
        logger.info("Read camera input", recurring=True)

        rgb = data['rgb']
        bbox = self.hd_model.estimate(rgb)["bbox"]

        logger.info("FOCUS detected", recurring=True)

        return {'rgb': rgb, 'bbox': bbox, 'fps_hd': self.fps()}


if __name__ == '__main__':
    hd = HumanDetection()
    hd.run()
