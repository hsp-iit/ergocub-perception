from loguru import logger

from utils.concurrency.utils.signals import Signals
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
        if rgb in Signals:
            return {}

        bbox = self.hd_model.estimate(rgb)["bbox"]

        output = {'fps_hd': self.fps()}

        logger.info("HUMAN detected", recurring=True)

        if bbox is None:
            return output

        output['rgb'] = rgb
        output['bbox'] = bbox

        return output


if __name__ == '__main__':
    hd = HumanDetection()
    hd.run()
