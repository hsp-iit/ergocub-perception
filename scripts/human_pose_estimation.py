from loguru import logger
from utils.logging import setup_logger
import tensorrt as trt
# https://github.com/NVIDIA/TensorRT/issues/1945
import torch
import pycuda.autoinit
from configs.human_pose_estimation_config import HPE, Network, Logging

setup_logger(**Logging.Logger.Params.to_dict())


class HumanPoseEstimation(Network.node):
    def __init__(self):
        super().__init__(**Network.Args.to_dict())
        self.hpe_model = None

    def startup(self):
        self.hpe_model = HPE.model(**HPE.Args.to_dict())

    def loop(self, data):
        logger.info("Estimating human pose...", recurring=True)

        rgb = data['rgb']
        bbox = data['bbox']
        if rgb is None or bbox is None:
            return {}

        ret = self.hpe_model.estimate(rgb, bbox)

        logger.info("Human pose estimated!", recurring=True)

        ret["fps_hpe"] = self.fps()

        return ret


if __name__ == '__main__':
    hd = HumanPoseEstimation()
    hd.run()
