from loguru import logger

from utils.concurrency.utils.signals import Signals
from utils.logging import setup_logger
import tensorrt as trt
# https://github.com/NVIDIA/TensorRT/issues/1945
import torch
import pycuda.autoinit
from configs.human_pose_estimation_config import HPE, Network, Logging
import yarp
import math
setup_logger(**Logging.Logger.Params.to_dict())
yarp.Network.init()

class HumanPoseEstimation(Network.node):
    def __init__(self):
        super().__init__(**Network.Args.to_dict())
        self.hpe_model = None
        self.direct_human_data_port = yarp.BufferedPortBottle()

    def startup(self):
        self.hpe_model = HPE.model(**HPE.Args.to_dict())
        self.direct_human_data_port.open("/humanDataPort")

    def loop(self, data):
        logger.info("Estimating human pose...", recurring=True)

        rgb = data['rgb']
        bbox = data['bbox']
        yarp_read_time = data['yarp_read_time']

        if rgb in Signals or bbox in Signals:
            return {}

        ret = self.hpe_model.estimate(rgb, bbox, yarp_read_time)
        logger.info("Human pose estimated!", recurring=True)
        bottle = self.direct_human_data_port.prepare()
        bottle.clear()
        for i in range(3):
            bottle.addFloat64(ret["human_position"][i])
        for i in range(4): 
            bottle.addFloat64(ret["human_occupancy"][i])
        split_time = math.modf(yarp_read_time)
        bottle.addInt64(int(split_time[1]))
        bottle.addInt64(int(split_time[0]*1e9))
        self.direct_human_data_port.write()

        ret["fps_hpe"] = self.fps()
        ret['yarp_read_time'] = yarp_read_time

        return ret


if __name__ == '__main__':
    hd = HumanPoseEstimation()
    hd.run()
