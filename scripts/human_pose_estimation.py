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
import os
setup_logger(**Logging.Logger.Params.to_dict())
yarp.Network.init()

class HumanPoseEstimation(Network.node):
    def __init__(self):
        super().__init__(**Network.Args.to_dict())
        self.hpe_model = None
        self.direct_human_data_port = yarp.BufferedPortBottle()
        self.previous_extreme = [None, None]
        self.previous_centre = [None, None]
        self.alpha_x = float(os.environ['HUMAN_X_FILTER_ALPHA'])
        self.alpha_y = float(os.environ['HUMAN_Y_FILTER_ALPHA'])


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
        human_position = [ret["human_position"][0], ret["human_position"][1],ret["human_position"][2]]
        human_occupancy = [ret["human_occupancy"][0],ret["human_occupancy"][1],ret["human_occupancy"][2],ret["human_occupancy"][3]]
        if self.previous_centre[0] is not None:
            human_position[0] = self.alpha_x*human_position[0]+ (1-self.alpha_x)*self.previous_centre[0]
            human_position[1] = self.alpha_y*human_position[1] + (1-self.alpha_y)*self.previous_centre[1]
            human_occupancy[0] = self.alpha_y*human_occupancy[0] + (1-self.alpha_y)*self.previous_extreme[0]
            human_occupancy[1] = self.alpha_y*human_occupancy[1] + (1-self.alpha_y)*self.previous_extreme[1]
        

        bottle = self.direct_human_data_port.prepare()
        bottle.clear()
        for i in range(3):
            bottle.addFloat64(human_position[i])
        for i in range(4): 
            bottle.addFloat64(human_occupancy[i])
        split_time = math.modf(yarp_read_time)
        bottle.addInt64(int(split_time[1]))
        bottle.addInt64(int(split_time[0]*1e9))
        self.direct_human_data_port.write()
        
        self.previous_extreme = human_occupancy[:2] 
        self.previous_centre = human_position[:2]
        ret["fps_hpe"] = self.fps()
        ret['yarp_read_time'] = yarp_read_time

        return ret


if __name__ == '__main__':
    hd = HumanPoseEstimation()
    hd.run()
