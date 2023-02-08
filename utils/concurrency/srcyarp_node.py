from abc import abstractmethod

import numpy as np
import yarp

from utils.concurrency.yarpsys_node import YarpSysNode
from loguru import logger

from utils.input import RealSense


class SrcYarpNode(YarpSysNode):

    def _startup(self):
        logger.info('Starting up...')
        self.startup()
        logger.success('Start up complete.')

    @abstractmethod
    def loop(self) -> dict:
        pass

    # @_exception_handler
    # @logger.catch(reraise=True)
    def run(self) -> None:
        self._startup()

        # This is to wait for the first message even in non-blocking mode
        while True:
            data = self.loop()
            self._send_all(data)
            # self._send_all(data, self.blocking)

    def _send_all(self, data):
        for dest in data:

            for k, v in data[dest].items():
                bottle = yarp.Bottle()
                bottle.clear()

                if isinstance(v, np.ndarray) and (v.ndim == 3):
                    # v = v.astype(np.float32)
                    # yarp_data = yarp.ImageRgb()
                    # yarp_data.setExternal(v, v.shape[1], v.shape[0])
                    yarp_image = yarp.ImageRgb()
                    yarp_image.resize(v.shape[1], v.shape[0])
                    yarp_image.setExternal(v.data, v.shape[1], v.shape[0])
                elif isinstance(v, np.ndarray) and (v.ndim == 2):
                    # old_pc = RealSense.depth_pointcloud(v)
                    v = v / 1000
                    v = v.astype(np.float32)

                    # v = (v*1000).astype(np.uint16)
                    # new_pc = RealSense.depth_pointcloud(v)

                    # yarp_data = yarp.ImageMono16()
                    # yarp_data.setExternal(v, v.shape[1], v.shape[0])
                    yarp_image = yarp.ImageFloat()
                    yarp_image.resize(v.shape[1], v.shape[0])
                    yarp_image.setExternal(v.data, v.shape[1], v.shape[0])
                else:
                    raise ValueError(f"Unsupported output type for key {dest}/{k}")

                self._out_queues[f'/{dest}/{k}'].write(yarp_image)

        return data



