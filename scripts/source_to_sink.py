import copy
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from loguru import logger

sys.path.insert(0, Path(__file__).parent.parent.as_posix())


from utils.logging import setup_logger

from configs.s2s_config import Network, Logging

setup_logger(**Logging.Logger.Params.to_dict())


class SourceToSink(Network.node):
    def __init__(self):
        super().__init__(**Network.Args.to_dict())

    def loop(self, data):
        output = {}
        # Input

        logger.info("Read camera input", recurring=True)

        rgb = data['rgb']
        depth = data['depth']

        output['rgb'] = rgb
        output['depth'] = depth

        return output


if __name__ == '__main__':
    s2s = SourceToSink()
    s2s.run()
