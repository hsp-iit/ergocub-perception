import copy
import sys
from pathlib import Path
from loguru import logger

sys.path.insert(0, Path(__file__).parent.parent.as_posix())

from configs.source_config import Logging, Network, Input
from utils.logging import setup_logger

setup_logger(level=Logging.level)


@logger.catch(reraise=True)
class Source(Network.node):
    def __init__(self):
        super().__init__(**Network.Args.to_dict())
        self.camera = None
        

    def startup(self):
        self.camera = Input.camera(**Input.Params.to_dict())

    def loop(self, _):

        while True:
            try:
                rgb, depth = self.camera.read()
                data = {'rgb': copy.deepcopy(rgb), 'depth': copy.deepcopy(depth)}

                return data

            except RuntimeError as e:
                logger.error("Realsense: frame didn't arrive")
                self.camera = Input.camera(**Input.Params.to_dict())


if __name__ == '__main__':
    source = Source()
    source.run()
