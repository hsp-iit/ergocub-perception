from logging import INFO

import numpy as np

from utils.concurrency.yarppy_node import YarpPyNode
from utils.confort import BaseConfig


class Logging(BaseConfig):
    class Logger:
        class Params:
            level = INFO  # Minimum logging level or list of logging levels
            recurring = False

    debug = True
    # options: rgb depth mask 'fps center hands partial scene reconstruction transform


class Network(BaseConfig):

    node = YarpPyNode

    class Args:
        ip = "localhost"
        port = 50000

        in_config = {'realsense': ['rgb', 'depth']}
        out_config = {'visualizer': {k: None for k in ['rgb', 'depth']}}
        # make the output queue blocking (can be used to put a breakpoint in the sink and debug the process output)
        blocking = False
