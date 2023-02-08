from logging import INFO

from utils.concurrency import PyPyNode
from utils.confort import BaseConfig


class Logging(BaseConfig):
    level = INFO


class Network(BaseConfig):
    node = PyPyNode

    class Args:
        ip = 'localhost'
        port = 50000
        in_queue = 'visualizer'
        out_queues = []

