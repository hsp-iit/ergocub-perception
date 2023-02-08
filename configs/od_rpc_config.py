from utils.concurrency.pysys_node import PySysNode
from utils.confort import BaseConfig


class Network(BaseConfig):
    node = PySysNode

    class Args:
        ip = "localhost"
        port = 50000

        in_config = 'object_detection_rpc'
        ipc_key = 1234

        blocking = False

