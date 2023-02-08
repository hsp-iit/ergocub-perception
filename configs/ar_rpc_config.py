from utils.concurrency.pysys_node import PySysNode
from utils.confort import BaseConfig


class Network(BaseConfig):
    node = PySysNode

    class Args:
        ip = "localhost"
        port = 50000

        in_config = 'action_recognition_rpc'
        ipc_key = 5678

        blocking = False

