import numpy as np

from loguru import logger
import yarp


class YarpQueue:

    def __init__(self, remote_port_name, local_port_name, data_type, read_format, blocking=True):

        self.type = data_type
        self.blocking = blocking
        self.read_format = read_format

        yarp.Network.init()

        if self.type == "depth":
            port = yarp.BufferedPortImageFloat()

            self.buffer = bytearray(
                np.zeros((480, 640), dtype=np.float32))
            self.image = yarp.ImageFloat()
            self.image.resize(640, 480)
            self.image.setExternal(self.buffer, 640, 480)

        elif self.type == "rgb":
            port = yarp.BufferedPortImageRgb()

            self.buffer = bytearray(np.zeros((480, 640, 3), dtype=np.uint8))
            self.image = yarp.ImageRgb()
            self.image.resize(640, 480)
            self.image.setExternal(self.buffer, 640, 480)
        elif self.type == "list":
            port = yarp.BufferedPortVector()

        else:
            raise ValueError('Wrong value for parameter type')

        self.port = port
        self.port.open(local_port_name)

        yarp.Network.connect(remote_port_name, local_port_name)
        logger.info(f'Connected remote port "{remote_port_name}" to local port "{local_port_name}"')

    def read(self, blocking=None):
        # TODO: add buffered port to switch from blocking to non blocking
        if blocking is None:
            blocking = self.blocking

        msg = {}

        while True:
            if (data := self.port.read(blocking)) is not None:
                break

        if self.type == 'rgb':
            self.image.copy(data)
            data = (np.frombuffer(self.buffer, dtype=np.uint8).reshape(480, 640, 3))

        elif self.type == 'depth':
            self.image.copy(data)
            data = (np.frombuffer(self.buffer, dtype=np.float32).reshape(480, 640) * 1000).astype(
                np.uint16)

        msg[self.read_format] = data

        return msg
