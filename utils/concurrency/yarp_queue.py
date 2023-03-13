import numpy as np

from loguru import logger
import yarp


class YarpQueue:

    def __init__(self, data_type, local_port_name, remote_port_name=None, read_format=None, write_format=None,
                 blocking=True):

        self.type = data_type
        self.blocking = blocking
        self.read_format = read_format
        self.write_format = write_format

        yarp.Network.init()

        # TODO this is ugly and the outer if should be removed
        # However if I do that, the writing part does not work
        if remote_port_name is not None:
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
        else:
            port = yarp.Port()

        self.port = port
        self.port.open(local_port_name)

        if remote_port_name is not None:
            yarp.Network.connect(remote_port_name, local_port_name)
            logger.info(f'Connected remote port "{remote_port_name}" to local port "{local_port_name}"')

    def read(self, blocking=None):
        if blocking is None:
            blocking = self.blocking

        msg = {}

        # TODO if block is set to false we are never out of this while loop
        blocking = True  # temporary
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

    def write(self, data):

        # msg = self.port.prepare()

        v = data[self.write_format]

        if self.type == 'rgb':
            msg = yarp.ImageRgb()
            msg.resize(v.shape[1], v.shape[0])
            msg.setExternal(v.data, v.shape[1], v.shape[0])
        elif self.type == 'depth':
            v = (v / 1000).astype(np.float32)
            msg = yarp.ImageFloat()
            msg.resize(v.shape[1], v.shape[0])
            msg.setExternal(v.data, v.shape[1], v.shape[0])

        self.port.write(msg)
