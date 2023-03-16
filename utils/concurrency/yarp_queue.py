import numpy as np

from loguru import logger
import yarp

from utils.concurrency.utils.signals import Signals


class YarpQueue:

    def __init__(self, data_type, local_port_name, remote_port_name=None,
                 read_format=None, read_default=None,
                 write_format=None, write_default=None,
                 blocking=True):

        self.local_port_name = local_port_name
        self.remote_port_name = remote_port_name

        self.type = data_type
        self.blocking = blocking
        self.read_format = read_format
        self.write_format = write_format
        self.read_default = read_default
        self.write_default = write_default

        self.latest = None

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
            style = yarp.ContactStyle()
            style.persistent = True
            yarp.Network.connect(remote_port_name, local_port_name, style)
            logger.info(f'Connected remote port "{remote_port_name}" to local port "{local_port_name}"')

    def connect(self):
        if self.remote_port_name is not None:
            yarp.Network.connect(self.remote_port_name, self.local_port_name)
        else:
            logger.error('Can\'t connnect as no local_port was specified')

    def startup(self):
        pass

    def read(self, blocking=None):

        if blocking is None:
            blocking = self.blocking

        msg = {self.read_format: self.read_default}

        # TODO if the connection dies and blocking is True the code remained
        #  block to the read call without any timeout. Have to use active wait.
        # blocking = True  # temporary

        if blocking:  # temporary
            while True:
                if (data := self.port.read(False)) is not None:
                    break
                if not yarp.Network.isConnected(self.remote_port_name, self.local_port_name):
                    data = None
                    break
        else:
            data = self.port.read(False)

        if data is not None:

            if self.type == 'rgb':
                self.image.copy(data)
                data = (np.frombuffer(self.buffer, dtype=np.uint8).reshape(480, 640, 3))

            elif self.type == 'depth':
                self.image.copy(data)
                data = (np.frombuffer(self.buffer, dtype=np.float32).reshape(480, 640) * 1000).astype(
                    np.uint16)

            self.latest = data
            data = {self.read_format: data}
        else:
            data = {}

        msg.update(data)

        return msg

    def write(self, data):

        v = data.get(self.write_format, self.write_default)

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
