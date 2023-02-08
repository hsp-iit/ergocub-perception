# import time
from abc import ABC, abstractmethod
# from collections import OrderedDict
from multiprocessing import Process

from queue import Empty, Full

import numpy as np
from loguru import logger
import yarp
import sysv_ipc
import struct

def _exception_handler(function):
    # Graceful shutdown
    def wrapper(*args):
        try:
            function(*args)
        except Exception as e:
            logger.exception(e)
            exit(1)

    return wrapper


class YarpSysNode(Process, ABC):

    def __init__(self, in_queues=None, out_queues=None, blocking=False):
        super(Process, self).__init__()
        self.ipc = sysv_ipc.MessageQueue(1234, sysv_ipc.IPC_CREAT)

        self.blocking = blocking
        self.np_buffer = {}
        self.yarp_data = {}

        yarp.Network.init()

        self._in_queues = {}
        if in_queues is not None:
            for in_q in in_queues:
                for port in in_queues[in_q]:
                    if port == "depth":
                        p = yarp.BufferedPortImageFloat()

                        depth_buffer = bytearray(
                            np.zeros((480, 640), dtype=np.float32))
                        depth_image = yarp.ImageFloat()
                        depth_image.resize(640, 480)
                        depth_image.setExternal(depth_buffer, 640, 480)

                        self.yarp_data[f'/{in_q}/{port}'] = depth_image
                        self.np_buffer[f'/{in_q}/{port}'] = depth_buffer

                    if port == "rgb":
                        p = yarp.BufferedPortImageRgb()

                        rgb_buffer = bytearray(np.zeros((480, 640, 3), dtype=np.uint8))
                        rgb_image = yarp.ImageRgb()
                        rgb_image.resize(640, 480)
                        rgb_image.setExternal(rgb_buffer, 640, 480)

                        self.yarp_data[f'/{in_q}/{port}'] = rgb_image
                        self.np_buffer[f'/{in_q}/{port}'] = rgb_buffer

                    port_id = np.random.randint(9999, size=4)
                    p.open(f'/{in_q}/{port}_{port_id}_in')
                    self._in_queues[f'/{in_q}/{port}'] = p

                    yarp.Network.connect(f'/{in_q}/{port}_out', f'/{in_q}/{port}_{port_id}_in')
                    print(f"Connecting /{in_q}/{port}_out to /{in_q}/{port}_{port_id}_in")

        self._out_queues = {}
        for out_q in out_queues:
            for port in out_queues[out_q]:
                p = yarp.Port()
                p.open(f'/{out_q}/{port}_out')
                self._out_queues[f'/{out_q}/{port}'] = p

        logger.info(f'Input queue: {", ".join(in_queues) if in_queues is not None else "None"}'
                    f' - Output queues: {", ".join(out_queues)}')

    def _startup(self):
        logger.info('Starting up...')
        self.startup()
        logger.info('Waiting for source startup...')
        data = self._recv()
        logger.success('Start up complete.')

    def _recv(self):
        msg = {}
        for name, port in self._in_queues.items():

            while True:
                if (data := port.read(False)) is not None:
                    break

            data_type = name.split('/')[-1]

            if data_type == 'rgb':
                self.yarp_data[name].copy(data)
                data = (np.frombuffer(self.np_buffer[name], dtype=np.uint8).reshape(480, 640, 3))

            if data_type == 'depth':
                self.yarp_data[name].copy(data)
                data = (np.frombuffer(self.np_buffer[name], dtype=np.float32).reshape(480, 640) * 1000).astype(np.uint16)

            msg[data_type] = data

        return msg

    def startup(self):
        pass

    def shutdown(self):
        pass

    def run(self) -> None:
        self._startup()

        # This is to wait for the first message even in non-blocking mode
        data = self._recv()

        while True:
            data = self.loop(data)
            self._send_all(data)
            data = self._recv()

    def _send_all(self, data):
        data = data
        msg = b""

        for k, v in data.items():
            if isinstance(v, int):
                out_v = struct.pack("h", v)
            elif isinstance(v, float):
                out_v = struct.pack("d", v)
            elif isinstance(v, np.ndarray):
                out_v = v.tobytes(order='C')
            elif isinstance(v, bool):
                out_v = struct.pack("?", v)
            else:
                raise Exception(f"Yarp node received unsupported data: {type(v)}")

            msg += out_v

        while self.ipc.current_messages > 0:
            self.ipc.receive(block=False)

        self.ipc.send(msg, False, type=1)

    def unpack(self, data):
        data.find()

if __name__ == '__main__':
    YarpNode('source', ['sink', 'action_recognition'])
