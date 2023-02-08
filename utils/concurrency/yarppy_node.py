# import time
import copy
import random
import time
from abc import ABC, abstractmethod
# from collections import OrderedDict
from multiprocessing import Process
from multiprocessing.managers import BaseManager

from queue import Empty, Full

import numpy as np
from loguru import logger
import yarp
import sysv_ipc
import struct
# print("WHA CI ARRIVO SENZA UN SENSO AHAHAHAHAH :D")

def _exception_handler(function):
    # Graceful shutdown
    def wrapper(*args):
        try:
            function(*args)
        except Exception as e:
            logger.exception(e)
            exit(1)

    return wrapper

def connect(manager):
    logger.info('Connecting to manager...')
    start = time.time()

    while True:
        try:
            manager.connect()
            break
        except ConnectionRefusedError as e:
            if time.time() - start > 120:
                logger.error('Connection refused.')
                raise e
            time.sleep(1)
    logger.success('Connected to manager.')

class YarpPyNode(Process, ABC):

    def __init__(self, ip, port, in_config=None, out_config=None, blocking=False):
        super(Process, self).__init__()

        self.out_config = out_config
        self.in_config = in_config

        BaseManager.register('get_queue')
        manager = BaseManager(address=(ip, port), authkey=b'abracadabra')
        connect(manager)

        self.blocking = blocking
        self.np_buffer = {}
        self.yarp_data = {}

        yarp.Network.init()

        self._in_queues = {}
        if in_config is not None:
            for in_q in in_config:
                for port in in_config[in_q]:
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

                    port_id = random.randint(0, 9999)
                    p.open(f'/{in_q}/{port}_{port_id:04}_in')
                    self._in_queues[f'/{in_q}/{port}'] = p

                    yarp.Network.connect(f'/{in_q}/{port}_out', f'/{in_q}/{port}_{port_id:04}_in')
                    print(f"Connecting /{in_q}/{port}_out to /{in_q}/{port}_{port_id:04}_in")


        self._out_queues = {k: manager.get_queue(k) for k in out_config}

        logger.info(f'Input queue: {", ".join(in_config) if in_config is not None else "None"}'
                    f' - Output queues: {", ".join(out_config)}')

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
            self._send_all(data, self.blocking)
            data = self._recv()

    def _send_all(self, data, blocking):
        for dest in self.out_config:
            data_dest = copy.deepcopy(self.out_config[dest])  # TODO BEFORE IT WAS {}
            data_dest.update(data)  # add computed values while keeping default ones
            data_dest = {k: v for k, v, in data_dest.items() if k in self.out_config[dest]}  # remove unnecessary keys

            msg = {}
            if not blocking:
                while not self._out_queues[dest].empty():
                    try:
                        msg = self._out_queues[dest].get(block=False)
                    except Empty:
                        break

            msg.update(data_dest)
            try:
                self._out_queues[dest].put(msg, block=blocking)
            except Full:
                pass


