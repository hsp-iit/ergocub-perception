import struct
import time
from abc import ABC, abstractmethod
from multiprocessing import Process
from multiprocessing.managers import BaseManager

import numpy as np
import sysv_ipc
from loguru import logger
from queue import Empty, Full


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


class PySysNode(Process):

    def __init__(self, ip, port, in_config=None, ipc_key=None, blocking=False):
        super(Process, self).__init__()

        self.blocking = blocking

        self.ipc = sysv_ipc.MessageQueue(ipc_key, sysv_ipc.IPC_CREAT)

        BaseManager.register('get_queue')
        manager = BaseManager(address=(ip, port), authkey=b'abracadabra')
        connect(manager)
        self._in_queue = manager.get_queue(in_config)

        logger.info(f'Input queue: {in_config} - Output queues: {ipc_key}')

    def _startup(self):
        logger.info('Starting up...')
        self.startup()
        logger.info('Waiting for source startup...')
        data = self._recv()
        logger.success('Start up complete.')

    def _recv(self):

        data = self._in_queue.get()

        return data

    def _recv_nowait(self):
        if not self._in_queue.empty():
            return self._recv()
        else:
            return None

    def _send_all(self, data, blocking):
        msg = b""

        for k, v in data.items():
            if isinstance(v, int):
                try:
                    out_v = struct.pack("h", v)
                except struct.error as e:
                    logger.warning(f'struct_error in pysys_node - line: 84. Value not in range: {v}')
            elif isinstance(v, float):
                out_v = struct.pack("d", v)
            elif isinstance(v, np.ndarray):
                out_v = v.tobytes(order='C')
            elif isinstance(v, bool):
                out_v = struct.pack("?", v)
            # elif isinstance(v, str):
            #     if len(v) < 32:
            #         v += ' ' * (32 - len(v))
            #     if len(v) > 32:
            #         v = v[:32]
            #     out_v = bytes(v, 'utf-8')
            #     out_v = struct.pack("I%ds" % (len(out_v),), len(out_v), out_v)  # THIS IS 36 bytes
            else:
                raise Exception(f"Yarp node received unsupported data: {k}: {type(v)}")

            msg += out_v

        if not blocking:
            while self.ipc.current_messages > 0:
                try:
                    self.ipc.receive(block=False)
                except sysv_ipc.BusyError:
                    logger.warning('Catched BusyError: pysys_node: 105')

        self.ipc.send(msg, blocking, type=1)

    def startup(self):
        pass

    def shutdown(self):
        pass

    def loop(self, data: dict) -> dict:
        return data

    def run(self) -> None:
        self._startup()

        # This is to wait for the first message even in non-blocking mode
        data = self._recv()
        while True:
            data = self.loop(data)
            self._send_all(data, self.blocking)

            data = self._recv()



