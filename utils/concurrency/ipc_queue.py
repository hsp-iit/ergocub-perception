import copy
import struct

import numpy as np
import sysv_ipc

from loguru import logger


class IPCQueue:

    def __init__(self, ipc_key, write_format, auto_read=True, auto_write=True, blocking=False):
        self.auto_write = auto_write
        self.auto_read = auto_read

        self.ipc = sysv_ipc.MessageQueue(ipc_key, sysv_ipc.IPC_CREAT)
        self.write_format = write_format
        self.blocking = blocking

    def startup(self):
        pass

    def write(self, data, blocking=None):
        if blocking is None:
            blocking = self.blocking

        data_dest = copy.deepcopy(self.write_format)
        data_dest.update(data)  # add computed values while keeping default ones
        data_dest = {k: v for k, v, in data_dest.items() if k in self.write_format}  # remove unnecessary keys

        msg = b""

        for k, v in data_dest.items():
            if isinstance(v, int):
                try:
                    out_v = struct.pack("h", v)
                except struct.error as e:
                    logger.warning(f'struct_error in ipc_queue. Value not in range: {v}')
            elif isinstance(v, float):
                out_v = struct.pack("d", v)
            elif isinstance(v, np.ndarray):
                out_v = v.tobytes(order='C')
            elif isinstance(v, bool):
                out_v = struct.pack("?", v)
            else:
                raise Exception(f"Yarp node received unsupported data: {k}: {type(v)}")

            msg += out_v

        if not blocking:
            while self.ipc.current_messages > 0:
                try:
                    self.ipc.receive(block=False)
                except sysv_ipc.BusyError:
                    logger.warning('Catched BusyError: ipc_queue')

        self.ipc.send(msg, blocking, type=1)
