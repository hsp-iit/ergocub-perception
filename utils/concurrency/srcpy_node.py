from abc import abstractmethod
from queue import Empty, Full

from utils.concurrency.pypy_node import _exception_handler, PyPyNode
from loguru import logger


class SrcPyNode(PyPyNode):

    def _startup(self):
        logger.info('Starting up...')
        self.startup()
        logger.success('Start up complete.')

    @abstractmethod
    def loop(self) -> dict:
        pass

    def _send_all(self, data, blocking):
        for dest in data:

            msg = {}
            if not blocking:
                while not self._out_queues[dest].empty():
                    try:
                        msg = self._out_queues[dest].get(block=False)
                    except Empty:
                        break

            try:
                self._out_queues[dest].put(data[dest], block=blocking)
            except Full:
                pass

    @_exception_handler
    @logger.catch(reraise=True)
    def run(self) -> None:
        self._startup()

        # This is to wait for the first message even in non-blocking mode
        while True:
            data = self.loop()
            self._send_all(data, self.blocking)
