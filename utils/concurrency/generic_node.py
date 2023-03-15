from abc import ABC, abstractmethod
from multiprocessing import Process

from loguru import logger


class GenericNode(Process, ABC):

    def __init__(self, in_queues={}, out_queues={}):
        super(Process, self).__init__()
        self.in_queues = in_queues
        self.out_queues = out_queues

        logger.info(f'Input queues: {", ".join(out_queues.keys())} '
                    f'- Output queues: {", ".join(out_queues.keys())}')

    def _startup(self):
        logger.info('Starting up...')
        self.startup()
        logger.info('Waiting for source startup...')
        # TODO maybe we don't wanna wait for every queue to be online?
        data = self._recv(blocking=True)
        logger.success('Start up complete.')
        return data

    def _recv(self, blocking=None):
        data = {}
        for queue in self.in_queues.values():
            data.update(queue.read(blocking))
        return data

    def _send_all(self, data):
        for queue in self.out_queues.values():
            queue.write(data)

    def read(self, queue, blocking=None):
        return self.in_queues[queue].read(blocking)

    def write(self, queue, data, blocking=None):
        return self.out_queues[queue].write(data, blocking)

    def startup(self):
        pass

    def shutdown(self):
        pass

    @abstractmethod
    def loop(self, data: dict) -> dict:
        pass

    def run(self) -> None:
        data = self._startup()

        while True:
            data = self.loop(data)
            self._send_all(data)

            data = self._recv()
