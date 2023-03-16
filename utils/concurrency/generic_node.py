from abc import ABC, abstractmethod
from multiprocessing import Process

from loguru import logger


class GenericNode(Process, ABC):

    def __init__(self, in_queues={}, out_queues={}, auto_write=True, auto_read=True):
        super(Process, self).__init__()
        self.in_queues = in_queues
        self.out_queues = out_queues

        self.auto_write = auto_write
        self.auto_read = auto_read

        logger.info(f'Input queues: {", ".join(out_queues.keys())} '
                    f'- Output queues: {", ".join(out_queues.keys())}')

    def _startup(self):
        logger.info('Starting up...')
        self.startup()
        logger.info('Waiting for source startup...')
        # TODO maybe we don't wanna wait for every queue to be online?
        self.startup()
        logger.success('Start up complete.')

    def read_all(self, blocking=None):
        data = {}
        for queue in self.in_queues.values():
            data.update(queue.read(blocking))
        return data

    def write_all(self, data):
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
        self._startup()
        data = {}

        while True:
            if self.auto_read:
                data = self.read_all()

            data = self.loop(data)

            if data is not None and self.auto_write:
                self.write_all(data)

