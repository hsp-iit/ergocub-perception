from abc import ABC, abstractmethod
from collections import defaultdict
from multiprocessing import Process

from loguru import logger

from utils.concurrency.utils.signals import Signals


class GenericNode(Process, ABC):

    def __init__(self, in_queues={}, out_queues={}, auto_write=True, auto_read=True):
        super(Process, self).__init__()
        self.in_queues = in_queues
        self.out_queues = out_queues
        self.auto_write = auto_write
        self.auto_read = auto_read

        self.latest = defaultdict(lambda: None)

    def _startup(self):
        logger.info('Input queues startup...')
        for queue in self.in_queues:
            self.in_queues[queue].startup()
            logger.success(f"   {queue} -> Success")

        logger.info('Output queues startup...')
        for queue in self.out_queues:
            self.out_queues[queue].startup()
            logger.success(f"   {queue} -> Success")

        logger.info('Node startup...')
        self.startup()

        logger.success('Start up complete.')

    def read_all(self, blocking=None):
        data = {}
        for queue in self.in_queues:
            data.update(self.read(queue, blocking))

        return data

    def write_all(self, data, blocking=None):
        for queue in self.out_queues:
            self.write(queue, data, blocking)

    def read(self, queue, blocking=None):
        data = self.in_queues[queue].read(blocking)
        check_format(data)

        self.latest = update_latest(self.latest, data)
        data = update_current(data, self.latest)

        return data

    def write(self, queue, data, blocking=None):
        check_format(data)
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

            if self.auto_write:
                self.write_all(data)


def check_format(data):
    if not isinstance(data, dict):
        raise ValueError('The returned value should be of type dict')


def update_latest(latest, current):
    latest.update({k: current[k] for k in current
                   if current[k] is not None and current[k] is not Signals.MISSING_VALUE and
                   current[k] is not Signals.USE_LATEST})
    return latest


def update_current(current, latest):
    current.update({k: latest[k] for k in current
                    if current[k] is Signals.USE_LATEST})
    return current
