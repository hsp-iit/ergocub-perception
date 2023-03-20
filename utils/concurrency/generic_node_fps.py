from abc import ABC
import time
from utils.concurrency.generic_node import GenericNode


class GenericNodeFPS(GenericNode, ABC):
    def __init__(self, in_queues={}, out_queues={}, auto_write=True, auto_read=True):
        super().__init__(in_queues=in_queues, out_queues=out_queues, auto_write=auto_write, auto_read=auto_read)
        self.times = [time.time()]

    def run(self) -> None:
        self._startup()
        data = {}

        while True:
            if self.auto_read:
                data = self.read_all()

            start = time.time()
            data = self.loop(data)
            self.times.append(time.time() - start)
            self.times = self.times[-10:]

            if self.auto_write:
                self.write_all(data)

    def fps(self):
        return 1/(sum(self.times)/len(self.times))
