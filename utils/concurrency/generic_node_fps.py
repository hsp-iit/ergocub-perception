from abc import ABC
import time
from utils.concurrency.generic_node import GenericNode


class GenericNodeFPS(GenericNode, ABC):
    def __init__(self, in_queues={}, out_queues={}, auto_write=True, auto_read=True, max_fps=30):
        super().__init__(in_queues=in_queues, out_queues=out_queues, auto_write=auto_write, auto_read=auto_read)
        self.times = [time.time()]
        self.last_time = time.time()
        self.max_fps = max_fps

    def run(self) -> None:
        self._startup()
        data = {}

        while True:

            while 1/(time.time() - self.last_time) > self.max_fps:
                time.sleep(0.01)

            self.times.append(time.time() - self.last_time)
            self.times = self.times[-10:]
            self.last_time = time.time()

            if self.auto_read:
                data = self.read_all()

            data = self.loop(data)

            if self.auto_write:
                self.write_all(data)

    def fps(self):
        return 1/(sum(self.times)/len(self.times))
