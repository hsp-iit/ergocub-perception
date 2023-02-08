import time


class Timer:
    def __init__(self, window):
        self.window = window
        self.times = []
        self.idx = 0

        self.t0 = None

    def start(self):
        self.t0 = time.perf_counter()

    def stop(self):
        if self.t0 is None:
            raise RuntimeError('Timer.stop should be called after Timer.start')

        elapsed = time.perf_counter() - self.t0

        if len(self.times) < self.window:
            self.times += [elapsed]
        else:
            self.times[self.idx] = elapsed

        self.idx = (self.idx + 1) % self.window
        self.t0 = None

    def compute(self, stop):
        if stop:
            self.stop()

        return sum(self.times) / len(self.times)


