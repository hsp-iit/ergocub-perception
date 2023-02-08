from dataclasses import dataclass, field
import time
from typing import Callable, ClassVar, Dict, Optional

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

@dataclass
class Timer:
    timers: ClassVar[Dict[str, float]] = dict()
    counters: ClassVar[Dict[str, int]] = dict()
    name: str = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization"""
        self.timers.setdefault(self.name, 0)
        self.counters.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        # if self.logger:
        #     self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time
            self.counters[self.name] += 1

        return elapsed_time

    def compute(self):
        return self.timers[self.name] / self.counters[self.name]

    @classmethod
    def reset(self):
        Timer.timers = dict()
        Timer.counters = dict()

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()

if __name__ == '__main__':
    t = Timer('test1')
    t.start()
    time.sleep(1)
    print(t.stop())
    t = Timer('test1')