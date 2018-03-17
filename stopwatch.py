import time
import typing


class Stopwatch:
    """
    A class that operates a stopwatch. Basic functions include start(), stop(), lap(),
    and get_lap_times(). All returned times are in seconds. The accuracy of the stopwatch
    is the accuracy of the time.perf_counter() command.
    """

    def __init__(self):
        self._is_stopped = True
        self._last_time: float = 0  # The last time that either start() or lap() was called.
        self._current_lap_time: float = 0
        self._lap_times: typing.List[float] = []

    def start(self):
        if self._is_stopped:
            self._last_time = time.perf_counter()
            self._is_stopped = False

    def stop(self):
        if not self._is_stopped:
            current_time = time.perf_counter()
            self._current_lap_time += current_time - self._last_time
            self._is_stopped = True

    def lap(self):
        if not self._is_stopped:
            current_time = time.perf_counter()
            self._lap_times.append(current_time - self._last_time + self._current_lap_time)
            self._last_time = current_time
            self._current_lap_time = 0

    def lap_times(self) -> typing.List[float]:
        return [t for t in self._lap_times]

    def reset(self):
        self._is_stopped = True
        self._current_lap_time = 0
        self._lap_times = []


def run_stopwatch():
    # Should print, approximately, '[1, 2, 3]\n[1, 2, 3]\n'
    s = Stopwatch()
    for _ in range(2):
        s.start()
        time.sleep(1)
        s.lap()
        time.sleep(1)
        s.stop()
        time.sleep(1)
        s.start()
        time.sleep(1)
        s.lap()
        time.sleep(3)
        s.lap()
        s.stop()
        print(s.lap_times())
        s.reset()
