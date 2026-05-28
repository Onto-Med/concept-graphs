"""Threading helpers."""

import threading


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition.
    From: https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread
    """

    def __init__(self, group, target, name, target_args=(), target_kwargs=None):
        super().__init__(
            args=target_args,
            group=group,
            target=target,
            name=name,
            kwargs=target_kwargs,
        )
        self._stop_event = threading.Event()
        self._hard_stop_event = threading.Event()
        self._return = None

    @property
    def set_to_stop(self):
        return self._stop_event.is_set() or self._hard_stop_event.is_set()

    @property
    def set_to_hard_stop(self):
        return self._stop_event.is_set() and self._hard_stop_event.is_set()

    @property
    def return_value(self):
        if self.is_alive():
            return False
        return self._return

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def stop(self, hard_stop=False):
        self._stop_event.set()
        if hard_stop:
            self._hard_stop_event.set()
