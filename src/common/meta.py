"""Metaclass helpers."""

from threading import Lock


class SingletonMeta(type):
    """
    Thread-safe Singleton metaclass.

    Adapted from https://refactoring.guru/design-patterns/singleton/.
    """

    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]
