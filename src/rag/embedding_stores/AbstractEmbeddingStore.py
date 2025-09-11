import abc
from typing import Any


class ChunkEmbeddingStore(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_config(cls, index_name: str, url: str, port: int, config: dict[str, Any]):
        raise NotImplementedError

    @abc.abstractmethod
    def is_filled(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def add_chunks(self, chunks: list[dict[str, Any]]):
        raise NotImplementedError

    @abc.abstractmethod
    def get_chunks(self, question: str):
        raise NotImplementedError