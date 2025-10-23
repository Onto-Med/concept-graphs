import abc
from typing import Any, Optional


class ChunkEmbeddingStore(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_config(
        cls,
        index_name: str,
        url: str,
        port: int,
        index_settings: dict[str, Any],
        force_init: bool,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def is_filled(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def reset_index(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        # field: str
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def get_chunks(
        self,
        question: str,
        filter_by: Optional[dict[str, list[str]]] = None,
        limit: int = 10,
    ):
        raise NotImplementedError
