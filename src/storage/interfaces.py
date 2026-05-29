"""Abstract interfaces for external document and embedding stores."""

import abc
import pathlib
from typing import Any, Iterable, Optional, Union

import numpy as np


class Document(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def embeddings(self) -> list[Union[np.ndarray, list]]:
        raise NotImplementedError

    @abc.abstractmethod
    def phrases(self) -> list[str]:
        raise NotImplementedError


class EmbeddingStore(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def existing_from_config(cls, config: Union[dict, pathlib.Path, str]):
        """Instantiates an object from a config."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def is_accessible(config: Union[dict, pathlib.Path, str]) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def store_embedding(
        self, embedding: Any, check_for_same: bool, **kwargs
    ) -> dict[str, set]:
        """Store the embedding and return ids/indexes grouped by result status."""
        raise NotImplementedError

    @abc.abstractmethod
    def store_embeddings(
        self,
        embeddings: Iterable,
        embeddings_repr: Iterable,
        vector_name: str,
        check_for_same: bool,
    ) -> dict[str, set]:
        """Store embeddings and return ids/indexes grouped by result status."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_embedding(self, embedding_id: str):
        raise NotImplementedError

    @abc.abstractmethod
    def get_embeddings(self, embedding_ids: Optional[Iterable]):
        raise NotImplementedError

    @abc.abstractmethod
    def update_embedding(self, embedding_id: str, **kwargs) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def update_embeddings(
        self, embedding_ids: list[str], values: list[dict]
    ) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def delete_embedding(self, embedding_id: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def delete_embeddings(self, embedding_ids: Iterable) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def best_hits_for_field(self, embedding: Union[str, np.ndarray, list[float], dict]):
        raise NotImplementedError


class DocumentStore(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add_document(
        self, document: Union[Document, tuple[Document, dict]], as_tuple: bool = False
    ) -> dict[str, dict[str, dict[str, list[str]]]]:
        """Adds a document to the store."""
        raise NotImplementedError

    @abc.abstractmethod
    def add_documents(
        self,
        document: Union[Iterable[Document], Iterable[tuple[Document, dict]]],
        as_tuple: bool = False,
    ) -> dict[str, dict[str, dict[str, dict[str, list[str]]]]]:
        """Adds documents from the iterable to the store."""
        raise NotImplementedError

    @abc.abstractmethod
    def suggest_graph_cluster(self, document: Document):
        raise NotImplementedError
