from typing import Any, Optional

from marqo import Client

from src.rag.embedding_stores.AbstractEmbeddingStore import ChunkEmbeddingStore
from src.rag.marqo_rag_utils import ResultsFields


class MarqoChunkEmbeddingStore(ChunkEmbeddingStore):
    def __init__(
            self,
            index_name: str,
            url: str = "http://localhost",
            port: int = 8888,
            config: Optional[dict[str, Any]] = None
    ):
        self._client: Optional[Client] = Client(url=f"{url}:{port}")
        self._index_name: str = index_name
        self._config: dict = config if config is not None else {}

    def _init_index(
            self
    ):
        self._client.create_index(
            index_name=self._index_name,
            settings_dict=self._config
        )

    def _has_index(self):
        return any([(self._index_name == d["indexName"]) for d in self._client.get_indexes()["results"]])

    @classmethod
    def from_config(
            cls,
            index_name: str,
            url: str = "http://localhost",
            port: int = 8888,
            config: dict[str, Any] = None
    ) -> "MarqoChunkEmbeddingStore":
        if config is None:
            config = {}
        _store = cls(index_name, url, port, config)
        if not _store._has_index():
            _store._init_index()
        return _store

    def is_filled(self) -> bool:
        return self._client.index(self._index_name).get_stats()["numberOfDocuments"] > 0

    def add_chunks(self, chunks: list[dict[str, Any]]):
        self._client.index(self._index_name).add_documents(
            documents=chunks,
            tensor_fields=["text"],
            client_batch_size=64
        )

    def get_chunks(self, question: str):
        results = self._client.index(self._index_name).search(question)
        return results[ResultsFields.hits]