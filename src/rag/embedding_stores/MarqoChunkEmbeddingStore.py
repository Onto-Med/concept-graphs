from typing import Any, Optional

from marqo import Client

from src.rag.embedding_stores.AbstractEmbeddingStore import ChunkEmbeddingStore
from src.rag.marqo_rag_utils import ResultsFields


class MarqoChunkEmbeddingStore(ChunkEmbeddingStore):
    def __init__(
            self,
            index_name: str,
            url: str = "http://localhost",
            port: int = 8882,
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
            port: int = 8882,
            index_settings: dict[str, Any] = None,
            force_init: bool = False,
    ) -> "MarqoChunkEmbeddingStore":
        if index_settings is None:
            index_settings = {}
        _store = cls(index_name, url, port, index_settings)
        if not _store._has_index():
            _store._init_index()
        else:
            if force_init:
                _store.reset_index(index_settings)
        return _store

    def is_filled(self) -> bool:
        return self._client.index(self._index_name).get_stats()["numberOfDocuments"] > 0

    def reset_index(self, with_settings: Optional[dict[str, Any]] = None) -> None:
        _settings = self._client.get_index(self._index_name).get_settings() if with_settings is None else with_settings
        self._client.delete_index(self._index_name, wait_for_readiness=True)
        self._client.create_index(
            index_name=self._index_name,
            settings_dict=_settings
        )

    def add_chunks(
            self,
            chunks: list[dict[str, Any]],
            # field: str = "text"
    ):
        self._client.index(self._index_name).add_documents(
            documents=chunks,
            # tensor_fields=[field],
            client_batch_size=64
        )

    def get_chunks(self, question: str, filter_by: Optional[dict[str, list[str]]] = None) -> list[Any]:
        limit = 10 #ToDo: limit not hard-coded?
        # ToDo: right now only filter on one field is allowed!
        filter_str = None if filter_by is None else (
            f"{list(filter_by.keys())[0]} IN ({', '.join(filter_by.get(list(filter_by.keys())[0]))})"
        )
        if filter_by is None:
            return self._client.index(self._index_name).search(question, limit=limit).get(ResultsFields.hits, [])
        else:
            result = []
            _offset = 0
            loop_control = 0
            while True:
                result.extend(
                    self._client.index(self._index_name)
                        .search(
                            question,
                            limit=limit,
                            offset=_offset,
                            filter_string=filter_str
                        )
                        .get(ResultsFields.hits, [])
                )
                _offset += 1
                loop_control += 1
                if len(result) >= limit or loop_control >= limit:
                    break
            return result