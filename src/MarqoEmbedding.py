from typing import Iterable, Union, Generator

import marqo

from util_functions import EmbeddingStore


class MarqoEmbeddingExternal(EmbeddingStore):
    def __init__(self, client_url: str, index_name: str, vector_dim: int = 1024):
        self._settings = None
        self._vector_dim = vector_dim
        self._index_name = index_name
        self._vector_name = 'phrase_vector'
        self.client = marqo.Client(url=client_url)

        self._create_index(index_name)

    @property
    def index_settings(self) -> dict:
        if self._settings is None:
            self._settings = {
                "treatUrlsAndPointersAsImages": False,
                "model": "no_model",
                "modelProperties": {
                    "dimensions": self._vector_dim,
                    "type": "no_model",
                },
            }
        return self._settings

    @property
    def vector_name(self) -> str:
        return self._vector_name

    def _create_index(self, index_name) -> marqo.Client:
        try:
            self.client.create_index(
                index_name=index_name,
                settings_dict=self.index_settings,
            )
        except:
            self.client.delete_index(index_name)
            self.client.create_index(
                index_name=index_name,
                settings_dict=self.index_settings,
            )
        return self.client

    def store_embedding(self, embedding_id: str, embedding: list, **kwargs) -> str:
        _doc = {
            "_id": embedding_id,
            self._vector_name: {
                "vector": embedding
            }
        }
        _doc.update(kwargs)
        _result = self.client.index(self._index_name).add_documents(
            documents=[_doc],
            client_batch_size=128,
            tensor_fields=[self._vector_name],
            mappings={
                self._vector_name: {
                    "type": "custom_vector"
                }
            },
        )
        return _result[0]['items'][0]['_id']

    def store_embeddings(self, embedding_dicts: Iterable, vector_name: str) -> Iterable:
        _result = self.client.index(self._index_name).add_documents(
            documents=list(embedding_dicts),
            client_batch_size=128,
            tensor_fields=[vector_name],
            mappings={
                vector_name: {
                    "type": "custom_vector"
                }
            },
        )
        _ids = []
        for batch in _result:
            for item in batch['items']:
                _ids.append(item['_id'])
        return _ids

    def get_embedding(self, embedding_id: str):
        pass

    def get_embeddings(self, embedding_ids: Iterable):
        pass
