import logging
from typing import Iterable, Union, Optional, Tuple

import marqo
import numpy as np
from marqo.errors import MarqoWebError

from util_functions import EmbeddingStore

#ToDo: right now, embeddings will be normalized
class MarqoEmbeddingExternal(EmbeddingStore):
    def __init__(
            self,
            client_url: str,
            index_name: str,
            create_index: bool = False,
            vector_dim: int = 1024,
            additional_index_settings: Optional[Union[dict, Iterable[str]]] = None
    ):
        self._settings = None
        self._vector_dim = vector_dim
        self._index_name = index_name
        self._vector_name = 'phrase_vector'
        self.client = marqo.Client(url=client_url)

        if create_index:
            if additional_index_settings is not None:
                self._update_index_settings(additional_index_settings)
            self._create_index(index_name)
        else:
            try:
                self._settings = self.marqo_index.get_settings()
                self._vector_dim = self._settings["modelProperties"]["dimensions"]
            except MarqoWebError as e:
                logging.error(f" Either there couldn't be a connection established or"
                              f" there seems to be no index '{self._index_name}'"
                              f" and 'create_if_not_exists' is set to False.")

    def _update_index_settings(
            self,
            index_settings: Union[dict, Iterable[str]]
    ):
        self.index_settings.update(index_settings)

    @property
    def marqo_index(self):
        return self.client.index(self._index_name)

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
                "normalizeEmbeddings": False,
            }
        return self._settings

    @property
    def vector_name(self) -> str:
        return self._vector_name

    @property
    def store_size(self) -> int:
        return self.marqo_index.get_stats()["numberOfDocuments"]

    def _create_index(
            self,
            index_name
    ) -> marqo.Client:
        try:
            self.client.create_index(
                index_name=index_name,
                settings_dict=self.index_settings,
            )
        except MarqoWebError as e:
            self.client.delete_index(index_name)
            self.client.create_index(
                index_name=index_name,
                settings_dict=self.index_settings,
            )
        return self.client

    def store_embedding(
            self,
            embedding: Union[Tuple[str, list], Tuple[str, np.ndarray], Union[list, np.ndarray]],
            **kwargs
    ) -> str:
        _id = str(self.store_size)
        if isinstance(embedding, tuple):
            content, embedding = embedding
        else:
            content = None
        _doc = {
            "_id": _id,
            self._vector_name: {
                "vector": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            }
        }
        if content is not None:
            _doc["phrase"] = content
        _doc.update(kwargs)
        _result = self.marqo_index.add_documents(
            documents=[_doc],
            client_batch_size=128,
            tensor_fields=[self._vector_name],
            mappings={
                self._vector_name: {
                    "type": "custom_vector"
                }
            },
        )
        return _id

    def store_embeddings(
            self,
            embeddings: Union[Iterable[Union[dict, list, Union[Tuple[str, list], Tuple[str, np.ndarray]]]], np.ndarray],
            embeddings_repr: list[str] = None,
            vector_name: Optional[str] = None
    ) -> Iterable:
        def _doc_representation(did: Union[str, int], vec: Union[list, np.ndarray], cont: Optional[str]):
            _d = {
                "_id": str(did),
                self._vector_name: {
                    "vector": vec.tolist() if isinstance(vec, np.ndarray) else vec
                }
            }
            if cont is not None:
                _d["phrase"] = cont
            return _d

        _vector_name = vector_name if vector_name is not None else self._vector_name
        _offset = self.store_size
        _embeddings = []
        if not isinstance(embeddings, np.ndarray) and isinstance(list(embeddings)[0], dict):
            for i, _dict in enumerate(embeddings):
                _dict["_id"] = str(_offset + i)
        elif not isinstance(embeddings, np.ndarray) and isinstance(list(embeddings)[0], tuple):
            for i, _tuple in enumerate(embeddings):
                _embeddings.append(_doc_representation(i, _tuple[1], _tuple[0]))
            embeddings = _embeddings
        else:
            if embeddings_repr is None:
                for i, _list in enumerate(embeddings):
                    _embeddings.append(_doc_representation(i, _list, None))
            else:
                for (i, (content, vector)) in enumerate(zip(embeddings_repr, embeddings)):
                    _embeddings.append(_doc_representation(i, vector, content))
            embeddings = _embeddings

        _result = self.marqo_index.add_documents(
            documents=list(embeddings),
            client_batch_size=128,
            tensor_fields=[_vector_name],
            mappings={
                _vector_name: {
                    "type": "custom_vector"
                }
            },
        )
        _ids = []
        for batch in _result:
            for item in batch['items']:
                _ids.append(item['_id'])
        return _ids

    def get_embedding(
            self,
            embedding_id: str
    ) -> np.ndarray:
        try:
            return np.asarray(
                self.marqo_index.get_document(embedding_id, expose_facets=True)
                ["_tensor_facets"][0]["_embedding"]
            )
        except MarqoWebError as e:
            logging.error(e)
            return np.array([])

    def get_embeddings(
            self,
            embedding_ids: Optional[Iterable[str]] = None
    ) -> np.ndarray:
        if embedding_ids is None:
            embedding_ids = [str(_id) for _id in range(self.store_size)]
            if len(embedding_ids) == 0:
                logging.warning(f" There are no embeddings stored in index '{self._index_name}'.")
                return np.array([])
        try:
            return np.asarray(
                [
                    _res["_tensor_facets"][0]["_embedding"]
                    for _res in self.marqo_index.get_documents(document_ids=embedding_ids, expose_facets=True)["results"]
                ]
            )
        except MarqoWebError as e:
            logging.error(e)
        return np.array([])
