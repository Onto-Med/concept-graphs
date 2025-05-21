import logging
from abc import ABC
from typing import Iterable, Union, Optional, Tuple, Generator

import marqo
import numpy as np
from marqo.errors import MarqoWebError

from util_functions import EmbeddingStore, DocumentStore, Document, harmonic_mean


class MarqoDocument(Document):
    def __init__(
            self,
            phrases: list[str],
            embeddings: list[Union[list, np.ndarray]]
    ):
        if len(phrases) != len(embeddings):
            raise ValueError(f"Phrases (len={len(phrases)}) and embeddings (len={len(embeddings)}) must have same length")
        self._embeddings = embeddings
        self._phrases = phrases

    @property
    def embeddings(self) -> list[Union[np.ndarray, list]]:
        return self._embeddings

    @property
    def phrases(self) -> list[str]:
        return self._phrases

    @property
    def as_tuples(self) -> list[tuple[str, Union[list, np.ndarray]]]:
        return list(zip(self.phrases, self.embeddings))


class MarqoEmbeddingStore(EmbeddingStore):
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

    def _doc_representation(
            self,
            did: Union[str, int],
            vec: Union[list, np.ndarray],
            cont: Optional[str]
    ) -> dict:
        _d = {
            "_id": str(did),
            "graph_cluster": [] ,
            "phrase": cont if cont is not None else "",
            self._vector_name: {
                "vector": vec.tolist() if isinstance(vec, np.ndarray) else vec
            }
        }
        return _d

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
        _doc = self._doc_representation(_id, embedding, content)
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
        _vector_name = vector_name if vector_name is not None else self._vector_name
        _offset = self.store_size
        _embeddings = []
        if not isinstance(embeddings, np.ndarray) and isinstance(list(embeddings)[0], dict):
            for i, _dict in enumerate(embeddings):
                _dict["_id"] = str(_offset + i)
        elif not isinstance(embeddings, np.ndarray) and isinstance(list(embeddings)[0], tuple):
            for i, _tuple in enumerate(embeddings):
                _embeddings.append(self._doc_representation(i, _tuple[1], _tuple[0]))
            embeddings = _embeddings
        else:
            if embeddings_repr is None:
                for i, _list in enumerate(embeddings):
                    _embeddings.append(self._doc_representation(i, _list, None))
            else:
                for (i, (content, vector)) in enumerate(zip(embeddings_repr, embeddings)):
                    _embeddings.append(self._doc_representation(i, vector, content))
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

    def delete_embedding(
            self,
            embedding_id: str
    ) -> bool:
        try:
            self.marqo_index.get_document(embedding_id)
        except MarqoWebError as e:
            logging.warning(f"Id doesn't exist '{embedding_id}'.")
            return False
        self.marqo_index.delete_documents([embedding_id])
        return True

    def delete_embeddings(
                self,
                embedding_ids: Iterable[str]
        ) -> bool:
        _ids = sorted(embedding_ids)
        try:
            self.marqo_index.get_document(_ids[0])
        except MarqoWebError as e:
            logging.warning(f"Already first id doesn't exist '{_ids[0]}'.")
            return False
        self.marqo_index.delete_documents(_ids)
        return True

    def best_hits_for_field(
            self,
            embedding: Union[str, np.ndarray, list[float], tuple[str, np.ndarray], tuple[str, list]],
            field: str = "graph_cluster",
            score_frac: float = 0.5,
            delete_if_not_similar: bool = True,
            force_delete_after: bool = False,
    ) -> list[tuple[str, float]]:
        if isinstance(embedding, str):
            try:
                _result = self.marqo_index.recommend(
                    documents=[embedding],
                    tensor_fields=[self._vector_name],
                    exclude_input_documents=False
                )
                _recommendations = []
                if _hits := _result.get("hits", []):
                    if len(_hits) < 2:
                        return []
                    _true_doc_score = _hits[0].get("_score", 0.0)
                    _recommendations = [
                        (c, h.get("_score"))
                        for h in _hits[1:] if h.get(field, False)
                        for c in ([h.get(field)] if not isinstance(h.get(field), list) else h.get(field))
                    ]
                    if len(_recommendations) > 0:
                        return harmonic_mean([x for x in _recommendations if x[1] >= _true_doc_score * score_frac])
            except MarqoWebError as e:
                logging.error(f"Document/Embedding with id '{embedding}' is not present in the index.")
        else:
            _id = self.store_embedding(embedding)
            _gen = list(self.best_hits_for_field(str(_id), field))
            if (delete_if_not_similar and len(_gen) == 0) or force_delete_after:
                self.marqo_index.delete_documents([_id])
            else:
                self.marqo_index.update_documents([{"_id": _id, field: [_gen[0][0]]}])
            return _gen
        return []

class MarqoDocumentStore(DocumentStore):
    def __init__(
            self,
            embedding_store: MarqoEmbeddingStore
    ):
        self._embedding_store = embedding_store

    def add_document(self, document: MarqoDocument):
        _ids = self._embedding_store.store_embeddings(
            embeddings=document.as_tuples
        )
        if len(list(_ids)) == 0:
            return

        _first_id = list(_ids)[0]
        _gcs = []
        for _id in _ids:
            _res = self._embedding_store.best_hits_for_field(_id)
            if _res:
                _gcs.append(_res[0][0])
            else:
                _gcs.append(False)
        if not any(_gcs):
            self._embedding_store.delete_embeddings(_ids)
            return




    def suggest_graph_cluster(self, document: MarqoDocument) -> Optional[str]:
        # _graph_cluster = self._embedding_store.best_hits_for_field(
        #     embedding=document.embedding,
        #     field="graph_cluster",
        #     score_frac=0.5,
        #     force_delete_after=True
        # )
        # if len(_graph_cluster) == 0:
        #     return None
        # return _graph_cluster[0][0]
        pass


if __name__ == "__main__":
    mqs = MarqoEmbeddingStore("http://localhost:8882", "grascco_lokal_test")
    print(mqs.store_size)
    print(list(mqs.best_hits_for_field(("handchirurgen", mqs.get_embedding("238")), force_delete_after=True)))
    print(mqs.store_size)