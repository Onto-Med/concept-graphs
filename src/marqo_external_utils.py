import itertools
import logging
import pathlib
from copy import copy
from typing import Iterable, Union, Optional, Tuple

import marqo
import numpy as np
from marqo.errors import MarqoWebError

from src.util_functions import (
    EmbeddingStore,
    DocumentStore,
    Document,
    harmonic_mean,
    ConfigLoadMethods,
)

CLIENT_BATCH_SIZE = 128


class MarqoDocument(Document):
    def __init__(self, phrases: list[str], embeddings: list[Union[list, np.ndarray]]):
        if len(phrases) != len(embeddings):
            raise ValueError(
                f"Phrases (len={len(phrases)}) and embeddings (len={len(embeddings)}) must have same length"
            )
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
        additional_index_settings: Optional[Union[dict, Iterable[str]]] = None,
    ):
        self._settings = None
        self._vector_dim = vector_dim
        self._index_name = index_name
        self._vector_name = "phrase_vector"
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
                logging.error(
                    f" Either there couldn't be a connection established or"
                    f" there seems to be no index '{self._index_name}'"
                    f" and 'create_if_not_exists' is set to False."
                )

    def _update_index_settings(self, index_settings: Union[dict, Iterable[str]]):
        self.index_settings.update(index_settings)

    def _doc_representation(
        self, did: Union[str, int], vec: Union[list, np.ndarray], cont: Optional[str]
    ) -> dict:
        _d = {
            "_id": str(did),
            "graph_cluster": [],
            "phrase": cont if cont is not None else "",
            self._vector_name: {
                "vector": vec.tolist() if isinstance(vec, np.ndarray) else vec
            },
        }
        return _d

    @staticmethod
    def _read_config(config: Union[dict, pathlib.Path, str]) -> dict:
        if isinstance(config, str):
            config = pathlib.Path(config)
        if not hasattr(config, "get"):
            config_obj = copy(config)
            config = ConfigLoadMethods.get(config_obj.suffix)(config_obj.open("rb"))
            config["index_name"] = config.get("index_name", config_obj.stem)
        else:
            config["index_name"] = config.get("index_name", "default")
        return config

    @staticmethod
    def is_accessible(config: Union[dict, pathlib.Path, str]) -> bool:
        config = MarqoEmbeddingStore._read_config(config)

        try:
            _client = marqo.Client(url=f"{config['client_url']}")
            _ = _client.get_indexes()
            return True
        except Exception as e:
            logging.error(
                f"There couldn't be a connection established for {config['client_url']}."
            )
            return False

    @classmethod
    def existing_from_config(cls, config: Union[dict, pathlib.Path, str]):
        config = MarqoEmbeddingStore._read_config(config)
        return cls(
            client_url=config.get("client_url", "http://localhost:8882"),
            index_name=config.get("index_name"),
            create_index=False,
        )

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

    def _create_index(self, index_name) -> marqo.Client:
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

    def _check_for_same_embedding(self, check_id: Union[str, Iterable[str]]):
        _field = "graph_cluster"
        _check_id_iter = lambda x: (
            check_id if isinstance(check_id, Iterable) else [check_id]
        )
        # _return_ids = []
        _additions = []
        _docs_to_add = []
        for _cid in _check_id_iter(check_id):
            _cid = str(_cid)
            # returns the added vector and the next similar;
            # if the embeddings are nearly the same (float scores are turned to int) the new one will be deleted
            _rec_result = self.marqo_index.recommend(
                documents=[_cid],
                tensor_fields=["phrase_vector"],
                limit=2,
                exclude_input_documents=False,
            ).get("hits", [])
            if len(_rec_result) >= 2:
                _that, _this = _rec_result
                if int(_this.get("_score", 0)) == int(_that.get("_score", -1)):
                    _old_id = (
                        _that.get("_id")
                        if _that.get("_id") != _cid
                        else _this.get("_id")
                    )
                    # _return_ids.append(_old_id)
                    logging.info(
                        f"For id '{_cid}' the same embedding is already in the index with id '{_old_id}'."
                    )
                else:
                    # {'_id': '11',
                    #  'phrase': 'li',
                    #  '_highlights': [{'phrase_vector': ''}],
                    #  '_score': 436.94110107421875},
                    _additions.append(_that if _that.get("_id") == _cid else _this)
                    # _return_ids.append(_cid)
        # for i, d in enumerate(_additions):
        # _recs = self.best_hits_for_field(d["_id"], delete_if_not_similar=False)
        # _doc = self._doc_representation(did=i, vec=self.get_embedding(d["_id"]), cont=d["phrase"])
        # if _recs:
        #     _doc[_field] = [_recs[0][0]]
        # _docs_to_add.append(_doc)
        _documents = [
            self._doc_representation(
                did=i, vec=self.get_embedding(d["_id"]), cont=d["phrase"]
            )
            for i, d in enumerate(_additions)
        ]
        self.delete_embeddings(_check_id_iter(check_id))
        # self.store_embeddings(_docs_to_add)
        return self.store_embeddings(_documents)
        # return _return_ids

    def store_embedding(
        self,
        embedding: Union[
            Tuple[str, list], Tuple[str, np.ndarray], Union[list, np.ndarray]
        ],
        check_for_same: bool = False,
        **kwargs,
    ) -> Iterable[str]:
        _id = str(self.store_size)
        if isinstance(embedding, tuple):
            content, embedding = embedding
        else:
            content = None
        _doc = self._doc_representation(_id, embedding, content)
        _doc.update(kwargs)
        _result = self.marqo_index.add_documents(
            documents=[_doc],
            client_batch_size=CLIENT_BATCH_SIZE,
            tensor_fields=[self._vector_name],
            mappings={self._vector_name: {"type": "custom_vector"}},
        )
        return self._check_for_same_embedding(_id) if check_for_same else [_id]

    def store_embeddings(
        self,
        embeddings: Union[
            Iterable[
                Union[dict, list, Union[Tuple[str, list], Tuple[str, np.ndarray]]]
            ],
            np.ndarray,
        ],
        embeddings_repr: list[str] = None,
        vector_name: Optional[str] = None,
        check_for_same: bool = False,
    ) -> Iterable[str]:
        _vector_name = vector_name if vector_name is not None else self._vector_name
        _offset = self.store_size
        _new_id = lambda x: str(_offset + x)
        _embeddings = []
        try:
            list(embeddings)[0]
        except IndexError:
            return []
        if not isinstance(embeddings, np.ndarray) and isinstance(
            list(embeddings)[0], dict
        ):
            for i, _dict in enumerate(embeddings):
                _dict["_id"] = _new_id(i)
        elif not isinstance(embeddings, np.ndarray) and isinstance(
            list(embeddings)[0], tuple
        ):
            for i, _tuple in enumerate(embeddings):
                _embeddings.append(
                    self._doc_representation(_new_id(i), _tuple[1], _tuple[0])
                )
            embeddings = _embeddings
        else:
            if embeddings_repr is None:
                for i, _list in enumerate(embeddings):
                    _embeddings.append(
                        self._doc_representation(_new_id(i), _list, None)
                    )
            else:
                for i, (content, vector) in enumerate(zip(embeddings_repr, embeddings)):
                    _embeddings.append(
                        self._doc_representation(_new_id(i), vector, content)
                    )
            embeddings = _embeddings

        _result = self.marqo_index.add_documents(
            documents=list(embeddings),
            client_batch_size=CLIENT_BATCH_SIZE,
            tensor_fields=[_vector_name],
            mappings={_vector_name: {"type": "custom_vector"}},
        )
        _ids = []
        for batch in _result:
            for item in batch["items"]:
                if item.get("status", -1) == 200:
                    _ids.append(item["_id"])
        return self._check_for_same_embedding(_ids) if check_for_same else _ids

    def get_embedding(self, embedding_id: str) -> Optional[np.ndarray]:
        try:
            return np.asarray(
                self.marqo_index.get_document(embedding_id, expose_facets=True)[
                    "_tensor_facets"
                ][0]["_embedding"]
            )
        except MarqoWebError as e:
            logging.error(e)
            return None

    def get_embeddings(
        self, embedding_ids: Optional[Iterable[str]] = None
    ) -> Optional[np.ndarray]:
        if embedding_ids is None:
            embedding_ids = [str(_id) for _id in range(self.store_size)]
            if len(embedding_ids) == 0:
                logging.warning(
                    f" There are no embeddings stored in index '{self._index_name}'."
                )
                return np.array([])
        try:
            return np.asarray(
                [
                    _res["_tensor_facets"][0]["_embedding"]
                    for _res in self.marqo_index.get_documents(
                        document_ids=embedding_ids, expose_facets=True
                    )["results"]
                ]
            )
        except MarqoWebError as e:
            logging.error(e)
        return None

    def update_embedding(self, embedding_id: str, **kwargs) -> bool:
        _doc = {"_id": embedding_id}
        _doc.update(kwargs)
        try:
            self.marqo_index.update_documents(
                documents=[_doc],
                client_batch_size=CLIENT_BATCH_SIZE,
            )
            return True
        except MarqoWebError as e:
            logging.error(e)
            return False

    def update_embeddings(
        self, embedding_ids: list[str], values: list[dict]
    ) -> list[str]:
        _docs = [{"_id": str(t[0]), **t[1]} for t in zip(embedding_ids, values)]
        try:
            self.marqo_index.update_documents(
                documents=_docs,
                client_batch_size=CLIENT_BATCH_SIZE,
            )
            # ToDo: is it possible to differentiate which docs were updated or if one fails all do?
            return embedding_ids
        except MarqoWebError as e:
            logging.error(e)
            return []

    def delete_embedding(self, embedding_id: str) -> bool:
        try:
            self.marqo_index.get_document(embedding_id)
        except MarqoWebError as e:
            logging.warning(f"Id doesn't exist '{embedding_id}'.")
            return False
        self.marqo_index.delete_documents([embedding_id])
        return True

    def delete_embeddings(self, embedding_ids: Iterable[str]) -> bool:
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
        embedding: Union[
            str,
            np.ndarray,
            list[float],
            tuple[str, np.ndarray],
            tuple[str, list[float]],
        ],
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
                    exclude_input_documents=False,
                )
                _recommendations = []
                if _hits := _result.get("hits", []):
                    if len(_hits) < 2:
                        return []
                    _true_doc_score = _hits[0].get("_score", 0.0)
                    _recommendations = [
                        (c, h.get("_score"))
                        for h in _hits[1:]
                        if h.get(field, False)
                        for c in (
                            [h.get(field)]
                            if not isinstance(h.get(field), list)
                            else h.get(field)
                        )
                    ]
                    if len(_recommendations) > 0:
                        return harmonic_mean(
                            [
                                x
                                for x in _recommendations
                                if x[1] >= _true_doc_score * score_frac
                            ]
                        )
            except MarqoWebError as e:
                logging.error(
                    f"Document/Embedding with id '{embedding}' is not present in the index."
                )
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
    def __init__(self, embedding_store: MarqoEmbeddingStore):
        self._embedding_store = embedding_store

    def add_document(self, document: MarqoDocument) -> Optional[set[str]]:
        _field = "graph_cluster"
        _last_store_id: int = self._embedding_store.store_size - 1
        _ids: Iterable[str] = self._embedding_store.store_embeddings(
            embeddings=document.as_tuples, check_for_same=True
        )
        if not any(_ids):
            return

        _added_ids = [(idx, i) for idx, i in enumerate(_ids) if int(i) > _last_store_id]
        _gcs = []
        for _id_tuple in _added_ids:
            _res = self._embedding_store.best_hits_for_field(
                embedding=_id_tuple[1], field=_field, delete_if_not_similar=False
            )
            if _res:
                _gcs.append(_res[0][0])
            else:
                _gcs.append(False)

        _updated_docs = []
        for _id_tuple in itertools.compress(_added_ids, _gcs):
            _updated_docs.append(
                {
                    "_id": _id_tuple[1],
                    _field: [_gcs[_id_tuple[0]]],
                }
            )
        self._embedding_store.marqo_index.update_documents(
            _updated_docs, client_batch_size=CLIENT_BATCH_SIZE
        )
        return {i["_id"] for i in _updated_docs}

    def add_documents(self, documents: Iterable[MarqoDocument]) -> Optional[set[str]]:
        return_set = set()
        for document in documents:
            return_set.update(self.add_document(document))
        return return_set

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
    print(
        list(
            mqs.best_hits_for_field(
                ("handchirurgen", mqs.get_embedding("238")), force_delete_after=True
            )
        )
    )
    print(mqs.store_size)
