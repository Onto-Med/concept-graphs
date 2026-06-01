import logging
import pathlib
from collections.abc import Iterable
from copy import copy

import marqo
import numpy as np
from marqo.errors import MarqoError, MarqoWebError

from src.common.config_loading import ConfigLoadMethods
from src.core.metrics import harmonic_mean
from src.storage.interfaces import EmbeddingStore

CLIENT_BATCH_SIZE = 128


class MarqoEmbeddingStore(EmbeddingStore):
    def __init__(
        self,
        client_url: str,
        index_name: str,
        create_index: bool = False,
        vector_dim: int = 1024,
        additional_index_settings: dict | Iterable[str] | None = None,
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
            except MarqoWebError:
                logging.error(
                    f" Either there couldn't be a connection established or"
                    f" there seems to be no index '{self._index_name}'"
                    f" and 'create_if_not_exists' is set to False."
                )

    def _update_index_settings(self, index_settings: dict | Iterable[str]):
        self.index_settings.update(index_settings)

    def _doc_representation(
        self,
        did: str | int,
        vec: list | np.ndarray,
        cont: str | None,
        metadata: dict | None = None,
    ) -> dict:
        _d = {
            "_id": str(did),
            "graph_cluster": [],
            "phrase": cont if cont is not None else "",
            self._vector_name: {
                "vector": vec.tolist() if isinstance(vec, np.ndarray) else vec
            },
        }
        if metadata:
            _d.update(metadata)
        return _d

    @staticmethod
    def _read_config(config: dict | pathlib.Path | str) -> dict:
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
    def is_accessible(config: dict | pathlib.Path | str) -> bool:
        config = MarqoEmbeddingStore._read_config(config)

        try:
            _client = marqo.Client(url=f"{config['client_url']}")
            _ = _client.get_indexes()
            return True
        except (KeyError, MarqoError, OSError) as e:
            logging.error(
                "There couldn't be a connection established for %s: %s",
                config.get("client_url", "<missing client_url>"),
                e,
            )
            return False

    @classmethod
    def existing_from_config(
        cls, config: dict | pathlib.Path | str
    ) -> "MarqoEmbeddingStore":
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
        except MarqoWebError:
            self.client.delete_index(index_name)
            self.client.create_index(
                index_name=index_name,
                settings_dict=self.index_settings,
            )
        return self.client

    def _check_for_same_embedding(
        self, check_id: str | Iterable[str]
    ) -> dict[str, list]:
        _field = "graph_cluster"

        def _check_id_iter(_):
            if isinstance(check_id, Iterable) and not isinstance(check_id, str):
                return enumerate(check_id)
            return [(0, check_id)]

        _additions = []
        _retained = []
        _max = 0
        for i, _cid in _check_id_iter(check_id):
            _max += 1
            _cid = str(_cid)
            try:
                _current_doc = self.marqo_index.get_document(_cid)
            except MarqoWebError:
                logging.warning("Id doesn't exist '%s'.", _cid)
                continue
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
                    _retained.append(_old_id)
                    logging.info(
                        f"For id '{_cid}' the same embedding is already in the index with id '{_old_id}'."
                    )
                else:
                    _additions.append(
                        (
                            i,
                            _current_doc,
                        )
                    )
        _documents = [
            {
                **self._doc_representation(
                    did=i, vec=self.get_embedding(d[1]["_id"]), cont=d[1]["phrase"]
                ),
                **{
                    k: v
                    for k, v in d[1].items()
                    if not k.startswith("_")
                    and k not in {"phrase", self._vector_name, "graph_cluster"}
                },
                "graph_cluster": d[1].get("graph_cluster", []),
            }
            for i, d in enumerate(_additions)
        ]
        self.delete_embeddings([x for _, x in _check_id_iter(check_id)])
        _added = self.store_embeddings(_documents).get("added", set())
        _added_idx = [i for i, _ in _additions]
        return {
            "added": _added,
            "retained": _retained,
            "retained_idx": [i for i in range(_max) if i not in set(_added_idx)],
            "added_idx": _added_idx,
        }

    def store_embedding(
        self,
        embedding: tuple[str, list] | tuple[str, np.ndarray] | list | np.ndarray,
        check_for_same: bool = False,
        **kwargs,
    ) -> dict[str, list]:
        """See `self.store_embeddings`

        :param embedding:
        :param check_for_same:
        :param kwargs:
        :return:
        """
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
        return (
            self._check_for_same_embedding(_id)
            if check_for_same
            else {"added": _id, "retained": [], "added_idx": [0], "retained_idx": []}
        )

    def store_embeddings(
        self,
        embeddings: Iterable[dict | list | tuple[str, list] | tuple[str, np.ndarray]]
        | np.ndarray,
        embeddings_repr: list[str] = None,
        vector_name: str | None = None,
        check_for_same: bool = False,
    ) -> dict[str, list]:
        """Takes embeddings (with optional textual representation), and stores them in the vector store.

        :param embeddings: Either an ::class::`Iterable` of vector representations (in various forms)
            or a :class::`numpy.ndarray`
        :param embeddings_repr: If for `embeddings` a ::class::`numpy.ndarray` was used,
            a list of textual representations can be provided, defaults to None
        :param vector_name: The class' vector name can be changed; not advised, defaults to None
        :param check_for_same: Whether embeddings aren't added when similar embeddings are found, defaults to False
        :return: A dictionary with the following keys: 'added', 'retained', 'added_idx', 'retained_idx'
            which gives a ::class::`Set` as ids in the vector store for the added embeddings
            (or the ones which were already present) as well as the indices (*_idx) of the input embeddings
        """
        _vector_name = vector_name if vector_name is not None else self._vector_name
        _offset = self.store_size

        def _new_id(value):
            return str(_offset + value)

        _embeddings = []
        try:
            list(embeddings)[0]
        except IndexError:
            return {
                "added": list(),
                "retained": list(),
                "added_idx": list(),
                "retained_idx": list(),
            }
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
        _idx = []
        _idx_count = 0
        for batch in _result:
            for item in batch["items"]:
                if item.get("status", -1) == 200:
                    _ids.append(item["_id"])
                    _idx.append(_idx_count)
                _idx_count += 1
        return (
            self._check_for_same_embedding(_ids)
            if check_for_same
            else {"added": _ids, "retained": [], "added_idx": _idx, "retained_idx": []}
        )

    def get_embedding(self, embedding_id: str) -> np.ndarray | None:
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
        self, embedding_ids: Iterable[str] | None = None
    ) -> np.ndarray | None:
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

    @staticmethod
    def _merge_document_provenance(
        existing_documents: list[dict], new_documents: Iterable[dict]
    ) -> list[dict]:
        merged_documents = [dict(document) for document in existing_documents]
        documents_by_id = {
            document.get("id"): document
            for document in merged_documents
            if document.get("id") is not None
        }
        for new_document in new_documents:
            document_id = new_document.get("id")
            if document_id is None or document_id not in documents_by_id:
                merged_documents.append(dict(new_document))
                if document_id is not None:
                    documents_by_id[document_id] = merged_documents[-1]
                continue
            existing_offsets = documents_by_id[document_id].setdefault("offsets", [])
            for offset in new_document.get("offsets", []):
                if offset not in existing_offsets:
                    existing_offsets.append(offset)
        return merged_documents

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

    @staticmethod
    def _remove_document_provenance(
        existing_documents: list[dict], document_id: str
    ) -> list[dict]:
        return [
            document
            for document in existing_documents
            if str(document.get("id")) != str(document_id)
        ]

    def add_document_provenance(
        self, embedding_id: str, documents: Iterable[dict]
    ) -> bool:
        try:
            existing_document = self.marqo_index.get_document(str(embedding_id))
        except MarqoWebError as e:
            logging.error(e)
            return False
        merged_documents = self._merge_document_provenance(
            existing_document.get("documents", []), documents
        )
        return self.update_embedding(str(embedding_id), documents=merged_documents)

    def remove_document_provenance(
        self,
        embedding_id: str,
        document_id: str,
        delete_if_unreferenced: bool = False,
    ) -> bool:
        try:
            existing_document = self.marqo_index.get_document(str(embedding_id))
        except MarqoWebError as e:
            logging.error(e)
            return False
        remaining_documents = self._remove_document_provenance(
            existing_document.get("documents", []), document_id
        )
        if delete_if_unreferenced and len(remaining_documents) == 0:
            return self.delete_embedding(str(embedding_id))
        return self.update_embedding(str(embedding_id), documents=remaining_documents)

    def find_embedding_ids_for_document(self, document_id: str) -> list[str]:
        """Find vector-store entries that reference a document ID in provenance.

        Marqo does not expose a project-specific manifest here, so this scans the
        currently known numeric embedding IDs. The surrounding code already treats
        Marqo IDs as numeric strings assigned from the current store size.
        """
        matching_ids = []
        for embedding_id in [str(_id) for _id in range(self.store_size)]:
            try:
                document = self.marqo_index.get_document(embedding_id)
            except MarqoWebError:
                continue
            if any(
                str(doc.get("id")) == str(document_id)
                for doc in document.get("documents", [])
            ):
                matching_ids.append(embedding_id)
        return matching_ids

    def remove_document_provenance_from_all(
        self, document_id: str, delete_if_unreferenced: bool = False
    ) -> dict[str, list[str]]:
        embedding_ids = self.find_embedding_ids_for_document(document_id)
        updated = []
        failed = []
        for embedding_id in embedding_ids:
            if self.remove_document_provenance(
                embedding_id,
                document_id,
                delete_if_unreferenced=delete_if_unreferenced,
            ):
                updated.append(embedding_id)
            else:
                failed.append(embedding_id)
        return {"matched": embedding_ids, "updated": updated, "failed": failed}

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
        except MarqoWebError:
            logging.warning(f"Id doesn't exist '{embedding_id}'.")
            return False
        self.marqo_index.delete_documents([embedding_id])
        return True

    def delete_embeddings(self, embedding_ids: Iterable[str]) -> bool:
        _ids = sorted(embedding_ids)
        try:
            self.marqo_index.get_document(_ids[0])
        except MarqoWebError:
            logging.warning(f"Already first id doesn't exist '{_ids[0]}'.")
            return False
        self.marqo_index.delete_documents(_ids)
        return True

    def best_hits_for_field(
        self,
        embedding: str
        | np.ndarray
        | list[float]
        | tuple[str, np.ndarray]
        | tuple[str, list[float]],
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
            except MarqoWebError:
                logging.error(
                    f"Document/Embedding with id '{embedding}' is not present in the index."
                )
        else:
            _id = list(self.store_embedding(embedding).get("added", set()))[0]
            _gen = list(self.best_hits_for_field(str(_id), field))
            if (delete_if_not_similar and len(_gen) == 0) or force_delete_after:
                self.marqo_index.delete_documents([_id])
            else:
                self.marqo_index.update_documents([{"_id": _id, field: [_gen[0][0]]}])
            return _gen
        return []
