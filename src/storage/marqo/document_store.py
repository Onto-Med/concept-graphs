import itertools
from collections.abc import Iterable

from src.storage.interfaces import DocumentStore
from src.storage.marqo.documents import MarqoDocument
from src.storage.marqo.embedding_store import MarqoEmbeddingStore

CLIENT_BATCH_SIZE = 128


class MarqoDocumentStore(DocumentStore):
    def __init__(self, embedding_store: MarqoEmbeddingStore):
        self._embedding_store = embedding_store

    @staticmethod
    def _document_obj(
        document: MarqoDocument | tuple[MarqoDocument, dict], as_tuple: bool
    ) -> MarqoDocument:
        return document[0] if as_tuple and isinstance(document, tuple) else document

    @staticmethod
    def _additional_info(
        document: MarqoDocument | tuple[MarqoDocument, dict], as_tuple: bool
    ) -> dict:
        return document[1] if as_tuple and isinstance(document, tuple) else {}

    def _provenance_for_phrase(
        self,
        document: MarqoDocument | tuple[MarqoDocument, dict],
        phrase_index: int,
        as_tuple: bool,
    ) -> dict:
        document_obj = self._document_obj(document, as_tuple)
        if document_obj.id is None:
            return {}
        additional_info = self._additional_info(document, as_tuple)
        offsets = additional_info.get("offsets", [])
        return {
            "documents": [
                {
                    "id": document_obj.id,
                    "offsets": offsets[phrase_index]
                    if phrase_index < len(offsets)
                    else [],
                }
            ],
            "source": "document_addition",
        }

    def _embedding_documents(
        self,
        document: MarqoDocument | tuple[MarqoDocument, dict],
        as_tuple: bool,
    ) -> list[dict]:
        document_obj = self._document_obj(document, as_tuple)
        return [
            self._embedding_store._doc_representation(
                did=idx,
                vec=embedding,
                cont=phrase,
                metadata=self._provenance_for_phrase(document, idx, as_tuple),
            )
            for idx, (phrase, embedding) in enumerate(document_obj.as_tuples)
        ]

    def add_document(
        self,
        document: MarqoDocument | tuple[MarqoDocument, dict],
        as_tuple: bool = False,
    ) -> dict[str, dict[str, dict[str, list]]]:
        # ToDo: this method doesn't utilize "score_frac" of "best_hits_for_field" which governs how similar phrases should be
        _field = "graph_cluster"
        _last_store_id: int = self._embedding_store.store_size - 1
        _stored = self._embedding_store.store_embeddings(
            embeddings=self._embedding_documents(document, as_tuple),
            check_for_same=True,
        )
        _ids = _stored.get("added", set())
        for retained_id, retained_idx in zip(
            _stored.get("retained", []), _stored.get("retained_idx", [])
        ):
            provenance = self._provenance_for_phrase(document, retained_idx, as_tuple)
            if provenance.get("documents"):
                self._embedding_store.add_document_provenance(
                    retained_id, provenance["documents"]
                )

        if not any(_ids):
            return {
                "with_graph": {"added": [], "incorporated": []},
                "without_graph": {"added": [], "incorporated": []},
            }

        _added_ids = [
            (idx, i, x)
            for idx, (i, x) in enumerate(zip(_ids, _stored.get("added_idx")))
            if int(i) > _last_store_id
        ]
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
            _to_append = {
                "_id": _id_tuple[1],
                _field: [_gcs[_id_tuple[0]]],
            }
            if as_tuple:
                _to_append = (
                    _to_append,
                    {k: v[_id_tuple[2]] for k, v in document[1].items()},
                )
            _updated_docs.append(_to_append)

        def read_tuple_value(values, index):
            return [(d[index] if as_tuple else d) for d in values]

        if _updated_docs:
            self._embedding_store.marqo_index.update_documents(
                read_tuple_value(_updated_docs, 0), client_batch_size=CLIENT_BATCH_SIZE
            )

        _x = [
            (
                (
                    {"_id": x.get("_id"), _field: x.get(_field, None)},
                    {
                        k: v[_stored.get("retained_idx")[i]]
                        for k, v in document[1].items()
                    },
                )
                if as_tuple
                else {"_id": x.get("_id"), _field: x.get(_field, None)}
            )
            for i, x in enumerate(
                self._embedding_store.marqo_index.get_documents(
                    list(_stored.get("retained", set()))
                ).get("results", [])
            )
        ]
        return_dict = {
            "with_graph": {
                "added": {"phrases": read_tuple_value(_updated_docs, 0)},
                "incorporated": {
                    "phrases": [
                        x for x in read_tuple_value(_x, 0) if x.get(_field) is not None
                    ]
                },
            },
            "without_graph": {
                "added": [
                    i
                    for i in _stored.get("added", [])
                    if i
                    not in set([x["_id"] for x in read_tuple_value(_updated_docs, 0)])
                ],
                "incorporated": [
                    i
                    for i in _stored.get("retained", [])
                    if i
                    not in set(
                        [
                            x["_id"]
                            for x in read_tuple_value(_x, 0)
                            if x.get(_field) is not None
                        ]
                    )
                ],
            },
        }
        if as_tuple:
            return_dict["with_graph"]["added"]["additional_info"] = read_tuple_value(
                _updated_docs, 1
            )
            return_dict["with_graph"]["incorporated"]["additional_info"] = [
                x
                for x, y in zip(read_tuple_value(_x, 1), read_tuple_value(_x, 0))
                if y.get(_field) is not None
            ]
        return return_dict

    def add_documents(
        self,
        documents: Iterable[MarqoDocument] | Iterable[tuple[MarqoDocument, dict]],
        as_tuple: bool = False,
    ) -> dict[str, dict[str, dict[str, dict[str, list]]]]:
        return_dict = dict()
        for document in documents:
            return_dict[
                (
                    document[0].id
                    if as_tuple and isinstance(document, tuple)
                    else document.id
                )
            ] = self.add_document(document, as_tuple)
        return return_dict

    def suggest_graph_cluster(self, document: MarqoDocument) -> str | None:
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
