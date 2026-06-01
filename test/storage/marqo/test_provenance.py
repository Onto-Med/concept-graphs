import numpy as np

from src.storage.marqo.document_store import MarqoDocumentStore
from src.storage.marqo.documents import MarqoDocument
from src.storage.marqo.embedding_store import MarqoEmbeddingStore


class FakeEmbeddingStore:
    store_size = 1

    def __init__(self):
        self.provenance_updates = []
        self._vector_name = "phrase_vector"

    def _doc_representation(self, did, vec, cont, metadata=None):
        document = {
            "_id": str(did),
            "graph_cluster": [],
            "phrase": cont,
            self._vector_name: {
                "vector": vec.tolist() if isinstance(vec, np.ndarray) else vec
            },
        }
        if metadata:
            document.update(metadata)
        return document

    def store_embeddings(self, embeddings, check_for_same=False):
        self.stored_embeddings = list(embeddings)
        return {
            "added": [],
            "retained": ["existing-phrase"],
            "added_idx": [],
            "retained_idx": [0],
        }

    def add_document_provenance(self, embedding_id, documents):
        self.provenance_updates.append((embedding_id, list(documents)))
        return True


class FakeMarqoIndex:
    def __init__(self):
        self.documents = {
            "0": {
                "_id": "0",
                "documents": [{"id": "doc-a", "offsets": [[1, 2]]}],
            },
            "1": {
                "_id": "1",
                "documents": [
                    {"id": "doc-a", "offsets": [[3, 4]]},
                    {"id": "doc-b", "offsets": [[5, 6]]},
                ],
            },
            "2": {"_id": "2", "documents": [{"id": "doc-c", "offsets": []}]},
            "42": {
                "_id": "42",
                "documents": [{"id": "doc-a", "offsets": [[1, 2]]}],
            },
        }
        self.updated_documents = []
        self.deleted_documents = []

    def get_document(self, embedding_id):
        if embedding_id not in self.documents:
            from marqo.errors import MarqoWebError

            raise MarqoWebError(message="missing", code="missing", error_type="missing")
        return self.documents[embedding_id]

    def update_documents(self, documents, client_batch_size=None):
        self.updated_documents.extend(documents)
        for document in documents:
            self.documents[document["_id"]].update(document)

    def delete_documents(self, embedding_ids):
        self.deleted_documents.extend(embedding_ids)


class FakeMarqoEmbeddingStore(MarqoEmbeddingStore):
    def __init__(self):
        self.index = FakeMarqoIndex()

    @property
    def store_size(self) -> int:
        return 4

    @property
    def marqo_index(self):
        return self.index


def test_document_store_adds_document_provenance_to_new_embedding_documents():
    fake_store = FakeEmbeddingStore()
    document_store = MarqoDocumentStore(fake_store)
    document = MarqoDocument(
        phrases=["heart failure"],
        embeddings=[np.array([0.1, 0.2])],
        doc_id="doc-a",
    )

    embedding_documents = document_store._embedding_documents(
        (document, {"offsets": [[[5, 18]]]}), as_tuple=True
    )

    assert embedding_documents[0]["documents"] == [
        {"id": "doc-a", "offsets": [[5, 18]]}
    ]
    assert embedding_documents[0]["source"] == "document_addition"


def test_document_store_adds_provenance_to_retained_duplicate_embeddings():
    fake_store = FakeEmbeddingStore()
    document_store = MarqoDocumentStore(fake_store)
    document = MarqoDocument(
        phrases=["heart failure"],
        embeddings=[np.array([0.1, 0.2])],
        doc_id="doc-b",
    )

    result = document_store.add_document(
        (document, {"offsets": [[[10, 23]]], "text": ["heart failure"]}),
        as_tuple=True,
    )

    assert fake_store.provenance_updates == [
        ("existing-phrase", [{"id": "doc-b", "offsets": [[10, 23]]}])
    ]
    assert result["with_graph"] == {"added": [], "incorporated": []}


def test_embedding_store_merges_document_provenance_without_duplicate_offsets():
    merged = MarqoEmbeddingStore._merge_document_provenance(
        [{"id": "doc-a", "offsets": [[1, 2]]}],
        [
            {"id": "doc-a", "offsets": [[1, 2], [3, 4]]},
            {"id": "doc-b", "offsets": [[5, 6]]},
        ],
    )

    assert merged == [
        {"id": "doc-a", "offsets": [[1, 2], [3, 4]]},
        {"id": "doc-b", "offsets": [[5, 6]]},
    ]


def test_embedding_store_removes_document_provenance():
    remaining = MarqoEmbeddingStore._remove_document_provenance(
        [
            {"id": "doc-a", "offsets": [[1, 2]]},
            {"id": "doc-b", "offsets": [[3, 4]]},
        ],
        "doc-a",
    )

    assert remaining == [{"id": "doc-b", "offsets": [[3, 4]]}]


def test_embedding_store_remove_document_provenance_deletes_unreferenced_embedding():
    store = FakeMarqoEmbeddingStore()

    assert store.remove_document_provenance("42", "doc-a", delete_if_unreferenced=True)

    assert store.index.deleted_documents == ["42"]
    assert store.index.updated_documents == []


def test_embedding_store_remove_document_provenance_updates_when_references_remain():
    store = FakeMarqoEmbeddingStore()

    assert store.remove_document_provenance("1", "doc-a", delete_if_unreferenced=True)

    assert store.index.deleted_documents == []
    assert store.index.updated_documents == [
        {"_id": "1", "documents": [{"id": "doc-b", "offsets": [[5, 6]]}]}
    ]
    assert store.index.documents["1"]["documents"] == [
        {"id": "doc-b", "offsets": [[5, 6]]}
    ]


def test_embedding_store_finds_embedding_ids_for_document():
    store = FakeMarqoEmbeddingStore()

    assert store.find_embedding_ids_for_document("doc-a") == ["0", "1"]
    assert store.find_embedding_ids_for_document("doc-c") == ["2"]
    assert store.find_embedding_ids_for_document("missing") == []


def test_embedding_store_removes_document_provenance_from_all_matches():
    store = FakeMarqoEmbeddingStore()

    result = store.remove_document_provenance_from_all(
        "doc-a", delete_if_unreferenced=True
    )

    assert result == {"matched": ["0", "1"], "updated": ["0", "1"], "failed": []}
    assert store.index.deleted_documents == ["0"]
    assert store.index.updated_documents == [
        {"_id": "1", "documents": [{"id": "doc-b", "offsets": [[5, 6]]}]}
    ]


def test_embedding_store_remove_document_provenance_returns_false_for_missing_id():
    store = FakeMarqoEmbeddingStore()

    assert not store.remove_document_provenance("missing", "doc-a")
