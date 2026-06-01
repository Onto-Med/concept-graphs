import networkx as nx

from src.api.responses import HTTPResponses
from src.common.io import load_pickle, save_pickle
from src.pipeline.document_addition import (
    _remove_document_from_graphs,
    delete_document_from_concept_graphs,
)


class FakeDeleteEmbeddingStore:
    last_instance = None

    def __init__(self, client_url, index_name, create_index=False, vector_dim=1024):
        self.client_url = client_url
        self.index_name = index_name
        self.create_index = create_index
        self.vector_dim = vector_dim
        self.calls = []
        FakeDeleteEmbeddingStore.last_instance = self

    def remove_document_provenance_from_all(
        self, document_id, delete_if_unreferenced=False
    ):
        self.calls.append((document_id, delete_if_unreferenced))
        return {
            "matched": ["10"],
            "updated": ["10"],
            "failed": [],
        }


def test_remove_document_from_graphs_removes_document_refs_and_empty_nodes():
    graph = nx.Graph()
    graph.add_node(
        1,
        documents=[
            {"id": "doc-a", "offsets": [[1, 2]]},
            {"id": "doc-b", "offsets": [[3, 4]]},
        ],
    )
    graph.add_node(2, documents=[{"id": "doc-a", "offsets": [[5, 6]]}])
    graph.add_edge(1, 2)

    result = _remove_document_from_graphs([graph], "doc-a")

    assert result == {
        "removed_graph_document_references": 2,
        "removed_graph_nodes": 1,
        "affected_graphs": 1,
    }
    assert 2 not in graph.nodes
    assert graph.nodes[1]["documents"] == [{"id": "doc-b", "offsets": [[3, 4]]}]


def test_delete_document_from_concept_graphs_updates_graph_and_vectorstore(tmp_path):
    process = "proc"
    process_path = tmp_path / process
    graph = nx.Graph()
    graph.add_node(1, documents=[{"id": "doc-a", "offsets": [[1, 2]]}])
    graph.add_node(2, documents=[{"id": "doc-b", "offsets": [[3, 4]]}])
    save_pickle([graph], process_path / f"{process}_graph")

    result, status = delete_document_from_concept_graphs(
        document_id="doc-a",
        storage_path=process_path,
        process_name=process,
        vectorstore_server={"client_url": "http://example.test", "index_name": "idx"},
        delete_unreferenced_embeddings=True,
        embedding_store_cls="test.pipeline.test_document_deletion.FakeDeleteEmbeddingStore",
    )

    assert status == HTTPResponses.OK
    assert result["graph"] == {
        "removed_graph_document_references": 1,
        "removed_graph_nodes": 1,
        "affected_graphs": 1,
    }
    assert result["vectorstore"]["updated"] == ["10"]
    assert result["vectorstore"]["skipped"] is False

    saved_graphs = load_pickle(process_path / f"{process}_graph")
    assert 1 not in saved_graphs[0].nodes
    assert 2 in saved_graphs[0].nodes


def test_delete_document_from_concept_graphs_returns_not_found_without_provenance(
    tmp_path,
):
    process = "proc"
    process_path = tmp_path / process
    graph = nx.Graph()
    graph.add_node(1, documents=[{"id": "doc-b", "offsets": [[3, 4]]}])
    save_pickle([graph], process_path / f"{process}_graph")

    result, status = delete_document_from_concept_graphs(
        document_id="doc-a",
        storage_path=process_path,
        process_name=process,
    )

    assert status == HTTPResponses.NOT_FOUND
    assert "No provenance" in result["message"]
