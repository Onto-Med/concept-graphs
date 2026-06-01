"""Document addition workflow for existing concept graphs."""

import logging
import pathlib
import uuid
from collections import defaultdict
from pydoc import locate

import networkx as nx
import numpy as np

from src.api.request_parsing import document_adding_json
from src.api.responses import HTTPResponses
from src.common.io import load_pickle, save_pickle
from src.core import data_functions, embedding_functions
from src.core.graph import GraphIncorp
from src.pipeline.document_results import transform_document_addition_results
from src.pipeline.load_utils import FactoryLoader
from src.pipeline.status import StepsName
from src.storage.interfaces import DocumentStore, EmbeddingStore


def _source_keys(source: dict) -> tuple[str | None, str | None]:
    client_keys = set(source.keys()).intersection(
        ["client_url", "url", "client", "clienturl"]
    )
    index_keys = set(source.keys()).intersection(["index_name", "index", "indexname"])
    return (
        next(iter(client_keys), None),
        next(iter(index_keys), None),
    )


def _remove_document_from_graphs(
    graphs: list[nx.Graph], document_id: str, remove_unreferenced_nodes: bool = True
) -> dict[str, int]:
    removed_references = 0
    removed_nodes = 0
    affected_graphs = 0
    for graph in graphs:
        graph_changed = False
        nodes_to_remove = []
        for node_id, node_data in list(graph.nodes(data=True)):
            documents = node_data.get("documents", [])
            remaining_documents = [
                document
                for document in documents
                if str(document.get("id")) != str(document_id)
            ]
            removed_from_node = len(documents) - len(remaining_documents)
            if removed_from_node == 0:
                continue
            removed_references += removed_from_node
            graph_changed = True
            if remove_unreferenced_nodes and len(remaining_documents) == 0:
                nodes_to_remove.append(node_id)
            else:
                graph.nodes[node_id]["documents"] = remaining_documents
        if nodes_to_remove:
            graph.remove_nodes_from(nodes_to_remove)
            removed_nodes += len(nodes_to_remove)
        if graph_changed:
            affected_graphs += 1
    return {
        "removed_graph_document_references": removed_references,
        "removed_graph_nodes": removed_nodes,
        "affected_graphs": affected_graphs,
    }


def add_documents_to_concept_graphs(
    # ToDo?: ``store_permanently`` only changes whether new phrases will be stored in graphs!
    #   --> regardless of the former argument:
    #       - docs won't be stored in the processed_data
    #       - docs (their phrase embeddings) will be stored in the vector store
    content_json: document_adding_json,
    data_processing: data_functions.DataProcessingFactory.DataProcessing | None = None,
    embedding_processing: embedding_functions.SentenceEmbeddingsFactory.SentenceEmbeddings
    | None = None,
    graph_processing: list[nx.Graph] | None = None,
    storage_path: str | pathlib.Path | None = None,
    process_name: str = "default",
    store_permanently: bool = True,
    document_store_cls: str = "src.storage.marqo.MarqoDocumentStore",
    embedding_store_cls: str = "src.storage.marqo.MarqoEmbeddingStore",
    document_cls: str = "src.storage.marqo.MarqoDocument",
):
    try:
        document_store = locate(document_store_cls)
        embedding_store = locate(embedding_store_cls)
        document = locate(document_cls)
        missing_classes = [
            class_path
            for class_path, located_class in (
                (document_store_cls, document_store),
                (embedding_store_cls, embedding_store),
                (document_cls, document),
            )
            if located_class is None
        ]
        if missing_classes:
            raise TypeError(
                "Could not locate configured document/vector-store classes: "
                + ", ".join(missing_classes)
            )

        if content_json.documents is None:
            return {"error": "No content provided."}, HTTPResponses.BAD_REQUEST

        ###
        try:
            data_processing = (
                FactoryLoader.load_data(str(storage_path.resolve()), process_name)
                if data_processing is None
                else data_processing
            )
            embedding_processing = (
                FactoryLoader.load_embedding(
                    str(storage_path.resolve()),
                    process_name,
                    data_processing,
                    (
                        None
                        if content_json.vectorstore_server is None
                        else content_json.vectorstore_server
                    ),
                )
                if embedding_processing is None
                else embedding_processing
            )
        except FileNotFoundError:
            _missing = "data" if data_processing is None else "embedding"
            return {
                "error": f"The serialized object for '{_missing}' doesn't seem to be present. Please finish the complete pipeline for the process '{process_name}' first."
            }, HTTPResponses.NOT_FOUND
        try:
            graph_processing = (
                FactoryLoader.load_graph(str(storage_path.resolve()), process_name)
                if graph_processing is None
                else graph_processing
            )
        except FileNotFoundError:
            logging.warning(
                "The serialized object for 'graph' doesn't seem to be present. Storing the document into the vector store will still be performed."
            )
        if (
            content_json.vectorstore_server is None
            and embedding_processing.source is None
        ):
            return {
                "error": "Only adding documents with a vectorstore server setup is supported; no vectorstore configured."
            }, HTTPResponses.NOT_IMPLEMENTED
        if embedding_processing.source is None:
            embedding_processing.source = content_json.vectorstore_server
        if len(content_json.documents) > 0 and isinstance(
            content_json.documents[0], dict
        ):
            # ToDo
            pass
        else:
            return {
                "error": "Right now only processing of documents as json is supported."
            }, HTTPResponses.NOT_IMPLEMENTED
        ###

        _source = embedding_processing.source
        _client_key, _index_key = (
            _source_keys(_source) if isinstance(_source, dict) else (None, None)
        )

        def doc_content(value):
            return value if isinstance(value, dict) else {"content": value}

        _chunk_result = data_processing.process_external_docs(
            content=[
                {
                    "id": doc_content(doc).get("id", str(uuid.uuid4())),
                    "name": doc_content(doc).get("name", None),
                    "content": doc_content(doc).get("content", ""),
                    "label": doc_content(doc).get("label", None),
                }
                for doc in content_json.documents
            ]
        )
        text_list = []
        idx_dict = defaultdict(list)
        for idx, _chunk in enumerate(
            # sorted(
            (
                _result["text"],
                set(_doc["id"] for _doc in _result["doc"]),
            )
            for _result in _chunk_result
            # ,
            # key=lambda _result: _result[0],
            # )
        ):
            for _doc in _chunk[1]:
                idx_dict[_doc].append(idx)
            text_list.append(_chunk[0])
        _embedding_result = embedding_processing.encode_external(content=text_list)

        embedding_store_impl: EmbeddingStore = embedding_store(
            client_url=_source.get(_client_key, "http://localhost:8882"),
            index_name=_source.get(_index_key, "default"),
            create_index=False,
            vector_dim=embedding_processing.embedding_dim,
        )
        doc_store_impl: DocumentStore = document_store(
            embedding_store=embedding_store_impl
        )
        added_embeddings = doc_store_impl.add_documents(
            [
                (
                    document(
                        phrases=np.take(text_list, idx, 0),
                        embeddings=np.take(_embedding_result.astype("float64"), idx, 0),
                        doc_id=_id,
                    ),
                    {
                        "offsets": [
                            x.get("offsets", [])
                            for _dict in np.take(_chunk_result, idx, 0)
                            for x in _dict.get("doc", [])
                            if x.get("id", "") == _id
                        ],
                        "text": [
                            _dict.get("text", "")
                            for _dict in np.take(_chunk_result, idx, 0)
                        ],
                    },
                )
                for _id, idx in idx_dict.items()
            ],
            as_tuple=True,
        )
        if store_permanently:
            graph_storage_path = pathlib.Path(
                storage_path / f"{process_name}_{StepsName.GRAPH}"
            ).resolve()
            save_pickle(
                GraphIncorp.with_graphs(graph_processing)
                .incorporate_phrases(
                    transform_document_addition_results(
                        (
                            (
                                k,
                                v.get("with_graph"),
                            )
                            for k, v in added_embeddings.items()
                        )
                    ).items()
                )
                .graphs,
                graph_storage_path,
            )
    except Exception as e:
        logging.exception(
            "Unhandled error while adding documents to process '%s'.", process_name
        )
        return {
            "error": str(e) + "\n--- please consult the logs!"
        }, HTTPResponses.INTERNAL_SERVER_ERROR
    return added_embeddings, HTTPResponses.OK


def delete_document_from_concept_graphs(
    document_id: str,
    embedding_processing: embedding_functions.SentenceEmbeddingsFactory.SentenceEmbeddings
    | None = None,
    graph_processing: list[nx.Graph] | None = None,
    storage_path: str | pathlib.Path | None = None,
    process_name: str = "default",
    vectorstore_server: dict | None = None,
    remove_unreferenced_nodes: bool = True,
    delete_unreferenced_embeddings: bool = False,
    embedding_store_cls: str = "src.storage.marqo.MarqoEmbeddingStore",
):
    try:
        if not document_id:
            return {"error": "No document id provided."}, HTTPResponses.BAD_REQUEST
        if storage_path is None:
            return {"error": "No storage path provided."}, HTTPResponses.BAD_REQUEST
        storage_path = pathlib.Path(storage_path)

        graph_storage_path = pathlib.Path(
            storage_path / f"{process_name}_{StepsName.GRAPH}"
        ).resolve()
        if graph_processing is None:
            graph_processing = load_pickle(graph_storage_path)
        if graph_processing is None:
            return {
                "error": f"The serialized graph object for process '{process_name}' doesn't seem to be present."
            }, HTTPResponses.NOT_FOUND

        if embedding_processing is None:
            try:
                embedding_processing = FactoryLoader.load_embedding(
                    str(storage_path.resolve()), process_name, None, vectorstore_server
                )
            except FileNotFoundError:
                embedding_processing = None

        vector_result = {
            "matched": [],
            "updated": [],
            "failed": [],
            "skipped": True,
            "reason": "No vector-store configuration available.",
        }
        source = (
            vectorstore_server
            if vectorstore_server is not None
            else getattr(embedding_processing, "source", None)
        )
        if isinstance(source, dict):
            embedding_store = locate(embedding_store_cls)
            if embedding_store is None:
                raise TypeError(
                    f"Could not locate configured vector-store class: {embedding_store_cls}"
                )
            client_key, index_key = _source_keys(source)
            embedding_store_impl: EmbeddingStore = embedding_store(
                client_url=source.get(client_key, "http://localhost:8882"),
                index_name=source.get(index_key, "default"),
                create_index=False,
                vector_dim=getattr(embedding_processing, "embedding_dim", 1024),
            )
            if hasattr(embedding_store_impl, "remove_document_provenance_from_all"):
                vector_result = (
                    embedding_store_impl.remove_document_provenance_from_all(
                        document_id,
                        delete_if_unreferenced=delete_unreferenced_embeddings,
                    )
                )
                vector_result["skipped"] = False
            else:
                vector_result["reason"] = (
                    "Configured vector-store implementation does not support document provenance removal."
                )

        graph_result = _remove_document_from_graphs(
            graph_processing,
            document_id,
            remove_unreferenced_nodes=remove_unreferenced_nodes,
        )
        if graph_result["removed_graph_document_references"] > 0:
            save_pickle(graph_processing, graph_storage_path)

        if (
            graph_result["removed_graph_document_references"] == 0
            and len(vector_result.get("matched", [])) == 0
        ):
            return {
                "document_id": document_id,
                "graph": graph_result,
                "vectorstore": vector_result,
                "message": f"No provenance for document '{document_id}' was found.",
            }, HTTPResponses.NOT_FOUND

        return {
            "document_id": document_id,
            "graph": graph_result,
            "vectorstore": vector_result,
        }, HTTPResponses.OK
    except Exception as e:
        logging.exception(
            "Unhandled error while deleting document '%s' from process '%s'.",
            document_id,
            process_name,
        )
        return {
            "error": str(e) + "\n--- please consult the logs!"
        }, HTTPResponses.INTERNAL_SERVER_ERROR
