"""RAG vector-store service helpers."""

import logging
import pathlib
from inspect import getfullargspec
from pydoc import locate
from typing import cast

from marqo.errors import MarqoError

from src.common.parsing import string_conformity
from src.pipeline.load_utils import FactoryLoader
from src.pipeline.status import StepsName
from src.rag.embedding_stores.base import ChunkEmbeddingStore
from src.rag.text_splitters import PreprocessedSpacyTextSplitter


def initialize_chunk_vectorstore(
    process_name: str,
    config: dict | None,
    chunk_store: str = "src.rag.embedding_stores.marqo.MarqoChunkEmbeddingStore",
    force_init: bool = False,
):
    process_name = string_conformity(process_name)
    if config is None:
        config = {"index_settings": None}
    else:
        config = dict(config)
    if config.get("index_settings", None) is None or len(config["index_settings"]) == 0:
        config["index_settings"] = {
            "type": "structured",
            "model": "hf/multilingual-e5-base",
            "normalizeEmbeddings": True,
            "textPreprocessing": {
                "splitLength": 3,
                "splitOverlap": 1,
                "splitMethod": "sentence",
            },
            "allFields": [
                {
                    "name": "doc_id",
                    "type": "text",
                    "features": ["lexical_search", "filter"],
                },
                {
                    "name": "doc_name",
                    "type": "text",
                    "features": ["lexical_search", "filter"],
                },
                {
                    "name": "chunk_index",
                    "type": "int",
                    "features": ["filter"],
                },
                {"name": "text", "type": "text", "features": ["lexical_search"]},
            ],
            "tensorFields": ["text"],
        }
    index_name = f"{process_name}_rag"
    url = config.pop("url", "http://localhost")
    port = config.pop("port", 8882)
    logging.info(
        "[initialize_chunk_vectorstore] Initializing RAG vector store index='%s' url='%s' port=%s force_init=%s",
        index_name,
        url,
        port,
        force_init,
    )
    chunk_store: ChunkEmbeddingStore = cast(
        ChunkEmbeddingStore, locate(chunk_store)
    ).from_config(
        index_name=index_name,
        url=url,
        port=port,
        force_init=force_init,
        **config,
    )
    try:
        logging.info(
            "[initialize_chunk_vectorstore] RAG vector store index='%s' filled=%s",
            index_name,
            chunk_store.is_filled(),
        )
    except Exception as exc:
        logging.warning(
            "[initialize_chunk_vectorstore] Could not inspect RAG vector store index='%s': %s",
            index_name,
            exc,
        )
    return chunk_store


def fill_chunk_vectorstore(process: str, rag, storage, pipeline, **kwargs) -> bool:
    """

    :param process:
    :param app_context:
    :param kwargs: e.g. splitter=splitter-config-dict
    :return:
    """
    process = string_conformity(process)
    logging.info(
        "[fill_chunk_vectorstore] Starting RAG indexing for process '%s'.", process
    )
    _splitter_class = PreprocessedSpacyTextSplitter
    _split_options = {
        "doc_metadata_key": kwargs.get("splitter", {}).pop(
            "doc_metadata_key", "doc_id"
        ),
        "keep_metadata": kwargs.get("splitter", {}).pop(
            "keep_metadata", ["doc_id", "doc_name"]
        ),
    }
    _splitter_options = {
        k: v
        for k, v in kwargs.pop(
            "splitter", {"chunk_size": 400, "chunk_overlap": 100}
        ).items()
        if k in getfullargspec(_splitter_class).args
    }
    _rag = rag.active_by_process.get(process)
    if _rag is None:
        logging.error(
            "[fill_chunk_vectorstore] No RAG initialized for process '%s'.", process
        )
        return False
    if not _rag.initializing:
        _rag.initializing = True
        data_path = str(pathlib.Path(storage.file_storage_dir, process).resolve())
        logging.info(
            "[fill_chunk_vectorstore] Loading DATA object for process '%s' from '%s'. Active pipeline keys: %s",
            process,
            data_path,
            list(pipeline.active_objects.keys()),
        )
        data_obj = FactoryLoader.with_active_objects(
            data_path,
            process,
            pipeline.active_objects,
            StepsName.DATA,
        )
        if data_obj is None:
            error = (
                f"Data object not initialized for process '{process}'. "
                "Run/load the pipeline data step before initializing RAG."
            )
            logging.error("[fill_chunk_vectorstore] %s", error)
            _rag.mark_not_ready(error)
            return False
        processed_docs = getattr(data_obj, "processed_docs", None)
        logging.info(
            "[fill_chunk_vectorstore] Loaded DATA object type=%s processed_docs=%s",
            type(data_obj).__name__,
            "missing" if processed_docs is None else len(processed_docs),
        )
        splitter = _splitter_class(**_splitter_options)

        try:
            _rag.vectorstore.reset_index()
        except (MarqoError, RuntimeError, ValueError, TypeError) as e:
            logging.warning(
                "[fill_chunk_vectorstore] Could not reset vectorstore: %s", e
            )

        try:
            documents = list(
                splitter.split_preprocessed_sentences(
                    data_obj.processed_docs, **_split_options
                )
            )
        except (AttributeError, TypeError, ValueError) as exc:
            error = f"Could not split processed documents for RAG process '{process}': {exc}"
            logging.error("[fill_chunk_vectorstore] %s", error)
            _rag.mark_not_ready(error)
            return False

        logging.info(
            "[fill_chunk_vectorstore] Split process '%s' into %s document chunk groups.",
            process,
            len(documents),
        )
        _field = "text"
        chunks = []
        for chunk_group, metadata in documents:
            kept_metadata = {
                key: metadata.get(key)
                for key in _split_options.get("keep_metadata", [])
                if key in metadata
            }
            for chunk_index, chunk in enumerate(chunk_group):
                chunks.append(
                    dict(
                        {_field: chunk, "chunk_index": chunk_index},
                        **kept_metadata,
                    )
                )
        logging.info(
            "[fill_chunk_vectorstore] Prepared %s chunks for RAG index '%s'.",
            len(chunks),
            getattr(_rag.vectorstore, "index_name", f"{process}_rag"),
        )
        if not chunks:
            error = f"No RAG chunks were produced for process '{process}'."
            logging.error("[fill_chunk_vectorstore] %s", error)
            _rag.mark_not_ready(error)
            return False

        _rag.vectorstore.add_chunks(
            chunks,
            # _field,
        )
        logging.info(
            "[fill_chunk_vectorstore] Submitted %s chunks to RAG index '%s'.",
            len(chunks),
            getattr(_rag.vectorstore, "index_name", f"{process}_rag"),
        )

        if not _rag.vectorstore.is_filled():
            error = f"RAG vector store for process '{process}' is still empty after filling."
            logging.error("[fill_chunk_vectorstore] %s", error)
            _rag.mark_not_ready(error)
            return False

        logging.info(
            "[fill_chunk_vectorstore] Finished RAG indexing for process '%s'. vectorstore_filled=True document_count=%s",
            process,
            getattr(_rag.vectorstore, "document_count", lambda: "unknown")(),
        )
        _rag.mark_ready()
        return True
    else:
        logging.warning("[fill_chunk_vectorstore] Already initializing")
        return False
