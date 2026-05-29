"""RAG vector-store service helpers."""

import logging
import pathlib
from inspect import getfullargspec
from pydoc import locate
from typing import Optional, cast

from src.pipeline.load_utils import FactoryLoader
from src.pipeline.status import StepsName
from src.rag.embedding_stores.base import ChunkEmbeddingStore
from src.rag.text_splitters import PreprocessedSpacyTextSplitter


def initialize_chunk_vectorstore(
    process_name: str,
    config: Optional[dict],
    chunk_store: str = "src.rag.embedding_stores.marqo.MarqoChunkEmbeddingStore",
    force_init: bool = False,
):
    if config is None:
        config = {"index_settings": None}
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
                {"name": "text", "type": "text", "features": ["lexical_search"]},
            ],
            "tensorFields": ["text"],
        }
    chunk_store: ChunkEmbeddingStore = cast(
        ChunkEmbeddingStore, locate(chunk_store)
    ).from_config(
        index_name=f"{process_name}_rag",
        url=config.pop("url", "http://localhost"),
        port=config.pop("port", 8882),
        force_init=force_init,
        **config,
    )
    return chunk_store


def fill_chunk_vectorstore(process: str, rag, storage, pipeline, **kwargs) -> bool:
    """

    :param process:
    :param app_context:
    :param kwargs: e.g. splitter=splitter-config-dict
    :return:
    """
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
    _rag = rag.active
    if not _rag.initializing:
        _rag.initializing = True
        data_obj = FactoryLoader.with_active_objects(
            str(pathlib.Path(storage.file_storage_dir, process).resolve()),
            process,
            pipeline.active_objects,
            StepsName.DATA,
        )
        if data_obj is None:
            logging.error(
                f"[fill_chunk_vectorstore] Data object not initialized for process '{process}'. See logs for more information."
            )
            return False
        splitter = _splitter_class(**_splitter_options)

        try:
            _rag.vectorstore.reset_index()
        except Exception as e:
            logging.warning(f"[fill_chunk_vectorstore] {e}")

        _documents = splitter.split_preprocessed_sentences(
            data_obj.processed_docs, **_split_options
        )
        _field = "text"
        _rag.vectorstore.add_chunks(
            [
                dict(
                    {_field: d},
                    **{k: t[1][k] for k in _split_options.get("keep_metadata", [])},
                )
                for t in _documents
                for d in t[0]
            ],
            # _field,
        )

        _rag.initializing = False
        _rag.switch_readiness()
        return True
    else:
        logging.warning("[fill_chunk_vectorstore] Already initializing")
        return False
