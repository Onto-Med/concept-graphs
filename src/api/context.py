"""Application context dataclasses."""

import pathlib
from dataclasses import dataclass

import flask

from src.rag.embedding_stores.base import ChunkEmbeddingStore
from src.rag.rag import RAG


@dataclass
class ActiveRAG:
    rag: RAG
    vectorstore: ChunkEmbeddingStore
    process: str
    ready: bool = False
    initializing: bool = False
    error: str | None = None

    def mark_ready(self) -> None:
        self.ready = True
        self.initializing = False
        self.error = None

    def mark_not_ready(self, error: str | None = None) -> None:
        self.ready = False
        self.initializing = False
        self.error = error

    def switch_readiness(self):
        self.ready = not self.ready
        if self.ready:
            self.error = None


@dataclass
class ProcessContext:
    """Runtime process/thread tracking state."""

    running: dict
    threads: dict


@dataclass
class PipelineContext:
    """Runtime state for active pipeline objects."""

    active_objects: dict


@dataclass
class StorageContext:
    """Filesystem locations used by the application."""

    file_storage_dir: pathlib.Path


@dataclass
class RagContext:
    """Runtime RAG state."""

    active_by_process: dict[str, ActiveRAG]

    @property
    def active(self) -> ActiveRAG | None:
        """Compatibility accessor returning the first active RAG, if any."""
        return next(iter(self.active_by_process.values()), None)

    @active.setter
    def active(self, value: ActiveRAG | None) -> None:
        self.active_by_process.clear()
        if value is not None:
            self.active_by_process[value.process] = value


@dataclass
class AppContext:
    """Shared Flask application context and grouped runtime state."""

    app: flask.Flask
    processes: ProcessContext
    pipeline: PipelineContext
    storage: StorageContext
    rag: RagContext

    @property
    def running_processes(self) -> dict:
        """Compatibility alias for ``processes.running``."""
        return self.processes.running

    @property
    def pipeline_threads_store(self) -> dict:
        """Compatibility alias for ``processes.threads``."""
        return self.processes.threads

    @property
    def current_active_pipeline_objects(self) -> dict:
        """Compatibility alias for ``pipeline.active_objects``."""
        return self.pipeline.active_objects

    @property
    def file_storage_dir(self) -> pathlib.Path:
        """Compatibility alias for ``storage.file_storage_dir``."""
        return self.storage.file_storage_dir

    @property
    def active_rag(self) -> ActiveRAG | None:
        """Compatibility alias for ``rag.active``."""
        return self.rag.active

    @active_rag.setter
    def active_rag(self, value: ActiveRAG | None) -> None:
        self.rag.active = value
