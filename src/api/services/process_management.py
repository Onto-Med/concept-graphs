"""Process management service helpers."""

import pathlib
import shutil
from typing import Optional

import flask

from src.common.threads import StoppableThread
from src.pipeline.steps.clustering_util import ClusteringUtil
from src.pipeline.steps.embedding_util import PhraseEmbeddingUtil
from src.pipeline.steps.graph_creation_util import GraphCreationUtil
from src.pipeline.steps.integration_util import ConceptGraphIntegrationUtil
from src.pipeline.steps.preprocessing_util import PreprocessingUtil


def delete_pipeline(
    app: flask.Flask,
    base_path: pathlib.Path,
    process_name: str,
    running_processes: dict,
    cached_processes: dict,
    wait_for_thread: Optional[StoppableThread] = None,
):
    app.logger.info(f"Deleting pipeline '{process_name}'.")
    if wait_for_thread is not None:
        app.logger.info(
            f"Waiting for running pipeline thread of '{process_name}' to stop before deleting..."
        )
        wait_for_thread.join()
    for _store in [running_processes, cached_processes]:
        _ = _store.pop(process_name, None)
    for _step in [
        PreprocessingUtil,
        PhraseEmbeddingUtil,
        ClusteringUtil,
        GraphCreationUtil,
        ConceptGraphIntegrationUtil,
    ]:
        _process_util = _step(
            app=app, file_storage=str(pathlib.Path(base_path / process_name).resolve())
        )
        _process_util.process_name = process_name
        _process_util.delete_process()
    shutil.rmtree(pathlib.Path(base_path / process_name))
