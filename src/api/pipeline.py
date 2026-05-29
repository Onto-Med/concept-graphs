"""Pipeline route implementation.

This module keeps the pipeline orchestration code out of ``main.py`` while
preserving the existing Flask route behavior.
"""

import logging
import pathlib
from dataclasses import dataclass
from typing import Optional

import flask
import yaml
from flask import jsonify, request
from werkzeug.datastructures import FileStorage

from src.pipeline.steps.clustering_util import ClusteringUtil
from src.pipeline.steps.embedding_util import PhraseEmbeddingUtil
from src.pipeline.steps.graph_creation_util import GraphCreationUtil
from src.pipeline.steps.integration_util import ConceptGraphIntegrationUtil
from src.pipeline.load_utils import FactoryLoader
from src.api.request_parsing import parse_pipeline_config_json, pipeline_json_config
from src.api.services.artifact_responses import graph_get_statistics
from src.api.services.configuration import read_config
from src.api.services.document_server import (
    check_data_server,
    get_data_server_config,
    get_documents_from_es_server,
)
from src.api.services.pipeline_params import get_pipeline_query_params
from src.pipeline.processes import start_processes, start_thread
from src.api.responses import HTTPResponses
from src.common.threads import StoppableThread
from src.pipeline.base import BaseUtil
from src.pipeline.status import (
    ProcessStatus,
    StepsName,
    add_status_to_running_process,
    pipeline_query_params,
)
from src.pipeline.steps.preprocessing_util import PreprocessingUtil
from src.core import (
    cluster_functions,
    data_functions,
    embedding_functions,
    integration_functions,
)
from src.storage import marqo_external_utils


DEFAULT_VECTOR_STORE = {"url": "http://localhost", "port": 8882}


@dataclass
class PipelineRouteContext:
    """Dependencies needed by pipeline route orchestration."""

    app: flask.Flask
    processes: object
    pipeline: object
    storage: object


@dataclass
class PipelineRequestData:
    data: object = False
    data_upload: bool = False
    document_server_config: object = False
    vector_store_config: Optional[dict] = None
    replace_keys: Optional[dict] = None
    label_getter: Optional[str] = None
    labels: object = None
    data_config: object = None
    embedding_config: object = None
    clustering_config: object = None
    graph_config: object = None
    content_type_json: bool = False


@dataclass
class PreparedPipeline:
    processes_threading: list[tuple[BaseUtil, object, str]]
    last_step: str


def _default_query_params() -> pipeline_query_params:
    """Return safe default query parameters used before request parsing completes."""
    return pipeline_query_params(
        process_name="not set",
        language="en",
        skip_present=True,
        omitted_pipeline_steps=[],
        return_statistics=False,
    )


def _temporary_upload_path(
    file_storage_dir: pathlib.Path, upload: FileStorage
) -> pathlib.Path:
    """Persist an uploaded file in the temporary stream directory and return its path."""
    target = file_storage_dir / ".tmp_streams" / upload.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    upload.save(target)
    return target


def _parse_json_pipeline_request(
    config_object_json: pipeline_json_config,
    query_params: pipeline_query_params,
):
    """Build pipeline request data from an application/json request body."""
    document_server = config_object_json.document_server
    if document_server is None:
        return None, (
            jsonify(
                name=(
                    config_object_json.name
                    if config_object_json.name is not None
                    else query_params.process_name
                ),
                error="No configuration entry for documents on a server provided.",
            ),
            int(HTTPResponses.BAD_REQUEST),
        )

    return (
        PipelineRequestData(
            document_server_config=document_server.copy(),
            vector_store_config=(
                config_object_json.vectorstore_server
                if config_object_json.vectorstore_server is not None
                else DEFAULT_VECTOR_STORE.copy()
            ),
            replace_keys=document_server.get("replace_keys", {"text": "content"}),
            label_getter=document_server.get("label_key", None),
            data_config=config_object_json.data,
            embedding_config=config_object_json.embedding,
            clustering_config=config_object_json.clustering,
            graph_config=config_object_json.graph,
            content_type_json=True,
        ),
        None,
    )


def _parse_multipart_pipeline_request(app_context, query_params: pipeline_query_params):
    """Build pipeline request data from multipart form fields and uploaded files."""
    data = request.files.get("data", False)
    document_server_config = request.files.get("document_server_config", False)
    vector_store_config = request.files.get("vectorstore_server_config", False)
    data_upload = False

    if vector_store_config:
        if isinstance(vector_store_config, FileStorage):
            vector_store_config = yaml.safe_load(vector_store_config.stream)
        else:
            vector_store_config = DEFAULT_VECTOR_STORE.copy()
    else:
        vector_store_config = DEFAULT_VECTOR_STORE.copy()

    if not data and not document_server_config:
        return None, (
            jsonify(
                name=query_params.process_name,
                error="Neither data provided for upload with 'data' key nor a config file for documents on a server",
            ),
            int(HTTPResponses.BAD_REQUEST),
        )

    if data and not document_server_config:
        data = _temporary_upload_path(app_context.storage.file_storage_dir, data)
        data_upload = True

    labels = request.files.get("labels", None)
    if labels is not None:
        labels = _temporary_upload_path(app_context.storage.file_storage_dir, labels)

    return (
        PipelineRequestData(
            data=data,
            data_upload=data_upload,
            document_server_config=document_server_config,
            vector_store_config=vector_store_config,
            labels=labels,
            data_config=request.files.get(f"{StepsName.DATA}_config", None),
            embedding_config=request.files.get(f"{StepsName.EMBEDDING}_config", None),
            clustering_config=request.files.get(f"{StepsName.CLUSTERING}_config", None),
            graph_config=request.files.get(f"{StepsName.GRAPH}_config", None),
        ),
        None,
    )


def _parse_pipeline_request(app_context, query_params, config_object_json):
    """Dispatch request parsing based on the incoming content type."""
    content_type_json = request.headers.get("Content-Type") == "application/json"
    if content_type_json:
        return _parse_json_pipeline_request(config_object_json, query_params)
    return _parse_multipart_pipeline_request(app_context, query_params)


def _normalize_vector_store_config(
    vector_store_config: Optional[dict],
) -> Optional[dict]:
    """Convert vector-store settings to client_url form and verify accessibility."""
    if vector_store_config is None:
        return None

    vector_store_config = dict(vector_store_config)
    url = vector_store_config.pop("url", "http://localhost")
    port = str(vector_store_config.pop("port", 8882))
    vector_store_config["client_url"] = f"{url}:{port}"

    if marqo_external_utils.MarqoEmbeddingStore.is_accessible(
        vector_store_config.copy()
    ):
        return vector_store_config

    logging.warning(
        "Vector store doesn't seem to be accessible under '%s'. Using 'pickle' storage.",
        vector_store_config["client_url"],
    )
    return None


def _load_data_from_document_server(
    app_context, query_params: pipeline_query_params, request_data: PipelineRequestData
):
    """Fetch documents from the configured document server into request data."""
    ds_base_config = get_data_server_config(
        request_data.document_server_config, app_context.app
    )
    if not check_data_server(ds_base_config):
        return (
            jsonify(
                name=query_params.process_name,
                error=f"There is no data server at the specified location ({ds_base_config}) or it contains no data.",
            ),
            int(HTTPResponses.NOT_FOUND),
        )

    request_data.data = get_documents_from_es_server(
        url=ds_base_config["url"],
        port=ds_base_config["port"],
        index=ds_base_config["index"],
        size=int(ds_base_config["size"]),
        other_id=ds_base_config["other_id"],
    )
    request_data.replace_keys = ds_base_config.get("replace_keys", {"text": "content"})
    request_data.label_getter = ds_base_config.get("label_key", None)
    return None


def _pipeline_process_definitions(vector_store_config: Optional[dict], request_data):
    """Return the ordered pipeline step definitions for the current request."""
    processes = [
        (
            StepsName.DATA,
            PreprocessingUtil,
            request_data.data_config,
            data_functions.DataProcessingFactory,
        ),
        (
            StepsName.EMBEDDING,
            PhraseEmbeddingUtil,
            request_data.embedding_config,
            embedding_functions.SentenceEmbeddingsFactory,
        ),
        (
            StepsName.CLUSTERING,
            ClusteringUtil,
            request_data.clustering_config,
            cluster_functions.PhraseClusterFactory,
        ),
        (
            StepsName.GRAPH,
            GraphCreationUtil,
            request_data.graph_config,
            cluster_functions.WordEmbeddingClustering,
        ),
    ]
    if vector_store_config is not None:
        processes.append(
            (
                StepsName.INTEGRATION,
                ConceptGraphIntegrationUtil,
                {},
                integration_functions.ConceptGraphIntegrationFactory,
            )
        )
    return processes


def _load_skipped_step(app_context, query_params, step_name: str, vector_store_config):
    """Load a serialized step result into the active object cache."""
    active_objects = app_context.pipeline.active_objects[query_params.process_name]
    active_objects[step_name] = FactoryLoader.load(
        step=step_name,
        path=str(
            pathlib.Path(
                app_context.storage.file_storage_dir,
                query_params.process_name,
            ).resolve()
        ),
        process=query_params.process_name,
        data_obj=active_objects.get(StepsName.DATA, None),
        emb_obj=active_objects.get(StepsName.EMBEDDING, None),
        vector_store=vector_store_config,
    )


def _is_omitted_step(query_params, step_name: str) -> bool:
    """Return whether a step was explicitly omitted by the request."""
    return step_name in query_params.omitted_pipeline_steps


def _should_skip_present_step(query_params, step_name: str) -> bool:
    """Return whether an existing step should be reused instead of recomputed."""
    return query_params.skip_present


def _configure_process_step(
    app_context,
    process_obj: BaseUtil,
    step_name: str,
    config,
    query_params,
    request_data: PipelineRequestData,
    vector_store_config: Optional[dict],
):
    """Read step configuration and apply step-specific runtime inputs."""
    read_config(
        app=app_context.app,
        processor=process_obj,
        process_type=step_name,
        process_name=query_params.process_name,
        config=config,
        language=query_params.language,
        mode="json" if request_data.content_type_json else "yaml",
    )

    if step_name == StepsName.DATA:
        process_obj.read_labels(
            request_data.labels
            if request_data.label_getter is None
            else request_data.label_getter
        )
        process_obj.read_data(
            request_data.data,
            replace_keys=request_data.replace_keys,
            label_getter=request_data.label_getter,
        )
    elif step_name == StepsName.EMBEDDING:
        process_obj.storage_method = _embedding_storage_method(
            process_obj, vector_store_config
        )
    elif step_name == StepsName.INTEGRATION:
        process_obj.config["check_for_reasonable_result"] = True


def _embedding_storage_method(process_obj, vector_store_config: Optional[dict]):
    """Choose pickle or vector-store backing for phrase embeddings."""
    if vector_store_config is None:
        return "pickle", None
    if process_obj.storage_method == "vectorstore":
        return "vectorstore", vector_store_config
    return "pickle", None


def _prepare_pipeline_processes(
    app_context,
    query_params: pipeline_query_params,
    request_data: PipelineRequestData,
    vector_store_config: Optional[dict],
) -> PreparedPipeline:
    """Create and configure pipeline processors that still need to run."""
    processes_threading = []
    app_context.pipeline.active_objects[query_params.process_name] = {
        key: None for key in StepsName.ALL
    }
    previous_step_present = True
    last_step = (
        StepsName.INTEGRATION if vector_store_config is not None else StepsName.GRAPH
    )

    for step_name, processor_cls, config, factory in _pipeline_process_definitions(
        vector_store_config, request_data
    ):
        process_obj: BaseUtil = processor_cls(
            app=app_context.app, file_storage=app_context.storage.file_storage_dir
        )
        add_status_to_running_process(
            query_params.process_name,
            step_name,
            ProcessStatus.STARTED,
            app_context.processes.running,
        )

        if _is_omitted_step(query_params, step_name):
            _mark_step_skipped(app_context, query_params, step_name)
            if (
                process_obj.has_process(query_params.process_name)
                and previous_step_present
            ):
                _load_skipped_step(
                    app_context, query_params, step_name, vector_store_config
                )
            else:
                previous_step_present = False
            continue

        if process_obj.has_process(query_params.process_name):
            if _should_skip_present_step(query_params, step_name):
                _mark_step_skipped(app_context, query_params, step_name)
                if previous_step_present:
                    _load_skipped_step(
                        app_context, query_params, step_name, vector_store_config
                    )
                continue
            process_obj.delete_process(query_params.process_name)
            last_step = step_name
        else:
            last_step = step_name
            previous_step_present = False

        _configure_process_step(
            app_context,
            process_obj,
            step_name,
            config,
            query_params,
            request_data,
            vector_store_config,
        )
        processes_threading.append((process_obj, factory, step_name))

    return PreparedPipeline(
        processes_threading=processes_threading, last_step=last_step
    )


def _mark_step_skipped(app_context, query_params, step_name: str) -> None:
    """Record a skipped existing step as finished for process status reporting."""
    logging.info(
        "Skipping %s because %s.",
        step_name,
        (
            "omitted"
            if step_name in query_params.omitted_pipeline_steps
            else "skip_present"
        ),
    )
    add_status_to_running_process(
        query_params.process_name,
        step_name,
        ProcessStatus.FINISHED,
        app_context.processes.running,
    )


def _start_pipeline_thread(
    app_context, query_params, prepared_pipeline: PreparedPipeline
):
    """Create, store, and start the background pipeline thread."""
    pipeline_thread = StoppableThread(
        target_args=(
            app_context.app,
            prepared_pipeline.processes_threading,
            query_params.process_name,
            app_context.processes.running,
            app_context.processes.threads,
            app_context.pipeline.active_objects,
            prepared_pipeline.last_step,
        ),
        group=None,
        target=start_processes,
        name=None,
    )
    app_context.processes.threads[query_params.process_name] = pipeline_thread
    start_thread(
        app_context.app,
        query_params.process_name,
        pipeline_thread,
        app_context.processes.threads,
    )
    return pipeline_thread


def _pipeline_response(app_context, query_params, pipeline_thread: StoppableThread):
    """Build the HTTP response for a started or completed pipeline."""
    if query_params.return_statistics:
        pipeline_thread.join()
        graph_stats = graph_get_statistics(
            app=app_context.app,
            data=query_params.process_name,
            path=app_context.storage.file_storage_dir,
        )
        return (
            jsonify(name=query_params.process_name, **graph_stats),
            (
                int(HTTPResponses.OK)
                if "error" not in graph_stats
                else int(HTTPResponses.INTERNAL_SERVER_ERROR)
            ),
        )

    return (
        jsonify(
            name=query_params.process_name,
            status=app_context.processes.running.get(
                query_params.process_name, {"status": []}
            ).get("status"),
        ),
        int(HTTPResponses.ACCEPTED),
    )


def run_complete_pipeline(app, processes, pipeline, storage):
    """Handle the /pipeline endpoint by preparing and starting a pipeline run."""
    app_context = PipelineRouteContext(
        app=app, processes=processes, pipeline=pipeline, storage=storage
    )
    query_params = _default_query_params()
    try:
        content_type_json = request.headers.get("Content-Type") == "application/json"
        config_object_json: Optional[pipeline_json_config] = None
        if content_type_json:
            config_object_json = parse_pipeline_config_json(request.json)

        query_params = get_pipeline_query_params(
            app_context.app,
            request,
            app_context.processes.running,
            config_object_json,
        )
        if isinstance(query_params, tuple) and isinstance(
            query_params[0], flask.Response
        ):
            return query_params

        request_data, error_response = _parse_pipeline_request(
            app_context, query_params, config_object_json
        )
        if error_response is not None:
            return error_response

        vector_store_config = _normalize_vector_store_config(
            request_data.vector_store_config
        )
        if not request_data.data_upload:
            error_response = _load_data_from_document_server(
                app_context, query_params, request_data
            )
            if error_response is not None:
                return error_response

        prepared_pipeline = _prepare_pipeline_processes(
            app_context, query_params, request_data, vector_store_config
        )
        pipeline_thread = _start_pipeline_thread(
            app_context, query_params, prepared_pipeline
        )
        return _pipeline_response(app_context, query_params, pipeline_thread)
    except Exception as e:
        return (
            jsonify(
                name=query_params.process_name,
                error=str(e),
            ),
            int(HTTPResponses.INTERNAL_SERVER_ERROR),
        )
