"""Request parsing and temporary upload handling for the pipeline route."""

import pathlib

import yaml
from flask import jsonify, request
from werkzeug.datastructures import FileStorage

from src.api.pipeline_support.models import DEFAULT_VECTOR_STORE, PipelineRequestData
from src.api.request_parsing import pipeline_json_config
from src.api.responses import HTTPResponses
from src.pipeline.status import StepsName, pipeline_query_params


def temporary_upload_path(
    file_storage_dir: pathlib.Path, upload: FileStorage
) -> pathlib.Path:
    """Persist an uploaded file in the temporary stream directory and return its path."""
    target = file_storage_dir / ".tmp_streams" / upload.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    upload.save(target)
    return target


def parse_json_pipeline_request(
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


def parse_multipart_pipeline_request(app_context, query_params: pipeline_query_params):
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
        data = temporary_upload_path(app_context.storage.file_storage_dir, data)
        data_upload = True

    labels = request.files.get("labels", None)
    if labels is not None:
        labels = temporary_upload_path(app_context.storage.file_storage_dir, labels)

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


def parse_pipeline_request(app_context, query_params, config_object_json):
    """Dispatch request parsing based on the incoming content type."""
    content_type_json = request.headers.get("Content-Type") == "application/json"
    if content_type_json:
        return parse_json_pipeline_request(config_object_json, query_params)
    return parse_multipart_pipeline_request(app_context, query_params)
