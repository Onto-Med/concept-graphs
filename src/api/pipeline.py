"""Pipeline route implementation.

This module keeps the public /pipeline route orchestration compact and delegates
request parsing, step preparation, vector-store handling, and response creation
to focused helpers under ``src.api.pipeline_support``.
"""

import flask
from flask import jsonify, request

from src.api.pipeline_support.document_server import load_data_from_document_server
from src.api.pipeline_support.execution import pipeline_response, start_pipeline_thread
from src.api.pipeline_support.models import PipelineRouteContext, default_query_params
from src.api.pipeline_support.request_data import parse_pipeline_request
from src.api.pipeline_support.steps import prepare_pipeline_processes
from src.api.pipeline_support.vectorstore import normalize_vector_store_config
from src.api.request_parsing import parse_pipeline_config_json, pipeline_json_config
from src.api.responses import HTTPResponses
from src.api.services.pipeline_params import get_pipeline_query_params


def run_complete_pipeline(app, processes, pipeline, storage):
    """Handle the /pipeline endpoint by preparing and starting a pipeline run."""
    app_context = PipelineRouteContext(
        app=app, processes=processes, pipeline=pipeline, storage=storage
    )
    query_params = default_query_params()
    try:
        content_type_json = request.headers.get("Content-Type") == "application/json"
        config_object_json: pipeline_json_config | None = None
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

        request_data, error_response = parse_pipeline_request(
            app_context, query_params, config_object_json
        )
        if error_response is not None:
            return error_response

        vector_store_config = normalize_vector_store_config(
            request_data.vector_store_config
        )
        if not request_data.data_upload:
            error_response = load_data_from_document_server(
                app_context, query_params, request_data
            )
            if error_response is not None:
                return error_response

        prepared_pipeline = prepare_pipeline_processes(
            app_context, query_params, request_data, vector_store_config
        )
        pipeline_thread = start_pipeline_thread(
            app_context, query_params, prepared_pipeline
        )
        return pipeline_response(app_context, query_params, pipeline_thread)
    except Exception as e:
        app_context.app.logger.exception(
            "Unhandled error while starting pipeline '%s'.", query_params.process_name
        )
        return (
            jsonify(
                name=query_params.process_name,
                error=str(e),
            ),
            int(HTTPResponses.INTERNAL_SERVER_ERROR),
        )
