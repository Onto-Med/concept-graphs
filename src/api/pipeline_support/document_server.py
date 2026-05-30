"""Document-server loading helpers for the pipeline route."""

from flask import jsonify

from src.api.pipeline_support.models import PipelineRequestData
from src.api.responses import HTTPResponses
from src.api.services.document_server import (
    check_data_server,
    get_data_server_config,
    get_documents_from_es_server,
)
from src.pipeline.status import pipeline_query_params


def load_data_from_document_server(
    app_context,
    query_params: pipeline_query_params,
    request_data: PipelineRequestData,
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
