"""Data containers for pipeline route orchestration."""

from dataclasses import dataclass
from typing import Optional

import flask

from src.pipeline.base import BaseUtil
from src.pipeline.status import pipeline_query_params

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
    """Parsed request inputs needed to configure a pipeline run."""

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
    """Configured pipeline processors ready for background execution."""

    processes_threading: list[tuple[BaseUtil, object, str]]
    last_step: str


def default_query_params() -> pipeline_query_params:
    """Return safe default query parameters used before request parsing completes."""
    return pipeline_query_params(
        process_name="not set",
        language="en",
        skip_present=True,
        omitted_pipeline_steps=[],
        return_statistics=False,
    )
