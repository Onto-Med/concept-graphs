"""Pipeline step preparation helpers for the /pipeline route."""

import logging
import pathlib
from typing import Optional

from src.api.pipeline_support.models import PipelineRequestData, PreparedPipeline
from src.api.services.configuration import read_config
from src.core import (
    cluster_functions,
    data_functions,
    embedding_functions,
    integration_functions,
)
from src.pipeline.base import BaseUtil
from src.pipeline.load_utils import FactoryLoader
from src.pipeline.status import (
    ProcessStatus,
    StepsName,
    add_status_to_running_process,
    pipeline_query_params,
)
from src.pipeline.steps.clustering_util import ClusteringUtil
from src.pipeline.steps.embedding_util import PhraseEmbeddingUtil
from src.pipeline.steps.graph_creation_util import GraphCreationUtil
from src.pipeline.steps.integration_util import ConceptGraphIntegrationUtil
from src.pipeline.steps.preprocessing_util import PreprocessingUtil


def pipeline_process_definitions(vector_store_config: Optional[dict], request_data):
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


def load_skipped_step(app_context, query_params, step_name: str, vector_store_config):
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


def is_omitted_step(query_params, step_name: str) -> bool:
    """Return whether a step was explicitly omitted by the request."""
    return step_name in query_params.omitted_pipeline_steps


def should_skip_present_step(query_params, step_name: str) -> bool:
    """Return whether an existing step should be reused instead of recomputed."""
    return query_params.skip_present


def configure_process_step(
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
        process_obj.storage_method = embedding_storage_method(
            process_obj, vector_store_config
        )
    elif step_name == StepsName.INTEGRATION:
        process_obj.config["check_for_reasonable_result"] = True


def embedding_storage_method(process_obj, vector_store_config: Optional[dict]):
    """Choose pickle or vector-store backing for phrase embeddings."""
    if vector_store_config is None:
        return "pickle", None
    if process_obj.storage_method == "vectorstore":
        return "vectorstore", vector_store_config
    return "pickle", None


def prepare_pipeline_processes(
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

    for step_name, processor_cls, config, factory in pipeline_process_definitions(
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

        if is_omitted_step(query_params, step_name):
            mark_step_skipped(app_context, query_params, step_name)
            if (
                process_obj.has_process(query_params.process_name)
                and previous_step_present
            ):
                load_skipped_step(
                    app_context, query_params, step_name, vector_store_config
                )
            else:
                previous_step_present = False
            continue

        if process_obj.has_process(query_params.process_name):
            if should_skip_present_step(query_params, step_name):
                mark_step_skipped(app_context, query_params, step_name)
                if previous_step_present:
                    load_skipped_step(
                        app_context, query_params, step_name, vector_store_config
                    )
                continue
            process_obj.delete_process(query_params.process_name)
            last_step = step_name
        else:
            last_step = step_name
            previous_step_present = False

        configure_process_step(
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


def mark_step_skipped(app_context, query_params, step_name: str) -> None:
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
