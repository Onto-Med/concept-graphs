"""Thread startup and response helpers for pipeline runs."""

from flask import jsonify

from src.api.responses import HTTPResponses
from src.api.services.artifact_responses import graph_get_statistics
from src.common.threads import StoppableThread
from src.pipeline.processes import start_processes, start_thread


def start_pipeline_thread(app_context, query_params, prepared_pipeline):
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


def pipeline_response(app_context, query_params, pipeline_thread: StoppableThread):
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
