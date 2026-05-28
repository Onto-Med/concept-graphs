"""Routes for adding documents to existing concept graphs."""

from flask import jsonify, request

from main_methods import (
    add_documents_to_concept_graphs,
    parse_document_adding_json,
    start_thread,
)
from main_utils import HTTPResponses, StepsName, StoppableThread, string_conformity


def register_graph_document_routes(app_context):
    """Register document addition routes for concept graphs."""

    @app_context.app.route("/graph/document/<path_arg>", methods=["POST", "DELETE"])
    def graph_document(path_arg=None):
        process = string_conformity(request.args.get("process", "default"))
        method = request.method
        if request.headers.get("Content-Type") == "application/json":
            content_json = parse_document_adding_json(request.get_json())
            if content_json is None:
                return (
                    jsonify(error="Could not parse json provided in request."),
                    HTTPResponses.BAD_REQUEST,
                )
        else:
            return (
                jsonify(error="Only json request body is supported."),
                HTTPResponses.NOT_IMPLEMENTED,
            )

        if method == "POST" and path_arg is not None and path_arg.lower() == "add":
            data_proc = app_context.current_active_pipeline_objects.get(
                process, {}
            ).get(StepsName.DATA, None)
            emb_proc = app_context.current_active_pipeline_objects.get(
                process, {}
            ).get(StepsName.EMBEDDING, None)
            graph_proc = app_context.current_active_pipeline_objects.get(
                process, {}
            ).get(StepsName.GRAPH, None)
            path_base = app_context.file_storage_dir / process
            document_adding_thread = StoppableThread(
                target_args=(content_json,),
                target_kwargs={
                    "data_processing": data_proc,
                    "embedding_processing": emb_proc,
                    "graph_processing": graph_proc,
                    "storage_path": path_base,
                    "process_name": process,
                },
                group=None,
                target=add_documents_to_concept_graphs,
                name=None,
            )
            app_context.pipeline_threads_store[f"document_addition_{process}"] = (
                document_adding_thread
            )
            start_thread(
                app_context.app,
                f"document_addition_{process}",
                document_adding_thread,
                None,
            )
            return (
                jsonify(f"Started thread for adding documents for process {process}."),
                HTTPResponses.OK,
            )
        if method == "DELETE" and path_arg is not None:
            return (
                jsonify(error="'Delete' not implemented."),
                HTTPResponses.NOT_IMPLEMENTED,
            )

        err_msg = f"Either method 'POST' or 'DELETE' expected, but '{method.upper()}' was given instead."
        if path_arg is None:
            if method == "DELETE":
                err_msg = "No path argument provided for 'DELETE' method; this method needs an id."
            elif method == "POST":
                err_msg = "Please use 'add' as path argument for 'POST' method."
        return jsonify(error=err_msg), HTTPResponses.BAD_REQUEST

    @app_context.app.route("/graph/document/add/status", methods=["GET"])
    def graph_document_status():
        process = string_conformity(request.args.get("process", "default"))
        thread_id = f"document_addition_{process}"
        if thread_id not in app_context.pipeline_threads_store:
            return (
                jsonify(
                    error=f"No document addition thread (running or completed) for '{process}' found."
                ),
                HTTPResponses.NOT_FOUND,
            )
        if return_value := app_context.pipeline_threads_store.get(
            thread_id
        ).return_value:
            return jsonify(return_value[0]), return_value[1]
        return (
            jsonify(
                f"Document addition thread for '{process}' seems to be still running."
            ),
            HTTPResponses.ACCEPTED,
        )
