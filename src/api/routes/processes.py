"""Routes for process status, stopping, and deletion."""

from typing import Optional

from flask import Response, jsonify, request

from main_methods import delete_pipeline, stop_thread
from main_utils import HTTPResponses, ProcessStatus, StoppableThread, string_conformity


def register_process_routes(main_objects):
    """Register process listing, status, deletion, and stop routes."""

    @main_objects.app.route("/processes", methods=["GET"])
    def get_all_processes_api():
        if len(main_objects.running_processes) > 0:
            return jsonify(
                processes=[p for p in main_objects.running_processes.values()]
            )
        return jsonify("No saved processes."), int(HTTPResponses.NOT_FOUND)

    @main_objects.app.route("/processes/<process_id>/delete", methods=["DELETE"])
    def delete_process(process_id):
        hard_stop = request.args.get("hard_stop", False)
        process_id = string_conformity(process_id)
        if process_id not in set(main_objects.running_processes.keys()).union(
            main_objects.current_active_pipeline_objects.keys()
        ):
            return Response(
                f"There is no such process '{process_id}'.\n",
                status=int(HTTPResponses.NOT_FOUND),
            )
        to_stop = None
        if any(
            [
                step.get("status") in [ProcessStatus.RUNNING, ProcessStatus.STARTED]
                for step in main_objects.running_processes.get(process_id).get(
                    "status", []
                )
            ]
        ):
            to_stop: Optional[StoppableThread]
            if to_stop := main_objects.pipeline_threads_store.get(process_id, None):
                stop_thread(
                    app=main_objects.app,
                    process_name=process_id,
                    threading_store=main_objects.pipeline_threads_store,
                    process_tracker=main_objects.running_processes,
                    hard_stop=hard_stop,
                )
        delete_thread = StoppableThread(
            target_args=(
                main_objects.app,
                main_objects.file_storage_dir,
                process_id,
                main_objects.running_processes,
                main_objects.current_active_pipeline_objects,
                to_stop,
            ),
            group=None,
            target=delete_pipeline,
            name=None,
        )
        delete_thread.start()
        return Response(
            f"Process '{process_id}' set to be deleted.", status=HTTPResponses.OK
        )

    @main_objects.app.route("/processes/<process_id>/stop", methods=["GET"])
    def stop_pipeline(process_id):
        if request.method == "GET":
            hard_stop = request.args.get("hard_stop", False)
            process_id = string_conformity(process_id)
            return stop_thread(
                app=main_objects.app,
                process_name=process_id,
                threading_store=main_objects.pipeline_threads_store,
                process_tracker=main_objects.running_processes,
                hard_stop=hard_stop,
            )
        return jsonify(f"Method not supported: {request.method}")

    @main_objects.app.route("/status", methods=["GET"])
    def get_status_of():
        process = string_conformity(request.args.get("process", "default"))
        if process is not None:
            response = main_objects.running_processes.get(process, None)
            if response is not None:
                return jsonify(response), int(HTTPResponses.OK)
        return jsonify(
            name=process, error=f"No such (running) process: '{process}'"
        ), int(HTTPResponses.NOT_FOUND)
