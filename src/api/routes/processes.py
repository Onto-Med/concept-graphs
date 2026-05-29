"""Routes for process status, stopping, and deletion."""

from typing import Optional

from flask import Blueprint, Response, jsonify, request

from src.api.responses import HTTPResponses
from src.api.services.process_management import delete_pipeline
from src.common.parsing import string_conformity
from src.common.threads import StoppableThread
from src.pipeline.processes import stop_thread
from src.pipeline.status import ProcessStatus


def create_process_blueprint(app, processes, pipeline, storage):
    """Create the blueprint for process listing, status, deletion, and stop routes."""
    blueprint = Blueprint("process_routes", __name__)

    @blueprint.route("/processes", methods=["GET"])
    def get_all_processes_api():
        if len(processes.running) > 0:
            return jsonify(processes=[p for p in processes.running.values()])
        return jsonify("No saved processes."), int(HTTPResponses.NOT_FOUND)

    @blueprint.route("/processes/<process_id>/delete", methods=["DELETE"])
    def delete_process(process_id):
        hard_stop = request.args.get("hard_stop", False)
        process_id = string_conformity(process_id)
        if process_id not in set(processes.running.keys()).union(
            pipeline.active_objects.keys()
        ):
            return Response(
                f"There is no such process '{process_id}'.\n",
                status=int(HTTPResponses.NOT_FOUND),
            )
        to_stop = None
        if any(
            [
                step.get("status") in [ProcessStatus.RUNNING, ProcessStatus.STARTED]
                for step in processes.running.get(process_id).get("status", [])
            ]
        ):
            to_stop: Optional[StoppableThread]
            if to_stop := processes.threads.get(process_id, None):
                stop_thread(
                    app=app,
                    process_name=process_id,
                    threading_store=processes.threads,
                    process_tracker=processes.running,
                    hard_stop=hard_stop,
                )
        delete_thread = StoppableThread(
            target_args=(
                app,
                storage.file_storage_dir,
                process_id,
                processes.running,
                pipeline.active_objects,
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

    @blueprint.route("/processes/<process_id>/stop", methods=["GET"])
    def stop_pipeline(process_id):
        if request.method == "GET":
            hard_stop = request.args.get("hard_stop", False)
            process_id = string_conformity(process_id)
            return stop_thread(
                app=app,
                process_name=process_id,
                threading_store=processes.threads,
                process_tracker=processes.running,
                hard_stop=hard_stop,
            )
        return jsonify(f"Method not supported: {request.method}")

    @blueprint.route("/status", methods=["GET"])
    def get_status_of():
        process = string_conformity(request.args.get("process", "default"))
        if process is not None:
            response = processes.running.get(process, None)
            if response is not None:
                return jsonify(response), int(HTTPResponses.OK)
        return jsonify(
            name=process, error=f"No such (running) process: '{process}'"
        ), int(HTTPResponses.NOT_FOUND)

    return blueprint
