"""Routes for external service and RAG status checks."""

from flask import Blueprint, jsonify, request

from main_methods import check_data_server, get_data_server_config
from main_utils import HTTPResponses, string_conformity


def create_status_blueprint(app, rag):
    """Create the blueprint for document-server and RAG status routes."""
    blueprint = Blueprint("status_routes", __name__)

    @blueprint.route("/status/document-server", methods=["POST", "GET"])
    def get_data_server():
        if request.method == "POST":
            if request.headers.get("Content-Type") == "application/json":
                document_server_config = request.json
            else:
                document_server_config = request.files.get(
                    "document_server_config", False
                )
            if not document_server_config:
                return jsonify(
                    name="document server check",
                    status="No document server config file provided",
                ), int(HTTPResponses.BAD_REQUEST)
            base_config = get_data_server_config(document_server_config, app)
            if not check_data_server(base_config):
                return (
                    jsonify(
                        f"There is no data server at the specified location ({base_config}) or its index '{base_config['index']}' contains no data."
                    ),
                    int(HTTPResponses.NOT_FOUND),
                )
            return (
                jsonify(
                    f"Data server reachable under: '{base_config['url']}:{base_config['port']}' with index '{base_config['index']}'"
                ),
                int(HTTPResponses.OK),
            )
        if request.method == "GET":
            return jsonify("Method 'GET' not yet implemented.")
        return jsonify(f"Method not supported: '{request.method}'.")

    @blueprint.route("/status/rag", methods=["GET"])
    def get_rag_status():
        process = string_conformity(request.args.get("process", "default"))
        has_rag = rag.active is not None
        if process is not None and has_rag and rag.active.process == process:
            return jsonify(active=rag.active.ready, name=process, error=None), int(
                HTTPResponses.OK
            )
        err_string = (
            f"RAG is active but it seems for the process: '{rag.active.process}'"
            if has_rag
            else "The RAG component is not initialized."
        )
        return jsonify(active=False, name=process, error=err_string), int(
            HTTPResponses.NOT_FOUND
        )

    return blueprint
