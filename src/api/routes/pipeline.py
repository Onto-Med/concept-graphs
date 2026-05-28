"""Routes for starting pipelines and retrieving pipeline configuration."""

import json
import logging
import pathlib

from flask import Blueprint, jsonify, request

from main_methods import load_configs
from main_utils import (
    HTTPResponses,
    PipelineLanguage,
    get_bool_expression,
    string_conformity,
)
from src.api.pipeline import run_complete_pipeline


def create_pipeline_blueprint(app_context):
    """Create the blueprint for pipeline execution and configuration routes."""
    blueprint = Blueprint("pipeline_routes", __name__)

    @blueprint.route("/pipeline", methods=["POST"])
    def complete_pipeline():
        return run_complete_pipeline(app_context)

    @blueprint.route("/pipeline/configuration", methods=["GET"])
    def get_pipeline_default_configuration():
        if request.method != "GET":
            return HTTPResponses.BAD_REQUEST

        is_default_conf = get_bool_expression(request.args.get("default", True))
        process = string_conformity(request.args.get("process", "default"))
        language = PipelineLanguage.language_from_string(
            request.args.get("language", "en")
        )
        if is_default_conf:
            default_conf = pathlib.Path(f"./conf/pipeline-config_{language}.json")
            if default_conf.exists() and default_conf.is_file():
                try:
                    return jsonify(**json.load(default_conf.open("rb"))), int(
                        HTTPResponses.OK
                    )
                except Exception as e:
                    logging.error(e)
            return jsonify(message="Couldn't find/read default configuration."), int(
                HTTPResponses.NOT_FOUND
            )

        logging.info("Returning configuration for '%s' pipeline.", process)
        try:
            config = load_configs(
                app=app_context.app,
                process_name=process,
                path_to_configs=app_context.storage.file_storage_dir,
            )
            return (
                jsonify(
                    name=process,
                    language=config.get("language", "en"),
                    config=config.get("config", {}),
                ),
                int(HTTPResponses.OK),
            )
        except Exception as e:
            logging.error(e)
        return jsonify(
            message=f"Couldn't find/read configuration for '{process}'."
        ), int(HTTPResponses.NOT_FOUND)

    return blueprint
