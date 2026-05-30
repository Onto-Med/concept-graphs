"""Routes for starting pipelines and retrieving pipeline configuration."""

import json
import logging
import pathlib

from flask import Blueprint, jsonify, request

from src.api.pipeline import run_complete_pipeline
from src.api.responses import HTTPResponses
from src.api.services.configuration import load_configs
from src.common.parsing import (
    get_bool_expression,
    string_conformity,
)
from src.pipeline.status import PipelineLanguage


def create_pipeline_blueprint(app, processes, pipeline, storage):
    """Create the blueprint for pipeline execution and configuration routes."""
    blueprint = Blueprint("pipeline_routes", __name__)

    @blueprint.route("/pipeline", methods=["POST"])
    def complete_pipeline():
        return run_complete_pipeline(app, processes, pipeline, storage)

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
                except (OSError, json.JSONDecodeError, TypeError) as e:
                    logging.warning(
                        "Couldn't read default pipeline configuration '%s': %s",
                        default_conf,
                        e,
                    )
            return jsonify(message="Couldn't find/read default configuration."), int(
                HTTPResponses.NOT_FOUND
            )

        logging.info("Returning configuration for '%s' pipeline.", process)
        try:
            config = load_configs(
                app=app,
                process_name=process,
                path_to_configs=storage.file_storage_dir,
            )
            return (
                jsonify(
                    name=process,
                    language=config.get("language", "en"),
                    config=config.get("config", {}),
                ),
                int(HTTPResponses.OK),
            )
        except (OSError, KeyError, TypeError, AttributeError) as e:
            logging.warning("Couldn't read configuration for '%s': %s", process, e)
        return jsonify(
            message=f"Couldn't find/read configuration for '{process}'."
        ), int(HTTPResponses.NOT_FOUND)

    return blueprint
