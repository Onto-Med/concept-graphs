"""Routes for starting pipelines and retrieving pipeline configuration."""

import json
import logging
import pathlib

from flask import jsonify, request

from main_methods import load_configs
from main_utils import (
    HTTPResponses,
    PipelineLanguage,
    get_bool_expression,
    string_conformity,
)
from src.api.pipeline import run_complete_pipeline


def register_pipeline_routes(main_objects):
    """Register pipeline execution and configuration routes."""

    @main_objects.app.route("/pipeline", methods=["POST"])
    def complete_pipeline():
        return run_complete_pipeline(main_objects)

    @main_objects.app.route("/pipeline/configuration", methods=["GET"])
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
                app=main_objects.app,
                process_name=process,
                path_to_configs=main_objects.file_storage_dir,
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
