"""Pipeline query parameter service helpers."""

import flask
from flask import jsonify

from src.api.request_parsing import pipeline_json_config
from src.api.responses import HTTPResponses
from src.common.parsing import (
    get_bool_expression,
    string_conformity,
)
from src.pipeline.status import (
    PipelineLanguage,
    ProcessStatus,
    pipeline_query_params,
)


def get_pipeline_query_params(
    app: flask.Flask,
    flask_request: flask.Request,
    running_processes: dict,
    config_obj_json: pipeline_json_config,
) -> pipeline_query_params | tuple:
    if config_obj_json is not None and config_obj_json.name is not None:
        corpus = string_conformity(config_obj_json.name)
    else:
        corpus = string_conformity(flask_request.args.get("process", "default"))
    if corpus_status := running_processes.get(corpus, False):
        if any(
            [
                v.get("status", None) == ProcessStatus.RUNNING
                for v in corpus_status.get("status", [])
            ]
        ):
            return jsonify(
                name=corpus,
                error=f"A process is currently running for this corpus. Use '/status?process={corpus}' for specifics.",
            ), int(HTTPResponses.FORBIDDEN)
    app.logger.info(f"Using process name '{corpus}'")
    if config_obj_json is not None and config_obj_json.language is not None:
        language = PipelineLanguage.language_from_string(config_obj_json.language)
    else:
        language = PipelineLanguage.language_from_string(
            str(flask_request.args.get("lang", "en"))
        )
    app.logger.info(
        f"Using preset language settings for '{language}' where specific configuration is not provided."
    )

    skip_present = flask_request.args.get("skip_present", True)
    if isinstance(skip_present, str):
        skip_present = get_bool_expression(skip_present, True)
    if skip_present:
        app.logger.info("Skipping present saved steps")

    skip_steps = flask_request.args.get("skip_steps", False)
    omit_pipeline_steps = []
    if skip_steps:
        omit_pipeline_steps = get_omit_pipeline_steps(skip_steps)

    return_statistics = flask_request.args.get("return_statistics", False)
    if isinstance(return_statistics, str):
        return_statistics = get_bool_expression(return_statistics, True)

    return pipeline_query_params(
        corpus, language, skip_present, omit_pipeline_steps, return_statistics
    )


def get_dict_expression(dict_str: str):
    if isinstance(dict_str, str):
        # e.g. "{'text': 'content'}"
        if not dict_str.startswith("{") and not dict_str.endswith("}"):
            return dict_str
        _str = dict_str[1:-1].split(",")
        _return_dict = dict()
        for _s in _str:
            _split_s = _s.split(":")
            if len(_split_s) != 2:
                break
            _return_dict[_split_s[0].strip().strip("'").strip('"')] = (
                _split_s[1].strip().strip("'").strip('"')
            )

        return _return_dict
    else:
        return dict_str


def get_query_param_help_text(param: str):
    return {
        "process": "process: name of the process (e.g. ``corpus_name`` in config); if not provided, uses 'default'",
        "exclusion_ids": "exclusion_ids: list of concept ids (as returned by e.g. ``/clustering/concepts``), "
        "that shall be excluded from the final graphs in the form of ``[ID1, ID2, etc.]``",
        "draw": "draw: `true` or `false` - whether the response shall be a rendered graph or plain json",
    }.get(param, "No help text available.")


def get_omit_pipeline_steps(steps: object) -> list[str]:
    step_set = {"data", "embedding", "clustering", "graph", "integration"}
    if isinstance(steps, str):
        steps = steps.strip("([{}])")
        return [s.lower() for s in steps.split(",") if s.lower() in step_set]
    return []
