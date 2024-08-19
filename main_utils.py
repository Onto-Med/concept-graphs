import pathlib
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum, IntEnum
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import flask
import yaml
from dataclass_wizard import JSONWizard


@dataclass
class NegspacyConfig(JSONWizard):
    chunk_prefix: str | list[str] | None = None
    neg_termset_file: str | None = None
    scope: int | None = None
    language: str | None = None
    feat_of_interest: str | None = None


class BaseUtil(ABC):
    def __init__(self, app: flask.app.Flask, file_storage: str, process_name: str, step_name: str):
        self._app = app
        self._file_storage = pathlib.Path(file_storage)
        self._process_step = step_name
        self._process_name = process_name
        self._base_config = None
        self._final_config = None

    @property
    @abstractmethod
    def base_config(self):
        return self._base_config

    @base_config.setter
    def base_config(self, config):
        self._base_config = config

    @property
    @abstractmethod
    def file_storage(self):
        return self._file_storage

    @file_storage.setter
    def file_storage(self, file_path):
        self._file_storage = file_path

    @property
    @abstractmethod
    def app(self):
        return self._app

    @app.setter
    def app(self, app_instance):
        self._app = app_instance

    @abstractmethod
    def read_config(self, config):
        if config is None:
            self.app.logger.info("No config file provided; using default values")
            self._final_config = self.base_config
        else:
            try:
                loaded_config = yaml.safe_load(config.stream)
                with Path(Path(self.file_storage) / loaded_config.get("corpus_name", "default") / "_config.yaml"
                          ).open('w') as config_save:
                    yaml.safe_dump(loaded_config, config_save)
            except Exception as e:
                self.app.logger.error(f"Couldn't read config file: {e}")
            self._final_config = loaded_config


class ProcessStatus(str, Enum):
    STARTED = "started"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"
    NOT_PRESENT = "not present"


class HTTPResponses(IntEnum):
    OK = 200
    ACCEPTED = 202
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    SERVICE_UNAVAILABLE = 503


class StepsName:
    DATA = "data"
    EMBEDDING = "embedding"
    CLUSTERING = "clustering"
    GRAPH = "graph"


pipeline_query_params = namedtuple(
    "PipelineQueryParams", ["process_name", "language", "skip_present", "omitted_pipeline_steps", "return_statistics"])

steps_relation_dict = {
    StepsName.DATA: 1,
    StepsName.EMBEDDING: 2,
    StepsName.CLUSTERING: 3,
    StepsName.GRAPH: 4
}


class PipelineLanguage:
    language_map = {
        "en": "en",
        "english": "en",
        "englisch": "en",
        "de": "de",
        "german": "de",
        "deutsch": "de"
    }

    @staticmethod
    def language_from_string(lang):
        return PipelineLanguage.language_map.get(lang.lower(), "en")


def add_status_to_running_process(
        process_name: str,
        step_name: StepsName,
        step_status: ProcessStatus,
        running_processes: dict
):
    _step = {
        "name": step_name,
        "rank": steps_relation_dict[step_name],
        "status": step_status
    }
    _remove = -1
    if not running_processes.get(process_name, False):
        running_processes[process_name] = {
            "name": process_name,
            "status": [],
        }
    else:
        for _i, _status in enumerate(running_processes[process_name]["status"]):
            if _status.get("name", False) == step_name:
                _remove = _i
                break
        if _remove >= 0:
            running_processes[process_name]["status"].pop(_remove)

    if _remove >= 0:
        running_processes[process_name]["status"].insert(_remove, _step)
    else:
        running_processes[process_name]["status"].append(_step)
    return running_processes


def get_bool_expression(str_bool: str, default: Union[bool, str] = False) -> bool:
    if isinstance(str_bool, bool):
        return str_bool
    elif isinstance(str_bool, str):
        return {
            'true': True, 'yes': True, 'y': True, 'ja': True, 'j': True,
            'false': False, 'no': False, 'n': False, 'nein': False,
        }.get(str_bool.lower(), default)
    else:
        return False
