import logging
import pathlib
import threading
import re
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum, IntEnum
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional, Any

from flask import jsonify, Response
from munch import Munch, unmunchify
from waiting import wait, TimeoutExpired

import flask
import spacy
import yaml
from dataclass_wizard import JSONWizard
from werkzeug.datastructures import FileStorage


class ProcessStatus(str, Enum):
    STARTED = "started"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"
    NOT_PRESENT = "not present"
    STOPPED = "stopped"


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
    ALL = [DATA, EMBEDDING, CLUSTERING, GRAPH]


@dataclass
class NegspacyConfig(JSONWizard):
    chunk_prefix: str | list[str] | None = None
    neg_termset_file: str | None = None
    scope: int | None = None
    language: str | None = None
    feat_of_interest: str | None = None


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition.
    From: https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread"""

    def __init__(self, group, target, name, target_args):
        super().__init__(args=target_args, group=group, target=target, name=name)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class BaseUtil(ABC):
    def __init__(
            self,
            app: flask.app.Flask,
            file_storage: str,
            step_name: str
    ):
        self._app = app
        self._file_storage = pathlib.Path(file_storage)
        self._process_step = step_name
        self._process_name = None
        self.config = None
        self.serializable_config = None

    def _complete_pickle_path(
            self,
            process: Optional[str]
    ) -> pathlib.Path:
        return Path(self._file_storage / (process if process is not None else "") /
                    f"{self.process_name if process is None else process}_{self.process_step}.pickle")

    @property
    @abstractmethod
    def default_config(self) -> dict:
        raise NotImplementedError()

    @property
    @abstractmethod
    def sub_config_names(self) -> list[str]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def necessary_config_keys(self) -> list[str]:
        raise NotImplementedError()

    @property
    def process_name(self) -> Optional[str]:
        return self._process_name

    @process_name.setter
    def process_name(
            self,
            name: str
    ) -> None:
        self._process_name = name

    @property
    def process_step(self) -> str:
        return self._process_step

    @property
    def file_storage_path(self) -> pathlib.Path:
        return self._file_storage

    @file_storage_path.setter
    def file_storage_path(
            self,
            sub_path: Union[str, pathlib.Path]
    ) -> None:
        self._file_storage = Path(self._file_storage / sub_path)
        self._file_storage.mkdir(exist_ok=True)  # ToDo: warning when folder exists

    @abstractmethod
    def has_process(
            self,
            process: Optional[str] = None
    ) -> bool:
        return self._complete_pickle_path(process).exists()

    @abstractmethod
    def delete_process(
            self,
            process: Optional[str] = None
    ) -> None:
        if self.has_process(process):
            self._complete_pickle_path(process).unlink()

    @abstractmethod
    def read_config(
            self,
            config: Optional[Union[FileStorage, dict]],
            process_name=None,
            language=None
    ) -> Optional[Response]:
        base_config = self.default_config
        is_default_config = True
        if isinstance(config, dict):
            if isinstance(config, Munch):
                _config = unmunchify(config)
            else:
                _config = config
            for _type in self.sub_config_names:
                _sub_config = _config.get(_type, {}).copy()
                for k, v in _sub_config.items():
                    _config[f"{_type}_{k}"] = v
                _config.pop(_type, None)
            base_config = _config
            is_default_config = False
        elif isinstance(config, FileStorage):
            try:
                base_config = yaml.safe_load(config.stream)
                is_default_config = False
            except Exception as e:
                self._app.logger.error(f"Couldn't read config file: {e}")
                return jsonify("Encountered error. See log.")
        else:
            is_default_config = True
            self._app.logger.info("No config file provided; using default values")

        if not is_default_config:
            _inter = set(base_config.keys()).intersection(self.necessary_config_keys)
            if not len(_inter) == len(self.necessary_config_keys):
                raise KeyError(f"Missing necessary config values: '{_inter}'.")
        base_config["corpus_name"] = process_name.lower() if process_name is not None else base_config[
            "corpus_name"].lower()
        self.config = base_config
        return None

    @abstractmethod
    def read_stored_config(
            self,
            ext: str = "yaml"
    ) -> tuple[str, dict]:
        _sub_configs = {k: {} for k in self.sub_config_names}
        _file_name = f"{self.process_name}_{self.process_step}_config.{ext}"
        _file = Path(self._file_storage / _file_name)
        if not _file.exists():
            return self.process_step, {}
        config_yaml = yaml.safe_load(_file.open('rb'))
        for key, value in config_yaml.copy().items():
            _sub_key_split = key.split("_")
            if len(_sub_key_split) > 1 and _sub_key_split[0] in _sub_configs.keys():
                _sub_configs[_sub_key_split[0]]["_".join(_sub_key_split[1:])] = value
                config_yaml.pop(key)
        config_yaml.update(_sub_configs)
        return self.process_step, config_yaml

    @abstractmethod
    def start_process(
            self,
            cache_name: str,
            process_factory,
            process_tracker: dict
    ) -> Optional[Any]:
        raise NotImplementedError()

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


def string_conformity(s: str):
    return re.sub(r"\s+", "_", s.lower())


def load_spacy_model(spacy_model: str, logger: logging.Logger, default_model: str):
    def is_valid_spacy_model(model: str):
        from spacy.cli.download import get_compatibility
        if model in get_compatibility():
            return True
        logger.error(f"'{model}' is not a valid model name.")
        return False


    def wait_for_download(model: str, time_out: int = 30):
        spacy.cli.download(model)
        wait_pred = lambda: model in spacy.util.get_installed_models()
        try:
            wait(wait_pred, timeout_seconds=time_out)
        except TimeoutExpired:
            logger.warning(f"TimeOut while waiting >{time_out} seconds for download to finish."
                           f" Hopefully this is just due to installed models not refreshing.")
    spacy_language = None
    try:
        spacy_language = spacy.load(spacy_model)
    except IOError as e:
        if spacy_model != default_model:
            if is_valid_spacy_model(spacy_model):
                logger.info(f"Model '{spacy_model}' doesn't seem to be installed; trying to download model.")
                wait_for_download(spacy_model)
                spacy_language = spacy.load(spacy_model)
            else:
                logger.error(f"{e}\nUsing default model {default_model}.")
                try:
                    spacy_language = spacy.load(default_model)
                except IOError as e:
                    logger.error(f"{e}\ntrying to download default model {default_model}.")
                    wait_for_download(default_model)
                    spacy_language = spacy.load(default_model)
        else:
            logger.error(f"{e}\ntrying to download default model {default_model}.")
            wait_for_download(default_model)
            spacy_language = spacy.load(default_model)
    return spacy_language

def get_default_spacy_model():
    return "en_core_web_trf"