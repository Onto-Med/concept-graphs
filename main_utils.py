import logging
import pathlib
import re
import threading
from abc import ABC, abstractmethod
from collections import namedtuple, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, IntEnum
from inspect import getfullargspec
from pathlib import Path
from typing import Union, Optional, Any, Callable, Tuple, Iterator

import flask
import spacy
import yaml
from dataclass_wizard import JSONWizard
from flask import jsonify, Response
from munch import Munch, unmunchify
from waiting import wait, TimeoutExpired
from werkzeug.datastructures import FileStorage

from src.rag.embedding_stores.AbstractEmbeddingStore import ChunkEmbeddingStore
from src.rag.rag import RAG


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
    INTEGRATION = "integration"
    ALL = [DATA, EMBEDDING, CLUSTERING, GRAPH, INTEGRATION]


@dataclass
class ActiveRAG:
    rag: RAG
    vectorstore: ChunkEmbeddingStore
    process: str
    ready: bool = False
    initializing: bool = False

    def switch_readiness(self):
        self.ready = not self.ready


@dataclass
class PersistentObjects:
    app: flask.Flask
    running_processes: dict
    pipeline_threads_store: dict
    current_active_pipeline_objects: dict
    file_storage_dir: pathlib.Path
    active_rag: Optional[ActiveRAG]


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
    From: https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread
    """

    def __init__(self, group, target, name, target_args=(), target_kwargs=None):
        super().__init__(args=target_args, group=group, target=target, name=name, kwargs=target_kwargs)
        self._stop_event = threading.Event()
        self._hard_stop_event = threading.Event()
        self._return = None

    @property
    def set_to_stop(self):
        return self._stop_event.is_set() or self._hard_stop_event.is_set()

    @property
    def set_to_hard_stop(self):
        return self._stop_event.is_set() and self._hard_stop_event.is_set()

    @property
    def return_value(self):
        if self.is_alive():
            return False
        return self._return

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def stop(self, hard_stop=False):
        self._stop_event.set()
        if hard_stop:
            self._hard_stop_event.set()


class BaseUtil(ABC):
    def __init__(self, app: flask.app.Flask, file_storage: str, step_name: str):
        self._app = app
        self._file_storage = pathlib.Path(file_storage)
        self._process_step = step_name
        self._process_name = None
        self._thread: Optional[StoppableThread] = None
        self.config = None

    def _complete_pickle_path(
        self,
        process: Optional[str],
        extension: str = "pickle",
    ) -> pathlib.Path:
        return Path(
            self._file_storage
            / (process if process is not None else "")
            / f"{self.process_name if process is None else process}_{self.process_step}.{extension}"
        )

    @property
    @abstractmethod
    def serializable_config(self) -> dict:
        raise NotImplementedError()

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
    @abstractmethod
    def protected_kwargs(self) -> list[str]:
        raise NotImplementedError()

    @property
    def this_thread(self):
        return self._thread

    @this_thread.setter
    def this_thread(self, thread: StoppableThread):
        self._thread = thread

    @property
    def is_threaded(self) -> bool:
        return self.this_thread is not None

    @property
    def thread_is_set_to_stop(self) -> bool:
        return (
            self.is_threaded
            and self.this_thread.is_alive()
            and self.this_thread.set_to_stop
        )

    @property
    def thread_is_set_to_hard_stop(self) -> bool:
        return (
            self.is_threaded
            and self.this_thread.is_alive()
            and self.this_thread.set_to_hard_stop
        )

    @property
    def process_name(self) -> Optional[str]:
        return self._process_name

    @process_name.setter
    def process_name(self, name: str) -> None:
        self._process_name = name

    @property
    def process_step(self) -> str:
        return self._process_step

    @property
    def file_storage_path(self) -> pathlib.Path:
        return self._file_storage

    @file_storage_path.setter
    def file_storage_path(self, sub_path: Union[str, pathlib.Path]) -> None:
        self._file_storage = Path(self._file_storage / sub_path)
        self._file_storage.mkdir(exist_ok=True)  # ToDo: warning when folder exists

    @abstractmethod
    def has_process(
        self, process: Optional[str] = None, extensions: Optional[list[str]] = None
    ) -> bool:
        if extensions is None:
            return self._complete_pickle_path(process).exists()
        return all(
            [self._complete_pickle_path(process, ext).exists() for ext in extensions]
        )

    @abstractmethod
    def delete_process(
        self,
        process: Optional[str] = None,
        extensions: Optional[list[str]] = None,
    ) -> None:
        if self.has_process(process, extensions):
            if extensions is None:
                extensions = ["pickle"]
            for ext in extensions:
                _pickle = self._complete_pickle_path(process, ext)
                _pickle.unlink()

    @abstractmethod
    def read_config(
        self,
        config: Optional[Union[FileStorage, dict]],
        process_name=None,
        language=None,
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
        base_config["corpus_name"] = (
            process_name.lower()
            if process_name is not None
            else base_config["corpus_name"].lower()
        )
        self.config = base_config
        # ToDo: Since n_process > 1 would induce Multiprocessing and this doesn't work with the Threading approach
        #  to keep the server able to respond, the value will be popped here.
        #  Maybe I can find a solution to this problem
        self.config.pop("n_process", None)
        return None

    @abstractmethod
    def read_stored_config(self, ext: str = "yaml") -> tuple[str, dict]:
        _sub_configs = {k: {} for k in self.sub_config_names}
        _file_name = f"{self.process_name}_{self.process_step}_config.{ext}"
        _file = Path(self._file_storage / _file_name)
        if not _file.exists():
            return self.process_step, {}
        config_yaml = yaml.safe_load(_file.open("rb"))
        for key, value in config_yaml.copy().items():
            _sub_key_split = key.split("_")
            if len(_sub_key_split) > 1 and _sub_key_split[0] in _sub_configs.keys():
                _sub_configs[_sub_key_split[0]]["_".join(_sub_key_split[1:])] = value
                config_yaml.pop(key)
        config_yaml.update(_sub_configs)
        return self.process_step, config_yaml

    @abstractmethod
    def _process_method(self) -> Optional[Callable]:
        raise NotImplementedError()

    @abstractmethod
    def _load_pre_components(
        self, cache_name, active_process_objs: Optional[dict[str, dict]] = None
    ) -> Optional[Union[tuple, list]]:
        """
        Pre Components should be returned as a tuple or list; they will be provided to
        '_start_process' as its args. So when implementing both methods, one should be
        aware of their positioning.
        """
        return []

    @abstractmethod
    def _start_process(
        self, process_factory, *args, **kwargs
    ) -> Tuple[bool, Union[str, Any]]:
        """
        Should return whether process was successful (and with it could provide the resulting object)
         and if not additional an error/exception message
        """
        raise NotImplementedError()

    def _in_protected_kwargs(self, kwarg: str) -> bool:
        if isinstance(kwarg, str):
            return any([kwarg.startswith(x) for x in self.protected_kwargs])
        else:
            return False

    @staticmethod
    def abort_chain(
        step: str,
    ) -> list:
        return {
            StepsName.DATA: StepsName.ALL,
            StepsName.EMBEDDING: [
                StepsName.EMBEDDING,
                StepsName.CLUSTERING,
                StepsName.GRAPH,
                StepsName.INTEGRATION,
            ],
            StepsName.CLUSTERING: [
                StepsName.CLUSTERING,
                StepsName.GRAPH,
                StepsName.INTEGRATION,
            ],
            StepsName.GRAPH: [StepsName.GRAPH, StepsName.INTEGRATION],
            StepsName.INTEGRATION: [StepsName.INTEGRATION],
        }.get(step, StepsName.ALL)

    def stop_thread(self, hard_stop: bool = False):
        if self.is_threaded:
            self.this_thread.stop(hard_stop=hard_stop)

    def start_process(
        self,
        cache_name: str,
        process_factory,
        process_tracker: dict,
        active_process_objs: Optional[dict[str, dict]] = None,
        return_result_obj: bool = False,
        thread: Optional[StoppableThread] = None,
        **kwargs,
    ):
        self.this_thread = thread
        add_status_to_running_process(
            self.process_name, self.process_step, ProcessStatus.RUNNING, process_tracker
        )
        _pre_components = self._load_pre_components(cache_name, active_process_objs)
        config = self.config.copy()
        try:
            _valid_config = (
                getfullargspec(self._process_method()).args
                if self._process_method() is not None
                else None
            )
            if _valid_config is not None:
                for _arg in config.copy().keys():
                    if _arg not in _valid_config and not self._in_protected_kwargs(
                        _arg
                    ):
                        config.pop(_arg)
                config.update(kwargs)
            _process_status = None
            if _pre_components is None:
                _process_status = self._start_process(process_factory, **config)
            else:
                _process_status = self._start_process(
                    process_factory, *_pre_components, **config
                )

            if _process_status[0]:
                add_status_to_running_process(
                    self.process_name,
                    self.process_step,
                    ProcessStatus.FINISHED,
                    process_tracker,
                )
            else:
                self._app.logger.error(
                    _process_status[1]
                    if (
                        len(_process_status) > 1 and (isinstance(_process_status[1], str) or isinstance(_process_status[1], Exception))
                    )
                    else "No error message given."
                )
                for _step in BaseUtil.abort_chain(self.process_step):
                    add_status_to_running_process(
                        self.process_name, _step, ProcessStatus.ABORTED, process_tracker
                    )

            if active_process_objs is not None and hasattr(
                active_process_objs, "update"
            ):
                active_process_objs[self.process_name][self.process_step] = deepcopy(
                    _process_status[1]
                )
            if return_result_obj:
                return _process_status[1] if len(_process_status) > 1 else None
        except Exception as e:
            for _step in BaseUtil.abort_chain(self.process_step):
                add_status_to_running_process(
                    self.process_name, _step, ProcessStatus.ABORTED, process_tracker
                )
            self._app.logger.error(e)
        return None


pipeline_query_params = namedtuple(
    "PipelineQueryParams",
    [
        "process_name",
        "language",
        "skip_present",
        "omitted_pipeline_steps",
        "return_statistics",
    ],
)

steps_relation_dict = {
    StepsName.DATA: 1,
    StepsName.EMBEDDING: 2,
    StepsName.CLUSTERING: 3,
    StepsName.GRAPH: 4,
    StepsName.INTEGRATION: 5,
}


class PipelineLanguage:
    language_map = {
        "en": "en",
        "english": "en",
        "englisch": "en",
        "de": "de",
        "german": "de",
        "deutsch": "de",
    }

    @staticmethod
    def language_from_string(lang):
        return PipelineLanguage.language_map.get(lang.lower(), "en")


def add_status_to_running_process(
    process_name: str,
    step_name: str,
    step_status: ProcessStatus,
    running_processes: dict,
):
    _step = {
        "name": step_name,
        "rank": steps_relation_dict[step_name],
        "status": step_status,
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
            "true": True,
            "yes": True,
            "y": True,
            "ja": True,
            "j": True,
            "false": False,
            "no": False,
            "n": False,
            "nein": False,
        }.get(str_bool.lower(), default)
    else:
        return False


def string_conformity(s: str):
    if s is None:
        return None
    return re.sub(r"\s+|-+", "_", s.lower())


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
            logger.warning(
                f"TimeOut while waiting >{time_out} seconds for download to finish."
                f" Hopefully this is just due to installed models not refreshing."
            )

    spacy_language = None
    try:
        spacy_language = spacy.load(spacy_model)
    except IOError as e:
        if spacy_model != default_model:
            if is_valid_spacy_model(spacy_model):
                logger.info(
                    f"Model '{spacy_model}' doesn't seem to be installed; trying to download model."
                )
                wait_for_download(spacy_model)
                spacy_language = spacy.load(spacy_model)
            else:
                logger.error(f"{e}\nUsing default model {default_model}.")
                try:
                    spacy_language = spacy.load(default_model)
                except IOError as e:
                    logger.error(
                        f"{e}\ntrying to download default model {default_model}."
                    )
                    wait_for_download(default_model)
                    spacy_language = spacy.load(default_model)
        else:
            logger.error(f"{e}\ntrying to download default model {default_model}.")
            wait_for_download(default_model)
            spacy_language = spacy.load(default_model)
    return spacy_language


def get_default_spacy_model():
    return "en_core_web_trf"


def transform_document_addition_results(iterator: Iterator):
    phrases_dict = dict()
    type_list = ["added", "incorporated"]
    _additional_info_key = "additional_info"
    _phrases_key = "phrases"
    _graph_field_key = "graph_cluster"
    _phrase_id_key = "_id"
    _text_key = "text"
    _offsets_key = "offsets"

    for doc_id, graph_dict in iterator:
        for _type in type_list:
            for _phrase, _additional_info in zip(graph_dict.get(_type, {}).get(_phrases_key, []), graph_dict.get(_type, {}).get(_additional_info_key, [])):
                if _phrase_id := _phrase.get(_phrase_id_key, None):
                    if _phrase_id not in phrases_dict:
                        phrases_dict[_phrase_id] = {
                            "graph": _phrase.get(_graph_field_key, [None])[0],
                            "id": _phrase_id,
                            "documents": list(),
                            "label": _additional_info.get(_text_key, ""),
                        }
                    phrases_dict[_phrase_id]["documents"].append({"id": doc_id, "offsets": _additional_info.get(_offsets_key, [])})
    return phrases_dict