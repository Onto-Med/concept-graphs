"""Base class for pipeline step utilities."""

import pathlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from inspect import getfullargspec
from pathlib import Path
from typing import Any

import flask
import yaml
from flask import Response, jsonify
from munch import Munch, unmunchify
from werkzeug.datastructures import FileStorage

from src.common.parsing import string_conformity
from src.common.threads import StoppableThread
from src.pipeline.status import ProcessStatus, StepsName, add_status_to_running_process


class BaseUtil(ABC):
    def __init__(self, app: flask.app.Flask, file_storage: str, step_name: str):
        self._app = app
        self._file_storage = pathlib.Path(file_storage)
        self._process_step = step_name
        self._process_name = None
        self._thread: StoppableThread | None = None
        self.config = None

    def _complete_pickle_path(
        self,
        process: str | None,
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
    def process_name(self) -> str | None:
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
    def file_storage_path(self, sub_path: str | pathlib.Path) -> None:
        self._file_storage = Path(self._file_storage / sub_path)
        self._file_storage.mkdir(exist_ok=True)  # ToDo: warning when folder exists

    @abstractmethod
    def has_process(
        self, process: str | None = None, extensions: list[str] | None = None
    ) -> bool:
        if extensions is None:
            return self._complete_pickle_path(process).exists()
        return all(
            [self._complete_pickle_path(process, ext).exists() for ext in extensions]
        )

    @abstractmethod
    def delete_process(
        self,
        process: str | None = None,
        extensions: list[str] | None = None,
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
        config: FileStorage | dict | None,
        process_name=None,
        language=None,
    ) -> Response | None:
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
            except (yaml.YAMLError, AttributeError, TypeError, UnicodeDecodeError) as e:
                self._app.logger.error("Couldn't read config file: %s", e)
                return jsonify("Encountered error. See log.")
        else:
            is_default_config = True
            self._app.logger.info("No config file provided; using default values")

        if not is_default_config:
            _inter = set(base_config.keys()).intersection(self.necessary_config_keys)
            if not len(_inter) == len(self.necessary_config_keys):
                raise KeyError(f"Missing necessary config values: '{_inter}'.")
        base_config["corpus_name"] = string_conformity(
            process_name if process_name is not None else base_config["corpus_name"]
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
    def _process_method(self) -> Callable | None:
        raise NotImplementedError()

    @abstractmethod
    def _load_pre_components(
        self, cache_name, active_process_objs: dict[str, dict] | None = None
    ) -> tuple | list | None:
        """
        Pre Components should be returned as a tuple or list; they will be provided to
        '_start_process' as its args. So when implementing both methods, one should be
        aware of their positioning.
        """
        return []

    @abstractmethod
    def _start_process(
        self, process_factory, *args, **kwargs
    ) -> tuple[bool, str | Any]:
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
        active_process_objs: dict[str, dict] | None = None,
        return_result_obj: bool = False,
        thread: StoppableThread | None = None,
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
                        len(_process_status) > 1
                        and (
                            isinstance(_process_status[1], str)
                            or isinstance(_process_status[1], Exception)
                        )
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
        except Exception:
            for _step in BaseUtil.abort_chain(self.process_step):
                add_status_to_running_process(
                    self.process_name, _step, ProcessStatus.ABORTED, process_tracker
                )
            self._app.logger.exception(
                "Unhandled error while executing '%s' step for process '%s'.",
                self.process_step,
                self.process_name,
            )
        return None
