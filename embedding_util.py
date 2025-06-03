import inspect
from pathlib import Path
from typing import Optional, Union

import flask
import yaml
import sys

from flask import jsonify
from munch import Munch, unmunchify
from werkzeug.datastructures import FileStorage

from main_utils import ProcessStatus, StepsName, add_status_to_running_process
from src.marqo_external_utils import MarqoEmbeddingStore
from src.util_functions import load_pickle

sys.path.insert(0, "src")
import util_functions


DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-albert-small-v2'


class PhraseEmbeddingUtil:

    def __init__(self, app: flask.app.Flask, file_storage: str, step_name: StepsName = StepsName.EMBEDDING):
        self._app = app
        self._file_storage = Path(file_storage)
        self._process_step = step_name
        self._process_name = None
        self._storage_method = None
        self.config = None

    @property
    def process_name(self):
        return self._process_name

    @process_name.setter
    def process_name(self, name):
        self._process_name = name

    @property
    def storage_method(self):
        return self._storage_method

    @storage_method.setter
    def storage_method(self, method: tuple[str, Optional[dict]]):
        self._storage_method = method

    @property
    def process_step(self):
        return self._process_step

    def read_config(self, config: Optional[Union[FileStorage, dict]], process_name=None, language=None):
        _language_model_map = {"en": DEFAULT_EMBEDDING_MODEL, "de": "Sahajtomar/German-semantic"}
        base_config = {'model': DEFAULT_EMBEDDING_MODEL, 'down_scale_algorithm': None, 'storage': {'method': 'pickle'}}
        if isinstance(config, dict):
            if isinstance(config, Munch):
                base_config = unmunchify(config)
            else:
                base_config = config
            _scaling = base_config.get("scaling", {}).copy()
            for k, v in _scaling.items():
                base_config[f"scaling_{k}"] = v
            base_config.pop("scaling", None)

            _storage = base_config.pop("storage", None)
            if _storage is not None and isinstance(_storage, dict):
                base_config["storage_method"] = _storage.get("method", ("pickle", None,))
                if isinstance(base_config["storage_method"], tuple) and base_config["storage_method"][0] == "pickle":
                    pass
                else:
                    for k, v in _storage.get("config", {}).items():
                        base_config[f"vectorstore_{k}"] = v
            else:
                base_config["storage_method"] = ('pickle', None,)

        elif isinstance(config, FileStorage):
            try:
                base_config = yaml.safe_load(config.stream)
                if not base_config.get('model', False):
                    raise KeyError(f"No model name provided in config: {base_config}")
            except Exception as e:
                self._app.logger.error(f"Couldn't read config file: {e}")
                return jsonify("Encountered error. See log.")
        else:
            self._app.logger.info("No config file provided; using default values")
            if language is not None:
                base_config["model"] = _language_model_map.get(language, DEFAULT_EMBEDDING_MODEL)

        # _storage = base_config.get("storage", None)
        # if _storage is None:
        #     base_config[""]
        if language is not None and not base_config.get("model", False):
            base_config["model"] = _language_model_map.get(language, DEFAULT_EMBEDDING_MODEL)

        base_config["corpus_name"] = process_name.lower() if process_name is not None else base_config["corpus_name"].lower()
        # ToDo: Since n_process > 1 would induce Multiprocessing and this doesn't work with the Threading approach
        #  to keep the server able to respond, the value will be popped here.
        #  Maybe I can find a solution to this problem
        base_config.pop("n_process", None)
        self.config = base_config

    def set_file_storage_path(self, sub_path):
        self._file_storage = Path(self._file_storage / sub_path)
        self._file_storage.mkdir(exist_ok=True)  # ToDo: warning when folder exists

    def has_process(self, process: Optional[str] = None):
        _pickle = Path(self._file_storage / (process if process is not None else "") /
                       f"{self.process_name if process is None else process}_{self.process_step}.pickle")
        return _pickle.exists()

    def delete_process(self, process: Optional[str] = None):
        if self.has_process(process):
            _pickle = Path(self._file_storage / (process if process is not None else "") /
                           f"{self.process_name if process is None else process}_{self.process_step}.pickle")
            if _pickle.stat().st_size < 500:
                # assume pickled config dict for vectorstore and not pickled embedding object
                _config: dict = load_pickle(_pickle)
                vector_store = MarqoEmbeddingStore.existing_from_config(_config)
                vector_store.marqo_index.delete()
            _pickle.unlink()

    def read_stored_config(self, ext: str = "yaml"):
        _file_name = f"{self.process_name}_{self.process_step}_config.{ext}"
        _file = Path(self._file_storage / _file_name)
        if not _file.exists():
            return self.process_step, {}
        config_yaml = yaml.safe_load(_file.open('rb'))
        return self.process_step, config_yaml

    def start_process(self, cache_name, process_factory, process_tracker):
        config = self.config.copy()
        # default_args = inspect.getfullargspec(process_factory.create)[0]
        # _ = [config.pop(x, None) for x in list(config.keys()) if x not in default_args]

        data_obj = util_functions.load_pickle(
            Path(self._file_storage / f"{cache_name}_data.pickle"))

        add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.RUNNING, process_tracker)
        _process = None
        if self.storage_method is None and config.get("storage_method", False):
            self.storage_method = config.pop("storage_method", ("pickle", None,))
        else:
            _ = config.pop("storage_method", None)
        try:
            _process = process_factory.create(
                data_obj=data_obj,
                cache_path=self._file_storage,
                cache_name=f"{cache_name}_{self.process_step}",
                model_name=config.pop("model", DEFAULT_EMBEDDING_MODEL),
                storage_method=self.storage_method,
                **config
            )
            add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.FINISHED, process_tracker)
        except Exception as e:
            add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.ABORTED, process_tracker)
            self._app.logger.error(e)

        return _process
