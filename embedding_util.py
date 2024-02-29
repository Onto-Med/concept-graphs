import inspect
from pathlib import Path
from typing import Optional

import flask
import yaml
import sys

from flask import jsonify

from main_utils import ProcessStatus

sys.path.insert(0, "src")
import util_functions


DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-albert-small-v2'


class PhraseEmbeddingUtil:

    def __init__(self, app: flask.app.Flask, file_storage: str, step_name: str = "embedding"):
        self._app = app
        self._file_storage = Path(file_storage)
        self._process_step = step_name
        self._process_name = None
        self.config = None

    @property
    def process_name(self):
        return self._process_name

    @process_name.setter
    def process_name(self, name):
        self._process_name = name

    @property
    def process_step(self):
        return self._process_step

    def read_config(self, config, process_name=None, language=None):
        _language_model_map = {"en": DEFAULT_EMBEDDING_MODEL, "de": "Sahajtomar/German-semantic"}
        base_config = {'model': DEFAULT_EMBEDDING_MODEL, 'down_scale_algorithm': None}
        if config is None:
            self._app.logger.info("No config file provided; using default values")
            if language is not None:
                base_config["model"] = _language_model_map.get(language, DEFAULT_EMBEDDING_MODEL)
        else:
            try:
                base_config = yaml.safe_load(config.stream)
                if not base_config.get('model', False):
                    raise KeyError(f"No model name provided in config: {base_config}")
            except Exception as e:
                self._app.logger.error(f"Couldn't read config file: {e}")
                return jsonify("Encountered error. See log.")
        if language is not None and not base_config.get("model", False):
            base_config["model"] = _language_model_map.get(language, DEFAULT_EMBEDDING_MODEL)

        if process_name is not None:
            base_config["corpus_name"] = process_name
        # ToDo: Since n_process > 1 would induce Multiprocessing and this doesn't work with the Threading approach
        #  to keep the server able to respond, the value will be popped here.
        #  Maybe I can find a solution to this problem
        base_config.pop("n_process", None)
        self.config = base_config

    def set_file_storage_path(self, sub_path):
        self._file_storage = Path(self._file_storage / sub_path)
        self._file_storage.mkdir(exist_ok=True)  # ToDo: warning when folder exists

    def has_pickle(self, process):
        _pickle = Path(self._file_storage / process / f"{process}_{self.process_step}.pickle")
        return _pickle.exists()

    def delete_pickle(self, process):
        if self.has_pickle(process):
            _pickle = Path(self._file_storage / process / f"{process}_{self.process_step}.pickle")
            _pickle.unlink()

    def start_process(self, cache_name, process_factory, process_tracker):
        config = self.config.copy()
        # default_args = inspect.getfullargspec(process_factory.create)[0]
        # _ = [config.pop(x, None) for x in list(config.keys()) if x not in default_args]

        data_obj = util_functions.load_pickle(
            Path(self._file_storage / f"{cache_name}_data.pickle"))

        process_tracker[self.process_name]["status"][self.process_step] = ProcessStatus.RUNNING
        _process = None
        try:
            _process = process_factory.create(
                data_obj=data_obj,
                cache_path=self._file_storage,
                cache_name=f"{cache_name}_{self.process_step}",
                model_name=config.pop("model", DEFAULT_EMBEDDING_MODEL),
                **config
            )
            process_tracker[self.process_name]["status"][self.process_step] = ProcessStatus.FINISHED
        except Exception as e:
            process_tracker[self.process_name]["status"][self.process_step] = ProcessStatus.ABORTED
            self._app.logger.error(e)

        return _process
