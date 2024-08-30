from pathlib import Path
from typing import Optional, Union

import flask
import yaml
import sys

from flask import jsonify
from munch import Munch, unmunchify
from werkzeug.datastructures import FileStorage

from main_utils import ProcessStatus, StepsName, add_status_to_running_process

sys.path.insert(0, "src")
import util_functions
import embedding_functions


class ClusteringUtil:

    def __init__(self, app: flask.app.Flask, file_storage: str, step_name: StepsName = StepsName.CLUSTERING):
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

    def read_config(self, config: Optional[Union[FileStorage, dict]], process_name=None, language=None):
        base_config = {"algorithm": "kmeans", "downscale": "umap", "scaling_n_neighbors": 10, "scaling_min_dist": 0.1,
                       "scaling_n_components": 100, "scaling_metric": 'euclidean', "scaling_random_state": 42,
                       "kelbow_k": (10, 100), "kelbow_show": False}
        if isinstance(config, dict):
            if isinstance(config, Munch):
                _config = unmunchify(config)
            else:
                _config = config
            for _type in ["scaling", "clustering", "kelbow"]:
                _sub_config = _config.get(_type, {}).copy()
                for k, v in _sub_config.items():
                    _config[f"{_type}_{k}"] = v
                _config.pop(_type, None)
            if _config.pop("missing_as_recommended", True):
                for k, v in base_config.items():
                    if k not in _config:
                        _config[k] = v
            base_config = _config
        elif isinstance(config, FileStorage):
            try:
                _config = yaml.safe_load(config.stream)
                if _config.pop("missing_as_recommended", True):
                    for k, v in base_config.items():
                        if k not in _config:
                            _config[k] = v
                base_config = _config
            except Exception as e:
                self._app.logger.error(f"Couldn't read config file: {e}")
                return jsonify("Encountered error. See log.")
        else:
            self._app.logger.info("No config file provided; using default values")

        base_config["corpus_name"] = process_name.lower() if process_name is not None else base_config["corpus_name"].lower()
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

    def read_stored_config(self, ext: str = "yaml"):
        _sub_configs = {"kelbow": {}, "scaling": {}, "clustering": {}}
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

    def start_process(self, cache_name, process_factory, process_tracker):
        config = self.config.copy()
        # default_args = inspect.getfullargspec(process_factory.create)[0]
        algorithm = config.pop("algorithm", "kmeans")
        downscale = config.pop("downscale", "umap")
        # _ = [config.pop(x, None) for x in list(config.keys()) if x not in default_args]

        emb_obj = util_functions.load_pickle(Path(self._file_storage / f"{cache_name}_embedding.pickle"))

        add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.RUNNING, process_tracker)
        cluster_obj = None
        try:
            cluster_obj = process_factory.create(
                sentence_embeddings=emb_obj,
                cache_path=self._file_storage,
                cache_name=f"{cache_name}_{self.process_step}",
                cluster_algorithm=algorithm,
                down_scale_algorithm=downscale,
                cluster_by_down_scale=True,  # ToDo: is this feasible to toggle via config?
                ** config
            )
            add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.FINISHED, process_tracker)
        except Exception as e:
            add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.ABORTED, process_tracker)
            self._app.logger.error(e)

        if cluster_obj is not None:
            return embedding_functions.show_top_k_for_concepts(cluster_obj=cluster_obj.concept_cluster,
                                                               embedding_object=emb_obj, yield_concepts=True)
        else:
            return []
