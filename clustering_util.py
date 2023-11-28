import inspect
from pathlib import Path

import flask
import yaml
import sys

from flask import jsonify

from main_utils import ProcessStatus

sys.path.insert(0, "src")
import util_functions
import embedding_functions


class ClusteringUtil:

    def __init__(self, app, file_storage):
        self._app = app
        self._file_storage = Path(file_storage)
        self._process_name = None
        self._process_step = "clustering"
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
        base_config = {"algorithm": "kmeans", "downscale": "umap", "scaling_n_neighbors": 10, "scaling_min_dist": 0.1,
                       "scaling_n_components": 100, "scaling_metric": 'euclidean', "scaling_random_state": 42,
                       "kelbow_k": (10, 100), "kelbow_show": False}
        if config is None:
            self._app.logger.info("No config file provided; using default values")
        else:
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
        if process_name is not None:
            base_config["corpus_name"] = process_name
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
        algorithm = config.pop("algorithm", "kmeans")
        downscale = config.pop("downscale", "umap")
        # _ = [config.pop(x, None) for x in list(config.keys()) if x not in default_args]

        emb_obj = util_functions.load_pickle(Path(self._file_storage / f"{cache_name}_embedding.pickle"))

        process_tracker[self.process_name]["status"][self.process_step] = ProcessStatus.RUNNING
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
            process_tracker[self.process_name]["status"][self.process_step] = ProcessStatus.FINISHED
        except Exception as e:
            process_tracker[self.process_name]["status"][self.process_step] = ProcessStatus.ABORTED
            self._app.logger.error(e)

        if cluster_obj is not None:
            return embedding_functions.show_top_k_for_concepts(cluster_obj=cluster_obj.concept_cluster,
                                                               embedding_object=emb_obj, yield_concepts=True)
        else:
            return []
