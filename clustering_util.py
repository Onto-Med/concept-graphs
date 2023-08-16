import inspect
from pathlib import Path

import flask
import yaml
import sys

from flask import jsonify

sys.path.insert(0, "src")
import util_functions
import embedding_functions


class ClusteringUtil:

    def __init__(self, app, file_storage):
        self._app = app
        self._file_storage = Path(file_storage)
        self.config = None

    def read_config(self, config):
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
                with Path(Path(self._file_storage) / base_config.get("corpus_name", "default") / "_config.yaml"
                          ).open('w') as config_save:
                    yaml.safe_dump(base_config, config_save)
            except Exception as e:
                self._app.logger.error(f"Couldn't read config file: {e}")
                return jsonify("Encountered error. See log.")
        self.config = base_config

    def start_clustering(self, cache_name, process_factory):
        config = self.config.copy()
        # default_args = inspect.getfullargspec(process_factory.create)[0]
        algorithm = config.pop("algorithm", "kmeans")
        downscale = config.pop("downscale", "umap")
        # _ = [config.pop(x, None) for x in list(config.keys()) if x not in default_args]

        emb_obj = util_functions.load_pickle(Path(self._file_storage / f"{cache_name}_embeddings.pickle"))

        cluster_obj = process_factory.create(
            sentence_embeddings=emb_obj,
            cache_path=self._file_storage,
            cache_name=f"{cache_name}_clustering",
            cluster_algorithm=algorithm,
            down_scale_algorithm=downscale,
            cluster_by_down_scale=True,  # ToDo: is this feasible to toggle via config?
            ** config
        )

        return embedding_functions.show_top_k_for_concepts(cluster_obj=cluster_obj.concept_cluster,
                                                           embedding_object=emb_obj, yield_concepts=True)
