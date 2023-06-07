import inspect
from pathlib import Path

import yaml
import sys

from flask import jsonify

sys.path.insert(0, "src")
import util_functions


class ClusteringUtil:

    def __init__(self, app, file_storage):
        self._app = app
        self._file_storage = Path(file_storage)
        self.config = None

    def read_config(self, config):
        base_config = {"algorithm": "kmeans", "downscale": "umap"}
        if config is None:
            self._app.logger.info("No config file provided; using default values")
        else:
            try:
                base_config = yaml.safe_load(config.stream)
            except Exception as e:
                self._app.logger.error(f"Couldn't read config file: {e}")
                return jsonify("Encountered error. See log.")
        self.config = base_config

    def start_clustering(self, cache_name, process_factory):
        config = self.config.copy()
        # default_args = inspect.getfullargspec(process_factory.create)[0]
        # _ = [config.pop(x, None) for x in list(config.keys()) if x not in default_args]

        emb_obj = util_functions.load_pickle(Path(self._file_storage / f"{cache_name}_embeddings.pickle"))
        process_factory.create(
            # sentence_embeddings: Union[SentenceEmbeddingsFactory.SentenceEmbeddings, np.ndarray],
            # cache_path: pathlib.Path,
            # cache_name: str,
            # cluster_algorithm: str = 'kmeans',
            # down_scale_algorithm: str = 'umap',
            # cluster_by_down_scale: bool = True,
            # ** kwargs
        )


        # cache_path=cache_path,
        # cache_name=f"{name_prefix}_phrase-cluster-obj{('_' + suffix) if suffix is not None else ''}",
        # cluster_algorithm=cluster_algorithm,
        # scaling_n_neighbors=scaling_n_neighbors, scaling_min_dist=scaling_min_dist, scaling_n_components=scaling_n_components,
        # scaling_metric='euclidean', scaling_random_state=42,
        # kelbow_k=(10, 100), kelbow_show=False,
        # cluster_by_down_scale=cluster_by_down_scale,
        # **kwargs