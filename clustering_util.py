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

    def read_config(self):
        pass

    def start_clustering(self):
        pass


        # cache_path=cache_path,
        # cache_name=f"{name_prefix}_phrase-cluster-obj{('_' + suffix) if suffix is not None else ''}",
        # cluster_algorithm=cluster_algorithm,
        # scaling_n_neighbors=scaling_n_neighbors, scaling_min_dist=scaling_min_dist, scaling_n_components=scaling_n_components,
        # scaling_metric='euclidean', scaling_random_state=42,
        # kelbow_k=(10, 100), kelbow_show=False,
        # cluster_by_down_scale=cluster_by_down_scale,
        # **kwargs