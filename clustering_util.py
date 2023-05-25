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
