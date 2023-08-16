from pathlib import Path

import yaml
import sys

from flask import jsonify

sys.path.insert(0, "src")
import util_functions


class GraphCreationUtil:

    def __init__(self, app, file_storage):
        self._app = app
        self._file_storage = Path(file_storage)
        self.config = None

    def read_config(self, config):
        base_config = {}
        if config is None:
            self._app.logger.info("No config file provided; using default values")
        else:
            try:
                base_config = yaml.safe_load(config.stream)
                if base_config.get('model', False):
                    raise KeyError(f"No model name provided in config: {base_config}")
            except Exception as e:
                self._app.logger.error(f"Couldn't read config file: {e}")
                return jsonify("Encountered error. See log.")
        self.config = base_config