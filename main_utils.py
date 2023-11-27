import enum
from abc import ABC, abstractmethod
from pathlib import Path

import yaml


class ProcessStatus(enum.Enum):
    STARTED = "started"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class BaseUtil(ABC):
    def __init__(self):
        self._app = None
        self._file_storage = None
        self._base_config = None
        self._final_config = None

    @property
    @abstractmethod
    def base_config(self):
        return self._base_config

    @base_config.setter
    def base_config(self, config):
        self._base_config = config

    @property
    @abstractmethod
    def file_storage(self):
        return self._file_storage

    @file_storage.setter
    def file_storage(self, file_path):
        self._file_storage = file_path

    @property
    @abstractmethod
    def app(self):
        return self._app

    @app.setter
    def app(self, app_instance):
        self._app = app_instance

    @abstractmethod
    def read_config(self, config):
        if config is None:
            self.app.logger.info("No config file provided; using default values")
            self._final_config = self.base_config
        else:
            try:
                loaded_config = yaml.safe_load(config.stream)
                with Path(Path(self.file_storage) / loaded_config.get("corpus_name", "default") / "_config.yaml"
                          ).open('w') as config_save:
                    yaml.safe_dump(loaded_config, config_save)
            except Exception as e:
                self.app.logger.error(f"Couldn't read config file: {e}")
            self._final_config = loaded_config
