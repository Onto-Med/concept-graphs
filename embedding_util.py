import inspect
from pathlib import Path

import yaml
import sys

from flask import jsonify

sys.path.insert(0, "src")
import util_functions


DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-albert-small-v2'


class PhraseEmbeddingUtil:

    def __init__(self, app, file_storage):
        self._app = app
        self._file_storage = Path(file_storage)
        self.config = None

    def read_config(self, config):
        base_config = {'model_name': DEFAULT_EMBEDDING_MODEL, 'down_scale_algorithm': None}
        if config is None:
            self._app.logger.info("No config file provided; using default values")
        else:
            try:
                base_config = yaml.safe_load(config.stream)
                if not base_config.get('model', False):
                    raise KeyError(f"No model name provided in config: {base_config}")
                with Path(Path(self._file_storage) / f"{base_config.get('corpus_name', 'default')}_embedding_config.yaml"
                          ).open('w') as config_save:
                    yaml.safe_dump(base_config, config_save)
            except Exception as e:
                self._app.logger.error(f"Couldn't read config file: {e}")
                return jsonify("Encountered error. See log.")
        self.config = base_config

    def start_phrase_embedding(self, cache_name, process_factory):
        config = self.config.copy()
        # default_args = inspect.getfullargspec(process_factory.create)[0]
        # _ = [config.pop(x, None) for x in list(config.keys()) if x not in default_args]

        data_obj = util_functions.load_pickle(Path(self._file_storage / f"{cache_name}_data-processed.pickle"))
        return process_factory.create(
            data_obj=data_obj,
            cache_path=self._file_storage,
            cache_name=f"{cache_name}_embeddings",
            model_name=config.pop("model", DEFAULT_EMBEDDING_MODEL),
            down_scale_algorithm=None,
            **config
        )
