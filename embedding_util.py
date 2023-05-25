import inspect
from pathlib import Path

import yaml
import sys

from flask import jsonify

sys.path.insert(0, "src")
import util_functions


class PhraseEmbeddingUtil:

    def __init__(self, app, file_storage):
        self._app = app
        self._file_storage = Path(file_storage)
        self.config = None

    #        data_obj: DataProcessingFactory.DataProcessing,
    #        cache_path: pathlib.Path,
    #        cache_name: str,
    #        model_name: str,
    #        n_process: int = 1,
    #        view_from_topics: Optional[Iterable[str]] = None,
    #        down_scale_algorithm: Optional[str] = 'umap',
    #        head_only: bool = False,
    #        **kwargs

    def read_config(self, config):
        base_config = {'model': 'sentence-transformers/paraphrase-albert-small-v2'}
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

    def start_phrase_embedding(self, cache_name, process_factory):
        config = self.config.copy()
        default_args = inspect.getfullargspec(process_factory.create)[0]
        _ = [config.pop(x, None) for x in list(config.keys()) if x not in default_args]

        data_obj = util_functions.load_pickle(Path(self._file_storage / f"{cache_name}_data-processed.pickle"))
        process_factory.create(
            data_obj,
            self._file_storage,
            f"{cache_name}_embeddings",
            self.config.get("model"),
            **config
        )
