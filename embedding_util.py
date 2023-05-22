import yaml


class PhraseEmbeddingUtil:

    def __init__(self, app, file_storage):
        self._app = app
        self._file_storage = file_storage
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
            except Exception as e:
                self._app.logger.error(f"Couldn't read config file: {e}")
        self.config = base_config

    def start_phrase_embedding(self, cache_name, process_factory):
        pass
