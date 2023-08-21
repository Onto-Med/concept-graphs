import inspect
import zipfile
from pathlib import Path
from typing import List, Dict

import spacy
import yaml


DEFAULT_SPACY_MODEL = "en_core_web_trf"


class PreprocessingUtil:

    def __init__(self, app, file_storage):
        self._app = app
        self._file_storage = Path(file_storage)
        self.config = None
        self.labels = None
        self.data = None

    def _read_zip_content(self, zip_archive, labels) -> List[Dict[str, str]]:
        extension = self.config.get("file_extension", "txt")
        return [{"name": Path(f.filename).stem,
                 "content": zip_archive.read(f.filename).decode(self.config.get('file_encoding', 'utf-8')),
                 "label": labels.get(Path(f.filename).stem, None)}
                for f in zip_archive.filelist if (not f.is_dir()) and (Path(f.filename).suffix.lstrip('.') == extension.lstrip('.'))]

    def set_file_storage_path(self, sub_path):
        self._file_storage = Path(self._file_storage / sub_path)
        self._file_storage.mkdir(exist_ok=True)  # ToDo: warning when folder exists

    def has_pickle(self, process):
        _step = "data-processed"
        _pickle = Path(self._file_storage / f"{process}_{_step}.pickle")
        return _pickle.exists()

    def read_data(self, data):
        try:
            archive_path = Path(self._file_storage / data.filename)
            data.save(archive_path)
            with zipfile.ZipFile(archive_path, mode='r') as archive:
                self.data = self._read_zip_content(archive, self.labels)
        except Exception as e:
            self._app.logger.error(f"Something went wrong with data file reading: {e}")

    def read_config(self, config, process_name=None, language=None):
        base_config = {'spacy_model': DEFAULT_SPACY_MODEL, 'file_encoding': 'utf-8'}
        if config is None:
            self._app.logger.info("No config file provided; using default values")
        else:
            try:
                base_config = yaml.safe_load(config.stream)
            except Exception as e:
                self._app.logger.error(f"Couldn't read config file: {e}")
        if language is not None and not base_config.get("spacy_model", False):
            base_config["spacy_model"] = {"en": DEFAULT_SPACY_MODEL, "de": "de_dep_news_trf"}.get(language, DEFAULT_SPACY_MODEL)

        self.config = base_config
        if process_name is not None:
            base_config["corpus_name"] = process_name
        sub_path = base_config.get('corpus_name', 'default')
        with Path(Path(self._file_storage) / f"{sub_path}_preprocessing_config.yaml"
                  ).open('w') as config_save:
            yaml.safe_dump(base_config, config_save)

    def read_labels(self, labels):
        base_labels = {}
        if labels is None:
            self._app.logger.info("No labels file provided; no labels will be added to text data")
        else:
            try:
                base_labels = yaml.safe_load(labels.stream)
            except Exception as e:
                self._app.logger.error(f"Couldn't read labels file: {e}")
        self.labels = base_labels

    def start_process(self, cache_name, process_factory):
        config = self.config.copy()
        default_args = inspect.getfullargspec(process_factory.create)[0]
        spacy_language = spacy.load(config.pop("spacy_model", DEFAULT_SPACY_MODEL))

        for x in list(config.keys()):
            if x not in default_args:
                config.pop(x)

        return process_factory.create(
            pipeline=spacy_language,
            base_data=self.data,
            cache_name=f"{cache_name}_data-processed",
            cache_path=self._file_storage,
            save_to_file=True,
            **config
        )
