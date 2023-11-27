import inspect
import zipfile
from pathlib import Path
from typing import List, Dict, Union

import spacy
import yaml
from werkzeug.datastructures import FileStorage

from main_utils import ProcessStatus

DEFAULT_SPACY_MODEL = "en_core_web_trf"


class PreprocessingUtil:

    def __init__(self, app, file_storage):
        self._app = app
        self._file_storage = Path(file_storage)
        self._process_name = None
        self._process_step = "data"
        self.config = None
        self.labels = None
        self.data = None

    def _read_zip_content(self, zip_archive, labels) -> List[Dict[str, str]]:
        extension = self.config.get("file_extension", "txt")
        return [{"name": Path(f.filename).stem,
                 "content": zip_archive.read(f.filename).decode(self.config.get('file_encoding', 'utf-8')),
                 "label": labels.get(Path(f.filename).stem, None)}
                for f in zip_archive.filelist if (not f.is_dir()) and (Path(f.filename).suffix.lstrip('.') == extension.lstrip('.'))]

    @property
    def process_name(self):
        return self._process_name

    @process_name.setter
    def process_name(self, name):
        self._process_name = name

    @property
    def process_step(self):
        return self._process_step

    def set_file_storage_path(self, sub_path):
        self._file_storage = Path(self._file_storage / sub_path)
        self._file_storage.mkdir(exist_ok=True)  # ToDo: warning when folder exists

    def has_pickle(self, process):
        _step = "data-processed"
        _pickle = Path(self._file_storage / process / f"{process}_{_step}.pickle")
        return _pickle.exists()

    def delete_pickle(self, process):
        if self.has_pickle(process):
            _step = "data-processed"
            _pickle = Path(self._file_storage / process / f"{process}_{_step}.pickle")
            _pickle.unlink()

    def read_data(self, data: Union[FileStorage, Path]):
        try:
            if isinstance(data, FileStorage):
                archive_path = Path(self._file_storage / data.filename)
                data.save(archive_path)
            elif isinstance(data, Path):
                archive_path = Path(self._file_storage / data.name)
                if archive_path.exists():
                    archive_path.unlink()
                with archive_path.open(mode='xb') as target:
                    target.write(data.read_bytes())
                data.unlink()
            else:
                self.data = None
                return
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

        if process_name is not None:
            base_config["corpus_name"] = process_name
        self.config = base_config

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

    def start_process(self, cache_name, process_factory, process_tracker):
        config = self.config.copy()
        default_args = inspect.getfullargspec(process_factory.create)[0]
        spacy_language = spacy.load(config.pop("spacy_model", DEFAULT_SPACY_MODEL))

        for x in list(config.keys()):
            if x not in default_args:
                config.pop(x)

        process_tracker[self.process_name]["status"][self.process_step] = ProcessStatus.RUNNING
        _process = None
        try:
            _process = process_factory.create(
                pipeline=spacy_language,
                base_data=self.data,
                cache_name=f"{cache_name}_data-processed",
                cache_path=self._file_storage,
                save_to_file=True,
                **config
            )
            process_tracker[self.process_name]["status"][self.process_step] = ProcessStatus.FINISHED
        except Exception as e:
            process_tracker[self.process_name]["status"][self.process_step] = ProcessStatus.ABORTED
            self._app.logger.error(e)

        return _process
