import inspect
import sys
import zipfile
from pathlib import Path
from types import GeneratorType
from typing import List, Dict, Union, Generator, Optional

import flask.app
import spacy
import yaml
from werkzeug.datastructures import FileStorage

from main_utils import ProcessStatus

DEFAULT_SPACY_MODEL = "en_core_web_trf"


class PreprocessingUtil:

    def __init__(self, app: flask.app.Flask, file_storage: str, step_name: str = "data"):
        self._app = app
        self._file_storage = Path(file_storage)
        self._process_step = step_name
        self._process_name = None
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
        _pickle = Path(self._file_storage / process / f"{process}_{self.process_step}.pickle")
        return _pickle.exists()

    def delete_pickle(self, process):
        if self.has_pickle(process):
            _pickle = Path(self._file_storage / process / f"{process}_{self.process_step}.pickle")
            _pickle.unlink()

    def read_data(self, data: Union[FileStorage, Path, Generator], replace_keys: Optional[dict]):
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
            elif isinstance(data, GeneratorType):
                if replace_keys is not None:
                    def _replace_keys():
                        for _data in data:
                            _replaced_data = {}
                            for key, repl in replace_keys.items():
                                _replaced_data[repl] = _data.pop(key)
                            _replaced_data.update(**_data)
                            yield _replaced_data
                    self.data = _replace_keys()
                else:
                    self.data = data
                return
            else:
                self.data = None
                return
            with zipfile.ZipFile(archive_path, mode='r') as archive:
                self.data = self._read_zip_content(archive, self.labels)
        except Exception as e:
            self._app.logger.error(f"Something went wrong with data file reading: {e}")

    def read_config(self, config, process_name=None, language=None):
        base_config = {'spacy_model': DEFAULT_SPACY_MODEL, 'file_encoding': 'utf-8'}
        _language_model_map = {"en": DEFAULT_SPACY_MODEL, "de": "de_dep_news_trf"}
        if config is None:
            self._app.logger.info("No config file provided; using default values")
            if language is not None:
                base_config["spacy_model"] = _language_model_map.get(language, DEFAULT_SPACY_MODEL)
        else:
            try:
                base_config = yaml.safe_load(config.stream)
            except Exception as e:
                self._app.logger.error(f"Couldn't read config file: {e}")

        if language is not None and not base_config.get("spacy_model", False):
            base_config["spacy_model"] = _language_model_map.get(language, DEFAULT_SPACY_MODEL)

        if process_name is not None:
            base_config["corpus_name"] = process_name
        # ToDo: Since n_process > 1 would induce Multiprocessing and this doesn't work with the Threading approach
        #  to keep the server able to respond, the value will be popped here.
        #  Maybe I can find a solution to this problem
        base_config.pop("n_process", None)
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
        _model = config.pop("spacy_model", DEFAULT_SPACY_MODEL)
        try:
            spacy_language = spacy.load(_model)
        except IOError as e:
            if _model != DEFAULT_SPACY_MODEL:
                self._app.logger.error(f"{e}\nUsing default model {DEFAULT_SPACY_MODEL}.")
                try:
                    spacy_language = spacy.load(DEFAULT_SPACY_MODEL)
                except IOError as e:
                    self._app.logger.error(f"{e}\ntrying to download default model {DEFAULT_SPACY_MODEL}.")
                    spacy.cli.download(DEFAULT_SPACY_MODEL)
                    spacy_language = spacy.load(DEFAULT_SPACY_MODEL)
            else:
                self._app.logger.error(f"{e}\ntrying to download default model {DEFAULT_SPACY_MODEL}.")
                spacy.cli.download(DEFAULT_SPACY_MODEL)
                spacy_language = spacy.load(DEFAULT_SPACY_MODEL)

        for x in list(config.keys()):
            if x not in default_args:
                config.pop(x)

        process_tracker[self.process_name]["status"][self.process_step] = ProcessStatus.RUNNING
        _process = None
        try:
            _process = process_factory.create(
                pipeline=spacy_language,
                base_data=self.data,
                cache_name=f"{cache_name}_{self.process_step}",
                cache_path=self._file_storage,
                save_to_file=True,
                **config
            )
            process_tracker[self.process_name]["status"][self.process_step] = ProcessStatus.FINISHED
        except Exception as e:
            process_tracker[self.process_name]["status"][self.process_step] = ProcessStatus.ABORTED
            self._app.logger.error(e)

        return _process
