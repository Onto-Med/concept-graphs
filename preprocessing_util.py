import inspect
import sys
import zipfile
from pathlib import Path
from types import GeneratorType
from typing import List, Dict, Union, Generator, Optional

import flask.app
import spacy
import yaml
from munch import Munch, unmunchify
from werkzeug.datastructures import FileStorage

from main_methods import get_bool_expression, NegspacyConfig
from main_utils import ProcessStatus, StepsName, add_status_to_running_process

DEFAULT_SPACY_MODEL = "en_core_web_trf"


class PreprocessingUtil:

    def __init__(self, app: flask.app.Flask, file_storage: str, step_name: StepsName = StepsName.DATA):
        self._app = app
        self._file_storage = Path(file_storage)
        self._process_step = step_name
        self._process_name = None
        self.config = None
        self.serializable_config = None
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

    def read_data(self, data: Union[FileStorage, Path, Generator], replace_keys: Optional[dict], label_getter: Optional[str]):
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
                            if (label_getter is not None) and (label_getter in _data):
                                _replaced_data['label'] = _data.pop(label_getter)
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

    def read_config(self, config: Optional[Union[FileStorage, dict]], process_name=None, language=None):
        base_config = {'spacy_model': DEFAULT_SPACY_MODEL, 'file_encoding': 'utf-8', "omit_negated_chunks": False}
        _language_model_map = {"en": DEFAULT_SPACY_MODEL, "de": "de_dep_news_trf"}

        if isinstance(config, dict):
            if isinstance(config, Munch):
                base_config = unmunchify(config)
            else:
                base_config = config
        elif isinstance(config, FileStorage):
            try:
                base_config = yaml.safe_load(config.stream)
            except Exception as e:
                self._app.logger.error(f"Couldn't read config file: {e}")
        else:
            self._app.logger.info("No config file provided; using default values")
            if language is not None:
                base_config["spacy_model"] = _language_model_map.get(language, DEFAULT_SPACY_MODEL)

        if language is not None and not base_config.get("spacy_model", False):
            base_config["spacy_model"] = _language_model_map.get(language, DEFAULT_SPACY_MODEL)

        base_config["corpus_name"] = process_name.lower() if process_name is not None else base_config["corpus_name"].lower()
        # ToDo: Since n_process > 1 would induce Multiprocessing and this doesn't work with the Threading approach
        #  to keep the server able to respond, the value will be popped here.
        #  Maybe I can find a solution to this problem
        base_config.pop("n_process", None)

        self.serializable_config = base_config.copy()
        if base_config.get("negspacy", False):
            _enabled = False
            _neg_config = NegspacyConfig()
            if isinstance(base_config["negspacy"], dict):
                for k, v in base_config["negspacy"].items():
                    if k.lower() == "enabled":
                        _enabled = get_bool_expression(v)
                    elif k.lower() == "configuration":
                        _neg_config = NegspacyConfig.from_dict(v)
            elif isinstance(base_config["negspacy"], list):
                for _c in base_config["negspacy"]:
                    if get_bool_expression(_c.get("enabled", "False")):
                        _enabled = True
                    elif _c.get("configuration", False):
                        _neg_config = NegspacyConfig.from_dict(_c.get("configuration")[0])
            base_config.pop("negspacy", None)
            base_config["negspacy_config"] = _neg_config
            base_config["omit_negated_chunks"] = _enabled

        self.config = base_config

    def read_labels(self, labels):
        base_labels = {}
        if labels is None:
            self._app.logger.info("No labels file provided; no labels will be added to text data")
        else:
            if isinstance(labels, str):
                self._app.logger.info(f"Labels will be extracted from the document server if the field '{labels}' is present.")
                return
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

        add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.RUNNING, process_tracker)
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
            add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.FINISHED, process_tracker)
        except Exception as e:
            add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.ABORTED, process_tracker)
            self._app.logger.error(e)

        return _process
