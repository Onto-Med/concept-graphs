import inspect
import zipfile
from pathlib import Path
from types import GeneratorType
from typing import List, Dict, Union, Generator, Optional

import flask.app
import yaml
from munch import Munch, unmunchify
from werkzeug.datastructures import FileStorage

from main_utils import ProcessStatus, StepsName, add_status_to_running_process, get_bool_expression, NegspacyConfig, \
    load_spacy_model, get_default_spacy_model
from src.negspacy.utils import FeaturesOfInterest


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
        self._ext = ["pickle", "spacy"]

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

    def has_process(self, process: Optional[str] = None):
        pickle = [Path(self._file_storage / (process if process is not None else "") /
                       f"{self.process_name if process is None else process}_{self.process_step}.{x}") for x in self._ext]
        return all([p.exists() for p in pickle])

    def delete_process(self, process: Optional[str] = None):
        if self.has_process(process):
            for p in self._ext:
                _pickle = Path(self._file_storage / (process if process is not None else "") /
                               f"{self.process_name if process is None else process}_{self.process_step}.{p}")
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
        base_config = {'spacy_model': get_default_spacy_model(), 'file_encoding': 'utf-8', "omit_negated_chunks": False}
        _language_model_map = {"en": get_default_spacy_model(), "de": "de_dep_news_trf"}

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
                base_config["spacy_model"] = _language_model_map.get(language, get_default_spacy_model())

        if language is not None and not base_config.get("spacy_model", False):
            base_config["spacy_model"] = _language_model_map.get(language, get_default_spacy_model())

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
            _foi_map = {
                "nc": FeaturesOfInterest.NOUN_CHUNKS,
                "ne": FeaturesOfInterest.NAMED_ENTITIES,
                "both": FeaturesOfInterest.BOTH
            }
            _neg_config.feat_of_interest = (
                _foi_map.get(_neg_config.feat_of_interest.lower(), FeaturesOfInterest.NAMED_ENTITIES)
                if isinstance(_neg_config.feat_of_interest, str) else FeaturesOfInterest.NAMED_ENTITIES
            )
            base_config.pop("negspacy", None)
            base_config["negspacy_config"] = _neg_config
            base_config["omit_negated_chunks"] = _enabled

        if base_config.get("tfidf_filter", False):
            _conf = base_config.pop("tfidf_filter")
            if _conf.get("enabled", True):
                base_config["filter_min_df"] = _conf.get("min_df", 1)
                base_config["filter_max_df"] = _conf.get("max_df", 1.0)
                base_config["filter_stop"] = _conf.get("stop", None)

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

    def read_stored_config(self, ext: str = "yaml"):
        _file_name = f"{self.process_name}_{self.process_step}_config.{ext}"
        _file = Path(self._file_storage / _file_name)
        if not _file.exists():
            return self.process_step, {}
        config_yaml = yaml.safe_load(_file.open('rb'))
        return self.process_step, config_yaml

    def start_process(self, cache_name, process_factory, process_tracker):
        config = self.config.copy()
        default_args = inspect.getfullargspec(process_factory.create)[0]
        _model = config.pop("spacy_model", get_default_spacy_model())
        spacy_language = load_spacy_model(_model, self._app.logger, get_default_spacy_model())

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