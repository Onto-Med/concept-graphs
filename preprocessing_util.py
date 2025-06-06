import sys
import zipfile
from pathlib import Path
from types import GeneratorType
from typing import List, Dict, Union, Generator, Optional, Callable

import flask.app
import yaml
from dataclass_wizard import JSONSerializable
from werkzeug.datastructures import FileStorage

sys.path.insert(0, "src")
from main_utils import StepsName, get_bool_expression, NegspacyConfig, load_spacy_model, get_default_spacy_model, BaseUtil
from src.data_functions import DataProcessingFactory
from src.negspacy.utils import FeaturesOfInterest


class PreprocessingUtil(BaseUtil):
    def __init__(
            self,
            app: flask.app.Flask,
            file_storage: str
    ):
        super().__init__(app, file_storage, StepsName.DATA)
        self.labels = None
        self.data = None
        self._ext = ["pickle", "spacy"]

    @property
    def default_model(self):
        return get_default_spacy_model()

    @property
    def language_model_map(self):
        return {
            "en": self.default_model,
            "de": "de_dep_news_trf"
        }

    @property
    def serializable_config(self) -> dict:
        _neg_spacy: JSONSerializable =  self.config.get("negspacy_config", NegspacyConfig())
        _serializable_conf =  self.config.copy()
        _serializable_conf["negspacy_config"] = _neg_spacy.to_dict()
        return _serializable_conf

    @property
    def default_config(self) -> dict:
        return {
            'spacy_model': self.default_model,
            'file_encoding': 'utf-8',
            "omit_negated_chunks": False
        }

    @property
    def sub_config_names(self) -> list[str]:
        return []

    @property
    def necessary_config_keys(self) -> list[str]:
        return []

    @property
    def protected_kwargs(self) -> list[str]:
        return ["spacy_model"]

    def _read_zip_content(self, zip_archive, labels) -> List[Dict[str, str]]:
        extension = self.config.get("file_extension", "txt")
        return [{"name": Path(f.filename).stem,
                 "content": zip_archive.read(f.filename).decode(self.config.get('file_encoding', 'utf-8')),
                 "label": labels.get(Path(f.filename).stem, None)}
                for f in zip_archive.filelist if (not f.is_dir()) and (Path(f.filename).suffix.lstrip('.') == extension.lstrip('.'))]

    def read_data(
            self,
            data: Union[FileStorage, Path, Generator],
            replace_keys: Optional[dict],
            label_getter: Optional[str]
    ):
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

    def read_config(
            self,
            config: Optional[Union[FileStorage, dict]],
            process_name=None,
            language=None
    ):
        _response = super().read_config(config, process_name, language)
        if _response is None:
            if language is not None and not self.config.get("spacy_model", False):
                self.config["spacy_model"] = self.language_model_map.get(language, self.default_model)

            if self.config.get("negspacy", False):
                _enabled = False
                _neg_config = NegspacyConfig()
                _neg_options = self.config.pop("negspacy")
                if isinstance(_neg_options, dict):
                    for k, v in _neg_options.items():
                        if k.lower() == "enabled":
                            _enabled = get_bool_expression(v)
                        elif k.lower() == "configuration":
                            _neg_config = NegspacyConfig.from_dict(v)
                elif isinstance(_neg_options, list):
                    for _c in _neg_options:
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
                self.config["negspacy_config"] = _neg_config
                self.config["omit_negated_chunks"] = _enabled

            if self.config.get("tfidf_filter", False):
                _conf = self.config.pop("tfidf_filter")
                if _conf.get("enabled", True):
                    self.config["filter_min_df"] = _conf.get("min_df", 1)
                    self.config["filter_max_df"] = _conf.get("max_df", 1.0)
                    self.config["filter_stop"] = _conf.get("stop", None)
        return _response

    def read_stored_config(self, ext: str = "yaml"):
        return super().read_stored_config(ext)

    def has_process(
            self,
            process: Optional[str] = None,
            extensions: Optional[list[str]] = None
    ):
        return super().has_process(process, extensions)

    def delete_process(
            self,
            process: Optional[str] = None,
            extensions: Optional[list[str]] = None
    ):
        super().delete_process(process, extensions)

    def _process_method(self) -> Optional[Callable]:
        return DataProcessingFactory.create

    def _load_pre_components(
            self,
            cache_name
    ):
        return None

    def _start_process(
            self,
            process_factory,
            *args,
            **kwargs
    ):
        _model = kwargs.pop("spacy_model", get_default_spacy_model())
        spacy_language = load_spacy_model(_model, self._app.logger, get_default_spacy_model())
        _process = None
        try:
            _process = process_factory.create(
                pipeline=spacy_language,
                base_data=self.data,
                cache_name=f"{self.process_name}_{self.process_step}",
                cache_path=self._file_storage,
                save_to_file=True,
                **kwargs
            )
        except Exception as e:
            raise e
        return _process
