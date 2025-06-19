import sys
from typing import Optional, Union, Callable

import flask

from werkzeug.datastructures import FileStorage

sys.path.insert(0, "src")
from main_utils import StepsName, BaseUtil
from src.marqo_external_utils import MarqoEmbeddingStore
from src.embedding_functions import SentenceEmbeddingsFactory
from src.util_functions import load_pickle
from load_utils import FactoryLoader


#ToDo: I need to check the storage_method config/kwarg; it works but there are some redundancies
class PhraseEmbeddingUtil(BaseUtil):

    def __init__(
            self,
            app: flask.app.Flask,
            file_storage: str
    ):
        super().__init__(app, file_storage, StepsName.EMBEDDING)

    @property
    def default_model(self):
        return 'sentence-transformers/paraphrase-albert-small-v2'

    @property
    def language_model_map(self):
        return {
            "en": self.default_model,
            "de": "Sahajtomar/German-semantic"
        }

    @property
    def default_storage_method(self):
        return "pickle", None,

    @property
    def storage_method(self) -> tuple[str, Optional[dict]]:
        if self.config is None or self.config.get("storage_method", "pickle") == "pickle":
            return self.default_storage_method
        else:
            return self.config.get("storage_method")

    @storage_method.setter
    def storage_method(
            self,
            storage_method: tuple[str, Optional[dict]]
    ):
        self.config["storage_method"] = storage_method

    def set_storage_method(
            self,
            storage_method: str,
            connection_dict: dict
    ):
        self.storage_method = (storage_method, connection_dict)

    @property
    def serializable_config(self) -> dict:
        return self.config.copy()

    @property
    def default_config(self) -> dict:
        return {
            'model': self.default_model,
            'down_scale_algorithm': None,
            'storage': {
                'method': 'pickle'
            }
        }

    @property
    def sub_config_names(self) -> list[str]:
        return ["scaling"]

    @property
    def necessary_config_keys(self) -> list[str]:
        return []

    @property
    def protected_kwargs(self) -> list[str]:
        return ["model", "scaling", "vectorstore"]

    def read_config(
            self,
            config: Optional[Union[FileStorage, dict]],
            process_name=None,
            language=None
    ):
        _response = super().read_config(config, process_name, language)
        if _response is None:
            _storage = self.config.pop("storage", None)
            if _storage is not None and isinstance(_storage, dict):
                self.config["storage_method"] = _storage.get("method", self.default_storage_method)
                if isinstance(self.config["storage_method"], tuple) and self.config["storage_method"][0] == "pickle":
                    pass
                else:
                    for k, v in _storage.get("config", {}).items():
                        self.config[f"vectorstore_{k}"] = v
            else:
                self.config["storage_method"] = self.default_storage_method

        if language is not None and not self.config.get("model", False):
            self.config["model"] = self.language_model_map.get(language, self.default_model)
        return _response

    def read_stored_config(
            self,
            ext: str = "yaml"
    ):
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
        if self.has_process(process):
            _pickle = self._complete_pickle_path(process)
            if _pickle.stat().st_size < 500:
                # assume pickled config dict for vectorstore and not pickled embedding object
                _config: dict = load_pickle(_pickle)
                vector_store = MarqoEmbeddingStore.existing_from_config(_config)
                vector_store.marqo_index.delete()
            _pickle.unlink()

    def _process_method(self) -> Callable:
        return SentenceEmbeddingsFactory.create

    def _load_pre_components(
            self,
            cache_name,
            active_process_objs: Optional[dict[str, dict]] = None
    ) -> Union[tuple, list]:
        _cached = active_process_objs.get(cache_name, {}).get(StepsName.DATA, None)
        data_obj = FactoryLoader.load_data(self._file_storage, cache_name) if _cached is None else _cached
        return (data_obj,)

    def _start_process(
            self,
            process_factory,
            *args,
            **kwargs
    ):
        data_obj, = args
        emb_obj = None
        try:
            emb_obj = process_factory.create(
                data_obj=data_obj,
                cache_path=self._file_storage,
                cache_name=f"{self.process_name}_{self.process_step}",
                model_name=kwargs.pop("model", self.default_model),
                # storage_method=self.storage_method,
                **kwargs
            )
        except Exception as e:
            return False, e
        return True, emb_obj
