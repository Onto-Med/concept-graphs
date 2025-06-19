import logging
from pathlib import Path
from pydoc import locate
from typing import Tuple, Union, Any, Optional, Callable

from flask import Response, Flask
from werkzeug.datastructures import FileStorage

from main_utils import BaseUtil, StepsName
from src.integration_functions import ConceptGraphIntegrationFactory
from src.util_functions import load_pickle, EmbeddingStore


class ConceptGraphIntegrationUtil(BaseUtil):

    def __init__(
            self,
            app: Flask,
            file_storage: str,
            embedding_store_cls: str = "src.marqo_external_utils.MarqoEmbeddingStore"
    ):
        super().__init__(app, file_storage, StepsName.INTEGRATION)
        self._embedding_store_cls = embedding_store_cls

    @property
    def serializable_config(self) -> dict:
        return {}

    @property
    def default_config(self) -> dict:
        return {}

    @property
    def sub_config_names(self) -> list[str]:
        return []

    @property
    def necessary_config_keys(self) -> list[str]:
        return []

    @property
    def protected_kwargs(self) -> list[str]:
        return []

    def has_process(
            self,
            process: Optional[str] = None,
            extensions: Optional[list[str]] = None
    ) -> bool:
        return super().has_process(process, extensions)

    def delete_process(
            self,
            process: Optional[str] = None,
            extensions: Optional[list[str]] = None
    ) -> None:
        super().delete_process(process, extensions)

    def read_config(
            self,
            config: Optional[Union[FileStorage, dict]],
            process_name=None,
            language=None
    ) -> Optional[Response]:
        return super().read_config(config, process_name, language)

    def read_stored_config(
            self,
            ext: str = "yaml"
    ) -> tuple[str, dict]:
        return super().read_stored_config(ext)

    def _process_method(self) -> Optional[Callable]:
        return ConceptGraphIntegrationFactory.create

    def _load_pre_components(
            self,
            cache_name,
            active_process_objs: Optional[dict[str, dict]] = None
    ) -> Optional[Union[tuple, list]]:
        _emb_config = Path(self._file_storage / f"{cache_name}_embedding.pickle")
        embedding_store: EmbeddingStore = locate(self._embedding_store_cls)
        if not embedding_store.is_accessible(_emb_config):
            _conf = load_pickle(_emb_config)
            logging.error(f"Couldn't access embedding store for '{cache_name}'. Check if the server is running for the following configuration:\n{_conf}.")
            raise RuntimeError("Embedding store not present.")
        embedding_store_impl: EmbeddingStore = embedding_store.existing_from_config(_emb_config)
        graph_list = load_pickle(Path(self._file_storage / f"{cache_name}_graph.pickle"))
        if len(graph_list) > 0:
            return embedding_store_impl, graph_list
        else:
            logging.warning("There were no graphs present.")
        raise RuntimeError("Couldn't find any graphs to update embeddings with.")

    def _start_process(
            self,
            process_factory,
            *args,
            **kwargs
    ):
        emb_store, graphs = args
        try:
            int_obj = process_factory.create(
                embedding_store=emb_store,
                graphs=graphs,
                cache_path=self._file_storage,
                cache_name=f"{self.process_name}_{self.process_step}",
            )
        except Exception as e:
            return False, e
        return True, int_obj