from pathlib import Path
from typing import Optional, Union, Callable

import flask
import sys

from werkzeug.datastructures import FileStorage

sys.path.insert(0, "src")
from main_utils import StepsName, BaseUtil
from src.data_functions import DataProcessingFactory
from src.embedding_functions import SentenceEmbeddingsFactory, show_top_k_for_concepts
from src.cluster_functions import PhraseClusterFactory
from src.util_functions import load_pickle


class ClusteringUtil(BaseUtil):

    def __init__(
            self,
            app: flask.app.Flask,
            file_storage: str
    ):
        super().__init__(app, file_storage, StepsName.CLUSTERING)

    @property
    def default_config(self) -> dict:
        return {
            "algorithm": "kmeans",
            "downscale": "umap",
            "scaling_n_neighbors": 10,
            "scaling_min_dist": 0.1,
            "scaling_n_components": 100,
            "scaling_metric": 'euclidean',
            "scaling_random_state": 42,
            "deduction_k_min": 2,
            "deduction_k_max": 100
        }

    @property
    def sub_config_names(self) -> list[str]:
        return ["scaling", "clustering", "deduction"]

    @property
    def necessary_config_keys(self) -> list[str]:
        return []

    @property
    def protected_kwargs(self) -> list[str]:
        return ["clustering", "scaling", "deduction"]

    def read_config(
            self,
            config: Optional[Union[FileStorage, dict]],
            process_name=None,
            language=None
    ):
        _response = super().read_config(config, process_name, language)
        if _response is None:
            if self.config.pop("missing_as_recommended", True):
                for k, v in self.default_config.items():
                    if k not in self.config:
                        self.config[k] = v
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
    ) -> bool:
        return super().has_process(process, extensions)

    def delete_process(
            self,
            process: Optional[str] = None,
            extensions: Optional[list[str]] = None
    ) -> None:
        return super().delete_process(process, extensions)

    def _process_method(self) -> Callable:
        return PhraseClusterFactory.create

    def _load_pre_components(
            self,
            cache_name
    ) -> tuple:
        sent_emb = load_pickle(Path(self._file_storage / f"{cache_name}_embedding.pickle"))
        if isinstance(sent_emb, dict):
            sent_emb = SentenceEmbeddingsFactory.load(
                embeddings_obj_path=Path(self._file_storage / f"{cache_name}_embedding.pickle"),
                data_obj=DataProcessingFactory.load(
                    Path(self._file_storage / f"{cache_name}_data.pickle")),
                storage_method=('vector_store', {},),
            )
        return (sent_emb,)

    def _start_process(self, process_factory, *args, **kwargs):
        sent_emb, = args
        algorithm = kwargs.pop("algorithm", "kmeans")
        downscale = kwargs.pop("downscale", "umap")

        cluster_obj = None
        try:
            cluster_obj = process_factory.create(
                sentence_embeddings=sent_emb,
                cache_path=self._file_storage,
                cache_name=f"{self.process_name}_{self.process_step}",
                cluster_algorithm=algorithm,
                down_scale_algorithm=downscale,
                cluster_by_down_scale=True,  # ToDo: is this feasible to toggle via config?
                **kwargs
            )
        except Exception as e:
            raise e
        if cluster_obj is not None:
            return show_top_k_for_concepts(cluster_obj=cluster_obj.concept_cluster,
                                           embedding_object=sent_emb, yield_concepts=True)
        else:
            return []
