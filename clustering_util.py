from pathlib import Path
from typing import Optional, Union, Callable

import flask
import sys

from werkzeug.datastructures import FileStorage

from main_utils import ProcessStatus, StepsName, add_status_to_running_process, BaseUtil
from src import data_functions

sys.path.insert(0, "src")
import util_functions
import embedding_functions
from src.cluster_functions import PhraseClusterFactory


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
            process: Optional[str] = None
    ) -> bool:
        return super().has_process(process)

    def delete_process(
            self,
            process: Optional[str] = None
    ) -> None:
        return super().delete_process(process)

    def _process_method(self) -> Callable:
        return PhraseClusterFactory.create

    def _load_pre_components(
            self,
            cache_name
    ) -> Union[tuple, list]:
        pass

    def _start_process(self, process_factory, *args, **kwargs):
        pass

    def start_process(
            self,
            cache_name,
            process_factory,
            process_tracker,
            **kwargs
    ):
        config = self.config.copy()
        # default_args = inspect.getfullargspec(process_factory.create)[0]
        algorithm = config.pop("algorithm", "kmeans")
        downscale = config.pop("downscale", "umap")
        # _ = [config.pop(x, None) for x in list(config.keys()) if x not in default_args]

        emb_obj = util_functions.load_pickle(Path(self._file_storage / f"{cache_name}_embedding.pickle"))
        if isinstance(emb_obj, dict):
            emb_obj = embedding_functions.SentenceEmbeddingsFactory.load(
                embeddings_obj_path=Path(self._file_storage / f"{cache_name}_embedding.pickle"),
                data_obj=data_functions.DataProcessingFactory.load(Path(self._file_storage / f"{cache_name}_data.pickle")),
                storage_method=('vector_store', {},),
            )

        add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.RUNNING, process_tracker)
        cluster_obj = None
        try:
            cluster_obj = process_factory.create(
                sentence_embeddings=emb_obj,
                cache_path=self._file_storage,
                cache_name=f"{cache_name}_{self.process_step}",
                cluster_algorithm=algorithm,
                down_scale_algorithm=downscale,
                cluster_by_down_scale=True,  # ToDo: is this feasible to toggle via config?
                ** config
            )
            add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.FINISHED, process_tracker)
        except Exception as e:
            add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.ABORTED, process_tracker)
            self._app.logger.error(e)

        if cluster_obj is not None:
            return embedding_functions.show_top_k_for_concepts(cluster_obj=cluster_obj.concept_cluster,
                                                               embedding_object=emb_obj, yield_concepts=True)
        else:
            return []
