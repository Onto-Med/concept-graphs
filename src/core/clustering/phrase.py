import inspect
import logging
import pathlib
from typing import Union

import numpy as np
import umap
from numpy import ndarray
from sklearn.cluster import (
    AffinityPropagation,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils._random import sample_without_replacement
from tqdm.autonotebook import tqdm
from yellowbrick.cluster import kelbow_visualizer

from src.common.io import load_pickle, save_pickle
from src.core.clustering_config import ClusterNumberDetection
from src.core.data import DataProcessingFactory
from src.core.embedding_functions import SentenceEmbeddingsFactory
from src.core.reduction import NoneDownScaleObj

logging.basicConfig()
logging.root.setLevel(logging.INFO)


class PhraseClusterFactory:
    @staticmethod
    def create(
        sentence_embeddings: Union[
            SentenceEmbeddingsFactory.SentenceEmbeddings, np.ndarray
        ],
        cache_path: pathlib.Path,
        cache_name: str,
        cluster_algorithm: str = "kmeans",
        down_scale_algorithm: str = "umap",
        cluster_by_down_scale: bool = True,
        **kwargs,
    ):
        _cluster_obj = PhraseClusterFactory.PhraseCluster(
            sentence_embeddings=sentence_embeddings,
            cluster_algorithm=cluster_algorithm,
            down_scale_algorithm=down_scale_algorithm,
            cluster_by_down_scale=cluster_by_down_scale,
            **kwargs,
        )
        _cluster_obj.sentence_embedding = None
        save_pickle(_cluster_obj, (cache_path / pathlib.Path(f"{cache_name}.pickle")))
        return _cluster_obj

    @classmethod
    def load(
        cls,
        cluster_obj_path: Union[pathlib.Path, str],
        embedding_obj: SentenceEmbeddingsFactory.SentenceEmbeddings,
    ):
        _data = load_pickle(pathlib.Path(cluster_obj_path).resolve())
        try:
            _ = _data.cluster_center
        except AttributeError:
            raise AssertionError(
                f"The loaded object '{_data}' is not a PhraseCluster object."
            )
        try:
            _ = embedding_obj.sentence_embeddings
        except AttributeError:
            raise AssertionError(
                f"The provided object '{embedding_obj}' is not a SentenceEmbeddings object."
            )
        _data.sentence_embedding = embedding_obj
        return _data

    class PhraseCluster:
        """

        **kwargs for the cluster algorithm, the scaling algorithm and the k-elbow algorithm have to be prefixed
        with 'clustering_', 'scaling_' & 'kelbow_' respectively
        """

        def __init__(
            self,
            sentence_embeddings: Union[
                SentenceEmbeddingsFactory.SentenceEmbeddings, np.ndarray
            ],
            cluster_algorithm: str = "kmeans",
            down_scale_algorithm: str = "umap",
            cluster_by_down_scale: bool = True,
            **kwargs,
        ):
            self._clustering_estimators_ref = {
                "kmeans": KMeans,
                "kmeans-mb": MiniBatchKMeans,
                "affinity-prop": AffinityPropagation,
            }
            self._cluster_alg_kwargs = {
                "_".join(key.split("_")[1:]): val
                for key, val in kwargs.items()
                if len(key.split("_")) > 1 and key.split("_")[0] == "clustering"
            }
            self._down_scale_alg_kwargs = {
                "_".join(key.split("_")[1:]): val
                for key, val in kwargs.items()
                if len(key.split("_")) > 1 and key.split("_")[0] == "scaling"
            }
            self._kelbow_alg_kwargs = {
                "_".join(key.split("_")[1:]): val
                for key, val in kwargs.items()
                if len(key.split("_")) > 1 and key.split("_")[0] == "deduction"
            }
            # if self._kelbow_alg_kwargs.get("estimator", False):
            #     if self._kelbow_alg_kwargs.get("estimator") != "affinity-prop":
            #         self._kelbow_alg_kwargs["estimator"] = self._clustering_estimators_ref.get(self._kelbow_alg_kwargs.get("estimator"))
            #     else:
            #         self._kelbow_alg_kwargs.pop("estimator")
            self.sentence_embedding = sentence_embeddings
            self._cluster_alg = (
                cluster_algorithm
                if cluster_algorithm in ["kmeans", "kmeans-mb", "affinity-prop"]
                else "kmeans"
            )
            self._cluster_obj = self._clustering_estimators_ref[self._cluster_alg]
            self._down_scale_alg = down_scale_algorithm.lower()

            if self._down_scale_alg_kwargs.get("n_neighbors", False) and isinstance(
                self._down_scale_alg_kwargs.get("n_neighbors"), float
            ):
                _n_neighbors = min(
                    self._down_scale_alg_kwargs["n_neighbors"]
                    * self._sentence_emb.shape[0],
                    100,
                )
                self._down_scale_alg_kwargs["n_neighbors"] = int(_n_neighbors)

            self._down_scale_obj = {
                "umap": umap.UMAP,
                # "cvae": CVAEMantle,
                None: NoneDownScaleObj,
            }[down_scale_algorithm.lower()](**self._down_scale_alg_kwargs)
            self._concept_cluster = None
            self._kelbow = None
            self._cluster_center = None

            self._build_concept_cluster(cluster_by_down_scale)

        @property
        def cluster_center(self):
            return self._cluster_center

        @property
        def concept_cluster(self):
            return self._concept_cluster

        @property
        def sentence_embedding(self):
            return self._sentence_emb

        @sentence_embedding.setter
        def sentence_embedding(self, value):
            self._sentence_emb = (
                value
                if isinstance(value, ndarray)
                else (None if value is None else value.sentence_embeddings)
            )

        @property
        def get_params(self):
            return {
                f"scaling ({self._down_scale_alg})": self._down_scale_alg_kwargs,
                f"cluster ({self._cluster_alg})": self._cluster_alg_kwargs,
                "deduction": self._kelbow_alg_kwargs,
            }

        def _build_concept_cluster(self, cluster_by_down_scale: bool = True):
            logging.info("Building Concept Cluster ...")
            _cluster_embeddings = self._sentence_emb

            _is_downscaling = cluster_by_down_scale and not isinstance(
                self._down_scale_obj, NoneDownScaleObj
            )
            if _is_downscaling:
                logging.info(
                    f"{self._down_scale_alg.upper()} arguments: {self._down_scale_obj.get_params()}"
                )
                _cluster_embeddings = self._down_scale_obj.fit_transform(
                    self._sentence_emb
                )

            _estimator = self._kelbow_alg_kwargs.get("estimator", False)
            logging.info("-- Clustering ...")
            if _is_downscaling and self._down_scale_alg != "cvae":
                logging.info(
                    "Using downscaled embeddings for clustering is only allowed for 'cvae' algorithm.\n"
                    "Cluster number deduction will be performed on downscaled embeddings but clustering needs to be done on the original embeddings!"
                )

            if (_estimator and (_estimator != AffinityPropagation)) or (
                not _estimator
                and self._cluster_alg != "affinity-prop"
                and _estimator is not None
            ):
                _n_clusters_given = self._cluster_alg_kwargs.get("n_clusters", False)
                if self._kelbow_alg_kwargs.get("enabled", False):
                    _deduction_kwargs = self._kelbow_alg_kwargs.copy()
                    _default_args = inspect.getfullargspec(ClusterNumberDetector).args
                    for _k in _deduction_kwargs.copy().keys():
                        if _k not in _default_args:
                            _deduction_kwargs.pop(_k, None)
                    cnd = ClusterNumberDetector(self, **_deduction_kwargs)
                    cnd.estimate_cluster_number(_cluster_embeddings)
                else:
                    # ToDo: here some default value -> not hard coded
                    self._cluster_alg_kwargs["n_clusters"] = (
                        _n_clusters_given if _n_clusters_given else 50
                    )

                logging.info(
                    f" ({self._cluster_alg}) with Arguments: {self._cluster_alg_kwargs}\n"
                    f"Number of Clusters: {self._cluster_alg_kwargs['n_clusters']}\n"
                )
                self._concept_cluster = self._cluster_obj(
                    **self._cluster_alg_kwargs
                ).fit(
                    (
                        _cluster_embeddings
                        if (_is_downscaling and self._down_scale_alg == "cvae")
                        else self._sentence_emb
                    ),
                )
            else:
                logging.info(
                    f" ({self._cluster_alg}) with Arguments: {self._cluster_alg_kwargs}"
                )
                self._concept_cluster = self._cluster_obj(
                    **self._cluster_alg_kwargs
                ).fit(
                    (
                        _cluster_embeddings
                        if (_is_downscaling and self._down_scale_alg == "cvae")
                        else self._sentence_emb
                    ),
                )

            if _is_downscaling and self._down_scale_alg == "cvae":
                self._cluster_center = self._down_scale_obj.inverse_transform(
                    self._concept_cluster.cluster_centers_
                )
            else:
                self._cluster_center = self._concept_cluster.cluster_centers_


class ClusterNumberDetector:
    def __init__(
        self,
        owner: PhraseClusterFactory.PhraseCluster,
        algorithm: ClusterNumberDetection = ClusterNumberDetection.SILHOUETTE,
        k_min: int = 2,
        k_max: int = 100,
        n_samples: int = 15,
        sample_fraction: int = 25,
        regression_poly_degree: int = 5,
    ):
        self._owner = owner
        self._algorithm = algorithm
        self._k_min = k_min
        self._k_max = k_max
        self._n_samples = n_samples
        self._sample_fraction = sample_fraction
        self._regression_poly_degree = regression_poly_degree

    def _fit_regression(self, x_reg, y_reg):
        poly = PolynomialFeatures(degree=self._regression_poly_degree)
        x_poly = poly.fit_transform(np.asarray(x_reg).reshape(-1, 1))

        model = LinearRegression()
        model.fit(x_poly, np.asarray(y_reg))

        x_lin = np.linspace(
            np.asarray(x_reg).min(), np.asarray(x_reg).max(), self._k_max
        )
        x_out = poly.transform(x_lin.reshape(-1, 1))
        y_out = model.predict(x_out)
        x_reg_recalc = list(range(self._k_min)) + x_reg
        max_reg = np.asarray(x_reg_recalc)[np.argmax(y_out)]

        return x_lin, y_out, max_reg

    def _k_elbow_estimation(self, projection: ndarray):
        _samples = []
        _kelbow_array = []
        _elbow_max = []

        for i in range(self._n_samples):
            _samples.append(
                sample_without_replacement(
                    projection.shape[0],
                    int(projection.shape[0] / self._sample_fraction),
                )
            )

        for _sample in tqdm(_samples):
            _kelbow = kelbow_visualizer(
                model=MiniBatchKMeans(n_init="auto"),
                X=projection[_sample],
                show=False,
                k=(self._k_min, self._k_max),
                metric={
                    ClusterNumberDetection.SILHOUETTE: "silhouette",
                    ClusterNumberDetection.DISTORTION: "distortion",
                    ClusterNumberDetection.CALINSKI_HARABASZ: "calinski_harabasz",
                }.get(self._algorithm, ClusterNumberDetection.SILHOUETTE),
            )
            _kelbow_array.append(_kelbow)

        for _kelbow in _kelbow_array:
            x_vals, y_regression, max_regression = self._fit_regression(
                _kelbow.k_values_, _kelbow.k_scores_
            )
            _elbow_max.append(max_regression)

        return np.average(np.asarray(_elbow_max))

    def estimate_cluster_number(self, projection: ndarray):
        logging.info("-- Calculating K-Elbow ...")
        logging.info(f"---- shape of embeddings: ({projection.shape})")
        logging.info(f"---- Arguments: {self._owner._kelbow_alg_kwargs}")

        # if "estimator" not in self._kelbow_alg_kwargs:
        #     _obj = self._cluster_obj
        # else:
        #     _obj = self._kelbow_alg_kwargs.pop("estimator", KMeans)
        #
        # if _obj in [KMeans, MiniBatchKMeans]:
        #     _model = _obj(n_init='auto')
        # else:
        #     _model = _obj()

        # self._kelbow = kelbow_visualizer(
        #     model=_model,
        #     X=_cluster_embeddings,
        #     **self._kelbow_alg_kwargs
        # )

        # _cluster_args = self._owner._cluster_alg_kwargs.copy()
        # if "n_clusters" in _cluster_args.keys():
        self._owner._cluster_alg_kwargs["n_clusters"] = int(
            self._k_elbow_estimation(projection)
        )
        # if self._cluster_obj in [KMeans, MiniBatchKMeans] and "n_init" not in _cluster_args.keys():
        #     _cluster_args["n_init"] = 'auto'


if __name__ == "__main__":
    data = DataProcessingFactory.load(
        pathlib.Path("../tmp/grascco/grascco_data.pickle")
    )
    emb = SentenceEmbeddingsFactory.load(
        pathlib.Path("../tmp/grascco/grascco_embedding.pickle"),
        data,
        storage_method=("vectorstore", None),
    )
    cluster = PhraseClusterFactory.load(
        pathlib.Path("../tmp/grascco/grascco_clustering.pickle"), emb
    )
