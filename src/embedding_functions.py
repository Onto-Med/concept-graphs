import logging
import pathlib
from typing import Optional, Union, List, Iterable, Tuple

import numpy as np
import torch
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from src.data_functions import DataProcessingFactory
from src.marqo_external_utils import MarqoEmbeddingStore
from src.util_functions import (
    NoneDownScaleObj,
    load_pickle,
    save_pickle,
    set_spacy_extensions,
)


class SentenceEmbeddingsFactory:
    storage_options = {
        "pickle": save_pickle,
        "vector_store": MarqoEmbeddingStore,
        "vectorstore": MarqoEmbeddingStore,
    }

    @classmethod
    def load(
        cls,
        embeddings_obj_path: Union[pathlib.Path, str],
        data_obj: DataProcessingFactory.DataProcessing,
        view_from_topics: Optional[Iterable[str]] = None,
        storage_method: tuple[str, Optional[dict]] = (
            "pickle",
            None,
        ),
    ):
        try:
            _ = data_obj.data_chunk_sets
        except AttributeError:
            raise AssertionError(
                f"The provided object '{data_obj}' is not a DataProcessing object."
            )
        set_spacy_extensions()
        if view_from_topics is not None:
            data_obj.set_view_by_labels(view_from_topics)

        _file_path = pathlib.Path(embeddings_obj_path).absolute()
        _loaded_obj = load_pickle(_file_path)

        if storage_method is None:
            storage_method = (
                ("pickle", None)
                if not isinstance(_loaded_obj, dict)
                else ("vectorstore", _loaded_obj)
            )
        if storage_method[0].lower() == "pickle" and not isinstance(_loaded_obj, dict):
            _embeddings_obj: SentenceEmbeddingsFactory.SentenceEmbeddings = _loaded_obj
            _embeddings_obj.data_processing_obj = data_obj
            _embeddings_obj.source = None
        elif storage_method[0].lower() in ["vector_store", "vectorstore"]:
            _config: dict = load_pickle(_file_path)
            vector_store = MarqoEmbeddingStore.existing_from_config(_config)
            _embeddings_obj = SentenceEmbeddingsFactory.SentenceEmbeddings(
                model_name=_config.get("model_name", None),
                data_obj=data_obj,
            )
            _embeddings_obj.sentence_embeddings = vector_store.get_embeddings()
            _embeddings_obj.source = _config
        else:
            logging.error("Unknown storage method")
            return None

        # _sent_emb = SentenceEmbeddingsFactory.SentenceEmbeddings(data_obj=_data_obj)
        # _sent_emb._embeddings = _embeddings_obj.sentence_embeddings
        # if _data_obj.chunk_sets_n != _embeddings_obj.sentence_embeddings.shape[0]:
        #     raise AssertionError("chunks ")
        # return _sent_emb
        return _embeddings_obj

    @staticmethod
    def create(
        data_obj: DataProcessingFactory.DataProcessing,
        cache_path: pathlib.Path,
        cache_name: str,
        model_name: str,
        n_process: int = 1,
        view_from_topics: Optional[Iterable[str]] = None,
        down_scale_algorithm: Optional[str] = None,
        head_only: bool = False,
        storage_method: tuple[str, Optional[dict]] = (
            "pickle",
            None,
        ),  # dict for vectorstore configures the url and index name (things that shouldn't go to config params in yaml)
        **kwargs,
    ):
        _down_scale_alg_kwargs = {
            "_".join(key.split("_")[1:]): val
            for key, val in kwargs.items()
            if len(key.split("_")) > 1 and key.split("_")[0] == "scaling"
        }
        _vector_store_kwargs = {
            "_".join(key.split("_")[1:]): val
            for key, val in kwargs.items()
            if len(key.split("_")) > 1 and key.split("_")[0] == "vectorstore"
        }
        _down_scale_obj = {"umap": umap.UMAP, None: NoneDownScaleObj}[
            down_scale_algorithm
        ](**_down_scale_alg_kwargs)
        for add_ in [
            (
                "scaling",
                _down_scale_alg_kwargs,
            ),
            (
                "vectorstore",
                _vector_store_kwargs,
            ),
        ]:
            for key in add_[1].keys():
                kwargs.pop(f"{add_[0]}_{key}")
        if view_from_topics is not None:
            data_obj.set_view_by_labels(view_from_topics)

        logging.info(f"Creating Sentence Embedding with '{_down_scale_obj}'")
        _sent_emb = SentenceEmbeddingsFactory.SentenceEmbeddings(
            model_name=model_name,
            data_obj=data_obj,
            down_scale_obj=_down_scale_obj,
            head_only=head_only,
        )
        _sent_emb._encode_data(n_process, **kwargs)
        _sent_emb.embedding_config = kwargs
        _file_path = cache_path / pathlib.Path(f"{cache_name}.pickle")

        _storage_method = storage_method[0].lower()
        if _storage_method == "pickle":
            _sent_emb.data_processing_obj = None
            _sent_emb.source = None
            SentenceEmbeddingsFactory.storage_options[_storage_method](
                _sent_emb, _file_path
            )
        elif _storage_method in ["vector_store", "vectorstore"]:
            _url_key = set(k.lower() for k in storage_method[1].keys()).intersection(
                ["client", "client_url", "url"]
            )
            _url_key = list(_url_key)[0] if len(_url_key) > 0 else "client_url"
            _client_url = storage_method[1].get(_url_key, "http://localhost:8882")
            _index_name = storage_method[1].get("index_name", cache_name)
            vector_store = SentenceEmbeddingsFactory.storage_options[_storage_method](
                client_url=_client_url,
                index_name=_index_name,
                create_index=True,
                vector_dim=_sent_emb.embedding_dim,
                additional_index_settings=_vector_store_kwargs,
            )
            _added_embeddings = vector_store.store_embeddings(
                embeddings=_sent_emb.sentence_embeddings,
                embeddings_repr=[dcs["text"] for dcs in data_obj.data_chunk_sets],
            ).get("added", set())
            save_pickle(
                {
                    "client_url": _client_url,
                    "index_name": _index_name,
                    "model_name": model_name,
                    "dtype": str(_sent_emb.sentence_embeddings.dtype),
                },
                _file_path,
            )
        return _sent_emb

    class SentenceEmbeddings:
        def __init__(
            self,
            model_name: Optional[str] = None,
            data_obj: Optional[DataProcessingFactory.DataProcessing] = None,
            down_scale_obj: Optional[object] = None,
            head_only: bool = False,
        ):
            self._model = (
                None if model_name is None else SentenceTransformer(model_name)
            )
            self._data_obj = data_obj
            self._down_scale_obj = down_scale_obj
            self._embeddings = None
            self._head_only = head_only  # ToDo?
            self._source = None
            self._encoding_config = None

        @property
        def sentence_embeddings(self) -> np.ndarray:
            return self._embeddings

        @sentence_embeddings.setter
        def sentence_embeddings(self, value: np.ndarray):
            self._embeddings = value.astype(dtype="float32")

        @property
        def data_processing_obj(self):
            return self._data_obj

        @data_processing_obj.setter
        def data_processing_obj(self, value: DataProcessingFactory.DataProcessing):
            self._data_obj = value

        @property
        def source(self) -> Optional[dict]:
            return self._source

        @source.setter
        def source(self, value: Tuple[str, Optional[dict]]):
            if isinstance(value, tuple):
                self._source = value[1] if len(value) > 1 else None
            else:
                self._source = value

        @property
        def encoding_config(self):
            return {} if self._encoding_config is None else self._encoding_config

        @encoding_config.setter
        def encoding_config(self, value: dict):
            self._encoding_config = value

        @property
        def embedding_dim(self) -> int:
            return self._embeddings.shape[1]

        # ToDo: maybe the tfidf filtering shall be applied already here?
        def _encode_data(
            self,
            n_process: int = 1,
            device: Union[str, List[str]] = "cpu",
            external: Optional[Iterable[str]] = None,
            **kwargs,
        ):
            if "convert_to_numpy" in kwargs.keys():
                kwargs.pop("convert_to_numpy")  # ToDo?

            for _key in list(kwargs.keys()):
                if _key not in ["batch_size", "chunk_size"]:
                    kwargs.pop(_key)

            if n_process > 1:
                logging.info(f"Using {n_process} processes.")
                pool = self._model.start_multi_process_pool(
                    [device] * n_process if isinstance(device, str) else device
                )
                _embeddings = self._model.encode_multi_process(
                    sentences=(
                        [
                            dcs["text"]
                            for dcs in self.data_processing_obj.data_chunk_sets
                        ]
                        if external is None
                        else external
                    ),
                    pool=pool,
                    **kwargs,
                )
            else:
                _embeddings = self._model.encode(
                    sentences=(
                        [
                            dcs["text"]
                            for dcs in self.data_processing_obj.data_chunk_sets
                        ]
                        if external is None
                        else external
                    ),
                    convert_to_numpy=True,
                    **kwargs,
                )
            if external is None:
                self._embeddings = _embeddings
                if (
                    not isinstance(self._down_scale_obj, NoneDownScaleObj)
                    and self._down_scale_obj is not None
                ):
                    self._embeddings = self._down_scale_obj.fit_transform(
                        self._embeddings
                    )
            else:
                if (
                    not isinstance(self._down_scale_obj, NoneDownScaleObj)
                    and self._down_scale_obj is not None
                ):
                    return self._down_scale_obj.fit_transform(_embeddings)
                return _embeddings
            return None

        def encode_external(self, content: Iterable[str]):
            _embeddings = self._encode_data(external=content, **self.encoding_config)
            return _embeddings


def cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def cosine_against_collection(v1, v2, vector_dim, as_tensor=True):
    if isinstance(v1, np.ndarray):
        _vector1 = torch.nn.functional.normalize(
            torch.reshape(torch.from_numpy(v1), [1, vector_dim])
        )
    else:
        _vector1 = torch.nn.functional.normalize(v1)
    if isinstance(v2, np.ndarray):
        if len(v2.shape) == 1:
            _shape = [1, vector_dim]
        else:
            _shape = [len(v2), vector_dim]
        _vector2 = torch.nn.functional.normalize(torch.reshape(torch.from_numpy(v2), _shape))
    else:
        _vector2 = torch.nn.functional.normalize(v2)

    _cosine_tensor = torch.matmul(_vector1, torch.transpose(_vector2, 0, 1))
    if as_tensor:
        return _cosine_tensor
    return _cosine_tensor.numpy().tolist()[0]


def top_k_cosine(
    single_embed,
    collection_embed,
    top_k=None,
    distance=0.7,
    vector_dim=768,
    to_sorted=True,
):
    _cosine_similarity = cosine_against_collection(
        single_embed, collection_embed, vector_dim
    )
    _reshaped_cs = torch.reshape(
        _cosine_similarity,
        [
            _cosine_similarity.shape[1],
        ],
    )

    if top_k is not None:
        return (
            torch.topk(
                _reshaped_cs, k=min(collection_embed.shape[0], top_k), sorted=to_sorted
            )
            .indices.numpy()
            .tolist()
        )

    _vals = torch.nonzero(torch.ge(_reshaped_cs, distance))
    _reshaped_vals = torch.reshape(
        _vals,
        [
            _vals.shape[0],
        ],
    )

    if to_sorted:
        # [::-1] reverses the array to give highest first
        return _reshaped_vals.numpy()[
            np.argsort(_reshaped_cs.numpy()[_reshaped_vals.numpy()])[::-1]
        ].tolist()
    else:
        return _reshaped_vals.numpy().tolist()


def show_top_k_for_concepts(
    cluster_obj: KMeans,
    embedding_object: SentenceEmbeddingsFactory.SentenceEmbeddings,
    top_k: int = 15,
    distance: float = 0.6,
    yield_concepts: bool = False,
):
    """
    :param cluster_obj: a fitted KMeans object
    :param embedding_object: a SentenceEmbeddings object on which the KMeans object was fitted
    :param top_k: k nearest phrases
    :param distance: distance threshold; only cosine values from the centre that are higher will be regarded
    :param yield_concepts: whether concepts and phrases will be yielded or printed

    :returns: Either None and prints the words encompassed by a concept or yields a tuple(concept_id, phrase_id, phrase)
    """
    sent_embeddings = embedding_object.sentence_embeddings
    for _c_id, _center in enumerate(cluster_obj.cluster_centers_):
        _indices = top_k_cosine(
            _center,
            sent_embeddings,
            distance=distance,
            vector_dim=sent_embeddings.shape[1],
        )
        if not yield_concepts:
            print(f"==Center {_c_id}==\n")
        for i in _indices[:top_k]:
            if not yield_concepts:
                print(
                    f"\t{embedding_object.data_processing_obj.data_chunk_sets[i]['text']}"
                )
            else:
                yield _c_id, i, embedding_object.data_processing_obj.data_chunk_sets[i][
                    "text"
                ]
        if not yield_concepts:
            print("\n")
