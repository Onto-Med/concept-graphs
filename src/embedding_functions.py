import logging
import pathlib
from typing import Optional, Union, List, Iterable

import numpy as np
import tensorflow as tf
import torch
import umap

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from data_functions import DataProcessingFactory
from src.util_functions import NoneDownScaleObj
from util_functions import load_pickle, save_pickle


# ToDo: somewhere else
def _set_extensions(
) -> None:
    from spacy.tokens import Doc
    if not Doc.has_extension("text_id"):
        Doc.set_extension("text_id", default=None)
    if not Doc.has_extension("doc_name"):
        Doc.set_extension("doc_name", default=None)
    if not Doc.has_extension("doc_topic"):
        Doc.set_extension("doc_topic", default=None)


class SentenceEmbeddingsFactory:

    @staticmethod
    def load(
            data_obj_path: Union[pathlib.Path, str],
            embeddings_path: Union[pathlib.Path, str],
            view_from_topics: Optional[Iterable[str]] = None,
    ):
        _set_extensions()
        _data_obj: DataProcessingFactory.DataProcessing = load_pickle(
            pathlib.Path(data_obj_path).absolute()
        )
        if view_from_topics is not None:
            _data_obj.set_view_by_labels(view_from_topics)
        _embeddings_obj: SentenceEmbeddingsFactory.SentenceEmbeddings = load_pickle(
            pathlib.Path(embeddings_path).absolute()
        )
        _sent_emb = SentenceEmbeddingsFactory.SentenceEmbeddings(data_obj=_data_obj)
        _sent_emb._embeddings = _embeddings_obj.sentence_embeddings
        assert _data_obj.chunk_sets_n == _embeddings_obj.sentence_embeddings.shape[0]
        return _sent_emb

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
            **kwargs
    ):
        _down_scale_alg_kwargs = {"_".join(key.split("_")[1:]): val for key, val in kwargs.items()
                                  if len(key.split("_")) > 1 and key.split("_")[0] == "scaling"}
        _down_scale_obj = {"umap": umap.UMAP, None: NoneDownScaleObj}[down_scale_algorithm](**_down_scale_alg_kwargs)
        for key in _down_scale_alg_kwargs.keys():
            kwargs.pop(f"scaling_{key}")
        if view_from_topics is not None:
            data_obj.set_view_by_labels(view_from_topics)

        logging.info(f"Creating Sentence Embedding with '{_down_scale_obj}'")
        _sent_emb = SentenceEmbeddingsFactory.SentenceEmbeddings(
            model_name=model_name,
            data_obj=data_obj,
            down_scale_obj=_down_scale_obj,
            head_only=head_only
        )
        _sent_emb._encode_data(n_process, **kwargs)
        save_pickle(_sent_emb, (cache_path / pathlib.Path(f"{cache_name}.pickle")))
        return _sent_emb

    class SentenceEmbeddings:
        def __init__(
                self,
                model_name: Optional[str] = None,
                data_obj: Optional[DataProcessingFactory.DataProcessing] = None,
                down_scale_obj: Optional[object] = None,
                head_only: bool = False
        ):
            self._model = None if model_name is None else SentenceTransformer(model_name)
            self._data_obj = data_obj
            self._down_scale_obj = down_scale_obj
            self._embeddings = None
            self._head_only = head_only #ToDo?

        @property
        def sentence_embeddings(
                self
        ) -> np.ndarray:
            return self._embeddings

        @property
        def data_processing_obj(
                self
        ):
            return self._data_obj

        @property
        def embedding_dim(
                self
        ) -> int:
            return self._embeddings.shape[1]

        # ToDo: maybe the tfidf filtering shall be applied already here?
        def _encode_data(
                self,
                n_process: int = 1,
                device: Union[str, List[str]] = 'cpu',
                **kwargs
        ):
            if "convert_to_numpy" in kwargs.keys():
                kwargs.pop("convert_to_numpy") #ToDo?
            if n_process > 1:
                for _key in list(kwargs.keys()):
                    if _key not in ["batch_size", "chunk_size"]:
                        kwargs.pop(_key)
                logging.info(f"Using {n_process} processes.")
                pool = self._model.start_multi_process_pool([device]*n_process if isinstance(device, str) else device)
                self._embeddings = self._model.encode_multi_process(
                    sentences=[dcs['text'] for dcs in self._data_obj.data_chunk_sets],
                    pool=pool,
                    **kwargs
                )
            else:
                self._embeddings = self._model.encode(
                    sentences=[dcs['text'] for dcs in self._data_obj.data_chunk_sets],
                    convert_to_numpy=True,
                    **kwargs
                )
            if not isinstance(self._down_scale_obj, NoneDownScaleObj):
                self._embeddings = self._down_scale_obj.fit_transform(self._embeddings)


def cosine(
        v1,
        v2
):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def cosine_against_collection(
        v1,
        v2,
        vector_dim,
        as_tensor=True
):
    if isinstance(v1, np.ndarray):
        _vector1 = tf.math.l2_normalize(torch.reshape(torch.from_numpy(v1), [1, vector_dim]), 1)
    else:
        _vector1 = tf.math.l2_normalize(v1, 1)
    if isinstance(v2, np.ndarray):
        if len(v2.shape) == 1:
            _shape = [1, vector_dim]
        else:
            _shape = [len(v2), vector_dim]
        _vector2 = tf.math.l2_normalize(torch.reshape(torch.from_numpy(v2), _shape), 1)
    else:
        _vector2 = tf.math.l2_normalize(v2, 1)

    _cosine_tensor = tf.matmul(_vector1, tf.transpose(_vector2, [1, 0]))
    if as_tensor:
        return _cosine_tensor
    return _cosine_tensor.numpy().tolist()[0]


def top_k_cosine(
        single_embed,
        collection_embed,
        top_k=None,
        distance=0.7,
        vector_dim=768,
        to_sorted=True
):
    _cosine_similarity = cosine_against_collection(single_embed, collection_embed, vector_dim)
    _reshaped_cs = tf.reshape(_cosine_similarity, [_cosine_similarity.shape[1], ])

    if top_k is not None:
        return tf.math.top_k(_reshaped_cs, k=min(collection_embed.shape[0], top_k),
                             sorted=to_sorted).indices.numpy().tolist()

    _vals = tf.where(tf.math.greater_equal(_reshaped_cs, tf.constant([distance])))
    _reshaped_vals = tf.reshape(_vals, [_vals.shape[0], ])

    if to_sorted:
        # [::-1] reverses the array to give highest first
        return _reshaped_vals.numpy()[np.argsort(_reshaped_cs.numpy()[_reshaped_vals.numpy()])[::-1]].tolist()
    else:
        return _reshaped_vals.numpy().tolist()


def show_top_k_for_concepts(
    cluster_obj: KMeans,
    embedding_object: SentenceEmbeddingsFactory.SentenceEmbeddings,
    top_k: int = 15,
    distance: float = 0.6,
    yield_concepts: bool = False
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
            _center, sent_embeddings, distance=distance, vector_dim=sent_embeddings.shape[1])
        if not yield_concepts:
            print(f"==Center {_c_id}==\n")
        for i in _indices[:top_k]:
            if not yield_concepts:
                print(f"\t{embedding_object.data_processing_obj.data_chunk_sets[i]['text']}")
            else:
                yield _c_id, i, embedding_object.data_processing_obj.data_chunk_sets[i]['text']
        if not yield_concepts:
            print("\n")
