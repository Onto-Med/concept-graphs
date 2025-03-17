import inspect
import itertools
import logging
import pathlib
import pickle
import re
import statistics
import sys
from collections import defaultdict, Counter
from functools import cache
from itertools import repeat
from typing import Iterable, Optional, Union, Any, Generator, List, Tuple
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import scipy.sparse.csgraph
from numpy import ndarray
from sklearn.linear_model import LinearRegression
from sklearn.utils._random import sample_without_replacement
from tqdm.autonotebook import tqdm

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans, AffinityPropagation
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize, MinMaxScaler, PolynomialFeatures
from yellowbrick.cluster import kelbow_visualizer
from scipy.sparse.csgraph import shortest_path, construct_dist_matrix, NegativeCycleError
from sklearn.feature_extraction.text import TfidfVectorizer as tfidfVec
#import sknetwork as skn

from data_functions import DataProcessingFactory, clean_span, get_actual_str
from pruning import unimodal

from embedding_functions import SentenceEmbeddingsFactory, top_k_cosine
from graph_functions import GraphCreator, unroll_graph, simplify_graph_naive, sub_clustering
from util_functions import load_pickle, save_pickle, pairwise, NoneDownScaleObj, ClusterNumberDetection

logging.basicConfig()
logging.root.setLevel(logging.INFO)


class WordEmbeddingClustering:
    """
    Parameters
    ----------

    sentence_embedding_obj: DataProcessingFactory.DataProcessing
    cluster_obj: sklearn.cluster.KMeans or sklearn.cluster.AgglomerativeClustering
    """

    def __init__(
            self,
            sentence_embedding_obj: SentenceEmbeddingsFactory.SentenceEmbeddings,
            cluster_obj: Optional[KMeans] = None,
            cluster_exclusion_ids: Optional[Iterable[int]] = None,
            text_field_value: str = 'text',
            lemma_field_value: str = 'lemma',
            text_id_field_value: str = 'doc_index'
    ):
        self._cluster_obj = cluster_obj
        self._exclusion_ids = [] if cluster_exclusion_ids is None else cluster_exclusion_ids
        self._text_field_value = text_field_value
        self._lemma_field_value = lemma_field_value
        self._text_id_field_value = text_id_field_value

        self._sentence_embed_obj = sentence_embedding_obj
        self._we_cluster = None
        self._concept_graph_cluster = None

    @property
    def concept_graph_cluster(self):
        return self._concept_graph_cluster

    @property
    def we_cluster(self):
        return self._we_cluster

    def create_we_clustering(
            self,
            use_lemma: bool = False
    ):
        self._we_cluster = self._WEClustering(self, use_lemma=use_lemma)
        return self._we_cluster

    def create_concept_graph_clustering(
            self,
    ):
        self._concept_graph_cluster = self._ConceptGraphClustering(self)
        return self._concept_graph_cluster

    # ToDo: add reference to paper
    class _WEClustering:
        def __init__(
                self,
                outer_instance: 'WordEmbeddingClustering',
                exclusion_ids: Optional[list] = None,
                use_lemma: bool = False,
                head_only: bool = False
        ):
            self._data_proc = outer_instance._sentence_embed_obj.data_processing_obj
            self._outer_instance = outer_instance
            self._concept_word_matrix = self._build_concept_word_matrix(exclusion_ids=exclusion_ids)
            self._document_word_matrix = self._build_document_word_matrix(use_lemma=use_lemma, head_only=head_only)
            self._document_concept_matrix = None

        @property
        def document_concept_matrix(
                self
        ) -> np.ndarray:
            return self._document_concept_matrix

        def get_norm_document_concept_matrix(
                self,
                norm='l2',
                **kwargs
        ) -> np.ndarray:
            if "X" in kwargs:
                kwargs.pop("X")
            return normalize(self.document_concept_matrix, norm=norm, **kwargs)

        l2_norm_document_concept_matrix = property(get_norm_document_concept_matrix)

        def _build_concept_word_matrix(
                self,
                exclusion_ids: Optional[list] = None
        ) -> list:
            return [
                " ".join([self._data_proc.data_chunk_sets[i][self._outer_instance._text_field_value]
                          for i, j in enumerate(self._outer_instance._cluster_obj.labels_ == n) if j])
                for n in range(self._outer_instance._cluster_obj.n_clusters) if
                n not in (self._outer_instance._exclusion_ids if exclusion_ids is None else exclusion_ids)
            ]

        def _build_document_word_matrix(
                self,
                use_lemma: bool = False,
                head_only: bool = False
        ) -> list:
            _dw_matrix = defaultdict(list)
            _outer = self._outer_instance
            _text_field = _outer._text_field_value if not use_lemma else _outer._lemma_field_value
            for _d in self._data_proc.noun_chunks_corpus:
                _chunk_dict = clean_span(_d["spacy_chunk"])

                _text = ""
                if _chunk_dict is not None:
                    _text = get_actual_str(_chunk_dict, (False, use_lemma, head_only,), case_sensitive=False)

                # if isinstance(_d[_text_field], str):
                #     _text = _d[_text_field]
                # elif _d[_text_field] is not None:
                #     _text = str(_d[_text_field])
                _dw_matrix[_d[_outer._text_id_field_value]].append(_text)
            # need to insert at least empty strings for doc_ids that aren't covered; else errors ahead
            # (different array shapes later)
            for _missing_id in set(range(self._data_proc.documents_n)).difference(set(_dw_matrix.keys())):
                _dw_matrix[_missing_id].append("")
            return [" ".join(text) for _, text in sorted(_dw_matrix.items(), key=lambda item: item[0])]

        @staticmethod
        def _build_document_concept_matrix(
                tfidf_concepts_vectorizer,
                tfidf_concepts_collection,
                tfidf_documents_vectorizer,
                tfidf_documents_collection,
                concept_num,
                doc_range
        ) -> np.ndarray:
            _start, _end = doc_range
            doc_concept_matrix = np.zeros((_end - _start, concept_num))
            _shape = doc_concept_matrix.shape
            logging.info(f"Build array ({_shape[0]}, {_shape[1]}) for documents {_start} to {_end} ...")
            for _doc, _concept in tqdm(itertools.product(range(_start, _end), range(concept_num)),
                                       total=np.prod(_shape)):
                _words_in_doc = tfidf_documents_vectorizer.get_feature_names_out()[
                    np.where(np.asarray(tfidf_documents_collection[_doc].todense())[0] != 0.)[0]]
                _search_idx = np.searchsorted(tfidf_concepts_vectorizer.get_feature_names_out(), _words_in_doc)
                if _search_idx.size <= 0:
                    continue
                while np.max(_search_idx) >= tfidf_concepts_collection[_concept].shape[1]:
                    _search_idx = _search_idx[_search_idx != np.max(_search_idx)]
                _scores_for_concept_doc = tfidf_concepts_collection[_concept].T.todense()[_search_idx]
                doc_concept_matrix[_doc - _start, _concept] = np.sum(_scores_for_concept_doc)
            return doc_concept_matrix

        def build_document_concept_matrix(
                self,
                n_process: int = 1,
                **kwargs
        ) -> None:
            _tfidf_concept_kwargs = {"_".join(key.split("_")[1:]): val for key, val in kwargs.items()
                                     if len(key.split("_")) > 1 and key.split("_")[0] == "concept"}
            _tfidf_document_kwargs = {"_".join(key.split("_")[1:]): val for key, val in kwargs.items()
                                      if len(key.split("_")) > 1 and key.split("_")[0] == "document"}

            from sklearn.feature_extraction.text import TfidfVectorizer as tfidfVec

            tfidf_cv = tfidfVec(**_tfidf_concept_kwargs)
            tfidf_cc = tfidf_cv.fit_transform(self._concept_word_matrix)
            tfidf_dv = tfidfVec(**_tfidf_document_kwargs)
            tfidf_dc = tfidf_dv.fit_transform(self._document_word_matrix)

            _shape = (len(self._document_word_matrix), len(self._concept_word_matrix))

            with ProcessPoolExecutor(n_process) as executor:
                _iter = executor.map(
                    self._build_document_concept_matrix,
                    repeat(tfidf_cv), repeat(tfidf_cc), repeat(tfidf_dv), repeat(tfidf_dc), repeat(_shape[1]),
                    ((_split * int(_shape[0] / n_process), (_split + 1) * int(_shape[0] / n_process)
                    if _split < n_process - 1 else ((_split + 1) * int(_shape[0] / n_process) + _shape[0] % n_process))
                     for _split in range(n_process))
                )
                self._document_concept_matrix = np.concatenate(tuple(i for i in _iter), axis=0)

    class _ConceptGraphClustering:
        def __init__(
                self,
                outer_instance: 'WordEmbeddingClustering'
        ):
            self._outer_instance = outer_instance
            self._sentence_embed = outer_instance._sentence_embed_obj
            self._data_proc = outer_instance._sentence_embed_obj.data_processing_obj
            self._concept_graphs = None
            self._document_concept_matrix = None

        @property
        def document_concept_matrix(
                self
        ) -> np.ndarray:
            return self._document_concept_matrix

        def get_norm_document_concept_matrix(
                self,
                norm='l2',
                **kwargs
        ) -> np.ndarray:
            if "X" in kwargs:
                kwargs.pop("X")
            if self.document_concept_matrix.shape[1] > 0:
                return normalize(self.document_concept_matrix, norm=norm, **kwargs)

        l2_norm_document_concept_matrix = property(get_norm_document_concept_matrix)

        def _filter_entries(self, iter: Iterable, filter_list: Optional[
            list] = None):  # ToDo: use vectorized methods since feature_names are already ndarrays
            if filter_list is None:
                return iter
            return [_idx for _idx in iter
                    if self._sentence_embed.data_processing_obj.data_chunk_sets[_idx][
                        self._outer_instance._text_field_value] in filter_list]

        @cache
        def _concept_clusters(
                self,
                cluster_distance: float = 0.6,
                cluster_min_size: Union[float, int] = 1,
                exclusion_ids: Optional[tuple] = None,
                restrict_to_cluster: bool = False,
        ) -> Iterable[List[int]]:
            _meaningful_clusters = []
            tfidf_filter = self._data_proc.tfidf_filter
            if tfidf_filter is not None: # ToDo need to check whether filtering is enabled!
                logging.info("Filtering phrases")
            for i, _center in enumerate(self._outer_instance._cluster_obj.cluster_centers_):
                if i in (self._outer_instance._exclusion_ids if exclusion_ids is None else exclusion_ids):
                    continue
                if restrict_to_cluster:  # ToDo: results in worse figures when using same parameters...
                    # restrict embeddings to those that are in the cluster
                    _idx = np.where(self._outer_instance._cluster_obj.labels_ == i)
                    _meaningful_clusters.append(
                        self._filter_entries(
                            _idx[0][  # get the original indices
                                top_k_cosine(
                                    _center, self._sentence_embed.sentence_embeddings[_idx],
                                    distance=cluster_distance, vector_dim=self._sentence_embed.embedding_dim
                                )
                            ],
                            tfidf_filter.get_feature_names_out().tolist() if tfidf_filter is not None else None
                        )
                    )
                else:
                    _meaningful_clusters.append(
                        self._filter_entries(
                            top_k_cosine(
                                _center, self._sentence_embed.sentence_embeddings,
                                distance=cluster_distance, vector_dim=self._sentence_embed.embedding_dim
                            ),
                            tfidf_filter.get_feature_names_out().tolist() if tfidf_filter is not None else None
                        )
                    )

            # ToDo: right now works just for int
            return [_cluster for _cluster in _meaningful_clusters if len(_cluster) >= cluster_min_size]

        @cache
        def _build_graph(
                self,
                cluster: Tuple[int],
                graph_cosine_weight: float = .5,
                graph_merge_threshold: float = .95,
                graph_weight_cut_off: float = .5
        ) -> nx.Graph:
            gc = GraphCreator(chunk_set_dict=self._data_proc.data_chunk_sets,
                              embeddings=self._sentence_embed.sentence_embeddings)
            graph = gc.build_graph_from_cluster(cluster=cluster, weight_on_cosine=graph_cosine_weight,
                                                merge_threshold=graph_merge_threshold,
                                                weight_cut_off=graph_weight_cut_off)
            return graph

        def _graph_list(
                self,
                graph_simplify: Optional[float],
                graph_unroll: bool,
                graph_simplify_alg: str,
                graph_sub_clustering: bool
        ) -> List[nx.Graph]:
            # Todo: some method to incorporate sub_clustering (e.g. adding weight/significance) when not unrolling
            _graph_list_gen = ((g, g) for g in self._concept_graphs)

            if graph_simplify_alg not in ['weight', 'significance']:
                raise ValueError(f"Parameter value '{graph_simplify_alg}' for 'graph_simplify_alg' not valid. Choose"
                                 f"one of 'weight, significance'.")

            if graph_simplify_alg == 'significance' and graph_simplify:
                mlf = unimodal.MLF(directed=False)
                _graph_list_gen = (
                    (mlf.fit_transform(g, weight_as_percentile=True),
                     g) for g in self._concept_graphs
                )

            if graph_simplify:
                _graph_list_gen = (
                    (simplify_graph_naive(g, gamma=graph_simplify, n_graph=None, weight=graph_simplify_alg),
                     g_ref) for i, (g, g_ref) in enumerate(_graph_list_gen) if len(g.edges) > 0
                )
            if graph_unroll:
                _graph_list_gen = (
                    (unroll_graph(
                        graph=simplify_graph_naive(g, gamma=graph_simplify, n_graph=None) if graph_simplify else g,
                        reference_graph=g,
                        weight=graph_simplify_alg),
                     g_ref) for i, (g, g_ref) in enumerate(_graph_list_gen) if len(g.edges) > 0
                )
                if graph_sub_clustering:
                    _graph_list_gen = (
                        (sub_clustering(g, g_ref),
                         g_ref) for g, g_ref in _graph_list_gen
                    )

            return [g for g, _ in tqdm(_graph_list_gen, total=len(self._concept_graphs)) if len(g.edges) > 0]

        def _calculate_connection_alg4(
                self,
                concept_graphs: List[nx.Graph],
                weight: str = 'weight',
                normalize: bool = False
        ):
            logging.info("Algorithm 4")
            _tfidf_filter = tfidfVec(min_df=1, max_df=1.0, stop_words=None, analyzer=lambda x: re.split(self._data_proc._chunk_boundary, x))
            _tfidf_filter_vec = _tfidf_filter.fit_transform(self._data_proc.document_chunk_matrix).todense()
            _tfidf_vocab = _tfidf_filter.vocabulary_
            _scaler = MinMaxScaler()
            _tfidf_filter_vec_norm = _scaler.fit_transform(_tfidf_filter_vec)

            _dump_list = []
            for j, concept_graph in tqdm(enumerate(concept_graphs), total=len(concept_graphs)):
                concept_graph: nx.Graph
                _graph = concept_graph.copy(as_view=False)
                for (nid, ndict) in concept_graph.nodes(data=True):
                    _graph.add_weighted_edges_from(((nid, f"d{d}", _tfidf_filter_vec_norm[d, _tfidf_vocab[ndict['label']]])
                                                    for d in ndict['documents']))
                _doc_array = []
                _doc_eigen_array = []
                for d, v in nx.pagerank_numpy(_graph, weight=weight).items():
                    if not (isinstance(d, str) and d.startswith('d')):
                        continue
                    _doc_array.append(int(d[1:]))
                    _doc_eigen_array.append(v)
                _doc_array = np.asarray(_doc_array)
                _doc_eigen_array = np.asarray(_doc_eigen_array)
                if normalize:
                    _doc_eigen_array *= (1.0/_doc_eigen_array.max())
                self._document_concept_matrix[_doc_array, j] += _doc_eigen_array

                _dump_list.append(_graph)
            pickle.dump(_dump_list, pathlib.Path("graph_dump.pickle").open('wb'))

        def _calculate_connection_alg3(
                self,
                concept_graphs: List[nx.Graph],
                cutoff: float = .5
        ):
            logging.info("Algorithm 3")
            for j, concept_graph in tqdm(enumerate(concept_graphs), total=len(concept_graphs)):
                louvain = skn.clustering.Louvain(modularity='Newman', sort_clusters=False)
                documents = np.array([n[1]['documents'] for n in concept_graph.nodes(data=True)])
                concept_graph_matrix = nx.to_numpy_array(concept_graph)
                graph_cluster = louvain.fit_transform(concept_graph_matrix)
                # min_max = MinMaxScaler()
                # ToDo: only works if significance was calculated
                # ToDo: think about this again
                concept_graph_matrix_significance = nx.to_numpy_array(concept_graph, weight="significance") * (
                        1.0 / nx.to_numpy_array(concept_graph, weight="significance").max())
                concept_graph_matrix_rev = (concept_graph_matrix > 0) - concept_graph_matrix
                try:
                    distance, predecessors = shortest_path(concept_graph_matrix_rev, return_predecessors=True, directed=False)
                except NegativeCycleError:  #ToDo: don't know if this is appropriate: it would skip th whole concept graph
                    continue
                bool_cut = ((distance <= cutoff) & (distance > 0))
                absolute_distance = construct_dist_matrix(np.ones(concept_graph_matrix.shape), predecessors, directed=False)
                distance_real = construct_dist_matrix(concept_graph_matrix, predecessors, directed=False)
                for i in range(concept_graph_matrix.shape[0]):
                    _gc_id = graph_cluster[i]
                    _idx = np.where(bool_cut[i, :])
                    _scores = ((distance_real[i, :][_idx] / np.exp(absolute_distance[i, :][_idx]))
                               * (graph_cluster[_idx] == _gc_id)
                               )  # product: only count score if both nodes are in same sub cluster: graph_cluster
                    self._document_concept_matrix[
                        list(set(_d for d in documents[_idx] for _d in d)), j] += np.sum(_scores)
                # ToDo: now concept_dist = construct_dist_matrix(concept_graph_matrix, predecessors, directed=False)
                # ToDo: ==> iterate over all rows
                # ToDo: np.where(bool_cut[i,:]) -> gives indices where there is a connection according to cutoff
                # ToDo: concept_dist[i,:] -> use this + former line result to get conn strength (scores) and node
                # ToDo: from nodes indices get document sets and add respective scores
                # concept_graph: nx.Graph
                # G = concept_graph.copy(as_view=False)
                # # reverse values for weight that high-weighted edges in subsequent dijkstra are favored
                # for e1, e2, d in concept_graph.edges(data=True):
                #     G[e1][e2]['weight'] = 1 - d['weight']
                # for n in concept_graph.nodes():
                #     for neighbor, distance in nx.dijkstra_predecessor_and_distance(G, n, cutoff=cutoff)[1].items():
                #         if n == neighbor:
                #             continue
                #         all_docs = [d for d in set(concept_graph.nodes(data=True)[n]['documents']).union(
                #             set(concept_graph.nodes(data=True)[neighbor]['documents']))]
                #         weight, path = nx.single_source_dijkstra(G, n, neighbor, weight="weight")
                #         n_edges = len(path) - 1
                #         self._document_concept_matrix[all_docs, j] = sum(
                #             [(concept_graph.edges[p]["weight"] / (i + 1)) for i, p in
                #              enumerate(pairwise(path)) if concept_graph.has_edge(*p)]
                #         )

        def _calculate_connection_alg1(
                self,
                concept_graphs: List[nx.Graph]
        ):
            logging.info("Iterating over edges ...")
            for j, concept_graph in tqdm(enumerate(concept_graphs), total=len(concept_graphs)):
                if not len(concept_graph.edges) > 0:
                    continue
                for n1, n2, edge in concept_graph.edges(data=True):
                    all_docs = [d for d in set(concept_graph.nodes(data=True)[n1]['documents']).union(
                        set(concept_graph.nodes(data=True)[n2]['documents']))]
                    self._document_concept_matrix[all_docs, j] += edge["weight"]

        def _calculate_connection_alg2(
                self,
                concept_graphs: List[nx.Graph],
                distance: int = 2,
                gamma: float = .5,
                sub_cluster_reward: float = 1.75,
                weight: str = 'weight'
        ):
            logging.info("Iterating over nodes...")
            for j, concept_graph in tqdm(enumerate(concept_graphs), total=len(concept_graphs)):
                _connected_nodes = defaultdict(set)
                _mean_w_graph = statistics.mean(d.get(weight, .0) for _, _, d in concept_graph.edges(data=True))
                for pos, (node, n_data) in enumerate(concept_graph.nodes(data=True)):
                    neighbors = ((n, d) for d in range(1, distance + 1)
                                 for n in nx.descendants_at_distance(concept_graph, node, d))
                    for target, dist in neighbors:
                        if target in _connected_nodes and node in _connected_nodes[target]:
                            continue
                        for path in nx.all_simple_edge_paths(concept_graph, node, target, dist):
                            _w_sum = 0
                            for i, path_elem in enumerate(path):
                                # _w = concept_graph.get_edge_data(path_elem[0], path_elem[1])[path_elem[2]]["weight"]
                                _edge_data = concept_graph.get_edge_data(path_elem[0], path_elem[1])
                                _w = _edge_data.get(weight, (
                                    0.0 if not _edge_data.get("sub_cluster", False) else _mean_w_graph))
                                # _w_sum += _w / np.log2(i + 2)
                                # ToDo: how to enhance the weight when two nodes are connected by sub_cluster
                                _w_sum += ((_w if not _edge_data.get("sub_cluster", False) else _w * sub_cluster_reward)
                                          / (i * gamma + 1))
                                # _w_sum += ((2 * _w / (1 + np.exp(-2*i))) - 1)
                            try:
                                all_docs = [d for d in set(concept_graph.nodes(data=True)[node]['documents']).union(
                                    set(concept_graph.nodes(data=True)[target]['documents']))]
                            except KeyError as err:
                                logging.warning(f"(node)   \t{concept_graph.nodes(data=True)[node]}\n"
                                                f"(target) \t{concept_graph.nodes(data=True)[target]}")
                                raise
                            self._document_concept_matrix[all_docs, j] += _w_sum / len(path)
                        _connected_nodes[node].add(target)

        def build_concept_graphs(
                self,
                cluster_distance: float = 0.6,
                cluster_min_size: Union[float, int] = 1,
                cluster_exclusion_ids: Optional[list] = None,
                graph_cosine_weight: float = .5,
                graph_merge_threshold: float = .95,
                graph_weight_cut_off: float = .5,  # edges where weight is smaller than this value are cut
                graph_simplify: Optional[float] = .5,
                graph_simplify_alg: str = 'weight',  # ToDo: class enumeration for simplify alg
                graph_unroll: bool = True,
                graph_sub_clustering: Union[float, bool] = False,
                connection_distance: int = 2,
                restrict_to_cluster: bool = False,
                filter_min_df: Union[int, float] = 1,
                filter_max_df: Union[int, float] = 1.,
                filter_stop: Optional[list] = None,
        ):
            filter_stop = filter_stop if (filter_stop not in [None, False]) else []
            if (self._data_proc.tfidf_filter is not None and (
                    self._data_proc.tfidf_filter.get_params().get("min_df", -1) != filter_min_df or
                    self._data_proc.tfidf_filter.get_params().get("max_df", -1) != filter_max_df or
                    self._data_proc.tfidf_filter.get_params().get("stop_words", -1) != filter_stop)) or (
                    self._data_proc.tfidf_filter is None and (
                    filter_min_df != 1 or filter_max_df != 1. or filter_stop is not None)):
                logging.info(
                    f"Resetting tfidf filter with min_df: {filter_min_df}, max_df: {filter_max_df}, stopwords: {filter_stop}")
                self._data_proc.reset_filter(filter_min_df=filter_min_df, filter_max_df=filter_max_df,
                                             filter_stop=filter_stop)
            logging.info(f"Building Document Concept Matrix with following arguments:\n{locals()}")
            _exclusion = self._outer_instance._exclusion_ids if cluster_exclusion_ids is None else cluster_exclusion_ids
            _tqdm_sum = (len(self._outer_instance._cluster_obj.cluster_centers_) - len(_exclusion))
            logging.info(f"Building Concept Graphs... (exclusion_ids: {_exclusion})")
            self._concept_graphs = [
                self._build_graph(tuple(_c), graph_cosine_weight, graph_merge_threshold, graph_weight_cut_off)
                for _c in tqdm(self._concept_clusters(cluster_distance, cluster_min_size,
                                                      tuple(_exclusion), restrict_to_cluster=restrict_to_cluster),
                               total=_tqdm_sum)
            ]

            if graph_simplify is not None:
                logging.info(f"Cutting edges ({graph_simplify_alg})...")
            return self._graph_list(graph_simplify=graph_simplify, graph_unroll=graph_unroll,
                                    graph_simplify_alg=graph_simplify_alg,
                                    graph_sub_clustering=True if isinstance(graph_sub_clustering, float) else False)

        def build_document_concept_matrix(
                self,
                graph_unroll: bool = True,
                graph_sub_clustering: Union[float, bool] = False,
                graph_distance_cutoff: float = .5,
                **kwargs
        ):
            kwargs["graph_unroll"] = graph_unroll
            kwargs["graph_sub_clustering"] = graph_sub_clustering
            kwargs["graph_distance_cutoff"] = graph_distance_cutoff
            _concept_graphs = self.build_concept_graphs(**kwargs)

            self._document_concept_matrix = np.zeros((self._data_proc.documents_n, len(_concept_graphs)))
            sub_cluster_reward = graph_sub_clustering if isinstance(graph_sub_clustering, float) else (
                1.75 if graph_sub_clustering is True else 1.0)

            # ToDo: best approach to evaluate the connection between documents via their concepts needs to be found
            logging.info("Calculating connections...")
            if not graph_unroll:
                # self._calculate_connection_alg1(_concept_graphs)
                # self._calculate_connection_alg4(_concept_graphs)
                self._calculate_connection_alg3(_concept_graphs, cutoff=graph_distance_cutoff)
            else:
                # ToDo: this one only when graphs are unrolled!
                # self._calculate_connection_alg2(concept_graphs=_concept_graphs, distance=connection_distance, gamma=.5,
                #                                 sub_cluster_reward=sub_cluster_reward)
                # self._calculate_connection_alg4(_concept_graphs)
                self._calculate_connection_alg3(_concept_graphs, cutoff=graph_distance_cutoff)

    # ToDo: cache this? -- doesnt really matter for low dimensional matrices but might be an issue when bigger
    @staticmethod
    def _create_clustering_obj(
            clustering_obj: Any,
            **kwargs
    ) -> Any:
        _cluster_alg_kwargs = {"_".join(key.split("_")[1:]): val for key, val in kwargs.items()
                               if len(key.split("_")) > 1 and key.split("_")[0] == "cluster"}
        return clustering_obj(**_cluster_alg_kwargs)

    def _get_scoring_cluster(
            self,
            embeddings_cluster_obj,
            clustering_obj,
            min_concepts: int = 10,  # ToDo: There were cases when nearly no meaningful concepts were created
            **kwargs
    ):
        _matrix = (embeddings_cluster_obj.l2_norm_document_concept_matrix if "norm" not in kwargs
                   else embeddings_cluster_obj.get_norm_document_concept_matrix(**kwargs))
        if _matrix is None or _matrix.shape[1] < min_concepts:
            return None, None
        _non_zero = np.any(_matrix, axis=1)
        _cluster = self._create_clustering_obj(clustering_obj, **kwargs).fit(_matrix[np.where(_non_zero)])
        return _cluster, _non_zero

    def ari_score(
            self,
            embeddings_cluster_obj: Union[_WEClustering, _ConceptGraphClustering],
            clustering_obj: [KMeans, AgglomerativeClustering],
            min_concepts: int = 10,
            **kwargs
    ):
        kwargs["cluster_n_clusters"] = len(set(embeddings_cluster_obj._data_proc.true_labels_vec))
        _cluster, _non_zero = self._get_scoring_cluster(embeddings_cluster_obj, clustering_obj, min_concepts, **kwargs)
        if _cluster is None or _non_zero is None:
            return 0.0
        return adjusted_rand_score(
            np.asarray(embeddings_cluster_obj._data_proc.true_labels_vec)[_non_zero],
            _cluster.labels_)

    def purity_score(
            self,
            embeddings_cluster_obj: Union[_WEClustering, _ConceptGraphClustering],
            clustering_obj: [KMeans, AgglomerativeClustering],
            min_concepts: int = 10,
            **kwargs
    ):
        kwargs["cluster_n_clusters"] = len(set(embeddings_cluster_obj._data_proc.true_labels))
        _cluster, _non_zero = self._get_scoring_cluster(embeddings_cluster_obj, clustering_obj, min_concepts, **kwargs)
        if _cluster is None or _non_zero is None:
            return 0.0
        counter = {}
        for c in range(_cluster.n_clusters):
            counter[c] = Counter()
            for i in np.asarray(embeddings_cluster_obj._data_proc.true_labels)[_non_zero][np.where(
                    _cluster.labels_ == c)]:
                counter[c].update({i: 1})

        df = pd.DataFrame.from_records([counter[i] for i in range(len(counter))])
        df.fillna(0, inplace=True)
        return df.max(axis=1).to_numpy().sum() / df.to_numpy().sum()


class PhraseClusterFactory:
    @staticmethod
    def create(
            sentence_embeddings: Union[SentenceEmbeddingsFactory.SentenceEmbeddings, np.ndarray],
            cache_path: pathlib.Path,
            cache_name: str,
            cluster_algorithm: str = 'kmeans',
            down_scale_algorithm: str = 'umap',
            cluster_by_down_scale: bool = True,
            **kwargs
    ):
        _cluster_obj = PhraseClusterFactory.PhraseCluster(
            sentence_embeddings=sentence_embeddings,
            cluster_algorithm=cluster_algorithm,
            down_scale_algorithm=down_scale_algorithm,
            cluster_by_down_scale=cluster_by_down_scale,
            **kwargs
        )
        save_pickle(_cluster_obj, (cache_path / pathlib.Path(f"{cache_name}.pickle")))
        return _cluster_obj

    @staticmethod
    def load(
            data_obj_path: Union[pathlib.Path, str],
    ):
        _data = load_pickle(pathlib.Path(data_obj_path).resolve())
        assert isinstance(_data, PhraseClusterFactory.PhraseCluster)
        return _data

    class PhraseCluster:
        """

        **kwargs for the cluster algorithm, the scaling algorithm and the k-elbow algorithm have to be prefixed
        with 'clustering_', 'scaling_' & 'kelbow_' respectively
        """

        def __init__(
                self,
                sentence_embeddings: Union[SentenceEmbeddingsFactory.SentenceEmbeddings, np.ndarray],
                cluster_algorithm: str = 'kmeans',
                down_scale_algorithm: str = 'umap',
                cluster_by_down_scale: bool = True,
                **kwargs
        ):
            self._clustering_estimators_ref = {"kmeans": KMeans,
                                               "kmeans-mb": MiniBatchKMeans,
                                               "affinity-prop": AffinityPropagation}
            self._cluster_alg_kwargs = {"_".join(key.split("_")[1:]): val for key, val in kwargs.items()
                                        if len(key.split("_")) > 1 and key.split("_")[0] == "clustering"}
            self._down_scale_alg_kwargs = {"_".join(key.split("_")[1:]): val for key, val in kwargs.items()
                                           if len(key.split("_")) > 1 and key.split("_")[0] == "scaling"}
            self._kelbow_alg_kwargs = {"_".join(key.split("_")[1:]): val for key, val in kwargs.items()
                                       if len(key.split("_")) > 1 and key.split("_")[0] == "deduction"}
            # if self._kelbow_alg_kwargs.get("estimator", False):
            #     if self._kelbow_alg_kwargs.get("estimator") != "affinity-prop":
            #         self._kelbow_alg_kwargs["estimator"] = self._clustering_estimators_ref.get(self._kelbow_alg_kwargs.get("estimator"))
            #     else:
            #         self._kelbow_alg_kwargs.pop("estimator")
            self._sentence_emb = sentence_embeddings.sentence_embeddings if not isinstance(sentence_embeddings,
                                                                                           np.ndarray) else sentence_embeddings
            self._cluster_alg = (cluster_algorithm if cluster_algorithm in ["kmeans", "kmeans-mb", "affinity-prop"]
                                 else "kmeans")
            self._cluster_obj = self._clustering_estimators_ref[self._cluster_alg]
            self._down_scale_alg = down_scale_algorithm

            if self._down_scale_alg_kwargs.get("n_neighbors", False) and isinstance(self._down_scale_alg_kwargs.get("n_neighbors"), float):
                _n_neighbors = min(self._down_scale_alg_kwargs["n_neighbors"] * self._sentence_emb.shape[0], 100)
                self._down_scale_alg_kwargs["n_neighbors"] = int(_n_neighbors)

            self._down_scale_obj = {"umap": umap.UMAP, None: NoneDownScaleObj}[down_scale_algorithm](**self._down_scale_alg_kwargs)
            self._concept_cluster = None
            self._kelbow = None

            self._build_concept_cluster(cluster_by_down_scale)

        @property
        def concept_cluster(
                self
        ):
            return self._concept_cluster

        @property
        def get_params(
                self
        ):
            return {
                f"scaling ({self._down_scale_alg})": self._down_scale_alg_kwargs,
                f"cluster ({self._cluster_alg})": self._cluster_alg_kwargs,
                f"deduction": self._kelbow_alg_kwargs
            }

        def _build_concept_cluster(
                self,
                cluster_by_down_scale: bool = True
        ):
            logging.info("Building Concept Cluster ...")
            _cluster_embeddings = self._sentence_emb

            if cluster_by_down_scale and not isinstance(self._down_scale_obj, NoneDownScaleObj):
                logging.info(f"{self._down_scale_alg.upper()} arguments: {self._down_scale_obj.get_params()}")
                _cluster_embeddings = self._down_scale_obj.fit_transform(self._sentence_emb)

            _estimator = self._kelbow_alg_kwargs.get("estimator", False)
            if ((_estimator and (_estimator != AffinityPropagation)) or
                    (not _estimator and self._cluster_alg != "affinity-prop" and _estimator is not None)):

                _n_clusters_given = self._cluster_alg_kwargs.get("n_clusters", False)
                if self._kelbow_alg_kwargs.get("enabled", False):
                    _deduction_kwargs = self._kelbow_alg_kwargs.copy()
                    _default_args = inspect.getfullargspec(ClusterNumberDetector).args
                    for _k in _deduction_kwargs.copy().keys():
                        if _k not in _default_args:
                            _deduction_kwargs.pop(_k, None)
                    cnd = ClusterNumberDetector(
                        self,
                        **_deduction_kwargs
                    )
                    cnd.estimate_cluster_number(_cluster_embeddings)
                else:
                    #ToDo: her some default value not hard coded
                    self._cluster_alg_kwargs['n_clusters'] = _n_clusters_given if _n_clusters_given else 50

                logging.info("-- Clustering ...")
                logging.info(f" ({self._cluster_alg}) with Arguments: {self._cluster_alg_kwargs}\n"
                             f"Number of Clusters: {self._cluster_alg_kwargs['n_clusters']}\n")
                self._concept_cluster = self._cluster_obj(**self._cluster_alg_kwargs).fit(
                    # self._sentence_emb #ToDo: why did I have this here (the original sentence embeddings and not the down-scaled ones)
                    _cluster_embeddings
                )
            else:
                logging.info("-- Clustering ...")
                logging.info(f" ({self._cluster_alg}) with Arguments: {self._cluster_alg_kwargs}")
                self._concept_cluster = self._cluster_obj(**self._cluster_alg_kwargs).fit(
                    _cluster_embeddings
                )


class ClusterNumberDetector:
    def __init__(
            self,
            owner: PhraseClusterFactory.PhraseCluster,
            algorithm: ClusterNumberDetection = ClusterNumberDetection.SILHOUETTE,
            k_min: int = 2,
            k_max: int = 100,
            n_samples: int = 15,
            sample_fraction: int = 25,
            regression_poly_degree: int = 5
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

        x_lin = np.linspace(np.asarray(x_reg).min(), np.asarray(x_reg).max(), self._k_max)
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
            _samples.append(sample_without_replacement(projection.shape[0], int(projection.shape[0] / self._sample_fraction)))

        for _sample in tqdm(_samples):
            _kelbow = kelbow_visualizer(
                model=MiniBatchKMeans(n_init='auto'),
                X=projection[_sample],
                show=False,
                k=(self._k_min, self._k_max),
                metric={
                    ClusterNumberDetection.SILHOUETTE: 'silhouette',
                    ClusterNumberDetection.DISTORTION: 'distortion',
                    ClusterNumberDetection.CALINSKI_HARABASZ: 'calinski_harabasz',
                }.get(self._algorithm, ClusterNumberDetection.SILHOUETTE),
            )
            _kelbow_array.append(_kelbow)

        for _kelbow in _kelbow_array:
            x_vals, y_regression, max_regression = self._fit_regression(_kelbow.k_values_, _kelbow.k_scores_)
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
        self._owner._cluster_alg_kwargs["n_clusters"] = int(self._k_elbow_estimation(projection))
        # if self._cluster_obj in [KMeans, MiniBatchKMeans] and "n_init" not in _cluster_args.keys():
        #     _cluster_args["n_init"] = 'auto'
