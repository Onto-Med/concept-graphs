from collections.abc import Iterable
from functools import lru_cache
from typing import Any

import networkx as nx
import numpy as np
from fuzzywuzzy import fuzz
from numpy import ndarray

from src.core.documents import add_offset_to_documents_dicts_by_id
from src.core.embedding_functions import cosine


class GraphCreator:
    def __init__(
        self,
        chunk_set_dict: list[dict],
        embeddings: np.ndarray,
        doc_text_value: str = "doc",
    ) -> None:
        self.chunk_set_dict = chunk_set_dict
        self.embeddings = embeddings
        self.doc_text_value = doc_text_value
        self.doc_count_array = np.asarray(
            [len(d[self.doc_text_value]) for d in chunk_set_dict]
        )

    @lru_cache
    def _chunks_as_np_array(self, text_value: str = "text") -> np.ndarray:
        return np.asarray([i[text_value] for i in self.chunk_set_dict])

    @lru_cache
    def _indices_and_some(
        self, cluster: tuple[int]
    ) -> tuple[ndarray, Iterable | tuple, Any]:
        # ToDo: think about batching this somehow -> when _cluster is relatively large, combinations (triu_indices)
        #  get's really large and memory allocation issues might occur when indexing self.embeddings (in _cosine_adj)
        _cluster = sorted(cluster)
        adj_matrix = np.zeros((len(_cluster), len(_cluster)), dtype=float)

        # Calculate indices for upper triangle
        tri = np.triu_indices(len(_cluster), 1)
        # Create combinations of upper triangle positions
        idx = np.asarray(_cluster)[np.transpose(tri)]

        return adj_matrix, tri, idx

    @lru_cache
    def _str_sim_adj(
        self,
        cluster: tuple[int],
        similarity_add: float = 0.05,
        merge_threshold: float | None = 0.95,
        text_value: str = "text",
        fuzzy_matching: str = "token_sort_ratio",
    ) -> dict:
        _fuzzy_matching = {
            "token_sort_ratio": fuzz.token_sort_ratio,
            "token_set_ratio": fuzz.token_set_ratio,
            "partial_ratio": fuzz.partial_ratio,
        }
        if _fuzzy_matching.get(fuzzy_matching, False):
            fuzzy_matching = "token_sort_ratio"
        _adj_matrix, _tri, _idx = self._indices_and_some(cluster)

        # Calculate string similarity for combinations and assigns them to upper triangle
        _adj_matrix[_tri] = (
            np.asarray(
                [
                    _fuzzy_matching[fuzzy_matching](*_arr)
                    for _arr in self._chunks_as_np_array(text_value)[_idx]
                ]
            )
            / 100
            + similarity_add
        )

        # Merge/Collapse phrases that have a string similarity higher than merge_threshold
        _cluster = sorted(cluster)
        if merge_threshold:
            _merge_pos = np.stack(
                np.asarray(_adj_matrix >= merge_threshold).nonzero(), axis=1
            )
            _merge_ids = np.asarray(_cluster)[_merge_pos]
            _top_docs = np.argmax(self.doc_count_array[_merge_ids], axis=1)
            _rem_ids = np.take_along_axis(_merge_pos, (_top_docs - 1)[:, None], axis=1)[
                :, 0
            ]
            _add_ids = np.take_along_axis(_merge_pos, _top_docs[:, None], axis=1)[:, 0]

            _adj_matrix_collapsed = np.delete(
                np.delete(_adj_matrix, _rem_ids, axis=0), _rem_ids, axis=1
            )
            _cluster_collapsed = sorted(np.delete(_cluster, _rem_ids))
            _merged_docs_add_id = self._combine_docs(_cluster, _add_ids, _rem_ids)

            return {
                "adjacency_matrix": _adj_matrix_collapsed,
                "sorted_cluster": _cluster_collapsed,
                "documents": [
                    (
                        self.chunk_set_dict[_id][self.doc_text_value]
                        if _cluster.index(_id) not in _add_ids
                        else list(
                            _merged_docs_add_id[
                                np.nonzero(_add_ids == _cluster.index(_id))[0][0]
                            ]
                        )
                    )
                    for _id in _cluster_collapsed
                ],
            }
        return {
            "adjacency_matrix": _adj_matrix,
            "sorted_cluster": sorted(np.asarray(cluster)),
            "documents": [
                self.chunk_set_dict[_id][self.doc_text_value]
                for _id in sorted(np.asarray(cluster))
            ],
        }

    @lru_cache
    def _cosine_adj(self, cluster: tuple[int]) -> np.ndarray:
        _adj_matrix, _tri, _idx = self._indices_and_some(cluster)

        # Calculate cosine similarity for combinations and assigns them to upper triangle
        _adj_matrix[_tri] = [cosine(x[0], x[1]) for x in self.embeddings[_idx]]

        # Copy upper triangle to lower triangle -- Don't need it for importing in networkx; it only uses upper triangle
        # _adj_matrix = _adj_matrix + _adj_matrix.T - np.diag(np.diag(_adj_matrix))

        return _adj_matrix

    def _combine_docs(
        self, cluster: list, add_ids: np.ndarray, rem_ids: np.ndarray
    ) -> list:
        return_list = []
        for i, _id in enumerate(sorted(rem_ids)):
            _rem_docs: list[dict] = self.chunk_set_dict[cluster[_id]][
                self.doc_text_value
            ]
            _add_docs: list[dict] = self.chunk_set_dict[cluster[add_ids[i]]][
                self.doc_text_value
            ].copy()
            for doc in _rem_docs:
                for _offset in doc.get("offsets", []).copy():
                    add_offset_to_documents_dicts_by_id(_add_docs, doc["id"], _offset)
            return_list.append(_add_docs)
        return return_list

        # return [set(_rem_docs + _add_docs) for i, _id in enumerate(sorted(rem_ids))]

    # ToDo: some form of correction parameter?
    #    Right now, for instance, if not the majority of docs for for each phrase share the same cluster,
    #    the final weight will be decreased
    # ToDo: insert fuzzy_matching parameter in calling methods/functions
    def build_graph_from_cluster(
        self,
        cluster: Iterable[int],
        similarity_add: float = 0.05,
        weight_on_cosine: float = 0.5,
        merge_threshold: float | None = 0.95,
        weight_cut_off: float | None = 0.5,
        text_value: str = "text",
        fuzzy_matching: str = "token_sort_ratio",
    ) -> nx.Graph:
        _string_sim_result = self._str_sim_adj(
            tuple(sorted(cluster)),
            merge_threshold=merge_threshold,
            similarity_add=similarity_add,
            text_value=text_value,
            fuzzy_matching=fuzzy_matching,
        )
        _adj_matrix_str_sim = _string_sim_result["adjacency_matrix"]
        _adapt_cluster = _string_sim_result["sorted_cluster"]
        _documents = _string_sim_result["documents"]

        _adj_matrix_cosine = self._cosine_adj(tuple(sorted(_adapt_cluster)))

        # Remove nodes where edge weight is smaller than weight_cut_off
        # ToDo: need to think abut this one!? Either weight is exactly weight of cosine (w_on_cosine == 1)
        #  or weight of string_similarity (w_on_cosine == 0)
        _final_adj_matrix = ((1 - weight_on_cosine) * _adj_matrix_str_sim) + (
            weight_on_cosine * _adj_matrix_cosine
        )
        if weight_cut_off:
            _final_adj_matrix[
                np.where(
                    np.logical_and(
                        _final_adj_matrix < weight_cut_off, _final_adj_matrix > 0.0
                    )
                )
            ] = 0.0

        # Build graph from adjacency matrix
        graph = nx.from_numpy_array(A=_final_adj_matrix, create_using=nx.Graph)
        # Relabel the nodes to conform to the indices in the embedding/chunk_set_dict
        nx.relabel_nodes(
            G=graph,
            mapping={n: nn.item() for n, nn in enumerate(_adapt_cluster)},
            copy=False,
        )
        nx.set_node_attributes(
            G=graph,
            values={
                p.item(): {
                    "label": self.chunk_set_dict[p][text_value],
                    "documents": _documents[i],
                }
                for i, p in enumerate(_adapt_cluster)
            },
        )

        return graph
