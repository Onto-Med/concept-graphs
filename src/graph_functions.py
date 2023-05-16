import logging
import random
import statistics
import warnings

import numpy as np
import networkx as nx
from typing import Union, List, Dict, Iterable, Optional, Tuple, Any
from inspect import getfullargspec

from numpy import ndarray

from fuzzywuzzy import fuzz
from functools import lru_cache

from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AffinityPropagation
from tqdm.auto import tqdm

from embedding_functions import cosine_against_collection, cosine


class GraphCreator:
    def __init__(
            self,
            chunk_set_dict: List[Dict],
            embeddings: np.ndarray,
            doc_text_value: str = "doc"
    ) -> None:
        self.chunk_set_dict = chunk_set_dict
        self.embeddings = embeddings
        self.doc_text_value = doc_text_value
        self.doc_count_array = np.asarray([len(d[self.doc_text_value]) for d in chunk_set_dict])

    @lru_cache()
    def _chunks_as_np_array(
            self,
            text_value: str = "text"
    ) -> np.ndarray:
        return np.asarray([i[text_value] for i in self.chunk_set_dict])

    @lru_cache()
    def _indices_and_some(
            self,
            cluster: Tuple[int]
    ) -> Tuple[ndarray, Union[Iterable, Tuple], Any]:
        # ToDo: think about batching this somehow -> when _cluster is relatively large, combinations (triu_indices)
        #  get's really large and memory allocation issues might occur when indexing self.embeddings (in _cosine_adj)
        _cluster = sorted(cluster)
        adj_matrix = np.zeros((len(_cluster), len(_cluster)), dtype=float)

        # Calculate indices for upper triangle
        tri = np.triu_indices(len(_cluster), 1)
        # Create combinations of upper triangle positions
        idx = np.asarray(_cluster)[np.transpose(tri)]

        return adj_matrix, tri, idx

    @lru_cache()
    def _str_sim_adj(
            self,
            cluster: Tuple[int],
            similarity_add: float = 0.05,
            merge_threshold: Union[float, None] = 0.95,
            text_value: str = "text",
            fuzzy_matching: str = "token_sort_ratio"
    ) -> dict:
        _fuzzy_matching = {
            "token_sort_ratio": fuzz.token_sort_ratio,
            "token_set_ratio": fuzz.token_set_ratio,
            "partial_ratio": fuzz.partial_ratio
        }
        if _fuzzy_matching.get(fuzzy_matching, False):
            fuzzy_matching = "token_sort_ratio"
        _adj_matrix, _tri, _idx = self._indices_and_some(cluster)

        # Calculate string similarity for combinations and assigns them to upper triangle
        _adj_matrix[_tri] = np.asarray([_fuzzy_matching[fuzzy_matching](*_arr)
                                        for _arr in self._chunks_as_np_array(text_value)[_idx]]) / 100 + similarity_add

        # Merge/Collapse phrases that have a string similarity higher than merge_threshold
        _cluster = sorted(cluster)
        if merge_threshold:
            _merge_pos = np.stack(np.asarray(_adj_matrix >= merge_threshold).nonzero(), axis=1)
            _merge_ids = np.asarray(_cluster)[_merge_pos]
            _top_docs = np.argmax(self.doc_count_array[_merge_ids], axis=1)
            _rem_ids = np.take_along_axis(_merge_pos, (_top_docs - 1)[:, None], axis=1)[:, 0]
            _add_ids = np.take_along_axis(_merge_pos, _top_docs[:, None], axis=1)[:, 0]

            _adj_matrix_collapsed = np.delete(np.delete(_adj_matrix, _rem_ids, axis=0), _rem_ids, axis=1)
            _cluster_collapsed = sorted(np.delete(_cluster, _rem_ids))
            _merged_docs_add_id = [
                set(self.chunk_set_dict[_cluster[_id]][self.doc_text_value] +
                    self.chunk_set_dict[_cluster[_add_ids[i]]][self.doc_text_value])
                for i, _id in enumerate(sorted(_rem_ids))
            ]

            return {'adjacency_matrix': _adj_matrix_collapsed,
                    'sorted_cluster': _cluster_collapsed,
                    'documents': [
                        self.chunk_set_dict[_id][self.doc_text_value] if _cluster.index(_id) not in _add_ids else
                        list(_merged_docs_add_id[np.nonzero(_add_ids == _cluster.index(_id))[0][0]])
                        for _id in _cluster_collapsed]
                    }
        return {'adjacency_matrix': _adj_matrix,
                'sorted_cluster': sorted(np.asarray(cluster)),
                'documents': [self.chunk_set_dict[_id][self.doc_text_value] for _id in sorted(np.asarray(cluster))]
                }

    @lru_cache()
    def _cosine_adj(
            self,
            cluster: Tuple[int]
    ) -> np.ndarray:
        _adj_matrix, _tri, _idx = self._indices_and_some(cluster)

        # Calculate cosine similarity for combinations and assigns them to upper triangle
        _adj_matrix[_tri] = [cosine(x[0], x[1]) for x in self.embeddings[_idx]]

        # Copy upper triangle to lower triangle -- Don't need it for importing in networkx; it only uses upper triangle
        # _adj_matrix = _adj_matrix + _adj_matrix.T - np.diag(np.diag(_adj_matrix))

        return _adj_matrix

    # ToDo: some form of correction parameter?
    #    Right now, for instance, if not the majority of docs for for each phrase share the same cluster,
    #    the final weight will be decreased
    # ToDo: insert fuzzy_matching parameter in calling methods/functions
    def build_graph_from_cluster(
            self,
            cluster: Iterable[int],
            similarity_add: float = 0.05,
            weight_on_cosine: float = 0.5,
            merge_threshold: Union[float, None] = 0.95,
            weight_cut_off: Union[float, None] = 0.5,
            text_value: str = "text",
            fuzzy_matching: str = "token_sort_ratio"
    ) -> nx.Graph:
        _string_sim_result = self._str_sim_adj(tuple(sorted(cluster)), merge_threshold=merge_threshold,
                                               similarity_add=similarity_add, text_value=text_value,
                                               fuzzy_matching=fuzzy_matching)
        _adj_matrix_str_sim = _string_sim_result['adjacency_matrix']
        _adapt_cluster = _string_sim_result['sorted_cluster']
        _documents = _string_sim_result['documents']

        _adj_matrix_cosine = self._cosine_adj(tuple(sorted(_adapt_cluster)))

        # Remove nodes where edge weight is smaller than weight_cut_off
        # ToDo: need to think abut this one!? Either weight is exactly weight of cosine (w_on_cosine == 1)
        #  or weight of string_similarity (w_on_cosine == 0)
        _final_adj_matrix = ((1 - weight_on_cosine) * _adj_matrix_str_sim) + (weight_on_cosine * _adj_matrix_cosine)
        if weight_cut_off:
            _final_adj_matrix[np.where(
                np.logical_and(_final_adj_matrix < weight_cut_off, _final_adj_matrix > 0.0)
            )] = 0.0

        # Build graph from adjacency matrix
        graph = nx.from_numpy_array(
            A=_final_adj_matrix, create_using=nx.Graph
        )
        # Relabel the nodes to conform to the indices in the embedding/chunk_set_dict
        nx.relabel_nodes(G=graph, mapping={n: nn.item() for n, nn in enumerate(_adapt_cluster)}, copy=False)
        nx.set_node_attributes(G=graph,
                               values={p.item(): {'label': self.chunk_set_dict[p][text_value],
                                                  'documents': _documents[i]}
                                       for i, p in enumerate(_adapt_cluster)})

        return graph


def rank_nodes(
        graph: nx.Graph,
        algorithm='naive',
        **kwargs
) -> dict:
    _alg_funcs = {
        'naive': lambda _g, **_: {n[0]: len(n[1]) for n in _g.adj.items()},
        'page_rank': nx.pagerank
    }
    _kwargs = {k: v for k, v in kwargs.items() if k in getfullargspec(_alg_funcs[algorithm]).args}
    return _alg_funcs[algorithm](graph, **_kwargs)


def unroll_graph(
        graph: nx.Graph,
        reference_graph: Optional[nx.Graph] = None,
        rank_algorithm: str = 'page_rank',
        weight: str = 'weight'
) -> nx.Graph:
    # maximum branching, cuts and reconnects the graph such that it becomes tree-like
    _isolates = list(nx.isolates(graph))
    graph.remove_nodes_from(_isolates)
    _max_branching = nx.maximum_branching(graph, preserve_attrs=True)  # , attr=weight)
    _reference_graph = reference_graph if reference_graph is not None else graph
    nx.set_node_attributes(_max_branching, dict(graph.nodes(data=True)))
    _has_root = False

    while True:  # Todo: hangs occasionally in this while loop!
        _connected_nodes = set()
        _nodes_ranked = rank_nodes(_reference_graph, algorithm=rank_algorithm)  # , weight=weight)
        _branch_comps = sorted(nx.connected_components(_max_branching),  ## <- sort subgraphs by average rank
                               key=lambda x: sum(_nodes_ranked[w] for w in x) / len(
                                   x))  ##  algorithm output (e.g. pagerank)
        if len(_branch_comps) <= 1:
            break

        for _i, _sub_graph in enumerate(_branch_comps):
            if _i == len(_branch_comps) - 1:
                _parent_node = max(_sub_graph,
                                   key=lambda _sg: _nodes_ranked[_sg])
                # _max_branching.nodes()[_parent_node]["parent"] = True
                if not _has_root:
                    # _max_branching.nodes()[_parent_node]["root"] = True
                    _has_root = True
                break
            has_conn = False
            for _parent_node in sorted(_sub_graph,
                                       key=lambda _sg: _nodes_ranked[_sg]):
                if has_conn:
                    break
                for _, _t, _d in sorted(_reference_graph.edges(_parent_node, data=True),
                                        key=lambda items: items[2]['weight'], reverse=True):
                    if _t not in _sub_graph and _t not in _connected_nodes:
                        _max_branching.add_edge(_parent_node, _t, **_d)
                        _connected_nodes.add(_parent_node)
                        # _max_branching.nodes()[_parent_node]["parent"] = True
                        has_conn = True
                        break

        while True:
            try:
                _set = set()
                [_set.update(i) for i in nx.find_cycle(_max_branching)]
                _edge = min(_max_branching.subgraph(_set).edges(data=True),
                            key=lambda items: items[2]['weight'])
                _max_branching.remove_edge(*_edge[:-1])
            except nx.NetworkXNoCycle:
                break
    return _max_branching


def simplify_graph_naive(
        g: nx.Graph,
        gamma: float = 0.5,
        n_graph: Optional[int] = None,
        weight: str = "weight",
        assert_connected: bool = True
) -> nx.Graph:
    """
    gamma: float btw. 0.0 and 1.0 - The higher, the more edges are pruned
    """

    if n_graph is not None:
        logging.info(f"Simplify Graph {n_graph}.")
    h = g.subgraph(max(nx.connected_components(g), key=len)).copy()

    if not 0.0 < gamma < 1.0:
        logging.warning("gamma needs to be between 0.0 and 1.0;"
                        " returning subgraph (if any) of max connected components.")
        return h

    _rand_edge_attr = list(h.edges(data=True))[random.randint(0, len(h.edges) - 1)][2]
    if weight not in _rand_edge_attr.keys():
        raise AttributeError(f"Edges of the graph seem to have no attribute '{weight}' to be pruned by:"
                             f" {_rand_edge_attr}.")

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        _adj = np.asarray(nx.adjacency_matrix(h, weight=weight).todense())

    # calculates the the offset for the upper triangle
    _cut = (_adj.shape[0] * _adj.shape[0]) - int((_adj.shape[0] * _adj.shape[0] - _adj.shape[0]) / 2)
    # returns indices for sorted (ascending) values
    _sorted_idx = np.unravel_index(np.argsort(np.triu(_adj, 1), axis=None)[_cut::], _adj.shape)
    # clears all mirrored connection in the lower triangle
    _adj[np.tril_indices(_adj.shape[0], k=-1)] = .0
    # new sorted idx with all 0.0 referring indices removed (i.e. real edges only)
    _edge_offset_idx = tuple(i[sum(_adj[_sorted_idx] == 0):] for i in _sorted_idx)

    # determine how many edges shall be removed
    _to_remove = int(_adj[_edge_offset_idx].shape[0] * gamma)
    if _to_remove == 0 or _to_remove == _adj[_edge_offset_idx].shape[0]:
        logging.info(f"{_to_remove} from total {_adj[_edge_offset_idx].shape[0]} will be removed. No point in doing.")
        return h

    # remove the n-lowest (_to_remove) weighted edges from _adj
    _cutted_idx = tuple(i[:_to_remove] for i in _edge_offset_idx)
    _adj[_cutted_idx] = 0.0
    # identify unconnected components in the new graph and their corresponding indices
    # (2, [0, 0, 0, 1, 1, 0]) --> 2 unconnected components; nodes 0 to 2 and 5 are one group, 3 and 4 the other
    _n_unconnected, _unconnected_idx = connected_components(_adj, directed=False)
    # restore connections (descending) until all sub-components are connected again
    if _n_unconnected > 1 and assert_connected:
        _reconnected = nx.Graph()
        _reconnected.add_nodes_from(range(_n_unconnected))
        _rev_cutted_idx = tuple(i[::-1] for i in _cutted_idx)
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            _ref_adj = np.asarray(nx.adjacency_matrix(h, weight=weight).todense())
        for i, j in zip(*_rev_cutted_idx):
            _edge = tuple(sorted([_unconnected_idx[i], _unconnected_idx[j]]))
            if (_edge[0] != _edge[1]) and not _reconnected.has_edge(_edge[0], _edge[1]):
                _adj[i, j] = _ref_adj[i, j]
                _reconnected.add_edge(_edge[0], _edge[1])
                if nx.is_connected(_reconnected):
                    break

    # create a new networkx graph from simplified _adj and restore proper values
    new_graph: nx.Graph = nx.from_numpy_array(_adj)
    new_graph = nx.relabel_nodes(new_graph, mapping={i: j for i, j in enumerate(h.nodes())}, copy=True)
    _edge_update = ((u, v, {_key: _val for _key, _val in d.items()}) for u, v, d in h.edges(data=True)
                    if new_graph.has_edge(u, v))
    new_graph.update(nodes=((n, d) for n, d in h.nodes(data=True) if new_graph.has_node(n)),
                     edges=None if weight == "weight" else _edge_update)
    return new_graph


def sub_clustering(
        g: nx.Graph,
        g_reference: nx.Graph,  # the graph that wasn't pruned or unrolled
        gamma: float = .01,  # the factor by which the string similarity is penalized if no weight was recorded in graph
        damping: Optional[float] = None,
        max_iter: Optional[int] = None,
        inplace: bool = False,
        retries: int = 1
) -> Optional[nx.Graph]:
    """
    Returns a copy (or modifies in place) of the graph with annotated sub clusters.
    The exemplar nodes of each sub cluster have a node attribute '{root: True}' whereas the example nodes are just
    connected to their exemplar with an edge with attribute '{sub_cluster: True}'.
    @param g:
    @param g_reference:
    @param gamma:
    @param damping:
    @param max_iter:
    @param inplace:
    @param retries:
    @return:
    """
    if not inplace:
        g_working = g.copy()
    else:
        g_working = g
    words = np.asarray([i[1]["label"] for i in g_reference.nodes(data=True)])
    nodes = np.asarray(g_reference.nodes())

    # calculate the similarity of two phrases by a combination of their string similarity and their cosine similarity
    _similarity = np.array([[
        fuzz.SequenceMatcher(None, w1, w2).ratio() * g_reference.get_edge_data(nodes[i1], nodes[i2], {'weight': gamma})["weight"]
        for i1, w1 in enumerate(words)] for i2, w2 in enumerate(words)])

    #
    _mean_weight = statistics.mean([e[2].get("weight", 0.0) for e in g_working.edges(data=True)])
    _mean_sign = statistics.mean([e[2].get("significance", 0.0) for e in g_working.edges(data=True)])

    _damping = [.55, .65, .75, .85, .95]
    _max_iter = [300, 600, 900, 1200, 1500]
    if damping is not None and max_iter is not None:
        retries = 1
    for retry in range(min(retries, 5)):
        affprop = AffinityPropagation(affinity='precomputed',
                                      damping=(_damping[retry] if damping is None else damping),
                                      max_iter=(_max_iter[retry] if max_iter is None else max_iter))
        affprop.fit(_similarity)

        if len(affprop.cluster_centers_indices_) == 0 or np.any(affprop.labels_ == -1):
            logging.info(f"No subclusters detected for"
                         f"Graph ({random.sample([n[1]['label'] for n in g_working.nodes(data=True)], min(5, len(g_working.nodes)))})\n"
                         f"with 'damping': {_damping[retry] if damping is None else damping},"
                         f" 'max_iter': {_max_iter[retry] if max_iter is None else max_iter}")
            continue

        for _cluster_id in np.unique(affprop.labels_):
            _exemplar_idx = affprop.cluster_centers_indices_[_cluster_id]
            _cluster_idx = np.nonzero(affprop.labels_ == _cluster_id)
            exemplar_id = int(nodes[_exemplar_idx])
            cluster_ids = nodes[_cluster_idx]

            _node_dict = g_reference.nodes(data=True)[exemplar_id]
            if exemplar_id in g_working.nodes:
                nx.set_node_attributes(g_working, {exemplar_id: dict({"root": True}, **_node_dict)})
            elif len(set(cluster_ids).intersection(g_working.nodes)) > 0:
                g_working.add_node(exemplar_id, root=True, **_node_dict)
                logging.debug(f"Added Exemplar: ({exemplar_id}): {g_working.nodes(data=True)[exemplar_id]} - ({cluster_ids})")
            for c_id in cluster_ids:
                c_id = int(c_id)
                if c_id not in g_working.nodes:
                    continue
                if c_id != exemplar_id:
                    _attr_dict = g_reference.get_edge_data(exemplar_id, c_id)
                    _sign = g.get_edge_data(exemplar_id, c_id)
                    _sign = _sign.get("significance", _mean_sign) if _sign else _mean_sign
                    if not g_working.has_edge(exemplar_id, c_id):
                        if not g_reference.has_edge(exemplar_id, c_id):
                            g_working.add_edge(exemplar_id, c_id, sub_cluster=True, weight=_mean_weight, significance=_mean_sign)
                            logging.debug(
                                f"Added edge: ({exemplar_id})-({c_id}): {g_working.get_edge_data(exemplar_id, c_id)}")
                        else:
                            if 'significance' in _attr_dict:
                                _attr_dict.pop("significance")
                            g_working.add_edge(exemplar_id, c_id, sub_cluster=True, significance=_sign, **_attr_dict)
                            logging.debug(
                                f"Added edge: ({exemplar_id})-({c_id}): {g_working.get_edge_data(exemplar_id, c_id)}")
                    else:
                        nx.set_edge_attributes(g_working, {(exemplar_id, c_id): dict({"sub_cluster": True}, **_attr_dict)})

    if not inplace:
        return g_working
