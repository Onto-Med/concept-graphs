import sys
from pathlib import Path
from typing import Optional, Union, Callable

import flask
import networkx as nx
from pyvis import network as net

from werkzeug.datastructures import FileStorage

sys.path.insert(0, "src")
from main_utils import StepsName, BaseUtil
from src.data_functions import DataProcessingFactory
from src.embedding_functions import SentenceEmbeddingsFactory
from src.cluster_functions import WordEmbeddingClustering
from src.util_functions import load_pickle


class GraphCreationUtil(BaseUtil):

    def __init__(
            self,
            app: flask.app.Flask,
            file_storage: str
    ):
        super().__init__(app, file_storage, StepsName.GRAPH)

    @property
    def serializable_config(self) -> dict:
        return self.config.copy()

    @property
    def default_config(self) -> dict:
        return {
            "cluster_distance": 0.7,
            "cluster_min_size": 4,
            "graph_cosine_weight": .6,
            "graph_merge_threshold": .95,
            "graph_weight_cut_off": .5,
            "graph_unroll": False,
            "graph_simplify": .5,
            "graph_simplify_alg": "significance",
            "graph_sub_clustering": False,
            "restrict_to_cluster": True,
        }

    @property
    def sub_config_names(self) -> list[str]:
        return ["graph", "cluster"]

    @property
    def necessary_config_keys(self) -> list[str]:
        return []

    @property
    def protected_kwargs(self) -> list[str]:
        return ["exclusion_ids"]

    def read_config(
            self,
            config: Optional[Union[FileStorage, dict]],
            process_name=None,
            language=None
    ):
        return super().read_config(config, process_name, language)

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
        return super().delete_process(process, extensions)

    def _process_method(self) -> Callable:
        return WordEmbeddingClustering._ConceptGraphClustering.build_concept_graphs

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
        cluster_obj = load_pickle(Path(self._file_storage / f"{cache_name}_clustering.pickle"))
        return sent_emb, cluster_obj

    def _start_process(
            self,
            process_factory,
            *args,
            **kwargs
    ):
        concept_graphs = []
        sent_emb, cluster_obj = args
        try:
            concept_graph_clustering = process_factory(
                sentence_embedding_obj=sent_emb,
                cluster_obj=cluster_obj,
                cluster_exclusion_ids=kwargs.pop("exclusion_ids", []),
            ).create_concept_graph_clustering()
            concept_graphs = concept_graph_clustering.build_concept_graphs(**kwargs)
        except Exception as e:
            raise e
        return concept_graphs


def visualize_graph(graph: nx.Graph, height="800px", directed=False, store="index.html"):
    g = net.Network(height=height, select_menu=False, filter_menu=False, notebook=True, width='100%',
                    directed=directed, cdn_resources="remote")
    g.barnes_hut(gravity=-5000)

    # if directed:
    #     g.from_nx(transform2directed(graph))
    #     return g
    for _node, _node_attrs in graph.nodes(data=True):
        _node_attrs.update({"size": 25, "title": str(_node), "font": {"size": 50}})
        if _node_attrs.get("parent", False):
            _node_attrs.update({"color": "red"})
        if _node_attrs.get("root", False):
            _node_attrs.update({"size": 50})
        g.add_node(_node, **_node_attrs)
    for _source_edge, _target_edge, _edge_attrs in graph.edges(data=True):
        if not directed:
            _edge_weight = _edge_attrs.get("weight", 0.0)
            _edge_sepcial = _edge_attrs.get("sub_cluster", False)
            if _edge_weight >= 0.9:
                _edge_attrs.update({"color": "red", "width": 5})
            elif 0.9 > _edge_weight >= 0.8:
                _edge_attrs.update({"color": "green", "width": 3})
            elif 0.8 > _edge_weight >= 0.65:
                _edge_attrs.update({"color": "blue", "width": 2})
            elif 0.65 > _edge_weight >= 0.5:
                _edge_attrs.update({"color": "blue", "dashes": True, "physics": True})
            else:
                _edge_attrs.update({"dashes": True, "physics": False, "color": "yellow"})
            if _edge_sepcial:
                #_edge_attrs.update({"color": "black", "dashes": True, "physics": True})
                continue # skip edge visualization
            _edge_attrs.update({"title": str(round(_edge_weight, 2))})
        g.add_edge(_source_edge, _target_edge, **_edge_attrs)
    g.write_html(name=store, notebook=True)
    return store
