import pathlib
import pickle
import sys
from pathlib import Path
from typing import Optional, Union
from inspect import getfullargspec

import flask
import yaml
import networkx as nx
from munch import Munch, unmunchify
from pyvis import network as net

from flask import jsonify
from werkzeug.datastructures import FileStorage

from main_utils import ProcessStatus, StepsName, add_status_to_running_process

sys.path.insert(0, "src")
import util_functions


class GraphCreationUtil:

    def __init__(self, app: flask.app.Flask, file_storage: str, step_name: StepsName = StepsName.GRAPH):
        self._app = app
        self._file_storage = Path(file_storage)
        self._process_step = step_name
        self._process_name = None
        self.config = None

    @property
    def process_name(self):
        return self._process_name

    @process_name.setter
    def process_name(self, name):
        self._process_name = name

    @property
    def process_step(self):
        return self._process_step

    def read_config(self, config: Optional[Union[FileStorage, dict]], process_name=None, language=None):
        base_config = {
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
        if isinstance(config, dict):
            if isinstance(config, Munch):
                _config = unmunchify(config)
            else:
                _config = config
            for _type in ["graph", "cluster"]:
                _sub_config = _config.get(_type, {}).copy()
                for k, v in _sub_config.items():
                    _config[f"{_type}_{k}"] = v
                _config.pop(_type, None)
            base_config = _config
        elif isinstance(config, FileStorage):
            try:
                base_config = yaml.safe_load(config.stream)
                if base_config.get('model', False):
                    raise KeyError(f"No model name provided in config: {base_config}")
            except Exception as e:
                self._app.logger.error(f"Couldn't read config file: {e}")
                return jsonify("Encountered error. See log.")
        else:
            self._app.logger.info("No config file provided; using default values")

        base_config["corpus_name"] = process_name.lower() if process_name is not None else base_config["corpus_name"].lower()
        self.config = base_config

    def set_file_storage_path(self, sub_path):
        self._file_storage = Path(self._file_storage / sub_path)
        self._file_storage.mkdir(exist_ok=True)  # ToDo: warning when folder exists

    def has_pickle(self, process):
        _pickle = Path(self._file_storage / process / f"{process}_{self.process_step}.pickle")
        return _pickle.exists()

    def delete_pickle(self, process):
        if self.has_pickle(process):
            _pickle = Path(self._file_storage / process / f"{process}_{self.process_step}.pickle")
            _pickle.unlink()

    def start_process(self, cache_name, process_factory, process_tracker, exclusion_ids=None):
        sent_emb = util_functions.load_pickle(Path(self._file_storage / f"{cache_name}_embedding.pickle"))
        cluster_obj = util_functions.load_pickle(Path(self._file_storage / f"{cache_name}_clustering.pickle"))

        config = self.config.copy()

        add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.RUNNING, process_tracker)
        concept_graphs = []
        try:
            concept_graph_clustering = process_factory(
                sentence_embedding_obj=sent_emb,
                cluster_obj=cluster_obj.concept_cluster,
                cluster_exclusion_ids=exclusion_ids
            ).create_concept_graph_clustering()

            # ToDo: should this go everywhere?
            _valid_config = getfullargspec(concept_graph_clustering.build_document_concept_matrix).args
            for _arg in config.copy().keys():
                if _arg not in _valid_config:
                    config.pop(_arg)
            concept_graphs = concept_graph_clustering.build_document_concept_matrix(
                break_after_graph_creation=True,
                **config
            )
            with pathlib.Path(self._file_storage / f"{cache_name}_{self.process_step}.pickle").open("wb") as graphs_out:
                pickle.dump(concept_graphs, graphs_out)
            add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.FINISHED, process_tracker)
        except Exception as e:
            add_status_to_running_process(self.process_name, self.process_step, ProcessStatus.ABORTED, process_tracker)
            self._app.logger.error(e)

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
