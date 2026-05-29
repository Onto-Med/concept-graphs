"""Artifact response service helpers."""

import os
import pathlib
import pickle
from collections import defaultdict
from typing import Union

import flask
import networkx as nx
from flask import jsonify, render_template_string, request

from src.api.services.configuration import read_config, read_exclusion_ids
from src.core import cluster_functions
from src.pipeline.status import StepsName
from src.pipeline.steps.graph_creation_util import GraphCreationUtil, visualize_graph


def data_get_statistics(data_obj):
    return jsonify(
        number_of_documents=data_obj.documents_n,
        number_of_data_chunks=data_obj.chunk_sets_n,
        number_of_label_types=len(data_obj.true_labels),
    )


def embedding_get_statistics(emb_obj):
    return jsonify(
        number_of_embeddings=emb_obj.sentence_embeddings.shape[0],
        embedding_dim=emb_obj.embedding_dim,
    )


def clustering_get_concepts(cluster_gen):
    _cluster_dict = defaultdict(list)
    for c_id, _, text in cluster_gen:
        _cluster_dict[f"concept-{c_id}"].append(text)
    return jsonify(**_cluster_dict)


def graph_get_statistics(
    app: flask.Flask, data: Union[str, list], path: Union[str, pathlib.Path]
) -> dict:
    if isinstance(data, str):
        _path = pathlib.Path(
            os.getcwd()
            / pathlib.Path(path)
            / pathlib.Path(data)
            / f"{data}_graph.pickle"
        )
        app.logger.info(f"Trying to open file '{_path}'")
        try:
            graph_list = pickle.load(_path.open("rb"))
        except FileNotFoundError as e:
            app.logger.info(e)
            return {
                "error": f"Couldn't find graph pickle '{data}_graph.pickle'. Probably steps before failed; check the logs."
            }
    elif isinstance(data, list):
        graph_list = data
    else:
        graph_list = []

    # return_dict = defaultdict(dict)
    return_dict = dict()
    cg_stats = list()
    for i, cg in enumerate(graph_list):
        cg_stats.append({"id": i, "edges": len(cg.edges), "nodes": len(cg.nodes)})
        # return_dict[f"concept_graph_{i}"]["edges"] = len(cg.edges)
        # return_dict[f"concept_graph_{i}"]["nodes"] = len(cg.nodes)
    return_dict["conceptGraphs"] = cg_stats
    return_dict["numberOfGraphs"] = len(cg_stats)
    # response = ["To get a specific graph (its nodes (with labels) and edges (with weight) as an adjacency list)"
    #             "use the endpoint '/graph/GRAPH-ID', where GRAPH-ID can be gleaned by 'concept_graph_GRAPH-ID",
    #             return_dict]
    return return_dict


def build_adjacency_obj(graph_obj: nx.Graph):
    _adj = []
    for node in graph_obj.nodes:
        _neighbors = []
        for _, neighbor, _data in graph_obj.edges(node, data=True):
            _neighbors.append(
                {
                    "id": neighbor,
                    "weight": _data.get("weight", None),
                    "significance": _data.get("significance", None),
                }
            )
        _adj.append({"id": node, "neighbors": _neighbors})
    return _adj


def graph_get_specific(
    process: Union[str, list], graph_nr, path: Union[str, pathlib.Path], draw=False
):
    try:
        if isinstance(process, str):
            store_path = pathlib.Path(pathlib.Path(path) / f"{process}")
            graph_list = pickle.load(
                pathlib.Path(store_path / f"{process}_graph.pickle").open("rb")
            )
        else:
            graph_list = process
        if (len(graph_list)) > graph_nr >= 0:
            if not draw:
                return jsonify(
                    {
                        "adjacency": build_adjacency_obj(graph_list[graph_nr]),
                        "nodes": [
                            dict(id=n, **v)
                            for n, v in graph_list[graph_nr].nodes(data=True)
                        ],
                    }
                )
            else:
                templates_path = pathlib.Path(path)
                templates_path.mkdir(exist_ok=True)
                graph_path = visualize_graph(
                    graph=graph_list[graph_nr],
                    store=str(pathlib.Path(templates_path / "graph.html").resolve()),
                    height="800px",
                )
                return render_template_string(
                    pathlib.Path(graph_path).resolve().read_text()
                )
        else:
            return jsonify(
                f"{graph_nr} is not in range [0, {len(graph_list) - 1}]; no such graph present."
            )
    except FileNotFoundError:
        return jsonify(f"There is no graph data present for '{process}'.")


def graph_create(app: flask.Flask, path: Union[str, pathlib.Path]):
    app.logger.info("=== Graph creation started ===")
    exclusion_ids_query = read_exclusion_ids(request.args.get("exclusion_ids", "[]"))
    # ToDo: files read doesn't work...
    # exclusion_ids_files = read_exclusion_ids(request.files.get("exclusion_ids", "[]"))
    if request.method in ["POST", "GET"]:
        graph_create = GraphCreationUtil(app, path)

        process_name = read_config(graph_create, StepsName.GRAPH)

        app.logger.info(f"Start Graph Creation '{process_name}' ...")
        try:
            _, concept_graphs = graph_create.start_process(
                process_name,
                cluster_functions.WordEmbeddingClustering,
                process_tracker={},
                exclusion_ids=exclusion_ids_query,
            )
            return graph_get_statistics(
                app, concept_graphs, path
            )  # ToDo: if concept_graphs -> need to adapt method
        except FileNotFoundError:
            return jsonify(
                f"There is no processed data for the '{process_name}' process to be embedded."
            )
    return jsonify("Nothing to do.")
