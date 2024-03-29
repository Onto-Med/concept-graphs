import enum
import os
import sys
import pathlib
import pickle
from collections import OrderedDict, defaultdict
from typing import Union, Optional, List

import flask
import networkx as nx
import requests
import yaml
from flask import request, jsonify, render_template_string
from werkzeug.datastructures import FileStorage

from main_utils import ProcessStatus

sys.path.insert(0, "src")
import graph_creation_util
import cluster_functions


class StepsName:
    DATA = "data"
    EMBEDDING = "embedding"
    CLUSTERING = "clustering"
    GRAPH = "graph"


steps_relation_dict = {
    StepsName.DATA: 1,
    StepsName.EMBEDDING: 2,
    StepsName.CLUSTERING: 3,
    StepsName.GRAPH: 4
}


def get_data_server_config(document_server_config: FileStorage, app: flask.Flask):
    base_config = {"url": "http://localhost", "port": "9008", "index": "documents", "size": "30"}
    try:
        _config = yaml.safe_load(document_server_config.stream)
        for k, v in base_config.items():
            if k not in _config:
                if v is None or (isinstance(v, str) and v.lower() == "none"):
                    _config[k] = None
                    continue
                _config[k] = get_bool_expression(v, v) if isinstance(v, str) else v
        base_config = _config
    except Exception as e:
        app.logger.error(f"Couldn't read config file: {e}\n Using default values '{base_config}'.")
    return base_config


def check_data_server(
        url: str, port: Union[int, str], index: str
):
    final_url = f"{url.rstrip('/')}:{port}/{index.lstrip('/').rstrip('/')}/_count"
    try:
        _response = requests.get(final_url)
    except requests.exceptions.RequestException as e:
        return False
    if _count := _response.json().get('count', False):
        if int(_count) > 0:
            return True
    return False


def check_es_source_for_id(document_hit: dict):
    _source = document_hit.get("_source")
    return _source if _source.get("id", False) else {"id": document_hit.get("_id", "")} | _source


def get_documents_from_es_server(
        url: str, port: Union[str, int], index: str, size: int = 30
):
    final_url = f"{url.rstrip('/')}:{port}/{index.lstrip('/').rstrip('/')}/_search"
    _first_page = requests.get(final_url, params={"size": f"{size}", "scroll": "1m"}).json()
    _scroll_id = _first_page.get("_scroll_id")
    _total_documents = _first_page.get("hits").get("total").get("value")

    for _scroll_index in range(0, _total_documents, size):
        if _scroll_index == 0:
            for document in _first_page.get("hits").get("hits"):
                yield check_es_source_for_id(document)
        else:
            _response = requests.post(
                url=f"{url.rstrip('/')}:{port}/_search/scroll",
                json={"scroll_id": _scroll_id, "scroll": "1m"}).json()
            for document in _response.get("hits").get("hits"):
                yield check_es_source_for_id(document)


def populate_running_processes(app: flask.Flask, path: str, running_processes: dict):
    for process in get_all_processes(path):
        _finished = [_finished_step.get("name") for _finished_step in process.get("finished_steps", [])]
        _name = process.get("name", None)
        if _name is None:
            app.logger.warning(f"Skipping process entry with no name and '{_finished}' steps.")
            continue

        running_processes[_name] = {"status": {}, "name": _name}
        for _step in steps_relation_dict.keys():
            running_processes[_name]["status"][_step] = ProcessStatus.NOT_PRESENT
            if _step in _finished:
                running_processes[_name]["status"][_step] = ProcessStatus.FINISHED


def get_all_processes(path: str):
    _process_detailed = list()
    for _proc in pathlib.Path(path).glob("*"):
        if _proc.is_dir() and not _proc.stem.startswith("."):
            _proc_name = _proc.stem.lower()
            _steps_list = list()
            for _pickle in pathlib.Path(pathlib.Path(path) / _proc_name).glob("*.pickle"):
                _pickle_stem = _pickle.stem.lower()
                _step = _pickle_stem.removeprefix(f"{_proc_name}_")
                if steps_relation_dict.get(_step, False):
                    _steps_list.append({"rank": steps_relation_dict.get(_step),
                                        "name": _step})
            _ord_dict = OrderedDict()
            _ord_dict["name"] = _proc_name
            _ord_dict["finished_steps"] = sorted(_steps_list, key=lambda x: x.get("rank", -1))
            _process_detailed.append(_ord_dict)
    return _process_detailed


def start_processes(app: flask.Flask, processes: tuple, process_name: str, process_tracker: dict):
    _name_marker = {
        StepsName.DATA: "**data**, embedding, clustering, graph",
        StepsName.EMBEDDING: "data, **embedding**, clustering, graph",
        StepsName.CLUSTERING: "data, embedding, **clustering**, graph",
        StepsName.GRAPH: "data, embedding, clustering, **graph**",
    }
    for process_obj, _fact, _name in processes:
        try:
            process_obj.start_process(
                cache_name=process_name,
                process_factory=_fact,
                process_tracker=process_tracker
            )
        except FileNotFoundError as e:
            app.logger.warning(f"Something went wrong with one of the previous steps: {_name_marker[_name]}."
                               f"\nThere is a pickle file missing: {e}")


def read_config(app: flask.Flask, processor, process_type, process_name=None, config=None, language=None):
    app.logger.info(f"Reading config ({process_type}) ...")
    processor.read_config(config=config if config is not None else request.files.get("config", None),
                          process_name=process_name,
                          language=None if process_type not in [StepsName.DATA, StepsName.EMBEDDING] else language)
    # pyyaml doesn't handle 'None' so we need to convert them
    for k, v in processor.config.items():
        if isinstance(v, str) and v.lower() == "none":
            processor.config[k] = None
    app.logger.info(f"Parsed the following arguments for {processor}:\n\t{processor.config}")
    process_name_conf = processor.config.pop("corpus_name", "default")
    if process_name is None:
        process_name = process_name_conf
    processor.set_file_storage_path(process_name)
    processor.process_name = process_name

    with pathlib.Path(
            pathlib.Path(processor._file_storage) /
            pathlib.Path(f"{process_name}_{process_type}_config.yaml")
    ).open('w') as config_save:
        yaml.safe_dump(processor.config, config_save)
    return process_name


def read_exclusion_ids(exclusion: Union[str, FileStorage]):
    if isinstance(exclusion, str):
        if exclusion.startswith("[") and exclusion.endswith("]"):
            try:
                return [int(i.strip()) for i in exclusion[1:-1].split(",")]
            except Exception:
                pass
    elif isinstance(exclusion, FileStorage):
        read_exclusion_ids(f"[{exclusion.stream.read().decode()}]")
    return []


def data_get_statistics(data_obj):
    return jsonify(
        number_of_documents=data_obj.documents_n,
        number_of_data_chunks=data_obj.chunk_sets_n,
        number_of_label_types=len(data_obj.true_labels)
    )


def embedding_get_statistics(emb_obj):
    return jsonify(
        number_of_embeddings=emb_obj.sentence_embeddings.shape[0],
        embedding_dim=emb_obj.embedding_dim
    )


def clustering_get_concepts(cluster_gen):
    _cluster_dict = defaultdict(list)
    for c_id, _, text in cluster_gen:
        _cluster_dict[f"concept-{c_id}"].append(text)
    return jsonify(**_cluster_dict)


def graph_get_statistics(app: flask.Flask, data: Union[str, list], path: str) -> dict:
    if isinstance(data, str):
        _path = pathlib.Path(
            os.getcwd() / pathlib.Path(path) / pathlib.Path(data) / f"{data}_graph.pickle")
        app.logger.info(f"Trying to open file '{_path}'")
        try:
            graph_list = pickle.load(_path.open('rb'))
        except FileNotFoundError as e:
            app.logger.info(e)
            return {"error": f"Couldn't find graph pickle '{data}_graph.pickle'. Probably steps before failed; check the logs."}
    elif isinstance(data, list):
        graph_list = data
    else:
        graph_list = []

    # return_dict = defaultdict(dict)
    return_dict = dict()
    cg_stats = list()
    for i, cg in enumerate(graph_list):
        cg_stats.append({
            "id": i,
            "edges": len(cg.edges),
            "nodes": len(cg.nodes)
        })
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
            _neighbors.append({
                "id": neighbor,
                "weight": _data.get("weight", None),
                "significance": _data.get("significance", None)
            })
        _adj.append({
            "id": node,
            "neighbors": _neighbors
        })
    return _adj


def graph_get_specific(process, graph_nr, path: str, draw=False):
    store_path = pathlib.Path(pathlib.Path(path) / f"{process}")
    try:
        graph_list = pickle.load(
            pathlib.Path(store_path / f"{process}_graph.pickle").open('rb')
        )
        if (len(graph_list)) > graph_nr >= 0:
            if not draw:
                return jsonify({
                    "adjacency": build_adjacency_obj(graph_list[graph_nr]),
                    "nodes": [dict(id=n, **v) for n, v in graph_list[graph_nr].nodes(data=True)]
                })
            else:
                templates_path = pathlib.Path(store_path)
                templates_path.mkdir(exist_ok=True)
                graph_path = graph_creation_util.visualize_graph(
                    graph=graph_list[graph_nr], store=str(pathlib.Path(templates_path / "graph.html").resolve()),
                    height="800px"
                )
                return render_template_string(pathlib.Path(graph_path).resolve().read_text())
        else:
            return jsonify(f"{graph_nr} is not in range [0, {len(graph_list) - 1}]; no such graph present.")
    except FileNotFoundError:
        return jsonify(f"There is no graph data present for '{process}'.")


def graph_create(app: flask.Flask, path: str):
    app.logger.info("=== Graph creation started ===")
    exclusion_ids_query = read_exclusion_ids(request.args.get("exclusion_ids", "[]"))
    # ToDo: files read doesn't work...
    # exclusion_ids_files = read_exclusion_ids(request.files.get("exclusion_ids", "[]"))
    if request.method in ["POST", "GET"]:
        graph_create = graph_creation_util.GraphCreationUtil(app, path)

        process_name = read_config(graph_create, StepsName.GRAPH)

        app.logger.info(f"Start Graph Creation '{process_name}' ...")
        try:
            concept_graphs = graph_create.start_process(process_name,
                                                        cluster_functions.WordEmbeddingClustering,
                                                        exclusion_ids_query)
            return graph_get_statistics(concept_graphs)  # ToDo: if concept_graphs -> need to adapt method
        except FileNotFoundError:
            return jsonify(f"There is no processed data for the '{process_name}' process to be embedded.")
    return jsonify("Nothing to do.")


def get_bool_expression(str_bool: str, default: Union[bool, str] = False) -> bool:
    if isinstance(str_bool, bool):
        return str_bool
    elif isinstance(str_bool, str):
        return {
            'true': True, 'yes': True, 'y': True, 'ja': True, 'j': True,
            'false': False, 'no': False, 'n': False, 'nein': False,
        }.get(str_bool.lower(), default)
    else:
        return False


def get_query_param_help_text(param: str):
    return {
        "process": "process: name of the process (e.g. ``corpus_name`` in config); if not provided, uses 'default'",
        "exclusion_ids": "exclusion_ids: list of concept ids (as returned by e.g. ``/clustering/concepts``), "
                         "that shall be excluded from the final graphs in the form of ``[ID1, ID2, etc.]``",
        "draw": "draw: `true` or `false` - whether the response shall be a rendered graph or plain json"
    }.get(param, "No help text available.")


def get_omit_pipeline_steps(steps: object) -> list[str]:
    step_set = {"data", "embedding", "clustering", "graph"}
    if isinstance(steps, str):
        steps = steps.strip('([{}])')
        return [s.lower() for s in steps.split(',') if s.lower() in step_set]
    return []


if __name__ == "__main__":
    count = 0
    for i in get_documents_from_es_server(url="http://localhost", port=9008, index="documents"):
        count += 1
        print(i.get('id'))
    print(count)
