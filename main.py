import logging
import os
import pathlib
import pickle
import sys

from collections import defaultdict
from typing import Union

import networkx as nx
import yaml
from flask import Flask, jsonify, request, render_template, render_template_string, Response
from flask.logging import default_handler
from werkzeug.datastructures import FileStorage

import graph_creation_util
from preprocessing_util import PreprocessingUtil
from embedding_util import PhraseEmbeddingUtil
from clustering_util import ClusteringUtil
from graph_creation_util import GraphCreationUtil

sys.path.insert(0, "src")
import data_functions
import embedding_functions
import cluster_functions
import util_functions
from util_functions import HTTPResponses

app = Flask(__name__)

root = logging.getLogger()
root.addHandler(default_handler)

FILE_STORAGE_TMP = "./tmp"  # ToDo: replace it with proper path in docker

steps_relation_dict = {
    "data-processed": 1,
    "embeddings": 2,
    "clustering": 3,
    "graphs": 4
}

# ToDo: file with stopwords will be POSTed: #filter_stop: Optional[list] = None,

# ToDo: I downscale the embeddings twice... (that snuck in somehow); once in SentenceEmbeddings via create(down_scale_algorithm)
# ToDo: and once PhraseCluster via create(down_scale_algorithm). I can't remember why I added this to SentenceEmbeddings later on...
# ToDo: but I should make sure, that there is a check somewhere that the down scaling is not applied twice!

# ToDo: make sure that no arguments can be supplied via config that won't work

# ToDo: endpoints with path arguments should throw a response/warning if there is no saved pickle

# ToDo: implement socket.io (or similar) so that the requests can update on the progress of each process

# ToDo: get info on each base endpoint, when providing no further args or params (if necessary)


@app.route("/")
def index():
    return jsonify(available_endpoints=['/preprocessing', '/embedding', '/clustering', '/graph', '/pipeline', '/processes'])


@app.route("/preprocessing", methods=['GET', 'POST'])
def data_preprocessing():
    app.logger.info("=== Preprocessing started ===")
    if request.method == "POST" and len(request.files) > 0 and "data" in request.files:
        pre_proc = PreprocessingUtil(app, FILE_STORAGE_TMP)

        process_name = read_config(pre_proc, "data")

        app.logger.info("Reading labels ...")
        pre_proc.read_labels(request.files.get("labels", None))
        app.logger.info(
            f"Gathered the following labels:\n\t{list(pre_proc.labels.values()) if pre_proc.labels is not None else []}")

        app.logger.info("Reading data ...")
        pre_proc.read_data(request.files.get("data", None))
        app.logger.info(f"Counted {len(pre_proc.data)} item ins zip file.")

        app.logger.info(f"Start preprocessing '{process_name}' ...")
        return data_get_statistics(pre_proc.start_process(process_name, data_functions.DataProcessingFactory))

    elif request.method == "GET":
        pass

    elif len(request.files) <= 0 or "data" not in request.files:
        app.logger.error("There were no files at all or no data files POSTed."
                         " At least a zip folder with the data is necessary!\n"
                         " It is also necessary to conform to the naming convention!\n"
                         "\t\ti.e.: curl -X POST -F data=@\"#SOME/PATH/TO/FILE.zip\"")
    return jsonify("Nothing to do.")


@app.route("/preprocessing/<path_arg>", methods=['GET'])
def data_preprocessing_with_arg(path_arg):
    process = request.args.get("process", "default")
    path_arg = path_arg.lower()

    _path_args = ["statistics", "noun_chunks"]
    if path_arg in _path_args:
        data_obj = data_functions.DataProcessingFactory.load(
            pathlib.Path(pathlib.Path(FILE_STORAGE_TMP) / pathlib.Path(process) / f"{process}_data-processed.pickle"))
    else:
        return jsonify(error=f"No such path argument '{path_arg}' for 'preprocessing' endpoint.",
                       possible_path_args=[f"/{p}" for p in _path_args])

    if path_arg == "statistics":
        return data_get_statistics(data_obj)
    elif path_arg == "noun_chunks":
        return jsonify(
            noun_chunks=data_obj.data_chunk_sets
        )


@app.route("/embedding", methods=['POST', 'GET'])
def phrase_embedding():
    app.logger.info("=== Phrase embedding started ===")
    if request.method in ["POST", "GET"]:
        phra_emb = PhraseEmbeddingUtil(app, FILE_STORAGE_TMP)

        process_name = read_config(phra_emb, "embedding")

        app.logger.info(f"Start phrase embedding '{process_name}' ...")
        try:
            return embedding_get_statistics(
                phra_emb.start_process(process_name, embedding_functions.SentenceEmbeddingsFactory))
        except FileNotFoundError:
            return jsonify(f"There is no processed data for the '{process_name}' process to be embedded.")
    return jsonify("Nothing to do.")


@app.route("/embedding/<path_arg>", methods=['GET'])
def phrase_embedding_with_arg(path_arg):
    process = request.args.get("process", "default")
    path_arg = path_arg.lower()

    _path_args = ["statistics"]
    if path_arg in _path_args:
        emb_obj = embedding_functions.SentenceEmbeddingsFactory.load(
            pathlib.Path(pathlib.Path(FILE_STORAGE_TMP) / pathlib.Path(process) / f"{process}_data-processed.pickle"),
            pathlib.Path(pathlib.Path(FILE_STORAGE_TMP) / pathlib.Path(process) / f"{process}_embeddings.pickle"),
        )
    else:
        return jsonify(error=f"No such path argument '{path_arg}' for 'embedding' endpoint.",
                       possible_path_args=[f"/{p}" for p in _path_args])

    if path_arg == "statistics":
        return embedding_get_statistics(emb_obj)


@app.route("/clustering", methods=['POST', 'GET'])
def phrase_clustering():
    app.logger.info("=== Phrase clustering started ===")
    if request.method in ["POST", "GET"]:
        saved_config = request.args.get("config", False)
        if not saved_config:
            phra_clus = ClusteringUtil(app, FILE_STORAGE_TMP)

            process_name = read_config(phra_clus, "clustering")

            app.logger.info(f"Start phrase clustering '{process_name}' ...")
            try:
                _cluster_gen = phra_clus.start_process(process_name, cluster_functions.PhraseClusterFactory)
                return clustering_get_concepts(_cluster_gen)
            except FileNotFoundError:
                return jsonify(f"There is no embedded data for the '{process_name}' process to be clustered.")
        else:
            return jsonify(saved_config)
    return jsonify("Nothing to do.")


@app.route("/clustering/<path_arg>", methods=['GET'])
def clustering_with_arg(path_arg):
    process = request.args.get("process", "default")
    top_k = int(request.args.get("top_k", 15))
    distance = float(request.args.get("distance", .6))
    path_arg = path_arg.lower()

    _path_args = ["concepts"]
    if path_arg in _path_args:
        cluster_obj = cluster_functions.PhraseClusterFactory.load(
            pathlib.Path(pathlib.Path(FILE_STORAGE_TMP) / pathlib.Path(process) / f"{process}_clustering.pickle"),
        )
    else:
        return jsonify(error=f"No such path argument '{path_arg}' for 'clustering' endpoint.",
                       possible_path_args=[f"/{p}" for p in _path_args])

    if path_arg == "concepts":
        emb_obj = util_functions.load_pickle(
            pathlib.Path(pathlib.Path(FILE_STORAGE_TMP) / f"{process}_embeddings.pickle"))
        _cluster_gen = embedding_functions.show_top_k_for_concepts(
            cluster_obj=cluster_obj.concept_cluster, embedding_object=emb_obj, yield_concepts=True,
            top_k=top_k, distance=distance
        )
        return clustering_get_concepts(_cluster_gen)


@app.route("/graph/<path_arg>", methods=['POST', 'GET'])
def graph_creation_with_arg(path_arg):
    process = request.args.get("process", "default")
    draw = True if request.args.get("draw", "false").lower() == "true" else False
    path_arg = path_arg.lower()

    _path_args = ["statistics", "creation"]
    if path_arg in _path_args:
        try:
            if path_arg == "statistics":
                return graph_get_statistics(process)
            elif path_arg == "creation":
                return graph_create()
        except FileNotFoundError:
            return Response(f"There is no graph data present for '{process}'.\n",
                            status=HTTPResponses.NOT_FOUND)
    elif path_arg.isdigit():
        graph_nr = int(path_arg)
        return graph_get_specific(process, graph_nr, draw=draw)
    else:
        return Response(
            f"No such path argument '{path_arg}' for 'graph' endpoint.\n"
            f"Possible path arguments are: {', '.join([p for p in _path_args] + ['#ANY_INTEGER'])}\n",
            status=HTTPResponses.BAD_REQUEST)


@app.route("/pipeline", methods=['POST'])
def complete_pipeline():
    process = request.args.get("process", "default")
    app.logger.info(f"Using process name '{process}'")
    language = {"en": "en", "de": "de"}.get(request.args.get("lang", "en"), "en")
    app.logger.info(f"Using preset language settings for '{language}'")
    skip_present = request.args.get("skip_present", True)
    if isinstance(skip_present, str):
        skip_present = {"true": True, "false": False}.get(skip_present.lower(), True)
    if skip_present:
        app.logger.info("Skipping present saved steps")

    data = request.files.get("data", False)
    if not data:
        return jsonify("No data provided with 'data' key")
    labels = request.files.get("labels", None)

    processes = [
        ("data", PreprocessingUtil, request.files.get("data_config", None), data_functions.DataProcessingFactory,),
        ("embedding", PhraseEmbeddingUtil, request.files.get("embedding_config", None),
         embedding_functions.SentenceEmbeddingsFactory,),
        ("clustering", ClusteringUtil, request.files.get("clustering_config", None),
         cluster_functions.PhraseClusterFactory,),
        (
        "graph", GraphCreationUtil, request.files.get("graph_config", None), cluster_functions.WordEmbeddingClustering,)
    ]

    for _name, _proc, _conf, _fact in processes:
        process_obj = _proc(app=app, file_storage=FILE_STORAGE_TMP)
        # process_obj.set_file_storage_path(process)
        if process_obj.has_pickle(process) and skip_present:
            continue
        read_config(processor=process_obj, process_type=_name, process_name=process, config=_conf, language=language)
        if _name == "data":
            process_obj.read_labels(labels)
            process_obj.read_data(data)
        process_obj.start_process(cache_name=process, process_factory=_fact)

    return graph_get_statistics(process)


@app.route("/processes", methods=['GET'])
def get_all_processes():
    _process_detailed = list()
    for _proc in pathlib.Path(FILE_STORAGE_TMP).glob("*"):
        if _proc.is_dir():
            _proc_name = _proc.stem.lower()
            _steps_list = list()
            for _pickle in pathlib.Path(pathlib.Path(FILE_STORAGE_TMP) / _proc_name).glob("*.pickle"):
                _pickle_stem = _pickle.stem.lower()
                _step = _pickle_stem.removeprefix(f"{_proc_name}_")
                _steps_list.append({"rank": steps_relation_dict.get(_step),
                                    "name": _step})
            _process_detailed.append({"name": _proc_name, "finished_steps": sorted(_steps_list, key=lambda x: x.get("rank", -1))})
    if len(_process_detailed) > 0:
        return jsonify(processes=_process_detailed)
    else:
        return Response("No saved processes.", HTTPResponses.NOT_FOUND)


def read_config(processor, process_type, process_name=None, config=None, language=None):
    app.logger.info("Reading config ...")
    processor.read_config(config=config if config is not None else request.files.get("config", None),
                          process_name=process_name,
                          language=None if process_type not in ["data", "embedding"] else language)
    # pyyaml doesn't handle 'None' so we need to convert them
    for k, v in processor.config.items():
        if isinstance(v, str) and v.lower() == "none":
            processor.config[k] = None
    app.logger.info(f"Parsed the following arguments for {processor}:\n\t{processor.config}")
    process_name_conf = processor.config.pop("corpus_name", "default")
    if process_name is None:
        process_name = process_name_conf
    processor.set_file_storage_path(process_name)

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


def graph_get_statistics(data: Union[str, list]):
    if isinstance(data, str):
        _path = pathlib.Path(
            os.getcwd() / pathlib.Path(FILE_STORAGE_TMP) / pathlib.Path(data) / f"{data}_graphs.pickle")
        app.logger.info(f"Trying to open file '{_path}'")
        graph_list = pickle.load(_path.open('rb'))
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
    return jsonify(return_dict)


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


def graph_get_specific(process, graph_nr, draw=False):
    store_path = pathlib.Path(pathlib.Path(FILE_STORAGE_TMP) / f"{process}")
    try:
        graph_list = pickle.load(
            pathlib.Path(store_path / f"{process}_graphs.pickle").open('rb')
        )
        if (len(graph_list) - 1) > graph_nr >= 0:
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
            return jsonify(f"{graph_nr} is not in range [0, {len(graph_list)}]; no such graph present.")
    except FileNotFoundError:
        return jsonify(f"There is no graph data present for '{process}'.")


def graph_create():
    app.logger.info("=== Graph creation started ===")
    exclusion_ids_query = read_exclusion_ids(request.args.get("exclusion_ids", "[]"))
    # ToDo: files read doesn't work...
    # exclusion_ids_files = read_exclusion_ids(request.files.get("exclusion_ids", "[]"))
    if request.method in ["POST", "GET"]:
        graph_create = GraphCreationUtil(app, FILE_STORAGE_TMP)

        process_name = read_config(graph_create, "graph")

        app.logger.info(f"Start Graph Creation '{process_name}' ...")
        try:
            concept_graphs = graph_create.start_process(process_name,
                                                        cluster_functions.WordEmbeddingClustering,
                                                        exclusion_ids_query)
            return graph_get_statistics(concept_graphs)  # ToDo: if concept_graphs -> need to adapt method
        except FileNotFoundError:
            return jsonify(f"There is no processed data for the '{process_name}' process to be embedded.")
    return jsonify("Nothing to do.")


# ToDo: set debug=False
if __name__ == "__main__":
    f_storage = pathlib.Path(FILE_STORAGE_TMP)
    if not f_storage.exists():
        f_storage.mkdir()
    app.run(debug=True, host='0.0.0.0', port=9007)
