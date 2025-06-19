import logging
import os
import sys
import pathlib
import pickle
from collections import OrderedDict, defaultdict, namedtuple
from time import sleep
from typing import Union, Iterable, Optional

import numpy as np
import flask
import networkx as nx
import requests
import yaml
from flask import request, jsonify, render_template_string, Response
from munch import Munch
from werkzeug.datastructures import FileStorage
from yaml.representer import RepresenterError
from pydoc import locate

from clustering_util import ClusteringUtil
from embedding_util import PhraseEmbeddingUtil
from preprocessing_util import PreprocessingUtil
from graph_creation_util import GraphCreationUtil, visualize_graph
from main_utils import ProcessStatus, HTTPResponses, StepsName, pipeline_query_params, steps_relation_dict, \
    add_status_to_running_process, PipelineLanguage, get_bool_expression, StoppableThread, string_conformity, BaseUtil

sys.path.insert(0, "src")
from src import data_functions, cluster_functions, embedding_functions
from src.util_functions import DocumentStore, EmbeddingStore

pipeline_json_config = namedtuple("pipeline_json_config",
                                  ["name", "language", "document_server", "vectorstore_server", "data", "embedding", "clustering", "graph"])

document_adding_json = namedtuple("document_adding_json",
                                  ["language", "documents", "document_server", "vectorstore_server"])


def parse_config_json(response_json) -> pipeline_json_config:
    config = Munch.fromDict(response_json)
    try:
        return pipeline_json_config(string_conformity(config.get("name", None)),
                                    config.get("language", None),
                                    config.get("document_server", None),
                                    config.get("vectorstore_server", None),
                                    config.config.get("data", Munch()),
                                    config.config.get("embedding", Munch()),
                                    config.config.get("clustering", Munch()),
                                    config.config.get("graph", Munch())
                                    )
    except AttributeError as e:
        logging.error(f"Json body/configuration seems to be malformed: no 'config' entry was provided.\n{e}")
        return pipeline_json_config(string_conformity(config.get("name", None)),
                                    config.get("language", None),
                                    config.get("document_server", None),
                                    config.get("vectorstore_server", None),
                                    Munch(),
                                    Munch(),
                                    Munch(),
                                    Munch()
                                    )


def parse_document_adding_json(response_json) -> Optional[document_adding_json]:
    """
        request.json = {
            vectorstore_server: Optional[dict]
            document_server: Optional[dict]
            language: str
            documents: [
                {
                    id: str
                    content: str
                    corpus: str
                    label: str
                    name: str
                }
            ]
        }
    """
    try:
        config = Munch.fromDict(response_json)
        # _id_key = list(set(config.keys()).intersection(["id", "_id"]))
        # _id = _id_key[0] if _id_key else "none"
        return document_adding_json(config.get("language", None),
                                    config.get("documents", []),
                                    config.get("document_server", None),
                                    config.get("vectorstore_server", None))
    except Exception as e:
        logging.error(f"Content json parsing error: '{e}'")
        return None



def read_config_json(app, processor, process_type, process_name, config, language):
    process_name = string_conformity(process_name)
    app.logger.info(f"Reading config ({process_type}) ...")
    _language = config.get("language", language)
    processor.read_config(config=config, process_name=config.get("name", process_name),
                          language=_language if process_type in [StepsName.DATA, StepsName.EMBEDDING] else None)
    app.logger.info(f"Parsed the following arguments for {processor}:\n\t{processor.config}")
    processor.file_storage_path = process_name
    processor.process_name = process_name

    with pathlib.Path(
            pathlib.Path(processor.file_storage_path) /
            pathlib.Path(f"{process_name}_{process_type}_config.yaml")
    ).open('w') as config_save:
        try:
            if _language is not None:
                processor.config["language"] = _language
            yaml.safe_dump(processor.config, config_save)
        except RepresenterError:
            yaml.safe_dump(processor.serializable_config, config_save)
    return process_name


def get_pipeline_query_params(
        app: flask.Flask,
        flask_request: flask.Request,
        running_processes: dict,
        config_obj_json: pipeline_json_config
) -> Union[pipeline_query_params, tuple]:
    if config_obj_json is not None and config_obj_json.name is not None:
        corpus = string_conformity(config_obj_json.name)
    else:
        corpus = string_conformity(flask_request.args.get("process", "default"))
    if corpus_status := running_processes.get(corpus, False):
        if any([v.get("status", None) == ProcessStatus.RUNNING for v in corpus_status.get("status", [])]):
            return jsonify(
                name=corpus,
                error=f"A process is currently running for this corpus. Use '/status?process={corpus}' for specifics."
            ), int(HTTPResponses.FORBIDDEN)
    app.logger.info(f"Using process name '{corpus}'")
    if config_obj_json is not None and config_obj_json.language is not None:
        language = PipelineLanguage.language_from_string(config_obj_json.language)
    else:
        language = PipelineLanguage.language_from_string(str(flask_request.args.get("lang", "en")))
    app.logger.info(f"Using preset language settings for '{language}' where specific configuration is not provided.")

    skip_present = flask_request.args.get("skip_present", True)
    if isinstance(skip_present, str):
        skip_present = get_bool_expression(skip_present, True)
    if skip_present:
        app.logger.info("Skipping present saved steps")

    skip_steps = flask_request.args.get("skip_steps", False)
    omit_pipeline_steps = []
    if skip_steps:
        omit_pipeline_steps = get_omit_pipeline_steps(skip_steps)

    return_statistics = flask_request.args.get("return_statistics", False)
    if isinstance(return_statistics, str):
        return_statistics = get_bool_expression(return_statistics, True)

    return pipeline_query_params(corpus, language, skip_present, omit_pipeline_steps, return_statistics)


def get_data_server_config(document_server_config: Union[FileStorage, dict], app: flask.Flask):
    base_config = {"url": "http://localhost", "port": "9008", "index": "documents", "size": "30", "label_key": "label",
                   "other_id": "id"}
    try:
        if isinstance(document_server_config, FileStorage):
            _config = yaml.safe_load(document_server_config.stream)
        elif isinstance(document_server_config, dict):
            _config = document_server_config.copy()
        else:
            raise Exception("Document server config is not of type 'FileStorage' or 'dict'!")
        for k, v in base_config.items():
            if k not in _config:
                if v is None or (isinstance(v, str) and v.lower() == "none"):
                    _config[k] = None
                    continue
                _config[k] = get_bool_expression(v, v) if isinstance(v, str) else v
        base_config = _config
        base_config["replace_keys"] = get_dict_expression(_config.pop("replace_keys", {'text': 'content'}))
    except Exception as e:
        app.logger.error(f"Couldn't read config file: {e}\n Using default values '{base_config}'.")
    return base_config


def check_data_server(
        config: Union[Munch, dict],
):
    for _url_key in ["url", "alternate_url"]:
        _url = config.get(_url_key, None)
        if _url is None:
            continue
        final_url = f"{_url.rstrip('/')}:{config.get('port', '9008')}/{config.get('index', 'documents').lstrip('/').rstrip('/')}/_count"
        try:
            _response = requests.get(final_url)
        except requests.exceptions.RequestException as e:
            continue
        if _count := _response.json().get('count', False):
            if int(_count) > 0:
                config["url"] = _url
                config.pop("alternate_url", None)
                return True
    return False


def check_es_source_for_id(document_hit: dict, other_id: str):
    _source = document_hit.get("_source")
    _other_id = False
    if isinstance(other_id, str):
        _other_id = not other_id.lower() == "id"

    if _other_id:
        if _source.get(other_id, False):
            _id = _source.pop(other_id)
            return _source | {"id": _id}
        else:
            return {"id": document_hit.get("_id", "")} | _source
    return _source if _source.get("id", False) else {"id": document_hit.get("_id", "")} | _source


def is_skip_doc(document_hit: dict, doc_filter: Iterable, inverse_filter: bool = False) -> bool:
    _source = document_hit.get("_source")
    _name = _source.get("name", False)
    if _name:
        _skip = any((f.lower() in _name.lower()) for f in doc_filter)
        if inverse_filter:
            return not _skip
        return _skip
    return False


def get_documents_from_es_server(
        url: str, port: Union[str, int], index: str, size: int = 30, other_id: str = "id", doc_name_filter: list = None,
        inverse_filter: bool = False,
):
    """Gets documents from a specified Elasticsearch server and index.

    Keyword arguments:
    url -- the base url of the Elasticsearch server
    port -- the port of the Elasticsearch server
    index -- the name of the Elasticsearch index
    size -- the size of each batch for the scroll
    other_id -- the field in the document source that should be used as id
    doc_name_filter -- list of name parts that should be filtered (i.e. not returned) works only on a 'name' field
    inverse_filter -- boolean indicating whether or not the doc_name_filter works as a positive filter (i.e. only those are returned)
    """
    _params = {"size": f"{size}", "scroll": "1m"}
    _filter = False
    if isinstance(doc_name_filter, Iterable) and not isinstance(doc_name_filter, (str, bytes)) and len(
            doc_name_filter) > 0:
        _filter = True
    final_url = f"{url.rstrip('/')}:{port}/{index.lstrip('/').rstrip('/')}/_search"
    _first_page = requests.get(final_url, params=_params).json()
    _scroll_id = _first_page.get("_scroll_id")
    _total_documents = _first_page.get("hits").get("total").get("value")

    for _scroll_index in range(0, _total_documents, size):
        if _scroll_index == 0:
            for document in _first_page.get("hits").get("hits"):
                if _filter and is_skip_doc(document, doc_name_filter, inverse_filter):
                    continue
                yield check_es_source_for_id(document, other_id)
        else:
            _response = requests.post(
                url=f"{url.rstrip('/')}:{port}/_search/scroll",
                json={"scroll_id": _scroll_id, "scroll": "1m"}).json()
            for document in _response.get("hits").get("hits"):
                if _filter and is_skip_doc(document, doc_name_filter, inverse_filter):
                    continue
                yield check_es_source_for_id(document, other_id)


def populate_running_processes(app: flask.Flask, path: str, running_processes: dict):
    for process in get_all_processes(path):
        _finished = [_finished_step.get("name") for _finished_step in process.get("status", [])]
        _process_name = process.get("name", None)
        if _process_name is None:
            app.logger.warning(f"Skipping process entry with no name and '{_finished}' steps.")
            continue

        for _step, _rank in steps_relation_dict.items():
            if _step not in _finished:
                add_status_to_running_process(_process_name, _step, ProcessStatus.NOT_PRESENT, running_processes)
            else:
                add_status_to_running_process(_process_name, _step, ProcessStatus.FINISHED, running_processes)
    return running_processes


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
                                        "name": _step,
                                        "status": ProcessStatus.FINISHED})
            _ord_dict = OrderedDict()
            _ord_dict["name"] = _proc_name
            _ord_dict["status"] = sorted(_steps_list, key=lambda x: x.get("rank", -1))
            _process_detailed.append(_ord_dict)
    return _process_detailed


def start_processes(
        app: flask.Flask,
        processes: tuple,
        process_name: str,
        process_tracker: dict[str, dict],
        thread_store: dict[str, StoppableThread]
):
    #ToDo: maybe think about persisting already finished objects so that each step
    # doesn't need to load them again?
    _name_marker = {
        StepsName.DATA: "**data**, embedding, clustering, graph, integration",
        StepsName.EMBEDDING: "data, **embedding**, clustering, graph, integration",
        StepsName.CLUSTERING: "data, embedding, **clustering**, graph, integration",
        StepsName.GRAPH: "data, embedding, clustering, **graph**, integration",
        StepsName.INTEGRATION: "data, embedding, clustering, graph, **integration**",
    }
    for process_obj, _fact, _name in processes:
        if thread_store.get(process_name, None) is not None:
            if thread_store[process_name].stopped():
                add_status_to_running_process(
                    process_name=process_name,
                    step_name=_name,
                    step_status=ProcessStatus.NOT_PRESENT,
                    running_processes=process_tracker
                )
                continue
        try:
            process_obj.start_process(
                cache_name=process_name,
                process_factory=_fact,
                process_tracker=process_tracker
            )
        except FileNotFoundError as e:
            app.logger.warning(f"Something went wrong with one of the previous steps: {_name_marker[_name]}."
                               f"\nThere is a pickle file missing: {e}")


def start_thread(
        app: flask.Flask,
        process_name: str,
        pipeline_thread: StoppableThread,
        threading_store: dict[str, StoppableThread]
):
    app.logger.info(f"Starting thread for '{process_name}'.")
    pipeline_thread.start()
    threading_store[process_name] = pipeline_thread
    sleep(1)
    return True


def stop_thread(
        app: flask.Flask,
        process_name: str,
        threading_store: dict[str, StoppableThread],
        process_tracker: dict[str, dict],
        hard_stop: bool = False
):
    # ToDo: delete config for aborted steps
    # ToDo: maybe allow for hard stop? -> terminate the thread even if a step is not yet finished
    app.logger.info(f"Trying to stop thread for '{process_name}'.")
    _thread = threading_store.get(process_name, None)
    _process = process_tracker.get(process_name, None)

    if _thread is None or _process is None:
        _msg = f"No thread/process for '{process_name}' was found."
        logging.error(_msg)
        return Response(_msg, status=int(HTTPResponses.INTERNAL_SERVER_ERROR))

    _current_step = next((step for step in sorted(_process.get('status', {}), key=lambda p: p.get('rank', 99)) if step.get('status', None) == ProcessStatus.RUNNING), None)
    if _current_step is None:
        _msg = f"Couldn't find a running step in the pipeline '{process_name}'."
        logging.error(_msg)
        return Response(_msg, status=int(HTTPResponses.INTERNAL_SERVER_ERROR))
    _thread.stop()

    try:
        for _step in sorted(_process.get('status', {}), key=lambda p: p.get('rank')):
            if _step.get('rank') <= _current_step.get('rank'):
                if _step.get('name') == _current_step.get('name'):
                    process_tracker[process_name]["status"][_step.get('rank') - 1]["status"] = ProcessStatus.STOPPED
                continue
            process_tracker[process_name]["status"][_step.get('rank') - 1]["status"] = ProcessStatus.ABORTED
    except Exception:
        pass

    app.logger.info(
        f"Thread for '{process_name}' will be stopped after the present step ('{_current_step.get('name', None)}') is completed.")
    return Response("Process will be stopped.", status=int(HTTPResponses.ACCEPTED))


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
        try:
            if language is not None:
                processor.config["language"] = language
            yaml.safe_dump(processor.config, config_save)
        except RepresenterError:
            yaml.safe_dump(processor.serializable_config, config_save)
    return process_name


def load_configs(app: flask.app, process_name: str, path_to_configs: Union[pathlib.Path, str], ext: str = "yaml"):
    final_config = {"config": {}}
    processes = [
        (StepsName.DATA, PreprocessingUtil,),
        (StepsName.EMBEDDING, PhraseEmbeddingUtil,),
        (StepsName.CLUSTERING, ClusteringUtil,),
        (StepsName.GRAPH, GraphCreationUtil,)
    ]
    _language = set()
    for _step, _proc in processes:
        process_obj: BaseUtil = _proc(app=app, file_storage=path_to_configs)
        process_obj.process_name = process_name
        process_obj.file_storage_path = process_name
        key, val = process_obj.read_stored_config()
        _language.add(val.pop('language', 'en'))
        final_config["config"][key] = val
    if len(_language) == 1:
        _language = _language.pop()
    else:
        _language = 'en'
    final_config["language"] = _language
    return final_config


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
            return {
                "error": f"Couldn't find graph pickle '{data}_graph.pickle'. Probably steps before failed; check the logs."}
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
                graph_path = visualize_graph(
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
        graph_create = GraphCreationUtil(app, path)

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


def get_dict_expression(dict_str: str):
    if isinstance(dict_str, str):
        # e.g. "{'text': 'content'}"
        if not dict_str.startswith("{") and not dict_str.endswith("}"):
            return dict_str
        _str = dict_str[1:-1].split(",")
        _return_dict = dict()
        for _s in _str:
            _split_s = _s.split(":")
            if len(_split_s) != 2:
                break
            _return_dict[_split_s[0].strip().strip("'").strip('"')] = _split_s[1].strip().strip("'").strip('"')

        return _return_dict
    else:
        return dict_str


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


def add_documents_to_concept_graphs(
        content: Iterable[Union[str, dict]],
        data_processing: Optional[data_functions.DataProcessingFactory.DataProcessing] = None,
        embedding_processing: Optional[embedding_functions.SentenceEmbeddingsFactory.SentenceEmbeddings] = None,
        document_store_cls: str = "src.marqo_external_utils.MarqoDocumentStore",
        embedding_store_cls: str = "src.marqo_external_utils.MarqoEmbeddingStore",
        document_cls: str = "src.marqo_external_utils.MarqoDocument",
):
    """
    content: [
        Union[
            {
                id: str
                content: str
                corpus: str
                label: str
                name: str
            },
            str
        ]
    ]
    """
    document_store = locate(document_store_cls)
    embedding_store = locate(embedding_store_cls)
    document = locate(document_cls)

    if content is None :
        return jsonify("No content provided.")
    _source = embedding_processing.source
    _client_key = list(set(_source.keys()).intersection(["client_url", "url", "client", "clienturl"])) if isinstance(_source, dict) else "none"
    _client_key = _client_key[0] if len(_client_key) > 0 else None
    _index_key = list(set(_source.keys()).intersection(["index_name", "index", "indexname"])) if isinstance(_source, dict) else "none"
    _index_key = _index_key[0] if len(_index_key) > 0 else None

    _chunk_result = data_processing.process_external_docs(
        content = [
            {"name": doc.get("name", None), "content": doc.get("content", ""), "label": doc.get("label", None)}
            for doc in content
        ]
    )
    text_list = []
    idx_dict = defaultdict(list)
    for idx, _chunk in enumerate(sorted(((_result["text"], set(_doc["id"] for _doc in _result["doc"]), )
                                         for _result in _chunk_result), key=lambda _result: _result[0])):
        for _doc in _chunk[1]:
            idx_dict[_doc].append(idx)
        text_list.append(_chunk[0])
    _embedding_result = embedding_processing.encode_external(content=text_list)

    embedding_store_impl: EmbeddingStore = embedding_store(
        client_url=_source.get(_client_key, "http://localhost:8882"),
        index_name=_source.get(_index_key, "default"),
        create_index=False,
        vector_dim=embedding_processing.embedding_dim
    )
    doc_store_impl: DocumentStore = document_store(
        embedding_store=embedding_store_impl
    )
    added_embeddings = doc_store_impl.add_documents(
        [document(phrases=np.take(text_list, idx, 0), embeddings=np.take(_embedding_result.astype("float64"), idx, 0),)
         for idx in idx_dict.values()]
    )
    return jsonify("Processed documents.")


if __name__ == "__main__":
    count = 0
    for i in get_documents_from_es_server(url="http://localhost", port=9008, index="documents"):
        count += 1
        print(i.get('id'))
    print(count)
