import collections
import logging
import os
import pathlib
import pickle
import shutil
import sys
import uuid
from collections import OrderedDict, defaultdict, namedtuple
from inspect import getfullargspec
from pydoc import locate
from time import sleep
from typing import Union, Iterable, Optional, cast

import flask
import networkx as nx
import numpy as np
import requests
import yaml
from flask import request, jsonify, render_template_string, Response
from munch import Munch
from werkzeug.datastructures import FileStorage
from yaml.representer import RepresenterError

from preprocessing_util import PreprocessingUtil
from embedding_util import PhraseEmbeddingUtil
from clustering_util import ClusteringUtil
from graph_creation_util import GraphCreationUtil, visualize_graph
from integration_util import ConceptGraphIntegrationUtil
from load_utils import FactoryLoader
from main_utils import (
    ProcessStatus,
    HTTPResponses,
    StepsName,
    pipeline_query_params,
    steps_relation_dict,
    add_status_to_running_process,
    PipelineLanguage,
    get_bool_expression,
    StoppableThread,
    string_conformity,
    BaseUtil,
    transform_document_addition_results,
    PersistentObjects,
)
from src.rag.TextSplitters import PreprocessedSpacyTextSplitter
from src.rag.embedding_stores.AbstractEmbeddingStore import ChunkEmbeddingStore

sys.path.insert(0, "src")
from src import data_functions, cluster_functions, embedding_functions
from src.graph_functions import GraphIncorp
from src.util_functions import DocumentStore, EmbeddingStore, save_pickle


pipeline_json_config = namedtuple(
    "pipeline_json_config",
    [
        "name",
        "language",
        "document_server",
        "vectorstore_server",
        "data",
        "embedding",
        "clustering",
        "graph",
    ],
)

document_adding_json = namedtuple(
    "document_adding_json",
    ["language", "documents", "document_server", "vectorstore_server"],
)


rag_config_json = namedtuple(
    "rag_config_json",
    ["chatter", "api_key", "language", "prompt_template", "vectorstore_server"],
)


def parse_pipeline_config_json(response_json) -> pipeline_json_config:
    config = Munch.fromDict(response_json)
    try:
        return pipeline_json_config(
            string_conformity(config.get("name", None)),
            config.get("language", None),
            config.get("document_server", None),
            config.get(
                "vectorstore_server",
                config.get(
                    "vector_store_server", config.get("vector-store_server", None)
                ),
            ),
            config.config.get("data", Munch()),
            config.config.get("embedding", Munch()),
            config.config.get("clustering", Munch()),
            config.config.get("graph", Munch()),
        )
    except AttributeError as e:
        logging.error(
            f"Json body/configuration seems to be malformed: no 'config' entry was provided.\n{e}"
        )
        return pipeline_json_config(
            string_conformity(config.get("name", None)),
            config.get("language", None),
            config.get("document_server", None),
            config.get("vectorstore_server", None),
            Munch(),
            Munch(),
            Munch(),
            Munch(),
        )


def parse_document_adding_json(response_json) -> Optional[document_adding_json]:
    try:
        config = Munch.fromDict(response_json)
        # _id_key = list(set(config.keys()).intersection(["id", "_id"]))
        # _id = _id_key[0] if _id_key else "none"
        return document_adding_json(
            config.get("language", None),
            config.get("documents", []),
            config.get("document_server", None),
            config.get("vectorstore_server", None),
        )
    except Exception as e:
        logging.error(f"Content json parsing error: '{e}'")
        return None


def parse_rag_config_json(response_json) -> Optional[rag_config_json]:
    try:
        config = Munch.fromDict(response_json)
        _chatter = config.get("chatter", None)
        _api_key = config.get("api_key", "")
        _lang = config.get("language", "en")
        _prompt = config.get("prompt_template", None)
        _vs = config.get(
            "vectorstore_server",
            config.get("vector_store_server", config.get("vector-store_server", None)),
        )
        return rag_config_json(
            {} if (_chatter is None or not isinstance(_chatter, dict)) else _chatter,
            _api_key,
            _lang,
            None if len(_prompt) == 0 else _prompt,
            _vs,
        )
    except Exception as e:
        logging.error(f"Content json parsing error: '{e}'")
        return None


def get_doc_ids(response_json: dict):
    try:
        if intersection := {"doc_ids", "doc_id", "ids", "id"}.intersection(
            k for k in response_json.keys()
        ):
            return response_json.get(list(intersection)[0])
    except Exception as e:
        logging.warning(f"Couldn't get document ids from request json: '{e}'")
    return []


def get_pipeline_query_params(
    app: flask.Flask,
    flask_request: flask.Request,
    running_processes: dict,
    config_obj_json: pipeline_json_config,
) -> Union[pipeline_query_params, tuple]:
    if config_obj_json is not None and config_obj_json.name is not None:
        corpus = string_conformity(config_obj_json.name)
    else:
        corpus = string_conformity(flask_request.args.get("process", "default"))
    if corpus_status := running_processes.get(corpus, False):
        if any(
            [
                v.get("status", None) == ProcessStatus.RUNNING
                for v in corpus_status.get("status", [])
            ]
        ):
            return jsonify(
                name=corpus,
                error=f"A process is currently running for this corpus. Use '/status?process={corpus}' for specifics.",
            ), int(HTTPResponses.FORBIDDEN)
    app.logger.info(f"Using process name '{corpus}'")
    if config_obj_json is not None and config_obj_json.language is not None:
        language = PipelineLanguage.language_from_string(config_obj_json.language)
    else:
        language = PipelineLanguage.language_from_string(
            str(flask_request.args.get("lang", "en"))
        )
    app.logger.info(
        f"Using preset language settings for '{language}' where specific configuration is not provided."
    )

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

    return pipeline_query_params(
        corpus, language, skip_present, omit_pipeline_steps, return_statistics
    )


def get_data_server_config(
    document_server_config: Union[FileStorage, dict], app: flask.Flask
):
    base_config = {
        "url": "http://localhost",
        "port": "9008",
        "index": "documents",
        "size": "30",
        "label_key": "label",
        "other_id": "id",
    }
    try:
        if isinstance(document_server_config, FileStorage):
            _config = yaml.safe_load(document_server_config.stream)
        elif isinstance(document_server_config, dict):
            _config = document_server_config.copy()
        else:
            raise Exception(
                "Document server config is not of type 'FileStorage' or 'dict'!"
            )
        for k, v in base_config.items():
            if k not in _config:
                if v is None or (isinstance(v, str) and v.lower() == "none"):
                    _config[k] = None
                    continue
                _config[k] = get_bool_expression(v, v) if isinstance(v, str) else v
        base_config = _config
        base_config["replace_keys"] = get_dict_expression(
            _config.pop("replace_keys", {"text": "content"})
        )
    except Exception as e:
        app.logger.error(
            f"Couldn't read config file: {e}\n Using default values '{base_config}'."
        )
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
        if _count := _response.json().get("count", False):
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
    return (
        _source
        if _source.get("id", False)
        else {"id": document_hit.get("_id", "")} | _source
    )


def is_skip_doc(
    document_hit: dict, doc_filter: Iterable, inverse_filter: bool = False
) -> bool:
    _source = document_hit.get("_source")
    _name = _source.get("name", False)
    if _name:
        _skip = any((f.lower() in _name.lower()) for f in doc_filter)
        if inverse_filter:
            return not _skip
        return _skip
    return False


def get_documents_from_es_server(
    url: str,
    port: Union[str, int],
    index: str,
    size: int = 30,
    other_id: str = "id",
    doc_name_filter: list = None,
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
    if (
        isinstance(doc_name_filter, Iterable)
        and not isinstance(doc_name_filter, (str, bytes))
        and len(doc_name_filter) > 0
    ):
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
                json={"scroll_id": _scroll_id, "scroll": "1m"},
            ).json()
            for document in _response.get("hits").get("hits"):
                if _filter and is_skip_doc(document, doc_name_filter, inverse_filter):
                    continue
                yield check_es_source_for_id(document, other_id)


def populate_running_processes(
    app: flask.Flask, path: Union[str, pathlib.Path], running_processes: dict
):
    for process in get_all_processes(path):
        _finished = [
            _finished_step.get("name") for _finished_step in process.get("status", [])
        ]
        _process_name = process.get("name", None)
        if _process_name is None:
            app.logger.warning(
                f"Skipping process entry with no name and '{_finished}' steps."
            )
            continue

        for _step, _rank in steps_relation_dict.items():
            if _step not in _finished:
                add_status_to_running_process(
                    _process_name, _step, ProcessStatus.NOT_PRESENT, running_processes
                )
            else:
                add_status_to_running_process(
                    _process_name, _step, ProcessStatus.FINISHED, running_processes
                )
    return running_processes


def get_all_processes(path: Union[str, pathlib.Path]):
    if isinstance(path, str):
        path = pathlib.Path(path)
    _process_detailed = list()
    for _proc in path.glob("*"):
        if _proc.is_dir() and not _proc.stem.startswith("."):
            _proc_name = _proc.stem.lower()
            _steps_list = list()
            for _pickle in pathlib.Path(path / _proc_name).glob("*.pickle"):
                _pickle_stem = _pickle.stem.lower()
                _step = _pickle_stem.removeprefix(f"{_proc_name}_")
                if steps_relation_dict.get(_step, False):
                    _steps_list.append(
                        {
                            "rank": steps_relation_dict.get(_step),
                            "name": _step,
                            "status": ProcessStatus.FINISHED,
                        }
                    )
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
    thread_store: dict[str, StoppableThread],
    active_process_objs: dict[str, dict],
    last_step: str,
):
    _name_marker = {
        StepsName.DATA: "**data**, embedding, clustering, graph, integration",
        StepsName.EMBEDDING: "data, **embedding**, clustering, graph, integration",
        StepsName.CLUSTERING: "data, embedding, **clustering**, graph, integration",
        StepsName.GRAPH: "data, embedding, clustering, **graph**, integration",
        StepsName.INTEGRATION: "data, embedding, clustering, graph, **integration**",
    }
    for process_obj, _fact, _name in processes:
        process_obj: BaseUtil
        this_thread: Optional[StoppableThread] = thread_store.get(process_name, None)
        if this_thread is not None and this_thread.set_to_stop:
            add_status_to_running_process(
                process_name=process_name,
                step_name=_name,
                step_status=ProcessStatus.NOT_PRESENT,
                running_processes=process_tracker,
            )
            continue
        log_warning = f"Something went wrong with one of the previous steps: {_name_marker[_name]}."
        if any(
            [
                True
                for d in process_tracker.get(process_name, {}).get("status", [])
                if (d.get("name") == _name and d.get("status") == ProcessStatus.ABORTED)
            ]
        ):
            app.logger.warning(log_warning + f"\n So this one was aborted: '{_name}'.")
            continue
        try:
            if this_thread.set_to_stop:
                app.logger.info(
                    f"Thread for '{process_name}' was stopped before this step '{_name}'. So subsequent steps will not be started."
                )
                return
            process_obj.start_process(
                cache_name=process_name,
                process_factory=_fact,
                process_tracker=process_tracker,
                active_process_objs=active_process_objs,
                thread=this_thread,
            )
            if _name == last_step:
                app.logger.info(f"Pipeline finished with last step: '{_name}'.")
        except FileNotFoundError as e:
            app.logger.warning(log_warning + f"\nThere is a pickle file missing: {e}")


def start_thread(
    app: flask.Flask,
    process_name: str,
    pipeline_thread: StoppableThread,
    threading_store: Optional[dict[str, StoppableThread]],
):
    app.logger.info(f"Starting thread for '{process_name}'.")
    pipeline_thread.start()
    # if threading_store is not None: threading_store[process_name] = pipeline_thread
    sleep(1.5)
    return True


def stop_thread(
    app: flask.Flask,
    process_name: str,
    threading_store: dict[str, StoppableThread],
    process_tracker: dict[str, dict],
    hard_stop: bool = False,
):
    # ToDo: delete config for aborted steps
    # ToDo: maybe allow for hard stop? -> terminate the thread even if a step is not yet finished
    app.logger.info(f"Trying to stop thread for '{process_name}'.")
    if hard_stop:
        app.logger.warning(f"Hard stopping a thread is not implemented.")
        hard_stop = False
    _thread = threading_store.get(process_name, None)
    _process = process_tracker.get(process_name, None)

    if _thread is None or _process is None:
        _msg = f"No thread/process for '{process_name}' was found."
        logging.error(_msg)
        return Response(_msg, status=int(HTTPResponses.INTERNAL_SERVER_ERROR))

    _current_step = next(
        (
            step
            for step in sorted(
                _process.get("status", {}), key=lambda p: p.get("rank", 99)
            )
            if step.get("status", None) == ProcessStatus.RUNNING
        ),
        None,
    )
    if _current_step is None:
        _msg = f"Couldn't find a running step in the pipeline '{process_name}'."
        logging.error(_msg)
        return Response(_msg, status=int(HTTPResponses.INTERNAL_SERVER_ERROR))
    _thread.stop(hard_stop=hard_stop)

    try:
        for _step in sorted(
            _process.get("status", {}), key=lambda p: p.get("rank", 99)
        ):
            if _step.get("rank") <= _current_step.get("rank"):
                if _step.get("name") == _current_step.get("name"):
                    process_tracker[process_name]["status"][_step.get("rank") - 1][
                        "status"
                    ] = ProcessStatus.STOPPED
                continue
            process_tracker[process_name]["status"][_step.get("rank") - 1][
                "status"
            ] = ProcessStatus.ABORTED
    except Exception:
        pass

    app.logger.info(
        f"Thread for '{process_name}' will be stopped after the present step ('{_current_step.get('name', None)}') is completed."
    )
    return Response("Process will be stopped.", status=int(HTTPResponses.ACCEPTED))


def read_config(
    app: flask.Flask,
    processor: any,
    process_type: str,
    process_name: Optional[str] = None,
    config: Optional[dict] = None,
    language: Optional[str] = None,
    mode: str = "yaml",
):
    _language = config.get("language", language) if config is not None else language
    app.logger.info(f"Reading config ({process_type}) ...")
    processor.read_config(
        config=config if config is not None else request.files.get("config", None),
        process_name=process_name,
        language=(
            _language if process_type in [StepsName.DATA, StepsName.EMBEDDING] else None
        ),
    )
    # pyyaml doesn't handle 'None' so we need to convert them
    if mode.lower() in ["yaml", "yml"]:
        for k, v in processor.config.items():
            if isinstance(v, str) and v.lower() == "none":
                processor.config[k] = None
    process_name_conf = processor.config.pop("corpus_name", "default")
    if process_name is None:
        process_name = process_name_conf
    process_name = string_conformity(process_name)
    processor.file_storage_path = process_name
    processor.process_name = process_name
    app.logger.info(
        f"Parsed the following arguments for '{processor.process_name}':\n\t{processor.config}"
    )

    with pathlib.Path(
        pathlib.Path(processor._file_storage)
        / pathlib.Path(f"{process_name}_{process_type}_config.yaml")
    ).open("w") as config_save:
        try:
            if _language is not None:
                processor.config["language"] = _language
            yaml.safe_dump(processor.config, config_save)
        except RepresenterError:
            yaml.safe_dump(processor.serializable_config, config_save)
    return process_name


def load_configs(
    app: flask.app,
    process_name: str,
    path_to_configs: Union[pathlib.Path, str],
    ext: str = "yaml",
):
    final_config = {"config": {}}
    processes = [
        (
            StepsName.DATA,
            PreprocessingUtil,
        ),
        (
            StepsName.EMBEDDING,
            PhraseEmbeddingUtil,
        ),
        (
            StepsName.CLUSTERING,
            ClusteringUtil,
        ),
        (
            StepsName.GRAPH,
            GraphCreationUtil,
        ),
        (
            StepsName.INTEGRATION,
            ConceptGraphIntegrationUtil,
        ),
    ]
    _language = collections.Counter()
    for _step, _proc in processes:
        process_obj: BaseUtil = _proc(app=app, file_storage=path_to_configs)
        process_obj.process_name = process_name
        process_obj.file_storage_path = process_name
        key, val = process_obj.read_stored_config()
        _language.update({val.pop("language", "en"): 1})
        final_config["config"][key] = val
    if len(_language) == 0:
        _language = "en"
    else:
        _language = _language.most_common(1)[0][0]
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
            _return_dict[_split_s[0].strip().strip("'").strip('"')] = (
                _split_s[1].strip().strip("'").strip('"')
            )

        return _return_dict
    else:
        return dict_str


def get_query_param_help_text(param: str):
    return {
        "process": "process: name of the process (e.g. ``corpus_name`` in config); if not provided, uses 'default'",
        "exclusion_ids": "exclusion_ids: list of concept ids (as returned by e.g. ``/clustering/concepts``), "
        "that shall be excluded from the final graphs in the form of ``[ID1, ID2, etc.]``",
        "draw": "draw: `true` or `false` - whether the response shall be a rendered graph or plain json",
    }.get(param, "No help text available.")


def get_omit_pipeline_steps(steps: object) -> list[str]:
    step_set = {"data", "embedding", "clustering", "graph", "integration"}
    if isinstance(steps, str):
        steps = steps.strip("([{}])")
        return [s.lower() for s in steps.split(",") if s.lower() in step_set]
    return []


def add_documents_to_concept_graphs(
        #ToDo?: ``store_permanently`` only changes whether new phrases will be stored in graphs!
        #   --> regardless of the former argument:
        #       - docs won't be stored in the processed_data
        #       - docs (their phrase embeddings) will be stored in the vector store
    content_json: document_adding_json,
    data_processing: Optional[
        data_functions.DataProcessingFactory.DataProcessing
    ] = None,
    embedding_processing: Optional[
        embedding_functions.SentenceEmbeddingsFactory.SentenceEmbeddings
    ] = None,
    graph_processing: Optional[list[nx.Graph]] = None,
    storage_path: Optional[Union[str, pathlib.Path]] = None,
    process_name: str = "default",
    store_permanently: bool = True,
    document_store_cls: str = "src.marqo_external_utils.MarqoDocumentStore",
    embedding_store_cls: str = "src.marqo_external_utils.MarqoEmbeddingStore",
    document_cls: str = "src.marqo_external_utils.MarqoDocument",
):
    try:
        document_store = locate(document_store_cls)
        embedding_store = locate(embedding_store_cls)
        document = locate(document_cls)

        if content_json.documents is None:
            return {"error": "No content provided."}, HTTPResponses.BAD_REQUEST

        ###
        try:
            data_processing = (
                FactoryLoader.load_data(str(storage_path.resolve()), process_name)
                if data_processing is None
                else data_processing
            )
            embedding_processing = (
                FactoryLoader.load_embedding(
                    str(storage_path.resolve()),
                    process_name,
                    data_processing,
                    (
                        None
                        if content_json.vectorstore_server is None
                        else content_json.vectorstore_server
                    ),
                )
                if embedding_processing is None
                else embedding_processing
            )
        except FileNotFoundError as e:
            _missing = "data" if data_processing is None else "embedding"
            return {
                "error": f"The serialized object for '{_missing}' doesn't seem to be present. Please finish the complete pipeline for the process '{process_name}' first."
            }, HTTPResponses.NOT_FOUND
        has_graph = True
        try:
            graph_processing = (
                FactoryLoader.load_graph(str(storage_path.resolve()), process_name)
                if graph_processing is None
                else graph_processing
            )
        except FileNotFoundError as e:
            has_graph = False
            logging.warning(
                f"The serialized object for 'graph' doesn't seem to be present. Storing the document into the vector store will still be performed."
            )
        if (
            content_json.vectorstore_server is None
            and embedding_processing.source is None
        ):
            return {
                "error": "Only adding documents with a vectorstore server setup is supported; no vectorstore configured."
            }, HTTPResponses.NOT_IMPLEMENTED
        if embedding_processing.source is None:
            embedding_processing.source = content_json.vectorstore_server
        if len(content_json.documents) > 0 and isinstance(
            content_json.documents[0], dict
        ):
            # ToDo
            pass
        else:
            return {
                "error": "Right now only processing of documents as json is supported."
            }, HTTPResponses.NOT_IMPLEMENTED
        ###

        _source = embedding_processing.source
        _client_key = (
            list(
                set(_source.keys()).intersection(
                    ["client_url", "url", "client", "clienturl"]
                )
            )
            if isinstance(_source, dict)
            else "none"
        )
        _client_key = _client_key[0] if len(_client_key) > 0 else None
        _index_key = (
            list(set(_source.keys()).intersection(["index_name", "index", "indexname"]))
            if isinstance(_source, dict)
            else "none"
        )
        _index_key = _index_key[0] if len(_index_key) > 0 else None

        doc_content = lambda x: x if isinstance(x, dict) else {"content": x}
        _chunk_result = data_processing.process_external_docs(
            content=[
                {
                    "id": doc_content(doc).get("id", str(uuid.uuid4())),
                    "name": doc_content(doc).get("name", None),
                    "content": doc_content(doc).get("content", ""),
                    "label": doc_content(doc).get("label", None),
                }
                for doc in content_json.documents
            ]
        )
        text_list = []
        idx_dict = defaultdict(list)
        for idx, _chunk in enumerate(
            # sorted(
            (
                (
                    _result["text"],
                    set(_doc["id"] for _doc in _result["doc"]),
                )
                for _result in _chunk_result
            )  # ,
            # key=lambda _result: _result[0],
            # )
        ):
            for _doc in _chunk[1]:
                idx_dict[_doc].append(idx)
            text_list.append(_chunk[0])
        _embedding_result = embedding_processing.encode_external(content=text_list)

        embedding_store_impl: EmbeddingStore = embedding_store(
            client_url=_source.get(_client_key, "http://localhost:8882"),
            index_name=_source.get(_index_key, "default"),
            create_index=False,
            vector_dim=embedding_processing.embedding_dim,
        )
        doc_store_impl: DocumentStore = document_store(
            embedding_store=embedding_store_impl
        )
        added_embeddings = doc_store_impl.add_documents(
            [
                (
                    document(
                        phrases=np.take(text_list, idx, 0),
                        embeddings=np.take(_embedding_result.astype("float64"), idx, 0),
                        doc_id=_id,
                    ),
                    {
                        "offsets": [
                            x.get("offsets", [])
                            for _dict in np.take(_chunk_result, idx, 0)
                            for x in _dict.get("doc", [])
                            if x.get("id", "") == _id
                        ],
                        "text": [
                            _dict.get("text", "")
                            for _dict in np.take(_chunk_result, idx, 0)
                        ],
                    },
                )
                for _id, idx in idx_dict.items()
            ],
            as_tuple=True,
        )
        if store_permanently:
            graph_storage_path = pathlib.Path(
                storage_path / f"{process_name}_{StepsName.GRAPH}"
            ).resolve()
            save_pickle(
                GraphIncorp.with_graphs(graph_processing)
                .incorporate_phrases(
                    transform_document_addition_results(
                        (
                            (
                                k,
                                v.get("with_graph"),
                            )
                            for k, v in added_embeddings.items()
                        )
                    ).items()
                )
                .graphs,
                graph_storage_path,
            )
    except Exception as e:
        return {
            "error": str(e) + "\n--- please consult the logs!"
        }, HTTPResponses.INTERNAL_SERVER_ERROR
    return added_embeddings, HTTPResponses.OK


def delete_pipeline(
    app: flask.Flask,
    base_path: pathlib.Path,
    process_name: str,
    running_processes: dict,
    cached_processes: dict,
    wait_for_thread: Optional[StoppableThread] = None,
):
    app.logger.info(f"Deleting pipeline '{process_name}'.")
    if wait_for_thread is not None:
        app.logger.info(
            f"Waiting for running pipeline thread of '{process_name}' to stop before deleting..."
        )
        wait_for_thread.join()
    for _store in [running_processes, cached_processes]:
        _ = _store.pop(process_name, None)
    for _step in [
        PreprocessingUtil,
        PhraseEmbeddingUtil,
        ClusteringUtil,
        GraphCreationUtil,
        ConceptGraphIntegrationUtil,
    ]:
        _process_util = _step(
            app=app, file_storage=str(pathlib.Path(base_path / process_name).resolve())
        )
        _process_util.process_name = process_name
        _process_util.delete_process()
    shutil.rmtree(pathlib.Path(base_path / process_name))


def initialize_chunk_vectorstore(
    process_name: str,
    config: Optional[dict],
    chunk_store: str = "src.rag.embedding_stores.MarqoChunkEmbeddingStore.MarqoChunkEmbeddingStore",
    force_init: bool = False,
):
    if config is None:
        config = {"index_settings": None}
    if config.get("index_settings", None) is None or len(config["index_settings"]) == 0:
        config["index_settings"] = {
            "type": "structured",
            "model": "multilingual-e5-base",
            "normalizeEmbeddings": True,
            "textPreprocessing": {
                "splitLength": 3,
                "splitOverlap": 1,
                "splitMethod": "sentence",
            },
            "allFields": [
                {
                    "name": "doc_id",
                    "type": "text",
                    "features": ["lexical_search", "filter"],
                },
                {
                    "name": "doc_name",
                    "type": "text",
                    "features": ["lexical_search", "filter"],
                },
                {"name": "text", "type": "text", "features": ["lexical_search"]},
            ],
            "tensorFields": ["text"],
        }
    chunk_store: ChunkEmbeddingStore = cast(
        ChunkEmbeddingStore, locate(chunk_store)
    ).from_config(
        index_name=f"{process_name}_rag",
        url=config.pop("url", "http://localhost"),
        port=config.pop("port", 8882),
        force_init=force_init,
        **config,
    )
    return chunk_store


def fill_chunk_vectorstore(
    process: str, persistent_objects: PersistentObjects, **kwargs
) -> bool:
    """

    :param process:
    :param persistent_objects:
    :param kwargs: e.g. splitter=splitter-config-dict
    :return:
    """
    _splitter_class = PreprocessedSpacyTextSplitter
    _split_options = {
        "doc_metadata_key": kwargs.get("splitter", {}).pop(
            "doc_metadata_key", "doc_id"
        ),
        "keep_metadata": kwargs.get("splitter", {}).pop(
            "keep_metadata", ["doc_id", "doc_name"]
        ),
    }
    _splitter_options = {
        k: v
        for k, v in kwargs.pop(
            "splitter", {"chunk_size": 400, "chunk_overlap": 100}
        ).items()
        if k in getfullargspec(_splitter_class).args
    }
    _rag = persistent_objects.active_rag
    if not _rag.initializing:
        _rag.initializing = True
        data_obj = FactoryLoader.with_active_objects(
            str(pathlib.Path(persistent_objects.file_storage_dir, process).resolve()),
            process,
            persistent_objects.current_active_pipeline_objects,
            StepsName.DATA,
        )
        if data_obj is None:
            logging.error(
                f"[fill_chunk_vectorstore] Data object not initialized for process '{process}'. See logs for more information."
            )
            return False
        splitter = _splitter_class(**_splitter_options)

        try:
            _rag.vectorstore.reset_index()
        except Exception as e:
            logging.warning(f"[fill_chunk_vectorstore] {e}")

        _documents = splitter.split_preprocessed_sentences(
            data_obj.processed_docs, **_split_options
        )
        _field = "text"
        _rag.vectorstore.add_chunks(
            [
                dict(
                    {_field: d},
                    **{k: t[1][k] for k in _split_options.get("keep_metadata", [])},
                )
                for t in _documents
                for d in t[0]
            ],
            # _field,
        )

        _rag.initializing = False
        _rag.switch_readiness()
        return True
    else:
        logging.warning(f"[fill_chunk_vectorstore] Already initializing")
        return False


if __name__ == "__main__":
    count = 0
    for i in get_documents_from_es_server(
        url="http://localhost", port=9008, index="documents"
    ):
        count += 1
        print(i.get("id"))
    print(count)
