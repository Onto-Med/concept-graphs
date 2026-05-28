import collections
import logging
import os
import pathlib
import pickle
import shutil
from collections import defaultdict
from inspect import getfullargspec
from pydoc import locate
from typing import Union, Iterable, Optional, cast

import flask
import networkx as nx
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
    PipelineLanguage,
    get_bool_expression,
    StoppableThread,
    string_conformity,
    BaseUtil,
    PersistentObjects,
)
from src.rag.TextSplitters import PreprocessedSpacyTextSplitter
from src.rag.embedding_stores.AbstractEmbeddingStore import ChunkEmbeddingStore
from src import data_functions, cluster_functions, embedding_functions


from src.api.request_parsing import (
    document_adding_json,
    get_doc_ids,
    parse_document_adding_json,
    parse_pipeline_config_json,
    parse_rag_config_json,
    pipeline_json_config,
    rag_config_json,
)


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


from src.pipeline.processes import (
    get_all_processes,
    populate_running_processes,
    start_processes,
    start_thread,
    stop_thread,
)


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


from src.pipeline.document_addition import add_documents_to_concept_graphs


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
            "model": "hf/multilingual-e5-base",
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
