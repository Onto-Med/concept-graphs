import json

from flask import Flask
from flask.logging import default_handler

from clustering_util import ClusteringUtil
from embedding_util import PhraseEmbeddingUtil
from graph_creation_util import GraphCreationUtil
from load_utils import FactoryLoader
from main_methods import *
from main_utils import (
    ProcessStatus,
    HTTPResponses,
    StepsName,
    add_status_to_running_process,
    get_bool_expression,
    StoppableThread,
)
from preprocessing_util import PreprocessingUtil

sys.path.insert(0, "src")
import data_functions
import embedding_functions
import cluster_functions
import integration_functions
import util_functions
import marqo_external_utils

werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.WARN)
root = logging.getLogger()
root.addHandler(default_handler)
[default_handler.setLevel(h) for h in werkzeug_logger.handlers]
app = Flask(__name__)


FILE_STORAGE_TMP = "./tmp"

running_processes = {}
pipeline_threads_store = {}
current_active_pipeline_objects = {}

f_storage = pathlib.Path(FILE_STORAGE_TMP)
if not f_storage.exists():
    f_storage.mkdir()
populate_running_processes(app, FILE_STORAGE_TMP, running_processes)

# ToDo: file with stopwords will be POSTed: #filter_stop: Optional[list] = None,

# ToDo: I downscale the embeddings twice... (that snuck in somehow); once in SentenceEmbeddings via create(down_scale_algorithm)
# ToDo: and once PhraseCluster via create(down_scale_algorithm). I can't remember why I added this to SentenceEmbeddings later on...
# ToDo: but I should make sure, that there is a check somewhere that the down scaling is not applied twice!

# ToDo: make sure that no arguments can be supplied via config that won't work

# ToDo: endpoints with path arguments should throw a response/warning if there is no saved pickle

# ToDo: get info on each base endpoint, when providing no further args or params (if necessary)

# ToDo: adapt README


@app.route("/")
def index():
    return jsonify(
        available_endpoints=[
            "/preprocessing",
            "/embedding",
            "/clustering",
            "/graph",
            "/pipeline",
            "/processes",
            "/status",
        ]
    )


@app.route("/preprocessing", methods=["GET", "POST"])
def data_preprocessing():
    app.logger.info("=== Preprocessing started ===")
    if request.method == "POST" and len(request.files) > 0 and "data" in request.files:
        pre_proc = PreprocessingUtil(app, FILE_STORAGE_TMP)

        process_name = read_config(app, pre_proc, StepsName.DATA)

        app.logger.info("Reading labels ...")
        pre_proc.read_labels(request.files.get("labels", None))
        _labels_str = (
            list(pre_proc.labels.values()) if pre_proc.labels is not None else []
        )
        app.logger.info(f"Gathered the following labels:\n\t{_labels_str}")

        app.logger.info("Reading data ...")
        pre_proc.read_data(request.files.get("data", None))
        app.logger.info(f"Counted {len(pre_proc.data)} item ins zip file.")

        app.logger.info(f"Start preprocessing '{process_name}' ...")
        return data_get_statistics(
            pre_proc.start_process(
                pre_proc.process_name,
                data_functions.DataProcessingFactory,
                running_processes,
            ),
        )

    elif request.method == "GET":
        jsonify(
            "'/preprocessing' endpoint needs some form of data. Please consult the README/manuals on how to provide it."
        )

    elif len(request.files) <= 0 or "data" not in request.files:
        app.logger.error(
            "There were no files at all or no data files POSTed."
            " At least a zip folder with the text data is necessary!\n"
            " It is also necessary to conform to the naming convention!\n"
            '\t\ti.e.: curl -X POST -F data=@"#SOME/PATH/TO/FILE.zip"'
        )
    return jsonify("Nothing to do.")


@app.route("/preprocessing/<path_arg>", methods=["GET"])
def data_preprocessing_with_arg(path_arg):
    process = request.args.get("process", "default").lower()
    path_arg = path_arg.lower()

    _path_args = ["statistics", "noun_chunks"]
    if path_arg in _path_args:
        data_obj = FactoryLoader.load_data(
            str(pathlib.Path(FILE_STORAGE_TMP, process).resolve()), process
        )
    else:
        return jsonify(
            error=f"No such path argument '{path_arg}' for 'preprocessing' endpoint.",
            possible_path_args=[f"/{p}" for p in _path_args],
        )

    if path_arg == "statistics":
        return data_get_statistics(data_obj)
    elif path_arg == "noun_chunks":
        return jsonify(noun_chunks=data_obj.data_chunk_sets)


@app.route("/embedding", methods=["POST", "GET"])
def phrase_embedding():
    app.logger.info("=== Phrase embedding started ===")
    if request.method in ["POST", "GET"]:
        phra_emb = PhraseEmbeddingUtil(app, FILE_STORAGE_TMP)

        process_name = read_config(app, phra_emb, StepsName.EMBEDDING)

        app.logger.info(f"Start phrase embedding '{process_name}' ...")
        try:
            return embedding_get_statistics(
                phra_emb.start_process(
                    process_name,
                    embedding_functions.SentenceEmbeddingsFactory,
                    running_processes,
                )
            )
        except FileNotFoundError:
            return jsonify(
                f"There is no processed data for the '{process_name}' process to be embedded."
            )
    return jsonify("Nothing to do.")


@app.route("/embedding/<path_arg>", methods=["GET"])
def phrase_embedding_with_arg(path_arg):
    process = request.args.get("process", "default").lower()
    path_arg = path_arg.lower()

    _path_args = ["statistics"]
    if path_arg in _path_args:
        emb_obj = embedding_functions.SentenceEmbeddingsFactory.load(
            pathlib.Path(
                pathlib.Path(FILE_STORAGE_TMP)
                / pathlib.Path(process)
                / f"{process}_data.pickle"
            ),
            pathlib.Path(
                pathlib.Path(FILE_STORAGE_TMP)
                / pathlib.Path(process)
                / f"{process}_embedding.pickle"
            ),
        )
    else:
        return jsonify(
            error=f"No such path argument '{path_arg}' for 'embedding' endpoint.",
            possible_path_args=[f"/{p}" for p in _path_args],
        )

    if path_arg == "statistics":
        return embedding_get_statistics(emb_obj)


@app.route("/clustering", methods=["POST", "GET"])
def phrase_clustering():
    app.logger.info("=== Phrase clustering started ===")
    if request.method in ["POST", "GET"]:
        saved_config = request.args.get("config", False)
        if not saved_config:
            phra_clus = ClusteringUtil(app, FILE_STORAGE_TMP)

            process_name = read_config(app, phra_clus, StepsName.CLUSTERING)

            app.logger.info(f"Start phrase clustering '{process_name}' ...")
            try:
                _cluster_obj = phra_clus.start_process(
                    process_name,
                    cluster_functions.PhraseClusterFactory,
                    running_processes,
                )
                return clustering_get_concepts(
                    embedding_functions.show_top_k_for_concepts(
                        cluster_obj=_cluster_obj.concept_cluster,
                        embedding_object=_cluster_obj.sentence_embeddings,
                        yield_concepts=True,
                    )
                )
            except FileNotFoundError:
                return jsonify(
                    f"There is no embedded data for the '{process_name}' process to be clustered."
                )
        else:
            return jsonify(saved_config)
    return jsonify("Nothing to do.")


@app.route("/clustering/<path_arg>", methods=["GET"])
def clustering_with_arg(path_arg):
    process = request.args.get("process", "default").lower()
    top_k = int(request.args.get("top_k", 15))
    distance = float(request.args.get("distance", 0.6))
    path_arg = path_arg.lower()

    _path_args = ["concepts"]
    if path_arg in _path_args:
        cluster_obj = cluster_functions.PhraseClusterFactory.load(
            pathlib.Path(
                pathlib.Path(FILE_STORAGE_TMP)
                / pathlib.Path(process)
                / f"{process}_clustering.pickle"
            ),
        )
    else:
        return jsonify(
            error=f"No such path argument '{path_arg}' for 'clustering' endpoint.",
            possible_path_args=[f"/{p}" for p in _path_args],
        )

    if path_arg == "concepts":
        emb_obj = util_functions.load_pickle(
            pathlib.Path(pathlib.Path(FILE_STORAGE_TMP) / f"{process}_embedding.pickle")
        )
        _cluster_gen = embedding_functions.show_top_k_for_concepts(
            cluster_obj=cluster_obj.concept_cluster,
            embedding_object=emb_obj,
            yield_concepts=True,
            top_k=top_k,
            distance=distance,
        )
        return clustering_get_concepts(_cluster_gen)


@app.route("/graph", methods=["GET"])
def graph_base_endpoint():
    return (
        jsonify(
            path_parameter=[
                {
                    "statistics": {
                        "query_parameters": [
                            get_query_param_help_text("process"),
                        ]
                    }
                },
                {
                    "creation": {
                        "query_parameters": [
                            get_query_param_help_text("process"),
                            get_query_param_help_text("exclusion_ids"),
                        ]
                    }
                },
                {
                    "#GRAPH_ID": {
                        "query_parameters": [
                            get_query_param_help_text("process"),
                            get_query_param_help_text("draw"),
                        ]
                    }
                },
                {
                    "document": {
                        {
                            "add": {
                                get_query_param_help_text("process"),
                            }
                        }
                    }
                },
            ],
        ),
        HTTPResponses.NOT_FOUND,
    )


@app.route("/graph/document/<path_arg>", methods=["POST"])
def graph_document(path_arg):
    # ToDo: resolve not implemented exceptions
    """
    request.json = {
        vectorstore_server: Optional[dict]
        document_server: Optional[dict]
        language: str
        documents: [
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
    }
    """
    process = request.args.get("process", "default").lower()
    if request.headers.get("Content-Type") == "application/json":
        content_json = parse_document_adding_json(request.get_json())
        if content_json is None:
            return jsonify(f"Could not parse json provided in request.")
    else:
        raise NotImplementedError("Only json request body is supported.")
    if path_arg.lower() == "add":
        # if content_json.id is None:
        #     return jsonify(f"No '_id' or 'id' key in content json: '{request.get_json().keys()}'")
        _path_base = pathlib.Path(FILE_STORAGE_TMP) / pathlib.Path(process)
        _data_proc = FactoryLoader.load_data(str(_path_base.resolve()), process)
        _emb_proc = FactoryLoader.load_embedding(
            str(_path_base.resolve()),
            process,
            _data_proc,
            (
                None
                if content_json.vectorstore_server is None
                else content_json.vectorstore_server
            ),
        )
        if content_json.vectorstore_server is None and _emb_proc.source is None:
            raise NotImplementedError(
                "Only adding documents with a setup vectorstore server is supported; no vectorstore configured."
            )
        if _emb_proc.source is None:
            _emb_proc.source = content_json.vectorstore_server
        if len(content_json.documents) > 0 and isinstance(
            content_json.documents[0], dict
        ):
            _response = add_documents_to_concept_graphs(
                content_json.documents,
                data_processing=_data_proc,
                embedding_processing=_emb_proc,
            )
        else:
            raise NotImplementedError(
                "Right now only processing of documents as json is supported."
            )
    elif path_arg.lower() == "delete":
        raise NotImplementedError("'Delete' not implemented.")
    else:
        return jsonify(error=f"No such path argument '{path_arg}' supported.")
    return graph_base_endpoint()


@app.route("/graph/<path_arg>", methods=["POST", "GET"])
def graph_creation_with_arg(path_arg):
    process = request.args.get("process", "default").lower()
    draw = get_bool_expression(request.args.get("draw", False))
    path_arg = path_arg.lower()

    _path_args = ["statistics", "creation"]
    if path_arg in _path_args:
        try:
            if path_arg == "statistics":
                _result = graph_get_statistics(app, process, FILE_STORAGE_TMP)
                _http_response = HTTPResponses.OK
                if "error" in _result:
                    _http_response = HTTPResponses.INTERNAL_SERVER_ERROR
                return jsonify(name=process, **_result), _http_response
            elif path_arg == "creation":
                return graph_create(app, FILE_STORAGE_TMP)
        except FileNotFoundError:
            return Response(
                f"There is no graph data present for '{process}'.\n",
                status=int(HTTPResponses.NOT_FOUND),
            )
    elif path_arg.isdigit():
        graph_nr = int(path_arg)
        return graph_get_specific(process, graph_nr, path=FILE_STORAGE_TMP, draw=draw)
    else:
        return Response(
            f"No such path argument '{path_arg}' for 'graph' endpoint.\n"
            f"Possible path arguments are: {', '.join([p for p in _path_args] + ['#ANY_INTEGER'])}\n",
            status=int(HTTPResponses.BAD_REQUEST),
        )


@app.route("/pipeline", methods=["POST"])
def complete_pipeline():
    DEFAULT_VECTOR_STORE = {"url": "http://localhost", "port": 8882}
    data = request.files.get("data", False)
    data_upload = False
    document_server_config = request.files.get("document_server_config", False)
    vector_store_config = request.files.get("vectorstore_server_config", False)
    replace_keys = None
    label_getter = None
    labels = None

    content_type = request.headers.get("Content-Type")
    content_type_json = False
    config_object_json = None
    if content_type == "application/json":
        content_type_json = True
        config_object_json: Optional[pipeline_json_config] = parse_config_json(
            request.json
        )

    query_params = get_pipeline_query_params(
        app, request, running_processes, config_object_json
    )
    if isinstance(query_params, tuple) and isinstance(query_params[0], flask.Response):
        return query_params

    if content_type_json:
        _document_server = config_object_json.document_server
        vector_store_config = (
            config_object_json.vectorstore_server
            if config_object_json.vectorstore_server is not None
            else DEFAULT_VECTOR_STORE
        )
        if _document_server is not None:
            replace_keys = _document_server.get("replace_keys", {"text": "content"})
            label_getter = _document_server.get("label_key", None)
            document_server_config = _document_server.copy()
        else:
            return jsonify(
                name=(
                    config_object_json.name
                    if config_object_json.name is not None
                    else query_params.process_name
                ),
                error="No configuration entry for documents on a server provided.",
            ), int(HTTPResponses.BAD_REQUEST)
        _data_config = config_object_json.data
        _embedding_config = config_object_json.embedding
        _clustering_config = config_object_json.clustering
        _graph_config = config_object_json.graph

    else:
        if vector_store_config:
            if isinstance(vector_store_config, FileStorage):
                vector_store_config = yaml.safe_load(vector_store_config.stream)
            else:
                vector_store_config = DEFAULT_VECTOR_STORE
        else:
            vector_store_config = DEFAULT_VECTOR_STORE
        if not data and not document_server_config:
            return jsonify(
                name=query_params.process_name,
                error="Neither data provided for upload with 'data' key nor a config file for documents on a server",
            ), int(HTTPResponses.BAD_REQUEST)
        elif data and not document_server_config:
            _tmp_data = pathlib.Path(
                pathlib.Path(FILE_STORAGE_TMP)
                / pathlib.Path(".tmp_streams")
                / data.filename
            )
            _tmp_data.parent.mkdir(parents=True, exist_ok=True)
            data.save(_tmp_data)
            data = _tmp_data
            data_upload = True

        labels = request.files.get("labels", None)
        if labels is not None:
            _tmp_labels = pathlib.Path(
                pathlib.Path(FILE_STORAGE_TMP)
                / pathlib.Path(".tmp_streams")
                / labels.filename
            )
            _tmp_labels.parent.mkdir(parents=True, exist_ok=True)
            labels.save(_tmp_labels)
            labels = _tmp_labels

        _data_config = request.files.get(f"{StepsName.DATA}_config", None)
        _embedding_config = request.files.get(f"{StepsName.EMBEDDING}_config", None)
        _clustering_config = request.files.get(f"{StepsName.CLUSTERING}_config", None)
        _graph_config = request.files.get(f"{StepsName.GRAPH}_config", None)

    if vector_store_config is not None:
        _url = vector_store_config.pop("url", "http://localhost")
        _port = str(vector_store_config.pop("port", 8882))
        vector_store_config["client_url"] = f"{_url}:{_port}"
        if not marqo_external_utils.MarqoEmbeddingStore.is_accessible(
            vector_store_config.copy()
        ):
            logging.warning(
                f"Vector store doesn't seem to be accessible under '{vector_store_config['client_url']}'."
                f" Using 'pickle' storage."
            )
            vector_store_config = None
    if not data_upload:
        ds_base_config = get_data_server_config(document_server_config, app)
        if not check_data_server(ds_base_config):
            return jsonify(
                name=query_params.process_name,
                error=f"There is no data server at the specified location ({ds_base_config}) or it contains no data.",
            ), int(HTTPResponses.NOT_FOUND)
        # ToDo: don't know if I want this, but 'get_documents_from_es_server' can now filter documents
        data = get_documents_from_es_server(
            url=ds_base_config["url"],
            port=ds_base_config["port"],
            index=ds_base_config["index"],
            size=int(ds_base_config["size"]),
            other_id=ds_base_config["other_id"],
        )
        replace_keys = ds_base_config.get("replace_keys", {"text": "content"})
        label_getter = ds_base_config.get("label_key", None)

    processes = [
        (
            StepsName.DATA,
            PreprocessingUtil,
            _data_config,
            data_functions.DataProcessingFactory,
        ),
        (
            StepsName.EMBEDDING,
            PhraseEmbeddingUtil,
            _embedding_config,
            embedding_functions.SentenceEmbeddingsFactory,
        ),
        (
            StepsName.CLUSTERING,
            ClusteringUtil,
            _clustering_config,
            cluster_functions.PhraseClusterFactory,
        ),
        (
            StepsName.GRAPH,
            GraphCreationUtil,
            _graph_config,
            cluster_functions.WordEmbeddingClustering,
        ),
        (
            StepsName.INTEGRATION,
            ConceptGraphIntegrationUtil,
            {},
            integration_functions.ConceptGraphIntegrationFactory,
        ),
    ]
    processes_threading = []
    current_active_pipeline_objects[query_params.process_name] = {
        k: None for k in StepsName.ALL
    }
    _prev_step_present = True
    _last_step = StepsName.INTEGRATION
    for _name, _proc, _conf, _fact in processes:
        process_obj: BaseUtil = _proc(app=app, file_storage=FILE_STORAGE_TMP)
        add_status_to_running_process(
            query_params.process_name, _name, ProcessStatus.STARTED, running_processes
        )
        if process_obj.has_process(query_params.process_name):
            if (
                _name in query_params.omitted_pipeline_steps
                or query_params.skip_present
            ):
                logging.info(
                    f"Skipping {_name} because "
                    f"{'omitted' if _name in query_params.omitted_pipeline_steps else 'skip_present'}."
                )
                add_status_to_running_process(
                    query_params.process_name,
                    _name,
                    ProcessStatus.FINISHED,
                    running_processes,
                )
                if _prev_step_present:
                    current_active_pipeline_objects[query_params.process_name][
                        _name
                    ] = FactoryLoader.load(
                        step=_name,
                        path=str(
                            pathlib.Path(
                                FILE_STORAGE_TMP, query_params.process_name
                            ).resolve()
                        ),
                        process=query_params.process_name,
                        data_obj=current_active_pipeline_objects[
                            query_params.process_name
                        ].get(StepsName.DATA, None),
                        emb_obj=current_active_pipeline_objects[
                            query_params.process_name
                        ].get(StepsName.EMBEDDING, None),
                        vector_store=vector_store_config,
                    )
                continue
            else:
                process_obj.delete_process(query_params.process_name)
                _last_step = _name
        else:
            _last_step = _name
            _prev_step_present = False

        read_config(
            app=app,
            processor=process_obj,
            process_type=_name,
            process_name=query_params.process_name,
            config=_conf,
            language=query_params.language,
            mode="json" if content_type_json else "yaml",
        )

        if _name == StepsName.DATA:
            process_obj: PreprocessingUtil
            process_obj.read_labels(labels if label_getter is None else label_getter)
            process_obj.read_data(
                data, replace_keys=replace_keys, label_getter=label_getter
            )
        if _name == StepsName.EMBEDDING:
            process_obj.storage_method = (
                (
                    "pickle",
                    None,
                )
                if vector_store_config is None
                else (
                    (
                        "vectorstore",
                        vector_store_config,
                    )
                    if process_obj.storage_method == "vectorstore"
                    else (
                        "pickle",
                        None,
                    )
                )
            )

        processes_threading.append(
            (
                process_obj,
                _fact,
                _name,
            )
        )

    pipeline_thread = StoppableThread(
        target_args=(
            app,
            processes_threading,
            query_params.process_name,
            running_processes,
            pipeline_threads_store,
            current_active_pipeline_objects,
            _last_step,
        ),
        group=None,
        target=start_processes,
        name=None,
    )

    pipeline_threads_store[query_params.process_name] = pipeline_thread
    start_thread(
        app, query_params.process_name, pipeline_thread, pipeline_threads_store
    )

    if query_params.return_statistics:
        pipeline_thread.join()
        _graph_stats_dict = graph_get_statistics(
            app=app, data=query_params.process_name, path=FILE_STORAGE_TMP
        )
        return (
            jsonify(name=query_params.process_name, **_graph_stats_dict),
            (
                int(HTTPResponses.OK)
                if "error" not in _graph_stats_dict
                else int(HTTPResponses.INTERNAL_SERVER_ERROR)
            ),
        )
    else:
        return (
            jsonify(
                name=query_params.process_name,
                status=running_processes.get(
                    query_params.process_name, {"status": []}
                ).get("status"),
            ),
            int(HTTPResponses.ACCEPTED),
        )


@app.route("/processes/<process_id>/stop", methods=["GET"])
def stop_pipeline(process_id):
    if request.method == "GET":
        hard_stop = request.args.get("hard_stop", False)
        process_id = process_id.lower()
        return stop_thread(
            app=app,
            process_name=process_id,
            threading_store=pipeline_threads_store,
            process_tracker=running_processes,
            hard_stop=hard_stop,
        )
    return jsonify(f"Method not supported: {request.method}")


@app.route("/pipeline/configuration", methods=["GET"])
def get_pipeline_default_configuration():
    if request.method == "GET":
        is_default_conf = get_bool_expression(request.args.get("default", True))
        process = request.args.get("process", "default")
        language = PipelineLanguage.language_from_string(
            request.args.get("language", "en")
        )
        if is_default_conf:
            default_conf = pathlib.Path(f"./conf/pipeline-config_{language}.json")
            if default_conf.exists() and default_conf.is_file():
                try:
                    return jsonify(**json.load(default_conf.open("rb"))), int(
                        HTTPResponses.OK
                    )
                except Exception as e:
                    logging.error(e)
            return jsonify(message="Couldn't find/read default configuration."), int(
                HTTPResponses.NOT_FOUND
            )
        else:
            logging.info(f"Returning configuration for '{process}' pipeline.")
            try:
                _config = load_configs(
                    app=app, process_name=process, path_to_configs=FILE_STORAGE_TMP
                )
                return (
                    jsonify(
                        name=process,
                        language=_config.get("language", "en"),
                        config=_config.get("config", {}),
                    ),
                    int(HTTPResponses.OK),
                )
            except Exception as e:
                logging.error(e)
            return jsonify(
                message=f"Couldn't find/read configuration for '{process}'."
            ), int(HTTPResponses.NOT_FOUND)
    else:
        return HTTPResponses.BAD_REQUEST


@app.route("/processes/<process_id>/delete", methods=["DELETE"])
def delete_process(process_id):
    hard_stop = request.args.get("hard_stop", False)
    process_id = process_id.lower()
    if process_id not in running_processes:
        return Response(
            f"There is no such process '{process_id}'.\n",
            status=int(HTTPResponses.NOT_FOUND),
        )
    to_stop = None
    if any(
        [
            step.get("status") in [ProcessStatus.RUNNING, ProcessStatus.STARTED]
            for step in running_processes.get(process_id).get("status", [])
        ]
    ):
        to_stop: Optional[StoppableThread]
        if to_stop := pipeline_threads_store.get(process_id, None):
            _stop = stop_thread(
                app=app,
                process_name=process_id,
                threading_store=pipeline_threads_store,
                process_tracker=running_processes,
                hard_stop=hard_stop,
            )
    _delete_thread = StoppableThread(
        target_args=(app, f_storage, process_id, running_processes, to_stop),
        group=None,
        target=delete_pipeline,
        name=None,
    )
    _delete_thread.start()
    return Response(
        f"Process '{process_id}' set to be deleted.", status=HTTPResponses.OK
    )


@app.route("/processes", methods=["GET"])
def get_all_processes_api():
    if len(running_processes) > 0:
        return jsonify(processes=[p for p in running_processes.values()])
    else:
        return jsonify("No saved processes."), int(HTTPResponses.NOT_FOUND)


@app.route("/status", methods=["GET"])
def get_status_of():
    _process = request.args.get("process", "default").lower()
    if _process is not None:
        _response = running_processes.get(_process, None)
        if _response is not None:
            return jsonify(_response), int(HTTPResponses.OK)
    return jsonify(
        name=_process, error=f"No such (running) process: '{_process}'"
    ), int(HTTPResponses.NOT_FOUND)


@app.route("/status/document-server", methods=["POST", "GET"])
def get_data_server():
    # if request.method == "GET" and request.args.get("port", False):
    #
    #     return jsonify()
    if request.method == "POST":
        document_server_config = request.files.get("document_server_config", False)
        if not document_server_config:
            return jsonify(
                name="document server check",
                status=f"No document server config file provided",
            ), int(HTTPResponses.BAD_REQUEST)
        base_config = get_data_server_config(document_server_config, app)
        if not check_data_server(
            url=base_config["url"], port=base_config["port"], index=base_config["index"]
        ):
            return (
                jsonify(
                    f"There is no data server at the specified location ({base_config}) or it contains no data."
                ),
                int(HTTPResponses.NOT_FOUND),
            )
        return (
            jsonify(
                f"Data server reachable under: '{base_config['url']}:{base_config['port']}/{base_config['index']}'"
            ),
            int(HTTPResponses.OK),
        )
    elif request.method == "POST":
        return jsonify("Method 'POST' not implemented.")
    else:
        return jsonify(f"Method not supported: '{request.method}'.")


if __name__ in ["__main__"]:
    app.run(host="0.0.0.0", port=9010)
