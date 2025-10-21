import json
import logging
from operator import itemgetter

from main_methods import *
from main_utils import StoppableThread, ActiveRAG
from src.rag.embedding_stores.AbstractEmbeddingStore import ChunkEmbeddingStore
from src.rag.marqo_rag_utils import extract_text_from_highlights
from src.rag.rag import RAG

sys.path.insert(0, "src")
from src import integration_functions
from src import marqo_external_utils


def unspecified_server_error():
    return (
        jsonify(error="Something went wrong; please consult the logs."),
        HTTPResponses.INTERNAL_SERVER_ERROR,
    )


def path_arg_error(parent_endpoint: str, path_arg: str, possible_path_args: list[str]):
    return (
        jsonify(
            error=f"No such path argument '{path_arg}' for '{parent_endpoint}' endpoint.",
            possible_path_args=[f"/{p}" for p in possible_path_args],
        ),
        HTTPResponses.BAD_REQUEST,
    )


def setup(
    static_folder: str = "api",
    static_url_path: str = "",
    file_storage_dir: str = "tmp",
    logging_setup_tuples: Optional[list[tuple]] = None,
) -> PersistentObjects:
    import logging

    if logging_setup_tuples is None:
        logging_setup_tuples = [
            (
                "werkzeug",
                logging.WARN,
            ),
            ("marqo", logging.WARN),
        ]
    _app = flask.Flask(
        __name__, static_folder=static_folder, static_url_path=static_url_path
    )
    for log_setup in logging_setup_tuples:
        _logger = logging.getLogger(log_setup[0])
        _logger.setLevel(log_setup[1])
    root_logger = logging.getLogger()
    root_logger.propagate = False
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(flask.logging.default_handler)

    _po = PersistentObjects(
        app=_app,
        running_processes={},
        pipeline_threads_store={},
        current_active_pipeline_objects={},
        file_storage_dir=pathlib.Path(file_storage_dir),
        active_rag=None,
    )
    if not _po.file_storage_dir.exists():
        _po.file_storage_dir.mkdir()
    populate_running_processes(_po.app, _po.file_storage_dir, _po.running_processes)
    return _po


main_objects = setup(static_folder="api", static_url_path="", file_storage_dir="tmp")


# ToDo: file with stopwords will be POSTed: #filter_stop: Optional[list] = None,

# ToDo: I downscale the embeddings twice... (that snuck in somehow); once in SentenceEmbeddings via create(down_scale_algorithm)
# ToDo: and once PhraseCluster via create(down_scale_algorithm). I can't remember why I added this to SentenceEmbeddings later on...
# ToDo: but I should make sure, that there is a check somewhere that the down scaling is not applied twice!

# ToDo: make sure that no arguments can be supplied via config that won't work

# ToDo: adapt README

# ToDo: main_objects.current_active_pipeline_objects stores all serialized step objects per process name; so this object
# ToDo: could become rather large (if a lot of processes get started); maybe have a config value that stores the last N
# ToDo: processes and have them have time stamps and delete accordingly


@main_objects.app.route("/", methods=["GET"])
def index():
    return openapi()


@main_objects.app.route("/openapi", methods=["GET"])
def openapi():
    return main_objects.app.send_static_file("index.html")


@main_objects.app.route("/preprocessing/<path_arg>", methods=["GET"])
def data_preprocessing_with_arg(path_arg):
    process = string_conformity(request.args.get("process", "default"))
    path_arg = path_arg.lower()

    _path_args = ["statistics", "noun_chunks"]
    if path_arg in _path_args:
        data_obj = FactoryLoader.with_active_objects(
            str(pathlib.Path(main_objects.file_storage_dir, process).resolve()),
            process,
            main_objects.current_active_pipeline_objects,
            StepsName.DATA,
        )
        if data_obj is None:
            return unspecified_server_error()
        if path_arg == "statistics":
            return data_get_statistics(data_obj), HTTPResponses.OK
        elif path_arg == "noun_chunks":
            return jsonify(noun_chunks=data_obj.data_chunk_sets), HTTPResponses.OK
    else:
        return path_arg_error("preprocessing", path_arg, _path_args)


@main_objects.app.route("/embedding/<path_arg>", methods=["GET"])
def phrase_embedding_with_arg(path_arg):
    process = string_conformity(request.args.get("process", "default"))
    path_arg = path_arg.lower()

    _path_args = ["statistics"]
    if path_arg in _path_args:
        emb_obj = FactoryLoader.with_active_objects(
            str(pathlib.Path(main_objects.file_storage_dir, process).resolve()),
            process,
            main_objects.current_active_pipeline_objects,
            StepsName.EMBEDDING,
        )
        if emb_obj is None:
            return unspecified_server_error()
        if path_arg == "statistics":
            return embedding_get_statistics(emb_obj)
    else:
        return path_arg_error("embedding", path_arg, _path_args)


@main_objects.app.route("/clustering/<path_arg>", methods=["GET"])
def clustering_with_arg(path_arg):
    process = string_conformity(request.args.get("process", "default"))
    top_k = int(request.args.get("top_k", 15))
    distance = float(request.args.get("distance", 0.6))
    path_arg = path_arg.lower()

    _path_args = ["concepts"]
    if path_arg in _path_args:
        cluster_obj = FactoryLoader.with_active_objects(
            str(pathlib.Path(main_objects.file_storage_dir, process).resolve()),
            process,
            main_objects.current_active_pipeline_objects,
            StepsName.CLUSTERING,
        )
        if cluster_obj is None:
            return unspecified_server_error()
        if path_arg == "concepts":
            emb_obj = FactoryLoader.with_active_objects(
                str(pathlib.Path(main_objects.file_storage_dir, process).resolve()),
                process,
                main_objects.current_active_pipeline_objects,
                StepsName.EMBEDDING,
            )
            _cluster_gen = embedding_functions.show_top_k_for_concepts(
                cluster_obj=cluster_obj.concept_cluster,
                embedding_object=emb_obj,
                yield_concepts=True,
                top_k=top_k,
                distance=distance,
            )
            return clustering_get_concepts(_cluster_gen)
    else:
        return path_arg_error("clustering", path_arg, _path_args)


@main_objects.app.route("/graph/<path_arg>", methods=["POST", "GET"])
def graph_with_arg(path_arg):
    process = string_conformity(request.args.get("process", "default"))
    draw = get_bool_expression(request.args.get("draw", False))
    path_arg = path_arg.lower()
    graph_list = FactoryLoader.with_active_objects(
        str(pathlib.Path(main_objects.file_storage_dir, process).resolve()),
        process,
        main_objects.current_active_pipeline_objects,
        StepsName.GRAPH,
    )

    _path_args = ["statistics"]
    if path_arg in _path_args:
        if graph_list is None:
            return unspecified_server_error()
        try:
            if path_arg == "statistics":
                _result = graph_get_statistics(
                    main_objects.app, graph_list, main_objects.file_storage_dir
                )
                _http_response = HTTPResponses.OK
                if "error" in _result:
                    _http_response = HTTPResponses.INTERNAL_SERVER_ERROR
                return jsonify(name=process, **_result), _http_response
        except FileNotFoundError:
            return Response(
                f"There is no graph data present for '{process}'.\n",
                status=int(HTTPResponses.NOT_FOUND),
            )
    elif path_arg.isdigit():
        graph_nr = int(path_arg)
        return graph_get_specific(
            graph_list, graph_nr, path=main_objects.file_storage_dir, draw=draw
        )
    else:
        return path_arg_error("graph", path_arg, _path_args + ["#ANY_INTEGER"])


@main_objects.app.route("/graph/document/<path_arg>", methods=["POST"])
def graph_document(path_arg):
    # ToDo: add getting documents from document_server
    # ToDo: resolve not implemented exceptions
    process = string_conformity(request.args.get("process", "default"))
    if request.headers.get("Content-Type") == "application/json":
        content_json = parse_document_adding_json(request.get_json())
        if content_json is None:
            return (
                jsonify(error=f"Could not parse json provided in request."),
                HTTPResponses.BAD_REQUEST,
            )
    else:
        return (
            jsonify(error="Only json request body is supported."),
            HTTPResponses.NOT_IMPLEMENTED,
        )

    if path_arg.lower() == "add":
        _data_proc = main_objects.current_active_pipeline_objects.get(process, {}).get(
            StepsName.DATA, None
        )
        _emb_proc = main_objects.current_active_pipeline_objects.get(process, {}).get(
            StepsName.EMBEDDING, None
        )
        _graph_proc = main_objects.current_active_pipeline_objects.get(process, {}).get(
            StepsName.GRAPH, None
        )
        _path_base = main_objects.file_storage_dir / process
        document_adding_thread = StoppableThread(
            target_args=(content_json,),
            target_kwargs={
                "data_processing": _data_proc,
                "embedding_processing": _emb_proc,
                "graph_processing": _graph_proc,
                "storage_path": _path_base,
                "process_name": process,
            },
            group=None,
            target=add_documents_to_concept_graphs,
            name=None,
        )
        main_objects.pipeline_threads_store[f"document_addition_{process}"] = (
            document_adding_thread
        )
        start_thread(
            main_objects.app,
            f"document_addition_{process}",
            document_adding_thread,
            None,
        )
        return (
            jsonify(f"Started thread for adding documents for process {process}."),
            HTTPResponses.OK,
        )
    elif path_arg.lower() == "delete":
        return jsonify(error="'Delete' not implemented."), HTTPResponses.NOT_IMPLEMENTED
    else:
        return path_arg_error("graph/document", path_arg, ["add", "delete"])


@main_objects.app.route("/graph/document/add/status", methods=["GET"])
def graph_document_status():
    process = string_conformity(request.args.get("process", "default"))
    _id = f"document_addition_{process}"
    if _id not in main_objects.pipeline_threads_store:
        return (
            jsonify(
                error=f"No document addition thread (running or completed) for '{process}' found."
            ),
            HTTPResponses.NOT_FOUND,
        )
    else:
        if return_value := main_objects.pipeline_threads_store.get(_id).return_value:
            return jsonify(return_value[0]), return_value[1]
        return (
            jsonify(
                f"Document addition thread for '{process}' seems to be still running."
            ),
            HTTPResponses.ACCEPTED,
        )


@main_objects.app.route("/pipeline", methods=["POST"])
def complete_pipeline():
    query_params = pipeline_query_params(
        process_name="not set",
        language="en",
        skip_present=True,
        omitted_pipeline_steps=[],
        return_statistics=False,
    )
    try:
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
            config_object_json: Optional[pipeline_json_config] = (
                parse_pipeline_config_json(request.json)
            )

        query_params = get_pipeline_query_params(
            main_objects.app,
            request,
            main_objects.running_processes,
            config_object_json,
        )
        if isinstance(query_params, tuple) and isinstance(
            query_params[0], flask.Response
        ):
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
                    main_objects.file_storage_dir / ".tmp_streams" / data.filename
                )
                _tmp_data.parent.mkdir(parents=True, exist_ok=True)
                data.save(_tmp_data)
                data = _tmp_data
                data_upload = True

            labels = request.files.get("labels", None)
            if labels is not None:
                _tmp_labels = pathlib.Path(
                    main_objects.file_storage_dir / ".tmp_streams" / labels.filename
                )
                _tmp_labels.parent.mkdir(parents=True, exist_ok=True)
                labels.save(_tmp_labels)
                labels = _tmp_labels

            _data_config = request.files.get(f"{StepsName.DATA}_config", None)
            _embedding_config = request.files.get(f"{StepsName.EMBEDDING}_config", None)
            _clustering_config = request.files.get(
                f"{StepsName.CLUSTERING}_config", None
            )
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
            ds_base_config = get_data_server_config(
                document_server_config, main_objects.app
            )
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
        ]
        if vector_store_config is not None:
            processes.append(
                (
                    StepsName.INTEGRATION,
                    ConceptGraphIntegrationUtil,
                    {},
                    integration_functions.ConceptGraphIntegrationFactory,
                )
            )
        processes_threading = []
        main_objects.current_active_pipeline_objects[query_params.process_name] = {
            k: None for k in StepsName.ALL
        }
        _prev_step_present = True
        _last_step = (
            StepsName.INTEGRATION
            if vector_store_config is not None
            else StepsName.GRAPH
        )
        for _name, _proc, _conf, _fact in processes:
            process_obj: BaseUtil = _proc(
                app=main_objects.app, file_storage=main_objects.file_storage_dir
            )
            add_status_to_running_process(
                query_params.process_name,
                _name,
                ProcessStatus.STARTED,
                main_objects.running_processes,
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
                        main_objects.running_processes,
                    )
                    if _prev_step_present:
                        main_objects.current_active_pipeline_objects[
                            query_params.process_name
                        ][_name] = FactoryLoader.load(
                            step=_name,
                            path=str(
                                pathlib.Path(
                                    main_objects.file_storage_dir,
                                    query_params.process_name,
                                ).resolve()
                            ),
                            process=query_params.process_name,
                            data_obj=main_objects.current_active_pipeline_objects[
                                query_params.process_name
                            ].get(StepsName.DATA, None),
                            emb_obj=main_objects.current_active_pipeline_objects[
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
                app=main_objects.app,
                processor=process_obj,
                process_type=_name,
                process_name=query_params.process_name,
                config=_conf,
                language=query_params.language,
                mode="json" if content_type_json else "yaml",
            )

            if _name == StepsName.DATA:
                process_obj: PreprocessingUtil
                process_obj.read_labels(
                    labels if label_getter is None else label_getter
                )
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
            if _name == StepsName.INTEGRATION:
                process_obj.config["check_for_reasonable_result"] = True

            processes_threading.append(
                (
                    process_obj,
                    _fact,
                    _name,
                )
            )

        pipeline_thread = StoppableThread(
            target_args=(
                main_objects.app,
                processes_threading,
                query_params.process_name,
                main_objects.running_processes,
                main_objects.pipeline_threads_store,
                main_objects.current_active_pipeline_objects,
                _last_step,
            ),
            group=None,
            target=start_processes,
            name=None,
        )

        main_objects.pipeline_threads_store[query_params.process_name] = pipeline_thread
        start_thread(
            main_objects.app,
            query_params.process_name,
            pipeline_thread,
            main_objects.pipeline_threads_store,
        )

        if query_params.return_statistics:
            pipeline_thread.join()
            _graph_stats_dict = graph_get_statistics(
                app=main_objects.app,
                data=query_params.process_name,
                path=main_objects.file_storage_dir,
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
                    status=main_objects.running_processes.get(
                        query_params.process_name, {"status": []}
                    ).get("status"),
                ),
                int(HTTPResponses.ACCEPTED),
            )
    except Exception as e:
        return (
            jsonify(
                name=query_params.process_name,
                error=str(e),
            ),
            int(HTTPResponses.INTERNAL_SERVER_ERROR),
        )


@main_objects.app.route("/pipeline/configuration", methods=["GET"])
def get_pipeline_default_configuration():
    if request.method == "GET":
        is_default_conf = get_bool_expression(request.args.get("default", True))
        process = string_conformity(request.args.get("process", "default"))
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
                    app=main_objects.app,
                    process_name=process,
                    path_to_configs=main_objects.file_storage_dir,
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


@main_objects.app.route("/processes", methods=["GET"])
def get_all_processes_api():
    if len(main_objects.running_processes) > 0:
        return jsonify(processes=[p for p in main_objects.running_processes.values()])
    else:
        return jsonify("No saved processes."), int(HTTPResponses.NOT_FOUND)


@main_objects.app.route("/processes/<process_id>/delete", methods=["DELETE"])
def delete_process(process_id):
    hard_stop = request.args.get("hard_stop", False)
    process_id = string_conformity(process_id)
    if process_id not in set(main_objects.running_processes.keys()).union(
        main_objects.current_active_pipeline_objects.keys()
    ):
        return Response(
            f"There is no such process '{process_id}'.\n",
            status=int(HTTPResponses.NOT_FOUND),
        )
    to_stop = None
    if any(
        [
            step.get("status") in [ProcessStatus.RUNNING, ProcessStatus.STARTED]
            for step in main_objects.running_processes.get(process_id).get("status", [])
        ]
    ):
        to_stop: Optional[StoppableThread]
        if to_stop := main_objects.pipeline_threads_store.get(process_id, None):
            _stop = stop_thread(
                app=main_objects.app,
                process_name=process_id,
                threading_store=main_objects.pipeline_threads_store,
                process_tracker=main_objects.running_processes,
                hard_stop=hard_stop,
            )
    _delete_thread = StoppableThread(
        target_args=(
            main_objects.app,
            main_objects.file_storage_dir,
            process_id,
            main_objects.running_processes,
            main_objects.current_active_pipeline_objects,
            to_stop,
        ),
        group=None,
        target=delete_pipeline,
        name=None,
    )
    _delete_thread.start()
    return Response(
        f"Process '{process_id}' set to be deleted.", status=HTTPResponses.OK
    )


@main_objects.app.route("/processes/<process_id>/stop", methods=["GET"])
def stop_pipeline(process_id):
    if request.method == "GET":
        hard_stop = request.args.get("hard_stop", False)
        process_id = string_conformity(process_id)
        return stop_thread(
            app=main_objects.app,
            process_name=process_id,
            threading_store=main_objects.pipeline_threads_store,
            process_tracker=main_objects.running_processes,
            hard_stop=hard_stop,
        )
    return jsonify(f"Method not supported: {request.method}")


@main_objects.app.route("/rag/init", methods=["POST"])
def init_rag():
    if request.method == "POST":
        if request.headers.get("Content-Type") == "application/json":
            process = string_conformity(request.args.get("process", "default"))
            force_init = get_bool_expression(request.args.get("force", "false"))
            _config = parse_rag_config_json(request.json)
            if _config is None:
                return jsonify(f"RAG config couldn't be parsed."), int(
                    HTTPResponses.BAD_REQUEST
                )
            else:
                _rag_thread_id = f"rag_fill_vectorstore_{process}"
                vector_store = initialize_chunk_vectorstore(
                    process, _config.vectorstore_server, force_init=force_init
                )
                _chatter = _config.chatter.pop(
                    "chatter", "src.rag.chatters.BlabladorChatter.BlabladorChatter"
                )
                main_objects.active_rag = ActiveRAG(
                    rag=RAG.with_chatter(
                        api_key=_config.api_key,
                        chatter=_chatter,
                        language=_config.language,
                        **_config.chatter,
                    ).with_prompt(prompt_template_config=_config.prompt_template),
                    vectorstore=vector_store,
                    process=process,
                )
                if (not vector_store.is_filled()) or force_init:
                    _rag_init_thread = StoppableThread(
                        target_args=(process, main_objects),
                        group=None,
                        target=fill_chunk_vectorstore,
                        name=None,
                    )
                    _rag_init_thread.start()
                    sleep(1.0)
                    main_objects.pipeline_threads_store[_rag_thread_id] = (
                        _rag_init_thread
                    )
                    return jsonify("Starting initializing RAG component."), int(
                        HTTPResponses.OK
                    )
                elif _init_thread := main_objects.pipeline_threads_store.get(
                    _rag_thread_id, None
                ):
                    _init_thread: StoppableThread
                    if not _init_thread.return_value:
                        return jsonify(
                            f"There already seems to be an initialization thread running for process {process}. Please wait for it to finish."
                        ), int(HTTPResponses.ACCEPTED)
                main_objects.active_rag.switch_readiness()
            return jsonify("Initialized RAG component."), int(HTTPResponses.OK)
        else:
            return jsonify(
                f"Wrong content type '{request.headers.get('Content-Type')}'; need 'application/json'"
            ), int(HTTPResponses.BAD_REQUEST)
    else:
        return jsonify(f"Method not supported: '{request.method}'."), int(
            HTTPResponses.BAD_REQUEST
        )


@main_objects.app.route("/rag/question", methods=["GET", "POST"])
def rag_question():
    _doc_ids = []
    if (
        request.method == "POST"
        and request.headers.get("Content-Type") == "application/json"
    ):
        if request.json is not None:
            _doc_ids = get_doc_ids(request.json)
    if request.method in ["GET", "POST"]:
        if main_objects.active_rag is None or not main_objects.active_rag.ready:
            return jsonify(
                f"No active and ready rag component found."
                f" You need to initialize it first and wait for it to be ready."
            ), int(HTTPResponses.NOT_FOUND)
        question = request.args.get("q", request.args.get("question", False))
        process = string_conformity(request.args.get("process", "default"))
        language = main_objects.active_rag.rag.language
        if not question:
            return jsonify("No question supplied."), int(HTTPResponses.BAD_REQUEST)
        elif main_objects.active_rag.process != process:
            return (
                jsonify(
                    f"There is no ready and active RAG component for '{process}'."
                    f" Currently active is : '{main_objects.active_rag.process}'; use the 'init' endpoint."
                ),
                int(HTTPResponses.BAD_REQUEST),
            )
        else:
            _documents = list(
                zip(
                    *itemgetter(1, -1)(
                        extract_text_from_highlights(
                            main_objects.active_rag.vectorstore.get_chunks(
                                question,
                                filter_by=(
                                    {"doc_id": _doc_ids} if len(_doc_ids) > 0 else None
                                ),
                            ),
                            token_limit=150,
                            lang=language,
                        )
                    )
                )
            )
            _success, _answer = main_objects.active_rag.rag.with_documents(
                _documents, concat_by="doc_id"
            ).build_and_invoke(question)
            _reference = {
                k: v.metadata for k, v in main_objects.active_rag.rag.documents.items()
            }
            if _success:
                return jsonify(
                    answer=_answer, info=json.dumps(_reference, ensure_ascii=False)
                ), int(HTTPResponses.OK)
            else:
                logging.error(f"[RAG Question]: {_answer}")
                return jsonify(error=_answer), int(HTTPResponses.INTERNAL_SERVER_ERROR)
    else:
        return jsonify(f"Method not supported: '{request.method}'."), int(
            HTTPResponses.BAD_REQUEST
        )


@main_objects.app.route("/status", methods=["GET"])
def get_status_of():
    _process = string_conformity(request.args.get("process", "default"))
    if _process is not None:
        _response = main_objects.running_processes.get(_process, None)
        if _response is not None:
            return jsonify(_response), int(HTTPResponses.OK)
    return jsonify(
        name=_process, error=f"No such (running) process: '{_process}'"
    ), int(HTTPResponses.NOT_FOUND)


@main_objects.app.route("/status/document-server", methods=["POST", "GET"])
def get_data_server():
    # if request.method == "GET" and request.args.get("port", False):
    #
    #     return jsonify()
    if request.method == "POST":
        if request.headers.get("Content-Type") == "application/json":
            document_server_config = request.json
        else:
            document_server_config = request.files.get("document_server_config", False)
        if not document_server_config:
            return jsonify(
                name="document server check",
                status=f"No document server config file provided",
            ), int(HTTPResponses.BAD_REQUEST)
        base_config = get_data_server_config(document_server_config, main_objects.app)
        if not check_data_server(base_config):
            return (
                jsonify(
                    f"There is no data server at the specified location ({base_config}) or its index '{base_config['index']}' contains no data."
                ),
                int(HTTPResponses.NOT_FOUND),
            )
        return (
            jsonify(
                f"Data server reachable under: '{base_config['url']}:{base_config['port']}' with index '{base_config['index']}'"
            ),
            int(HTTPResponses.OK),
        )
    elif request.method == "GET":
        return jsonify("Method 'GET' not yet implemented.")
    else:
        return jsonify(f"Method not supported: '{request.method}'.")


@main_objects.app.route("/status/rag", methods=["GET"])
def get_rag_status():
    _process = string_conformity(request.args.get("process", "default"))
    _has_rag = main_objects.active_rag is not None
    if _process is not None and _has_rag and main_objects.active_rag.process == _process:
        return jsonify(active=main_objects.active_rag.ready, name=_process, error=None), int(HTTPResponses.OK)
    _err_string = f"RAG is active but it seems for the process: '{main_objects.active_rag.process}'" if _has_rag else "The RAG component is not initialized."
    return jsonify(
        active=False, name=_process, error=_err_string
    ), int(HTTPResponses.NOT_FOUND)


if __name__ in ["__main__"]:
    main_objects.app.run(host="0.0.0.0", port=9010)
