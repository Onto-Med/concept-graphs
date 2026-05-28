"""Routes for inspecting pipeline artifacts."""

import pathlib

from flask import Blueprint, Response, jsonify, request

from src.pipeline.load_utils import FactoryLoader
from src.api.services.artifact_responses import (
    clustering_get_concepts,
    data_get_statistics,
    embedding_get_statistics,
    graph_get_specific,
    graph_get_statistics,
)
from main_utils import HTTPResponses, StepsName, get_bool_expression, string_conformity
from src.core import embedding_functions
from src.api.routes.common import path_arg_error, unspecified_server_error


def create_artifact_blueprint(app, storage, pipeline):
    """Create the blueprint for preprocessing, embedding, clustering, and graph routes."""
    blueprint = Blueprint("artifact_routes", __name__)

    @blueprint.route("/preprocessing/<path_arg>", methods=["GET"])
    def data_preprocessing_with_arg(path_arg):
        process = string_conformity(request.args.get("process", "default"))
        path_arg = path_arg.lower()

        path_args = ["statistics", "noun_chunks"]
        if path_arg in path_args:
            data_obj = FactoryLoader.with_active_objects(
                str(pathlib.Path(storage.file_storage_dir, process).resolve()),
                process,
                pipeline.active_objects,
                StepsName.DATA,
            )
            if data_obj is None:
                return unspecified_server_error()
            if path_arg == "statistics":
                return data_get_statistics(data_obj), HTTPResponses.OK
            if path_arg == "noun_chunks":
                return jsonify(noun_chunks=data_obj.data_chunk_sets), HTTPResponses.OK
        return path_arg_error("preprocessing", path_arg, path_args)

    @blueprint.route("/embedding/<path_arg>", methods=["GET"])
    def phrase_embedding_with_arg(path_arg):
        process = string_conformity(request.args.get("process", "default"))
        path_arg = path_arg.lower()

        path_args = ["statistics"]
        if path_arg in path_args:
            emb_obj = FactoryLoader.with_active_objects(
                str(pathlib.Path(storage.file_storage_dir, process).resolve()),
                process,
                pipeline.active_objects,
                StepsName.EMBEDDING,
            )
            if emb_obj is None:
                return unspecified_server_error()
            if path_arg == "statistics":
                return embedding_get_statistics(emb_obj)
        return path_arg_error("embedding", path_arg, path_args)

    @blueprint.route("/clustering/<path_arg>", methods=["GET"])
    def clustering_with_arg(path_arg):
        process = string_conformity(request.args.get("process", "default"))
        top_k = int(request.args.get("top_k", 15))
        distance = float(request.args.get("distance", 0.6))
        path_arg = path_arg.lower()

        path_args = ["concepts"]
        if path_arg in path_args:
            cluster_obj = FactoryLoader.with_active_objects(
                str(pathlib.Path(storage.file_storage_dir, process).resolve()),
                process,
                pipeline.active_objects,
                StepsName.CLUSTERING,
            )
            if cluster_obj is None:
                return unspecified_server_error()
            if path_arg == "concepts":
                emb_obj = FactoryLoader.with_active_objects(
                    str(pathlib.Path(storage.file_storage_dir, process).resolve()),
                    process,
                    pipeline.active_objects,
                    StepsName.EMBEDDING,
                )
                cluster_gen = embedding_functions.show_top_k_for_concepts(
                    cluster_obj=cluster_obj.concept_cluster,
                    embedding_object=emb_obj,
                    yield_concepts=True,
                    top_k=top_k,
                    distance=distance,
                )
                return clustering_get_concepts(cluster_gen)
        return path_arg_error("clustering", path_arg, path_args)

    @blueprint.route("/graph/<path_arg>", methods=["POST", "GET"])
    def graph_with_arg(path_arg):
        process = string_conformity(request.args.get("process", "default"))
        draw = get_bool_expression(request.args.get("draw", False))
        path_arg = path_arg.lower()
        graph_list = FactoryLoader.with_active_objects(
            str(pathlib.Path(storage.file_storage_dir, process).resolve()),
            process,
            pipeline.active_objects,
            StepsName.GRAPH,
        )

        path_args = ["statistics"]
        if path_arg in path_args:
            if graph_list is None:
                return unspecified_server_error()
            try:
                if path_arg == "statistics":
                    result = graph_get_statistics(
                        app,
                        graph_list,
                        storage.file_storage_dir,
                    )
                    http_response = HTTPResponses.OK
                    if "error" in result:
                        http_response = HTTPResponses.INTERNAL_SERVER_ERROR
                    return jsonify(name=process, **result), http_response
            except FileNotFoundError:
                return Response(
                    f"There is no graph data present for '{process}'.\n",
                    status=int(HTTPResponses.NOT_FOUND),
                )
        elif path_arg.isdigit():
            graph_nr = int(path_arg)
            return graph_get_specific(
                graph_list,
                graph_nr,
                path=storage.file_storage_dir,
                draw=draw,
            )
        return path_arg_error("graph", path_arg, path_args + ["#ANY_INTEGER"])

    return blueprint
