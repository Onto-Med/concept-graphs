"""Routes for RAG initialization and question answering."""

import json
import logging
from operator import itemgetter
from time import sleep

from flask import Blueprint, jsonify, request

from src.api.request_parsing import get_doc_ids, parse_rag_config_json
from src.api.services.rag_vectorstore import (
    fill_chunk_vectorstore,
    initialize_chunk_vectorstore,
)
from src.api.context import ActiveRAG
from src.api.responses import HTTPResponses
from src.common.parsing import (
    get_bool_expression,
    string_conformity,
)
from src.common.threads import StoppableThread
from src.rag.marqo_rag_utils import extract_text_from_highlights
from src.rag.rag import RAG


def create_rag_blueprint(rag, processes, storage, pipeline):
    """Create the blueprint for RAG initialization and question routes."""
    blueprint = Blueprint("rag_routes", __name__)

    @blueprint.route("/rag/init", methods=["POST"])
    def init_rag():
        if request.method == "POST":
            if request.headers.get("Content-Type") == "application/json":
                process = string_conformity(request.args.get("process", "default"))
                force_init = get_bool_expression(request.args.get("force", "false"))
                config = parse_rag_config_json(request.json)
                if config is None:
                    return jsonify("RAG config couldn't be parsed."), int(
                        HTTPResponses.BAD_REQUEST
                    )

                rag_thread_id = f"rag_fill_vectorstore_{process}"
                vector_store = initialize_chunk_vectorstore(
                    process, config.vectorstore_server, force_init=force_init
                )
                chatter = config.chatter.pop(
                    "chatter", "src.rag.chatters.blablador.BlabladorChatter"
                )
                rag.active = ActiveRAG(
                    rag=RAG.with_chatter(
                        api_key=config.api_key,
                        chatter=chatter,
                        language=config.language,
                        **config.chatter,
                    ).with_prompt(prompt_template_config=config.prompt_template),
                    vectorstore=vector_store,
                    process=process,
                )
                if (not vector_store.is_filled()) or force_init:
                    rag_init_thread = StoppableThread(
                        target_args=(process, rag, storage, pipeline),
                        group=None,
                        target=fill_chunk_vectorstore,
                        name=None,
                    )
                    rag_init_thread.start()
                    sleep(1.0)
                    processes.threads[rag_thread_id] = rag_init_thread
                    return jsonify("Starting initializing RAG component."), int(
                        HTTPResponses.OK
                    )
                if init_thread := processes.threads.get(rag_thread_id, None):
                    if not init_thread.return_value:
                        return jsonify(
                            f"There already seems to be an initialization thread running for process {process}. Please wait for it to finish."
                        ), int(HTTPResponses.ACCEPTED)
                rag.active.switch_readiness()
                return jsonify("Initialized RAG component."), int(HTTPResponses.OK)
            return jsonify(
                f"Wrong content type '{request.headers.get('Content-Type')}'; need 'application/json'"
            ), int(HTTPResponses.BAD_REQUEST)
        return jsonify(f"Method not supported: '{request.method}'."), int(
            HTTPResponses.BAD_REQUEST
        )

    @blueprint.route("/rag/question", methods=["GET", "POST"])
    def rag_question():
        doc_ids = []
        doc_part_limit = 15
        if (
            request.method == "POST"
            and request.headers.get("Content-Type") == "application/json"
            and request.json is not None
        ):
            doc_ids = get_doc_ids(request.json)
            doc_part_limit = request.json.get("limit", 15)
        if request.method in ["GET", "POST"]:
            if rag.active is None or not rag.active.ready:
                return jsonify(
                    "No active and ready rag component found."
                    " You need to initialize it first and wait for it to be ready."
                ), int(HTTPResponses.NOT_FOUND)
            question = request.args.get("q", request.args.get("question", False))
            process = string_conformity(request.args.get("process", "default"))
            language = rag.active.rag.language
            if not question:
                return jsonify("No question supplied."), int(HTTPResponses.BAD_REQUEST)
            if rag.active.process != process:
                return (
                    jsonify(
                        f"There is no ready and active RAG component for '{process}'."
                        f" Currently active is : '{rag.active.process}'; use the 'init' endpoint."
                    ),
                    int(HTTPResponses.BAD_REQUEST),
                )

            documents = list(
                zip(
                    *itemgetter(1, -1)(
                        extract_text_from_highlights(
                            rag.active.vectorstore.get_chunks(
                                question,
                                filter_by=(
                                    {"doc_id": doc_ids} if len(doc_ids) > 0 else None
                                ),
                                limit=doc_part_limit,
                            ),
                            token_limit=150,
                            lang=language,
                        )
                    )
                )
            )
            success, answer = rag.active.rag.with_documents(
                documents, concat_by="doc_id"
            ).build_and_invoke(question)
            reference = {k: v.metadata for k, v in rag.active.rag.documents.items()}
            if success:
                return jsonify(
                    answer=answer, info=json.dumps(reference, ensure_ascii=False)
                ), int(HTTPResponses.OK)
            logging.error("[RAG Question]: %s", answer)
            return jsonify(error=answer), int(HTTPResponses.INTERNAL_SERVER_ERROR)
        return jsonify(f"Method not supported: '{request.method}'."), int(
            HTTPResponses.BAD_REQUEST
        )

    return blueprint
