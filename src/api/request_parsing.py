"""Request parsing helpers for API endpoints."""

import logging
from collections import namedtuple
from typing import Optional

from munch import Munch

from main_utils import string_conformity


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
        _prompt = config.get("prompt_template", {})
        _vs = config.get(
            "vectorstore_server",
            config.get("vector_store_server", config.get("vector-store_server", None)),
        )
        return rag_config_json(
            {} if (_chatter is None or not isinstance(_chatter, dict)) else _chatter,
            _api_key,
            _lang,
            None if (_prompt is not None and len(_prompt) == 0) else _prompt,
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
