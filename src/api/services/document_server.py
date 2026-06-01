"""Document server service helpers."""

from collections.abc import Iterable

import flask
import requests
import yaml
from munch import Munch
from werkzeug.datastructures import FileStorage

from src.api.services.pipeline_params import get_dict_expression
from src.common.parsing import get_bool_expression


def get_data_server_config(
    document_server_config: FileStorage | dict, app: flask.Flask
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
            raise TypeError(
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
    except (yaml.YAMLError, AttributeError, TypeError, ValueError) as e:
        app.logger.error(
            "Couldn't read document server config: %s\nUsing default values '%s'.",
            e,
            base_config,
        )
    return base_config


def check_data_server(
    config: Munch | dict,
):
    for _url_key in ["url", "alternate_url"]:
        _url = config.get(_url_key, None)
        if _url is None:
            continue
        final_url = f"{_url.rstrip('/')}:{config.get('port', '9008')}/{config.get('index', 'documents').lstrip('/').rstrip('/')}/_count"
        try:
            _response = requests.get(final_url)
        except requests.exceptions.RequestException:
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
    port: str | int,
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
