"""Configuration service helpers."""

import collections
import pathlib
from typing import Optional, Union

import flask
import yaml
from flask import request
from werkzeug.datastructures import FileStorage
from yaml.representer import RepresenterError

from src.common.parsing import string_conformity
from src.pipeline.base import BaseUtil
from src.pipeline.status import StepsName
from src.pipeline.steps.clustering_util import ClusteringUtil
from src.pipeline.steps.embedding_util import PhraseEmbeddingUtil
from src.pipeline.steps.graph_creation_util import GraphCreationUtil
from src.pipeline.steps.integration_util import ConceptGraphIntegrationUtil
from src.pipeline.steps.preprocessing_util import PreprocessingUtil


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
