import itertools
import logging
import random
import re
import sys
import time
import os
import zipfile
import inspect

from pathlib import Path
from typing import Optional, Callable, Union, Dict, Generator, Iterable, List
from collections import Counter, defaultdict, namedtuple

import yaml
import spacy

from flask import Flask, jsonify, request
from flask.logging import default_handler

sys.path.insert(0, "src")
# import cluster_functions
import data_functions
# import embedding_functions
# import util_functions

app = Flask(__name__)

root = logging.getLogger()
root.addHandler(default_handler)

FILE_STORAGE_TMP = "./tmp" #ToDo: replace it with proper path in docker
# ToDo: file with stopwords will be POSTed: #filter_stop: Optional[list] = None,
# ToDo: evaluate 'None' values (yaml reader converts it to str) or maybe use boolean values only

@app.route("/preprocessing", methods=['POST'])
def data_preprocessing():
    app.logger.info("Preprocessing started")
    if request.method == "POST" and len(request.files) > 0 and "data" in request.files:

        app.logger.info("Reading config ...")
        config = read_config(request.files.get("config", None))
        app.logger.info(f"Parsed the following arguments for preprocessing:\n\t{config}")
        process_name = config.get("corpus_name",
                                  request.files.get("data", namedtuple('Corpus', ['name'])("default")).name)

        app.logger.info("Reading labels ...")
        labels = read_labels(request.files.get("labels", None))
        app.logger.info(f"Gathered the following labels:\n\t{list(labels.values()) if labels is not None else []}")

        app.logger.info("Reading data ...")
        data = read_data(request.files.get("data", None), config, labels)
        app.logger.info(f"Counted {len(data)} item ins zip file.")

        app.logger.info(f"Start preprocessing '{process_name}' ...")
        start_preprocessing(data, config, process_name)

    elif len(request.files) <= 0 or "data" not in request.files:
        app.logger.error("There were no files at all or no data files POSTed."
                         " At least a zip folder with the data is necessary!\n"
                         " It is also necessary to conform to the naming convention!\n"
                         "\t\ti.e.: curl -X POST -F data=@\"#SOME/PATH/TO/FILE.zip\"")
    return jsonify("Done.")


def read_data(data, config, labels) -> List[Dict[str, str]]:
    try:
        archive_path = f"{FILE_STORAGE_TMP}/{data.filename}"
        data.save(archive_path)
        with zipfile.ZipFile(archive_path, mode='r') as archive:
            return read_zip_content(archive, config, labels)
    except Exception as e:
        app.logger.error(f"Something went wrong with data file reading: {e}")


def read_config(config) -> Dict[str, str]:
    base_config = {'pipeline': 'en_core_web_trf', 'file_encoding': 'utf-8'}
    if config is None:
        app.logger.info("No config file provided; using default values")
    else:
        try:
            base_config = yaml.safe_load(config.stream)
        except Exception as e:
            app.logger.error(f"Couldn't read config file: {e}")
    return base_config


def read_labels(labels) -> Dict[str, str]:
    base_labels = {}
    if labels is None:
        app.logger.info("No labels file provided; no labels will be added to text data")
    else:
        try:
            base_labels = yaml.safe_load(labels.stream)
        except Exception as e:
            app.logger.error(f"Couldn't read labels file: {e}")
    return base_labels


def read_zip_content(zip_archive, config, labels) -> List[Dict[str, str]]:
    extension = config.get("file_extension", "txt")
    return [{"name": Path(f.filename).stem,
             "content": zip_archive.read(f.filename).decode(config.get('file_encoding', 'utf-8')),
             "label": labels.get(Path(f.filename).stem, None)}
            for f in zip_archive.filelist if (not f.is_dir()) and (Path(f.filename).suffix.lstrip('.') == extension.lstrip('.'))]


def start_preprocessing(data, config, cache_name):
    default_args = inspect.getfullargspec(data_functions.DataProcessingFactory.create)[0]
    spacy_language = spacy.load(config.pop("spacy_model"))
    _ = [config.pop(x, None) for x in list(config.keys()) if x not in default_args]
    data_functions.DataProcessingFactory.create(
        pipeline=spacy_language,
        base_data=data,
        cache_name=f"{cache_name}_data-processed",
        cache_path=Path(FILE_STORAGE_TMP),
        save_to_file=True,
        **config
    )


# label tab separated list of file name (minus its extension like '.txt') with according label
# config in yaml format
# curl -X POST -F data=@"LICENSE" -F config=@"conf/preprocessing_config.yaml" -F labels=@"..." http://localhost:9007/preprocessing
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9007)
