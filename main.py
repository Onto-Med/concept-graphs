import itertools
import logging
import random
import re
import sys
import time
import os
import zipfile

from pathlib import Path
from typing import Optional, Callable, Union
from collections import Counter, defaultdict

import yaml
import pandas as pd
import spacy
import ir_datasets

from sklearn.datasets import fetch_20newsgroups
from flask import Flask, jsonify, request
from flask.logging import default_handler

sys.path.insert(0, "src")
# import cluster_functions
# import data_functions
# import embedding_functions
# import util_functions

app = Flask(__name__)

root = logging.getLogger()
root.addHandler(default_handler)

TMP = "./tmp"

@app.route("/preprocessing", methods=['POST'])
def data_preprocessing():
    app.logger.info("Preprocessing started")
    if request.method == "POST" and len(request.files) > 0 and "data" in request.files:
        data = request.files.get("data", None)
        config = request.files.get("config", None)
        labels = request.files.get("labels", None)

        app.logger.info("Reading data ...")
        try:
            archive_path = f"{TMP}/{data.filename}"
            data.save(archive_path)
            with zipfile.ZipFile(archive_path, mode='r') as archive:
                app.logger.info(read_zip_content(archive).__next__())
        except Exception as e:
            app.logger.error(f"Something went wrong with data file reading: {e}")

        if config is not None:
            app.logger.info("Reading config ...")
            # ToDo: read config yaml
        else:
            app.logger.info("No config file provided; using default values")

        # ToDo: start preprocessing with config values on data
    return jsonify("Done.")


def read_zip_content(zip_archive):
    #ToDo: name - split ext according to extension in config
    #ToDo: content - decode according to encoding in config
    #ToDo: label - set according to separate label file if present
    return ({"name": f.filename.split("/")[-1], "content": zip_archive.read(f.filename).decode("utf-8"), "label": None}
            for f in zip_archive.filelist if not f.is_dir())


# data in zip format
# config in yaml format
# curl -X POST -F data=@"LICENSE" -F config=@"conf/preprocessing_config.yaml" -F labels=@"..." http://localhost:9007/preprocessing
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9007)
