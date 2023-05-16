import itertools
import logging
import random
import re
import sys
import time
import os
from pathlib import Path
from typing import Optional, Callable, Union
from collections import Counter, defaultdict

import pandas as pd
import spacy
import ir_datasets
from sklearn.datasets import fetch_20newsgroups
from flask import Flask, jsonify, request

sys.path.insert(0, "src")
import cluster_functions
import data_functions
import embedding_functions
import util_functions

logging.basicConfig()
logging.root.setLevel(logging.INFO)

app = Flask(__name__)


@app.route("/preprocessing", methods=['POST'])
def data_preprocessing():
    if request.method == "POST" and len(request.files) > 0 and "data" in request.files:
        data = request.files.get("data", None)
        config = request.files.get("config", None)
        logging.info("Preprocessing started")
        #ToDo: unzip data
        #ToDo: read config yaml
        #ToDo: start preprocessing with config values on data
    return jsonify("Done.")


# data in zip format
# config in yaml format
# curl -X POST -F data=@"LICENSE" -F config=@"conf/preprocessing_config.yaml" http://localhost:9007/preprocessing
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9007)
