import logging
import pathlib
import sys

from collections import namedtuple

from flask import Flask, jsonify, request
from flask.logging import default_handler

from preprocessing_util import PreprocessingUtil
from embedding_util import PhraseEmbeddingUtil
from clustering_util import ClusteringUtil

sys.path.insert(0, "src")
import data_functions
import embedding_functions
import cluster_functions


app = Flask(__name__)

root = logging.getLogger()
root.addHandler(default_handler)

FILE_STORAGE_TMP = "./tmp"  # ToDo: replace it with proper path in docker


# ToDo: file with stopwords will be POSTed: #filter_stop: Optional[list] = None,
# ToDo: evaluate 'None' values (yaml reader converts it to str) or maybe use boolean values only

# ToDo: I downscale the embeddings twice... (that snuck in somehow); once in SentenceEmbeddings via create(down_scale_algorithm)
# ToDo: and once PhraseCluster via create(down_scale_algorithm). I can't remember why I added this to SentenceEmbeddings later on...
# ToDo: but I should make sure, that there is a check somewhere that the down scaling is not applied twice!

# ToDo: replace `jsonify` output with something more meaningful

# ToDo: uncommented sknet in cluster_functions (otherwise I could not debug)

# ToDo: make sure that no arguments can be supplied via config that won't work


@app.route("/preprocessing", methods=['POST'])
def data_preprocessing():
    app.logger.info("=== Preprocessing started ===")
    if request.method == "POST" and len(request.files) > 0 and "data" in request.files:
        pre_proc = PreprocessingUtil(app, FILE_STORAGE_TMP)

        process_name = read_config(pre_proc)

        app.logger.info("Reading labels ...")
        pre_proc.read_labels(request.files.get("labels", None))
        app.logger.info(
            f"Gathered the following labels:\n\t{list(pre_proc.labels.values()) if pre_proc.labels is not None else []}")

        app.logger.info("Reading data ...")
        pre_proc.read_data(request.files.get("data", None))
        app.logger.info(f"Counted {len(pre_proc.data)} item ins zip file.")

        app.logger.info(f"Start preprocessing '{process_name}' ...")
        pre_proc.start_preprocessing(process_name, data_functions.DataProcessingFactory)

    elif len(request.files) <= 0 or "data" not in request.files:
        app.logger.error("There were no files at all or no data files POSTed."
                         " At least a zip folder with the data is necessary!\n"
                         " It is also necessary to conform to the naming convention!\n"
                         "\t\ti.e.: curl -X POST -F data=@\"#SOME/PATH/TO/FILE.zip\"")
    return jsonify("Done.")


@app.route("/embedding", methods=['POST', 'GET'])
def phrase_embedding():
    app.logger.info("=== Phrase embedding started ===")
    if request.method in ["POST", "GET"]:
        phra_emb = PhraseEmbeddingUtil(app, FILE_STORAGE_TMP)

        process_name = read_config(phra_emb)

        app.logger.info(f"Start phrase embedding '{process_name}' ...")
        phra_emb.start_phrase_embedding(process_name, embedding_functions.SentenceEmbeddingsFactory)
    return jsonify("Done.")


@app.route("/clustering", methods=['POST', 'GET'])
def phrase_clustering():
    app.logger.info("=== Phrase clustering started ===")
    if request.method in ["POST", "GET"]:
        phra_clus = ClusteringUtil(app, FILE_STORAGE_TMP)

        process_name = read_config(phra_clus)

        app.logger.info(f"Start phrase clustering '{process_name}' ...")
        phra_clus.start_clustering(process_name, cluster_functions.PhraseClusterFactory)
    return jsonify("Done.")


def read_config(processor):
    app.logger.info("Reading config ...")
    processor.read_config(request.files.get("config", None))
    app.logger.info(f"Parsed the following arguments for {processor}:\n\t{processor.config}")
    return processor.config.pop("corpus_name", "default")
                                # request.files.get("data", namedtuple('Corpus', ['name'])("default")).name)


# ToDo: set debug=False
if __name__ == "__main__":
    f_storage = pathlib.Path(FILE_STORAGE_TMP)
    if not f_storage.exists():
        f_storage.mkdir()
    app.run(debug=True, host='0.0.0.0', port=9007)
