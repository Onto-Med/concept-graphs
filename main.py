import logging
import sys

from collections import namedtuple

from flask import Flask, jsonify, request
from flask.logging import default_handler

from preprocessing_util import PreprocessingUtil
from embedding_util import PhraseEmbeddingUtil

sys.path.insert(0, "src")
import data_functions
import embedding_functions
# import cluster_functions
# import util_functions


app = Flask(__name__)

root = logging.getLogger()
root.addHandler(default_handler)

FILE_STORAGE_TMP = "./tmp" #ToDo: replace it with proper path in docker
# ToDo: file with stopwords will be POSTed: #filter_stop: Optional[list] = None,
# ToDo: evaluate 'None' values (yaml reader converts it to str) or maybe use boolean values only


@app.route("/preprocessing", methods=['POST'])
def data_preprocessing():
    app.logger.info("=== Preprocessing started ===")
    if request.method == "POST" and len(request.files) > 0 and "data" in request.files:
        pre_proc = PreprocessingUtil(app, FILE_STORAGE_TMP)

        app.logger.info("Reading config ...")
        pre_proc.read_config(request.files.get("config", None))
        app.logger.info(f"Parsed the following arguments for preprocessing:\n\t{pre_proc.config}")
        process_name = pre_proc.config.get("corpus_name",
                                           request.files.get("data", namedtuple('Corpus', ['name'])("default")).name)

        app.logger.info("Reading labels ...")
        pre_proc.read_labels(request.files.get("labels", None))
        app.logger.info(f"Gathered the following labels:\n\t{list(pre_proc.labels.values()) if pre_proc.labels is not None else []}")

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

        app.logger.info("Reading config ...")
        phra_emb.read_config(request.files.get("config", None))
        app.logger.info(f"Parsed the following arguments for phrase embedding:\n\t{phra_emb.config}")
        process_name = phra_emb.config.get("corpus_name",
                                           request.files.get("data", namedtuple('Corpus', ['name'])("default")).name)

        app.logger.info(f"Start phrase embedding '{process_name}' ...")
        phra_emb.start_phrase_embedding(process_name, embedding_functions.SentenceEmbeddingsFactory)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9007)
