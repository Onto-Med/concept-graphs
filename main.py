import logging
import pathlib
import sys

from collections import namedtuple, defaultdict

from flask import Flask, jsonify, request
from flask.logging import default_handler

from preprocessing_util import PreprocessingUtil
from embedding_util import PhraseEmbeddingUtil
from clustering_util import ClusteringUtil
from graph_creation_util import GraphCreationUtil

sys.path.insert(0, "src")
import data_functions
import embedding_functions
import cluster_functions
import util_functions


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

# ToDo: add PATH and QUERY args to README


@app.route("/")
def index():
    return jsonify(available_endpoints=['/preprocessing', '/embedding', '/clustering', '/graph'])


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
        return get_data_statistics(pre_proc.start_preprocessing(process_name, data_functions.DataProcessingFactory))

    elif len(request.files) <= 0 or "data" not in request.files:
        app.logger.error("There were no files at all or no data files POSTed."
                         " At least a zip folder with the data is necessary!\n"
                         " It is also necessary to conform to the naming convention!\n"
                         "\t\ti.e.: curl -X POST -F data=@\"#SOME/PATH/TO/FILE.zip\"")
    return jsonify("Nothing to do.")


@app.route("/preprocessing/<path_arg>", methods=['GET'])
def data_preprocessing_with_arg(path_arg):
    process = request.args.get("process", "default")
    path_arg = path_arg.lower()

    _path_args = ["statistics", "noun_chunks"]
    if path_arg in _path_args:
        data_obj = data_functions.DataProcessingFactory.load(
            pathlib.Path(pathlib.Path(FILE_STORAGE_TMP) / f"{process}_data-processed.pickle"))
    else:
        return jsonify(error=f"No such path argument '{path_arg}' for 'preprocessing' endpoint.",
                       possible_path_args=[f"/{p}" for p in _path_args])

    if path_arg.lower() == "statistics":
        return get_data_statistics(data_obj)
    elif path_arg.lower() == "noun_chunks":
        return jsonify(
            noun_chunks=data_obj.data_chunk_sets
        )


@app.route("/embedding", methods=['POST', 'GET'])
def phrase_embedding():
    app.logger.info("=== Phrase embedding started ===")
    if request.method in ["POST", "GET"]:
        phra_emb = PhraseEmbeddingUtil(app, FILE_STORAGE_TMP)

        process_name = read_config(phra_emb)

        app.logger.info(f"Start phrase embedding '{process_name}' ...")
        try:
            return get_embedding_statistics(
                phra_emb.start_phrase_embedding(process_name, embedding_functions.SentenceEmbeddingsFactory))
        except FileNotFoundError:
            return jsonify(f"There is no processed data for the '{process_name}' process to be embedded.")
    return jsonify("Nothing to do.")


@app.route("/embedding/<path_arg>", methods=['GET'])
def phrase_embedding_with_arg(path_arg):
    process = request.args.get("process", "default")
    path_arg = path_arg.lower()

    _path_args = ["statistics"]
    if path_arg in _path_args:
        emb_obj = embedding_functions.SentenceEmbeddingsFactory.load(
            pathlib.Path(pathlib.Path(FILE_STORAGE_TMP) / f"{process}_data-processed.pickle"),
            pathlib.Path(pathlib.Path(FILE_STORAGE_TMP) / f"{process}_embeddings.pickle"),
        )
    else:
        return jsonify(error=f"No such path argument '{path_arg}' for 'preprocessing' endpoint.",
                       possible_path_args=[f"/{p}" for p in _path_args])

    if path_arg.lower() == "statistics":
        return get_embedding_statistics(emb_obj)


@app.route("/clustering", methods=['POST', 'GET'])
def phrase_clustering():
    app.logger.info("=== Phrase clustering started ===")
    if request.method in ["POST", "GET"]:
        saved_config = request.args.get("config", False)
        if not saved_config:
            phra_clus = ClusteringUtil(app, FILE_STORAGE_TMP)

            process_name = read_config(phra_clus)

            app.logger.info(f"Start phrase clustering '{process_name}' ...")
            try:
                _cluster_gen = phra_clus.start_clustering(process_name, cluster_functions.PhraseClusterFactory)
                get_clustering_concepts(_cluster_gen)
            except FileNotFoundError:
                return jsonify(f"There is no embedded data for the '{process_name}' process to be clustered.")
        else:
            return jsonify(saved_config)
    return jsonify("Nothing to do.")


@app.route("/clustering/<path_arg>", methods=['GET'])
def clustering_with_arg(path_arg):
    process = request.args.get("process", "default")
    path_arg = path_arg.lower()

    _path_args = ["concepts"]
    if path_arg in _path_args:
        cluster_obj = cluster_functions.PhraseClusterFactory.load(
            pathlib.Path(pathlib.Path(FILE_STORAGE_TMP) / f"{process}_clustering.pickle"),
        )
    else:
        return jsonify(error=f"No such path argument '{path_arg}' for 'preprocessing' endpoint.",
                       possible_path_args=[f"/{p}" for p in _path_args])

    if path_arg.lower() == "concepts":
        emb_obj = util_functions.load_pickle(pathlib.Path(pathlib.Path(FILE_STORAGE_TMP) / f"{process}_embeddings.pickle"))
        _cluster_gen = embedding_functions.show_top_k_for_concepts(
            cluster_obj=cluster_obj.concept_cluster, embedding_object=emb_obj, yield_concepts=True,
            top_k=int(request.args.get("top_k", 15)), distance=float(request.args.get("distance", .6))
        )
        return get_clustering_concepts(_cluster_gen)


@app.route("/graph", methods=['POST', 'GET'])
def graph_creation():
    app.logger.info("=== Graph creation started ===")
    if request.method in ["POST", "GET"]:
        graph_create = GraphCreationUtil(app, FILE_STORAGE_TMP)


def read_config(processor):
    app.logger.info("Reading config ...")
    processor.read_config(request.files.get("config", None))
    app.logger.info(f"Parsed the following arguments for {processor}:\n\t{processor.config}")
    return processor.config.pop("corpus_name", "default")
                                # request.files.get("data", namedtuple('Corpus', ['name'])("default")).name)


def get_data_statistics(data_obj):
    return jsonify(
        number_of_documents=data_obj.documents_n,
        number_of_data_chunks=data_obj.chunk_sets_n,
        number_of_label_types=len(data_obj.true_labels)
    )


def get_embedding_statistics(emb_obj):
    return jsonify(
        number_of_embeddings=emb_obj.sentence_embeddings.shape[0],
        embedding_dim=emb_obj.embedding_dim
    )


def get_clustering_concepts(cluster_gen):
    _cluster_dict = defaultdict(list)
    for c_id, _, text in cluster_gen:
        _cluster_dict[f"concept-{c_id}"].append(text)
    return jsonify(**_cluster_dict)


# ToDo: set debug=False
if __name__ == "__main__":
    f_storage = pathlib.Path(FILE_STORAGE_TMP)
    if not f_storage.exists():
        f_storage.mkdir()
    app.run(debug=True, host='0.0.0.0', port=9007)
