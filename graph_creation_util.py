import pathlib
import pickle
from pathlib import Path

import yaml
import sys

from flask import jsonify

sys.path.insert(0, "src")
import util_functions


class GraphCreationUtil:

    def __init__(self, app, file_storage):
        self._app = app
        self._file_storage = Path(file_storage)
        self.config = None

    def read_config(self, config, process_name=None, language=None):
        base_config = {}
        if config is None:
            self._app.logger.info("No config file provided; using default values")
        else:
            try:
                base_config = yaml.safe_load(config.stream)
                if base_config.get('model', False):
                    raise KeyError(f"No model name provided in config: {base_config}")
            except Exception as e:
                self._app.logger.error(f"Couldn't read config file: {e}")
                return jsonify("Encountered error. See log.")
        self.config = base_config
        if process_name is not None:
            base_config["corpus_name"] = process_name
        sub_path = base_config.get('corpus_name', 'default')
        with Path(Path(self._file_storage) / f"{sub_path}_graph_config.yaml"
                  ).open('w') as config_save:
            yaml.safe_dump(base_config, config_save)

    def set_file_storage_path(self, sub_path):
        self._file_storage = Path(self._file_storage / sub_path)
        self._file_storage.mkdir(exist_ok=True)  # ToDo: warning when folder exists

    def has_pickle(self, process):
        _step = "graphs"
        _pickle = Path(self._file_storage / f"{process}_{_step}.pickle")
        return _pickle.exists()

    def start_process(self, cache_name, process_factory, exclusion_ids=None):
        sent_emb = util_functions.load_pickle(Path(self._file_storage / f"{cache_name}_embeddings.pickle"))
        cluster_obj = util_functions.load_pickle(Path(self._file_storage / f"{cache_name}_clustering.pickle"))

        config = self.config.copy()

        concept_graph_clustering = process_factory(
            sentence_embedding_obj=sent_emb,
            cluster_obj=cluster_obj.concept_cluster,
            cluster_exclusion_ids=exclusion_ids
        ).create_concept_graph_clustering()

        concept_graphs = concept_graph_clustering.build_document_concept_matrix(
            break_after_graph_creation=True,
            **config
        )
        with pathlib.Path(self._file_storage / f"{cache_name}_graphs.pickle").open("wb") as graphs_out:
            pickle.dump(concept_graphs, graphs_out)

        return concept_graphs