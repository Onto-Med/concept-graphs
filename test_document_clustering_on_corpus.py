import pathlib
import unittest
import sys

from sklearn.cluster import KMeans

from load_utils import FactoryLoader
from main_utils import StepsName
# from src.cluster_functions import WordEmbeddingClustering
sys.path.insert(0, "src")
from src import (
    data_functions,
    embedding_functions,
    cluster_functions,
    graph_functions
)


class TestDocumentClusteringOnCorpus(unittest.TestCase):
    test_path = pathlib.Path("./tmp/grascco")

    def test_clustering(self):
        data = FactoryLoader.load(
            StepsName.DATA, str(self.test_path.resolve()), "grascco"
        )
        emb = FactoryLoader.load(
            StepsName.EMBEDDING, str(self.test_path.resolve()), "grascco", data
        )
        cluster = FactoryLoader.load(
            StepsName.CLUSTERING, str(self.test_path.resolve()), "grascco", data, emb
        )
        graph = FactoryLoader.load(
            StepsName.GRAPH, str(self.test_path.resolve()), "grascco"
        )
        cg_clustering = cluster_functions.WordEmbeddingClustering.with_objects(sentence_embedding_obj=emb).create_concept_graph_clustering()
        cg_clustering.build_document_concept_matrix(external_graphs=graph)
        _ari = cluster_functions.WordEmbeddingClustering.ari_score(cg_clustering, KMeans)


if __name__ == "__main__":
    unittest.main()
