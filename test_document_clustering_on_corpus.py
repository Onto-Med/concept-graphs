import pathlib
import unittest
import sys

from sklearn.cluster import KMeans

from load_utils import FactoryLoader
from main_utils import StepsName
from src.cluster_functions import WordEmbeddingClustering

# from src.cluster_functions import WordEmbeddingClustering
sys.path.insert(0, "src")
from src import data_functions, embedding_functions, cluster_functions, graph_functions


class TestDocumentClusteringOnCorpus(unittest.TestCase):
    process_name = "grascco_stem"
    test_path = pathlib.Path(f"./tmp/{process_name}")

    def test_clustering(self):
        data = FactoryLoader.load(
            StepsName.DATA, str(self.test_path.resolve()), self.process_name
        )
        emb = FactoryLoader.load(
            StepsName.EMBEDDING, str(self.test_path.resolve()), self.process_name, data
        )
        cluster = FactoryLoader.load(
            StepsName.CLUSTERING,
            str(self.test_path.resolve()),
            self.process_name,
            data,
            emb,
        )
        graph = FactoryLoader.load(
            StepsName.GRAPH, str(self.test_path.resolve()), self.process_name
        )
        word_embedding_clustering = (
            cluster_functions.WordEmbeddingClustering.with_objects(
                sentence_embedding_obj=emb
            )
        )
        cg_clustering: WordEmbeddingClustering._ConceptGraphClustering
        cg_clustering = word_embedding_clustering.create_concept_graph_clustering()
        cg_clustering.build_document_concept_matrix(external_graphs=graph)
        _ari = word_embedding_clustering.ari_score(
            embeddings_cluster_obj=cg_clustering, clustering_obj=KMeans
        )
        print(_ari)


if __name__ == "__main__":
    unittest.main()
