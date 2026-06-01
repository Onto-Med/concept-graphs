import pathlib
import random
from unittest import TestCase

from src.core.graph import GraphCreator
from src.pipeline.load_utils import FactoryLoader


class TestGraphCreator(TestCase):
    def setUp(self) -> None:
        self.process_name = "grascco"
        self.test_path = pathlib.Path("test/data/results/grascco")
        data = FactoryLoader.load_data(str(self.test_path.resolve()), self.process_name)
        embedding = FactoryLoader.load_embedding(
            str(self.test_path.resolve()), self.process_name, data
        )
        self.chunk_set_dict = data.data_chunk_sets
        self.embeddings = embedding.sentence_embeddings

    def test_build_graph_from_cluster(self):
        gc = GraphCreator(self.chunk_set_dict, self.embeddings)
        gc.build_graph_from_cluster(
            random.choices(range(0, len(self.chunk_set_dict)), k=40)
        )
