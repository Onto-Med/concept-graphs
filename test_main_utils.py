import pathlib
import unittest

from load_utils import FactoryLoader
from main_utils import StepsName


class TestLoadFactory(unittest.TestCase):
    test_path = pathlib.Path("./tmp/grascco")

    def test_load_data(self):
        _ = FactoryLoader.load_data(str(self.test_path.resolve()), "grascco")

    def test_load_embedding(self):
        _ = FactoryLoader.load_embedding(str(self.test_path.resolve()), "grascco")

    def test_load_clustering(self):
        _ = FactoryLoader.load_clustering(str(self.test_path.resolve()), "grascco")

    def test_load_graph(self):
        _ = FactoryLoader.load_graph(str(self.test_path.resolve()), "grascco")

    def test_load(self):
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


if __name__ == "__main__":
    unittest.main()
