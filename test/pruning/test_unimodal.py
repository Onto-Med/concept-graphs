import unittest

import networkx as nx
import numpy as np

from src.pruning import unimodal


class TestMLF(unittest.TestCase):
    def test_rejects_non_networkx_inputs(self):
        mlf = unimodal.MLF()
        with self.assertRaises(TypeError):
            mlf.fit_transform([(0, 1, 1)])

    def test_rejects_invalid_networkx_graphs(self):
        mlf = unimodal.MLF()

        graph_without_weights = nx.Graph()
        graph_without_weights.add_edge(0, 1)
        with self.assertRaises(ValueError):
            mlf.fit_transform(graph_without_weights)

        graph_with_non_numeric_weight = nx.Graph()
        graph_with_non_numeric_weight.add_edge(0, 1, weight="invalid")
        with self.assertRaises(TypeError):
            mlf.fit_transform(graph_with_non_numeric_weight)

        graph_with_loop = nx.Graph()
        graph_with_loop.add_edge(0, 0, weight=1)
        with self.assertRaises(ValueError):
            mlf.fit_transform(graph_with_loop)

        multigraph = nx.MultiGraph()
        multigraph.add_edge(0, 1, weight=1)
        with self.assertRaises(TypeError):
            mlf.fit_transform(multigraph)

    def test_fit_transform_adds_significance(self):
        graph = nx.barabasi_albert_graph(100, 3, seed=42)
        for source, target in graph.edges:
            graph[source][target]["weight"] = np.random.randint(1, 200)

        result = unimodal.MLF(directed=False).fit_transform(graph)

        self.assertIs(result, graph)
        for _, _, data in result.edges(data=True):
            self.assertIn("significance", data)
            self.assertIsNotNone(data["significance"])

    def test_return_copy(self):
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=1)

        result = unimodal.MLF(directed=False).fit_transform(graph, return_copy=True)

        self.assertIsNot(result, graph)
        self.assertIn("significance", result[0][1])
        self.assertNotIn("significance", graph[0][1])


if __name__ == "__main__":
    unittest.main()
