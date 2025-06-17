from itertools import tee

import networkx as nx

from src.util_functions import EmbeddingStore


class ConceptGraphIntegrationFactory:

    @staticmethod
    def create(
            embedding_store: EmbeddingStore,
            graphs: list[nx.Graph],
    ):
        doc_dict = {}
        for i, g in enumerate(graphs):
            for n, d in g.nodes(True):
                doc_dict[str(n)] = {
                    "_id": str(n),
                    "graph_cluster": doc_dict.get(str(n), {}).get("graph_cluster", []) + [str(i)],
                }
        ids, values = tee(doc_dict.items(), 2)
        _res = embedding_store.update_embeddings(
            embedding_ids=[i[0] for i in ids],
            values=[v[1] for v in values],
        )
        return _res