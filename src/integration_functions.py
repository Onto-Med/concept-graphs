import pathlib
from itertools import tee

import networkx as nx

from src.util_functions import EmbeddingStore, save_pickle


class ConceptGraphIntegrationFactory:

    @staticmethod
    def create(
        embedding_store: EmbeddingStore,
        graphs: list[nx.Graph],
        cache_path: pathlib.Path,
        cache_name: str,
        check_for_reasonable_result: bool = False,
    ):
        _file_path = cache_path / pathlib.Path(f"{cache_name}.pickle")
        doc_dict = {}
        for i, g in enumerate(graphs):
            for n, d in g.nodes(True):
                doc_dict[str(n)] = {
                    "_id": str(n),
                    "graph_cluster": doc_dict.get(str(n), {}).get("graph_cluster", [])
                    + [str(i)],
                }
        ids, values = tee(doc_dict.items(), 2)
        _res = embedding_store.update_embeddings(
            embedding_ids=[i[0] for i in ids],
            values=[v[1] for v in values],
        )
        if check_for_reasonable_result and (len(_res) != len(doc_dict)):
            raise AssertionError(
                f"Count of input embeddings ({len(doc_dict)}) should be equal to count of result ids ({len(_res)}). If you're certain it doesn't need to be this way, set the argument 'check_for_reasonable_result' to 'false' and run this step again."
            )
        save_pickle({"updated_embeddings": sorted([int(x) for x in _res])}, _file_path)
        return _res
