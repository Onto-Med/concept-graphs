import logging
from typing import Iterable, Optional, Union

import networkx as nx


class GraphIncorp:
    def __init__(
        self,
        graphs: Iterable[nx.Graph],
    ):
        self._graphs = list(graphs)
        self._last_result: Optional[list[tuple[str, bool]]] = None

    @classmethod
    def with_graphs(cls, graphs: Iterable[nx.Graph]):
        return cls(graphs)

    @property
    def last_result(self):
        return self._last_result

    @property
    def graphs(self):
        return self._graphs

    @staticmethod
    def _update_documents_in_graph(
        graph: nx.Graph, phrase_id: int, documents: list[dict]
    ):
        _docs = nx.get_node_attributes(graph, "documents").get(phrase_id, [])
        _present_doc_ids = set(d.get("id") for d in _docs if d.get("id", False))
        if any(
            _inter := _present_doc_ids.intersection(
                d.get("id") for d in documents if d.get("id", False)
            )
        ):
            # There shouldn't be a document where suddenly a phrase gets new additional offsets
            # if I come up with a situation where this can happen, something needs to be done here
            logging.warning(
                f"Somehow a phrase tries to update one or more already existing document/s:"
                f" phrase '{phrase_id}' --> document ids: '{_inter}'."
                f" Skipping these updates."
            )
        else:
            _docs.extend(documents)
        nx.set_node_attributes(graph, {phrase_id: _docs}, "documents")

    @staticmethod
    def _adding_phrase_to_graph(
        graph: nx.Graph, phrase_id: int, documents: list, label: str
    ):
        graph.add_node(phrase_id, documents=documents, label=label)

    def get_graph_by_id(self, graph_id: str) -> nx.Graph:
        return self.graphs[int(graph_id)]

    def get_graph_by_idx(self, graph_idx: int) -> nx.Graph:
        return self.graphs[graph_idx]

    def incorporate_phrase(
        self, phrase: dict, from_within: bool = False
    ) -> Union[bool, "GraphIncorp"]:
        graph_id = int(phrase.get("graph")) if phrase.get("graph", False) else None
        phrase_id = int(phrase.get("id")) if phrase.get("id", False) else None
        documents = phrase.get("documents", [])
        label = phrase.get("label")
        if graph_id is None or phrase_id is None:
            logging.error(
                f"'graph_id' or 'phrase_id' not in '{phrase}'. Can't incorporate phrase without."
            )
            if not from_within:
                return False
            else:
                self._last_result = [
                    (
                        phrase_id,
                        False,
                    )
                ]
                return self

        graph = self.get_graph_by_idx(graph_id)
        if phrase_id in graph.nodes():
            logging.info(
                f" Phrase with id '{phrase_id}' seems already to be present in graph '{graph_id}'."
                f" Updating document info."
            )
            GraphIncorp._update_documents_in_graph(graph, phrase_id, documents)
        else:
            logging.info(
                f"Phrase with id '{phrase_id}' is not present in graph '{graph_id}'. Adding it."
            )
            GraphIncorp._adding_phrase_to_graph(graph, phrase_id, documents, label)
        if not from_within:
            return True
        else:
            self._last_result = [
                (
                    phrase_id,
                    True,
                )
            ]
            return self

    def incorporate_phrases(self, phrases: Iterable[tuple[str, dict]]):
        _res = []
        for _id, d in phrases:
            self.incorporate_phrase(d, True)
            _res.extend(self.last_result)
        self._last_result = _res
        return self
