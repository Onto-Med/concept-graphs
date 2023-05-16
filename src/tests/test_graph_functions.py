import pathlib
import re
import random
from unittest import TestCase
from graph_functions import GraphCreator
from util_functions import unpickle_or_run
from data_functions import clean_span
from spacy.tokens import Doc


class TestGraphCreator(TestCase):

    def setUp(self) -> None:
        pickle_base = pathlib.Path("../../pickles/")

        # if not Doc.has_extension("text_id"):
        #     Doc.set_extension("text_id", default=None)
        # if not Doc.has_extension("doc_name"):
        #     Doc.set_extension("doc_name", default=None)
        # if not Doc.has_extension("doc_topic"):
        #     Doc.set_extension("doc_topic", default=None)
        #
        # self.embeddings = unpickle_or_run(base_path=pickle_base, filename="sent_transformer_embeddings_w-life")
        # self._processed_docs = unpickle_or_run(base_path=pickle_base, filename="processed_docs_w-life")
        # self._noun_chunks = [{"chunk": l, "text_id": x._.text_id, "doc_name": x._.doc_name}
        #                      for x in self._processed_docs for l in x.noun_chunks
        #                      if not (re.match(r"\W", l.text) and len(l.text) == 1)]
        # self._data_dict = [{"text": clean_span(c["chunk"]), "text_id": c["text_id"], "doc_name": c["doc_name"]} for c in
        #                    self._noun_chunks]
        # self.build_chunk_set_dict()

    def build_chunk_set_dict(self):
        chunk_set_dict_temp = {}
        # for c in self._data_dict:
        #     _text = c["text"]
        #     if _text is None:
        #         continue
        #
        #     if chunk_set_dict_temp.get(_text, False):
        #         _docs = set(chunk_set_dict_temp[_text]["doc"])
        #         _docs.add(c["text_id"])
        #         chunk_set_dict_temp[_text]["doc"] = list(_docs)
        #         chunk_set_dict_temp[_text]["count"] += 1
        #     else:
        #         chunk_set_dict_temp[_text] = {"doc": [c["text_id"]], "count": 1}
        # self.chunk_set_dict = [{"text": t, "doc": ch["doc"], "count": ch["count"]}
        #                        for t, ch in chunk_set_dict_temp.items()]

    def test_build_graph_from_cluster(self):
        gc = GraphCreator(self.chunk_set_dict, self.embeddings)
        gc.build_graph_from_cluster(random.choices(range(0, len(self.chunk_set_dict)), k=40))
