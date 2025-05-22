import sys
from unittest import TestCase

sys.path.insert(0, "src")
from src.MarqoExternalUtils import MarqoDocument, MarqoEmbeddingStore, MarqoDocumentStore
from src import embedding_functions, data_functions


class TestMarqoFunctions(TestCase):
    def test(self):
        base_name = "grascco_lokal"
        path_name = lambda x: f"../tmp/{base_name}/{base_name}_{x}.pickle"

        grascco_embedding = embedding_functions.SentenceEmbeddingsFactory.load(
            data_obj_path=path_name("data"),
            embeddings_path=path_name("embedding")
        )

        embedded_phrases = [
            (chunk["text"], grascco_embedding.sentence_embeddings[idx]) for idx, chunk in
            enumerate(grascco_embedding.data_processing_obj.data_chunk_sets)
            if 8 in [d["id"] for d in chunk["doc"]]
        ]

        phrases, embeddings = zip(*embedded_phrases)
        test_doc = MarqoDocument(
            phrases=phrases,
            embeddings=embeddings,
        )

        mqs = MarqoEmbeddingStore("http://localhost:8882", "grascco_lokal_test")
        mqds = MarqoDocumentStore(embedding_store=mqs)

        mqds.add_document(test_doc)