import logging
from pathlib import Path
from typing import Optional, Union

import networkx as nx

from main_utils import StepsName
from src.cluster_functions import PhraseClusterFactory
from src.data_functions import DataProcessingFactory
from src.embedding_functions import SentenceEmbeddingsFactory
from src.util_functions import load_pickle


class FactoryLoader:
    @classmethod
    def load_data(
            cls,
            path: str,
            process: str
    ) -> DataProcessingFactory.DataProcessing:
        data = DataProcessingFactory.load(
            data_obj_path=Path(Path(path) / f"{process}_data.pickle")
        )
        if not hasattr(data, "data_chunk_sets"):
            raise AttributeError("The provided object is not a DataProcessing object.")
        return data

    @classmethod
    def load_embedding(
            cls,
            path: str,
            process: str,
            data_obj: Optional[DataProcessingFactory.DataProcessing] = None,
            vector_store: Optional[dict] = None,
    ) -> SentenceEmbeddingsFactory.SentenceEmbeddings:
        sent_emb = load_pickle(Path(Path(path) / f"{process}_embedding.pickle"))
        if data_obj is not None:
            if not hasattr(data_obj, "data_chunk_sets"):
                logging.warning("The provided data_obj seems not to be a DataProcessing object. Trying to load it instead.")
                data_obj = None
        if isinstance(sent_emb, dict) or vector_store is not None:
            _vec = {} if vector_store is None else vector_store
            sent_emb = SentenceEmbeddingsFactory.load(
                embeddings_obj_path=Path(Path(path) / f"{process}_embedding.pickle"),
                data_obj=cls.load_data(path, process) if data_obj is None else data_obj,
                storage_method=('vector_store', _vec,)
            )
        if not hasattr(sent_emb, "sentence_embeddings"):
            raise AttributeError("The provided object is not a SentenceEmbeddings object.")
        else:
            if sent_emb.sentence_embeddings is None:
                raise AttributeError("This SentenceEmbeddings object has no embeddings.")
        return sent_emb

    @classmethod
    def load_clustering(
            cls,
            path: str,
            process: str,
            data_obj: Optional[DataProcessingFactory.DataProcessing] = None,
            emb_obj: Optional[SentenceEmbeddingsFactory.SentenceEmbeddings] = None
    ) -> PhraseClusterFactory.PhraseCluster:
        if emb_obj is not None:
            if not hasattr(emb_obj, "sentence_embeddings"):
                logging.warning("The provided emb_obj seems not to be a SentenceEmbedding object. Trying to load it instead.")
                emb_obj = None
        phrase = PhraseClusterFactory.load(
            cluster_obj_path=Path(Path(path) / f"{process}_clustering.pickle"),
            embedding_obj=cls.load_embedding(path, process, data_obj) if emb_obj is None else emb_obj,
        )
        if not hasattr(phrase, "cluster_center"):
            raise AttributeError("The provided object is not a PhraseCluster object.")
        return phrase

    @classmethod
    def load_graph(
            cls,
            path: str,
            process: str
    ) -> list[nx.Graph]:
        graphs = load_pickle(Path(Path(path) / f"{process}_graph.pickle"))
        if not isinstance(graphs, list):
            raise AttributeError("The provided object doesn't seem to be a list of graphs.")
        else:
            if len(graphs) > 0 and hasattr(graphs[0], "nodes"):
                return graphs
            return []

    @classmethod
    def load_integration(
            cls,
            path: str,
            process: str
    ):
        raise NotImplementedError()

    @classmethod
    def load(
            cls,
            step: str,
            path: str,
            process: str,
            data_obj: Optional[DataProcessingFactory.DataProcessing] = None,
            emb_obj: Optional[SentenceEmbeddingsFactory.SentenceEmbeddings] = None,
            vector_store: Optional[dict] = None,
    ) -> Union[
        DataProcessingFactory.DataProcessing,
        SentenceEmbeddingsFactory.SentenceEmbeddings,
        PhraseClusterFactory.PhraseCluster,
        list[nx.Graph]
    ]:
        if step == StepsName.DATA:
            return cls.load_data(path, process)
        elif step == StepsName.EMBEDDING:
            return cls.load_embedding(path, process, data_obj, vector_store)
        elif step == StepsName.CLUSTERING:
            return cls.load_clustering(path, process, data_obj, emb_obj)
        elif step == StepsName.GRAPH:
            return cls.load_graph(path, process)
        elif step == StepsName.INTEGRATION:
            return cls.load_integration(path, process)
        else:
            raise NotImplementedError
