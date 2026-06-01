from src.api.context import ActiveRAG, RagContext
from src.rag.rag import RAG


class DummyVectorStore:
    pass


def test_rag_context_keeps_active_rag_per_process():
    rag_context = RagContext(active_by_process={})
    process_a = ActiveRAG(
        rag=RAG.with_chatter(chatter="src.rag.chatters.blablador.BlabladorChatter"),
        vectorstore=DummyVectorStore(),
        process="a",
    )
    process_b = ActiveRAG(
        rag=RAG.with_chatter(chatter="src.rag.chatters.blablador.BlabladorChatter"),
        vectorstore=DummyVectorStore(),
        process="b",
    )

    rag_context.active_by_process["a"] = process_a
    rag_context.active_by_process["b"] = process_b

    assert rag_context.active_by_process["a"] is process_a
    assert rag_context.active_by_process["b"] is process_b
    assert (
        rag_context.active_by_process["a"].rag
        is not rag_context.active_by_process["b"].rag
    )


def test_rag_documents_from_does_not_mutate_rag_documents():
    rag = RAG.with_chatter(chatter="src.rag.chatters.blablador.BlabladorChatter")

    documents = rag.documents_from([("content", {"doc_id": "doc-a"})])

    assert documents["[0]"].metadata == {"doc_id": "doc-a"}
    assert rag.documents is None
