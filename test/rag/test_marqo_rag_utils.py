from src.rag.marqo_rag_utils import extract_text_from_highlights


def test_extract_text_from_highlights_adds_snippet_offsets_and_metadata():
    highlights, texts, metadata = extract_text_from_highlights(
        [
            {
                "text": "Alpha beta inflammation gamma.",
                "doc_id": "doc-1",
                "doc_name": "doc.txt",
                "chunk_index": 2,
                "chunk_start": 100,
                "chunk_end": 130,
                "_highlights": [{"text": "inflammation"}],
                "_score": 0.9,
            }
        ],
        truncate=False,
    )

    assert highlights == ["inflammation"]
    assert texts == ["Alpha beta inflammation gamma."]
    assert metadata == [
        {
            "doc_id": "doc-1",
            "doc_name": "doc.txt",
            "chunk_index": 2,
            "chunk_start": 100,
            "chunk_end": 130,
            "retrieved_snippet": "Alpha beta inflammation gamma.",
            "retrieved_snippet_start": 0,
            "retrieved_snippet_end": 30,
            "highlight": "inflammation",
            "highlight_field": "text",
            "highlight_start": 11,
            "highlight_end": 23,
            "offset_unit": "retrieved_snippet_char",
            "document_highlight_start": 111,
            "document_highlight_end": 123,
            "document_offset_unit": "document_char",
            "retrieved_snippet_index": 0,
        }
    ]


def test_extract_text_from_highlights_handles_empty_highlights():
    highlights, texts, metadata = extract_text_from_highlights(
        [
            {
                "text": "Alpha beta inflammation gamma.",
                "doc_id": "doc-1",
                "chunk_start": 100,
                "_highlights": [],
            }
        ],
        truncate=False,
    )

    assert highlights == [""]
    assert texts == ["Alpha beta inflammation gamma."]
    assert metadata[0]["highlight"] == ""
    assert metadata[0]["highlight_field"] == "text"
    assert metadata[0]["highlight_start"] is None
    assert metadata[0]["highlight_end"] is None
    assert metadata[0]["document_highlight_start"] is None
    assert metadata[0]["document_highlight_end"] is None
