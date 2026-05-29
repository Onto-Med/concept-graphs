"""Document-related helper functions."""


def add_offset_to_documents_dicts_by_id(
    documents: list[dict], doc_id: str, offset: tuple[int, int]
):
    # ---> _docs = [{"id": doc_id, "offsets": [offset_of_nc_in_doc]}, ...]
    _added_offset = False
    for doc in documents:
        if doc_id == doc.get("id", None):
            doc.get("offsets", []).append(offset)
            _added_offset = True
            break
    if not _added_offset:
        documents.append({"id": doc_id, "offsets": [offset]})
