"""Project-specific spaCy extension registration."""


def set_spacy_extensions() -> None:
    from spacy.tokens import Doc

    if not Doc.has_extension("doc_id"):
        Doc.set_extension("doc_id", default=None)
    if not Doc.has_extension("doc_index"):
        Doc.set_extension("doc_index", default=None)
    if not Doc.has_extension("doc_name"):
        Doc.set_extension("doc_name", default=None)
    if not Doc.has_extension("doc_topic"):
        Doc.set_extension("doc_topic", default=None)
    if not Doc.has_extension("offset_in_doc"):
        Doc.set_extension("offset_in_doc", default=None)
