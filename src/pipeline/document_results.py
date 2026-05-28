"""Document addition result transformation helpers."""

from typing import Iterator


def transform_document_addition_results(iterator: Iterator):
    phrases_dict = dict()
    type_list = ["added", "incorporated"]
    _additional_info_key = "additional_info"
    _phrases_key = "phrases"
    _graph_field_key = "graph_cluster"
    _phrase_id_key = "_id"
    _text_key = "text"
    _offsets_key = "offsets"

    for doc_id, graph_dict in iterator:
        for _type in type_list:
            for _phrase, _additional_info in zip(
                graph_dict.get(_type, {}).get(_phrases_key, []),
                graph_dict.get(_type, {}).get(_additional_info_key, []),
            ):
                if _phrase_id := _phrase.get(_phrase_id_key, None):
                    if _phrase_id not in phrases_dict:
                        phrases_dict[_phrase_id] = {
                            "graph": _phrase.get(_graph_field_key, [None])[0],
                            "id": _phrase_id,
                            "documents": list(),
                            "label": _additional_info.get(_text_key, ""),
                        }
                    phrases_dict[_phrase_id]["documents"].append(
                        {
                            "id": doc_id,
                            "offsets": _additional_info.get(_offsets_key, []),
                        }
                    )
    return phrases_dict
