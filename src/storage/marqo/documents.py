from typing import Optional, Union

import numpy as np

from src.storage.interfaces import Document


class MarqoDocument(Document):
    def __init__(
        self,
        phrases: list[str],
        embeddings: list[Union[list, np.ndarray]],
        doc_id: Optional[str] = None,
    ):
        if len(phrases) != len(embeddings):
            raise ValueError(
                f"Phrases (len={len(phrases)}) and embeddings (len={len(embeddings)}) must have same length"
            )
        self._id = doc_id
        self._embeddings = embeddings
        self._phrases = phrases

    @property
    def id(self) -> Optional[str]:
        return self._id

    @property
    def embeddings(self) -> list[Union[np.ndarray, list]]:
        return self._embeddings

    @property
    def phrases(self) -> list[str]:
        return self._phrases

    @property
    def as_tuples(self) -> list[tuple[str, Union[list, np.ndarray]]]:
        return list(zip(self.phrases, self.embeddings))
