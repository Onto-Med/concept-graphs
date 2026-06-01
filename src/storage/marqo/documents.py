import numpy as np

from src.storage.interfaces import Document


class MarqoDocument(Document):
    def __init__(
        self,
        phrases: list[str],
        embeddings: list[list | np.ndarray],
        doc_id: str | None = None,
    ):
        if len(phrases) != len(embeddings):
            raise ValueError(
                f"Phrases (len={len(phrases)}) and embeddings (len={len(embeddings)}) must have same length"
            )
        self._id = doc_id
        self._embeddings = embeddings
        self._phrases = phrases

    @property
    def id(self) -> str | None:
        return self._id

    @property
    def embeddings(self) -> list[np.ndarray | list]:
        return self._embeddings

    @property
    def phrases(self) -> list[str]:
        return self._phrases

    @property
    def as_tuples(self) -> list[tuple[str, list | np.ndarray]]:
        return list(zip(self.phrases, self.embeddings))
