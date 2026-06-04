"""Local file based grounding sources."""

import json
from pathlib import Path
from typing import Any

import yaml

from src.query_expansion.models import GeneratedExpansionCandidate, GroundingEvidence
from src.query_expansion.sources.base import ExpansionSource


class LocalTerminologySource(ExpansionSource):
    """Ground candidates against a small local terminology file.

    Supported file shapes:

    ```yaml
    terms:
      - id: C001
        term: myocardial infarction
        synonyms: [heart attack, MI]
    ```

    or directly:

    ```yaml
    myocardial infarction:
      synonyms: [heart attack, MI]
    ```
    """

    def __init__(self, name: str, path: str | Path):
        """Load a local terminology source from YAML or JSON."""
        self.name = name
        self.path = Path(path)
        self.entries = self._load_entries(self.path)

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.lower().split())

    @classmethod
    def _load_entries(cls, path: Path) -> list[dict[str, Any]]:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
        else:
            data = yaml.safe_load(path.read_text())
        if data is None:
            return []
        if isinstance(data, dict) and isinstance(data.get("terms"), list):
            return data["terms"]
        if isinstance(data, dict):
            return [
                dict({"term": term}, **(payload or {}))
                for term, payload in data.items()
                if isinstance(payload, dict)
            ]
        if isinstance(data, list):
            return [entry for entry in data if isinstance(entry, dict)]
        return []

    @staticmethod
    def _entry_categories(entry: dict[str, Any]) -> set[str]:
        """Return explicit stable category IDs configured for an entry.

        If an entry does not declare a category, it can ground candidates from any
        category for backwards compatibility. If it declares ``category`` or
        ``categories``, only candidates with one of those exact category IDs are
        grounded.
        """
        categories = entry.get("categories", entry.get("category"))
        if categories is None:
            return set()
        if isinstance(categories, str):
            return {categories}
        if isinstance(categories, list):
            return {category for category in categories if isinstance(category, str)}
        return set()

    def ground(self, candidate: GeneratedExpansionCandidate) -> list[GroundingEvidence]:
        """Return exact/synonym matches for a generated candidate."""
        candidate_term = self._normalize(candidate.term)
        evidence = []
        for entry in self.entries:
            entry_categories = self._entry_categories(entry)
            if entry_categories and candidate.category not in entry_categories:
                continue

            terms = [entry.get("term", "")]
            terms.extend(entry.get("synonyms", []) or [])
            normalized_terms = {self._normalize(term): term for term in terms if term}
            if candidate_term in normalized_terms:
                evidence.append(
                    GroundingEvidence(
                        source=self.name,
                        matched_term=normalized_terms[candidate_term],
                        score=1.0,
                        relation="exact_or_synonym",
                        source_id=entry.get("id"),
                        metadata={
                            k: v
                            for k, v in entry.items()
                            if k not in {"id", "term", "synonyms"}
                        },
                    )
                )
        return evidence
