"""Query-expansion category definitions."""

from typing import Literal

ExpansionCategory = Literal[
    "synonym",
    "medication",
    "diagnosis",
    "symptom",
    "procedure",
    "abbreviation",
    "broader_term",
    "narrower_term",
    "related_term",
]

DEFAULT_EXPANSION_CATEGORIES: tuple[ExpansionCategory, ...] = (
    "synonym",
    "medication",
    "diagnosis",
    "symptom",
    "procedure",
)

CATEGORY_DESCRIPTIONS: dict[ExpansionCategory, str] = {
    "synonym": "Synonyms, near-synonyms, spelling variants, or lay terms.",
    "medication": "Medications or drug classes associated with the input term.",
    "diagnosis": "Diagnoses or diagnostic entities associated with the input term.",
    "symptom": "Symptoms, signs, or clinical findings associated with the input term.",
    "procedure": "Procedures, interventions, diagnostics, or treatments associated with the input term.",
    "abbreviation": "Common abbreviations or expanded forms.",
    "broader_term": "Broader parent concepts.",
    "narrower_term": "Narrower child concepts.",
    "related_term": "Other clinically or semantically related terms.",
}
