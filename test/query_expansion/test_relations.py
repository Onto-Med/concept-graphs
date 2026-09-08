from src.query_expansion.models import ExpansionConcept, ExpansionSemanticRelation
from src.query_expansion.relations import MEDICAL_RELATION_DEFINITIONS


def test_medical_relations_are_semantic_not_search_engine_methods():
    relation = next(
        item for item in MEDICAL_RELATION_DEFINITIONS if item.id == "may_indicate"
    )

    assert relation.source_categories == ("symptom",)
    assert relation.target_categories == ("diagnosis",)
    assert "search" not in relation.model_dump()


def test_investigated_by_and_confirmed_by_have_distinct_directions():
    investigated_by = next(
        item for item in MEDICAL_RELATION_DEFINITIONS if item.id == "investigated_by"
    )
    confirmed_by = next(
        item for item in MEDICAL_RELATION_DEFINITIONS if item.id == "confirmed_by"
    )

    assert investigated_by.source_categories == ("symptom",)
    assert investigated_by.target_categories == ("procedure",)
    assert confirmed_by.source_categories == ("diagnosis",)
    assert confirmed_by.target_categories == ("procedure",)


def test_query_expansion_relation_output_is_backend_neutral_json():
    concept = ExpansionConcept(
        id="abdominal_pain",
        label="abdominal pain",
        category="symptom",
        terms=["abdominal pain", "stomach ache"],
    )
    relation = ExpansionSemanticRelation(
        source_concept_id="abdominal_pain",
        relation="may_indicate",
        target_concept_id="appendicitis",
        confidence=0.8,
    )

    assert concept.model_dump(mode="json")["terms"] == [
        "abdominal pain",
        "stomach ache",
    ]
    assert relation.model_dump(mode="json") == {
        "source_concept_id": "abdominal_pain",
        "relation": "may_indicate",
        "target_concept_id": "appendicitis",
        "confidence": 0.8,
        "evidence": [],
        "metadata": {},
    }
