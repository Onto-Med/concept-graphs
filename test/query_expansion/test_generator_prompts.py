from src.query_expansion.generator import build_generation_prompt
from src.query_expansion.models import LLMConfig, QueryExpansionRequest


def test_build_generation_prompt_uses_german_profile():
    request = QueryExpansionRequest(
        term="Herzinfarkt",
        language="de",
        categories=["synonym", "medication"],
        llm=LLMConfig(model="test-model"),
    )

    prompt = build_generation_prompt(request)

    assert "Erzeuge medizinische Query-Expansion-Kandidaten" in prompt
    assert "Sprache der Kandidaten: Deutsch (de)" in prompt
    assert "Erlaubte Kategorie-IDs" in prompt
    assert "synonym" in prompt
    assert "Medikamente" in prompt
    assert "JSON-Feldnamen MÜSSEN exakt unverändert bleiben" in prompt
    assert '"candidates", "term", "category", "rationale"' in prompt


def test_build_generation_prompt_accepts_request_template_override():
    request = QueryExpansionRequest(
        term="myocardial infarction",
        language="en",
        categories=["synonym"],
        llm=LLMConfig(model="test-model"),
        prompt={
            "template": "Term={term}; language={language}; categories={categories_json}; {schema_instruction}",
            "category_descriptions": {"synonym": "custom synonyms"},
        },
    )

    prompt = build_generation_prompt(request)

    assert "Term=myocardial infarction" in prompt
    assert "language=en" in prompt
    assert "custom synonyms" in prompt
    assert "ExpansionGeneration" in prompt


def test_build_generation_prompt_falls_back_to_english_for_unknown_profile():
    request = QueryExpansionRequest(
        term="infarctus du myocarde",
        language="fr",
        categories=["synonym"],
        llm=LLMConfig(model="test-model"),
    )

    prompt = build_generation_prompt(request)

    assert "Generate medical query-expansion candidates" in prompt
    assert "Candidate language: English (fr)" in prompt
