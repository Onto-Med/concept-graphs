import yaml

from src.query_expansion.categories import ALL_EXPANSION_CATEGORIES
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
    assert '"candidates", "term", "category", "rationale", "concepts", "relations"' in prompt
    assert "equivalent_to" in prompt


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


def test_builtin_domain_profiles_cover_fallback_medical_vocabulary():
    expected_categories = set(ALL_EXPANSION_CATEGORIES)
    for profile_path in (
        "conf/query-expansion/profiles/medical-en.yml",
        "conf/query-expansion/profiles/medical-de.yml",
    ):
        with open(profile_path, encoding="utf-8") as file:
            profile = yaml.safe_load(file)

        assert set(profile["category_descriptions"]) == expected_categories
        assert set(profile["default_categories"]).issubset(expected_categories)


def test_build_generation_prompt_accepts_profile_defined_categories(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "custom.yml").write_text(
        """
language_name: Custom
category_descriptions:
  lab_value: Laboratory values associated with the input term.
default_categories:
  - lab_value
prompt_template: |
  Categories: {categories_json}
  {schema_instruction}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.query_expansion.prompts.DEFAULT_PROMPT_DIR", profile_dir)
    request = QueryExpansionRequest(
        term="CRP",
        language="custom",
        categories=["lab_value"],
        llm=LLMConfig(model="test-model"),
    )

    prompt = build_generation_prompt(request)

    assert "lab_value" in prompt
    assert "Laboratory values" in prompt


def test_build_generation_prompt_uses_profile_default_categories(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "custom.yml").write_text(
        """
language_name: Custom
category_descriptions:
  lab_value: Laboratory values associated with the input term.
default_categories:
  - lab_value
prompt_template: |
  Categories: {categories_json}
  {schema_instruction}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.query_expansion.prompts.DEFAULT_PROMPT_DIR", profile_dir)
    request = QueryExpansionRequest(
        term="CRP",
        language="custom",
        llm=LLMConfig(model="test-model"),
    )

    prompt = build_generation_prompt(request)

    assert "lab_value" in prompt
    assert "synonym" not in prompt


def test_build_generation_prompt_rejects_categories_outside_profile(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "custom.yml").write_text(
        """
language_name: Custom
category_descriptions:
  lab_value: Laboratory values associated with the input term.
default_categories:
  - lab_value
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.query_expansion.prompts.DEFAULT_PROMPT_DIR", profile_dir)
    request = QueryExpansionRequest(
        term="CRP",
        language="custom",
        categories=["diagnosis"],
        llm=LLMConfig(model="test-model"),
    )

    try:
        build_generation_prompt(request)
    except ValueError as exc:
        assert "diagnosis" in str(exc)
    else:
        raise AssertionError("Expected profile category validation to fail")


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
