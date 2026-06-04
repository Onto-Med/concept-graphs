from src.rag.prompts import resolve_rag_prompt_config


def test_resolve_rag_prompt_config_loads_language_profile():
    config = resolve_rag_prompt_config("de")

    assert "Beantworte die Frage ausschließlich anhand" in config.template
    assert "FRAGE:\n{question}" in config.template
    assert config.input_variables == ["summaries", "question"]


def test_resolve_rag_prompt_config_keeps_legacy_template_override():
    config = resolve_rag_prompt_config(
        "de",
        {
            "templates": {"de": "Legacy {question} {summaries}"},
            "input_variables": ["question", "summaries"],
        },
    )

    assert config.template == "Legacy {question} {summaries}"
    assert config.input_variables == ["question", "summaries"]


def test_resolve_rag_prompt_config_supports_direct_template_override():
    config = resolve_rag_prompt_config(
        "en",
        {
            "profile": "de",
            "template": "Custom {question} {summaries}",
        },
    )

    assert config.template == "Custom {question} {summaries}"
    assert config.input_variables == ["summaries", "question"]
