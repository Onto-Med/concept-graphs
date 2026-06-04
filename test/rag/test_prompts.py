from src.rag.prompts import resolve_rag_prompt_config
from src.rag.rag import _clean_answer


def test_resolve_rag_prompt_config_loads_language_profile():
    config = resolve_rag_prompt_config("de")

    assert "Beantworte die Frage anhand" in config.template
    assert "teilweise Antwort" in config.template
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


def test_clean_answer_removes_reasoning_markup():
    assert (
        _clean_answer(
            "<think>hidden reasoning</think>Finale Antwort auf Deutsch: Beleg [0]."
        )
        == "Beleg [0]."
    )
    assert (
        _clean_answer("Analysis: hidden\nFinal answer: Supported answer [0].")
        == "Supported answer [0]."
    )


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
