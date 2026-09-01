from src.rag.chatters.blablador import BlabladorChatter


def test_blablador_chatter_forwards_openai_extra_body(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("src.rag.chatters.blablador.ChatOpenAI", FakeChatOpenAI)

    BlabladorChatter.with_kwargs(
        api_key="token",
        model="alias-fast",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
