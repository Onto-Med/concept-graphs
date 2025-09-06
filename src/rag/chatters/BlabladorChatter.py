from langchain_core.language_models import BaseLLM
from langchain_openai import OpenAI
from pydantic import SecretStr

from src.rag.chatters.AbstractChatter import Chatter


class BlabladorChatter(Chatter):
    base_url = "https://api.helmholtz-blablador.fz-juelich.de/v1/"

    @classmethod
    def with_kwargs(cls, **kwargs) -> BaseLLM:
        return OpenAI(
            base_url=kwargs.get("base_url", cls.base_url),
            model=kwargs.get("model", "alias-large"),
            temperature=kwargs.get("temperature", 0.8),
            api_key=SecretStr(kwargs.get("api_key", "")),
        )
