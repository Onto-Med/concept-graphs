from typing import Optional

from langchain_core.language_models import BaseLLM
from langchain_ollama import OllamaLLM

from src.rag.chatters.AbstractChatter import Chatter


class OllamaChatter(Chatter):
    base_url = "http://localhost:11434"

    @classmethod
    def with_kwargs(cls, **kwargs) -> Optional[BaseLLM]:
        if _model := kwargs.get("model", None):
            return OllamaLLM(
                base_url=kwargs.get("base_url", cls.base_url),
                model=_model,
                temperature=kwargs.get("temperature", 0.5),
                # max_tokens=-1
            )
        return None
