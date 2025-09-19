from langchain_core.language_models import BaseLLM

from src.rag.chatters.AbstractChatter import Chatter


class OllamaChatter(Chatter):
    base_url = "http://localhost:11434/api"

    @classmethod
    def with_kwargs(cls, **kwargs) -> BaseLLM:
        return
