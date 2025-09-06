from pydoc import locate
from typing import Union

from langchain_core.language_models import BaseLLM
from langchain_text_splitters import

from chatters.AbstractChatter import Chatter

class RAG:
    def __init__(self, chatter: Union[Chatter, str] = "src.rag.chatters.BlabladorChatter.BlabladorChatter"):
        if isinstance(chatter, str):
            self._chatter = locate(chatter)
        elif isinstance(chatter, Chatter):
            self._chatter = chatter
        else:
            raise TypeError(f"'chatter' must be an implementation of the Chatter class, or a string denoting the location of said class.")

    def build_chatter(self, api_key: str, **kwargs) -> BaseLLM:
        if not "api_key" in kwargs:
            kwargs["api_key"] = api_key
        return self._chatter.with_kwargs(**kwargs)

if __name__ == "__main__":
    llm = RAG().build_chatter(api_key="")
    print(llm)