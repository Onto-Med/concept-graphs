import abc

from langchain_core.language_models import BaseLLM


class Chatter(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def with_kwargs(cls, **kwargs) -> BaseLLM:
        raise NotImplementedError
