from abc import ABC, abstractmethod
from functools import lru_cache
from langchain.chat_models.base import init_chat_model
from langchain_core.messages import HumanMessage


class AbstractLLM(ABC):
    """
    Abstract base class for language models.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generates a response to a prompt.

        Args:
            prompt: The prompt to generate a response to.

        Returns:
            The generated response.
        """
        pass

    @abstractmethod
    async def agenerate(self, prompt: str) -> str:
        """
        Asynchronously generates a response to a prompt.

        Args:
            prompt: The prompt to generate a response to.

        Returns:
            The generated response.
        """
        pass


class LangChainLLM(AbstractLLM):
    """
    A language model that uses the LangChain library to initialize a chat model.

    Args:
        model_name: The name of the model to initialize.
        model_provider: The provider of the model.
    """

    def __init__(self, model: str, **kwargs):
        self.model = init_chat_model(model, **kwargs)

    @lru_cache(maxsize=128)
    async def agenerate(self, prompt: str) -> str:
        """
        Asynchronously generates a response to a prompt.

        Args:
            prompt: The prompt to generate a response to.

        Returns:
            The generated response.
        """
        response = await self.model.ainvoke([HumanMessage(content=prompt)])
        return response.content

    def generate(self, prompt: str) -> str:
        """
        Generates a response to a prompt.

        Args:
            prompt: The prompt to generate a response to.

        Returns:
            The generated response.
        """
        import asyncio
        return asyncio.run(self.agenerate(prompt))