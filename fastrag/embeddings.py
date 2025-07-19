from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

class AbstractEmbeddingsModel(ABC):
    """
    Abstract base class for embedding models.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embeddings.
        """
        pass

    @abstractmethod
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously embeds a list of texts.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embeddings.
        """
        pass

class SentenceTransformerEmbeddings(AbstractEmbeddingsModel):
    """
    An embedding model that uses the Sentence Transformers library.

    Args:
        model_name: The name of the Sentence Transformers model to use.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    @lru_cache(maxsize=128)
    async def aembed_documents(self, texts_tuple: Tuple[str, ...]) -> List[List[float]]:
        """
        Asynchronously embeds a tuple of texts.

        Args:
            texts_tuple: A tuple of texts to embed.

        Returns:
            A list of embeddings.
        """
        import asyncio
        return await asyncio.to_thread(self.model.encode, list(texts_tuple)).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embeddings.
        """
        import asyncio
        return asyncio.run(self.aembed_documents(tuple(texts)))
