from abc import ABC, abstractmethod
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

class SentenceTransformerEmbeddings(AbstractEmbeddingsModel):
    """
    An embedding model that uses the Sentence Transformers library.

    Args:
        model_name: The name of the Sentence Transformers model to use.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embeddings.
        """
        return self.model.encode(texts).tolist()
