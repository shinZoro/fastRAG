from abc import ABC, abstractmethod
from typing import List

class TextSplitter(ABC):
    """
    Abstract base class for text splitters.
    """

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Splits a text into chunks.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks.
        """
        pass

class RecursiveCharacterTextSplitter(TextSplitter):
    """
    A simple text splitter that splits text by recursively trying different separators.

    Args:
        chunk_size: The maximum size of a chunk.
        chunk_overlap: The overlap between chunks.
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Splits a text into chunks.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks
