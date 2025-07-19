import faiss
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
import chromadb

from fastrag.data_sources import Document

class AbstractVectorStore(ABC):
    """
    Abstract base class for vector stores.
    """

    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """
        Adds documents and their embeddings to the vector store.

        Args:
            documents: A list of documents to add.
            embeddings: A list of embeddings corresponding to the documents.
        """
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], k: int) -> List[Document]:
        """
        Searches for the most similar documents to a query embedding.

        Args:
            query_embedding: The embedding of the query.
            k: The number of documents to return.

        Returns:
            A list of the most similar documents.
        """
        pass

    @abstractmethod
    def save_index(self, path: str):
        """
        Saves the index to a file.

        Args:
            path: The path to save the index to.
        """
        pass

    @abstractmethod
    def load_index(self, path: str):
        """
        Loads the index from a file.

        Args:
            path: The path to load the index from.
        """
        pass

class FAISSVectorStore(AbstractVectorStore):
    """
    A vector store that uses FAISS for efficient similarity search.
    """

    def __init__(self):
        self.index: faiss.IndexFlatL2 | None = None
        self.documents: List[Document] = []
        """
        Initializes the FAISS vector store.

        Attributes:
            index: The FAISS index for storing embeddings.
            documents: A list of Document objects corresponding to the embeddings.
        """

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """
        Adds documents and their embeddings to the vector store.

        Args:
            documents: A list of documents to add.
            embeddings: A list of embeddings corresponding to the documents.
        """
        if not embeddings:
            return # No embeddings to add

        if self.index is None:
            dimension = len(embeddings[0])
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))
        self.documents.extend(documents)

    def search(self, query_embedding: List[float], k: int) -> List[Document]:
        """
        Searches for the most similar documents to a query embedding.

        Args:
            query_embedding: The embedding of the query.
            k: The number of documents to return.

        Returns:
            A list of the most similar documents.
        """
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [self.documents[i] for i in indices[0]]

    def save_index(self, path: str):
        """
        Saves the index to a file.

        Args:
            path: The path to save the index to.
        """
        faiss.write_index(self.index, path)

    def load_index(self, path: str):
        """
        Loads the index from a file.

        Args:
            path: The path to load the index from.
        """
        self.index = faiss.read_index(path)


class ChromaVectorStore(AbstractVectorStore):
    """
    A vector store that uses ChromaDB.
    """

    def __init__(self, path: str = "./chroma_db", collection_name: str = "fastrag_collection"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """
        Adds documents and their embeddings to the vector store.

        Args:
            documents: A list of documents to add.
            embeddings: A list of embeddings corresponding to the documents.
        """
        if not documents or not embeddings:
            return

        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = [doc.metadata for doc in documents]
        contents = [doc.content for doc in documents]

        self.collection.add(
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query_embedding: List[float], k: int) -> List[Document]:
        """
        Searches for the most similar documents to a query embedding.

        Args:
            query_embedding: The embedding of the query.
            k: The number of documents to return.

        Returns:
            A list of the most similar documents.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas']
        )
        retrieved_docs = []
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                content = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                retrieved_docs.append(Document(content=content, metadata=metadata))
        return retrieved_docs

    def save_index(self, path: str):
        """
        ChromaDB automatically persists data to the path provided during initialization.
        This method is a placeholder for compatibility with the AbstractVectorStore interface.
        """
        print(f"ChromaDB data is persisted at: {self.client.path}")

    def load_index(self, path: str):
        """
        ChromaDB automatically loads data from the path provided during initialization.
        This method is a placeholder for compatibility with the AbstractVectorStore interface.
        """
        print(f"ChromaDB data is loaded from: {self.client.path}")