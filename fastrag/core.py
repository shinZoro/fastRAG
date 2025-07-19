from typing import List, Optional

from fastrag.chunking import TextSplitter, RecursiveCharacterTextSplitter
from fastrag.data_sources import Document, load_documents
from fastrag.embeddings import AbstractEmbeddingsModel, SentenceTransformerEmbeddings
from fastrag.llm_integrations import LangChainLLM
from fastrag.vector_stores import AbstractVectorStore, FAISSVectorStore

class FastRAG:
    """
    A developer-friendly SDK for building Retrieval-Augmented Generation (RAG) systems.

    This class provides a streamlined interface for ingesting data, building a searchable
    index, and querying Large Language Models (LLMs) with retrieved context.
    """

    def __init__(
        self,
        data_source_path: str,
        embedding_model: Optional[AbstractEmbeddingsModel] = None,
        llm: str = "openai:gpt-3.5-turbo",
        text_splitter: Optional[TextSplitter] = None,
        vector_store: Optional[AbstractVectorStore] = None,
        prompt_template: Optional[str] = None,
        **llm_kwargs,
    ):
        self.data_source_path = data_source_path
        self.embedding_model = embedding_model or SentenceTransformerEmbeddings()
        self.llm = LangChainLLM(model=llm, **llm_kwargs)
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter()
        self.vector_store = vector_store or FAISSVectorStore()
        self.prompt_template = prompt_template or "        prompt = self.prompt_template.format(context=context, query=query_text)"

    def set_vector_store(self, vector_store: AbstractVectorStore):
        """
        Sets the vector store for the FastRAG instance.

        Args:
            vector_store: An instance of a class inheriting from AbstractVectorStore.
        """
        self.vector_store = vector_store

    async def abuild_index(self):
        """
        Asynchronously builds the RAG index from the data source.
        """
        documents = load_documents(self.data_source_path)
        chunked_documents = []
        for doc in documents:
            split_texts = self.text_splitter.split_text(doc.content)
            for text in split_texts:
                chunked_documents.append(Document(content=text, metadata=doc.metadata))
        embeddings = await self.embedding_model.aembed_documents(tuple([doc.content for doc in chunked_documents]))
        await self.vector_store.aadd_documents(chunked_documents, embeddings)

    def build_index(self):
        """
        Builds the RAG index from the data source.
        """
        import asyncio
        asyncio.run(self.abuild_index())

    async def aquery(self, query_text: str, k: int = 5) -> dict:
        """
        Asynchronously queries the RAG system.

        Args:
            query_text: The query text.
            k: The number of documents to retrieve.

        Returns:
            A dictionary containing the answer and the sources.
        """
        query_embedding = (await self.embedding_model.aembed_documents([query_text]))[0]
        retrieved_docs = await self.vector_store.asearch(query_embedding, k)
        context = "\n".join([doc.content for doc in retrieved_docs])
        prompt = self.prompt_template.format(context=context, query=query_text)
        answer = await self.llm.agenerate(prompt)
        return {"answer": answer, "sources": retrieved_docs}

    def query(self, query_text: str, k: int = 5) -> dict:
        """
        Queries the RAG system.

        Args:
            query_text: The query text.
            k: The number of documents to retrieve.

        Returns:
            A dictionary containing the answer and the sources.
        """
        import asyncio
        return asyncio.run(self.aquery(query_text, k))
