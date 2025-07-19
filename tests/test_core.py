import os
import unittest
from unittest.mock import MagicMock, patch
import shutil

from fastrag.core import FastRAG
from fastrag.data_sources import Document
from fastrag.vector_stores import ChromaVectorStore

class TestFastRAG(unittest.TestCase):
    def setUp(self):
        # Create a dummy data source file
        self.data_source_path = "./test_data"
        os.makedirs(self.data_source_path, exist_ok=True)
        with open(os.path.join(self.data_source_path, "test.txt"), "w") as f:
            f.write("This is a test document.")

        # Setup for ChromaDB test
        self.chroma_db_path = "./test_chroma_db"

    def tearDown(self):
        # Clean up the dummy data source file
        os.remove(os.path.join(self.data_source_path, "test.txt"))
        os.rmdir(self.data_source_path)

        # Clean up ChromaDB directory
        if os.path.exists(self.chroma_db_path):
            shutil.rmtree(self.chroma_db_path)

    @patch('fastrag.core.LangChainLLM')
    def test_build_index_and_query(self, MockLangChainLLM):
        # Create mock objects
        mock_embedding_model = MagicMock()
        mock_vector_store = MagicMock()
        mock_llm_instance = MockLangChainLLM.return_value

        # Configure mock return values
        mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_llm_instance.generate.return_value = "This is the answer."
        mock_vector_store.search.return_value = [Document(content="This is a test document.", metadata={})]

        # Initialize FastRAG with mocked dependencies
        rag = FastRAG(
            data_source_path=self.data_source_path,
            embedding_model=mock_embedding_model,
            llm="openai:gpt-3.5-turbo",
            vector_store=mock_vector_store,
        )
        rag.llm = mock_llm_instance

        # Build the index
        rag.build_index()

        # Assert that the vector store's add_documents method was called
        mock_vector_store.add_documents.assert_called_once()

        # Query the RAG system
        result = rag.query("What is this?")

        # Assertions
        self.assertEqual(result["answer"], "This is the answer.")
        self.assertIsInstance(result["sources"], list)
        self.assertIsInstance(result["sources"][0], Document)
        mock_vector_store.search.assert_called_once()
        mock_llm_instance.generate.assert_called_once()

    @patch('fastrag.core.LangChainLLM')
    def test_fastrag_with_chromadb(self, MockLangChainLLM):
        # Create mock objects
        mock_embedding_model = MagicMock()
        mock_llm_instance = MockLangChainLLM.return_value

        # Configure mock return values
        mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_llm_instance.generate.return_value = "This is the answer."

        # Initialize FastRAG with ChromaVectorStore
        rag = FastRAG(
            data_source_path=self.data_source_path,
            embedding_model=mock_embedding_model,
            llm="openai:gpt-3.5-turbo",
            vector_store=self.chroma_vector_store,
        )
        rag.llm = mock_llm_instance

        # Build the index
        rag.build_index()

        # Query the RAG system
        result = rag.query("What is this?")

        # Assertions
        self.assertEqual(result["answer"], "This is the answer.")
        self.assertIsInstance(result["sources"], list)
        self.assertIsInstance(result["sources"][0], Document)
        mock_llm_instance.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
