# FastRAG - Developer-Friendly Retrieval-Augmented Generation SDK

**Core Idea:** To create a Python SDK that simplifies the integration of RAG into other applications, providing a clean, efficient, and extensible API for developers. It will prioritize speed and ease of use for developers building RAG-powered features.

**Target Audience:** Other developers, particularly those building LLM-powered applications, chatbots, or knowledge retrieval systems who want to easily incorporate RAG without deep diving into vector databases or complex pipeline orchestration.

---

## Phase 1: Core Functionality (MVP)

**Goal:** Establish the fundamental RAG pipeline as a Python SDK with a clean API for basic data ingestion, indexing, and querying.

### Step 1.0: Project Setup & Initial Structure

* **1.1.0 Initialize Git Repository:**
    * `git init`
    * Create `.gitignore` (e.g., for `.venv/`, `__pycache__/`, `*.pyc`, `.env`, index files)
* **1.1.1 Create Project Directory Structure:**
    ```
    fastrag/
    ├── fastrag/
    │   ├── __init__.py          # Package initializer
    │   ├── core.py              # Main FastRAG class
    │   ├── data_sources.py      # Data source handling (local files initially)
    │   ├── chunking.py          # Text chunking logic
    │   ├── embeddings.py        # Embedding model interface & implementations
    │   ├── vector_stores.py     # Vector store interface & FAISS implementation
    │   └── llm_integrations.py  # LLM API interaction
    ├── tests/
    │   └── test_core.py         # Basic tests
    ├── docs/
    │   └── README.md            # Project description (this file!)
    ├── pyproject.toml           # Project metadata & dependencies (modern standard)
    └── requirements.txt         # For initial dependency tracking (can be removed later if pyproject.toml is comprehensive)
    ```
* **1.1.2 Set up `pyproject.toml`:**
    * Define project name (`fastrag`), version (`0.1.0`), author, license.
    * Specify core dependencies (e.g., `sentence-transformers`, `faiss-cpu`, `openai` or `ollama-python`, `tiktoken`).
    * Define development dependencies (e.g., `pytest`).
* **1.1.3 Create Virtual Environment:**
    * `python -m venv .venv`
    * Activate: `. .venv/bin/activate` (Linux/macOS) or `.\.venv\Scripts\activate` (Windows PowerShell)
* **1.1.4 Install Development Dependencies:**
    * `pip install -e .` (installs `fastrag` in editable mode)
    * `pip install pytest`

### Step 2.0: Data Ingestion & Chunking

* **2.1.0 Implement Basic Document Loading (`fastrag/data_sources.py`):**
    * Create a `Document` dataclass: `content: str`, `metadata: dict`.
    * Implement functions to read text content from local files (`.txt`, `.md`).
    * (Future: Add simple PDF parsing using `pypdf` or `PyMuPDF`).
* **2.2.0 Implement Simple Text Splitter (`fastrag/chunking.py`):**
    * Create a `TextSplitter` abstract base class.
    * Implement `RecursiveCharacterTextSplitter` (or a basic fixed-size splitter) as a concrete class.

### Step 3.0: Embedding Generation

* **3.1.0 Define Embedding Model Interface (`fastrag/embeddings.py`):**
    * Create an `AbstractEmbeddingsModel` class.
    * Define abstract method `embed_documents(texts: list[str]) -> list[list[float]]`.
* **3.2.0 Implement Sentence Transformers Integration:**
    * Create `SentenceTransformerEmbeddings` class inheriting from `AbstractEmbeddingsModel`.
    * Load and use a model (e.g., `"sentence-transformers/all-MiniLM-L6-v2"`).
    * Make the model name configurable in its constructor.

### Step 4.0: Vector Store (FAISS MVP)

* **4.1.0 Define Vector Store Interface (`fastrag/vector_stores.py`):**
    * Create an `AbstractVectorStore` class.
    * Define methods:jjjsssssssssddddd
        * `add_documents(documents: list[Document], embeddings: list[list[float]])`
        * `search(query_embedding: list[float], k: int) -> list[Document]`
        * `save_index(path: str)`
        * `load_index(path: str)`
* **4.2.0 Implement FAISS Vector Store:**
    * Create `FAISSVectorStore` inheriting from `AbstractVectorStore`.
    * Utilize `faiss-cpu` for in-memory index creation, adding vectors, and searching.

### Step 5.0: LLM Integration

* **5.1.0 Define LLM Interface (`fastrag/llm_integrations.py`):**
    * Create an `AbstractLLM` class.
    * Define abstract method `generate(prompt: str) -> str`.
* **5.2.0 Implement OpenAI/Ollama LLM:**
    * Create `OpenAILLM` and/or `OllamaLLM` concrete implementations.
    * Handle API key loading for OpenAI (from env var) or endpoint configuration for Ollama.

### Step 6.0: Core `FastRAG` Class

* **6.1.0 Design `FastRAG` Constructor (`fastrag/core.py`):**
    * Parameters: `data_source_path: str`, `embedding_model: AbstractEmbeddingsModel`, `llm: AbstractLLM`, `text_splitter: TextSplitter`, `vector_store: AbstractVectorStore`. (Initial `FastRAG` can instantiate defaults if not provided).
* **6.2.0 Implement `build_index()`:**
    * Load raw documents from `data_source_path`.
    * Chunk documents using `text_splitter`.
    * Generate embeddings for chunks using `embedding_model`.
    * Add embeddings and original document content/metadata to `vector_store`.
* **6.3.0 Implement `query(query_text: str)`:**
    * Generate embedding for the `query_text`.
    * Search `vector_store` for top-k relevant documents.
    * Construct an augmented prompt (query + retrieved context).
    * Call the `llm` to generate the answer.
    * Return a structured `FastRAGResponse` object (e.g., `answer: str`, `sources: list[Document]`).
* **6.4.0 Basic Prompt Template:** Hardcode a simple prompt template initially within the `query` method (e.g., "Use the following context to answer the question: {context}\nQuestion: {query}\nAnswer:").

### Step 7.0: Initial Testing & Documentation

* **7.1.0 Write Unit Tests (`tests/test_*.py`):**
    * For `TextSplitter`, `EmbeddingsModel`, `VectorStore`, `LLM` interfaces and their default implementations.
* **7.2.0 Write Integration Test:**
    * A simple end-to-end test for `FastRAG.build_index()` and `FastRAG.query()`.
* **7.3.0 Update `README.md`:**
    * Basic project description.
    * Installation instructions (`pip install fastrag`).
    * Simple usage examples for `FastRAG` class.

---

## Phase 2: Performance & Extensibility (Intermediate)

**Goal:** Make the SDK "Fast" and more configurable, allowing developers more control and flexibility.

### Step 8.0: Pluggable Vector Stores

* **8.1.0 Implement ChromaDB Integration:**
    * Add `chromadb` to `pyproject.toml` (as an optional dependency or core if decided).
    * Create `ChromaVectorStore` in `fastrag/vector_stores.py`, implementing `AbstractVectorStore`.
    * Allow `FastRAG` constructor to explicitly accept a `vector_store` instance.
* **8.2.0 Vector Store Persistence:**
    * Ensure `save_index(path)` and `load_index(path)` methods are fully implemented for both `FAISSVectorStore` and `ChromaVectorStore`.

### Step 9.0: Customizable Chunking Strategies

* **9.1.0 Add More Text Splitters:**
    * Implement `MarkdownTextSplitter`, `HTMLTextSplitter` (requires `bs4`).
    * Allow `FastRAG` constructor to accept an `AbstractTextSplitter` instance.
* **9.2.0 Configuration for Chunking:** Expose `chunk_size` and `chunk_overlap` parameters via `TextSplitter` constructors.

### Step 10.0: Asynchronous API Support

* **10.1.0 Convert Core Methods to `async`:**
    * Modify `FastRAG.build_index()` and `FastRAG.query()` (and underlying LLM/embedding calls) to be `async def`.
    * Ensure all internal I/O operations (LLM calls, embedding generation, vector store operations) are `await`able.
* **10.2.0 Provide Synchronous Wrappers:**
    * For developer convenience, offer `build_index_sync()` and `query_sync()` methods that internally use `asyncio.run()`.

### Step 11.0: Caching Mechanism

* **11.1.0 Implement In-Memory Caching:**
    * Use `functools.lru_cache` or a custom dictionary-based cache for `embedding_model.embed_documents` (for identical text inputs).
    * Consider a simple cache for `LLM.generate` based on prompt hash.
* **11.2.0 Make Caching Configurable:** Add a `cache_enabled: bool` parameter to `FastRAG` constructor.

### Step 12.0: Metadata Handling & Filtering

* **12.1.0 Enhance `Document` Class:** Confirm `Document` dataclass can store arbitrary metadata (`dict`).
* **12.2.0 Pass Metadata to Vector Store:** Ensure `vector_store.add_documents` stores metadata alongside embeddings.
* **12.3.0 Implement Metadata Filtering in `search()`:**
    * Modify `AbstractVectorStore.search` method to accept `metadata_filters: dict`.
    * Implement filtering logic within concrete vector store implementations (FAISS will require manual filtering after retrieval, others like ChromaDB can do it natively).

### Step 13.0: Prompt Templating

* **13.1.0 Allow Custom Prompt Templates:**
    * Add a `prompt_template: str` parameter to `FastRAG` constructor or `query` method.
    * Use Python's f-strings or `string.Template` for easy injection of `{query}` and `{context}`.

---

## Phase 3: Advanced Features & Refinements (Polishing)

**Goal:** Add more sophisticated RAG techniques, robust error handling, and comprehensive documentation for production readiness.

### Step 14.0: Query Re-ranking

* **14.1.0 Implement Re-ranker Interface:**
    * Define an `AbstractReranker` class with a `rerank(query: str, documents: list[Document]) -> list[Document]` method.
* **14.2.0 Integrate a Cross-Encoder Model:**
    * Add `CrossEncoderReranker` implementation using `sentence-transformers` (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).
* **14.3.0 Integrate into `FastRAG.query()`:** Make re-ranking optional via a `reranker: AbstractReranker` parameter in `FastRAG` constructor.

### Step 15.0: Query Transformation/Expansion

* **15.1.0 Implement Query Transformer Interface:**
    * An `AbstractQueryTransformer` with `transform_query(query: str) -> str | list[str]`.
* **15.2.0 LLM-based Query Transformation (Optional):**
    * Use a small LLM to rephrase or generate sub-queries (e.g., for multi-hop questions). This can be a separate `LLMQueryTransformer` class.
* **15.3.0 Integrate into `FastRAG.query()`:** Pre-process the user query before sending to embedding model.

### Step 16.0: Observability & Metrics

* **16.1.0 Add Basic Timing Decorators:**
    * Create decorators to measure execution time of key methods (embedding generation, vector search, LLM call).
    * Log these timings.
* **16.2.0 Implement Structured Logging:**
    * Use Python's `logging` module throughout for `DEBUG`, `INFO`, `WARNING`, `ERROR` messages.
    * Consider `loguru` for easier logging.

### Step 17.0: Robust Error Handling & Validation

* **17.1.0 Input Validation:**
    * Use `pydantic` for validating input parameters to `FastRAG` constructor and its methods.
* **17.2.0 Custom Exceptions:**
    * Define specific custom exceptions for common failure modes (e.g., `FastRAGError`, `DataSourceError`, `EmbeddingError`, `LLMError`, `VectorStoreError`).
* **17.3.0 Graceful Fallbacks (Consideration):**
    * For example, if LLM fails, should it still return the retrieved documents? Provide options for this behavior.

### Step 18.0: Documentation & Examples

* **18.1.0 Detailed API Documentation:**
    * Write clear docstrings for all classes, methods, and parameters.
    * Set up `Sphinx` or `MkDocs` to auto-generate documentation from docstrings.
* **18.2.0 Comprehensive Examples:**
    * Create an `examples/` directory with detailed, runnable Python scripts demonstrating all features and configurations.
* **18.3.0 Publish to ReadTheDocs (Optional but excellent):** If using Sphinx, host your documentation on ReadTheDocs for easy access.

### Step 19.0: Testing Suite Expansion

* **19.1.0 Increase Test Coverage:** Aim for high test coverage (>80%) for all modules and features.
* **19.2.0 Add Performance Benchmarks:**
    * Use `pytest-benchmark` to measure the speed of core operations (indexing, querying, embedding).

### Step 20.0: Packaging and Distribution Readiness

* **20.1.0 Refine `pyproject.toml`:**
    * Add full metadata (long description, keywords, classifiers).
    * Define "extra" dependencies (e.g., `chromadb`, `pypdf`, `bs4`) for optional features.
* **20.2.0 Build Distribution Files:**
    * Confirm `python -m build` correctly generates `sdist` (`.tar.gz`) and `wheel` (`.whl`) files.
* **20.3.0 Publish to TestPyPI (Practice):**
    * Practice the upload process to TestPyPI using `twine`.

---



