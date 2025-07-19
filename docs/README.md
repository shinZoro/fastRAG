# FastRAG

A developer-friendly SDK for building Retrieval-Augmented Generation (RAG) systems.

## Installation

```bash
pip install fastrag
```

## Usage

```python
from fastrag.core import FastRAG

# Initialize FastRAG with the path to your data
rag = FastRAG(data_source_path="./data")

# Build the index
rag.build_index()

# Query the RAG system
result = rag.query("What is the meaning of life?")

print(result["answer"])
print(result["sources"])
```
