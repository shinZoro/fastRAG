from dataclasses import dataclass
from pathlib import Path
from typing import IO, List

@dataclass
class Document:
    """
    Represents a single document with its content and metadata.

    Attributes:
        content: The text content of the document.
        metadata: A dictionary of metadata associated with the document.
    """
    content: str
    metadata: dict



def load_documents(data_path: str, glob_pattern: str = "**/*") -> List[Document]:
    """
    Loads documents from a directory or a single file.

    Args:
        data_path: The path to the directory or file.
        glob_pattern: The glob pattern to match files against. Defaults to "**/*".

    Returns:
        A list of Document objects.
    """
    path = Path(data_path)
    documents = []
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(
                Document(content=content, metadata={"source": str(path)})
            )
    else:
        for file_path in path.glob(glob_pattern):
            if file_path.is_file() and file_path.suffix in [".txt", ".md"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(
                        Document(content=content, metadata={"source": str(file_path)})
                    )
    return documents
