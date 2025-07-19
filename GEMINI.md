# Gemini CLI Best Practices for FastRAG Development

This document outlines recommended coding practices and guidelines for developing the `FastRAG` SDK, especially when interacting with the Gemini CLI. Adhering to these practices will ensure code quality, maintainability, and prevent unintended modifications by the CLI.

---

## 1. Project Scope and File Management

The Gemini CLI is a powerful tool, but it's crucial to define and respect your project's boundaries to prevent it from modifying files outside your intended scope.

* **Explicitly Define Project Root:** Always ensure your terminal's current working directory is the root of your `fastrag` project when invoking the Gemini CLI. This helps the CLI understand the context of your request.

* **Target Specific Files/Directories:** When asking the CLI to generate or modify code, be as specific as possible about the file or directory you want it to interact with.

    * **Example:** Instead of "add a new class," try "add a new class to `fastrag/core.py`."

    * If generating a new file, specify the exact path: "create `fastrag/new_module.py` with a function for X."

* **Review Changes Carefully:** Before accepting any changes proposed by the Gemini CLI, **always review the diff** thoroughly. This is your primary safeguard against unintended modifications.

* **Use Version Control (Git):** This is non-negotiable. Commit your changes frequently. If the CLI makes an undesirable change, you can easily revert it using Git.

    * `git status` to see modified files.

    * `git diff <file>` to see specific changes.

    * `git restore <file>` or `git checkout <file>` to discard changes.

    * `git commit -m "Descriptive message"` for saving good changes.

---

## 2. Python Coding Best Practices

These are general best practices for writing clean, maintainable Python code, which are even more important when collaborating with an AI assistant.

* **PEP 8 Compliance:** Adhere to Python's official style guide (PEP 8) for consistent formatting, naming conventions, and code structure. This makes your code more readable for both humans and AI.

    * Use linters (e.g., `flake8`, `ruff`) and formatters (e.g., `black`, `isort`) to automate this.

* **Clear and Concise Code:** Write self-documenting code wherever possible.

    * Use meaningful variable and function names (e.g., `calculate_average_score` instead of `calc_avg`).

    * Break down complex logic into smaller, focused functions.

* **Type Hinting:** Use type hints (`def func(arg: str) -> int:`) for all function arguments and return values. This improves code clarity, enables static analysis, and helps the CLI understand the expected data types.

* **Error Handling:** Implement robust error handling using `try-except` blocks. Provide informative exception messages.

* **Modularity:** Design your code in a modular fashion, separating concerns into different files and classes as outlined in the `FastRAG` plan (e.g., `chunking.py`, `embeddings.py`, `vector_stores.py`).

---

## 3. Documentation: Docstrings Over Excessive Comments

For an SDK, clear and comprehensive documentation is paramount. Prioritize well-structured docstrings.

* **Use Docstrings for Functions, Classes, and Modules:**

    * **Purpose:** Explain *what* a function/class does, its parameters, what it returns, and any exceptions it might raise.

    * **Format:** Use a consistent docstring format (e.g., Google, NumPy, reStructuredText). Google style is often recommended for readability.

    * **Example (Google Style):**

        ```python
        def calculate_similarity(embedding1: list[float], embedding2: list[float]) -> float:
            """Calculates the cosine similarity between two embedding vectors.

            Args:
                embedding1: The first embedding vector.
                embedding2: The second embedding vector.

            Returns:
                The cosine similarity score (float between -1 and 1).
            """
            # Function implementation
            pass

        class FastRAG:
            """A developer-friendly SDK for building Retrieval-Augmented Generation (RAG) systems.

            This class provides a streamlined interface for ingesting data, building a searchable
            index, and querying Large Language Models (LLMs) with retrieved context.
            """
            # Class implementation
            pass
        ```

* **Minimize Inline Comments:**

    * **Purpose:** Use inline comments *sparingly* to explain *why* a particular piece of non-obvious code is written, or to clarify complex algorithms.

    * **Avoid:** Do not use comments to explain *what* the code is doing if it's already clear from the code itself (e.g., `x = x + 1 # Increment x`).

    * **Rationale:** Well-named variables, functions, and clear logic make most "what" comments redundant. Overly commented code can become stale and harder to maintain than the code it describes.

    * **Example of appropriate inline comment:**

        ```python
        # Apply a custom re-ranking algorithm to prioritize newer documents
        if self.reranker:
            documents = self.reranker.rerank(query_text, documents)
        ```

---

## 4. Interacting with the Gemini CLI

* **Be Explicit in Prompts:** When asking the CLI to write code, provide as much context as possible:

    * "In `fastrag/embeddings.py`, add a class `OpenAIEmbeddings` that implements `AbstractEmbeddingsModel`."

    * "Modify the `FastRAG.query` method in `fastrag/core.py` to accept a `prompt_template` argument."

* **Iterative Development:** Use the CLI for small, focused tasks rather than large, complex generations. This makes review easier and reduces the chance of errors.

* **Test After Generation:** Always run your tests after incorporating changes from the CLI to ensure functionality hasn't been broken.

* **Provide Feedback:** If the CLI doesn't produce the desired output, refine your prompt. Explain *why* the previous attempt wasn't suitable.

By following these practices, you'll create a high-quality `FastRAG` SDK and leverage the Gemini CLI effectively as a development assistant.