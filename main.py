import os
from fastrag.core import FastRAG


rag = FastRAG(
         data_source_path=r"Z:\Projects\fastRAG\policy_docs\axis_policy_docs.txt",
         llm="groq:llama3-8b-8192",
         groq_api_key=os.environ.get("GROQ_API_KEY")
)

rag.build_index()
response = rag.query("What is the key policy for personal loan ?")
print(response["answer"])
