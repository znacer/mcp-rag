import logging
from mcp.server.fastmcp import FastMCP
from typing import List
from bm25_retriever import (
    BM25Retriever,
)  # Assuming the previous code is in bm25_retriever.py

# Initialize the MCP server
mcp = FastMCP("BM25RAGServer", port=8000)

# Initialize the BM25 retriever with the path to our corpus
corpus_path = "./corpus"  # Replace with the actual path to your corpus directory
bm25_retriever = BM25Retriever(corpus_path)


@mcp.tool()
def perform_rag(query: str, top_n: int = 3) -> List[str]:
    """
    Performs Retrieval-Augmented Generation using BM25 on a local corpus.

    Args:
        query: The user's search query.
        top_n: The number of chunks to retrieve

    Returns:
        A list of the top_n most relevant documents from the corpus.
    """
    retrieved_documents = bm25_retriever.retrieve(query, top_n=top_n)
    retrieved_documents = retrieved_documents.flatten().tolist()
    logging.info(retrieved_documents)
    return retrieved_documents


if __name__ == "__main__":
    mcp.run(transport="sse")
