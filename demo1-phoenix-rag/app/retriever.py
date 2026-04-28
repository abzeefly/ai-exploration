import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from app.config import (
    OLLAMA_BASE_URL, CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION
)


def get_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
    )


def retrieve(query: str, k: int = 4) -> list:
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k=k)
