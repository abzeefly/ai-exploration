import os
import chromadb
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from app.config import (
    OLLAMA_BASE_URL, CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, DOCS_DIR
)


def load_documents(docs_dir: str = DOCS_DIR) -> list:
    docs = []
    for fname in os.listdir(docs_dir):
        path = os.path.join(docs_dir, fname)
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif fname.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} document(s) from {docs_dir}")
    return docs


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def get_chroma_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def embed_and_store(chunks: list, collection_name: str = CHROMA_COLLECTION) -> Chroma:
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    client = get_chroma_client()
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    vectorstore.add_documents(chunks)
    print(f"Stored {len(chunks)} chunks in collection '{collection_name}'")
    return vectorstore


def collection_is_empty(collection_name: str = CHROMA_COLLECTION) -> bool:
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        return collection.count() == 0
    except Exception:
        return True


def run_ingestion() -> Chroma:
    docs = load_documents()
    chunks = chunk_documents(docs)
    return embed_and_store(chunks)
