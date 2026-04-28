import sys
import time
import urllib.request
from app.config import CHROMA_HOST, CHROMA_PORT, PHOENIX_COLLECTOR_ENDPOINT
from app.ingest import collection_is_empty, run_ingestion
from app.rag_pipeline import query_rag


def wait_for_service(url: str, name: str, retries: int = 30, delay: int = 3) -> None:
    for i in range(retries):
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"  {name} ready.")
            return
        except Exception:
            print(f"  Waiting for {name}... ({i + 1}/{retries})")
            time.sleep(delay)
    raise RuntimeError(f"{name} did not become ready at {url}")

SAMPLE_QUESTIONS = [
    "What is the standard deduction for a single filer in 2024?",
    "How are long-term capital gains taxed for someone earning $100,000?",
    "What is the 401(k) contribution limit for 2024, including catch-up contributions?",
    "Can I deduct student loan interest if I earn $80,000 as a single filer?",
    "What are the income phase-out ranges for Roth IRA contributions in 2024?",
]


def main():
    print("=== Tax Policy RAG Demo — Powered by Arize Phoenix ===\n")
    print("Waiting for services...")
    wait_for_service(f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v2/heartbeat", "ChromaDB")
    wait_for_service(f"{PHOENIX_COLLECTOR_ENDPOINT}/healthz", "Phoenix")
    print()

    if collection_is_empty():
        print("ChromaDB collection is empty. Running ingestion...\n")
        run_ingestion()
        print("\nIngestion complete.\n")
    else:
        print("ChromaDB collection already populated. Skipping ingestion.\n")

    print("--- Running sample questions to warm up traces ---\n")
    for q in SAMPLE_QUESTIONS:
        print(f"Q: {q}")
        result = query_rag(q)
        print(f"A: {result['answer'][:300]}...")
        print(f"   Sources: {result['sources']}\n")

    print("\n--- Interactive Mode (type 'exit' to quit) ---\n")
    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question or question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        result = query_rag(question)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {result['sources']}\n")


if __name__ == "__main__":
    main()
