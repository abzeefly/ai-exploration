"""
MLflow AI Gateway — Demo Script

Shows the core value proposition: the same Python call shape routes to
different LLM backends simply by changing the route name.

Run inside Docker:  docker compose run --rm demo-app
Run locally:        MLFLOW_GATEWAY_URI=http://localhost:5000 python -m app.demo_routes
"""
import time
import math
import httpx
from mlflow.deployments import get_deploy_client

from app.config import MLFLOW_GATEWAY_URI


def get_client():
    return get_deploy_client(MLFLOW_GATEWAY_URI)


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def demo_list_routes(client):
    section("1. Available Routes")
    routes = client.list_endpoints()
    for route in routes:
        print(f"  • {route['name']:25s}  type={route['endpoint_type']}")


def demo_chat_comparison(client):
    section("2. Same Question → Two Models (llama3.2 vs mistral)")
    question = "In 2024, what is the standard deduction for a single filer? Answer in one sentence."
    payload = {"messages": [{"role": "user", "content": question}]}

    for route in ["llama-chat", "mistral-chat"]:
        t0 = time.time()
        try:
            resp = client.predict(endpoint=route, inputs=payload)
            elapsed = time.time() - t0
            answer = resp["choices"][0]["message"]["content"]
            label = "🦙 llama3.2 (3B)" if route == "llama-chat" else "⚡ mistral (7B)"
            print(f"\n[{label}]  ({elapsed:.2f}s)")
            print(f"  {answer}")
        except Exception as exc:
            print(f"\n[{route}]  ERROR: {exc}")


def demo_embeddings(client):
    section("3. Embeddings via Gateway")
    texts = [
        "standard deduction single filer",
        "capital gains tax rates",
        "401k contribution limit catch-up",
        "Roth IRA income phase-out",
    ]
    payload = {"input": texts}
    try:
        resp = client.predict(endpoint="ollama-embeddings", inputs=payload)
        vecs = [item["embedding"] for item in resp["data"]]
        print(f"  Embedded {len(vecs)} texts, dimension={len(vecs[0])}")

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            return dot / (math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b)))

        print("\n  Cosine similarities:")
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = cosine(vecs[i], vecs[j])
                print(f"    '{texts[i][:30]}' <-> '{texts[j][:30]}'  →  {sim:.3f}")
    except Exception as exc:
        print(f"  ERROR: {exc}")


def demo_direct_http(gateway_uri: str):
    section("4. Raw HTTP — OpenAI-Compatible Endpoint")
    url = f"{gateway_uri}/gateway/mistral-chat/invocations"
    payload = {
        "messages": [
            {"role": "user", "content": "What is the 2024 Roth IRA contribution limit? One sentence."}
        ]
    }
    try:
        r = httpx.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        print(f"  Status: {r.status_code}")
        print(f"  Answer: {data['choices'][0]['message']['content']}")
    except Exception as exc:
        print(f"  ERROR: {exc}")


def main():
    print("\n=== MLflow AI Gateway Demo — llama3.2 vs mistral ===")
    print(f"Gateway URI: {MLFLOW_GATEWAY_URI}\n")

    client = get_client()

    demo_list_routes(client)
    demo_chat_comparison(client)
    demo_embeddings(client)
    demo_direct_http(MLFLOW_GATEWAY_URI)

    print("\n\nDemo complete.")


if __name__ == "__main__":
    main()
