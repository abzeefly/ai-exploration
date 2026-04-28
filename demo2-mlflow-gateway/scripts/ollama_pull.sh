#!/bin/sh
set -e

echo "Waiting for Ollama to be ready..."
until curl -sf http://ollama:11434/api/tags > /dev/null; do
  echo "  Ollama not ready yet, retrying in 3s..."
  sleep 3
done
echo "Ollama is ready."

echo "Pulling llama3.2 (fast, 3B)..."
curl -sf -X POST http://ollama:11434/api/pull \
  -H 'Content-Type: application/json' \
  -d '{"name":"llama3.2","stream":false}'

echo "Pulling mistral (quality, 7B)..."
curl -sf -X POST http://ollama:11434/api/pull \
  -H 'Content-Type: application/json' \
  -d '{"name":"mistral","stream":false}'

echo "Pulling nomic-embed-text (embeddings)..."
curl -sf -X POST http://ollama:11434/api/pull \
  -H 'Content-Type: application/json' \
  -d '{"name":"nomic-embed-text","stream":false}'

echo "All models pulled successfully."
