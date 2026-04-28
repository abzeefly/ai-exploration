import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_GATEWAY_URI = os.getenv("MLFLOW_GATEWAY_URI", "http://localhost:5000")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
