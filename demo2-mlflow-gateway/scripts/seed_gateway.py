"""
Reads gateway/config.yaml and registers each endpoint in the MLflow tracking
server's integrated gateway (3.0 API) so they appear in the UI at /#/gateway.

Flow per endpoint:  secret  →  model definition  →  endpoint
The secret stores the API key + auth_config (base URL for non-standard providers).
"""
import sys
import time
import yaml
import requests

TRACKING = "http://mlflow-server:5001"
BASE = f"{TRACKING}/ajax-api/3.0/mlflow/gateway"
HEADERS = {"Host": "localhost"}


def post(path, body):
    r = requests.post(f"{BASE}/{path}", json=body, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def get(path):
    r = requests.get(f"{BASE}/{path}", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def wait_for_server():
    for _ in range(30):
        try:
            requests.get(f"{TRACKING}/health", headers=HEADERS, timeout=5).raise_for_status()
            return
        except Exception:
            time.sleep(2)
    sys.exit("mlflow-server never became ready")


def already_exists(name):
    try:
        data = get("endpoints/list")
        return any(e.get("name") == name for e in data.get("endpoints", []))
    except Exception:
        return False


def seed(config_path):
    wait_for_server()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for ep in cfg.get("endpoints", []):
        name = ep["name"]
        ep_type = ep["endpoint_type"]
        model = ep["model"]
        provider = model["provider"]
        model_name = model["name"]
        model_cfg = model.get("config", {})
        api_key = model_cfg.get("openai_api_key", "")
        api_base = model_cfg.get("openai_api_base", "")

        if already_exists(name):
            print(f"  skip  {name} (already registered)")
            continue

        # 1. secret
        secret_body = {
            "secret_name": f"{name}-key",
            "secret_value": {"api_key": api_key},
            "provider": provider,
        }
        if api_base:
            secret_body["auth_config"] = {"openai_api_base": api_base}

        secret_resp = post("secrets/create", secret_body)
        secret_id = secret_resp["secret"]["secret_id"]

        # 2. model definition
        model_def_resp = post("model-definitions/create", {
            "name": f"{name}-def",
            "secret_id": secret_id,
            "provider": provider,
            "model_name": model_name,
        })
        model_def_id = model_def_resp["model_definition"]["model_definition_id"]

        # 3. endpoint
        post("endpoints/create", {
            "name": name,
            "model_configs": [{"model_definition_id": model_def_id, "linkage_type": "PRIMARY"}],
        })

        print(f"  registered  {name}  ({provider}/{model_name})")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "/app/config.yaml"
    print(f"Seeding gateway endpoints from {config} ...")
    seed(config)
    print("Done.")
