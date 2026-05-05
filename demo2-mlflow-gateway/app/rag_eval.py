"""
MLflow RAG Evaluation — llama3.2 vs mistral

Routes generation through the gateway (llama3.2 and mistral), then runs
mlflow.evaluate() using Ollama's OpenAI-compatible endpoint as the judge.

Usage:
  docker compose run --rm demo-app python -m app.rag_eval
  # locally:
  MLFLOW_GATEWAY_URI=http://localhost:5000 OLLAMA_BASE_URL=http://localhost:11434 python -m app.rag_eval
"""
import time
import pandas as pd
import mlflow
from mlflow.deployments import get_deploy_client
from mlflow.metrics.genai import faithfulness, answer_relevance, relevance

from app.config import MLFLOW_GATEWAY_URI, OLLAMA_BASE_URL, MLFLOW_TRACKING_URI

EVAL_DATA = [
    {
        "question": "What is the standard deduction for a single filer in 2024?",
        "context": "For tax year 2024, the standard deduction amounts are: Single or Married Filing Separately: $14,600. Married Filing Jointly or Qualifying Surviving Spouse: $29,200. Head of Household: $21,900.",
        "ground_truth": "The standard deduction for a single filer in 2024 is $14,600.",
    },
    {
        "question": "What is the long-term capital gains rate for a single filer earning $100,000?",
        "context": "2024 Long-Term Capital Gains Tax Rates: 0%: up to $47,025 (single). 15%: $47,026 to $518,900 (single). 20%: above $518,900 (single).",
        "ground_truth": "A single filer earning $100,000 pays 15% on long-term capital gains.",
    },
    {
        "question": "What is the 401(k) employee contribution limit for 2024?",
        "context": "401(k), 403(b), AND 457(b) PLANS: Employee Contribution Limit: $23,000. Catch-Up Contribution (age 50 or older): Additional $7,500 (total $30,500).",
        "ground_truth": "The 401(k) employee contribution limit for 2024 is $23,000 ($30,500 with catch-up for age 50+).",
    },
    {
        "question": "What is the Roth IRA income phase-out range for single filers in 2024?",
        "context": "ROTH IRA: Income phase-out ranges for contributions: Single/Head of Household: $146,000 – $161,000 MAGI.",
        "ground_truth": "The Roth IRA income phase-out range for single filers in 2024 is $146,000–$161,000 MAGI.",
    },
    {
        "question": "Can I deduct mortgage interest on a second home?",
        "context": "Mortgage Interest: Interest on loans secured by your primary or secondary residence is deductible on up to $750,000 of acquisition debt (for loans originated after December 15, 2017).",
        "ground_truth": "Yes, mortgage interest on a second home is deductible on up to $750,000 of acquisition debt for loans originated after Dec 15, 2017.",
    },
]


def generate_answers(gateway_client, route: str, rows: list[dict]) -> pd.DataFrame:
    results = []
    for row in rows:
        prompt = (
            f"Answer the following tax question based only on the provided context.\n\n"
            f"Context: {row['context']}\n\n"
            f"Question: {row['question']}\n\nAnswer concisely."
        )
        t0 = time.time()
        try:
            resp = gateway_client.predict(
                endpoint=route,
                inputs={"messages": [{"role": "user", "content": prompt}]},
            )
            answer = resp["choices"][0]["message"]["content"]
        except Exception as exc:
            answer = f"ERROR: {exc}"
        results.append({**row, "answer": answer, "latency_s": round(time.time() - t0, 2)})
    return pd.DataFrame(results)


def run_mlflow_eval(df: pd.DataFrame, run_name: str, route: str, judge_base_url: str) -> dict:
    ollama_v1 = f"{judge_base_url}/v1"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("gateway_route", route)
        mlflow.log_param("judge_model", "mistral")
        mlflow.log_param("judge_endpoint", ollama_v1)
        mlflow.log_param("num_questions", len(df))
        mlflow.log_metric("mean_latency_s", df["latency_s"].mean())
        mlflow.log_metric("max_latency_s", df["latency_s"].max())
        mlflow.log_metric("min_latency_s", df["latency_s"].min())

        eval_result = mlflow.evaluate(
            data=df[["question", "context", "answer", "ground_truth"]],
            targets="ground_truth",
            predictions="answer",
            model_type="question-answering",
            extra_metrics=[
                faithfulness(model="openai:/mistral"),
                answer_relevance(model="openai:/mistral"),
                relevance(model="openai:/mistral"),
            ],
            evaluator_config={
                "col_mapping": {"inputs": "question", "context": "context"},
                "openai_api_key": "ollama",
                "openai_api_base": ollama_v1,
            },
        )
        return {"run_id": run.info.run_id, "metrics": eval_result.metrics}


def section(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("tax-rag-evaluation")

    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

    gw = get_deploy_client(MLFLOW_GATEWAY_URI)

    section("Run 1: llama3.2 (3B, fast) as answer generator")
    llama_df = generate_answers(gw, "llama-chat", EVAL_DATA)
    print(f"Generated {len(llama_df)} answers, avg latency {llama_df['latency_s'].mean():.2f}s")
    llama_eval = run_mlflow_eval(llama_df, "llama3.2-rag-eval", "llama-chat", OLLAMA_BASE_URL)
    print("llama3.2 metrics:", {k: round(v, 3) for k, v in llama_eval["metrics"].items() if isinstance(v, float)})

    section("Run 2: mistral (7B, quality) as answer generator")
    mistral_df = generate_answers(gw, "mistral-chat", EVAL_DATA)
    print(f"Generated {len(mistral_df)} answers, avg latency {mistral_df['latency_s'].mean():.2f}s")
    mistral_eval = run_mlflow_eval(mistral_df, "mistral-rag-eval", "mistral-chat", OLLAMA_BASE_URL)
    print("mistral metrics:", {k: round(v, 3) for k, v in mistral_eval["metrics"].items() if isinstance(v, float)})

    section("Comparison Summary")
    for key in ["faithfulness/v1/mean", "answer_relevance/v1/mean", "relevance/v1/mean"]:
        l = llama_eval["metrics"].get(key, 0)
        m = mistral_eval["metrics"].get(key, 0)
        label = key.split("/")[0]
        print(f"  {label:25s}  llama3.2={l:.2f}  mistral={m:.2f}  diff={m-l:+.2f}")

    print(f"\nView results in MLflow UI  →  {MLFLOW_TRACKING_URI}")
    print(f"  Experiment: tax-rag-evaluation → compare 'llama3.2-rag-eval' vs 'mistral-rag-eval'")


if __name__ == "__main__":
    main()
