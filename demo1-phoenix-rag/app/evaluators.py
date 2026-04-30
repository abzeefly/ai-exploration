import pandas as pd
from langchain_ollama import ChatOllama
from phoenix.evals.llm.adapters.langchain.adapter import LangChainModelAdapter
from phoenix.evals.llm.registries import PROVIDER_REGISTRY
from phoenix.evals import LLM, evaluate_dataframe
from phoenix.evals.metrics import FaithfulnessEvaluator, CorrectnessEvaluator, DocumentRelevanceEvaluator

from app.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL

_judge_llm: LLM | None = None


def _get_judge_llm() -> LLM:
    global _judge_llm
    if _judge_llm is not None:
        return _judge_llm

    def _create_ollama_client(model: str, is_async: bool = False, **kwargs):
        return ChatOllama(model=model, **kwargs)

    if "ollama" not in PROVIDER_REGISTRY._providers:
        PROVIDER_REGISTRY.register_provider(
            provider="ollama",
            adapter_class=LangChainModelAdapter,
            client_factory=_create_ollama_client,
            dependencies=["langchain-ollama"],
        )

    _judge_llm = LLM(provider="ollama", model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
    return _judge_llm


def evaluate_batch(rows: list[dict]) -> list[dict]:
    """
    Evaluate a batch of RAG results with LLM-as-judge.

    Each row must contain: question, context (retrieved chunks joined), answer.
    Optionally include reference for correctness scoring.

    Returns one result dict per row with keys: faithfulness, relevance,
    and correctness (only when all rows supply a reference).
    """
    judge = _get_judge_llm()
    df = pd.DataFrame(rows)

    faith_df = df.rename(columns={"question": "input", "answer": "output"})
    faith_results = evaluate_dataframe(
        dataframe=faith_df[["input", "context", "output"]],
        evaluators=[FaithfulnessEvaluator(judge)],
    )

    rel_df = df.rename(columns={"question": "input", "context": "document_text"})
    rel_results = evaluate_dataframe(
        dataframe=rel_df[["input", "document_text"]],
        evaluators=[DocumentRelevanceEvaluator(judge)],
    )

    has_reference = all(r.get("reference") for r in rows)
    if has_reference:
        corr_df = df.rename(columns={"question": "input", "answer": "output"})
        corr_results = evaluate_dataframe(
            dataframe=corr_df[["input", "output"]],
            evaluators=[CorrectnessEvaluator(judge)],
        )

    out = []
    for i in range(len(rows)):
        result = {
            "faithfulness": _extract_label(faith_results, "faithfulness_score", i),
            "relevance": _extract_label(rel_results, "document_relevance_score", i),
        }
        if has_reference:
            result["correctness"] = _extract_label(corr_results, "correctness_score", i)
        out.append(result)

    return out


def print_eval_summary(questions: list[str], eval_results: list[dict]) -> None:
    faithful = sum(1 for r in eval_results if r.get("faithfulness") == "faithful")
    relevant = sum(1 for r in eval_results if r.get("relevance") == "relevant")
    n = len(eval_results)

    print("\n--- Evaluation Summary ---")
    print(f"Faithful (no hallucination) : {faithful}/{n}")
    print(f"Context relevant            : {relevant}/{n}")

    if "correctness" in eval_results[0]:
        correct = sum(1 for r in eval_results if r.get("correctness") == "correct")
        print(f"Correct answers            : {correct}/{n}")

    print()
    for q, r in zip(questions, eval_results):
        flags = " | ".join(f"{k}={v}" for k, v in r.items())
        print(f"  {q[:60]}")
        print(f"    {flags}")


def _extract_label(results_df: pd.DataFrame, col: str, i: int) -> str | None:
    val = results_df[col].iloc[i]
    return val.get("label") if isinstance(val, dict) else None
