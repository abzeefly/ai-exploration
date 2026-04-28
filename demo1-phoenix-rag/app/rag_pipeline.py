from opentelemetry import trace as otel_trace
from opentelemetry.trace import StatusCode
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.trace import SpanAttributes
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, PHOENIX_COLLECTOR_ENDPOINT
from app import retriever

# Instrumentation must happen before any clients are instantiated
register(
    endpoint=f"{PHOENIX_COLLECTOR_ENDPOINT}/v1/traces",
    project_name="tax-rag-demo",
    protocol="http/protobuf",
)
LangChainInstrumentor().instrument()

_llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
_tracer = otel_trace.get_tracer(__name__)

SYSTEM_PROMPT = """You are a knowledgeable tax advisor assistant. Answer the user's question
based solely on the provided context excerpts from official tax policy documents.
Be precise, cite specific figures when relevant, and state if the context doesn't
contain enough information to fully answer the question."""


def query_rag(question: str, k: int = 4) -> dict:
    with _tracer.start_as_current_span("rag_query") as span:
        span.set_attribute(SpanAttributes.INPUT_VALUE, question)
        span.set_attribute("rag.retrieval.k", k)
        span.set_attribute("llm.model", OLLAMA_LLM_MODEL)

        # --- Retrieval ---
        with _tracer.start_as_current_span("retrieve") as ret_span:
            ret_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "RETRIEVER")
            ret_span.set_attribute(SpanAttributes.INPUT_VALUE, question)
            docs = retriever.retrieve(question, k=k)

            for i, doc in enumerate(docs):
                src = doc.metadata.get("source", "unknown")
                ret_span.set_attribute(f"retrieval.documents.{i}.document.content", doc.page_content)
                ret_span.set_attribute(f"retrieval.documents.{i}.document.metadata.source", src)

            ret_span.set_attribute("retrieval.document_count", len(docs))

        sources = list({doc.metadata.get("source", "unknown") for doc in docs})
        context_chunks = [doc.page_content for doc in docs]
        context = "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

        # --- Generation (traced automatically by LangChainInstrumentor) ---
        response = _llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ])
        answer = response.content

        usage = response.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
        span.set_attribute("rag.sources", str(sources))
        span.set_attribute("rag.chunk_count", len(docs))
        span.set_attribute("llm.token_count.prompt", input_tokens)
        span.set_attribute("llm.token_count.completion", output_tokens)
        span.set_status(StatusCode.OK)

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(docs),
        "context_chunks": context_chunks,
        "token_usage": {"input": input_tokens, "output": output_tokens},
    }
