from typing import Any, Dict

from utils.logging import log_panel, log_rule
from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    question = state["question"]

    log_rule("RETRIEVE  ·  vector search", "bold cyan")
    log_panel("Question", question, "cyan")

    documents = retriever.invoke(question)
    log_panel("Hits", f"{len(documents)} document(s)", "blue")
    return {"documents": documents, "question": question}
