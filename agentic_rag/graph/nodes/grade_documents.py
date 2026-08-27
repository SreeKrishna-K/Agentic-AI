from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState
from utils.logging import log_panel, log_rule, log_table, preview_text


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    question = state["question"]
    documents = state["documents"]

    log_rule("GRADE  ·  document relevance", "bold yellow")
    log_panel("Question", question, "cyan")

    filtered_docs = []
    web_search = False
    rows: list[tuple[str, str, str]] = []
    for index, d in enumerate(documents):
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        preview = preview_text(d.page_content)
        if grade.lower() == "yes":
            filtered_docs.append(d)
            rows.append((str(index), "[green]yes[/green]", preview))
        else:
            web_search = True
            rows.append((str(index), "[red]no[/red]", preview))

    log_table(
        "Document grades",
        ["#", "Grade", "Preview"],
        rows,
        justify={"#": "right"},
    )
    log_panel(
        "Result",
        f"kept {len(filtered_docs)}/{len(documents)}  ·  web_search={web_search}",
        "green" if not web_search else "yellow",
    )
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
